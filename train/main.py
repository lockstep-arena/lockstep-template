"""train -> export -> parity-check -> stage the submittable agent bundle.

One command (``task train AGENT=<name>`` runs it), for ANY published
environment::

    python -m train.main --agent <name> --steps 8192

The agent's ``agent.toml`` (written by ``task create-agent``) names the
environment and mode, so the right engine is resolved from the keyed cache
and a mode mismatch is impossible. Checkpoints, the exported ONNX and the
staged bundle all land under ``agents/<name>/out/``.

Nothing here is per-environment: the env, its spaces and its reward come
from the engine wasm's own declaration (``lockstep_train``
reads it), and the staged bundle pairs your trained ``policy.onnx`` with
the GENERIC agent shell fetched from the same release (the keyed cache
under ``out/cache/<env>/<mode>/`` — see ``train.core.engine``). The staged bundle is
what the platform actually consumes — and what ``lockstep match run`` /
``lockstep agent upload`` take directly::

    out/agent-bundle/
      lockstep.toml        declares the `policy` artifact by NAME
      component.wasm       the generic ONNX agent shell (from the release)
      artifacts/policy.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .core import utf8_output
from .core.export import export, verify
from .core.policy import policy_from_signature
from .core.self_play import train_self_play
from .core.stage import stage
from .core.train import default_num_envs, train

OUT_DIR = Path("out")
BUNDLE_DIR = OUT_DIR / "agent-bundle"


def engine_identity(engine: Path) -> tuple[str, int]:
    """(mode, payload_schema_version), read from the ENGINE's own descriptor.

    Never typed by hand and never fetched separately: the wasm you train
    against is the wasm that knows what it is. ``lockstep_train`` exposes
    the descriptor on its session handle.
    """
    from lockstep_train import Session

    session = Session(engine_source=str(engine))
    return session.mode, session.payload_schema_version


def main() -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--agent",
        default=None,
        help="agent name (AGENT= on the task line): env/mode come from its "
        "agent.toml and outputs land in agents/<name>/out/",
    )
    p.add_argument(
        "--env",
        default=None,
        help="environment slug — only without --agent; outputs land in out/",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=8192,
        help="environment steps. The default is a SHORT run: it produces a real "
        "policy, not a good one — and on sparse rewards, steps alone will not "
        "either (see the README's reward-landscape section).",
    )
    p.add_argument("--time-limit-ticks", type=int, default=1800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--mode",
        default=None,
        help="mode key (MODE= on the task line); default the release's default_mode",
    )
    p.add_argument(
        "--version",
        default=None,
        help="release version assertion; default the published release",
    )
    p.add_argument(
        "--engine",
        default=None,
        help="override: a local engine wasm path (default: the keyed cache "
        "fetches this env/mode's released engine)",
    )
    p.add_argument(
        "--shell",
        default=None,
        help="override: the generic ONNX agent shell to stage (default: from "
        "the same cached release as the engine)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="continue a crashed/stopped run from out/checkpoint.pt (written "
        "every rollout) up to --steps",
    )
    p.add_argument(
        "--from-weights",
        default=None,
        help="skip training and export an existing .pt (for iterating on export/stage)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="update-pass device (cuda/mps/cpu); default: best available",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help=f"parallel engine instances for collection (default {default_num_envs()} "
        "here; one env pins one core)",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="self-play: train BOTH seats at once with one shared policy over "
        "the generic PettingZoo parallel env (multi-seat engines only)",
    )
    args = p.parse_args()

    if args.parallel and args.num_envs not in (None, 1):
        raise SystemExit(
            "--parallel drives ONE parallel env (PettingZoo has no vector "
            "API; the engine is not the bottleneck) — drop --num-envs"
        )

    if args.agent or not args.env:
        from .agents import resolve_agent

        cfg = resolve_agent(args.agent)
        if cfg.lang != "python":
            raise SystemExit(
                f"agent {cfg.name} is written in {cfg.lang} — training is the "
                f"python path; build it instead: task build AGENT={cfg.name}"
            )
        env_slug, env_mode = cfg.env, args.mode or cfg.mode
        out_dir, bundle_dir = cfg.out_dir, cfg.bundle_dir
    else:
        env_slug, env_mode = args.env, args.mode
        out_dir, bundle_dir = OUT_DIR, BUNDLE_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.engine:
        engine, shell = Path(args.engine), Path(args.shell or "")
        if not args.shell:
            raise SystemExit("--engine override needs --shell too (the staged bundle ships it)")
    else:
        from .core.engine import ensure_engine

        paths = ensure_engine(env_slug, env_mode, args.version)
        engine, shell = paths.engine, paths.shell
    mode, payload_schema_version = engine_identity(engine)

    if args.from_weights:
        print(f"── loading weights {args.from_weights}")
        blob = torch.load(args.from_weights, weights_only=True, map_location="cpu")
        net = policy_from_signature(blob["spaces"])
        net.load_state_dict(blob["state_dict"])
    elif args.parallel:
        print(
            f"── self-play training {env_slug} [{mode}] for {args.steps} "
            "seat-steps (both seats learning, one shared policy)"
        )
        net = train_self_play(
            steps=args.steps,
            engine=str(engine),
            time_limit_ticks=args.time_limit_ticks,
            seed=args.seed,
            device=args.device,
            out_dir=out_dir,
            resume=args.resume,
        )
    else:
        print(f"── training {env_slug} [{mode}] for {args.steps} steps")
        net = train(
            steps=args.steps,
            engine=str(engine),
            time_limit_ticks=args.time_limit_ticks,
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.device,
            out_dir=out_dir,
            resume=args.resume,
        )
        weights = out_dir / "policy.pt"
        torch.save(
            {
                "action_len": net.action_len,
                "spaces": net.space_signature(),
                "state_dict": net.state_dict(),
            },
            weights,
        )
        print(f"→ weights: {weights}")

    onnx = export(net, out_dir / "policy.onnx")
    print(f"→ onnx: {onnx} ({onnx.stat().st_size} bytes)")

    diff = verify(net, onnx)
    print(f"✓ torch/onnxruntime parity: max abs diff {diff:.3e}")

    bundle = stage(env_slug, mode, payload_schema_version, onnx, shell, bundle_dir)
    print(f"→ bundle: {bundle}")
    print("\nRun it:   task match\nCompete:  task upload")


if __name__ == "__main__":
    sys.exit(main())
