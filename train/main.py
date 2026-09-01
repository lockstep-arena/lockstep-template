"""train -> export -> parity-check -> stage the submittable agent bundle.

One command (``task train`` runs it), for ANY published environment::

    python -m train.main --env <slug> --steps 8192

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
        "--env",
        required=True,
        help="environment slug (ENV= on the task line) — the identity the "
        "staged bundle declares; browse the catalog at https://lockstep.it/arenas",
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.engine:
        engine, shell = Path(args.engine), Path(args.shell or "")
        if not args.shell:
            raise SystemExit("--engine override needs --shell too (the staged bundle ships it)")
    else:
        from .core.engine import ensure_engine

        paths = ensure_engine(args.env, args.mode, args.version)
        engine, shell = paths.engine, paths.shell
    mode, payload_schema_version = engine_identity(engine)

    if args.from_weights:
        print(f"── loading weights {args.from_weights}")
        blob = torch.load(args.from_weights, weights_only=True, map_location="cpu")
        net = policy_from_signature(blob["spaces"])
        net.load_state_dict(blob["state_dict"])
    elif args.parallel:
        print(
            f"── self-play training {args.env} [{mode}] for {args.steps} "
            "seat-steps (both seats learning, one shared policy)"
        )
        net = train_self_play(
            steps=args.steps,
            engine=str(engine),
            time_limit_ticks=args.time_limit_ticks,
            seed=args.seed,
            device=args.device,
            out_dir=OUT_DIR,
            resume=args.resume,
        )
    else:
        print(f"── training {args.env} [{mode}] for {args.steps} steps")
        net = train(
            steps=args.steps,
            engine=str(engine),
            time_limit_ticks=args.time_limit_ticks,
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.device,
            out_dir=OUT_DIR,
            resume=args.resume,
        )
        weights = OUT_DIR / "policy.pt"
        torch.save(
            {
                "action_len": net.action_len,
                "spaces": net.space_signature(),
                "state_dict": net.state_dict(),
            },
            weights,
        )
        print(f"→ weights: {weights}")

    onnx = export(net, OUT_DIR / "policy.onnx")
    print(f"→ onnx: {onnx} ({onnx.stat().st_size} bytes)")

    diff = verify(net, onnx)
    print(f"✓ torch/onnxruntime parity: max abs diff {diff:.3e}")

    bundle = stage(args.env, mode, payload_schema_version, onnx, shell, BUNDLE_DIR)
    print(f"→ bundle: {bundle}")
    print("\nRun it:   task match\nCompete:  task upload")


if __name__ == "__main__":
    sys.exit(main())
