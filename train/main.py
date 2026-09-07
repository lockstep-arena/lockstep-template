"""train -> export -> parity-check -> stage the submittable agent bundle.

One command (``task train AGENT=<name>`` runs it), for ANY published
environment::

    python -m train.main --agent <name> --steps 8192                 # PPO
    python -m train.main --agent <name> --recipe supervised --seeds 16  # fit the revealed truth

The agent's ``agent.toml`` (written by ``task create-agent``) names the
environment and mode, so the right engine is resolved from the keyed cache
and a mode mismatch is impossible. Checkpoints, the exported ONNX and the
staged bundle all land under ``agents/<name>/out/``.

Two recipes, one network. ``ppo`` (the default) runs the PPO loop in
``train/core/train.py``; ``supervised`` runs ``train/core/supervised.py``
— collect ``(observation, truth)`` pairs from the engine's feedback (or
its lagged reward) and fit a classifier head. Both build the network from
the agent's OWN ``model.py`` (``build_policy``, scaffolded from the
declaration; the stock flat-stream network when the file is absent), and
both end at the same export → parity → stage path.

Nothing here is per-environment: the env, its spaces and its reward come
from the engine wasm's own declaration (``lockstep_train`` reads it), and
the staged bundle pairs your trained ``policy.onnx`` with the GENERIC
agent shell fetched from the same release (the keyed cache under
``out/cache/<env>/<mode>/`` — see ``train.core.engine``). The staged
bundle is what the platform actually consumes — and what ``lockstep match
run`` / ``lockstep agent upload`` take directly::

    agents/<name>/out/bundle/
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
from .core.policy import Policy, policy_from_signature
from .core.self_play import train_self_play
from .core.stage import provenance, stage
from .core.supervised import LabelSource, SupervisedPolicy, train_supervised
from .core.train import default_num_envs, train

OUT_DIR = Path("out")
BUNDLE_DIR = OUT_DIR / "agent-bundle"

RECIPES = ("ppo", "supervised")


def engine_identity(engine: Path) -> tuple[str, int]:
    """(mode, payload_schema_version), read from the ENGINE's own descriptor.

    Never typed by hand and never fetched separately: the wasm you train
    against is the wasm that knows what it is. ``lockstep_train`` exposes
    the descriptor on its session handle.
    """
    from lockstep_train import Session

    session = Session(engine_source=str(engine))
    return session.mode, session.payload_schema_version


def save_weights(path: Path, net, recipe: str) -> None:
    """The weights file ``--from-weights`` reloads: the space signature,
    the state dict, the recipe, and — for a supervised head — where its
    prediction lands and which codes it chooses between."""
    blob = {
        "recipe": recipe,
        "action_len": net.action_len,
        "spaces": net.space_signature(),
        "state_dict": net.state_dict(),
    }
    if isinstance(net, SupervisedPolicy):
        blob["supervised"] = net.head_signature()
    torch.save(blob, path)


def load_weights(path: Path, build) -> torch.nn.Module:
    """Rebuild the network a weights file describes — through the agent's
    ``build_policy`` so the streams match — and load its state."""
    from gymnasium import spaces

    blob = torch.load(path, weights_only=True, map_location="cpu")
    base = policy_from_signature(blob["spaces"], build)
    if blob.get("recipe") == "supervised":
        head = blob["supervised"]
        action_space = spaces.Box(-1.0, 1.0, (blob["action_len"],), dtype="float32")
        net = SupervisedPolicy(base, action_space, head["element"], head["classes"])
    else:
        net = base
    net.load_state_dict(blob["state_dict"])
    return net


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
        "--recipe",
        default="ppo",
        choices=RECIPES,
        help="ppo (default): reinforcement learning on the reward; supervised: "
        "fit the truth the environment reveals (RECIPE= on the task line)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=8192,
        help="environment steps (ppo). The default is a SHORT run: it produces a real "
        "policy, not a good one — and on sparse rewards, steps alone will not "
        "either (see the README's reward-landscape section).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=16,
        help="supervised: how many public seeds to collect labelled observations from (SEEDS=)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="supervised: passes over the collected pairs (EPOCHS=)",
    )
    p.add_argument(
        "--label-value",
        default=None,
        help="supervised: the feedback-window observation to read labels from "
        "(overrides agent.toml's [supervised]; columns keep the documented "
        "names age_ticks / your_decision / truth / valid)",
    )
    p.add_argument(
        "--reward-lag",
        type=int,
        default=None,
        help="supervised: label each decision with the reward this many ticks "
        "later instead of a feedback window",
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
        help="ppo: continue a crashed/stopped run from out/checkpoint.pt (written "
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
        help=f"ppo: parallel engine instances for collection (default {default_num_envs()} "
        "here; one env pins one core)",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="ppo self-play: train BOTH seats at once with one shared policy over "
        "the generic PettingZoo parallel env (multi-seat engines only)",
    )
    args = p.parse_args()

    if args.parallel and args.num_envs not in (None, 1):
        raise SystemExit(
            "--parallel drives ONE parallel env (PettingZoo has no vector "
            "API; the engine is not the bottleneck) — drop --num-envs"
        )
    if args.parallel and args.recipe == "supervised":
        raise SystemExit("--parallel is a PPO option; the supervised recipe drives one seat")

    build = Policy
    supervised_block: dict | None = None
    if args.agent or not args.env:
        from .agents import policy_builder, resolve_agent

        cfg = resolve_agent(args.agent)
        if cfg.lang != "python":
            raise SystemExit(
                f"agent {cfg.name} is written in {cfg.lang} — training is the "
                f"python path; build it instead: task build AGENT={cfg.name}"
            )
        env_slug, env_mode = cfg.env, args.mode or cfg.mode
        out_dir, bundle_dir = cfg.out_dir, cfg.bundle_dir
        build = policy_builder(cfg)
        supervised_block = cfg.supervised
        if (cfg.dir / "model.py").is_file():
            print(f"── network: {cfg.dir / 'model.py'} (build_policy)")
        else:
            print("── network: the stock flat-stream policy (no model.py in the agent dir)")
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

    recipe = args.recipe
    if args.from_weights:
        print(f"── loading weights {args.from_weights}")
        net = load_weights(Path(args.from_weights), build)
        recipe = "supervised" if isinstance(net, SupervisedPolicy) else "ppo"
    elif recipe == "supervised":
        if args.reward_lag is not None:
            source = LabelSource(reward_lag=args.reward_lag)
        elif args.label_value:
            source = LabelSource(value=args.label_value, valid_col="valid")
        elif supervised_block:
            source = LabelSource.from_block(supervised_block)
        else:
            raise SystemExit(
                "this agent's agent.toml has no [supervised] block — the declaration "
                "reveals no feedback window with the documented columns and no "
                "reward lag. Name the source yourself: --label-value <obs> or "
                "--reward-lag <ticks> (task train RECIPE=supervised passes them through), "
                "or train with PPO."
            )
        print(f"── supervised training {env_slug} [{mode}] over {args.seeds} seeds, {args.epochs} epochs")
        net = train_supervised(
            engine=str(engine),
            source=source,
            build=build,
            seeds=args.seeds,
            epochs=args.epochs,
            time_limit_ticks=args.time_limit_ticks,
            seed=args.seed,
            device=args.device,
            out_dir=out_dir,
        )
        weights = out_dir / "policy.pt"
        save_weights(weights, net, "supervised")
        print(f"→ weights: {weights}")
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
            build=build,
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
            build=build,
        )
        weights = out_dir / "policy.pt"
        save_weights(weights, net, "ppo")
        print(f"→ weights: {weights}")

    onnx = export(net, out_dir / "policy.onnx")
    print(f"→ onnx: {onnx} ({onnx.stat().st_size} bytes)")

    diff = verify(net, onnx)
    print(f"✓ torch/onnxruntime parity: max abs diff {diff:.3e}")

    if args.from_weights:
        provenance_table = None
    elif recipe == "supervised":
        provenance_table = provenance(steps=args.seeds, num_envs=1, trained=True)
        provenance_table["recipe"] = "supervised"
    else:
        provenance_table = provenance(
            steps=args.steps,
            num_envs=1 if args.parallel else (args.num_envs or default_num_envs()),
            trained=True,
        )
        provenance_table["recipe"] = "ppo"
    bundle = stage(
        env_slug,
        mode,
        payload_schema_version,
        onnx,
        shell,
        bundle_dir,
        provenance_table=provenance_table,
    )
    print(f"→ bundle: {bundle}")
    print("\nRun it:   task match\nCompete:  task upload")


if __name__ == "__main__":
    sys.exit(main())
