"""train -> export -> parity-check -> stage the submittable agent bundle.

One command (``task train`` runs it), for ANY installed game::

    python -m train.main --game <slug> --steps 8192 --engine out/engine.wasm

The game is discovered through the ``lockstep.training_games`` entry point —
``pip install lockstep-game-<slug>`` is all it takes for ``--game <slug>``
to work; this repo names no game. The staged bundle is what the platform
actually consumes — and what ``lockstep match run`` / ``lockstep agent
upload`` take directly::

    out/agent-bundle/
      lockstep.toml        declares the `policy` artifact by NAME
      component.wasm       the mode's prebuilt agent shell (from the wheel)
      artifacts/policy.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .core import utf8_output
from .core.discovery import parallel_env_id, resolve_game
from .core.export import export, verify
from .core.policy import policy_from_signature
from .core.self_play import train_self_play
from .core.stage import stage
from .core.train import default_num_envs, train

OUT_DIR = Path("out")
BUNDLE_DIR = OUT_DIR / "agent-bundle"


def main() -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--game",
        default=None,
        help="game slug; the matching lockstep-game-<slug> package must be "
        "installed (task setup GAME=<slug>). Defaults to the one installed "
        "game when there is exactly one.",
    )
    p.add_argument(
        "--mode",
        default=None,
        help="game mode key; defaults to the game's default mode",
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
        "--engine",
        required=True,
        help="engine wasm: a local path (task engine downloads one) or an https URL",
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
        "--opponent",
        default=None,
        help="fill seat 1 with a frozen policy (.onnx path, e.g. a previous "
        "run's out/agent-bundle/artifacts/policy.onnx) instead of the idle "
        "baseline — the self-play hook. Only for games whose env has an "
        "opponent seat to fill; other games refuse with a message",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="parallel engine instances for collection; 1 = in-process "
        f"(debuggable). Default: min(8, cores-2) = {default_num_envs()} here",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="self-play with BOTH seats learning at once: one policy trained "
        "on every seat's experience of the game's PettingZoo parallel env "
        "(train/core/self_play.py). Only for games that declare one "
        "(training contract v2, adversarial seats); others refuse with a "
        "message. Incompatible with --opponent and --num-envs",
    )
    args = p.parse_args()

    spec, spec_module = resolve_game(args.game)
    mode = args.mode or spec["default_mode"]
    if mode not in spec["modes"]:
        raise SystemExit(
            f"game {spec['slug']!r} has no mode {mode!r} "
            f"(modes: {', '.join(sorted(spec['modes']))})"
        )

    if args.parallel:
        # Pre-flight the contract: the spec says whether the game has a
        # parallel env at all, so the refusal can name the game up front.
        if not parallel_env_id(spec):
            raise SystemExit(
                f"--parallel: {spec['slug']!r} declares no parallel env — its "
                "seats are not adversarial (training contract "
                f"v{spec['training_contract_version']}, no parallel_env_id). "
                "Train one seat with plain `task train`."
            )
        if args.opponent:
            raise SystemExit(
                "--parallel and --opponent are different self-play rungs: with "
                "--parallel seat 1 IS the learning policy, not a frozen one. "
                "Pick one."
            )
        if args.num_envs not in (None, 1):
            raise SystemExit(
                "--parallel drives ONE parallel env (PettingZoo has no vector "
                "API; the engine is not the bottleneck) — drop --num-envs"
            )

    if args.opponent:
        # Pre-flight here, where the error can name the game: inside a
        # spawned env worker this surfaces as a pickled TypeError.
        import inspect

        import gymnasium

        env_spec = gymnasium.registry.get(spec["env_id"])
        ctor = env_spec.entry_point
        if isinstance(ctor, str):
            module, _, attr = ctor.partition(":")
            import importlib

            ctor = getattr(importlib.import_module(module), attr)
        if "opponent" not in inspect.signature(ctor).parameters:
            raise SystemExit(
                f"--opponent: {spec['slug']!r} has no opponent seat to fill "
                "(its env takes no `opponent` option)"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_weights:
        print(f"── loading weights {args.from_weights}")
        blob = torch.load(args.from_weights, weights_only=True, map_location="cpu")
        net = policy_from_signature(blob["spaces"])
        net.load_state_dict(blob["state_dict"])
    elif args.parallel:
        print(
            f"── self-play training {spec['slug']} [{mode}] for {args.steps} "
            "seat-steps (both seats learning, one shared policy)"
        )
        net = train_self_play(
            spec=spec,
            mode=mode,
            steps=args.steps,
            engine=args.engine,
            time_limit_ticks=args.time_limit_ticks,
            seed=args.seed,
            device=args.device,
            out_dir=OUT_DIR,
            resume=args.resume,
        )
    else:
        print(f"── training {spec['slug']} [{mode}] for {args.steps} steps")
        net = train(
            spec=spec,
            spec_module=spec_module,
            mode=mode,
            steps=args.steps,
            engine=args.engine,
            time_limit_ticks=args.time_limit_ticks,
            num_envs=args.num_envs,
            seed=args.seed,
            device=args.device,
            out_dir=OUT_DIR,
            resume=args.resume,
            opponent=args.opponent,
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

    bundle = stage(spec, mode, onnx, BUNDLE_DIR)
    print(f"→ bundle: {bundle}")
    print("\nRun it:   task match\nCompete:  task upload")


if __name__ == "__main__":
    sys.exit(main())
