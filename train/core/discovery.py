"""Find installed game training packages and gate on the contract version.

Game packages declare a ``lockstep.training_games`` entry point resolving to
a zero-arg ``game_spec()`` that returns the GameSpec dict (the versioned
training-metadata contract). This module is the ONLY place the template
touches a game package, and it does so without ever importing one by name —
``pip install lockstep-game-<slug>`` is all it takes for a game to appear.

The spec (training contract v1) carries:

- ``training_contract_version``: int — gated against
  :data:`SUPPORTED_CONTRACT_VERSIONS` below. The contract is EXPECTED to
  widen as future games need shapes it does not cover; the gate turns each
  widening into a clear "upgrade the template" instead of a silent break.
- ``slug``, ``env_id``, ``default_mode``
- ``modes``: per-mode ``payload_schema_version`` (int) + ``engine_url`` (str)
- ``agent_component_path``: callable(mode) -> Path to the mode's prebuilt
  agent-shell component

Contract v2 adds (additively — a v1 spec is a valid v2 spec):

- ``parallel_env_id``: ``str | None`` — for games whose seats are genuinely
  adversarial, a ``module:callable`` locator of a PettingZoo ``ParallelEnv``
  factory where every seat learns at once (``task train PARALLEL=1``; see
  :mod:`train.core.self_play`). ``None``/absent = single-agent only.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

ENTRY_POINT_GROUP = "lockstep.training_games"

#: Training-contract versions this template understands. v2 only ADDS the
#: optional ``parallel_env_id`` key, so accepting both costs nothing — and a
#: v3 that changes something this loop relies on will be refused here.
SUPPORTED_CONTRACT_VERSIONS = frozenset({1, 2})

#: Keys a v1 spec must carry (checked so a broken package fails with a
#: message instead of an AttributeError three modules later).
_REQUIRED_KEYS = ("training_contract_version", "slug", "env_id", "default_mode", "modes")


def installed_games() -> dict[str, importlib.metadata.EntryPoint]:
    """slug -> entry point, for every installed game training package."""
    return {
        ep.name: ep
        for ep in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    }


def resolve_game(slug: str | None) -> tuple[dict[str, Any], str]:
    """Like :func:`load_game`, but ``None`` means "the installed game".

    With exactly one game package installed there is nothing to choose;
    with zero or several, the caller has to say (``--game`` / ``GAME=``).
    """
    if slug:
        return load_game(slug)
    games = installed_games()
    if len(games) == 1:
        return load_game(next(iter(games)))
    installed = ", ".join(sorted(games)) if games else "none"
    raise SystemExit(
        f"pass --game <slug> (installed games: {installed})"
        + ("" if games else " — install one with: task setup GAME=<slug>")
    )


def load_game(slug: str) -> tuple[dict[str, Any], str]:
    """Load ``slug``'s GameSpec. Returns ``(spec, module_name)``.

    ``module_name`` is the module the entry point lives in — spawn-safe env
    factories re-import it in worker processes, because env registration is
    an import side effect and spawned workers start blank.
    """
    games = installed_games()
    if slug not in games:
        installed = ", ".join(sorted(games)) if games else "none"
        raise SystemExit(
            f"no training package provides game {slug!r} "
            f"(installed games: {installed}).\n"
            f"Install it with:  pip install lockstep-game-{slug}\n"
            f"           or:    task setup GAME={slug}"
        )
    ep = games[slug]
    spec = ep.load()()

    missing = [k for k in _REQUIRED_KEYS if k not in spec]
    if missing:
        raise SystemExit(
            f"game package {slug!r} returned a GameSpec missing {missing} — "
            "the package is broken or predates the training contract"
        )
    version = spec["training_contract_version"]
    if version not in SUPPORTED_CONTRACT_VERSIONS:
        supported = ", ".join(f"v{v}" for v in sorted(SUPPORTED_CONTRACT_VERSIONS))
        raise SystemExit(
            f"game package {slug!r} speaks training contract v{version}; this "
            f"template supports {supported}. Upgrade the template "
            "(git pull) — or install an older release of the game package."
        )
    return spec, ep.module


def parallel_env_id(spec: dict[str, Any]) -> str | None:
    """The spec's PettingZoo factory locator, or ``None`` for single-agent
    games (and for every v1 spec, which predates the key)."""
    return spec.get("parallel_env_id") or None
