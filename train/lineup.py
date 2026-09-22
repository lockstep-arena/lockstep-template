"""Who sits where in `task match`: your agent in seat 0, then the agents
named in ``OPPONENTS="…"``, in that order.

    python -m train.lineup [--agent NAME] --opponents "bot-2 bot-3"

prints the bundle paths for every seat, space-separated, for the Taskfile to
hand to `lockstep match run --agents`. Every opponent must be an agent under
``agents/`` built for the same environment and mode as yours; there can be at
most as many as the roster has free seats; and when the environment needs
more seats than were named, your agent fills the rest.

Without opponents `task match` does not call this: your agent takes every
seat, as before.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agents import AgentConfig, resolve_agent


def lineup(
    agent: AgentConfig,
    opponents: list[AgentConfig],
    roster_min: int,
    roster_max: int,
) -> list[Path]:
    """Bundle path per seat, or ``SystemExit`` saying exactly what is wrong."""
    for opp in opponents:
        if (opp.env, opp.mode) != (agent.env, agent.mode):
            raise SystemExit(
                f"OPPONENTS: {opp.name} is an agent for {opp.env} [{opp.mode}], but "
                f"{agent.name} plays {agent.env} [{agent.mode}] — every seat in a match "
                "plays the same environment and mode"
            )
    free = roster_max - 1
    if len(opponents) > free:
        seats = "seat" if roster_max == 1 else "seats"
        raise SystemExit(
            f"OPPONENTS: {len(opponents)} named, but {agent.env} [{agent.mode}] seats at "
            f"most {roster_max} {seats} — room for {free} opponent(s) besides {agent.name}"
        )
    seats = [agent, *opponents]
    while len(seats) < roster_min:
        seats.append(agent)
    for cfg in dict.fromkeys(seats):
        if not (cfg.bundle_dir / "lockstep.toml").is_file():
            raise SystemExit(f"no bundle for {cfg.name} yet — build it first: task build AGENT={cfg.name}")
    return [cfg.bundle_dir for cfg in seats]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", default=None, help="your agent (AGENT=; default the sole agent)")
    p.add_argument("--opponents", required=True, help='space-separated agent names (OPPONENTS="…")')
    args = p.parse_args(argv)

    agent = resolve_agent(args.agent)
    names = args.opponents.split()
    if not names:
        raise SystemExit('OPPONENTS is empty — name at least one agent: OPPONENTS="other-bot"')
    opponents = [resolve_agent(n) for n in names]

    from lockstep_train.info import from_engine

    from .core.engine import ensure_engine

    engine = ensure_engine(agent.env, agent.mode).engine
    _, budgets, _ = from_engine(str(engine), seat=0)
    roster_min = budgets.roster_min or 1
    roster_max = budgets.roster_max or roster_min
    print(" ".join(str(p) for p in lineup(agent, opponents, roster_min, roster_max)))


if __name__ == "__main__":
    sys.exit(main())
