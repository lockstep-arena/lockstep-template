"""What to type next, printed at the end of `task build` and `task train`.

    Run it:               task match AGENT=my-bot   (vs. other bots: OPPONENTS="other-bot")
    Debug it:             task match AGENT=my-bot LOGS=1 STEP=1
    Submit final bundle:  task upload AGENT=my-bot

The opponents hint appears only when the environment seats more than one
agent, and it names as many opponents as the roster has room for — read off
the engine's own declaration, never assumed.
"""

from __future__ import annotations

from pathlib import Path


def roster_max(engine: Path) -> int | None:
    """The most seats this engine's mode takes, or ``None`` when the engine
    cannot be asked (the hint then leaves opponents out)."""
    try:
        from lockstep_train.info import from_engine

        _, budgets, _ = from_engine(str(engine), seat=0)
    except Exception:  # noqa: BLE001 — a hint must never fail a build
        return None
    return budgets.roster_max


def opponents_example(max_seats: int | None) -> str | None:
    """``OPPONENTS="…"`` with one placeholder per free seat, or ``None`` for
    an environment where your agent plays alone."""
    if not max_seats or max_seats < 2:
        return None
    others = max_seats - 1
    if others == 1:
        names = "other-bot"
    elif others <= 3:
        names = " ".join(f"bot-{i}" for i in range(2, max_seats + 1))
    else:
        names = f"bot-2 bot-3 … bot-{max_seats}"
    return f'OPPONENTS="{names}"'


def next_steps(agent: str | None, engine: Path | None) -> str:
    """The closing lines of a build: run, debug, submit."""
    who = f" AGENT={agent}" if agent else ""
    run = f"task match{who}"
    opponents = opponents_example(roster_max(engine)) if engine else None
    run_line = f"{run}   (vs. other bots: {opponents})" if opponents else run
    return "\n".join(
        [
            f"Run it:               {run_line}",
            f"Debug it:             {run} LOGS=1 STEP=1",
            f"Submit final bundle:  task upload{who}",
        ]
    )
