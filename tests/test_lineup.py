"""`task match OPPONENTS="…"`: who sits where, and every refusal names the fix."""

from __future__ import annotations

import pytest

from train import agents as agents_mod
from train.agents import AgentConfig
from train.lineup import lineup


def agent(root, name, env="invented-env", mode="default", built=True) -> AgentConfig:
    cfg = AgentConfig(name, env, mode, "rust", "0.1.0", 1)
    if built:
        (root / name / "out" / "bundle").mkdir(parents=True)
        (root / name / "out" / "bundle" / "lockstep.toml").write_text("", encoding="utf-8")
    return cfg


@pytest.fixture
def root(tmp_path, monkeypatch):
    root = tmp_path / "agents"
    monkeypatch.setattr(agents_mod, "AGENTS_ROOT", root)
    monkeypatch.chdir(tmp_path)
    return root


def names(paths):
    return [p.parent.parent.name for p in paths]


def test_seat_zero_is_yours_then_opponents_in_order(root):
    me, a, b = agent(root, "me"), agent(root, "a"), agent(root, "b")
    assert names(lineup(me, [b, a], roster_min=2, roster_max=4)) == ["me", "b", "a"]


def test_your_agent_fills_seats_the_roster_requires(root):
    me, a = agent(root, "me"), agent(root, "a")
    assert names(lineup(me, [a], roster_min=4, roster_max=4)) == ["me", "a", "me", "me"]


def test_too_many_opponents_is_refused_with_the_room_left(root):
    me, a, b = agent(root, "me"), agent(root, "a"), agent(root, "b")
    with pytest.raises(SystemExit, match="room for 1 opponent"):
        lineup(me, [a, b], roster_min=2, roster_max=2)
    with pytest.raises(SystemExit, match="room for 0 opponent"):
        lineup(me, [a], roster_min=1, roster_max=1)


def test_an_opponent_for_another_mode_is_refused(root):
    me, other = agent(root, "me"), agent(root, "other", mode="hard")
    with pytest.raises(SystemExit, match=r"other is an agent for invented-env \[hard\]"):
        lineup(me, [other], roster_min=2, roster_max=2)


def test_an_unbuilt_opponent_says_how_to_build_it(root):
    me, lazy = agent(root, "me"), agent(root, "lazy", built=False)
    with pytest.raises(SystemExit, match="task build AGENT=lazy"):
        lineup(me, [lazy], roster_min=2, roster_max=2)
