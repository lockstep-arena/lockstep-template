"""The scaffolder's contract, hermetically (a synthetic declaration — no
network, no engine):

- every declared value and slice appears in each language's interface file
  with its doc, unit and index range;
- the generated python interface actually executes and its slices are right;
- re-running refreshes generated files but NEVER touches policy files;
- ``agent.toml`` round-trips through the loader.

The rust/c scaffolds COMPILING to components is covered by
``test_scaffold_builds.py`` (toolchain-gated locally, unconditional in CI).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lockstep_train.info import Budgets
from lockstep_train.wire import SeatBrief, SeatInit, SliceSpec, ValueSpec

from train import agents as agents_mod
from train import scaffold as scaffold_mod
from train.agents import AgentConfig, list_agents, load_agent, resolve_agent, write_agent_toml


def synthetic_init() -> SeatInit:
    obs = [
        ValueSpec(
            name="marquee",
            dtype="u8",
            shape=(1, 4, 8),
            low=0.0,
            high=255.0,
            doc="a tiny grayscale strip, row-major, y-down",
        ),
        ValueSpec(
            name="body",
            dtype="f32",
            shape=(5,),
            low=-1.0,
            high=1.0,
            doc="the body state",
            slices=[
                SliceSpec(name="joint_pos", start=0, len=2, doc="joint angles", unit="rad"),
                SliceSpec(name="joint_vel", start=2, len=2, doc="joint velocities", unit="rad/s"),
                SliceSpec(name="time_left", start=4, len=1, doc="episode fraction left", unit=""),
            ],
        ),
    ]
    actions = [
        ValueSpec(
            name="action",
            dtype="f32",
            shape=(3,),
            low=-1.0,
            high=1.0,
            doc="joint torques",
            slices=[SliceSpec(name="torque", start=0, len=3, doc="per-joint torque", unit="N·m")],
        )
    ]
    return SeatInit(
        seat=0,
        obs=obs,
        actions=actions,
        meta=[("control_hz", "50")],
        brief=SeatBrief(goal="reach the target", reward="progress", ends="time out"),
    )


BUDGETS = Budgets(tick_rate_hz=50, environment_version="1.2.3", payload_schema_version=7)


@pytest.fixture
def agent_env(tmp_path, monkeypatch):
    """Scaffold into an isolated agents/ root."""
    root = tmp_path / "agents"
    monkeypatch.setattr(agents_mod, "AGENTS_ROOT", root)
    return root


def cfg_for(lang: str) -> AgentConfig:
    return AgentConfig(
        name="testling",
        env="fixture-env",
        mode="default",
        lang=lang,
        environment_version="1.2.3",
        payload_schema_version=7,
    )


EXPECTED_FRAGMENTS = {
    # (name-ish, doc, unit) fragments that must appear in EVERY language's
    # interface file, spelled per that language's constant style.
    "python": [
        "OBS_MARQUEE_INDEX = 0",
        "OBS_BODY_JOINT_POS = slice(0, 2)",
        "OBS_BODY_JOINT_VEL = slice(2, 4)",
        "OBS_BODY_TIME_LEFT = slice(4, 5)",
        "ACT_ACTION_TORQUE = slice(0, 3)",
        "joint angles",
        "[rad]",
        "[rad/s]",
        "per-joint torque",
        "[N·m]",
        "reach the target",
        "PAYLOAD_SCHEMA_VERSION = 7",
    ],
    "rust": [
        "pub const INDEX: usize = 0;",
        "pub const JOINT_POS: core::ops::Range<usize> = 0..2;",
        "pub const JOINT_VEL: core::ops::Range<usize> = 2..4;",
        "pub const TIME_LEFT: core::ops::Range<usize> = 4..5;",
        "pub const TORQUE: core::ops::Range<usize> = 0..3;",
        "joint angles",
        "[rad]",
        "per-joint torque",
        "reach the target",
        "PAYLOAD_SCHEMA_VERSION: u32 = 7",
    ],
    "c": [
        "#define OBS_MARQUEE_INDEX 0",
        "#define OBS_BODY_JOINT_POS_START 0",
        "#define OBS_BODY_JOINT_POS_LEN 2",
        "#define OBS_BODY_TIME_LEFT_START 4",
        "#define ACT_ACTION_TORQUE_START 0",
        "#define ACT_ACTION_TORQUE_LEN 3",
        "joint angles",
        "[rad]",
        "per-joint torque",
        "reach the target",
        "#define AGENT_PAYLOAD_SCHEMA_VERSION 7",
    ],
}

INTERFACE_FILE = {
    "python": Path("interface.py"),
    "rust": Path("src/interface.rs"),
    "c": Path("interface.h"),
}

GENERATOR = {
    "python": scaffold_mod.scaffold_python,
    "rust": scaffold_mod.scaffold_rust,
    "c": scaffold_mod.scaffold_c,
}

POLICY_FILE = {
    "python": Path("policy.py"),
    "rust": Path("src/lib.rs"),
    "c": Path("agent.c"),
}


@pytest.mark.parametrize("lang", ["python", "rust", "c"])
def test_every_slice_lands_in_the_interface_file(agent_env, lang):
    cfg = cfg_for(lang)
    GENERATOR[lang](cfg, synthetic_init(), BUDGETS, "fixture-env · default")
    text = (cfg.dir / INTERFACE_FILE[lang]).read_text()
    for fragment in EXPECTED_FRAGMENTS[lang]:
        assert fragment in text, f"{lang} interface is missing {fragment!r}"


def test_python_interface_executes_and_slices_are_right(agent_env):
    cfg = cfg_for("python")
    scaffold_mod.scaffold_python(cfg, synthetic_init(), BUDGETS, "t")
    ns: dict = {}
    exec((cfg.dir / "interface.py").read_text(), ns)
    assert ns["OBS_BODY"] == "body"
    assert ns["OBS_BODY_JOINT_VEL"] == slice(2, 4)
    assert ns["OBS_MARQUEE_SHAPE"] == (1, 4, 8)
    assert ns["ACT_ACTION_TORQUE"] == slice(0, 3)
    # neutral action: bounds midpoint of [-1, 1] = 0 for all 3 elements
    assert ns["NEUTRAL_ACTION"] == [[0.0, 0.0, 0.0]]


@pytest.mark.parametrize("lang", ["python", "rust", "c"])
def test_regeneration_refreshes_generated_but_never_policy(agent_env, lang):
    cfg = cfg_for(lang)
    init = synthetic_init()
    GENERATOR[lang](cfg, init, BUDGETS, "t")
    policy = cfg.dir / POLICY_FILE[lang]
    iface = cfg.dir / INTERFACE_FILE[lang]
    policy.write_text("# MY EDITS\n" + policy.read_text())
    iface_marker = "SHOULD BE REGENERATED AWAY"
    iface.write_text(iface_marker)
    GENERATOR[lang](cfg, init, BUDGETS, "t")
    assert policy.read_text().startswith("# MY EDITS"), f"{lang} policy was clobbered"
    assert iface_marker not in iface.read_text(), f"{lang} interface was not regenerated"


def test_agent_toml_round_trips(agent_env):
    cfg = cfg_for("rust")
    write_agent_toml(cfg)
    back = load_agent("testling", agents_mod.AGENTS_ROOT)
    assert back == cfg


def test_resolve_agent_defaults_and_errors(agent_env):
    with pytest.raises(SystemExit, match="no agents yet"):
        resolve_agent(None, agents_mod.AGENTS_ROOT)
    write_agent_toml(cfg_for("python"))
    assert resolve_agent(None, agents_mod.AGENTS_ROOT).name == "testling"
    two = AgentConfig(
        name="other",
        env="fixture-env",
        mode="default",
        lang="c",
        environment_version="1.2.3",
        payload_schema_version=7,
    )
    write_agent_toml(two)
    assert len(list_agents(agents_mod.AGENTS_ROOT)) == 2
    with pytest.raises(SystemExit, match="say which"):
        resolve_agent(None, agents_mod.AGENTS_ROOT)
    assert resolve_agent("other", agents_mod.AGENTS_ROOT).lang == "c"
