"""The scaffolder's contract, hermetically (a synthetic declaration — no
network, no engine):

- the three languages are at PARITY: from ONE declaration every value,
  slice and column lands in each interface file with the same index,
  shape, doc, unit and code table, and every metric, tag and the reward
  lag reach every header;
- the generated python interface executes: typed views are
  declaration-shaped, column constants index the right column, the
  action builder maps declared units onto the shell's [-1, 1];
- the generated ``model.py`` builds a policy with one stream per value;
- re-running refreshes generated files but NEVER touches policy files;
- ``agent.toml`` round-trips through the loader, ``[supervised]`` block
  included, and the block is scaffolded exactly when the declaration
  reveals truth.

The rust/c scaffolds COMPILING to components is covered by
``test_scaffold_builds.py`` (toolchain-gated locally, unconditional in CI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lockstep_train.info import Budgets
from lockstep_train.wire import ColumnSpec, MetricSpec, SeatBrief, SeatInit, SliceSpec, ValueSpec

from train import agents as agents_mod
from train import scaffold as scaffold_mod
from train.agents import AgentConfig, list_agents, load_agent, resolve_agent, write_agent_toml

#: A column doc that carries its own code table (docs/wire.md: `unit =
#: "code"` and the table lives in that column's doc).
CODE_TABLE = "what you decided about the transaction:\n  0 = approve\n  1 = decline\n  2 = hold for review"


def synthetic_init() -> SeatInit:
    obs = [
        ValueSpec(
            name="marquee",
            dtype="u8",
            shape=(1, 4, 8),
            low=0.0,
            high=255.0,
            doc="a tiny grayscale strip, channel-first, row-major, y-down",
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
        # A feedback window: rank 2, documented columns, the documented
        # names — what the supervised block is scaffolded from.
        ValueSpec(
            name="feedback",
            dtype="f32",
            shape=(3, 5),
            low=-1.0,
            high=1000.0,
            doc="decisions that resolved this tick, newest first; zero rows are padding",
            columns=[
                ColumnSpec("age_ticks", "how many ticks ago the decision was made", "ticks"),
                ColumnSpec("your_decision", CODE_TABLE, "code"),
                ColumnSpec("truth", "1 when the transaction was fraudulent, else 0", "flag"),
                ColumnSpec("amount", "the transaction amount", "USD"),
                ColumnSpec("valid", "1 for a real row, 0 for padding", "flag"),
            ],
        ),
        ValueSpec(
            name="table",
            dtype="i32",
            shape=(2, 3),
            low=0.0,
            high=9.0,
            doc="one row per counter, index = counter id",
            columns=[
                ColumnSpec("hits", "count of hits", ""),
                ColumnSpec("misses", "count of misses", ""),
                ColumnSpec("kind", "0 = soft, 1 = hard", "code"),
            ],
        ),
    ]
    actions = [
        ValueSpec(
            name="decision",
            dtype="f32",
            shape=(3,),
            low=-2.0,
            high=2.0,
            doc="joint torques and a flag; on a tie the lower code wins",
            elem_bounds=(
                np.array([-1.0, -1.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0, 2.0], dtype=np.float32),
            ),
            slices=[
                SliceSpec(name="torque", start=0, len=2, doc="per-joint torque", unit="N·m"),
                SliceSpec(name="flag", start=2, len=1, doc="the decision flag", unit="code"),
            ],
        )
    ]
    metrics = [
        MetricSpec("score", "the 0-100 episode score", "", "score", "higher", 0.0, 100.0, True),
        MetricSpec("false-alarms", "declines of legitimate transactions", "count", "count", "lower"),
    ]
    return SeatInit(
        seat=0,
        obs=obs,
        actions=actions,
        meta=[
            ("control_hz", "50"),
            ("tags", "skill:classification, level:entry"),
            ("reward_lag_ticks", "40"),
        ],
        brief=SeatBrief(goal="reach the target", reward="progress", ends="time out"),
        metrics=metrics,
    )


BUDGETS = Budgets(
    tick_rate_hz=50,
    environment_version="1.2.3",
    payload_schema_version=7,
    min_ticks=1,
    max_ticks=100,
    roster_min=1,
    roster_max=1,
)


@pytest.fixture
def agent_env(tmp_path, monkeypatch):
    """Scaffold into an isolated agents/ root."""
    root = tmp_path / "agents"
    monkeypatch.setattr(agents_mod, "AGENTS_ROOT", root)
    return root


def cfg_for(lang: str, supervised: dict | None = None) -> AgentConfig:
    return AgentConfig(
        name="testling",
        env="fixture-env",
        mode="default",
        lang=lang,
        environment_version="1.2.3",
        payload_schema_version=7,
        supervised=supervised,
    )


#: Facts that must reach EVERY language's interface file, verbatim.
SHARED_FRAGMENTS = [
    # the header
    "reach the target",
    "progress",
    "time out",
    "score — the 0-100 episode score (higher is better, headline)",
    "false-alarms — declines of legitimate transactions (count, lower is",
    "skill:classification, level:entry",
    "40 ticks after a decision its truth reaches the reward",
    "tick rate: 50 Hz",
    "NEUTRAL action",
    "neutral: decision → per-element bounds midpoint",
    # every value: dtype, shape, element count, bounds, doc
    "u8[1, 4, 8] — 32 element(s), bounds [0, 255] every element",
    "f32[5] — 5 element(s), bounds [-1, 1] every element",
    "f32[3, 5] — 15 element(s), bounds [-1, 1000] every element",
    "i32[2, 3] — 6 element(s), bounds [0, 9] every element",
    "f32[3] — 3 element(s), per-element bounds (see below): overall [-1, 2]",
    "a tiny grayscale strip, channel-first, row-major, y-down",
    "decisions that resolved this tick, newest first; zero rows are padding",
    "joint torques and a flag; on a tie the lower code wins",
    # slices and columns: doc + unit, and the code table verbatim
    "joint angles [rad]",
    "joint velocities [rad/s]",
    "per-joint torque [N·m] — [-1, 1] per element",
    "the decision flag [code] — [0, 2] per element",
    "how many ticks ago the decision was made [ticks]",
    "what you decided about the transaction: [code]",
    "  0 = approve",
    "  1 = decline",
    "  2 = hold for review",
    "1 when the transaction was fraudulent, else 0 [flag]",
    "0 = soft, 1 = hard [code]",
]

EXPECTED_FRAGMENTS = {
    # The same constants, spelled per language.
    "python": [
        "OBS_MARQUEE_INDEX = 0",
        "OBS_MARQUEE_SHAPE = (1, 4, 8)",
        "OBS_BODY_JOINT_POS = slice(0, 2)",
        "OBS_BODY_JOINT_VEL = slice(2, 4)",
        "OBS_BODY_TIME_LEFT = slice(4, 5)",
        "OBS_FEEDBACK_ROWS = 3",
        "OBS_FEEDBACK_COLS = 5",
        "OBS_FEEDBACK_COL_AGE_TICKS = 0",
        "OBS_FEEDBACK_COL_YOUR_DECISION = 1",
        "OBS_FEEDBACK_COL_TRUTH = 2",
        "OBS_FEEDBACK_COL_VALID = 4",
        "OBS_TABLE_COL_KIND = 2",
        "ACT_DECISION_TORQUE = slice(0, 2)",
        "ACT_DECISION_ELEM_LOW = [-1.0, -1.0, 0.0]",
        "PAYLOAD_SCHEMA_VERSION = 7",
        "class Obs:",
        "def action(*, torque=None, flag=None):",
        "def normalized(*, torque=None, flag=None) -> np.ndarray:",
    ],
    "rust": [
        "pub const INDEX: usize = 0;",
        "pub const SHAPE: [u32; 3] = [1, 4, 8];",
        "pub const JOINT_POS: core::ops::Range<usize> = 0..2;",
        "pub const JOINT_VEL: core::ops::Range<usize> = 2..4;",
        "pub const TIME_LEFT: core::ops::Range<usize> = 4..5;",
        "pub const ROWS: usize = 3;",
        "pub const COLS: usize = 5;",
        "pub const COL_AGE_TICKS: usize = 0;",
        "pub const COL_YOUR_DECISION: usize = 1;",
        "pub const COL_TRUTH: usize = 2;",
        "pub const COL_VALID: usize = 4;",
        "pub const COL_KIND: usize = 2;",
        "pub const TORQUE: core::ops::Range<usize> = 0..2;",
        "pub const ELEM_LOW: [f32; 3] = [-1.0, -1.0, 0.0];",
        "PAYLOAD_SCHEMA_VERSION: u32 = 7",
        "pub struct Obs<'a>",
        "pub fn feedback(&self) -> &[[f32; 5]; 3]",
        "pub fn table(&self) -> &[[i32; 3]; 2]",
        "pub fn marquee(&self) -> &'a [u8]",
        "pub fn body(&self) -> &[f32; 5]",
        "pub struct Action {",
        "pub torque: [f32; 2],",
        "pub flag: f32,",
        "pub fn encode(&self) -> Vec<u8>",
    ],
    "c": [
        "#define OBS_MARQUEE_INDEX 0",
        "OBS_MARQUEE_SHAPE[3] AGENT_UNUSED = {1, 4, 8};",
        "#define OBS_BODY_JOINT_POS_START 0",
        "#define OBS_BODY_JOINT_POS_LEN 2",
        "#define OBS_BODY_TIME_LEFT_START 4",
        "#define OBS_FEEDBACK_ROWS 3",
        "#define OBS_FEEDBACK_COLS 5",
        "#define OBS_FEEDBACK_COL_AGE_TICKS 0",
        "#define OBS_FEEDBACK_COL_YOUR_DECISION 1",
        "#define OBS_FEEDBACK_COL_TRUTH 2",
        "#define OBS_FEEDBACK_COL_VALID 4",
        "#define OBS_TABLE_COL_KIND 2",
        "#define ACT_DECISION_TORQUE_START 0",
        "#define ACT_DECISION_TORQUE_LEN 2",
        "ACT_DECISION_ELEM_LOW[3] AGENT_UNUSED = {-1.0f, -1.0f, 0.0f};",
        "#define AGENT_PAYLOAD_SCHEMA_VERSION 7",
        "static inline int obs_feedback(const wire_view_t *v, float out[3][5])",
        "static inline int obs_table(const wire_view_t *v, int32_t out[2][3])",
        "static inline const uint8_t *obs_marquee(const wire_view_t *v)",
        "static inline int obs_body(const wire_view_t *v, float out[5])",
        "float torque[2];",
        "float flag;",
        "} agent_action_t;",
        "static inline void agent_action_neutral(agent_action_t *a)",
        "static inline void agent_action_encode(const agent_action_t *a, uint8_t **ptr, size_t *len)",
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
def test_every_fact_lands_in_every_interface_file(agent_env, lang):
    cfg = cfg_for(lang)
    GENERATOR[lang](cfg, synthetic_init(), BUDGETS, "fixture-env · default")
    text = (cfg.dir / INTERFACE_FILE[lang]).read_text(encoding="utf-8")
    for fragment in SHARED_FRAGMENTS + EXPECTED_FRAGMENTS[lang]:
        assert fragment in text, f"{lang} interface is missing {fragment!r}"


@pytest.mark.parametrize("lang", ["python", "rust", "c"])
def test_the_stub_reads_through_the_typed_accessor_and_answers_neutral(agent_env, lang):
    cfg = cfg_for(lang)
    GENERATOR[lang](cfg, synthetic_init(), BUDGETS, "t")
    stub = (cfg.dir / POLICY_FILE[lang]).read_text(encoding="utf-8")
    expect = {
        "python": ["iface.OBS_MARQUEE_INDEX", "iface.normalized()"],
        "rust": ["Obs::read(&view)", "obs.marquee()", "Action::neutral().encode()"],
        "c": ["obs_marquee(&view)", "agent_action_neutral(&action)", "agent_action_encode(&action"],
    }[lang]
    for fragment in expect:
        assert fragment in stub, f"{lang} stub is missing {fragment!r}"


def test_python_interface_executes_with_typed_views_and_builder(agent_env):
    cfg = cfg_for("python")
    scaffold_mod.scaffold_python(cfg, synthetic_init(), BUDGETS, "t")
    ns: dict = {}
    exec((cfg.dir / "interface.py").read_text(encoding="utf-8"), ns)
    assert ns["OBS_BODY"] == "body"
    assert ns["OBS_BODY_JOINT_VEL"] == slice(2, 4)
    assert ns["OBS_MARQUEE_SHAPE"] == (1, 4, 8)
    assert ns["OBS_FEEDBACK_COL_TRUTH"] == 2
    assert ns["ACT_DECISION_TORQUE"] == slice(0, 2)
    # neutral action: per-element bounds midpoint — [-1,1]→0, [-1,1]→0, [0,2]→1
    assert ns["NEUTRAL_ACTION"] == [[0.0, 0.0, 1.0]]

    feedback = np.zeros((3, 5), dtype=np.float32)
    feedback[0] = [4, 1, 1, 250.0, 1]
    obs = ns["Obs"](
        {
            "marquee": np.zeros((1, 4, 8), dtype=np.uint8),
            "body": np.arange(5, dtype=np.float32),
            "feedback": feedback,
            "table": np.array([[1, 2, 0], [3, 4, 1]], dtype=np.int32),
        }
    )
    assert obs.marquee.shape == (1, 4, 8) and obs.marquee.dtype == np.uint8
    np.testing.assert_array_equal(obs.body_joint_vel, [2.0, 3.0])
    np.testing.assert_array_equal(obs.feedback_truth, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(obs.feedback_age_ticks, [4.0, 0.0, 0.0])
    np.testing.assert_array_equal(obs.table_kind, [0, 1])
    assert obs.table.dtype == np.int32

    # The builder: declared units in; missing fields are neutral.
    np.testing.assert_allclose(ns["action"](torque=[0.5, 0.25]), [0.5, 0.25, 1.0])
    np.testing.assert_allclose(ns["action"](), [0.0, 0.0, 1.0])
    # The ONNX-output twin: the shell's affine map, inverted, per element.
    np.testing.assert_allclose(ns["normalized"](torque=[0.5, 0.25], flag=2.0), [0.5, 0.25, 1.0])
    np.testing.assert_allclose(ns["normalized"](), [0.0, 0.0, 0.0])


def test_generated_model_builds_one_stream_per_value(agent_env):
    torch = pytest.importorskip("torch")
    from gymnasium import spaces

    from train.agents import import_agent_module, policy_builder
    from train.core.policy import Policy

    cfg = cfg_for("python")
    scaffold_mod.scaffold_python(cfg, synthetic_init(), BUDGETS, "t")
    text = (cfg.dir / "model.py").read_text(encoding="utf-8")
    assert '"feedback": flat_stream((3, 5))' in text
    assert '"marquee": flat_stream((1, 4, 8))' in text
    assert "nn.Conv2d(1, 16" in text, "the u8 rank-3 value gets a commented conv example"
    build = policy_builder(cfg)
    assert build is not Policy
    obs = spaces.Dict(
        {
            "marquee": spaces.Box(0, 255, (1, 4, 8), dtype=np.uint8),
            "body": spaces.Box(-1, 1, (5,), dtype=np.float32),
            "feedback": spaces.Box(-1, 1000, (3, 5), dtype=np.float32),
            "table": spaces.Box(0, 9, (2, 3), dtype=np.int32),
        }
    )
    net = build(obs, spaces.Box(-1, 1, (3,), dtype=np.float32))
    assert isinstance(net, Policy)
    assert set(net.streams.keys()) == {"marquee", "body", "feedback", "table"}
    inputs = [
        torch.zeros(2, *obs[n].shape, dtype=torch.int32 if n == "table" else torch.float32)
        for n in net.input_names
    ]
    assert net(*inputs).shape == (2, 3)
    # The scripted policy imports the generated interface and answers neutral.
    policy = import_agent_module(cfg, "policy")
    scripted = policy.ScriptedPolicy(
        {n: obs[n].shape for n in net.input_names},
        {n: net.input_dtypes[n] for n in net.input_names},
        3,
    )
    out = scripted(*inputs)
    assert out.shape == (2, 3) and float(out.abs().max()) == 0.0


@pytest.mark.parametrize("lang", ["python", "rust", "c"])
def test_regeneration_refreshes_generated_but_never_policy(agent_env, lang):
    cfg = cfg_for(lang)
    init = synthetic_init()
    GENERATOR[lang](cfg, init, BUDGETS, "t")
    policy = cfg.dir / POLICY_FILE[lang]
    iface = cfg.dir / INTERFACE_FILE[lang]
    policy.write_text("# MY EDITS\n" + policy.read_text(encoding="utf-8"), encoding="utf-8")
    iface_marker = "SHOULD BE REGENERATED AWAY"
    iface.write_text(iface_marker, encoding="utf-8")
    if lang == "python":
        model = cfg.dir / "model.py"
        model.write_text("# MY NET\n" + model.read_text(encoding="utf-8"), encoding="utf-8")
    GENERATOR[lang](cfg, init, BUDGETS, "t")
    assert policy.read_text(encoding="utf-8").startswith("# MY EDITS"), f"{lang} policy was clobbered"
    assert iface_marker not in iface.read_text(encoding="utf-8"), f"{lang} interface was not regenerated"
    if lang == "python":
        assert (cfg.dir / "model.py").read_text(encoding="utf-8").startswith("# MY NET"), "model.py was clobbered"


def test_supervised_block_follows_the_declaration():
    init = synthetic_init()
    block = scaffold_mod.supervised_block(init)
    assert block == {
        "value": "feedback",
        "age_col": "age_ticks",
        "decision_col": "your_decision",
        "truth_col": "truth",
        "valid_col": "valid",
        "action": "decision",
    }
    # No feedback window: the reward lag from meta is the source.
    init.obs = [t for t in init.obs if t.name != "feedback"]
    assert scaffold_mod.supervised_block(init) == {"reward_lag_ticks": 40, "action": "decision"}
    # Neither: no block at all.
    init.meta = [(k, v) for k, v in init.meta if k != "reward_lag_ticks"]
    assert scaffold_mod.supervised_block(init) is None


def test_agent_toml_round_trips_with_the_supervised_block(agent_env):
    cfg = cfg_for("python", supervised=scaffold_mod.supervised_block(synthetic_init()))
    write_agent_toml(cfg)
    back = load_agent("testling", agents_mod.AGENTS_ROOT)
    assert back == cfg
    assert back.supervised["truth_col"] == "truth"
    plain = cfg_for("rust")
    write_agent_toml(plain)
    assert load_agent("testling", agents_mod.AGENTS_ROOT).supervised is None


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
