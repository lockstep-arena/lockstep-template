"""The supervised recipe, with no engine: a synthetic labelled stream
stands in for the environment.

A fake feedback window (the documented ``age_ticks`` / ``your_decision``
/ ``truth`` / ``valid`` columns) is joined by ``lockstep_train``'s
``LabelledStream`` — the same join the real recipe runs — into
``(observation, truth)`` pairs where the truth is a deterministic
function of the observation. The classifier head on the stock trunk must
learn it, export with the declared signature, and emit each class as its
declared code in the shell's [-1, 1] convention. The regression head and
the agent.toml block round-trip are covered too.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

torch = pytest.importorskip("torch")

from lockstep_train import LabelledStream  # noqa: E402
from lockstep_train.wire import ColumnSpec, SeatInit, ValueSpec  # noqa: E402

from train.core.export import export, verify  # noqa: E402
from train.core.policy import Policy, obs_to_tensors  # noqa: E402
from train.core.supervised import (  # noqa: E402
    Dataset,
    LabelSource,
    SupervisedPolicy,
    classes_of,
    make_stream,
)

LAG = 3
COLS = ["age_ticks", "your_decision", "truth", "valid"]


def synthetic_init() -> SeatInit:
    return SeatInit(
        seat=0,
        obs=[
            ValueSpec("features", "f32", (4,), -1.0, 1.0, doc="what you see"),
            ValueSpec(
                "feedback",
                "f32",
                (2, 4),
                -1.0,
                1000.0,
                doc="resolved decisions",
                columns=[ColumnSpec(c, c, "") for c in COLS],
            ),
        ],
        actions=[ValueSpec("decision", "f32", (1,), 0.0, 1.0, doc="1 = flag")],
    )


def synthetic_pairs(ticks: int = 400, seed: int = 0) -> Dataset:
    """Drive a LabelledStream with a fake engine: the truth of tick t is
    whether features[0] > features[1], revealed LAG ticks later as a
    feedback row (second row padding). (A difference, not a sum: the stock
    stream normalizes each sample across its elements, which keeps the
    order of two elements but not their sum's sign.)"""
    rng = np.random.default_rng(seed)
    init = synthetic_init()
    stream = make_stream(init, LabelSource.from_block({"value": "feedback", "valid_col": "valid", "action": "decision"}))
    features = rng.uniform(-1, 1, size=(ticks, 4)).astype(np.float32)
    truth = (features[:, 0] > features[:, 1]).astype(np.float32)
    obs_rows, labels = [], []
    for t in range(ticks):
        window = np.zeros((2, 4), dtype=np.float32)
        if t >= LAG:
            window[0] = [LAG, 0.0, truth[t - LAG], 1.0]
        pairs = stream.feed(t, {"features": features[t], "feedback": window}, 0.0)
        for obs_then, label in pairs:
            obs_rows.append(obs_then)
            labels.append(label.truth)
    assert len(labels) == ticks - LAG and stream.dropped == 0
    return Dataset(
        obs={
            "features": np.stack([o["features"] for o in obs_rows]),
            "feedback": np.stack([o["feedback"] for o in obs_rows]),
        },
        labels=np.asarray(labels, dtype=np.float32),
        seeds=1,
        ticks=ticks,
        dropped=0,
    )


OBS = spaces.Dict(
    {
        "features": spaces.Box(-1, 1, (4,), dtype=np.float32),
        "feedback": spaces.Box(-1, 1000, (2, 4), dtype=np.float32),
    }
)
ACT = spaces.Box(0.0, 1.0, (1,), dtype=np.float32)


def test_label_source_from_the_agent_toml_block():
    feedback = LabelSource.from_block(
        {"value": "feedback", "age_col": "age_ticks", "decision_col": "your_decision", "truth_col": "truth", "valid_col": "valid", "action": "decision"}
    )
    assert feedback.mode == "feedback" and "valid" in feedback.describe()
    lagged = LabelSource.from_block({"reward_lag_ticks": 40, "action": "decision"})
    assert lagged.mode == "reward-lag" and lagged.reward_lag == 40
    assert isinstance(make_stream(synthetic_init(), lagged), LabelledStream)
    with pytest.raises(SystemExit, match="label source"):
        make_stream(synthetic_init(), LabelSource())


def test_classes_of_tells_classification_from_regression():
    assert classes_of(np.array([0.0, 1.0, 1.0, 0.0])) == [0.0, 1.0]
    assert classes_of(np.array([2.0, 0.0, 1.0])) == [0.0, 1.0, 2.0]
    assert classes_of(np.array([0.5, 1.0])) is None, "non-integer truths regress"
    assert classes_of(np.array([1.0, 1.0])) is None, "one class is nothing to choose"


def test_classifier_head_learns_a_synthetic_truth_and_exports(tmp_path):
    torch.manual_seed(0)
    data = synthetic_pairs()
    classes = classes_of(data.labels)
    assert classes == [0.0, 1.0]
    net = SupervisedPolicy(Policy(OBS, ACT), ACT, element=0, classes=classes)
    targets = torch.from_numpy(data.labels.astype(np.int64))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    idx = np.arange(len(data))
    for _ in range(150):
        inputs = obs_to_tensors({n: data.obs[n][idx] for n in net.input_names}, net)
        loss = loss_fn(net.logits(*inputs), targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        inputs = obs_to_tensors({n: data.obs[n] for n in net.input_names}, net)
        predicted = net.logits(*inputs).argmax(dim=1)
        accuracy = float((predicted == targets).float().mean())
        out = net(*inputs)
    assert accuracy > 0.9, f"the head did not learn a linear truth: {accuracy:.2f}"
    # The exported value is the class's declared code in the shell's
    # [-1, 1]: code 0 -> -1, code 1 -> +1 for bounds [0, 1].
    assert set(out[:, 0].unique().tolist()) <= {-1.0, 1.0}
    assert out.shape == (len(data), 1)
    # Same signature and numbers under onnxruntime.
    path = export(net, tmp_path / "policy.onnx")
    assert verify(net, path) < 1e-4


def test_regression_head_targets_the_normalized_truth(tmp_path):
    net = SupervisedPolicy(Policy(OBS, ACT), ACT, element=0, classes=None)
    targets = net.normalized_target(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    np.testing.assert_allclose(targets.numpy(), [-1.0, 0.0, 1.0])
    inputs = obs_to_tensors({"features": np.zeros((2, 4), np.float32), "feedback": np.zeros((2, 2, 4), np.float32)}, net)
    assert net(*inputs).shape == (2, 1)
    assert verify(net, export(net, tmp_path / "policy.onnx")) < 1e-4


def test_head_signature_round_trips_through_weights(tmp_path):
    from train.main import load_weights, save_weights

    net = SupervisedPolicy(Policy(OBS, ACT), ACT, element=0, classes=[0.0, 1.0])
    save_weights(tmp_path / "policy.pt", net, "supervised")
    back = load_weights(tmp_path / "policy.pt", Policy)
    assert isinstance(back, SupervisedPolicy)
    assert back.classes == [0.0, 1.0] and back.element == 0
    inputs = obs_to_tensors({"features": np.ones((1, 4), np.float32), "feedback": np.zeros((1, 2, 4), np.float32)}, net)
    assert torch.equal(back(*inputs), net(*inputs))
