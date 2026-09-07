"""The generic network + export seam, with no engine at all.

Builds the policy from a synthetic Dict space that mixes every dtype and
several ranks the Lockstep wire carries — an f32 rank-2 window, a u8
rank-3 crop, an i32 table, an f32 vector — through the ONE stream rule
(flatten → LayerNorm → Linear, no shape branches), exports it, and holds
torch and onnxruntime to the same numbers under the declared signature:
one ONNX input per obs tensor BY NAME at the declared shape and dtype
(u8 as f32 ÷ 255, i32 as int32), one ``action`` output, fixed batch of 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

torch = pytest.importorskip("torch")

from train.core.export import export, verify  # noqa: E402 — after the skip
from train.core.policy import (  # noqa: E402
    Policy,
    env_to_normalized,
    actions_to_env,
    flat_stream,
    obs_to_tensors,
    policy_from_signature,
)

OBS = spaces.Dict(
    {
        # An f32 window: rows are ticks, columns are features.
        "window": spaces.Box(-np.inf, np.inf, (16, 6), dtype=np.float32),
        # A u8 crop, channel-first, exactly as the wire declares images.
        "crop": spaces.Box(0, 255, (1, 12, 20), dtype=np.uint8),
        # An i32 table of codes.
        "table": spaces.Box(0, 9, (3, 4), dtype=np.int32),
        # A plain f32 vector.
        "state": spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32),
    }
)
ACTION = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)


def test_policy_derives_its_signature_from_the_spaces():
    net = Policy(OBS, ACTION)
    # gymnasium's Dict canonicalizes key order; binding is BY NAME on every
    # side (obs dicts, ONNX inputs), so order is a net-internal detail.
    assert set(net.input_names) == {"window", "crop", "table", "state"}
    assert net.input_shapes["window"] == (16, 6)  # declared shape, no shuffle
    assert net.input_shapes["crop"] == (1, 12, 20)
    assert net.input_shapes["table"] == (3, 4)
    assert net.input_dtypes == {"window": "float32", "crop": "uint8", "table": "int32", "state": "float32"}
    assert net.action_len == 4
    # One stream rule for every value: no rank is special.
    for stream in net.streams.values():
        assert isinstance(stream[0], torch.nn.Flatten)
        assert isinstance(stream[1], torch.nn.LayerNorm)


def test_export_parity_and_declared_signature(tmp_path):
    net = Policy(OBS, ACTION)
    path = export(net, tmp_path / "policy.onnx")
    assert verify(net, path) < 1e-4
    import onnxruntime as ort

    inputs = {i.name: i for i in ort.InferenceSession(str(path)).get_inputs()}
    assert inputs["window"].shape == [1, 16, 6] and inputs["window"].type == "tensor(float)"
    assert inputs["crop"].shape == [1, 1, 12, 20] and inputs["crop"].type == "tensor(float)"
    assert inputs["table"].shape == [1, 3, 4] and inputs["table"].type == "tensor(int32)"


def test_a_custom_stream_from_model_py_exports_too(tmp_path):
    """The agent's model.py can hand any stream in — a convolution for the
    crop here — and the export contract is unchanged."""
    streams = {
        "window": flat_stream((16, 6), 64),
        "crop": torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        ),
        "table": flat_stream((3, 4), 32),
        "state": flat_stream((5,)),
    }
    net = Policy(OBS, ACTION, streams)
    assert verify(net, export(net, tmp_path / "policy.onnx")) < 1e-4


def test_streams_must_cover_every_declared_value():
    with pytest.raises(ValueError, match="missing \\['crop'"):
        Policy(OBS, ACTION, {"window": flat_stream((16, 6)), "table": flat_stream((3, 4)), "state": flat_stream((5,))})


def test_obs_normalization_matches_the_shell():
    """u8 values are fed as f32 ÷ 255 at the DECLARED shape, i32 values as
    int32 — the same rule the generic ONNX shell applies at match time."""
    net = Policy(OBS, ACTION)
    obs = {
        "window": np.ones((2, 16, 6), dtype=np.float32),
        "crop": np.full((2, 1, 12, 20), 255, dtype=np.uint8),
        "table": np.full((2, 3, 4), 7, dtype=np.int32),
        "state": np.ones((2, 5), dtype=np.float32),
    }
    tensors = dict(zip(net.input_names, obs_to_tensors(obs, net)))
    assert tensors["crop"].shape == (2, 1, 12, 20)
    assert torch.allclose(tensors["crop"], torch.ones(2, 1, 12, 20))
    assert tensors["table"].dtype == torch.int32 and int(tensors["table"][0, 0, 0]) == 7
    assert tensors["window"].shape == (2, 16, 6)
    assert net(*tensors.values()).shape == (2, 4)


def test_signature_round_trips_through_checkpoints():
    net = Policy(OBS, ACTION)
    rebuilt = policy_from_signature(net.space_signature())
    assert rebuilt.input_names == net.input_names
    assert rebuilt.input_shapes == net.input_shapes
    assert rebuilt.input_dtypes == net.input_dtypes
    assert rebuilt.action_len == net.action_len


def test_normalized_is_the_inverse_of_the_shell_map():
    box = spaces.Box(np.array([0.0, -2.0]), np.array([1.0, 2.0]), dtype=np.float32)
    declared = np.array([1.0, -1.0], dtype=np.float32)
    normalized = env_to_normalized(declared, box)
    np.testing.assert_allclose(normalized, [1.0, -0.5])
    np.testing.assert_allclose(actions_to_env(normalized, box), declared, atol=1e-6)
