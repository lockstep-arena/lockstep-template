"""The generic network + export seam, with no engine at all.

Builds the policy from a synthetic Dict space of the two stream kinds the
Lockstep wire carries today — a channel-first uint8 image and a float32
vector — exports it, and holds torch and onnxruntime to the same numbers.
This is the whole train-side contract in one test: one ONNX input per obs
tensor BY NAME, one ``action`` output, fixed batch of 1.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

torch = pytest.importorskip("torch")

from train.core.export import export, verify  # noqa: E402 — after the skip
from train.core.policy import Policy, obs_to_tensors, policy_from_signature  # noqa: E402

OBS = spaces.Dict(
    {
        # Channel-first, exactly as the wire declares images.
        "strip": spaces.Box(0, 255, (1, 48, 96), dtype=np.uint8),
        "state": spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32),
    }
)
ACTION = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)


def test_policy_derives_its_signature_from_the_spaces():
    net = Policy(OBS, ACTION)
    # gymnasium's Dict canonicalizes key order; binding is BY NAME on every
    # side (obs dicts, ONNX inputs), so order is a net-internal detail.
    assert set(net.input_names) == {"strip", "state"}
    assert net.input_shapes["strip"] == (1, 48, 96)  # declared shape, no shuffle
    assert net.input_shapes["state"] == (5,)
    assert net.action_len == 4


def test_export_parity_and_signature(tmp_path):
    net = Policy(OBS, ACTION)
    path = export(net, tmp_path / "policy.onnx")
    diff = verify(net, path)
    assert diff < 1e-4


def test_obs_normalization_matches_the_shell():
    """u8 images are fed as f32/255 at the DECLARED shape — the same rule
    the generic ONNX shell applies at match time."""
    net = Policy(OBS, ACTION)
    obs = {
        "strip": np.full((2, 1, 48, 96), 255, dtype=np.uint8),
        "state": np.ones((2, 5), dtype=np.float32),
    }
    tensors = dict(zip(net.input_names, obs_to_tensors(obs, net)))
    assert tensors["strip"].shape == (2, 1, 48, 96)
    assert torch.allclose(tensors["strip"], torch.ones(2, 1, 48, 96))
    assert tensors["state"].shape == (2, 5)


def test_signature_round_trips_through_checkpoints():
    net = Policy(OBS, ACTION)
    rebuilt = policy_from_signature(net.space_signature())
    assert rebuilt.input_names == net.input_names
    assert rebuilt.input_shapes == net.input_shapes
    assert rebuilt.action_len == net.action_len
