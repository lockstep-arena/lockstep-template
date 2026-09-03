"""The training loop steps the env through the SAME [-1, 1] → bounds map the
generic ONNX shell applies at match time. Hermetic: a Box is enough.

Why this matters: the env's action Box is the engine's DECLARED bounds (the
honest space), and the network's head is tanh. Feeding the tanh output
straight into the env means "radians in [-1, 1], clamped by the engine";
the shell instead rescales the same output onto each joint's full range.
A policy trained the first way scores well in training and near zero once
sealed. This pins the map to the shell's documented convention
(`agent-onnx`: clamp, affine per element, open bounds pass through).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from train.core.policy import actions_to_env


def test_affine_onto_per_element_bounds():
    box = spaces.Box(
        low=np.array([-0.863, -0.686, -2.818], dtype=np.float32),
        high=np.array([0.863, 4.501, -0.888], dtype=np.float32),
    )
    out = actions_to_env(np.array([-1.0, 0.0, 1.0], dtype=np.float32), box)
    np.testing.assert_allclose(out, [-0.863, (4.501 - 0.686) / 2, -0.888], atol=1e-6)
    # 0 → midpoint on every element: the shell's neutral.
    mid = actions_to_env(np.zeros(3, dtype=np.float32), box)
    np.testing.assert_allclose(mid, (box.low + box.high) / 2, atol=1e-6)


def test_clamps_before_scaling_and_passes_open_bounds_through():
    box = spaces.Box(
        low=np.array([0.0, np.finfo(np.float32).min], dtype=np.float32),
        high=np.array([1.0, np.finfo(np.float32).max], dtype=np.float32),
    )
    out = actions_to_env(np.array([7.0, 7.0], dtype=np.float32), box)
    assert out[0] == 1.0, "clamped to [-1, 1] first, then scaled"
    assert out[1] == 1.0, "an open bound passes the (clamped) value through unscaled"
    assert out.dtype == np.float32


def test_batched_rows_scale_independently():
    box = spaces.Box(low=np.array([-2.0, 0.0]), high=np.array([2.0, 10.0]), dtype=np.float32)
    batch = np.array([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.5]], dtype=np.float32)
    out = actions_to_env(batch, box)
    np.testing.assert_allclose(out, [[-2.0, 0.0], [2.0, 10.0], [0.0, 7.5]], atol=1e-6)
