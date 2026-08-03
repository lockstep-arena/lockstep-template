"""The template's own env-semantics proof, generic over the installed game.

Needs a game package installed (``task setup``) and an engine at
``out/engine.wasm`` (``task engine``) — CI provides both. The game is found
through discovery, exactly like training does; no game is named here.

The SAME_STEP check is the load-bearing one: the loop stores every
transition it sees, which is only sound if the obs arriving with
``done=True`` is already the NEXT episode's reset obs. The test drives a
SAME_STEP vector env and a NEXT_STEP twin through the same seed and action
sequence — NEXT_STEP delivers the reset obs one step late (action ignored),
so SAME_STEP's done-step obs must equal NEXT_STEP's following obs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from train.core.discovery import resolve_game
from train.core.train import make_vec_env

ENGINE = Path("out/engine.wasm")

pytestmark = pytest.mark.skipif(
    not ENGINE.is_file(), reason="no engine at out/engine.wasm — run: task engine"
)


@pytest.fixture(scope="module")
def game():
    spec, module = resolve_game(os.environ.get("LOCKSTEP_TEST_GAME") or None)
    return spec, module


def _vec(spec, module, autoreset_mode, num_envs=1):
    from gymnasium.vector import AutoresetMode, SyncVectorEnv

    from train.core.train import make_env_factory

    fns = [
        make_env_factory(
            module, spec["env_id"], spec["default_mode"], str(ENGINE), 120
        )
        for _ in range(num_envs)
    ]
    return SyncVectorEnv(fns, autoreset_mode=AutoresetMode[autoreset_mode])


def test_same_step_autoreset(game):
    spec, module = game
    same = _vec(spec, module, "SAME_STEP")
    nxt = _vec(spec, module, "NEXT_STEP")
    try:
        obs_s, _ = same.reset(seed=0)
        obs_n, _ = nxt.reset(seed=0)
        for key in obs_s:
            assert np.array_equal(obs_s[key], obs_n[key])

        action = np.zeros((1, *same.single_action_space.shape), dtype=np.float32)
        done_obs = None
        for _ in range(600):
            obs_s, _r, term, trunc, _ = same.step(action)
            if term[0] or trunc[0]:
                done_obs = {k: v.copy() for k, v in obs_s.items()}
                nxt.step(action)
                break
            nxt.step(action)
        assert done_obs is not None, "episode never ended within 600 steps"

        obs_n, _r, term_n, _tr, _ = nxt.step(action)  # NEXT_STEP's reset step
        assert not term_n[0]
        for key in done_obs:
            assert np.array_equal(done_obs[key], obs_n[key]), (
                f"SAME_STEP {key!r} on the done step must be the reset obs"
            )
    finally:
        same.close()
        nxt.close()


def test_vec_env_construction_uses_same_step(game):
    """The loop's own constructor must pin SAME_STEP (not gymnasium's default)."""
    spec, module = game
    env = make_vec_env(module, spec["env_id"], spec["default_mode"], str(ENGINE), 120, 1)
    try:
        # Gymnasium stores the mode in metadata["autoreset_mode"].
        from gymnasium.vector import AutoresetMode

        assert env.metadata.get("autoreset_mode") is AutoresetMode.SAME_STEP
    finally:
        env.close()
