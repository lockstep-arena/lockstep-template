"""The template's own env-semantics proof, against a REAL engine.

Needs an engine wasm: anything in the keyed cache (``out/cache/…`` — any
task that needs an engine fills it) or ``LOCKSTEP_TEST_ENGINE=<path>`` —
CI and the local e2e provide one. Nothing
per-environment: the env is the generic ``Lockstep/Env-v0``.

The SAME_STEP check is the load-bearing one: the loop stores every
transition it sees, which is only sound if the obs arriving with
``done=True`` is already the NEXT episode's reset obs. Asserted directly
from the generic per-step info: on the done step a SAME_STEP env reports
``tick == 0`` (a fresh episode's first view) with the dying episode's last
obs preserved in ``infos["final_obs"]``, while a NEXT_STEP twin reports the
TERMINAL tick on its done step and only resets one step later. (Fresh
episodes draw fresh session seeds, so comparing the two envs' post-reset
observations byte-for-byte — the old formulation — is not a property this
env promises.)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from train.core.train import make_env_factory, make_vec_env

def _default_engine() -> str:
    cached = sorted(Path("out/cache").glob("*/*/engine.wasm"))
    return str(cached[0]) if cached else "out/cache/<env>/<mode>/engine.wasm"


ENGINE = Path(os.environ.get("LOCKSTEP_TEST_ENGINE") or _default_engine())

pytestmark = pytest.mark.skipif(
    not ENGINE.is_file(),
    reason=f"no engine at {ENGINE} — run: task info ENV=<slug> (or set LOCKSTEP_TEST_ENGINE)",

)


def _vec(autoreset_mode, num_envs=1):
    from gymnasium.vector import AutoresetMode, SyncVectorEnv

    fns = [make_env_factory(str(ENGINE), 120) for _ in range(num_envs)]
    return SyncVectorEnv(fns, autoreset_mode=AutoresetMode[autoreset_mode])


def test_same_step_autoreset():
    same = _vec("SAME_STEP")
    nxt = _vec("NEXT_STEP")
    try:
        obs_s, _ = same.reset(seed=0)
        obs_n, _ = nxt.reset(seed=0)
        # Same seed → identical first episodes (the env IS deterministic).
        for key in obs_s:
            assert np.array_equal(obs_s[key], obs_n[key])

        action = np.zeros((1, *same.single_action_space.shape), dtype=np.float32)
        done_infos = None
        for _ in range(600):
            obs_s, _r, term, trunc, infos = same.step(action)
            if term[0] or trunc[0]:
                done_infos = infos
                break
            nxt.step(action)
        assert done_infos is not None, "episode never ended within 600 steps"

        # SAME_STEP: the done step already delivered the NEXT episode's
        # reset obs — the generic `tick` info says 0 — and the dying
        # episode's terminal obs is preserved under final_obs.
        assert done_infos["tick"][0] == 0, "done-step obs must be a fresh reset"
        final_key = "final_obs" if "final_obs" in done_infos else "final_observation"
        assert final_key in done_infos, f"terminal obs missing (infos: {list(done_infos)})"
        final = done_infos[final_key][0]
        assert set(final) == set(obs_s), "final_obs carries the full obs dict"

        # NEXT_STEP twin, same tick count: its done step returns the
        # TERMINAL obs (tick > 0); the reset only lands one step later,
        # burning that step's action.
        obs_n, _r, term_n, trunc_n, infos_n = nxt.step(action)
        assert term_n[0] or trunc_n[0]
        assert infos_n["tick"][0] > 0, "NEXT_STEP's done step is the terminal obs"
        _obs, _r, term_after, trunc_after, infos_after = nxt.step(action)
        assert not (term_after[0] or trunc_after[0])
        assert infos_after["tick"][0] == 0, "NEXT_STEP resets one step late"
    finally:
        same.close()
        nxt.close()


def test_vec_env_construction_uses_same_step():
    """The loop's own constructor must pin SAME_STEP (not gymnasium's default)."""
    env = make_vec_env(str(ENGINE), 120, 1)
    try:
        from gymnasium.vector import AutoresetMode

        assert env.metadata.get("autoreset_mode") is AutoresetMode.SAME_STEP
    finally:
        env.close()


def test_parallel_env_smoke():
    """The generic PettingZoo env drives every seat of a multi-seat engine;
    single-seat engines are skipped (nothing to self-play)."""
    pettingzoo = pytest.importorskip("pettingzoo")  # noqa: F841 — optional extra
    from lockstep_train.env import LockstepParallelEnv

    env = LockstepParallelEnv(engine_source=str(ENGINE), time_limit_ticks=60)
    try:
        if len(env.possible_agents) < 2:
            pytest.skip("single-seat engine — self-play does not apply")
        obs, _infos = env.reset(seed=0)
        assert set(obs) == set(env.possible_agents)
        actions = {
            a: np.zeros(env.action_space(a).shape, dtype=np.float32)
            for a in env.possible_agents
        }
        obs, rewards, terms, truncs, _infos = env.step(actions)
        assert set(rewards) == set(env.possible_agents)
    finally:
        env.close()
