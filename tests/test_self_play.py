"""The self-play loop, proven without an engine.

A synthetic two-seat PettingZoo ``ParallelEnv`` with the contract's shape
(``Dict`` of one float32 vector per seat, a ``Box`` action, zero-sum rewards,
both seats ending together) drives :func:`train.core.self_play.train_self_play`
through real rollouts and PPO updates. What is pinned:

- both seats' experience is pooled (``steps`` counts seat-steps, and the
  checkpoint's ``collected`` agrees),
- an episode end resets the env INSIDE the rollout (the next stored
  observation is a fresh first observation — the SAME_STEP discipline),
- the result is ONE policy with the single-seat signature, which the
  standard export + parity check accepts unchanged,
- ``--resume`` continues from the checkpoint,
- a spec without ``parallel_env_id`` is refused by name.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from gymnasium import spaces

pytest.importorskip("pettingzoo")
from pettingzoo import ParallelEnv  # noqa: E402

from train.core.export import export, verify  # noqa: E402
from train.core.self_play import (  # noqa: E402
    SELF_PLAY_METRICS_FIELDS,
    _batch_seats,
    resolve_parallel_env,
    train_self_play,
)

OBS_LEN = 6
ACTION_LEN = 3
EPISODE_LEN = 20


class Duel(ParallelEnv):
    """Two seats, fixed-length episodes, zero-sum reward: seat 0 is paid the
    first action channel's sign, seat 1 the negative of it."""

    metadata = {"name": "duel_v0", "render_modes": [], "is_parallelizable": True}
    resets = 0  # class-level so the test can count through the factory

    def __init__(self, mode: str, engine_source=None, time_limit_ticks=None):
        self.mode = mode
        self.possible_agents = ["seat_0", "seat_1"]
        self.agents = []
        self._obs_spaces = {
            a: spaces.Dict({"obs": spaces.Box(-np.inf, np.inf, (OBS_LEN,), dtype=np.float32)})
            for a in self.possible_agents
        }
        self._act_spaces = {
            a: spaces.Box(-1.0, 1.0, (ACTION_LEN,), dtype=np.float32) for a in self.possible_agents
        }
        self._t = 0
        self._rng = np.random.default_rng(0)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._obs_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._act_spaces[agent]

    def _obs(self):
        # Seat i's first channel is its seat index: the two views differ.
        return {
            a: {"obs": np.concatenate([[i, self._t / EPISODE_LEN], self._rng.normal(size=OBS_LEN - 2)]).astype(np.float32)}
            for i, a in enumerate(self.possible_agents)
        }

    def reset(self, seed=None, options=None):
        Duel.resets += 1
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._t = 0
        return self._obs(), {a: {} for a in self.agents}

    def step(self, actions):
        assert set(actions) == set(self.agents), "every live seat acts every tick"
        self._t += 1
        gain = float(np.sign(actions["seat_0"][0]))
        rewards = {"seat_0": gain, "seat_1": -gain}
        done = self._t >= EPISODE_LEN
        terms = {a: done for a in self.agents}
        truncs = {a: False for a in self.agents}
        obs = self._obs()
        infos = {a: {} for a in self.agents}
        if done:
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def close(self):
        pass


def parallel_env(**kwargs):
    return Duel(**kwargs)


SPEC = {
    "training_contract_version": 2,
    "slug": "synthetic-duel",
    "env_id": "Synthetic/Duel-v0",
    "parallel_env_id": f"{__name__}:parallel_env",
    "default_mode": "only",
    "modes": {"only": {"payload_schema_version": 1, "engine_url": "https://example.invalid/engine.wasm"}},
}


def test_batch_seats_stacks_in_seat_order():
    obs = {
        "seat_0": {"obs": np.zeros(OBS_LEN, dtype=np.float32)},
        "seat_1": {"obs": np.ones(OBS_LEN, dtype=np.float32)},
    }
    batched = _batch_seats(obs, ["seat_0", "seat_1"])
    assert batched["obs"].shape == (2, OBS_LEN)
    assert batched["obs"][0].sum() == 0 and batched["obs"][1].sum() == OBS_LEN


def test_spec_without_parallel_env_is_refused_by_kind():
    # A v1 spec cannot say whether the game is single-agent or the wheel is
    # just old, so the refusal carries the upgrade hint.
    v1 = {**SPEC, "training_contract_version": 1}
    del v1["parallel_env_id"]
    with pytest.raises(SystemExit, match="contract v1, which has no parallel env"):
        resolve_parallel_env(v1, "only", None, None)
    # A v2 spec with None is a deliberate "no adversarial seats".
    v2_none = {**SPEC, "parallel_env_id": None}
    with pytest.raises(SystemExit, match="has no parallel env — its seats are not adversarial"):
        resolve_parallel_env(v2_none, "only", None, None)


def test_resolve_parallel_env_builds_the_factory():
    env = resolve_parallel_env(SPEC, "only", None, None)
    assert isinstance(env, Duel) and env.mode == "only"


def test_self_play_pools_both_seats_and_exports_one_policy(tmp_path):
    Duel.resets = 0
    rollout = 32
    steps = rollout * 2 * 3  # three rollouts of 32 ticks × 2 seats
    net = train_self_play(
        SPEC,
        mode="only",
        steps=steps,
        rollout=rollout,
        minibatch=32,
        epochs=1,
        device="cpu",
        out_dir=tmp_path,
    )

    # Pooled accounting: every seat-step counted, episodes reset in-loop.
    import torch

    blob = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    assert blob["collected"] == steps
    ticks = steps // 2
    assert Duel.resets == 1 + ticks // EPISODE_LEN  # initial + one per finished duel

    rows = (tmp_path / "metrics.csv").read_text().splitlines()
    assert rows[0].split(",") == SELF_PLAY_METRICS_FIELDS
    assert len(rows) == 1 + 3
    last = dict(zip(SELF_PLAY_METRICS_FIELDS, rows[-1].split(",")))
    assert int(last["steps"]) == steps
    # Zero-sum pooled return is ~0; |return| is what shows activity.
    assert abs(float(last["mean_return"])) < 1e-6
    assert float(last["mean_abs_return"]) > 0

    # ONE policy with the single-seat signature: the unchanged export path
    # (and its parity check against onnxruntime) accepts it.
    assert net.input_names == ["obs"] and net.action_len == ACTION_LEN
    onnx = export(net, tmp_path / "policy.onnx")
    assert verify(net, onnx) < 1e-4


def test_self_play_resumes_from_the_checkpoint(tmp_path):
    rollout = 16
    train_self_play(
        SPEC, mode="only", steps=rollout * 2, rollout=rollout, minibatch=16, epochs=1,
        device="cpu", out_dir=tmp_path,
    )
    train_self_play(
        SPEC, mode="only", steps=rollout * 2 * 2, rollout=rollout, minibatch=16, epochs=1,
        device="cpu", out_dir=tmp_path, resume=True,
    )
    import torch

    assert torch.load(tmp_path / "checkpoint.pt", weights_only=True)["collected"] == rollout * 4
    rows = (tmp_path / "metrics.csv").read_text().splitlines()
    assert len(rows) == 1 + 2, "resume appends to metrics.csv rather than restarting it"
