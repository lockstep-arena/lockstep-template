"""A small, real PPO loop over any discovered game's Gymnasium env.

Real in the sense that matters here: it steps the ACTUAL engine wasm through
``lockstep_train``, with the same physics and the same observations a ranked
match uses. It is not tuned, and a short run produces a weak agent — that is
expected, and the README says so. What it produces is a genuine policy whose
weights came from playing the game, which is what makes the exported ONNX
worth shipping.

Kept deliberately small (no RL framework) because the interesting part of
this repo is the train -> export -> bundle -> compete path, not the
optimizer. Swap this whole loop for your own stack freely: the env is plain
``gymnasium.make`` and composes with the standard vector API.

Collection is vectorized: ``num_envs`` engine instances step in parallel
worker processes (the standard Gymnasium ``AsyncVectorEnv``), which is both
the speed story — the engine is the wall-clock bottleneck, and one env pins
one core — and the compatibility story: swapping this loop for your own
framework needs nothing from us.

Long-run behavior is engineered, not hoped for:

- a checkpoint is written atomically EVERY rollout (a crash loses at most
  one rollout; ``--resume`` continues it),
- learning rate and entropy coefficient decay linearly to zero over the run
  (exploration pressure early; and on a reward landscape that sits at zero,
  entropy is otherwise the only surviving gradient late in a run — the
  demonstrated failure mode the decay plus the log-std clamp bound),
- ``out/metrics.csv`` gets one row per rollout for offline inspection.
"""

from __future__ import annotations

import csv
import importlib
import os
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

from .policy import Policy, obs_to_tensors

#: metrics.csv column order (one row per rollout).
METRICS_FIELDS = [
    "steps",
    "wall_seconds",
    "episodes",
    "mean_return",
    "policy_loss",
    "value_loss",
    "entropy",
    "log_std_mean",
    "clip_fraction",
    "steps_per_second",
]


def default_num_envs() -> int:
    """Leave two cores for the learner + OS instead of pinning them all.

    A flat default equal to the machine's core count oversubscribes small
    machines (the workers fight the update pass) — this is the cross-OS-safe
    default; explicit ``--num-envs`` always wins.
    """
    return min(8, max(1, (os.cpu_count() or 4) - 2))


def pick_device() -> torch.device:
    """Best available device for the PPO update pass.

    Only the update runs there: small-batch rollout inference is faster on
    CPU than on MPS/CUDA (per-op dispatch overhead dominates a net this
    small), so collection stays on CPU regardless.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env_factory(
    spec_module: str,
    env_id: str,
    mode: str,
    engine: str | None,
    time_limit_ticks: int | None,
):
    """A spawn-safe env factory.

    Spawned workers (the default on macOS and Windows) start with a fresh
    interpreter, and env registration is an import side effect — so the
    factory re-imports the game package's module before ``gymnasium.make``
    can resolve the id. Closing over plain strings keeps it picklable.
    """

    def make():
        importlib.import_module(spec_module)
        import gymnasium

        return gymnasium.make(
            env_id,
            mode=mode,
            engine_source=engine,
            time_limit_ticks=time_limit_ticks,
        )

    return make


def make_vec_env(
    spec_module: str,
    env_id: str,
    mode: str,
    engine: str | None,
    time_limit_ticks: int | None,
    num_envs: int,
):
    """``num_envs`` engines stepping in parallel.

    Three paths, best available first:

    - A game that registers a NATIVE vector env (``vector_entry_point`` —
      N engines on Rust threads in THIS process, GIL released) gets it by
      default: no worker processes, no per-step IPC, fastest on every OS.
      The games-side conformance suite holds it step-for-step identical to
      the process path, so this is a throughput choice, never a semantics
      one.
    - Otherwise ``AsyncVectorEnv``, one worker process per env — the
      portable standard-API path every game supports.
    - ``num_envs=1`` uses the in-process ``SyncVectorEnv`` — same API, no
      parallelism — which is the one to debug under (breakpoints reach the
      env).

    SAME_STEP autoreset is load-bearing, not taste: with it, every
    transition the loop stores is one the env actually executed (the obs
    arriving with ``done=True`` is already the next episode's reset obs).
    The default NEXT_STEP mode instead burns the following step on the reset
    — the action is ignored — and a loop that doesn't mask that step out
    trains on a transition that never happened.

    Async workers use the SPAWN start method explicitly, on every OS. Linux
    is the one platform whose default is fork, and forking a process that
    already imported torch (thread pools, allocator locks) deadlocks the
    workers — it hung CI's Linux leg while macOS/Windows sailed through.
    One start method everywhere is the whole point: no OS-specific behavior
    to debug.
    """
    if num_envs > 1:
        importlib.import_module(spec_module)  # registration side effect
        import gymnasium

        spec = gymnasium.registry.get(env_id)
        if spec is not None and spec.vector_entry_point is not None:
            env = gymnasium.make_vec(
                env_id,
                num_envs=num_envs,
                vectorization_mode="vector_entry_point",
                mode=mode,
                engine_source=engine,
                time_limit_ticks=time_limit_ticks,
            )
            if env.metadata.get("autoreset_mode") is not AutoresetMode.SAME_STEP:
                raise RuntimeError(
                    f"{env_id} native vector env declares "
                    f"{env.metadata.get('autoreset_mode')!r}; the loop "
                    "requires SAME_STEP autoreset semantics"
                )
            return env

    fns = [
        make_env_factory(spec_module, env_id, mode, engine, time_limit_ticks)
        for _ in range(num_envs)
    ]
    if num_envs == 1:
        return SyncVectorEnv(fns, autoreset_mode=AutoresetMode.SAME_STEP)
    return AsyncVectorEnv(fns, autoreset_mode=AutoresetMode.SAME_STEP, context="spawn")


def discounted_advantages(rewards, values, dones, last_value, gamma, lam):
    """Generalized Advantage Estimation over one ``[T, N]`` rollout.

    Envs never mix: each column carries its own recursion, cut wherever that
    env's episode ended. ``done`` zeroes the bootstrap for truncations too —
    a deliberate simplification.
    """
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    next_value = last_value
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        running = delta + gamma * lam * non_terminal * running
        advantages[t] = running
        next_value = values[t]
    return advantages, advantages + values


def save_checkpoint(path: Path, net: Policy, opt: torch.optim.Optimizer, collected: int):
    """Atomically write the resume point (``.tmp`` then ``os.replace``).

    The format is a boring small dict on purpose — researchers will depend
    on it: ``action_len`` (int), ``spaces`` (the net's space signature),
    ``state_dict`` (CPU tensors), ``optimizer_state``, ``collected`` (env
    steps done so far).
    """
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    opt_state = opt.state_dict()

    def to_cpu(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_cpu(v) for v in obj]
        return obj

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "action_len": net.action_len,
            "spaces": net.space_signature(),
            "state_dict": state,
            "optimizer_state": to_cpu(opt_state),
            "collected": collected,
        },
        tmp,
    )
    os.replace(tmp, path)


def train(
    spec: dict,
    spec_module: str,
    mode: str,
    steps: int,
    engine: str | None = None,
    time_limit_ticks: int | None = None,
    num_envs: int | None = None,
    rollout: int = 128,
    epochs: int = 4,
    minibatch: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip: float = 0.2,
    entropy_coef: float = 0.005,
    seed: int = 0,
    device: str | None = None,
    out_dir: Path = Path("out"),
    resume: bool = False,
) -> Policy:
    torch.manual_seed(seed)
    num_envs = num_envs if num_envs is not None else default_num_envs()
    env = make_vec_env(
        spec_module, spec["env_id"], mode, engine, time_limit_ticks, num_envs
    )
    net = Policy(env.single_observation_space, env.single_action_space)

    dev = torch.device(device) if device else pick_device()
    if dev.type == "cuda":
        # Every conv sees the same shapes all run, the case autotune exists for.
        torch.backends.cudnn.benchmark = True
    if dev.type == "cpu":
        learner = net
    else:
        # `net` stays the CPU rollout copy; `learner` takes the gradient
        # steps on the accelerator and syncs weights back each rollout.
        learner = Policy(env.single_observation_space, env.single_action_space).to(dev)
        learner.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    metrics_path = out_dir / "metrics.csv"

    collected = 0
    if resume:
        if not checkpoint_path.is_file():
            raise SystemExit(f"--resume: no checkpoint at {checkpoint_path}")
        blob = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        learner.load_state_dict(blob["state_dict"])
        if learner is not net:
            net.load_state_dict(blob["state_dict"])
        opt.load_state_dict(blob["optimizer_state"])
        collected = int(blob["collected"])
        print(f"  resumed at {collected} steps from {checkpoint_path}", flush=True)
    print(
        f"  update device: {dev}   envs: {num_envs} ({type(env).__name__})",
        flush=True,
    )

    metrics_file = open(  # noqa: SIM115 — held across the whole run
        metrics_path, "a" if resume and metrics_path.is_file() else "w", newline=""
    )
    metrics = csv.DictWriter(metrics_file, fieldnames=METRICS_FIELDS)
    if metrics_file.tell() == 0:
        metrics.writeheader()

    obs, _ = env.reset(seed=seed)  # seeds env i with seed + i
    episode_return = np.zeros(num_envs, dtype=np.float64)
    episode_returns: list[float] = []
    started = time.time()
    started_at = collected

    while collected < steps:
        # Linear anneal of lr and entropy pressure over the FULL run, from
        # this rollout's starting point. Both hit zero together at --steps.
        frac = max(0.0, 1.0 - collected / steps)
        for group in opt.param_groups:
            group["lr"] = lr * frac
        ent_coef = entropy_coef * frac

        buf_obs: list[tuple[torch.Tensor, ...]] = []
        buf_raw, buf_logp = [], []
        buf_val = np.zeros((rollout, num_envs), dtype=np.float32)
        buf_rew = np.zeros((rollout, num_envs), dtype=np.float32)
        buf_done = np.zeros((rollout, num_envs), dtype=np.float32)

        for t in range(rollout):
            tensors = obs_to_tensors(obs, net)
            with torch.no_grad():
                action, raw, log_prob, value = net.act(*tensors)

            obs, reward, terminated, truncated, _ = env.step(
                action.numpy().astype(np.float32)
            )
            done = np.logical_or(terminated, truncated)

            buf_obs.append(tensors)
            buf_raw.append(raw)
            buf_logp.append(log_prob)
            buf_val[t] = value.numpy()
            buf_rew[t] = reward
            buf_done[t] = done

            episode_return += reward
            collected += num_envs
            for i in np.flatnonzero(done):
                episode_returns.append(float(episode_return[i]))
                episode_return[i] = 0.0

        with torch.no_grad():
            last_value = net.act(*obs_to_tensors(obs, net))[3].numpy()

        advantages, returns = discounted_advantages(
            buf_rew, buf_val, buf_done, last_value, gamma, lam
        )
        # Normalizing advantages is what keeps the update scale sane when the
        # reward is a raw score delta that can sit at zero for long stretches.
        adv_t = torch.from_numpy(advantages.reshape(-1)).to(dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # [T, N, ...] -> [T*N, ...]; every buffer flattens in the same
        # time-major order, so row i of one lines up with row i of the rest.
        obs_t = tuple(
            torch.cat([step[i] for step in buf_obs]).to(dev)
            for i in range(len(net.input_names))
        )
        raw_t = torch.cat(buf_raw).to(dev)
        logp_t = torch.cat(buf_logp).to(dev)
        ret_t = torch.from_numpy(returns.reshape(-1)).to(dev)

        batch = rollout * num_envs
        stat_policy, stat_value, stat_entropy, stat_clip = [], [], [], []
        for _ in range(epochs):
            for idx in torch.randperm(batch).split(minibatch):
                new_logp, entropy, value = learner.evaluate(
                    *(t[idx] for t in obs_t), raw=raw_t[idx]
                )
                ratio = (new_logp - logp_t[idx]).exp()
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
                policy_loss = -torch.min(ratio * adv_t[idx], clipped * adv_t[idx]).mean()
                value_loss = torch.nn.functional.mse_loss(value, ret_t[idx])
                loss = policy_loss + 0.5 * value_loss - ent_coef * entropy.mean()

                opt.zero_grad()
                loss.backward()
                # PPO without gradient clipping diverges readily on a reward
                # this spiky; cheap insurance.
                torch.nn.utils.clip_grad_norm_(learner.parameters(), 0.5)
                opt.step()

                with torch.no_grad():
                    stat_policy.append(float(policy_loss))
                    stat_value.append(float(value_loss))
                    stat_entropy.append(float(entropy.mean()))
                    stat_clip.append(float(((ratio - 1.0).abs() > clip).float().mean()))

        if learner is not net:
            net.load_state_dict(learner.state_dict())

        # A crash from here back loses at most this one rollout.
        save_checkpoint(checkpoint_path, net, opt, collected)

        recent = episode_returns[-5:]
        mean_return = sum(recent) / len(recent) if recent else float("nan")
        elapsed = time.time() - started
        metrics.writerow(
            {
                "steps": collected,
                "wall_seconds": round(elapsed, 3),
                "episodes": len(episode_returns),
                "mean_return": mean_return,
                "policy_loss": np.mean(stat_policy),
                "value_loss": np.mean(stat_value),
                "entropy": np.mean(stat_entropy),
                "log_std_mean": float(net.log_std.detach().mean()),
                "clip_fraction": np.mean(stat_clip),
                "steps_per_second": round((collected - started_at) / elapsed, 1),
            }
        )
        metrics_file.flush()
        print(
            f"  {collected:>7}/{steps} steps  "
            f"episodes={len(episode_returns):<4} "
            f"mean_return={mean_return:8.2f}  "
            f"{elapsed:6.1f}s",
            flush=True,
        )

    env.close()
    metrics_file.close()
    return net
