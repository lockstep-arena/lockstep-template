"""A small, real PPO loop over the dance-off Gymnasium env.

Real in the sense that matters here: it steps the ACTUAL engine wasm through
``lockstep_train``, with the same physics and the same host-rasterized
observations a ranked match uses. It is not tuned, and a short run produces a
weak dancer — that is expected, and the README says so. What it produces is a
genuine policy whose weights came from playing the game, which is what makes
the exported ONNX worth shipping as a test agent.

Kept deliberately small (one file, no RL framework) because the alternative was
a stable-baselines3 dependency for a loop this size, and the interesting part of
this repo is the train -> export -> bundle -> compete path, not the optimizer.

Collection is vectorized: ``num_envs`` engine instances step in parallel
worker processes (the standard Gymnasium ``AsyncVectorEnv``), which is both
the speed story — the engine is the wall-clock bottleneck, and one env pins
one core — and the compatibility story: the env composes with the vector API
every mainstream RL stack builds on, so swapping this loop for your own
framework needs nothing from us.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import gymnasium  # noqa: F401  (registers the env)
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
import lockstep_dance_off  # noqa: F401
from lockstep_dance_off import MODE_RAW_TORQUE, MODE_SERVO_ASSIST

from .policy import Policy


def pick_device() -> torch.device:
    """Best available device for the PPO update pass.

    Only the update runs there: small-batch rollout inference is faster on CPU
    than on MPS/CUDA (per-op dispatch overhead dominates a net this small),
    so collection stays on CPU regardless.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env(mode: str, engine: str | None, time_limit_ticks: int | None):
    return gymnasium.make(
        "Lockstep/DanceOff-v0",
        mode=mode,
        engine_source=engine,
        time_limit_ticks=time_limit_ticks,
    )


def make_vec_env(
    mode: str, engine: str | None, time_limit_ticks: int | None, num_envs: int
):
    """``num_envs`` engines stepping in parallel, one per worker process.

    ``num_envs=1`` uses the in-process ``SyncVectorEnv`` — same API, no
    subprocesses — which is the one to debug under (breakpoints reach the env).

    SAME_STEP autoreset is load-bearing, not taste: with it, every transition
    the loop stores is one the env actually executed (the obs arriving with
    ``done=True`` is already the next episode's reset obs). The default
    NEXT_STEP mode instead burns the following step on the reset — the action
    is ignored — and a loop that doesn't mask that step out trains on a
    transition that never happened.
    """
    fns = [
        (lambda: make_env(mode, engine, time_limit_ticks)) for _ in range(num_envs)
    ]
    cls = SyncVectorEnv if num_envs == 1 else AsyncVectorEnv
    return cls(fns, autoreset_mode=AutoresetMode.SAME_STEP)


def as_tensors(obs) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched vector-env observation -> the network's two inputs.

    The marquee is scaled to 0..1 HERE, matching
    ``schema::policy::marquee_normalized`` exactly — the observation space is
    declared ``uint8`` (it is an image, and saying otherwise would be a lie
    about the space), so the scaling has to happen on the way into the network
    on both sides. The agent vector arrives already built by the shared
    encoder and needs nothing.
    """
    marquee = torch.from_numpy(obs["marquee"].astype(np.float32) / 255.0)
    marquee = marquee.permute(0, 3, 1, 2)  # NHWC -> NCHW
    agent = torch.from_numpy(np.asarray(obs["agent"], dtype=np.float32))
    return marquee, agent


def discounted_advantages(rewards, values, dones, last_value, gamma, lam):
    """Generalized Advantage Estimation over one ``[T, N]`` rollout.

    Envs never mix: each column carries its own recursion, cut wherever that
    env's episode ended. ``done`` zeroes the bootstrap for truncations too —
    the same simplification the single-env loop made.
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


def train(
    mode: str,
    steps: int,
    engine: str | None = None,
    time_limit_ticks: int | None = None,
    num_envs: int = 8,
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
) -> Policy:
    torch.manual_seed(seed)
    env = make_vec_env(mode, engine, time_limit_ticks, num_envs)
    (action_len,) = env.single_action_space.shape
    net = Policy(action_len)

    dev = torch.device(device) if device else pick_device()
    if dev.type == "cuda":
        # Every conv sees the same shapes all run, the case autotune exists for.
        torch.backends.cudnn.benchmark = True
    if dev.type == "cpu":
        learner = net
    else:
        # `net` stays the CPU rollout copy; `learner` takes the gradient
        # steps on the accelerator and syncs weights back each rollout.
        learner = Policy(action_len).to(dev)
        learner.load_state_dict(net.state_dict())
    print(f"  update device: {dev}   envs: {num_envs}", flush=True)
    opt = torch.optim.Adam(learner.parameters(), lr=lr)

    obs, _ = env.reset(seed=seed)  # seeds env i with seed + i
    episode_return = np.zeros(num_envs, dtype=np.float64)
    episode_returns: list[float] = []
    started = time.time()
    collected = 0

    while collected < steps:
        buf_marquee, buf_agent, buf_raw, buf_logp = [], [], [], []
        buf_val = np.zeros((rollout, num_envs), dtype=np.float32)
        buf_rew = np.zeros((rollout, num_envs), dtype=np.float32)
        buf_done = np.zeros((rollout, num_envs), dtype=np.float32)

        for t in range(rollout):
            marquee, agent = as_tensors(obs)
            with torch.no_grad():
                action, raw, log_prob, value = net.act(marquee, agent)

            obs, reward, terminated, truncated, _ = env.step(
                action.numpy().astype(np.float32)
            )
            done = np.logical_or(terminated, truncated)

            buf_marquee.append(marquee)
            buf_agent.append(agent)
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
            marquee, agent = as_tensors(obs)
            last_value = net.act(marquee, agent)[3].numpy()

        advantages, returns = discounted_advantages(
            buf_rew, buf_val, buf_done, last_value, gamma, lam
        )
        # Normalizing advantages is what keeps the update scale sane when the
        # reward is a raw score delta that can sit at zero for long stretches
        # (a dancer who is not hitting cards earns nothing at all).
        adv_t = torch.from_numpy(advantages.reshape(-1)).to(dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # [T, N, ...] -> [T*N, ...]; every buffer flattens in the same
        # time-major order, so row i of one lines up with row i of the rest.
        marquee_t = torch.cat(buf_marquee).to(dev)
        agent_t = torch.cat(buf_agent).to(dev)
        raw_t = torch.cat(buf_raw).to(dev)
        logp_t = torch.cat(buf_logp).to(dev)
        ret_t = torch.from_numpy(returns.reshape(-1)).to(dev)

        batch = rollout * num_envs
        for _ in range(epochs):
            for idx in torch.randperm(batch).split(minibatch):
                new_logp, entropy, value = learner.evaluate(
                    marquee_t[idx], agent_t[idx], raw_t[idx]
                )
                ratio = (new_logp - logp_t[idx]).exp()
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
                policy_loss = -torch.min(ratio * adv_t[idx], clipped * adv_t[idx]).mean()
                value_loss = torch.nn.functional.mse_loss(value, ret_t[idx])
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy.mean()

                opt.zero_grad()
                loss.backward()
                # PPO without gradient clipping diverges readily on a reward
                # this spiky; cheap insurance.
                torch.nn.utils.clip_grad_norm_(learner.parameters(), 0.5)
                opt.step()

        if learner is not net:
            net.load_state_dict(learner.state_dict())

        recent = episode_returns[-5:]
        mean_return = sum(recent) / len(recent) if recent else float("nan")
        print(
            f"  {collected:>7}/{steps} steps  "
            f"episodes={len(episode_returns):<4} "
            f"mean_return={mean_return:8.2f}  "
            f"{time.time() - started:6.1f}s",
            flush=True,
        )

    env.close()
    return net


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=[MODE_SERVO_ASSIST, MODE_RAW_TORQUE], required=True)
    p.add_argument("--steps", type=int, default=8192, help="environment steps")
    p.add_argument("--engine", default=None, help="engine wasm (else $LOCKSTEP_ENGINE_WASM)")
    p.add_argument(
        "--time-limit-ticks",
        type=int,
        default=1800,
        help="force-end backstop; shorter episodes mean more of them per run",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the trained .pt here")
    p.add_argument(
        "--device",
        default=None,
        help="update-pass device (cuda/mps/cpu); default: best available",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="parallel engine instances for collection; 1 = in-process (debuggable)",
    )
    args = p.parse_args()

    net = train(
        mode=args.mode,
        steps=args.steps,
        engine=args.engine,
        time_limit_ticks=args.time_limit_ticks,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
    )
    if args.out:
        torch.save({"action_len": net.action_len, "state_dict": net.state_dict()}, args.out)
        print(f"→ weights: {args.out}")


if __name__ == "__main__":
    main()
