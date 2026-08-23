"""Shared-policy self-play PPO over a game's PettingZoo ``ParallelEnv`` —
BOTH seats learning at once.

``train.core.train`` trains one seat of a Gymnasium env (optionally against
a frozen ``--opponent``). This loop is the step beyond that for games whose
seats are genuinely adversarial: the game's parallel env steps every seat
every tick, ONE network chooses every seat's action from that seat's own
observation, and every seat's transition lands in the same PPO buffer. The
opponent is never frozen — it is the same policy, one update behind
nothing. This is the first rung of self-play (the standard one); league or
population play is deliberately out of scope here.

Why one policy and not one per seat: competition is one agent per seat
answering ``on_tick`` from its own view, and the seats are symmetric (same
observation layout, same action space — the games-side conformance suite
asserts it). A single policy is exactly what gets exported, so the artifact
is unchanged: ONE ``policy.onnx`` per bundle, through the same export →
parity → stage path as the single-seat loop. PettingZoo is a training-side
view only.

What is shared with :mod:`train.core.train`, on purpose: the network
(:class:`train.core.policy.Policy`), the GAE helper, the checkpoint format
(``--resume`` works across both loops), the device choice and the
``metrics.csv`` layout — so the two loops are the same experiment with one
variable changed. The rollout and update below are written out in full
rather than hidden behind an adapter, so what actually happens to two seats'
experience is on this page.

Not a vector API: this loop drives ONE parallel env. PettingZoo has no
vector API and this repo does not invent one — and it does not need to: the
engine steps at tens of thousands of ticks per second, so a two-seat loop is
bound by the network, not the env. If a game ever ships a "stacked
sessions" helper, this is the place it would plug in.
"""

from __future__ import annotations

import csv
import importlib
import time
from pathlib import Path

import numpy as np
import torch

from .discovery import require_parallel_env_id
from .policy import Policy, obs_to_tensors
from .train import METRICS_FIELDS, discounted_advantages, pick_device, save_checkpoint

#: metrics.csv columns — the single-seat layout plus one self-play column.
#: In a zero-sum duel the pooled ``mean_return`` is ~0 by construction (one
#: seat's gain is the other's loss), so it says nothing about whether any
#: jousting happened; ``mean_abs_return`` does.
SELF_PLAY_METRICS_FIELDS = [*METRICS_FIELDS, "mean_abs_return"]


def resolve_parallel_env(spec: dict, mode: str, engine: str | None, time_limit_ticks: int | None):
    """Construct the game's PettingZoo ``ParallelEnv`` from its GameSpec.

    Contract v2 carries ``parallel_env_id``, a ``module:callable`` locator
    (PettingZoo has no registry like Gymnasium's, so the spec names the
    factory itself). The factory takes the same keywords the contract fixes
    for the Gymnasium env — ``mode``, ``engine_source``, ``time_limit_ticks``.
    """
    locator = require_parallel_env_id(spec)
    module_name, _, attr = locator.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(
            f"could not import the parallel env {locator!r}: {e}\n"
            "The game wheel keeps PettingZoo as an optional extra — install it with:\n"
            f'  pip install "lockstep-game-{spec["slug"]}[pettingzoo]"'
        ) from e
    factory = getattr(module, attr)
    return factory(mode=mode, engine_source=engine, time_limit_ticks=time_limit_ticks)


def _batch_seats(obs: dict, agents: list[str]) -> dict:
    """PettingZoo's per-agent observation dict -> one batched observation
    dict (seats stacked along axis 0, in seat order), so both seats go
    through the network in a single forward pass."""
    first = obs[agents[0]]
    return {key: np.stack([obs[a][key] for a in agents]) for key in first}


def train_self_play(
    spec: dict,
    mode: str,
    steps: int,
    engine: str | None = None,
    time_limit_ticks: int | None = None,
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
    """Train one policy on every seat's experience of one parallel env.

    ``steps`` counts seat-steps (every seat's transition is one sample), so
    a 2-seat env collects two samples per engine tick — the same accounting
    as ``num_envs=2`` in the single-seat loop.
    """
    torch.manual_seed(seed)
    env = resolve_parallel_env(spec, mode, engine, time_limit_ticks)
    agents = list(env.possible_agents)
    num_seats = len(agents)
    if num_seats < 2:
        raise SystemExit(f"parallel env for {spec['slug']!r} has {num_seats} seat(s); self-play needs 2+")

    # One network for every seat — so every seat must see and act in the
    # same spaces. (The games-side conformance suite already asserts this
    # equals the Gymnasium env's spaces; checked here anyway, because a
    # mismatch would otherwise surface as a tensor-shape error mid-rollout.)
    obs_space = env.observation_space(agents[0])
    act_space = env.action_space(agents[0])
    for a in agents[1:]:
        if env.observation_space(a) != obs_space or env.action_space(a) != act_space:
            raise SystemExit(
                f"seat {a!r} has different spaces from {agents[0]!r}; a shared "
                "policy needs symmetric seats"
            )
    net = Policy(obs_space, act_space)

    dev = torch.device(device) if device else pick_device()
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if dev.type == "cpu":
        learner = net
    else:
        # `net` is the CPU rollout copy; `learner` takes the gradient steps
        # on the accelerator and syncs back each rollout.
        learner = Policy(obs_space, act_space).to(dev)
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
        f"  update device: {dev}   seats: {num_seats} ({type(env).__name__}, "
        "shared policy, experience pooled)",
        flush=True,
    )

    metrics_file = open(  # noqa: SIM115 — held across the whole run
        metrics_path, "a" if resume and metrics_path.is_file() else "w", newline=""
    )
    metrics = csv.DictWriter(metrics_file, fieldnames=SELF_PLAY_METRICS_FIELDS)
    if metrics_file.tell() == 0:
        metrics.writeheader()

    obs, _ = env.reset(seed=seed)
    episode_return = np.zeros(num_seats, dtype=np.float64)
    episode_returns: list[float] = []  # one entry per seat per finished episode
    started = time.time()
    started_at = collected

    while collected < steps:
        # Linear anneal of lr and entropy pressure over the FULL run (the
        # same schedule as the single-seat loop, for the same reason).
        frac = max(0.0, 1.0 - collected / steps)
        for group in opt.param_groups:
            group["lr"] = lr * frac
        ent_coef = entropy_coef * frac

        # Buffers are [T, seats, ...]: row t holds BOTH seats' transitions
        # of the same engine tick. GAE runs down each seat's column
        # separately; the update then pools everything.
        buf_obs: list[tuple[torch.Tensor, ...]] = []
        buf_raw, buf_logp = [], []
        buf_val = np.zeros((rollout, num_seats), dtype=np.float32)
        buf_rew = np.zeros((rollout, num_seats), dtype=np.float32)
        buf_done = np.zeros((rollout, num_seats), dtype=np.float32)

        for t in range(rollout):
            # Both seats' observations through the ONE network at once.
            tensors = obs_to_tensors(_batch_seats(obs, agents), net)
            with torch.no_grad():
                action, raw, log_prob, value = net.act(*tensors)
            actions = action.numpy().astype(np.float32)

            obs, rewards, terminations, truncations, _ = env.step(
                {a: actions[i] for i, a in enumerate(agents)}
            )
            reward = np.array([rewards[a] for a in agents], dtype=np.float32)
            done = np.array(
                [terminations[a] or truncations[a] for a in agents], dtype=np.float32
            )

            buf_obs.append(tensors)
            buf_raw.append(raw)
            buf_logp.append(log_prob)
            buf_val[t] = value.numpy()
            buf_rew[t] = reward
            buf_done[t] = done

            episode_return += reward
            collected += num_seats

            if not env.agents:
                # The engine ended the episode for every seat together.
                # Book each seat's return, then start the next duel NOW so
                # the observation stored for t+1 is a real first observation
                # (the same SAME_STEP discipline the single-seat loop relies
                # on: nothing in the buffer is a transition that never ran).
                episode_returns.extend(float(r) for r in episode_return)
                episode_return[:] = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            last_value = net.act(*obs_to_tensors(_batch_seats(obs, agents), net))[3].numpy()

        advantages, returns = discounted_advantages(
            buf_rew, buf_val, buf_done, last_value, gamma, lam
        )
        adv_t = torch.from_numpy(advantages.reshape(-1)).to(dev)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # [T, seats, ...] -> [T*seats, ...]: the pooled batch. Every buffer
        # flattens in the same time-major, seat-minor order, so row i of one
        # lines up with row i of the rest.
        obs_t = tuple(
            torch.cat([step[i] for step in buf_obs]).to(dev)
            for i in range(len(net.input_names))
        )
        raw_t = torch.cat(buf_raw).to(dev)
        logp_t = torch.cat(buf_logp).to(dev)
        ret_t = torch.from_numpy(returns.reshape(-1)).to(dev)

        # The clipped PPO update — identical to the single-seat loop's; the
        # only thing that differs is where the rows came from.
        batch = rollout * num_seats
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
                torch.nn.utils.clip_grad_norm_(learner.parameters(), 0.5)
                opt.step()

                with torch.no_grad():
                    stat_policy.append(float(policy_loss))
                    stat_value.append(float(value_loss))
                    stat_entropy.append(float(entropy.mean()))
                    stat_clip.append(float(((ratio - 1.0).abs() > clip).float().mean()))

        if learner is not net:
            net.load_state_dict(learner.state_dict())

        save_checkpoint(checkpoint_path, net, opt, collected)

        # Recent = the last 5 episodes × every seat.
        recent = episode_returns[-5 * num_seats :]
        mean_return = sum(recent) / len(recent) if recent else float("nan")
        mean_abs = sum(abs(r) for r in recent) / len(recent) if recent else float("nan")
        elapsed = time.time() - started
        metrics.writerow(
            {
                "steps": collected,
                "wall_seconds": round(elapsed, 3),
                "episodes": len(episode_returns) // num_seats,
                "mean_return": mean_return,
                "policy_loss": np.mean(stat_policy),
                "value_loss": np.mean(stat_value),
                "entropy": np.mean(stat_entropy),
                "log_std_mean": float(net.log_std.detach().mean()),
                "clip_fraction": np.mean(stat_clip),
                "steps_per_second": round((collected - started_at) / elapsed, 1),
                "mean_abs_return": mean_abs,
            }
        )
        metrics_file.flush()
        print(
            f"  {collected:>7}/{steps} steps  "
            f"episodes={len(episode_returns) // num_seats:<4} "
            f"mean_return={mean_return:8.2f}  "
            f"mean|return|={mean_abs:6.2f}  "
            f"{elapsed:6.1f}s",
            flush=True,
        )

    env.close()
    metrics_file.close()
    return net
