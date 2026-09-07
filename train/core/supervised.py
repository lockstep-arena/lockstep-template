"""The supervised recipe: learn from the truth the environment reveals.

Data environments score a decision AFTER the fact: the truth of the
transaction you saw at tick ``t`` arrives later — as a row of a feedback
window (columns saying how old the decision is, what you decided and what
was true) or as the reward ``reward_lag_ticks`` later. That makes them
supervised-learning problems as much as control problems, and this recipe
treats them so::

    task train AGENT=<name> RECIPE=supervised [SEEDS=16] [EPOCHS=20]

1. **Collect.** Run the engine over ``seeds`` public seeds playing the
   NEUTRAL action (the labels do not depend on what you decide; a decision
   is scored against what was true). Every tick's observation goes into
   ``lockstep_train.LabelledStream``, which hands back ``(observation at
   decision time, label)`` pairs the tick each label lands.
2. **Fit.** A classification head (cross-entropy, class-weighted, since
   the interesting class is usually the rare one) on the SAME trunk the
   PPO recipe uses — the streams in the agent's ``model.py`` — when the
   truth takes a few distinct integer values; a regression head (tanh +
   MSE on the normalized target) otherwise, and always for a lagged
   reward. 20% of the pairs are held out and scored every epoch.
3. **Export.** The exported graph keeps the declared signature: one input
   per observation by name, one ``action`` output in [-1, 1]. The head's
   prediction lands on ONE element of the declared action (element 0 of
   the action ``agent.toml`` names) as the value the shell maps back onto
   that element's bounds — a class is emitted as its declared code, a
   regression as its value — and every other element is neutral.

Where the truth is found is read from the ``[supervised]`` block of
``agent.toml`` (scaffolded from the declaration; see ``train/agents.py``),
never guessed: the value and column names are the declaration's own, and
an environment that names them differently is joined by naming them.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from .policy import Policy, PolicyBuilder, env_to_normalized

#: Distinct integer truth values up to this many make a classification
#: problem; more (or non-integers) make a regression.
MAX_CLASSES = 32

#: supervised.csv column order (one row per epoch).
METRICS_FIELDS = ["epoch", "wall_seconds", "train_loss", "val_loss", "val_score"]


@dataclass(frozen=True)
class LabelSource:
    """Where the truth of a decision shows up — the ``[supervised]`` block."""

    #: Feedback mode: the rank-2 observation whose rows are resolved decisions.
    value: str | None = None
    age_col: str = "age_ticks"
    decision_col: str = "your_decision"
    truth_col: str = "truth"
    valid_col: str | None = None
    #: Reward-lag mode: the reward at tick t labels the decision at t - lag.
    reward_lag: int | None = None
    #: The declared action the truth trains (element 0); default the first.
    action: str | None = None

    @classmethod
    def from_block(cls, block: dict) -> LabelSource:
        return cls(
            value=block.get("value"),
            age_col=block.get("age_col", "age_ticks"),
            decision_col=block.get("decision_col", "your_decision"),
            truth_col=block.get("truth_col", "truth"),
            valid_col=block.get("valid_col"),
            reward_lag=int(block["reward_lag_ticks"]) if block.get("reward_lag_ticks") is not None else None,
            action=block.get("action"),
        )

    @property
    def mode(self) -> str:
        return "reward-lag" if self.reward_lag is not None else "feedback"

    def describe(self) -> str:
        if self.reward_lag is not None:
            return f"the reward {self.reward_lag} ticks after each decision"
        cols = f"{self.age_col} / {self.decision_col} / {self.truth_col}"
        if self.valid_col:
            cols += f" (rows with {self.valid_col} = 0 are padding)"
        return f"rows of `{self.value}` — columns {cols}"


def make_stream(init, source: LabelSource):
    """The ``LabelledStream`` for this declaration and label source."""
    from lockstep_train import LabelledStream

    if source.reward_lag is not None:
        return LabelledStream(init, reward_lag=source.reward_lag)
    if not source.value:
        raise SystemExit(
            "the supervised recipe needs a label source: a [supervised] block in "
            "agent.toml (task create-agent writes one when the declaration reveals "
            "truth — a feedback window or meta.reward_lag_ticks), or --label-value / "
            "--reward-lag on the command line"
        )
    return LabelledStream(
        init,
        value=source.value,
        age_col=source.age_col,
        decision_col=source.decision_col,
        truth_col=source.truth_col,
        valid_col=source.valid_col,
    )


@dataclass
class Dataset:
    """Collected pairs: one stacked array per observation, one label each."""

    obs: dict[str, np.ndarray]
    labels: np.ndarray
    seeds: int
    ticks: int
    dropped: int

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def collect(
    engine: str,
    source: LabelSource,
    seeds: range,
    ticks_per_seed: int | None = None,
    time_limit_ticks: int | None = None,
) -> Dataset:
    """Drive the engine over ``seeds`` with the neutral action and join
    every label to the observation it belongs to."""
    from lockstep_train.env import LockstepEnv

    env = LockstepEnv(engine_source=engine, time_limit_ticks=time_limit_ticks)
    try:
        init = env.seat_init
        stream = make_stream(init, source)
        names = [t.name for t in init.obs]
        columns: dict[str, list[np.ndarray]] = {n: [] for n in names}
        labels: list[float] = []
        ticks = 0
        neutral = _neutral_action(env)
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            stream.reset()
            _take(stream.feed(info["tick"], obs, 0.0), source, columns, labels)
            steps = 0
            while True:
                obs, reward, terminated, truncated, info = env.step(neutral)
                ticks += 1
                steps += 1
                _take(stream.feed(info["tick"], obs, float(reward)), source, columns, labels)
                if terminated or truncated or (ticks_per_seed and steps >= ticks_per_seed):
                    break
        return Dataset(
            obs={n: np.stack(columns[n]) if columns[n] else np.zeros((0, *env.observation_space[n].shape)) for n in names},
            labels=np.asarray(labels, dtype=np.float32),
            seeds=len(seeds),
            ticks=ticks,
            dropped=stream.dropped,
        )
    finally:
        env.close()


def _neutral_action(env):
    init = env.seat_init
    if len(init.actions) == 1:
        t = init.actions[0]
        return t.neutral_f32().reshape(t.shape)
    return {t.name: t.neutral_f32().reshape(t.shape) for t in init.actions}


def _take(pairs, source: LabelSource, columns: dict, labels: list[float]) -> None:
    for obs_then, label in pairs:
        target = label.reward if source.reward_lag is not None else label.truth
        if target is None:
            continue
        for name, arr in obs_then.items():
            columns[name].append(np.asarray(arr))
        labels.append(float(target))


# ---------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------


def classes_of(labels: np.ndarray) -> list[float] | None:
    """The sorted distinct truth values when they make a classification
    problem (integers, 2..MAX_CLASSES of them), else ``None``."""
    distinct = np.unique(labels)
    if len(distinct) < 2 or len(distinct) > MAX_CLASSES:
        return None
    if not np.all(np.isfinite(distinct)) or not np.all(np.rint(distinct) == distinct):
        return None
    return [float(v) for v in distinct]


class SupervisedPolicy(nn.Module):
    """A classification or regression head on an agent's trunk, exported
    exactly like :class:`Policy` (same ``input_names`` / ``input_shapes``
    / ``input_dtypes`` / ``action_len`` surface, same ``action`` output).

    ``element`` is the index in the flat action the prediction lands on;
    ``classes`` the declared codes a classification head chooses between
    (``None`` for regression). Both are recorded in the weights file.
    """

    def __init__(
        self,
        base: Policy,
        action_space: spaces.Box,
        element: int,
        classes: list[float] | None,
    ):
        super().__init__()
        self.base = base
        self.input_names = base.input_names
        self.input_shapes = base.input_shapes
        self.input_dtypes = base.input_dtypes
        self.action_len = base.action_len
        self.element = int(element)
        self.classes = list(classes) if classes else None
        width = len(self.classes) if self.classes else 1
        self.head = nn.Linear(base.feature_width, width)
        # The [-1, 1] value the shell maps back onto this element's bounds
        # for each class code — the whole affine map, precomputed.
        low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
        elem_space = spaces.Box(low[self.element : self.element + 1], high[self.element : self.element + 1])
        if self.classes:
            values = np.stack([env_to_normalized([c], elem_space) for c in self.classes]).reshape(-1)
            self.register_buffer("class_values", torch.from_numpy(values.astype(np.float32)))
        self._elem_space = elem_space
        mask = np.zeros(self.action_len, dtype=np.float32)
        mask[self.element] = 1.0
        self.register_buffer("mask", torch.from_numpy(mask))

    def logits(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.base.features(*inputs))

    def normalized_target(self, truth: np.ndarray) -> torch.Tensor:
        """Regression targets in the graph's [-1, 1] for truths in declared units."""
        return torch.from_numpy(
            np.stack([env_to_normalized([v], self._elem_space) for v in truth]).reshape(-1).astype(np.float32)
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """The EXPORTED path: every element neutral (0) except the one the
        head decides — a class's declared code, or the regressed value."""
        out = self.logits(*inputs)
        if self.classes:
            value = self.class_values[out.argmax(dim=1)]
        else:
            value = torch.tanh(out).squeeze(1)
        return self.mask * value.unsqueeze(1)

    def space_signature(self) -> dict:
        return self.base.space_signature()

    def head_signature(self) -> dict:
        return {"element": self.element, "classes": self.classes}


def action_element(init, action_name: str | None) -> int:
    """The flat index of element 0 of the named action (the first when
    unnamed) — where the prediction lands in the exported ``action``."""
    if len(init.actions) != 1:
        raise SystemExit(
            f"the supervised recipe trains a single declared action; this "
            f"engine declares {len(init.actions)}"
        )
    t = init.actions[0]
    if action_name and action_name != t.name:
        raise SystemExit(
            f"[supervised] action = {action_name!r} but the engine declares "
            f"{t.name!r} — regenerate: task create-agent"
        )
    return 0


def _tensors(data: Dataset, net: nn.Module, idx: np.ndarray) -> tuple[torch.Tensor, ...]:
    from .policy import obs_to_tensors

    return obs_to_tensors({n: data.obs[n][idx] for n in net.input_names}, net)


def train_supervised(
    engine: str,
    source: LabelSource,
    build: PolicyBuilder = Policy,
    seeds: int = 16,
    epochs: int = 20,
    ticks_per_seed: int | None = None,
    time_limit_ticks: int | None = None,
    minibatch: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    out_dir: Path = Path("out"),
) -> SupervisedPolicy:
    """Collect → fit → return the head-on-trunk network, ready to export."""
    from lockstep_train.env import LockstepEnv

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"  label source: {source.describe()}", flush=True)
    started = time.time()
    data = collect(engine, source, range(seeds), ticks_per_seed, time_limit_ticks)
    print(
        f"  collected {len(data)} labelled observations from {data.seeds} seeds "
        f"({data.ticks} ticks, {data.dropped} labels with no observation to join) "
        f"in {time.time() - started:.1f}s",
        flush=True,
    )
    if len(data) < 2:
        raise SystemExit(
            "no labels arrived: the engine revealed no truth over these seeds — "
            "check the [supervised] block in agent.toml against `task info`, or "
            "raise SEEDS="
        )

    env = LockstepEnv(engine_source=engine)
    try:
        obs_space, act_space = env.observation_space, env.action_space
        element = action_element(env.seat_init, source.action)
    finally:
        env.close()
    if not isinstance(act_space, spaces.Box):
        raise SystemExit("the supervised recipe needs a single declared action (a Box)")

    classes = classes_of(data.labels)
    net = SupervisedPolicy(build(obs_space, act_space), act_space, element, classes)
    if classes:
        counts = np.array([(data.labels == c).sum() for c in classes], dtype=np.float32)
        shares = ", ".join(f"{c:g}: {int(n)}" for c, n in zip(classes, counts))
        print(f"  classification over declared codes {{{shares}}}", flush=True)
        # Inverse-frequency weights: the rare class is usually the one that
        # matters, and an unweighted head learns to never predict it.
        weights = torch.from_numpy((counts.sum() / np.maximum(counts, 1.0)) / len(classes))
        targets = torch.from_numpy(np.searchsorted(np.array(classes), data.labels).astype(np.int64))
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        print(
            f"  regression on the truth (range {data.labels.min():g} .. {data.labels.max():g})",
            flush=True,
        )
        targets = net.normalized_target(data.labels)
        loss_fn = nn.MSELoss()

    order = rng.permutation(len(data))
    n_val = max(1, len(data) // 5)
    val_idx, train_idx = order[:n_val], order[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx

    dev = torch.device(device) if device else torch.device("cpu")
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "supervised.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            net.train()
            perm = rng.permutation(train_idx)
            losses = []
            for start in range(0, len(perm), minibatch):
                idx = perm[start : start + minibatch]
                inputs = tuple(t.to(dev) for t in _tensors(data, net, idx))
                target = targets[idx].to(dev)
                out = net.logits(*inputs)
                loss = loss_fn(out, target) if classes else loss_fn(torch.tanh(out).squeeze(1), target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
            net.eval()
            with torch.no_grad():
                inputs = tuple(t.to(dev) for t in _tensors(data, net, val_idx))
                target = targets[val_idx].to(dev)
                out = net.logits(*inputs)
                if classes:
                    val_loss = float(loss_fn(out, target))
                    val_score = float((out.argmax(dim=1) == target).float().mean())
                    score_name = "val_accuracy"
                else:
                    pred = torch.tanh(out).squeeze(1)
                    val_loss = float(loss_fn(pred, target))
                    val_score = float((pred - target).abs().mean())
                    score_name = "val_mae"
            elapsed = time.time() - started
            writer.writerow(
                {
                    "epoch": epoch,
                    "wall_seconds": round(elapsed, 3),
                    "train_loss": float(np.mean(losses)),
                    "val_loss": val_loss,
                    "val_score": val_score,
                }
            )
            f.flush()
            print(
                f"  epoch {epoch:>3}/{epochs}  train_loss={np.mean(losses):.4f}  "
                f"val_loss={val_loss:.4f}  {score_name}={val_score:.3f}  {elapsed:6.1f}s",
                flush=True,
            )
    net.to(torch.device("cpu"))
    return net
