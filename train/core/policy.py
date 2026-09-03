"""A policy network built FROM the env's observation/action spaces.

No environment constants anywhere: each entry of the observation ``Dict``
becomes an input stream by its own shape and dtype, and the action head is
sized from the action space. The ONNX signature is DERIVED — input names
are the obs Dict keys, the output is ``action`` — which is exactly the
signature the generic ONNX agent shell binds at match time, because both
sides read the same declaration.

Stream kinds (the value shapes the Lockstep wire carries today):

- ``uint8`` rank-3 ``Box`` — a CNN stream. The wire declares images
  CHANNEL-FIRST (an image strip declares as ``u8[1, H, W]``), and the shell
  feeds the graph the DECLARED shape with a batch dim prepended and values
  as f32/255 — so the graph input here is exactly that: declared shape,
  normalized. :func:`obs_to_tensors` does the same normalization on the way
  into the network during training, so there is nothing to drift.
- ``int32`` ``Box`` (any rank) — cast to float, flattened, LayerNorm+MLP.
  The wire uses i32 for discrete values (cue cards, mode switches); the
  graph INPUT stays int32 (that is what the shell feeds) and the cast to
  float is part of the graph.
- ``float32`` vector ``Box`` (D,) — a LayerNorm+MLP stream. LayerNorm FIRST,
  and it is not decoration: such vectors routinely mix bounded components
  (quaternions in [-1, 1]) with raw counts in the thousands and unbounded
  scores. Fed straight into a Linear those channels dominate and the rest is
  lost in them. (The vector layout is frozen by the wire declaration, so it
  cannot normalize for you.)

Streams are reduced separately and fused late; the value head exists for PPO
and is deliberately NOT exported.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

#: The derived ONNX output name. Input names are the obs Dict keys.
OUTPUT_ACTION = "action"

#: Bounds on the state-independent log-std wherever a std is materialized.
#: The lower bound is ordinary numerical hygiene. The UPPER bound exists
#: because of a demonstrated failure: on a reward landscape that stays at
#: zero, the entropy bonus is the only nonzero gradient, and over thousands
#: of updates it monotonically inflates log_std — progressively wilder
#: actions, a policy that falls over more at the end of a long run than at
#: the start. The clamp makes that failure structurally bounded.
LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0


def _is_image(box: spaces.Box) -> bool:
    """A rank-3 uint8 tensor — declared CHANNEL-FIRST by the wire."""
    return box.dtype == np.uint8 and len(box.shape) == 3


def _is_vector(box: spaces.Box) -> bool:
    return box.dtype == np.float32 and len(box.shape) == 1


def _is_ints(box: spaces.Box) -> bool:
    """Discrete-valued tensors (the wire's i32), any rank."""
    return box.dtype == np.int32


class Policy(nn.Module):
    """Per-obs-entry streams -> late-fusion trunk -> tanh action + value."""

    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Box):
        super().__init__()
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                f"observation space must be a Dict of named Boxes, "
                f"got {type(observation_space).__name__}"
            )
        (self.action_len,) = action_space.shape

        #: Obs Dict keys in declaration order — the network's input order and
        #: the ONNX input names. The platform host binds tensors by NAME, so
        #: these strings are as load-bearing as the shapes.
        self.input_names: list[str] = list(observation_space.spaces)
        #: name -> ONNX/graph input shape (sans batch), derived per stream.
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        #: name -> graph input dtype ("float32" | "int32"). u8 images enter
        #: the GRAPH as normalized float32 (the shell feeds f32/255); i32
        #: tensors enter as int32 and are cast inside the graph.
        self.input_dtypes: dict[str, str] = {}

        self._streams = nn.ModuleDict()
        feature_widths = []
        for name, box in observation_space.spaces.items():
            if not isinstance(box, spaces.Box):
                raise TypeError(f"obs {name!r} must be a Box, got {type(box).__name__}")
            if _is_image(box):
                # The wire declares images channel-first; the graph input is
                # the declared shape as normalized float — see module docs.
                c, h, w = box.shape
                self.input_shapes[name] = (c, h, w)
                self.input_dtypes[name] = "float32"
                # Aggressive early striding: env strips are big, mostly-empty
                # images whose signal is coarse.
                stream = nn.Sequential(
                    nn.Conv2d(c, 16, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=2),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                with torch.no_grad():
                    width = stream(torch.zeros(1, c, h, w)).shape[1]
            elif _is_vector(box):
                (d,) = box.shape
                self.input_shapes[name] = (d,)
                self.input_dtypes[name] = "float32"
                stream = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, 128),
                    nn.ReLU(),
                )
                width = 128
            elif _is_ints(box):
                numel = int(np.prod(box.shape))
                self.input_shapes[name] = tuple(int(d) for d in box.shape)
                self.input_dtypes[name] = "int32"
                # The int->float cast happens in `features` (it is part of
                # the exported graph); the stream sees flat floats.
                stream = nn.Sequential(
                    nn.Flatten(),
                    nn.LayerNorm(numel),
                    nn.Linear(numel, 64),
                    nn.ReLU(),
                )
                width = 64
            else:
                raise TypeError(
                    f"obs {name!r} is {box.dtype} {box.shape}; this template "
                    "handles uint8 rank-3 images (channel-first, as the wire "
                    "declares them) and float32 vectors — add a stream kind "
                    "here for anything else"
                )
            self._streams[name] = stream
            feature_widths.append(width)

        self.trunk = nn.Sequential(
            nn.Linear(sum(feature_widths), 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, self.action_len)
        self.value = nn.Linear(256, 1)
        # State-independent log-std, the standard continuous-control choice.
        self.log_std = nn.Parameter(torch.zeros(self.action_len) - 0.5)

    # ── forward paths ────────────────────────────────────────────

    def features(self, *inputs: torch.Tensor) -> torch.Tensor:
        parts = []
        for name, tensor in zip(self.input_names, inputs, strict=True):
            if self.input_dtypes[name] == "int32":
                # Part of the exported graph: the shell feeds int32.
                tensor = tensor.float()
            parts.append(self._streams[name](tensor))
        return self.trunk(torch.cat(parts, dim=1))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """The EXPORTED path: observation -> bounded action.

        ``tanh`` is what makes the shells' ``[-1, 1]`` assumption true. They
        clamp anyway, but the bound belongs here where training can see it.
        """
        return torch.tanh(self.mu(self.features(*inputs)))

    def _std(self) -> torch.Tensor:
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def act(self, *inputs: torch.Tensor):
        """Sample an action for rollout, with its log-prob and value.

        Sampling happens BEFORE the tanh, and the log-prob is of the
        pre-squash Gaussian sample — mixing squashed and unsquashed spaces
        between collection and update is the usual PPO trap here.
        """
        feats = self.features(*inputs)
        mu = self.mu(feats)
        dist = torch.distributions.Normal(mu, self._std())
        raw = dist.sample()
        log_prob = dist.log_prob(raw).sum(dim=-1)
        return torch.tanh(raw), raw, log_prob, self.value(feats).squeeze(-1)

    def evaluate(self, *inputs: torch.Tensor, raw: torch.Tensor):
        """Re-score stored pre-squash actions under the CURRENT parameters."""
        feats = self.features(*inputs)
        dist = torch.distributions.Normal(self.mu(feats), self._std())
        return (
            dist.log_prob(raw).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.value(feats).squeeze(-1),
        )

    # ── (de)serialization helpers ────────────────────────────────

    def space_signature(self) -> dict:
        """A JSON-able record of the spaces this net was built from.

        Stored in checkpoints/weights so a policy can be rebuilt for export
        without constructing an env. Shapes are the DECLARED wire shapes —
        graph shapes and obs shapes are the same thing now that images stay
        channel-first end to end.
        """
        obs = {}
        for name in self.input_names:
            shape = self.input_shapes[name]
            if self.input_dtypes[name] == "int32":
                dtype = "int32"
            elif len(shape) == 3:
                dtype = "uint8"
            else:
                dtype = "float32"
            obs[name] = {"dtype": dtype, "shape": list(shape)}
        return {"obs": obs, "action_len": self.action_len}


def policy_from_signature(sig: dict) -> Policy:
    """Rebuild an (untrained) net from :meth:`Policy.space_signature`."""
    obs = {}
    for name, entry in sig["obs"].items():
        if entry["dtype"] == "uint8":
            obs[name] = spaces.Box(0, 255, tuple(entry["shape"]), dtype=np.uint8)
        elif entry["dtype"] == "int32":
            obs[name] = spaces.Box(
                np.iinfo(np.int32).min,
                np.iinfo(np.int32).max,
                tuple(entry["shape"]),
                dtype=np.int32,
            )
        else:
            obs[name] = spaces.Box(
                -np.inf, np.inf, tuple(entry["shape"]), dtype=np.float32
            )
    action = spaces.Box(-1.0, 1.0, (sig["action_len"],), dtype=np.float32)
    return Policy(spaces.Dict(obs), action)


#: Beyond this magnitude a Box bound is the dtype range standing in for
#: "open" (``lockstep_train`` clamps ±inf to the dtype's range so the space
#: stays a valid Box); the shell passes an open bound through unscaled.
_OPEN_BOUND = 1e30


def actions_to_env(action: np.ndarray, action_space: spaces.Box) -> np.ndarray:
    """The network's ``[-1, 1]`` output -> what the env actually steps on.

    This is the SAME map the generic ONNX shell applies at match time
    (``agent-onnx``'s ``denormalize``): clamp to ``[-1, 1]``, then affinely
    onto each element's declared bounds; an open bound passes the value
    through unscaled. Training MUST step the env through this map, or the
    policy learns radians-in-``[-1, 1]`` (clamped by the engine) and the
    exported bundle, which the shell rescales onto the full joint range,
    plays a completely different action from the one that earned the
    training return. That is exactly how a go1 policy scored 80 in
    training and 0 on every sealed seed.
    """
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    bounded = (np.abs(low) < _OPEN_BOUND) & (np.abs(high) < _OPEN_BOUND)
    scaled = low + (a + 1.0) * 0.5 * (high - low)
    return np.where(bounded, scaled, a).astype(np.float32)


def obs_to_tensors(obs: dict, net: Policy) -> tuple[torch.Tensor, ...]:
    """Batched vector-env observation dict -> the network's input tuple.

    Images are scaled to 0..1 HERE, exactly as the generic shell does at
    match time — the observation space declares ``uint8`` (it is an image,
    and saying otherwise would be a lie about the space), so the scaling has
    to happen on the way into the network on both sides. No axis shuffling:
    the wire declares images channel-first and the shell feeds the declared
    shape verbatim. Vectors pass through.
    """
    tensors = []
    for name in net.input_names:
        arr = obs[name]
        if net.input_dtypes[name] == "int32":
            t = torch.from_numpy(np.asarray(arr, dtype=np.int32))
        elif len(net.input_shapes[name]) == 3:
            t = torch.from_numpy(np.asarray(arr, dtype=np.float32) / 255.0)
        else:
            t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        tensors.append(t)
    return tuple(tensors)
