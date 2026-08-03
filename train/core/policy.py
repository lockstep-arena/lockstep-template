"""A policy network built FROM the env's observation/action spaces.

No game constants anywhere: each entry of the observation ``Dict`` becomes an
input stream by its own shape and dtype, and the action head is sized from
the action space. The ONNX signature is DERIVED — input names are the obs
Dict keys, the output is ``action`` — which is exactly the signature the
game's agent shell states, because both sides read the same spaces.

Stream kinds (the two the training contract allows):

- ``uint8`` image ``Box`` (H×W×C) — a CNN stream. The graph input is the
  normalized NCHW float tensor (pixels / 255), matching what the shells feed
  at match time; :func:`obs_to_tensors` does the same normalization on the
  way into the network during training, so there is nothing to drift.
- ``float32`` vector ``Box`` (D,) — a LayerNorm+MLP stream. LayerNorm FIRST,
  and it is not decoration: such vectors routinely mix bounded components
  (quaternions in [-1, 1]) with raw counts in the thousands and unbounded
  scores. Fed straight into a Linear those channels dominate and the rest is
  lost in them. (The vector layout is frozen by the game's codec, so it
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
    return box.dtype == np.uint8 and len(box.shape) == 3


def _is_vector(box: spaces.Box) -> bool:
    return box.dtype == np.float32 and len(box.shape) == 1


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

        self._streams = nn.ModuleDict()
        feature_widths = []
        for name, box in observation_space.spaces.items():
            if not isinstance(box, spaces.Box):
                raise TypeError(f"obs {name!r} must be a Box, got {type(box).__name__}")
            if _is_image(box):
                h, w, c = box.shape
                # The graph input is normalized NCHW float — see module docs.
                self.input_shapes[name] = (c, h, w)
                # Aggressive early striding: game strips are big, mostly-empty
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
                stream = nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Linear(d, 128),
                    nn.ReLU(),
                )
                width = 128
            else:
                raise TypeError(
                    f"obs {name!r} is {box.dtype} {box.shape}; the training "
                    "contract allows uint8 H×W×C images and float32 vectors"
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
        parts = [
            self._streams[name](tensor)
            for name, tensor in zip(self.input_names, inputs, strict=True)
        ]
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
        without constructing an env. Shapes are the Gymnasium OBS shapes
        (H×W×C for images), not the graph shapes.
        """
        obs = {}
        for name in self.input_names:
            shape = self.input_shapes[name]
            if len(shape) == 3:
                c, h, w = shape
                obs[name] = {"dtype": "uint8", "shape": [h, w, c]}
            else:
                obs[name] = {"dtype": "float32", "shape": list(shape)}
        return {"obs": obs, "action_len": self.action_len}


def policy_from_signature(sig: dict) -> Policy:
    """Rebuild an (untrained) net from :meth:`Policy.space_signature`."""
    obs = {}
    for name, entry in sig["obs"].items():
        if entry["dtype"] == "uint8":
            obs[name] = spaces.Box(0, 255, tuple(entry["shape"]), dtype=np.uint8)
        else:
            obs[name] = spaces.Box(
                -np.inf, np.inf, tuple(entry["shape"]), dtype=np.float32
            )
    action = spaces.Box(-1.0, 1.0, (sig["action_len"],), dtype=np.float32)
    return Policy(spaces.Dict(obs), action)


def obs_to_tensors(obs: dict, net: Policy) -> tuple[torch.Tensor, ...]:
    """Batched vector-env observation dict -> the network's input tuple.

    Images are scaled to 0..1 HERE, exactly as the shells do at match time —
    the observation space declares ``uint8`` (it is an image, and saying
    otherwise would be a lie about the space), so the scaling has to happen
    on the way into the network on both sides. Vectors pass through.
    """
    tensors = []
    for name in net.input_names:
        arr = obs[name]
        if len(net.input_shapes[name]) == 3:
            t = torch.from_numpy(np.asarray(arr, dtype=np.float32) / 255.0)
            t = t.permute(0, 3, 1, 2)  # NHWC -> NCHW
        else:
            t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
        tensors.append(t)
    return tuple(tensors)
