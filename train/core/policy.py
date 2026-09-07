"""A policy network built FROM the env's observation/action spaces.

No environment constants anywhere, and no shape rules either: every entry
of the observation ``Dict`` — whatever its dtype and rank — becomes one
named input stream, and the action head is sized from the action space.
The ONNX signature is DERIVED (input names are the obs Dict keys, the
output is ``action``), which is exactly the signature the generic ONNX
agent shell binds at match time, because both sides read the same
declaration.

The starter stream is the same for every value: flatten → LayerNorm →
Linear → ReLU. LayerNorm first, and it is not decoration: a declared
vector routinely mixes bounded components (quaternions in [-1, 1]) with
raw counts in the thousands; fed straight into a Linear, those channels
dominate and the rest is lost. (The layout is frozen by the wire
declaration, so it cannot normalize for you.) Which stream serves which
value is the AGENT's decision: ``task create-agent`` writes a ``model.py``
beside your policy with one named stream per observation — replace a
line there to give an image a convolution or a window a recurrent layer.
:class:`Policy` takes that ``streams`` dict and adds the trunk and heads.

Two dtype facts are part of the exported graph's contract, not shape
rules (``docs/wire.md``, *The ONNX signature*):

- ``u8`` values are fed to the graph as ``f32`` divided by 255 — that is
  what the shell feeds — so :func:`obs_to_tensors` divides on the way in
  during training and the graph input is the declared shape as ``f32``.
- ``i32`` values are fed as ``int32``; the cast to float happens INSIDE
  the graph (:meth:`Policy.features`), so the exported input keeps the
  declared dtype.

Streams are reduced separately and fused late; the value head exists for
PPO and is deliberately NOT exported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

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

#: Width of the default per-value stream and of the fused trunk.
STREAM_WIDTH = 128
TRUNK_WIDTH = 256


def declared_dtype(box: spaces.Box) -> str:
    """The wire dtype of a Box, as the declaration spelled it:
    ``"uint8"`` (u8), ``"int32"`` (i32) or ``"float32"`` (f32)."""
    if box.dtype == np.uint8:
        return "uint8"
    if box.dtype == np.int32:
        return "int32"
    return "float32"


def graph_dtype(declared: str) -> str:
    """The dtype of the ONNX graph INPUT for a declared dtype: ``i32`` is
    fed as ``int32``; ``u8`` arrives as ``f32 ÷ 255``; ``f32`` as is."""
    return "int32" if declared == "int32" else "float32"


def flat_stream(shape: tuple[int, ...], width: int = STREAM_WIDTH, norm: bool = True) -> nn.Module:
    """The starter stream for ANY value: flatten → LayerNorm → Linear → ReLU.

    Takes ``[batch, *shape]`` (already float), returns ``[batch, width]``.
    ``norm=False`` drops the LayerNorm — for a value with few elements
    whose absolute scale or mean IS the signal (LayerNorm removes both,
    per sample).
    """
    numel = int(np.prod(shape)) if shape else 1
    layers: list[nn.Module] = [nn.Flatten()]
    if norm:
        layers.append(nn.LayerNorm(numel))
    layers += [nn.Linear(numel, width), nn.ReLU()]
    return nn.Sequential(*layers)


def default_streams(observation_space: spaces.Dict) -> dict[str, nn.Module]:
    """One :func:`flat_stream` per declared observation — what an agent
    gets when its ``model.py`` is absent."""
    return {
        name: flat_stream(tuple(int(d) for d in box.shape))
        for name, box in observation_space.spaces.items()
    }


class Policy(nn.Module):
    """Per-obs-entry streams -> late-fusion trunk -> tanh action + value.

    ``streams`` maps every observation name to a module taking that
    value's batched float tensor (declared shape, ``u8`` already ÷ 255,
    ``i32`` already cast) and returning ``[batch, width]``; each stream's
    width is measured with a dry run, so a stream can be anything.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        streams: Mapping[str, nn.Module] | None = None,
        trunk_width: int = TRUNK_WIDTH,
    ):
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
        #: name -> the DECLARED shape (sans batch) — also the graph's.
        self.input_shapes: dict[str, tuple[int, ...]] = {}
        #: name -> the DECLARED dtype ("uint8" | "int32" | "float32"). The
        #: graph input dtype follows from it (:func:`graph_dtype`).
        self.input_dtypes: dict[str, str] = {}
        for name, box in observation_space.spaces.items():
            if not isinstance(box, spaces.Box):
                raise TypeError(f"obs {name!r} must be a Box, got {type(box).__name__}")
            self.input_shapes[name] = tuple(int(d) for d in box.shape)
            self.input_dtypes[name] = declared_dtype(box)

        if streams is None:
            streams = default_streams(observation_space)
        missing = [n for n in self.input_names if n not in streams]
        extra = [n for n in streams if n not in self.input_names]
        if missing or extra:
            raise ValueError(
                f"streams must name every declared observation exactly once: "
                f"missing {missing}, unknown {extra} (declared: {self.input_names})"
            )
        self.streams = nn.ModuleDict({name: streams[name] for name in self.input_names})

        widths = []
        with torch.no_grad():
            for name in self.input_names:
                probe = torch.zeros(1, *self.input_shapes[name])
                out = self.streams[name](probe)
                if out.dim() != 2 or out.shape[0] != 1:
                    raise ValueError(
                        f"stream {name!r} must return [batch, width], got {tuple(out.shape)}"
                    )
                widths.append(int(out.shape[1]))

        self.trunk = nn.Sequential(
            nn.Linear(sum(widths), trunk_width),
            nn.ReLU(),
        )
        self.feature_width = trunk_width
        self.mu = nn.Linear(trunk_width, self.action_len)
        self.value = nn.Linear(trunk_width, 1)
        # State-independent log-std, the standard continuous-control choice.
        self.log_std = nn.Parameter(torch.zeros(self.action_len) - 0.5)

    # ── forward paths ────────────────────────────────────────────

    def features(self, *inputs: torch.Tensor) -> torch.Tensor:
        parts = []
        for name, tensor in zip(self.input_names, inputs, strict=True):
            if self.input_dtypes[name] == "int32":
                # Part of the exported graph: the shell feeds int32.
                tensor = tensor.float()
            parts.append(self.streams[name](tensor))
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
        without constructing an env. Shapes and dtypes are the DECLARED
        ones — the graph is derived from them the same way every time.
        """
        obs = {
            name: {"dtype": self.input_dtypes[name], "shape": list(self.input_shapes[name])}
            for name in self.input_names
        }
        return {"obs": obs, "action_len": self.action_len}


#: Anything that builds a policy from the two spaces — :class:`Policy`
#: itself, or an agent's ``model.build_policy``.
PolicyBuilder = Callable[[spaces.Dict, spaces.Box], Policy]


def spaces_from_signature(sig: dict) -> tuple[spaces.Dict, spaces.Box]:
    """The (observation, action) spaces a :meth:`Policy.space_signature`
    describes — bounds are the dtype's full range (bounds are not part of
    the network)."""
    obs = {}
    for name, entry in sig["obs"].items():
        shape = tuple(entry["shape"])
        if entry["dtype"] == "uint8":
            obs[name] = spaces.Box(0, 255, shape, dtype=np.uint8)
        elif entry["dtype"] == "int32":
            info = np.iinfo(np.int32)
            obs[name] = spaces.Box(info.min, info.max, shape, dtype=np.int32)
        else:
            obs[name] = spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
    action = spaces.Box(-1.0, 1.0, (sig["action_len"],), dtype=np.float32)
    return spaces.Dict(obs), action


def policy_from_signature(sig: dict, build: PolicyBuilder = Policy) -> Policy:
    """Rebuild an (untrained) net from :meth:`Policy.space_signature`,
    through ``build`` — the agent's ``model.build_policy`` when it has one,
    so the rebuilt streams match the checkpoint's."""
    obs, action = spaces_from_signature(sig)
    return build(obs, action)


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


def env_to_normalized(value: np.ndarray, action_space: spaces.Box) -> np.ndarray:
    """The inverse of :func:`actions_to_env`: a value in the DECLARED
    units -> the ``[-1, 1]`` the graph must emit for the shell to play it.
    An open bound passes through unscaled; the result is clamped."""
    v = np.asarray(value, dtype=np.float32)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    bounded = (np.abs(low) < _OPEN_BOUND) & (np.abs(high) < _OPEN_BOUND)
    span = np.where(bounded, high - low, 1.0)
    span = np.where(span == 0, 1.0, span)
    scaled = (v - low) / span * 2.0 - 1.0
    return np.clip(np.where(bounded, scaled, v), -1.0, 1.0).astype(np.float32)


def obs_to_tensors(obs: dict, net: nn.Module) -> tuple[torch.Tensor, ...]:
    """Batched vector-env observation dict -> the network's input tuple.

    Keyed on the DECLARED dtype only. ``u8`` values are scaled to 0..1
    HERE, exactly as the generic shell does at match time — the
    observation space declares ``uint8`` (saying otherwise would be a lie
    about the space), so the scaling happens on the way into the network
    on both sides. ``i32`` values stay int32 (the graph casts). ``f32``
    values pass through. No axis shuffling, ever: the shell feeds the
    declared shape verbatim.
    """
    tensors = []
    for name in net.input_names:
        arr = obs[name]
        declared = net.input_dtypes[name]
        if declared == "int32":
            t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int32))
        elif declared == "uint8":
            t = torch.from_numpy(np.asarray(arr, dtype=np.float32) / 255.0)
        else:
            t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        tensors.append(t)
    return tuple(tensors)
