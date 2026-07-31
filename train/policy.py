"""The policy network, and the ONNX signature it is exported under.

The network is small on purpose: a short laptop run produces a real trained
policy that loads and infers in a real match — not a strong dancer. Crank
``--steps`` (by orders of magnitude) for that, or replace the whole network;
only the signature below is load-bearing.

# The ONNX signature is the contract

Two inputs and one output, and the names matter — the platform's inference
host binds tensors by string, and the agent-shell component (shipped inside
the ``lockstep-game-dance-off`` wheel) states the same signature::

    inputs:  "marquee" f32[1, 1, 64, 256]   pixels / 255
             "agent"   f32[1, 62]           proprioception
    output:  "action"  f32[1, ACTION_LEN]   tanh-bounded

``ACTION_LEN`` is the only thing that differs between the tiers: 48 for
``servo-assist`` (36 pose targets + 12 effort shares) and 36 for
``raw-torque`` (per-joint torque). A bundle built for one tier must not be
handed to the other — the shells check the output width and refuse.

Observations come from the env already shaped; the encoding is the same Rust
(``lockstep_dance_off._native``) the shells call in a real match, so there is
nothing here to drift.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lockstep_dance_off import _native

MARQUEE_H: int = _native.MARQUEE_H
MARQUEE_W: int = _native.MARQUEE_W
AGENT_LEN: int = _native.AGENT_LEN

#: ONNX input/output names. The host looks these up by string at call time,
#: so they are as load-bearing as the shapes.
INPUT_MARQUEE = "marquee"
INPUT_AGENT = "agent"
OUTPUT_ACTION = "action"


class Policy(nn.Module):
    """Marquee CNN + proprioception MLP -> tanh action, with a value head.

    The two input streams are different kinds of signal: the marquee is an
    IMAGE (what am I being asked to do, and how soon), the agent vector is
    BODY STATE (where am I now). Each is reduced on its own terms and fused
    late. The value head exists for PPO and is deliberately NOT exported.
    """

    def __init__(self, action_len: int):
        super().__init__()
        self.action_len = action_len

        # Aggressive early striding: the strip is 64x256 of mostly-empty
        # background and the signal (card glyphs crossing a hit line) is
        # coarse.
        self.vision = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            vision_out = self.vision(torch.zeros(1, 1, MARQUEE_H, MARQUEE_W)).shape[1]

        # LayerNorm FIRST, and it is not decoration: the agent vector mixes
        # quaternion components in [-1, 1] with a raw tick count in the
        # thousands and an unbounded score. Fed straight into a Linear those
        # channels dominate and the body state is lost in them. (The vector
        # layout is frozen platform-side, so it cannot normalize for you.)
        self.proprio = nn.Sequential(
            nn.LayerNorm(AGENT_LEN),
            nn.Linear(AGENT_LEN, 128),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(vision_out + 128, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_len)
        self.value = nn.Linear(256, 1)

        # State-independent log-std, the standard continuous-control choice.
        self.log_std = nn.Parameter(torch.zeros(action_len) - 0.5)

    def features(self, marquee: torch.Tensor, agent: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([self.vision(marquee), self.proprio(agent)], dim=1))

    def forward(self, marquee: torch.Tensor, agent: torch.Tensor) -> torch.Tensor:
        """The EXPORTED path: observation -> bounded action.

        ``tanh`` is what makes the shell's ``[-1, 1]`` assumption true. It
        clamps anyway, but the bound belongs here where training can see it.
        """
        return torch.tanh(self.mu(self.features(marquee, agent)))

    def act(self, marquee: torch.Tensor, agent: torch.Tensor):
        """Sample an action for rollout, with its log-prob and value.

        Sampling happens BEFORE the tanh, and the log-prob is of the
        pre-squash Gaussian sample — mixing squashed and unsquashed spaces
        between collection and update is the usual PPO trap here.
        """
        feats = self.features(marquee, agent)
        mu = self.mu(feats)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        raw = dist.sample()
        log_prob = dist.log_prob(raw).sum(dim=-1)
        return torch.tanh(raw), raw, log_prob, self.value(feats).squeeze(-1)

    def evaluate(self, marquee: torch.Tensor, agent: torch.Tensor, raw: torch.Tensor):
        """Re-score stored pre-squash actions under the CURRENT parameters."""
        feats = self.features(marquee, agent)
        dist = torch.distributions.Normal(self.mu(feats), self.log_std.exp())
        return (
            dist.log_prob(raw).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.value(feats).squeeze(-1),
        )
