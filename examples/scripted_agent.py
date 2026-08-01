"""A NON-trained agent: hand-built policy, no RL, same pipeline.

The platform doesn't care where your ONNX came from — an agent is any graph
with the right signature (see ``train/policy.py``) placed in a bundle next to
the shell. This example builds the dumbest thing that visibly dances: it
ignores the marquee entirely, reads the tick out of the agent vector, and
sways a few joints on a slow sine while holding the rest neutral.

It exports and parity-checks through the exact same code path as the trained
policy, and stages its own bundle (``out/scripted-bundle``) so it can live
next to a trained one::

    task scripted
    task match BUNDLE=out/scripted-bundle

The observation/action types this policy sees are documented at
https://lockstep.games/games/dance-off/interface?mode=servo-assist
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from lockstep_dance_off import ACTION_LEN, MODE_SERVO_ASSIST

from train.export import export, verify
from train.main import OUT_DIR, stage

BUNDLE = OUT_DIR / "scripted-bundle"


class Sway(nn.Module):
    """Ignore the marquee; sway to a beat only the policy can hear.

    The agent vector's LAST element is the current tick (see the layout in
    ``lockstep_dance_off.env``); at 60 ticks/second, ``sin(tick/60 * 3)``
    is a lazy half-hertz-ish sway. Channels: the first 36 action values are
    per-joint rotation vectors, the last 12 are effort shares — even effort,
    a gentle lean on the first two joints, everything else neutral.
    """

    #: `train.export.verify` checks the exported action width against this.
    action_len = ACTION_LEN

    def forward(self, marquee: torch.Tensor, agent: torch.Tensor) -> torch.Tensor:
        tick_seconds = agent[:, -1:] / 60.0
        wave = 0.35 * torch.sin(tick_seconds * 3.0)
        pose = torch.cat([wave.expand(-1, 2), wave.new_zeros(wave.shape[0], 34)], dim=1)
        effort = wave.new_full((wave.shape[0], 12), 0.5)
        return torch.cat([pose, effort], dim=1)


def main() -> None:
    net = Sway()
    onnx = export(net, BUNDLE.parent / "scripted-policy.onnx")
    print(f"→ onnx: {onnx} ({onnx.stat().st_size} bytes)")
    diff = verify(net, onnx)
    print(f"✓ torch/onnxruntime parity: max abs diff {diff:.3e}")
    bundle = stage(MODE_SERVO_ASSIST, onnx, bundle=BUNDLE)
    print(f"→ bundle: {bundle}")
    print("\nRun it:   task match BUNDLE=out/scripted-bundle")


if __name__ == "__main__":
    sys.exit(main())
