# Lockstep agent template

Train an agent for [Lockstep](https://lockstep.games) **Dance-Off**, watch it
play a real local match, and upload it to compete — starting from nothing but
this repo. No access to the platform's source is needed: the engine you train
against is the *same WASM binary* that runs ranked matches.

## Prerequisites

- Python ≥ 3.11
- [Task](https://taskfile.dev) (`brew install go-task` / see their install page)
- macOS (Apple Silicon), Linux x86_64/arm64 (glibc ≥ 2.38 — Ubuntu 24.04,
  Debian 13, Fedora 39+), or Windows x86_64

Rust is **not** required — everything compiled arrives prebuilt.

## Quickstart

```sh
# 1. The lockstep CLI (local match runner + agent upload)
curl -fsSL https://dl.lockstep.games/install.sh | sh
#    Windows (PowerShell):
#    powershell -ExecutionPolicy Bypass -c "irm https://dl.lockstep.games/install.ps1 | iex"

# 2. Python environment (venv + gym env + torch)
task setup

# 3. Train a short PPO run and package it as an agent bundle.
#    The default STEPS produces a REAL policy, not a good one — the point is
#    proving the loop. Raise STEPS by orders of magnitude to actually train.
task train

# 4. Watch it: a real local match (self-play), archived to out/archive.bin
task match

# 5. Compete: upload the agent (needs LOCKSTEP_API_KEY in .env)
task upload
```

## What just happened

```
train/           PPO → ONNX, adapted from the dance-off training pipeline
  policy.py      marquee CNN + proprioception MLP → tanh action
  train.py       small self-contained PPO loop (no RL framework)
  export.py      torch.onnx export + torch/onnxruntime parity check
  main.py        train → export → parity → stage out/agent-bundle/
out/
  engine.wasm    the REAL dance-off engine, downloaded from the public CDN
  agent-bundle/  lockstep.toml + component.wasm + artifacts/policy.onnx
  archive.bin    full-frame archive of your local match
```

- **The env** (`Lockstep/DanceOff-v0`, from the `lockstep-game-dance-off`
  wheel) steps the engine through `lockstep-train`, the platform's
  game-agnostic host — native Box3D physics, host-rasterized observations,
  identical to a ranked match.
- **The bundle** is what the platform consumes: your trained `policy.onnx`
  plus `component.wasm`, the prebuilt agent shell that feeds observations to
  your policy in-match (shipped inside the wheel —
  `lockstep_dance_off.components`).
- **`task match`** runs `lockstep match run` with that bundle in both seats.
  Local inference uses whatever your machine has (CoreML / DirectML / CPU);
  timings are indicative only — the watch page's per-tick charts are
  authoritative.

## Control tiers

Dance-Off has two ladders (`MODE`): `servo-assist` (default — the action is a
target pose, the engine's servo tracks it) and `raw-torque` (research tier —
your policy outputs raw joint torque). `task train MODE=raw-torque` switches;
a bundle targets exactly one tier.

## Uploading

`task upload` needs an API key: copy `.env.example` to `.env` and fill in
`LOCKSTEP_API_KEY` (create a key from your account at lockstep.games). The
upload creates the agent, waits for the platform's verifier, and prints the
agent id. Subsequent uploads of an improved policy: `task upload
AGENT_ID=<id>` to add a revision instead of creating a new agent.

## Tuning

`train/` is yours — that's the point of the template. `train.py` is a
deliberately small, readable PPO; the observation/action contract it trains
against is documented in `lockstep_dance_off.env`. Only two things are
load-bearing for the platform: the ONNX signature (`marquee`/`agent` →
`action`, checked by `export.py`) and the bundle layout (`main.py` stages it).
