# Lockstep agent template

Train an agent for [Lockstep](https://lockstep.games) **Dance-Off**, watch it
play a real local match, and upload it to compete — starting from nothing but
this repo. No access to the platform's source is needed: the engine you train
against is the *same WASM binary* that runs ranked matches.

## Table of contents

- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Commands](#commands)
- [Recipes](#recipes)
  - [Train for real](#train-for-real)
  - [Train the research tier (raw-torque)](#train-the-research-tier-raw-torque)
  - [Iterate on export without retraining](#iterate-on-export-without-retraining)
  - [Upload a new revision of an existing agent](#upload-a-new-revision-of-an-existing-agent)
  - [Pin or change the engine version](#pin-or-change-the-engine-version)
  - [Validate a bundle without uploading](#validate-a-bundle-without-uploading)
- [What's in the box](#whats-in-the-box)
- [The contract](#the-contract)
- [License](#license)

## Prerequisites

- **Python ≥ 3.11**
- **[Task](https://taskfile.dev)** (`brew install go-task`, or see their
  install page)
- **The `lockstep` CLI** — local match runner + agent upload:

  ```sh
  curl -fsSL https://dl.lockstep.games/install.sh | sh
  ```

  Windows (PowerShell):

  ```powershell
  powershell -ExecutionPolicy Bypass -c "irm https://dl.lockstep.games/install.ps1 | iex"
  ```

- A supported platform: macOS arm64 (11.0+), Linux x86_64/arm64
  (glibc ≥ 2.38 — Ubuntu 24.04, Debian 13, Fedora 39+), or Windows x86_64.

Rust is **not** required — everything compiled arrives prebuilt (the CLI as a
binary, the engine and agent shell as WASM inside the wheels).

For `task upload` only: an API key. Copy `.env.example` to `.env` and fill in
`LOCKSTEP_API_KEY` (create a key from your account at lockstep.games).

## Quickstart

```sh
task setup            # venv + gym env + torch
task train            # short PPO → ONNX → out/agent-bundle (a REAL policy, not a good one)
task match            # a real local match, archived to out/archive.bin
task upload           # compete (needs LOCKSTEP_API_KEY in .env)
```

## Commands

| Command | What it does |
|---|---|
| `task setup` | Creates `.venv` and installs `requirements.txt` (the dance-off Gymnasium env, torch, onnx toolchain). |
| `task engine` | Downloads the pinned engine wasm to `out/engine.wasm` (automatic before train/match; no-op if present). |
| `task train` | PPO against the real engine → ONNX export → torch/onnxruntime parity check → stages `out/agent-bundle/`. Vars: `STEPS` (default 8192), `MODE` (default `servo-assist`). |
| `task match` | `lockstep match run` with your bundle in both seats (self-play); writes `out/archive.bin`. |
| `task upload` | `lockstep agent upload` of the bundle. Vars: `NAME` (display name), `AGENT_ID` (upload as a revision instead of creating). |

## Recipes

### Train for real

The default `STEPS=8192` proves the loop in minutes and produces a weak
dancer. Real training is orders of magnitude longer:

```sh
task train STEPS=2000000
```

Progress prints every rollout (512 steps): episode count, mean return, wall
time. Weights land in `out/policy.pt`, so a crashed run's last export can be
recovered (see the next recipe).

### Train the research tier (raw-torque)

`servo-assist` (default) asks your policy for target poses the engine's servo
tracks; `raw-torque` gives it direct joint torques — no servo, much harder,
its own ladder:

```sh
task train MODE=raw-torque STEPS=500000
task match
```

A bundle targets exactly one tier — the shells check the action width and
refuse a wrong-tier policy rather than misbehave quietly.

### Iterate on export without retraining

`train/main.py` can re-export existing weights, skipping the PPO run:

```sh
.venv/bin/python -m train.main --mode servo-assist \
  --engine out/engine.wasm --from-weights out/policy.pt
```

### Upload a new revision of an existing agent

The first `task upload` creates the agent and prints its id. Later uploads of
an improved policy should be revisions of that same agent:

```sh
task upload AGENT_ID=<id-from-the-first-upload>
```

### Pin or change the engine version

The engine is pinned in `Taskfile.yml` (`ENGINE_URL`) so training and matches
are reproducible. Newer releases appear under
`https://cdn.lockstep.mediabucket.io/games/dance-off/releases/` — bump the
URL, `rm out/engine.wasm`, and re-run. The env asserts the engine's mode
matches the one you're training, so a wrong-mode URL fails loudly.

### Validate a bundle without uploading

```sh
lockstep agent validate --bundle out/agent-bundle
```

Checks the manifest schema, path safety, and deterministic ZIP build — the
same local checks upload runs first, with no credentials needed.

## What's in the box

```
train/           the training pipeline (yours to gut and replace)
  policy.py      marquee CNN + proprioception MLP → tanh action
  train.py       small self-contained PPO loop (no RL framework)
  export.py      torch.onnx export + torch/onnxruntime parity check
  main.py        train → export → parity → stage out/agent-bundle/
out/             build products (gitignored)
  engine.wasm    the REAL dance-off engine, from the public CDN
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
- Local inference uses whatever your machine has (CoreML / DirectML / CPU);
  timings are indicative only — the watch page's per-tick charts are
  authoritative.

## The contract

Everything in `train/` is replaceable. Only two things are load-bearing for
the platform:

1. **The ONNX signature** — inputs `marquee` `f32[1,1,64,256]` and `agent`
   `f32[1,62]`, output `action` `f32[1,ACTION_LEN]` (48 servo-assist / 36
   raw-torque), bound by NAME. `export.py`'s parity check enforces it.
2. **The bundle layout** — `lockstep.toml` + `component.wasm` +
   `artifacts/policy.onnx`, staged by `main.py`.

The full observation/action semantics are documented in
`lockstep_dance_off.env` (installed with the wheel).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
