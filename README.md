# Lockstep agent template

Train an agent for any [Lockstep](https://lockstep.it) environment, watch it
play a real local match, and upload it to compete — starting from nothing but
this repo. No access to the platform's source is needed: the engine you train
against is the *same WASM binary* that runs ranked matches.

The template itself is **environment-agnostic**, and so is everything it
installs: there are no per-environment packages at all. An environment's
published release carries its engine and the generic agent shell, and the
engine **self-describes** its whole interface (named observation/action
tensors, dtypes, shapes, bounds — the [tensor wire](#the-tensor-wire)).
`train/` builds everything — the network, the ONNX signature, the bundle —
from that declaration. **Dance-Off** is the default `ENV=` used in the
examples below; every other environment is the same commands with a different
slug. Browse them at [lockstep.it/arenas](https://lockstep.it/arenas).

## Table of contents

- [Tutorial](TUTORIAL.md) — the worked example: a non-trained agent and a
  trained one, end to end with real output
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Commands](#commands)
- [Bring your own training stack](#bring-your-own-training-stack)
- [The reward landscape (read before a long run)](#the-reward-landscape-read-before-a-long-run)
- [Recipes](#recipes)
  - [Train for real](#train-for-real)
  - [Resume a crashed or stopped run](#resume-a-crashed-or-stopped-run)
  - [Watch a run: metrics.csv](#watch-a-run-metricscsv)
  - [Train another mode](#train-another-mode)
  - [Iterate on export without retraining](#iterate-on-export-without-retraining)
  - [Upload a new revision of an existing agent](#upload-a-new-revision-of-an-existing-agent)
  - [Pin or change the release version](#pin-or-change-the-release-version)
  - [Validate a bundle without uploading](#validate-a-bundle-without-uploading)
  - [Taking a hiring assessment](#taking-a-hiring-assessment)
- [Any environment, one variable](#any-environment-one-variable)
- [What's in the box](#whats-in-the-box)
- [The wasm story](#the-wasm-story)
- [The tensor wire](#the-tensor-wire)
- [License](#license)

## Prerequisites

- **Python ≥ 3.11**
- **[Task](https://taskfile.dev)** (`brew install go-task`, or see their
  install page)
- **The `lockstep` CLI** — local match runner + agent upload:

  ```sh
  curl -fsSL https://dl.lockstep.it/install.sh | sh
  ```

  Windows (PowerShell):

  ```powershell
  powershell -ExecutionPolicy Bypass -c "irm https://dl.lockstep.it/install.ps1 | iex"
  ```

- A supported platform: macOS arm64 (11.0+), Linux x86_64/arm64
  (glibc ≥ 2.38 — Ubuntu 24.04, Debian 13, Fedora 39+), or Windows x86_64.
  CI proves the whole pipeline on all three on every change.

Rust is **not** required for the training path — everything compiled arrives
prebuilt (the CLI as a binary, the engine and the generic agent shell as WASM
from the release). The optional [pure-Rust agent example](#the-wasm-story)
needs a Rust toolchain with `rustup target add wasm32-wasip2`.

For `task upload` only: an API key. Copy `.env.example` to `.env` and fill in
`LOCKSTEP_API_KEY` (create a key from your account at lockstep.it).

## Quickstart

```sh
task setup            # venv + the training stack (nothing per-environment)
task train            # short PPO → ONNX → out/agent-bundle (a REAL policy, not a good one)
task match            # a real local match, archived to out/archive.bin
task upload           # compete (needs LOCKSTEP_API_KEY in .env)
```

Every command takes `ENV=<slug>` (default `dance-off`):

```sh
task train ENV=panda-pick
task match ENV=panda-pick
```

First time? [TUTORIAL.md](TUTORIAL.md) walks this end to end — including a
hand-written no-training agent — with real captured output.

## Commands

| Command | What it does |
|---|---|
| `task setup` | Create `.venv`, install the training stack + `lockstep-train`. One-time; nothing per-environment. |
| `task engine ENV= MODE= VERSION=` | Download the release's `engine.wasm` + generic `agent-onnx.wasm` into `out/` (no-op when unchanged). |
| `task train ENV= STEPS= NUM_ENVS= RESUME=1 PARALLEL=1` | PPO → ONNX → parity check → stage `out/agent-bundle`. |
| `task scripted ENV=` | The no-training example agent → `out/scripted-bundle`. |
| `task rust-agent` | The pure-Rust example agent → a bare wasm component. |
| `task match ENV= BUNDLE=` | A real local match through the CLI, both seats your bundle, archived to `out/archive.bin`. |
| `task upload ENV= NAME= AGENT_ID=` | Upload the bundle to compete (or seal it for an assessment). |
| `task test` | The template's own tests (hermetic + engine-backed). |

## Bring your own training stack

The training loop in `train/` is deliberately small — the interesting part of
this repo is the train → export → bundle → compete path, not the optimizer.
The env underneath is plain Gymnasium, provided by `lockstep-train`
(`pip install lockstep-train`):

```python
import gymnasium, lockstep_train.env  # registration side effect

env = gymnasium.make("Lockstep/Env-v0", engine_source="out/engine.wasm")
obs, info = env.reset(seed=0)
```

- Observations are a `Dict` of named `Box`es, actions a `Box` with
  per-element bounds — both derived from the engine's own declaration.
- The per-tick `reward` and `done` come from the engine too: training and
  competition are literally the same computation.
- `gymnasium.make_vec(..., num_envs=8)` gets you the native vector env
  (N engines on Rust threads, GIL released, SAME_STEP autoreset).
- Multi-seat environments: `lockstep_train.env.LockstepParallelEnv` is the
  PettingZoo view — every seat, every tick.

Ship whatever you train as ONNX with the derived signature (one input per
observation tensor, by name; one `action` output in [-1, 1]) and stage it
with `python -m train.main --from-weights` or your own copy of
`train/core/stage.py`. The platform does not care what trained the graph.

## The reward landscape (read before a long run)

The default `task train` run is SHORT — it proves the pipeline, not the
policy. Before spending hours of compute, read the environment's page
(`lockstep.it/environments/<slug>`) and skim its reward: several
environments pay a dense shaping signal (progress, posture) on top of sparse
task rewards, but some landscapes still sit near zero for a long time. Two
structural guards in this loop exist because of that: the entropy bonus and
learning rate decay to zero over the run, and the policy log-std is clamped —
otherwise a zero-gradient landscape slowly inflates exploration until the
policy is noise. `out/metrics.csv` shows all of it per rollout.

## Recipes

### Train for real

```sh
task train STEPS=2000000 NUM_ENVS=8
```

Expect hours, not minutes, for a visible policy on most environments. The
checkpoint is written every rollout; a crash costs at most one rollout.

### Resume a crashed or stopped run

```sh
task train STEPS=2000000 RESUME=1
```

### Watch a run: metrics.csv

```sh
column -s, -t out/metrics.csv | less -S
```

One row per rollout: mean return, losses, entropy, log-std, clip fraction,
steps/second.

### Train another mode

Some environments publish several modes — separate ladders with separate
engines (Dance-Off's `servo-assist` vs `raw-torque`, say). The engine you
download IS the mode:

```sh
task train ENV=dance-off MODE=raw-torque
task match ENV=dance-off        # the bundle's lockstep.toml picks the mode
```

### Iterate on export without retraining

```sh
.venv/bin/python -m train.main --env dance-off --from-weights out/policy.pt --engine out/engine.wasm
```

### Upload a new revision of an existing agent

```sh
task upload AGENT_ID=<id-from-the-first-upload>
```

### Pin or change the release version

```sh
task engine ENV=dance-off VERSION=0.6.0
```

`latest.json` on the CDN names the current release; `VERSION=` pins an
older one (its engine, its shell). The staged bundle declares the payload
schema version read from the engine itself, so the api will refuse a stale
bundle rather than let it misread observations.

### Validate a bundle without uploading

```sh
lockstep agent validate --bundle out/agent-bundle
```

### Taking a hiring assessment

Some companies use Lockstep environments as hands-on hiring assessments. If
you received an invite link, the flow is this exact repo: practice locally
(`task train` / `task match`), `task upload` your best agent, then **seal**
it from your invite page. The invite page lists your verified agents — the
upload you just made is what you seal. Practice runs and the final
evaluation both run the same engine you trained against here.

## Any environment, one variable

There is nothing to add to this repo when a new environment releases:

```sh
task engine ENV=go1-beacon    # resolve + fetch its current release
task train  ENV=go1-beacon    # the network is built from ITS declaration
```

`train/core/discovery.py` resolves `environments/<slug>/latest.json` on the
CDN (base overridable via `LOCKSTEP_CDN_URL`); the engine's tensor-wire
declaration does the rest. The catalog lives at
[lockstep.it/arenas](https://lockstep.it/arenas); each environment's
Interface page renders the same declaration this template trains from.

## What's in the box

```
Taskfile.yml            the whole surface (setup/engine/train/match/upload/test)
train/
  main.py               train → export → parity-check → stage
  core/discovery.py     CDN release resolution (latest.json)
  core/engine.py        engine + generic-shell download, .url stamps
  core/policy.py        the network, derived from the declared spaces
  core/train.py         a small real PPO loop (vectorized, SAME_STEP)
  core/self_play.py     shared-policy self-play over the PettingZoo view
  core/export.py        ONNX export + torch/onnxruntime parity proof
  core/stage.py         the submittable bundle (manifest v2)
examples/
  scripted_agent.py     a no-training agent through the same pipeline
  rust-agent/           a pure-Rust agent: hand-written wire decoder, no deps
tests/                  hermetic tests + engine-backed semantics proofs
```

## The wasm story

Your agent ships as a WASM **component** — either the generic ONNX shell
plus your `policy.onnx` (the trained path above), or a component you wrote
yourself. `examples/rust-agent/` is the second kind: ~150 lines of Rust
that decode the tensor wire BY HAND (no generated code, no private
dependencies — the wire is a published spec) and answer every tick. It
plays a visible sway on Dance-Off and the correct neutral on any other
environment, and the same `task match` runs it:

```sh
task rust-agent
task match BUNDLE=examples/rust-agent/target/wasm32-wasip2/release/rust_agent.wasm
```

Determinism is the platform's core bet: the engine is bit-identical across
machines, so a local match IS a ranked match with different seats. The
archive your match writes (`out/archive.bin`) drops straight onto
[lockstep.it/replay](https://lockstep.it/replay).

## The tensor wire

Every engine describes itself at seat-init: named observation/action
tensors with dtype, shape, bounds (per-element where it matters), slice
names for documentation, and free-form metadata. Little-endian, no codegen
— the spec fits on a page (`docs/wire.md` in the public
[lockstep-interface](https://github.com/lockstep-arena/lockstep-interface)
repo), and `examples/rust-agent/src/wire.rs` re-implements it from scratch
to prove the point. The generic ONNX shell, the Python env and your
hand-written agent all read the same declaration; the environment's
Interface page (`lockstep.it/environments/<slug>/interface`) renders it.

## License

Apache-2.0 OR MIT, at your option.
