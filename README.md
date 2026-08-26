# Lockstep agent template

Build an agent for any [Lockstep](https://lockstep.it) environment, watch it
play a real local match, and upload it to compete — from nothing but this
repo. The engine you train against is the *same WASM binary* that runs
ranked matches, and it **documents itself**: goal, reward, what ends an
episode, every tensor and slice with units. Nothing here is per-environment.

## Ten lines to a real match

```sh
git clone https://github.com/lockstep-arena/lockstep-template
cd lockstep-template
task doctor                      # what this machine is missing, and the exact fix for each
task quickstart ENV=go1-beacon   # setup → engine → a scripted agent → a real local match
```

That ends with `out/archive.bin`: a whole match, scored by the real engine.
Drop it on <https://lockstep.it/replay> to watch it. Then:

```sh
task info  ENV=go1-beacon        # what the environment IS — the brief and the wire, in this terminal
task train ENV=go1-beacon        # a short PPO run → ONNX → parity check → out/agent-bundle
task match ENV=go1-beacon        # your policy in every seat, archived to out/archive.bin
task upload ENV=go1-beacon       # compete (needs LOCKSTEP_API_KEY in .env)
```

Every command takes `ENV=<slug>` (default `dance-off`) and, on multi-mode
environments, `MODE=<key>`. Browse the catalog at
[lockstep.it/competitions](https://lockstep.it/competitions); each
environment's **Interface** tab is `task info` rendered as a page, and its
**Practice** tab is this README, per environment.

Want the long version with real output? [TUTORIAL.md](TUTORIAL.md).

## Prerequisites

`task doctor` checks all of these and prints the fix for anything missing.

- **Python ≥ 3.11**
- **[Task](https://taskfile.dev)** — the task runner every command here is a target of
- **The `lockstep` CLI** — the local match runner + agent uploader:

  ```sh
  curl -fsSL https://dl.lockstep.it/install.sh | sh
  ```

  Windows (PowerShell):

  ```powershell
  powershell -ExecutionPolicy Bypass -c "irm https://dl.lockstep.it/install.ps1 | iex"
  ```

  Supported: macOS arm64 (11.0+), Linux x86_64/arm64 (glibc ≥ 2.38 —
  Ubuntu 24.04, Debian 13, Fedora 39+), Windows x86_64. CI proves the whole
  pipeline on all three on every change.

Rust is **not** required for the training path — everything compiled arrives
prebuilt (the CLI as a binary, the engine and the generic agent shell as WASM
from the release). The optional [pure-Rust agent](#the-wasm-story) needs
`rustup target add wasm32-wasip2`.

For `task upload` only: an API key. Copy `.env.example` to `.env` and fill in
`LOCKSTEP_API_KEY` (create a key from your account settings at lockstep.it).

## Commands

| Command | What it does |
|---|---|
| `task doctor` | Check the machine: Python, Task, the CLI, the venv, the API key, the Rust toolchain — each with its fix. |
| `task quickstart ENV=` | Zero decisions: `setup` → `engine` → `scripted` → `match`, then hands you the archive. |
| `task setup` | Create `.venv`, install the training stack + `lockstep-train`. One-time; nothing per-environment. |
| `task info ENV= MODE=` | The environment's brief and wire layout, from the engine you will train against. |
| `task engine ENV= MODE= VERSION=` | Download the release's `engine.wasm` + generic `agent-onnx.wasm` into `out/` (no-op when unchanged). |
| `task train ENV= MODE= STEPS= NUM_ENVS= RESUME=1 PARALLEL=1` | PPO → ONNX → parity check → stage `out/agent-bundle`. |
| `task scripted ENV=` | The no-training example agent → `out/scripted-bundle`. |
| `task rust-agent` | The pure-Rust example agent → a bare wasm component. |
| `task match ENV= BUNDLE=` | A real local match through the CLI, both seats your bundle, archived to `out/archive.bin`. |
| `task upload ENV= NAME= AGENT_ID=` | Upload the bundle to compete (or seal it for an assessment). |
| `task test` | The template's own tests (hermetic + engine-backed). |

<details>
<summary><strong>Reading an environment: <code>task info</code></strong></summary>

The engine declares everything an agent needs — and everything a person
needs. `task info` prints, for the mode you are about to train against:

- **The brief** — goal, what earns reward, what ends an episode, in the
  engine's words.
- **What you see each tick** — every observation tensor with dtype, shape
  and bounds, then every slice with its index range, unit and meaning.
- **What you send back** — every action tensor with per-slice bounds, the
  neutral action the world plays when you miss a tick, and the ONNX output
  convention.
- **Budgets** — control rate, wall-clock per tick, missed ticks allowed,
  memory cap, episode length, players.

`python -m lockstep_train.info --env <slug>` reads the platform's record of
the release instead (no download, no wasm); `--json` emits it for scripts.
An `UNDOCUMENTED` section at the end means the engine left something
unexplained — ask its maintainers.

</details>

<details>
<summary><strong>Bring your own training stack</strong></summary>

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

Ship whatever you train as ONNX with the declared signature — one input per
observation tensor by name (`u8` images arrive as `f32 ÷ 255`), one output
per action tensor by name in `[-1, 1]`, which the shell maps affinely onto
the declared bounds — and stage it with `python -m train.main --from-weights`
or your own copy of `train/core/stage.py`. The platform does not care what
trained the graph.

</details>

<details>
<summary><strong>The reward landscape (read before a long run)</strong></summary>

The default `task train` run is SHORT — it proves the pipeline, not the
policy. Before spending hours of compute, run `task info` and read the
brief's REWARD paragraph: several environments pay a dense shaping signal
(progress, posture) on top of sparse task rewards, but some landscapes
still sit near zero for a long time. Two structural guards in this loop
exist because of that: the entropy bonus and learning rate decay to zero
over the run, and the policy log-std is clamped — otherwise a zero-gradient
landscape slowly inflates exploration until the policy is noise.
`out/metrics.csv` shows all of it per rollout.

</details>

<details>
<summary><strong>Recipes</strong></summary>

**Train for real**

```sh
task train STEPS=2000000 NUM_ENVS=8
```

Expect hours, not minutes, for a visible policy on most environments. The
checkpoint is written every rollout; a crash costs at most one rollout.

**Resume a crashed or stopped run**

```sh
task train STEPS=2000000 RESUME=1
```

**Watch a run: metrics.csv**

```sh
column -s, -t out/metrics.csv | less -S
```

One row per rollout: mean return, losses, entropy, log-std, clip fraction,
steps/second.

**Train another mode**

Some environments publish several modes — separate ladders with separate
engines (Dance-Off's `servo-assist` vs `raw-torque`, say). The engine you
download IS the mode:

```sh
task info  ENV=dance-off MODE=raw-torque
task train ENV=dance-off MODE=raw-torque
task match ENV=dance-off        # the bundle's lockstep.toml picks the mode
```

**Two seats learning at once**

On a multi-seat environment, `task train PARALLEL=1` trains BOTH seats with
one shared policy over the generic PettingZoo view — the classic first rung
of self-play. The exported artifact is unchanged.

**Iterate on export without retraining**

```sh
.venv/bin/python -m train.main --env dance-off --from-weights out/policy.pt --engine out/engine.wasm
```

**Upload a new revision of an existing agent**

```sh
task upload AGENT_ID=<id-from-the-first-upload>
```

**Pin or change the release version**

```sh
task engine ENV=dance-off VERSION=0.8.0
```

The platform API names the current release (its directory on the CDN is
unguessable, so the path is never derived from the slug); `VERSION=` asserts
that release's version rather than selecting an older one. For an
assessment-only environment the API answers only invited candidates — put
your key in `.env` as `LOCKSTEP_API_KEY` (the same one `task upload` uses)
before `task engine`. The staged bundle declares the payload schema version
read from the engine itself, so the api will refuse a stale bundle rather
than let it misread observations.

**Validate a bundle without uploading**

```sh
lockstep agent validate --bundle out/agent-bundle
```

**Taking a hiring assessment**

Some companies use Lockstep environments as hands-on hiring assessments. If
you received an invite link, the flow is this exact repo: practice locally
(`task train` / `task match`), `task upload` your best agent, then **seal**
it from your invite page. The invite page lists your verified agents — the
upload you just made is what you seal. Practice runs and the final
evaluation both run the same engine you trained against here.

</details>

<details>
<summary><strong>Any environment, one variable</strong></summary>

There is nothing to add to this repo when a new environment releases:

```sh
task info   ENV=panda-pick    # read it
task engine ENV=panda-pick    # resolve + fetch its current release
task train  ENV=panda-pick    # the network is built from ITS declaration
```

`train/core/discovery.py` resolves the release through the platform API
(`LOCKSTEP_API_URL`; artifacts then come from the CDN, `LOCKSTEP_CDN_URL`);
the engine's tensor-wire declaration does the rest.

</details>

<details>
<summary><strong>What's in the box</strong></summary>

```
Taskfile.yml            the whole surface (doctor/quickstart/setup/info/engine/train/match/upload/test)
train/
  doctor.py             `task doctor` — prerequisites, each with its fix
  main.py               train → export → parity-check → stage
  core/discovery.py     release resolution through the platform API
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

</details>

<details>
<summary><strong>The wasm story</strong></summary>

Your agent ships as a WASM **component** — either the generic ONNX shell
plus your `policy.onnx` (the trained path above), or a component you wrote
yourself. `examples/rust-agent/` is the second kind: ~200 lines of Rust
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

</details>

<details>
<summary><strong>The tensor wire</strong></summary>

Every engine describes itself at seat-init: named observation/action
tensors with dtype, shape, bounds (per-element where it matters),
documented slices with units, free-form metadata, and the seat's brief —
goal, reward, what ends an episode. Little-endian, no codegen — the spec
fits on a page ([docs/wire.md](docs/wire.md) — vendored here from the
platform's private interface repo at every release), and `examples/rust-agent/src/wire.rs` re-implements it from scratch
to prove the point. The generic ONNX shell, the Python env, `task info`,
the environment's Interface page and your hand-written agent all read the
same declaration.

</details>

## License

Apache-2.0 OR MIT, at your option.
