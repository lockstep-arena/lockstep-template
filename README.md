# Lockstep agent template

Train an agent for any [Lockstep](https://lockstep.it) game, watch it play
a real local match, and upload it to compete — starting from nothing but this
repo. No access to the platform's source is needed: the engine you train
against is the *same WASM binary* that runs ranked matches.

The template itself is **game-agnostic**: `train/` discovers whichever game
package is installed and builds everything — the network, the ONNX signature,
the bundle — from that game's own metadata and Gymnasium spaces.
**Dance-Off** is the shipped reference game (installed by default, used in
every example below); other games are one `task setup GAME=<slug>` away as
they release.

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
  - [Train the research tier (raw-torque)](#train-the-research-tier-raw-torque)
  - [Iterate on export without retraining](#iterate-on-export-without-retraining)
  - [Upload a new revision of an existing agent](#upload-a-new-revision-of-an-existing-agent)
  - [Pin or change the engine version](#pin-or-change-the-engine-version)
  - [Validate a bundle without uploading](#validate-a-bundle-without-uploading)
- [Adding a game](#adding-a-game)
- [What's in the box](#whats-in-the-box)
- [The wasm story](#the-wasm-story)
- [The contract](#the-contract)
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
prebuilt (the CLI as a binary, the engine and agent shell as WASM inside the
wheels). The optional [pure-Rust agent example](#the-wasm-story) needs a Rust
toolchain with `rustup target add wasm32-wasip2`.

For `task upload` only: an API key. Copy `.env.example` to `.env` and fill in
`LOCKSTEP_API_KEY` (create a key from your account at lockstep.it).

## Quickstart

```sh
task setup            # venv + training stack + the game package (GAME=dance-off)
task train            # short PPO → ONNX → out/agent-bundle (a REAL policy, not a good one)
task match            # a real local match, archived to out/archive.bin
task upload           # compete (needs LOCKSTEP_API_KEY in .env)
```

First time? [TUTORIAL.md](TUTORIAL.md) walks this end to end — including a
hand-written no-training agent — with real captured output.

## Commands

Every task takes `GAME=<slug>` (default `dance-off`, the shipped reference
game). The game package — not this repo — pins the engine release, declares
the modes, and carries the agent shell.

| Command | What it does |
|---|---|
| `task setup` | Creates `.venv`, installs `requirements.txt` (torch + onnx toolchain, no game), then `lockstep-game-<GAME>`. |
| `task engine` | Downloads the game's pinned engine wasm to `out/engine.wasm` (automatic before train/match; no-op if unchanged). `MODE=` picks a non-default mode's engine. |
| `task train` | PPO against the real engine → ONNX export → torch/onnxruntime parity check → stages `out/agent-bundle/`. Vars: `STEPS` (default 8192), `MODE` (default: the game's default mode), `NUM_ENVS` (default: `min(8, cores-2)`), `RESUME=1` (continue from `out/checkpoint.pt`), `OPPONENT=<policy.onnx>` (seat 1 = a frozen policy, on games with an opponent seat), `PARALLEL=1` (both seats learn at once — one shared policy over the game's PettingZoo env, on games that declare one; see [`train/core/self_play.py`](train/core/self_play.py)). |
| `task scripted` | Builds the NON-trained example agent ([`examples/scripted_agent.py`](examples/scripted_agent.py)) → `out/scripted-bundle/`. No RL involved. |
| `task match` | `lockstep match run` with a bundle in both seats (self-play); writes `out/archive.bin`. Var: `BUNDLE` — a bundle dir (`out/agent-bundle`, `out/scripted-bundle`) or a bare `.wasm` component (the Rust agent). |
| `task rust-agent` | Builds [`examples/rust-agent/`](examples/rust-agent/) — an agent authored directly in Rust → wasm component, from public contracts only. Needs `rustup target add wasm32-wasip2`. |
| `task contract` | Refreshes the vendored `dance-off.fbs` from the CDN release the game package pins (`…/<version>/<mode>/contract.fbs` — the contract of the exact engine you train against). The agent WIT is vendored in this repo. |
| `task upload` | `lockstep agent upload` of the bundle. Vars: `NAME` (display name), `AGENT_ID` (upload as a revision instead of creating). |
| `task test` | The template's own tests: the genericity gate (no game imports in `train/`) and the vector-env autoreset semantics proof. |

## Bring your own training stack

The reference PPO in `train/` is deliberately small and yours to gut — the
platform is **not** opinionated about how you train. Two seams make any stack
work:

**The env is standard Gymnasium.** Every game env registers a plain
`gymnasium.make` id and composes with the standard vector API that
stable-baselines3, CleanRL, torchrl and friends build on:

```python
import gymnasium, functools
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import lockstep_dance_off  # any game package; registers its env id

env = AsyncVectorEnv(
    [functools.partial(
        gymnasium.make, "Lockstep/DanceOff-v0",
        engine_source="out/engine.wasm",
     ) for _ in range(8)],
    autoreset_mode=AutoresetMode.SAME_STEP,   # see train/core/train.py for why
    context="spawn",                          # one start method on every OS
)
```

Games that ship a native vector env (dance-off does) also work through
`gymnasium.make_vec("Lockstep/DanceOff-v0", num_envs=8,
engine_source="out/engine.wasm")` — same API, N engines on Rust threads
instead of worker processes, ~35% faster on the reference machine. The
conformance suite holds it step-for-step identical to the process path.

**The export/stage seam is stable.** `train/core/export.py` and
`train/core/stage.py` accept ANY torch module that exposes three attributes —
`input_names` (the obs Dict keys), `input_shapes` (per-input graph shapes) and
`action_len` — and produce a parity-checked, submittable bundle. Train with
whatever you like; if your policy answers the game's derived ONNX signature
(obs keys in, `action` out, bound by NAME), it competes:

```python
from train.core.discovery import resolve_game
from train.core.export import export, verify
from train.core.stage import stage

spec, _ = resolve_game(None)          # the installed game
export(net, "out/policy.onnx")        # net: your policy, your framework
verify(net, "out/policy.onnx")        # torch ↔ onnxruntime parity or raise
stage(spec, spec["default_mode"], Path("out/policy.onnx"), Path("out/agent-bundle"))
```

(`examples/scripted_agent.py` is exactly this path with a hand-written
"policy" — no RL anywhere.)

## The reward landscape (read before a long run)

The reference loop is honest machinery, not a scoring recipe. A real 2M-step
run of this exact PPO on dance-off (64 minutes at ~525 steps/s) ended with
**`mean_return` at exactly 0.00** — random flailing never hits a card, the
score floors at zero, and with an all-zero reward the only surviving gradient
was the entropy bonus, which slowly made the policy *wilder* (more falls,
worse posture) until the log-std clamp and entropy decay now bound that
failure. The trained agent lost to the 70-line scripted one.

That is the starting line, not a bug: sparse-reward control is genuinely
hard, and **raising `STEPS` alone will not produce a scoring dancer**.
The interesting work — the part that is yours — looks like reward shaping
against the match metrics (`info["score"]` is the raw per-step delta; posture
and falls are visible in every archive), curriculum over
`--time-limit-ticks`, bigger networks (`train/core/policy.py` builds from
spaces — widen it freely), imitation from archived matches, or replacing PPO
outright ([bring your own stack](#bring-your-own-training-stack)). The
pipeline's job is to make each attempt cheap: crash-safe checkpoints,
comparable `metrics.csv` across runs, and a parity-checked bundle at the end.

## Recipes

### Train for real

The default `STEPS=8192` proves the loop in minutes and produces a weak
agent. Long runs are where the machinery earns its keep — but read
[the reward landscape](#the-reward-landscape-read-before-a-long-run) first:

```sh
task train STEPS=2000000
```

Progress prints every rollout: episode count, mean return, wall time — and
`out/metrics.csv` gets a row per rollout with the diagnostics that matter.
`out/checkpoint.pt` is rewritten atomically every rollout, so a crash loses
at most one rollout (~2 s of work).

Two things make a long run fast, both visible in the startup line
(`update device: mps   envs: 8 (DanceOffVectorEnv)`):

- **Collection is parallel — natively when the game supports it.** A game
  package that registers a native vector env (N engines on Rust threads in
  this process, GIL released — dance-off does) is picked up automatically;
  otherwise `NUM_ENVS` engine instances step in separate worker processes
  via Gymnasium's standard `AsyncVectorEnv`. The games-side conformance
  suite holds the two paths step-for-step identical, so which one you get
  changes throughput, never semantics. (Measured on an 8-perf-core laptop
  at N=8: ~2,640 env-steps/s native vs ~1,930 process-based.) The default
  (`min(8, cores-2)`) leaves headroom for the learner and the OS;
  `NUM_ENVS=1` runs in-process (breakpoints reach the env; useful for
  debugging).
- **The PPO update pass runs on the best available accelerator** (CUDA, then
  Apple MPS, then CPU); `--device cpu`/`cuda`/`mps` overrides. Rollout
  inference deliberately stays on CPU, where small batches through a net this
  small beat a GPU's per-op dispatch overhead.

Learning rate and entropy coefficient decay linearly to zero over `STEPS`,
so exploration pressure is front-loaded and a long run cannot end up driven
by the entropy bonus alone.

### Resume a crashed or stopped run

```sh
task train STEPS=2000000 RESUME=1
```

`out/checkpoint.pt` (rewritten every rollout) carries the network, the
optimizer state and the step count; `RESUME=1` restores all three and
continues to `STEPS`. The checkpoint is a boring dict —
`{action_len, spaces, state_dict, optimizer_state, collected}` — load it in
your own scripts freely (`train/core/train.py::save_checkpoint` documents
it).

### Watch a run: metrics.csv

One row per rollout, dependency-free, safe to read mid-run:

```
steps, wall_seconds, episodes, mean_return, policy_loss, value_loss,
entropy, log_std_mean, clip_fraction, steps_per_second
```

`log_std_mean` climbing monotonically means exploration noise is inflating
(the burn-in failure this file exists to expose); `clip_fraction` far from
~0.1–0.3 is the standard "learning rate is wrong" signal; `mean_return`
stuck at 0.00 means nothing is being scored and more steps will not help —
see [the reward landscape](#the-reward-landscape-read-before-a-long-run).

### Train the research tier (raw-torque)

Dance-off ships two modes. `servo-assist` (default) asks your policy for
target poses the engine's servo tracks; `raw-torque` gives it direct joint
torques — no servo, much harder, its own ladder:

```sh
task train MODE=raw-torque STEPS=500000
task match
```

A bundle targets exactly one mode — the shells check the action width and
refuse a wrong-mode policy rather than misbehave quietly. (`task engine
MODE=raw-torque` swaps `out/engine.wasm` to that mode's pinned release; the
env asserts the engine's mode matches and fails loudly otherwise.)

### Iterate on export without retraining

`train/main.py` can re-export existing weights, skipping the PPO run:

```sh
.venv/bin/python -m train.main --engine out/engine.wasm --from-weights out/policy.pt
```

### Upload a new revision of an existing agent

The first `task upload` creates the agent and prints its id. Later uploads of
an improved policy should be revisions of that same agent:

```sh
task upload AGENT_ID=<id-from-the-first-upload>
```

### Pin or change the engine version

The engine release is pinned by the **game package**, not this repo — the
wheel's codec and the engine's schema are version-coupled on purpose, so the
pin travels with the code that speaks it. `task engine` fetches whatever the
installed package pins (stamped in `out/engine.wasm.url`; unchanged re-runs
are no-ops). To move to a newer engine release, upgrade the game package:

```sh
.venv/bin/pip install --upgrade lockstep-game-dance-off && task engine
```

### Validate a bundle without uploading

```sh
lockstep agent validate --bundle out/agent-bundle
```

Checks the manifest schema, path safety, and deterministic ZIP build — the
same local checks upload runs first, with no credentials needed.

## Adding a game

The template never changes when a game ships. A game training package
declares one entry point in the `lockstep.training_games` group, resolving to
a zero-arg `game_spec()` that returns the **GameSpec** — the versioned
training-metadata contract:

```toml
[project.entry-points."lockstep.training_games"]
my-game = "lockstep_my_game.training_spec:game_spec"
```

The spec carries `training_contract_version` (this template supports **v1
and v2**), `slug`, `env_id`, `default_mode`, per-mode
`payload_schema_version` + `engine_url`, and an `agent_component_path(mode)`
callable to the mode's prebuilt shell. Loading the entry point registers the
Gymnasium env. Contract v1 also requires: env constructor kwargs `mode` /
`engine_source` / `time_limit_ticks`, a `Dict`-of-named-`Box` observation
space (uint8 images and float32 vectors — `train/core/policy.py` builds one
stream per entry), spawn-safe env factories, and SAME_STEP autoreset
semantics.

Contract **v2** adds one optional key, `parallel_env_id` — for games whose
seats are genuinely adversarial, a `module:callable` locator of a PettingZoo
`ParallelEnv` factory (same constructor kwargs) where every seat learns at
once. `task train PARALLEL=1` uses it; a v1 spec is a valid v2 spec, and a
game without the key simply has no parallel path (the pre-flight says so).

With that in place, `task setup GAME=my-game && task train GAME=my-game` is
the whole story: the network, ONNX signature and bundle are all derived. The
contract is expected to keep widening (new action-space kinds, …); each
widening bumps the version, and the template refuses versions it does not
know with an "upgrade the template" message rather than guessing.

## What's in the box

```
train/                 the reference pipeline (yours to gut and replace)
  core/
    discovery.py       finds installed games via the entry point; contract gate
    policy.py          builds the network FROM the env's spaces (CNN / MLP streams)
    train.py           small self-contained PPO loop; checkpoints + metrics.csv
    self_play.py       the same PPO over a game's PettingZoo env: both seats learning, one policy (PARALLEL=1)
    export.py          torch.onnx export + torch/onnxruntime parity check
    stage.py           writes the submittable bundle from the GameSpec
    engine.py          fetches the game's pinned engine wasm
  main.py              train → export → parity → stage out/agent-bundle/
tests/                 the genericity gate + vec-env semantics proof + an engine-free self-play run (CI runs these)
examples/              dance-off reference-game pedagogy (see TUTORIAL.md)
  scripted_agent.py    the NON-trained agent (part 1 of the tutorial)
  rust-agent/          a pure-Rust wasm agent against the public contract
out/                   build products (gitignored)
  engine.wasm          the REAL engine, from the public CDN (+ .url pin stamp)
  checkpoint.pt        rewritten every rollout; task train RESUME=1 continues
  metrics.csv          one row per rollout
  agent-bundle/        lockstep.toml + component.wasm + artifacts/policy.onnx
  archive.bin          full-frame archive of your local match
```

- **The env** (e.g. `Lockstep/DanceOff-v0`, from the `lockstep-game-dance-off`
  wheel) steps the engine through `lockstep-train`, the platform's
  game-agnostic host — native Box3D physics, host-rasterized observations,
  identical to a ranked match.
- **The bundle** is what the platform consumes: your trained `policy.onnx`
  plus `component.wasm`, the prebuilt agent shell that feeds observations to
  your policy in-match (shipped inside the game wheel).
- Local inference uses whatever your machine has (CoreML / DirectML / CPU);
  timings are indicative only — the watch page's per-tick charts are
  authoritative.

## The wasm story

An agent IS a wasm component implementing the `lockstep:agent` world
(vendored under [`examples/rust-agent/wit/`](examples/rust-agent/wit/)):
opaque bytes in (the View), opaque bytes out (the Input). What those bytes
mean is the game's wire contract — for dance-off, **FlatBuffers**, published
as `contract.fbs` next to every engine release and rendered for the current
one on the game's [Interface tab](https://lockstep.it/games/dance-off/interface)
(vendored here at [`examples/rust-agent/contract/dance-off.fbs`](examples/rust-agent/contract/dance-off.fbs),
refreshed by `task contract`)
— codegen-able for any language that compiles to a wasm component. Two ways
to get a component:

1. **Don't write one** (the training path): the game wheel ships a prebuilt
   shell that feeds `artifacts/policy.onnx` to the host inference capability —
   `task train` / `task scripted` stage it for you.
2. **Write your own** ([`examples/rust-agent/`](examples/rust-agent/)): ~70
   lines of Rust against the vendored WIT + planus-generated wire types,
   `task rust-agent` → a 64 KB component, runnable and uploadable as a bare
   `.wasm`. Any flatc-supported language works the same way.

## The contract

Everything in `train/` is replaceable. Only these are load-bearing for the
platform:

1. **The wire contract** — the game's published schema (dance-off:
   `dance-off.fbs` above): the View you receive and the mode's Input you
   return, versioned per mode by `payload_schema_version`.
2. **The ONNX signature** (training path only) — DERIVED from the env's
   observation space: one input per obs Dict key (bound by NAME), one
   `action` output sized from the action space. For dance-off that is
   `marquee` `f32[1,1,64,256]` + `agent` `f32[1,62]` → `action`
   `f32[1,48]` (servo-assist) / `f32[1,36]` (raw-torque).
   `train/core/export.py`'s parity check enforces it.
3. **The bundle layout** (training path only) — `lockstep.toml` +
   `component.wasm` + `artifacts/policy.onnx`, staged by
   `train/core/stage.py`.

The full observation/action semantics are documented in the game's `.fbs`,
its env module (e.g. `lockstep_dance_off.env`, installed with the wheel), and
its [interface page](https://lockstep.it/games/dance-off/interface?mode=servo-assist).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
