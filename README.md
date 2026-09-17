# Lockstep agent template

Build an agent for any [Lockstep](https://lockstep.it) environment, watch it
play a real local match, and upload it to compete — from nothing but this
repo. The engine you build against is the *same WASM binary* that runs
ranked matches, and it **documents itself**: goal, reward, what ends an
episode, how you are scored, every observation with its slices or columns
and their units. `task create-agent` turns that self-description into a
ready-to-edit agent project — every field named, every bound stated, typed
access to every value — in Python, Rust or C, at parity. Nothing here is
per-environment.

## Ten lines to a real match

```sh
git clone https://github.com/lockstep-arena/lockstep-template
cd lockstep-template
task doctor                      # what this machine is missing, and the exact fix for each
task quickstart ENV=go1-beacon   # setup → create-agent → build → a real local match
```

Not sure which environment? `task envs` lists every slug the platform
currently publishes, with its modes and each mode's tags (the skill, the
level, the domain) — nothing is named in this repo, so `ENV=` is always
yours to pass.

That ends with `out/archive.bin`: a whole match, scored by the real engine.
Drop it on <https://lockstep.it/replay> to watch it. Then:

```sh
task info ENV=go1-beacon         # what the environment IS — the brief and the wire, in this terminal
# edit agents/my-bot/policy.py — every observation is a named slice or column away
task build AGENT=my-bot          # your hand-written policy, no training
task train AGENT=my-bot          # or: a short PPO run → ONNX → parity check → the same bundle
task train AGENT=my-bot RECIPE=supervised   # or: fit the truth the environment reveals (data environments)
task match AGENT=my-bot          # your agent in every seat, archived to out/archive.bin
task upload AGENT=my-bot         # compete (needs LOCKSTEP_API_KEY in .env)
```

The unit of work is an **agent**: `agents/<name>/`, scaffolded by
`task create-agent NAME=… ENV=…` for one environment + mode + language.
Its `agent.toml` records that identity, every other command reads it —
so `AGENT=` can never pair your policy with the wrong engine, and may be
omitted whenever exactly one agent exists. Engines download themselves
into a keyed cache (`out/cache/<env>/<mode>/`) the first time something
needs one.

Browse the catalog at
[lockstep.it/exhibitions](https://lockstep.it/exhibitions); each
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

Rust and C are **not** required: the default language is Python, and
everything compiled arrives prebuilt (the CLI as a binary, the engine and
the generic agent shell as WASM from the release). The
[hand-written wasm track](#the-wasm-story) is one `task setup LANGS=rust`
(or `LANGS=c`) away.

For `task upload` only: an API key. Copy `.env.example` to `.env` and fill in
`LOCKSTEP_API_KEY` (create a key on the Authorization page at lockstep.it).

## Commands

| Command | What it does |
|---|---|
| `task doctor` | Check the machine: Python, Task, the CLI, the venv, the API key, and — exactly when you have agents in that language — the Rust/C toolchains. Each with its fix. Also what the engine cache holds, per environment and mode: its size and the data blobs or ONNX artifacts fetched beside the engine. |
| `task quickstart ENV=` | Zero decisions: `setup` → `create-agent` (python) → `build` → `match`, then hands you the archive. |
| `task envs` | List the environments you can create an agent for, from the platform — the slugs `ENV=` takes, each mode with its tags. |
| `task setup LANGS=` | Provision toolchains — the ONE place that installs anything. `python` (default): `.venv` + the training stack. `rust`: the wasm target. `c`: wasi-sdk + wit-bindgen (detects an existing install first). |
| `task create-agent NAME= ENV= MODE= LANG=` | Scaffold `agents/<name>/` from the engine's own declaration: a generated interface file (regenerable — every value, slice and column a named constant with its doc and unit, typed accessors, a typed action builder) + a policy stub and, for python, the network `model.py`, both YOURS (never overwritten). |
| `task info ENV= MODE=` | The environment's brief and wire layout — goal, reward, what ends an episode, how you are scored, every value with its slices or columns and units, the tags, the budgets — from the engine you will build against. Fetches the release (engine, shell, any data blobs) into the cache. |
| `task train AGENT= RECIPE= STEPS= NUM_ENVS= RESUME=1 PARALLEL=1 SEEDS= EPOCHS=` | Train → ONNX → parity check → stage `agents/<name>/out/bundle` (python agents). `RECIPE=ppo` (default) learns from the reward; `RECIPE=supervised` fits the truth the environment reveals. The network is the agent's own `model.py` either way. |
| `task build AGENT=` | Build the bundle without training: python exports YOUR `policy.py`; rust/c compile the wasm component. |
| `task match AGENT=` | A real local match through the CLI, every seat your agent, archived to `out/archive.bin`. |
| `task upload AGENT= NAME= AGENT_ID= APPROVE=1` | Upload the agent's bundle to compete. On an assessment the first upload that verifies is your submission, and it is final; `APPROVE=1` skips the confirmation prompt. |
| `task test` | The template's own tests (hermetic + engine-backed + the wire references vs the spec goldens). |

<details>
<summary><strong>Reading an environment: <code>task info</code> and the generated interface</strong></summary>

The engine declares everything an agent needs — and everything a person
needs. `task info` prints, for the mode you are about to build against:

- **The brief** — goal, what earns reward, what ends an episode, in the
  engine's words.
- **How you are scored** — every metric the session reports, with its
  unit and which direction is better, the headline ones first.
- **What you see each tick** — every observation with dtype, shape and
  bounds, then every slice of a vector with its index range, unit and
  meaning — or, for a table-shaped value, every column of its last axis
  with its unit and meaning. A column whose unit is `code` carries its
  code table right there.
- **What you send back** — every action with per-slice bounds, the
  neutral action the world plays when you miss a tick, and the ONNX output
  convention.
- **Tags and budgets** — the mode's assessment tags; control rate,
  wall-clock per tick, missed ticks allowed, memory cap, episode length,
  players.

`task create-agent` writes the same facts into your agent as code —
`interface.py` / `src/interface.rs` / `interface.h`. The three files carry
the same header (the brief, the metrics, the tags, the budgets, the
neutral-action rule) and, per value, the same constants: name, dtype,
shape, element count, bounds, the doc as a comment; per slice and per
column an index constant with its doc and unit (and any code table,
verbatim). Then each language's own typed access:

- Python: `Obs(obs).<value>` — declaration-shaped numpy views, plus
  `Obs(obs).<value>_<slice>` / `<value>_<column>`; `action(torque=…)`
  builds one action in declared units, `normalized(...)` the same action
  in the graph's `[-1, 1]` output convention.
- Rust: a generated `Obs<'a>` with one method per value returning its
  natural type (`&[f32; N]`, `&[[f32; F]; T]`, `&[u8]`, `&[i32]`) and an
  `Action` struct with named fields and `encode()`.
- C: `obs_<value>(&view, out)` typed accessors, `OBS_<VALUE>_ROWS` /
  `_COLS` and `OBS_<VALUE>_COL_<NAME>` defines, an `agent_action_t`
  struct with `agent_action_neutral` / `agent_action_encode`.

The scaffolded policy stub in every language reads one observation
through the typed accessor and answers the neutral action through the
typed builder — so you never transcribe an index range from a web page
again. Re-run `task create-agent` after a release bump to refresh (your
policy files are never touched).

`python -m lockstep_train.info --env <slug>` reads the platform's record of
the release instead (no download, no wasm); `--json` emits it for scripts.
An `UNDOCUMENTED` section at the end means the engine left something
unexplained — ask its maintainers.

</details>

<details>
<summary><strong>Bring your own training stack</strong></summary>

The training loop in `train/` is deliberately small — the interesting part of
this repo is the create → build → compete path, not the optimizer. The env
underneath is plain Gymnasium, provided by `lockstep-train`
(`pip install lockstep-train`):

```python
import gymnasium, lockstep_train.env  # registration side effect

env = gymnasium.make("Lockstep/Env-v0", engine_source="out/cache/<env>/<mode>/engine.wasm")
obs, info = env.reset(seed=0)
```

- Observations are a `Dict` of named `Box`es, actions a `Box` with
  per-element bounds — both derived from the engine's own declaration.
  Your generated `interface.py` names every slice and column of every
  observation, so feature engineering reads
  `obs["body"][iface.OBS_BODY_JOINT_VEL]` or
  `obs["feedback"][:, iface.OBS_FEEDBACK_COL_TRUTH]`, not `obs["body"][7:19]`.
- The network `task train` builds is the agent's own `model.py` —
  `build_policy(observation_space, action_space)`, scaffolded with one
  named stream per observation. Keep it, or ignore it and bring your own.
- Data environments reveal the truth of a decision after the fact;
  `lockstep_train.LabelledStream` joins it back to the observation it
  belongs to (see the supervised recipe below).
- The per-tick `reward` and `done` come from the engine too: training and
  a live match are literally the same computation.
- `gymnasium.make_vec(..., num_envs=8)` gets you the native vector env
  (N engines on Rust threads, GIL released, SAME_STEP autoreset).
- Multi-seat environments: `lockstep_train.env.LockstepParallelEnv` is the
  PettingZoo view — every seat, every tick.

Ship whatever you train as ONNX with the declared signature — one input per
declared observation by name (`u8` images arrive as `f32 ÷ 255`), one output
per declared action by name in `[-1, 1]`, which the shell maps affinely onto
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
`agents/<name>/out/metrics.csv` shows all of it per rollout.

</details>

<details>
<summary><strong>Recipes</strong></summary>

**Train for real**

```sh
task train AGENT=my-bot STEPS=2000000 NUM_ENVS=8
```

Expect hours, not minutes, for a visible policy on most environments. The
checkpoint is written every rollout; a crash costs at most one rollout.

**Resume a crashed or stopped run**

```sh
task train AGENT=my-bot STEPS=2000000 RESUME=1
```

**Watch a run: metrics.csv**

```sh
column -s, -t agents/my-bot/out/metrics.csv | less -S
```

One row per rollout: mean return, losses, entropy, log-std, clip fraction,
steps/second.

**An agent for another mode**

Some environments publish several modes — separate ladders with separate
engines; `task envs` shows each environment's modes and `task info
ENV=<slug>` describes them. A mode is part of an agent's identity, fixed at
create time:

```sh
task create-agent NAME=torquey ENV=<slug> MODE=<mode>
task train AGENT=torquey
task match AGENT=torquey        # always that mode's engine — agent.toml says so
```

Two agents on two modes coexist happily: the engine cache is keyed by
(environment, mode), so neither ever clobbers the other's engine.

**Two seats learning at once**

On a multi-seat environment, `task train PARALLEL=1` trains BOTH seats with
one shared policy over the generic PettingZoo view — the classic first rung
of self-play. The exported artifact is unchanged.

**Change the network**

`agents/<name>/model.py` is yours. `task create-agent` writes it with one
named stream per declared observation — the starter for every value,
whatever its dtype or rank, is flatten → LayerNorm → Linear → ReLU — and
`train/core/policy.py` adds the shared trunk and heads. Replace one line
to give an image a convolution (a commented example is generated for an
image-shaped `u8` value) or a history window a recurrent layer; the ONNX
signature is unchanged because the graph's inputs are still the declared
values at their declared shapes.

**Learn from the truth the environment reveals (supervised)**

Data environments score a decision after the fact: what was true about
the transaction you saw at tick `t` arrives later — as a row of a
*feedback window* (an observation whose rows are the decisions that
resolved this tick, with columns saying how old each one is, what you
decided and what was true) or as the reward `reward_lag_ticks` later.
That makes them supervised-learning problems as much as control
problems, and the template treats them so:

```sh
task train AGENT=my-bot RECIPE=supervised SEEDS=16 EPOCHS=20
```

It runs the engine over `SEEDS` public seeds playing the neutral action,
joins every revealed truth to the observation it belongs to
(`lockstep_train.LabelledStream`), fits a classifier head — or a
regression head when the truth is not a small set of codes — on the same
`model.py` trunk PPO uses, and exports with the declared signature: the
prediction lands on the declared action as its code, everything else
neutral. `agents/<name>/out/supervised.csv` has the per-epoch train and
held-out loss and accuracy.

Where the truth is found is never guessed. `task create-agent` writes a
`[supervised]` block into `agent.toml` when the declaration reveals it — a
value whose columns carry the documented names (`age_ticks`,
`your_decision`, `truth`, and `valid` for padding) or a
`reward_lag_ticks` in the metadata:

```toml
[supervised]
value = "feedback"
age_col = "age_ticks"
decision_col = "your_decision"
truth_col = "truth"
valid_col = "valid"
action = "decision"
```

Edit it if the environment names things differently (the names are the
declaration's own — `task info` shows them), or pass the source on the
command line: `.venv/bin/python -m train.main --agent my-bot --recipe
supervised --label-value <obs>` or `--reward-lag <ticks>`. An agent whose
declaration reveals no truth has no block, and the recipe says so
instead of training on nothing.

**Iterate on export without retraining**

```sh
.venv/bin/python -m train.main --agent my-bot --from-weights agents/my-bot/out/policy.pt
```

**Upload a new revision of an existing agent**

```sh
task upload AGENT=my-bot AGENT_ID=<id-from-the-first-upload>
```

**Refresh after a release bump**

```sh
task create-agent NAME=my-bot ENV=<slug>       # regenerates interface.py + agent.toml (policy.py, model.py untouched)
task build AGENT=my-bot                        # rebuild against the new release
```

The platform API names the current release (its directory on the CDN is
unguessable, so the path is never derived from the slug). For an
assessment-only environment the API answers only invited candidates — put
your key in `.env` as `LOCKSTEP_API_KEY` (the same one `task upload` uses).
The staged bundle declares the payload schema version read from the engine
itself, so the api will refuse a stale bundle rather than let it misread
observations.

**Validate a bundle without uploading**

```sh
lockstep agent validate --bundle agents/my-bot/out/bundle
```

**Taking a hiring assessment**

Some companies use Lockstep environments as hands-on hiring assessments. If
you received an invite link, the flow is this exact repo: practice locally
as much as you like (`task build` / `task train` / `task match` on random
scenarios, on the same engine that scores you), then `task upload` your
best agent. **Upload is submit**: the first upload that verifies is your
submission and it is final — the platform runs that bundle once over the
role's frozen scenario suite, the score is what it is, and later uploads
are refused. Run `task match` until you are happy before you upload.

What the employer receives, so there are no surprises:

- **The score and every replay**, scenario by scenario, next to two
  reference points: our scripted reference controller and this template's
  stock trainer run on the same scenarios.
- **The per-run metrics** (energy, smoothness, drops, time to reach, failure
  reason) and your **approach notes** from the invite page.
- **Your source**, if you let `task upload` send it (the default — it ships
  the text files under `agents/<name>/`, skipping `out/` and build
  directories; `SOURCE=0` uploads the bundle alone). The employer reads it
  next to your replays; it never reaches other candidates or the match server.
- **An interview guide** written by the environment's authors, keyed to the
  failure modes and metrics on your report. Expect to be asked about your
  weakest scenario.

Two invite options an employer may have switched on: **hand-written policy
only** (no ONNX artifact — `task create-agent LANG=rust` or `LANG=c`; the
Python path always ships an ONNX policy and will be refused at upload), and a
**follow-up round** — a second, 48-hour invite on fresh scenarios with one
stated change, which you take with the same workflow.

AI-assisted work is permitted. Describe what you generated and what you
fixed in your approach notes: the interviewer will ask, and the source and
the replays make the answer checkable.

</details>

<details>
<summary><strong>Any environment, one variable</strong></summary>

There is nothing to add to this repo when a new environment releases:

```sh
task info ENV=panda-pick                       # read it
task create-agent NAME=picky ENV=panda-pick    # scaffold from ITS declaration
task train AGENT=picky                         # the network is built from the same declaration
```

`train/core/discovery.py` resolves the release through the platform API, the
artifacts come from the CDN, and the engine's own declaration does the rest.

</details>

<details>
<summary><strong>What's in the box</strong></summary>

```
Taskfile.yml            the whole surface (doctor/quickstart/setup/envs/create-agent/info/train/build/match/upload/test)
train/
  doctor.py             `task doctor` — prerequisites, each with its fix
  scaffold.py           `task create-agent` — agent projects from the engine's declaration
  agents.py             agents/<name>/agent.toml — identity + AGENT= resolution
  build.py              `task build` — python export / rust cargo / c wasi-sdk → bundle
  toolchain.py          `task setup LANGS=` — detect-first toolchain provisioning
  main.py               `task train` — train → export → parity-check → stage
  core/discovery.py     `task envs` + release resolution, both through the platform API
  core/engine.py        the keyed engine cache (out/cache/<env>/<mode>/ — engine, shell, data blobs, ONNX artifacts)
  core/policy.py        the trunk and heads around the agent's streams; obs → tensor plumbing; the ONNX contract
  core/train.py         a small real PPO loop (vectorized, SAME_STEP)
  core/self_play.py     shared-policy self-play over the PettingZoo view
  core/supervised.py    the supervised recipe: collect the revealed truth, fit a head, export
  core/export.py        ONNX export + torch/onnxruntime parity proof
  core/stage.py         the submittable bundle (manifest v2)
reference/
  rust-wire/            the hand-written Rust wire reader the rust scaffold vendors, pinned by the spec goldens
  c-wire/               its C99 twin (wire.h/wire.c), pinned by the same goldens
wit/                    the agent WIT world every wasm agent targets (vendored)
agents/                 YOUR agents (created by task create-agent: interface + policy + model.py; build products gitignored)
tests/                  hermetic tests + engine-backed semantics proofs + scaffold compile tests
```

</details>

<details>
<summary><strong>The wasm story</strong></summary>

Your agent ships as a WASM **component** — either the generic ONNX shell
plus your `policy.onnx` (the python paths above), or a component you wrote
yourself. That second kind is the advanced track, and it is one command per
language:

```sh
task setup LANGS=rust                                  # once
task create-agent NAME=ferrous ENV=<slug> LANG=rust
task build AGENT=ferrous && task match AGENT=ferrous
```

```sh
task setup LANGS=c                                     # once — finds /opt/wasi-sdk or $WASI_SDK, else downloads
task create-agent NAME=clanger ENV=<slug> LANG=c
task build AGENT=clanger && task match AGENT=clanger
```

The scaffold is a complete project: the vendored WIT world, a hand-written
wire reader (`wire.rs` / `wire.c` — ~530 dependency-free lines,
re-implemented from the published spec and pinned by its goldens under
`reference/`), a generated `interface.rs` / `interface.h` naming every
declared value, slice and column with typed accessors and a typed action
builder, and a stub that reads one observation through them and answers
the neutral action until you edit it. No ONNX, no Python at match time —
`on-tick` in, action out.

Determinism is the platform's core bet: the engine is bit-identical across
machines, so a local match IS a ranked match with different seats. The
archive your match writes (`out/archive.bin`) drops straight onto
[lockstep.it/replay](https://lockstep.it/replay).

</details>

<details>
<summary><strong>The Lockstep wire</strong></summary>

Every engine describes itself at seat-init: named observations and actions
with dtype, shape, bounds (per-element where it matters), documented
slices with units (or documented columns, for a table-shaped value),
free-form metadata, the seat's brief — goal, reward, what ends an episode
— and the metrics the session will report, each with a unit and a
direction. Every tick after that is positional, near-raw blobs the
declaration explains. Little-endian, no codegen — the spec fits on a page
([docs/wire.md](docs/wire.md) — copied verbatim from the platform's
interface repo at every release), and `reference/rust-wire` +
`reference/c-wire` re-implement it from scratch to prove the point. The
generic ONNX shell, the Python env, `task info`, the environment's
Interface page, your generated interface files and your hand-written agent
all read the same declaration.

</details>

## License

Apache-2.0 OR MIT, at your option.
