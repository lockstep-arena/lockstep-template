# Tutorial: from a clean machine to the ladder

This walks the whole path with real captured output: what the environment
is, an agent scaffolded from the engine's own declaration, a hand-written
policy through the full pipeline, then a trained one. go1-beacon (a
quadruped walking to a beacon) is the worked example because its brief is
short; every other slug is the same commands.

```sh
git clone https://github.com/lockstep-arena/lockstep-template
cd lockstep-template
task doctor
```

```
lockstep template doctor
running checks… (the venv check imports the training stack; give it a moment)

  ✓ Python            3.14.6 at /opt/homebrew/opt/python@3.14/bin/python3.14
  ✓ Task              3.53.1 at /opt/homebrew/bin/task
  ✓ lockstep CLI      lockstep 0.1.4 at ~/.cargo/bin/lockstep
  ✗ venv              no .venv/ yet
    fix → task setup   (creates .venv and installs the training stack)
  ✓ LOCKSTEP_API_KEY  set (only `task upload` needs it)
  · Rust toolchain    not installed — only needed for LANG=rust agents
    fix → task setup LANGS=rust   (or https://rustup.rs then: rustup target add wasm32-wasip2)
  · C toolchain       wasi-sdk + wit-bindgen missing — only needed for LANG=c agents
    fix → task setup LANGS=c   (detects /opt/wasi-sdk or $WASI_SDK, else downloads wasi-sdk into out/toolchains/)
  · engine cache      no cached engines yet — every task that needs one (info / train / build / match) fetches it on demand

1 required item(s) missing — fix the ✗ lines above, then run `task doctor` again.
```

Every ✗ comes with its fix. The · lines are optional and say what they
unlock — the Rust/C toolchains become REQUIRED (✗) exactly when an agent in
that language exists under `agents/`.

## Contents

- [Zero decisions: `task quickstart`](#zero-decisions-task-quickstart)
- [Part 0 — read the environment](#part-0--read-the-environment)
- [Part 1 — create an agent](#part-1--create-an-agent)
- [Part 2 — a hand-written policy (no training)](#part-2--a-hand-written-policy-no-training)
- [Part 3 — a trained agent](#part-3--a-trained-agent)
- [Data environments: the supervised recipe](#data-environments-the-supervised-recipe)
- [Part 4 — reading a match](#part-4--reading-a-match)
- [Part 5 — compete](#part-5--compete)
- [The advanced track: Rust and C](#the-advanced-track-rust-and-c)
- [Two seats learning at once](#two-seats-learning-at-once)
- [Where to go from here](#where-to-go-from-here)

## Zero decisions: `task quickstart`

```sh
task quickstart ENV=go1-beacon
```

runs `setup` → `create-agent` → `build` → `match` back to back and ends with:

```
quickstart: done — out/archive.bin is a real go1-beacon match with agents/my-bot in every seat.
Watch it:   drop out/archive.bin on https://lockstep.it/replay
Read it:    task info ENV=go1-beacon    (the brief and the wire, in this terminal)
Make it yours: edit agents/my-bot/policy.py, then task build AGENT=my-bot
Train it:      task train AGENT=my-bot  then task match, then task upload
```

The rest of this tutorial is those same steps, one at a time, with what
each one prints.

## Part 0 — read the environment

The engine describes itself — nothing below was typed into the template:

```sh
task setup
task info ENV=go1-beacon
```

```
go1-beacon · default  (out/cache/go1-beacon/default/engine.wasm)
================================================================
  v0.2.0 · payload schema 2 · seat 0

THE BRIEF
  GOAL
    You drive a Unitree Go1 quadruped on flat ground. A beacon sits 2–4 m away
    at any heading; walk to it and stay within 0.35 m of it for half a second
    without falling. Harder seeds shove the trunk sideways one to three times
    mid-episode.
  REWARD
    The episode score is 0–100: progress toward the beacon (40, best fraction
    of the starting distance closed), reaching it (30), time to spare after
    reaching (15), staying upright (15, mean uprightness), minus energy (up to
    −10, mean actuator power) and jerk (up to −5, mean change in your
    targets). Each tick's reward is that tick's change in score plus
    potential-based shaping on distance to the beacon, so rewards sum to the
    score.
  ENDS
    The episode ends the moment the robot falls (trunk below 0.15 m or
    roll/pitch past 1 rad — counted as a failure), when it has held the beacon
    for 0.5 s, or after 20 s (1000 ticks at 50 Hz).

WHAT YOU SEE EACH TICK
  obs  f32[54]  unbounded
      Everything the policy sees this tick: trunk pose and rates, the 12
      joints, the previous targets, where the beacon is from the trunk, and
      how much time is left. Directions marked trunk frame are rotated into
      the robot's own frame so a yawed robot sees the same world.
          0–3  trunk_quat    [4] · unit quaternion wxyz, world frame
                            trunk orientation
          4–6  trunk_angvel  [3] · rad/s, trunk frame
                            trunk angular velocity
          7–9  trunk_linvel  [3] · m/s, trunk frame
                            trunk linear velocity
        10–21  joint_pos     [12] · rad
                            joint angles, leg order FR FL RR RL × (hip
                            abduction, thigh, calf)
        22–33  joint_vel     [12] · rad/s
                            joint angular velocities, same order as joint_pos
        34–45  last_action   [12] · rad
                            the targets you sent last tick, so a policy can
                            penalize its own jerk
        46–48  beacon_body   [3] · m, trunk frame
                            beacon position minus trunk position, rotated into
                            the trunk frame
           49  beacon_dist   [1] · m
                            straight-line horizontal distance to the beacon
           50  time_left     [1]
                            fraction of the 20 s episode remaining, 1 at the
                            start, 0 at the end
        51–53  gravity_body  [3] · unit vector, trunk frame
                            world down (0,0,-1) rotated into the trunk frame —
                            the tilt an IMU reads

WHAT YOU SEND BACK
  action  f32[12]  -4.501 .. 4.501
      Joint position targets for the 12 actuators; the model's position
      controller tracks them. Per-element bounds are each joint's ctrlrange.
      Sending HOME_POSE (hip 0, thigh 0.9, calf -1.8 on every leg) stands
      still.
          0–2  fr  [3] · rad · -0.863..0.863, -0.686..4.501, -2.818..-0.888
                  front-right leg: hip abduction, thigh, calf targets
          3–5  fl  [3] · rad · -0.863..0.863, -0.686..4.501, -2.818..-0.888
                  front-left leg: hip abduction, thigh, calf targets
          6–8  rr  [3] · rad · -0.863..0.863, -0.686..4.501, -2.818..-0.888
                  rear-right leg: hip abduction, thigh, calf targets
         9–11  rl  [3] · rad · -0.863..0.863, -0.686..4.501, -2.818..-0.888
                  rear-left leg: hip abduction, thigh, calf targets
  Rules the world applies to your input:
    Out-of-range values are clamped into bounds. A missing, malformed or non-
    finite input plays the NEUTRAL action — the midpoint of each element's
    bounds, or 0 when a bound is open — and counts as a missed tick.
    neutral: action → per-element bounds midpoint
    ONNX policies: the shell reads each action output in [-1, 1] and maps it
    affinely onto the bounds above (per element where they differ; an open
    bound passes through unscaled) — a tanh head is the right shape.

BUDGETS
  control rate   50 Hz — one decision every 20.0 ms of sim time
  time per tick  20 ms of wall clock for your policy to answer
  missed ticks   50 allowed before the seat forfeits
  episode        1 tick (under a second) to 1,000 ticks (20 s)
  players        1
```

That is the whole contract: 54 floats in, 12 joint targets out, and the
reward you will train on spelled out by the engine that computes it. An
environment with table-shaped values — a history window, a feedback
window of resolved decisions — prints each column of the last axis the
same way, with its unit and meaning (and a column whose unit is `code`
carries its code table right there); a `HOW YOU ARE SCORED` block names
every metric the report will show, and a `TAGS` line the mode's
assessment tags. The
same declaration renders as the environment's
[Interface tab](https://lockstep.it/exhibitions/environment/go1-beacon/interface);
`python -m lockstep_train.info --env go1-beacon` prints it without
downloading anything. The first `task info` also fetched the release into
the keyed cache (`out/cache/go1-beacon/default/`) — nothing else will ever
download it again until the release changes.

Multi-mode environments take `MODE=` — separate ladders with separate
engines; `task envs` lists each environment's modes and `task info
ENV=<slug>` describes them. The cache keys on (environment, mode), so
nothing ever clobbers anything.

## Part 1 — create an agent

```sh
task create-agent NAME=walker ENV=go1-beacon
```

```
→ agents/walker/interface.py  (generated)
→ agents/walker/policy.py  (yours — edit it)
→ agents/walker/model.py  (yours — edit it)
→ agents/walker/agent.toml  (generated)

create-agent: walker ready — go1-beacon [default] in python.
next: edit agents/walker/  then  task train AGENT=walker
```

Everything `task info` printed is now CODE in your agent —
`agents/walker/interface.py` opens with the brief, the metrics, the
budgets and the neutral-action rule as its docstring, then names every
value, slice and column:

```python
# ── obs ── f32[54] — 54 element(s), bounds [-inf, inf] every element
# Everything the policy sees this tick: trunk pose and rates, the 12
# joints, the previous targets, where the beacon is from the trunk, and
# how much time is left. …
OBS_OBS = "obs"
OBS_OBS_INDEX = 0
OBS_OBS_SHAPE = (54,)
# trunk orientation [unit quaternion wxyz, world frame]
OBS_OBS_TRUNK_QUAT = slice(0, 4)
# joint angles, leg order FR FL RR RL × (hip abduction, thigh, calf) [rad]
OBS_OBS_JOINT_POS = slice(10, 22)
# beacon position minus trunk position, rotated into the trunk frame [m, trunk frame]
OBS_OBS_BEACON_BODY = slice(46, 49)
…
ACT_ACTION_SHAPE = (12,)
# front-right leg: hip abduction, thigh, calf targets [rad] — per-element bounds [-2.818..4.501], varying
ACT_ACTION_FR = slice(0, 3)
```

A table-shaped value gets `OBS_<VALUE>_ROWS`, `OBS_<VALUE>_COLS` and one
`OBS_<VALUE>_COL_<NAME>` constant per column, each with its doc and unit.
Below the constants come the typed accessors: `Obs(obs).obs_joint_pos`
is the joint-angle slice as a numpy array, `Obs(obs).<value>_<column>`
a column of every row, and `action(fr=[0, 0.9, -1.8], ...)` builds one
action in declared units (`normalized(...)` the same action in the
graph's `[-1, 1]` output convention). The scaffolded `policy.py` reads
one observation that way and answers the neutral action.

The third file, `model.py`, is the network `task train` builds: one
named stream per declared observation, with its shape in a comment, and
the starter stream (flatten → LayerNorm → Linear → ReLU) for every value
whatever its dtype or rank. Replace one line to change a stream — an
image-shaped `u8` value comes with a commented-out convolution.

You never transcribe an index range from a web page again — and after a
release bump, re-running `task create-agent NAME=walker ENV=go1-beacon`
refreshes `interface.py` and `agent.toml` while **never touching**
`policy.py` or `model.py`. `agent.toml` is the agent's identity —
environment, mode, language, the release it was generated from — and
every later command reads it, which is why none of them need `ENV=`.

## Part 2 — a hand-written policy (no training)

`agents/walker/policy.py` is YOURS. As scaffolded it plays the neutral
action (every output 0 → each action element's bounds midpoint), and its
docstring shows how to read any slice by name. Build and run it as-is
first:

```sh
task build AGENT=walker
```

```
→ onnx: agents/walker/out/policy.onnx (573 bytes)
✓ torch/onnxruntime parity: max abs diff 0.000e+00
→ bundle: agents/walker/out/bundle

Run it:   task match AGENT=walker
Compete:  task upload AGENT=walker
```

(torch's ONNX exporter prints a few opset warnings above these lines; they
are noise.) `task build` exported your `policy.py` to ONNX, proved
torch/onnxruntime parity, and staged the submittable bundle — no training
anywhere.

```sh
task match AGENT=walker
```

```
match finished: 13 frames, final tick 12, rankings [0], winner None — archive out/archive.bin (6437 bytes)
        metrics: [ score 4.87, success 0, reached 0, max-tilt 0.196, energy 349, pushes 2, … ]
✓ match archived → out/archive.bin
Watch it: drop out/archive.bin on https://lockstep.it/replay
```

(Condensed — the CLI prints every metric as a struct.) Thirteen frames:
the neutral action is each joint's range **midpoint**, which is not a
standing pose, so the Go1 folds and the brief's ENDS rule fires at tick 12
— "the moment the robot falls". Score 4.87 out of 100, all of it stability
points earned before the fall. That is the real engine scoring a real
policy; it is just a bad one.

Now make it yours: the action doc in Part 0 says the pose that stands is
hip 0, thigh 0.9, calf −1.8. Edit `forward` in `agents/walker/policy.py`
to emit that pose (remember the shell maps `[-1, 1]` onto each joint's
bounds — `interface.py` has them), `task build AGENT=walker` again, and
watch the match run to the 1000-tick time-out instead.

## Part 3 — a trained agent

```sh
task train AGENT=walker STEPS=64 NUM_ENVS=1     # tiny — proves the pipeline in ~a minute
```

```
── network: agents/walker/model.py (build_policy)
── training go1-beacon [default] for 64 steps
  update device: mps   envs: 1 (SyncVectorEnv)
      128/64 steps  episodes=3    mean_return=   -0.86     0.4s
→ weights: agents/walker/out/policy.pt
→ onnx: agents/walker/out/policy.onnx (174546 bytes)
✓ torch/onnxruntime parity: max abs diff 7.451e-08
→ bundle: agents/walker/out/bundle

Run it:   task match
Compete:  task upload
```

The network is the one in `agents/walker/model.py`, scaffolded from the
declaration in Part 0: a flatten → LayerNorm → Linear stream for `obs`,
then the template's trunk and a 12-wide tanh head — and that tanh head is
not a style choice: the ONNX shell reads outputs in `[-1, 1]` and maps
each one onto its joint's declared range. The parity line is the load-bearing one:
the exported graph and the trained network are held to the same numbers
under onnxruntime — the exact runtime the platform's inference host uses.
The trained bundle lands in the same slot `task build` used —
`agents/walker/out/bundle` — so `match` and `upload` don't care which path
produced it.

For a policy that actually walks, `STEPS=64` becomes millions and
`NUM_ENVS=8`; the brief's REWARD paragraph tells you what the gradient will
look like before you spend the compute. See the README's
[reward landscape](README.md#the-reward-landscape-read-before-a-long-run)
note first.

## Data environments: the supervised recipe

Some environments are not control problems at all: you see a
transaction, a ticket, a sensor window, and decide; what was actually
true arrives later — as a row of a *feedback window* (an observation
whose rows are the decisions that resolved this tick, with columns saying
how old each one is, what you decided and what was true) or as the reward
`reward_lag_ticks` later. `task info` shows both: the window's columns
with their units, or a `REWARD LAG` line.

When the declaration reveals its truth that way, `task create-agent`
writes where to find it into `agent.toml`:

```toml
[supervised]
value = "feedback"
age_col = "age_ticks"
decision_col = "your_decision"
truth_col = "truth"
valid_col = "valid"
action = "decision"
```

and the second recipe reads it:

```sh
task train AGENT=triager RECIPE=supervised SEEDS=16 EPOCHS=20
```

```
── network: agents/triager/model.py (build_policy)
── supervised training <slug> [<mode>] over 16 seeds, 20 epochs
  label source: rows of `feedback` — columns age_ticks / your_decision / truth (rows with valid = 0 are padding)
  collected 3812 labelled observations from 16 seeds (4800 ticks, 0 labels with no observation to join) in 3.1s
  classification over declared codes {0: 3488, 1: 324}
  epoch   1/20  train_loss=0.6421  val_loss=0.5133  val_accuracy=0.812     3.9s
  …
→ weights: agents/triager/out/policy.pt
→ onnx: agents/triager/out/policy.onnx (160022 bytes)
✓ torch/onnxruntime parity: max abs diff 7.451e-08
→ bundle: agents/triager/out/bundle
```

The engine ran over 16 public seeds playing the neutral action; every
revealed truth was joined back to the observation it belongs to
(`lockstep_train.LabelledStream`, the same join you can use in your own
loop); a classifier head was fit on the `model.py` trunk with the rare
class weighted up; and the export keeps the declared signature — the
prediction lands on the declared action as its code, everything else
neutral. `agents/triager/out/supervised.csv` has the per-epoch numbers.
The bundle is the same shape as a PPO one, so `task match` and `task
upload` do not care which recipe made it.

Where the truth lives is never guessed: the block above is generated from
the declaration's own column names. If an environment names them
differently, edit the block (`task info` shows the names), or say it on
the command line — `.venv/bin/python -m train.main --agent triager
--recipe supervised --label-value <obs>` or `--reward-lag <ticks>`. An
agent whose declaration reveals no truth gets no block, and the recipe
refuses with that reason instead of training on nothing.

## Part 4 — reading a match

The archive is the whole match, deterministically replayable. Drop
`out/archive.bin` on <https://lockstep.it/replay> to watch it rendered
(its **View results** button shows the graded view before playback even
starts), or read the same thing in the terminal — `task match` prints it
at the end, and `task report` prints it again for any archive
(`ARCHIVE=<path>`): pass/fail, score, the headline metrics, then every
metric with its label and unit — exactly the row this match would be on
an assessment report. Matches end when the engine says so — the brief's
ENDS paragraph — never on a wall clock.

## Part 5 — compete

```sh
task upload AGENT=walker
```

```
uploaded bundle agents/…/8c7b581b54e499ae….wasm (1199629 bytes, sha256 7e1491b8…)
        mode: "create",
        verification: "verified",
        display_name: "walker",
        mode_key: "default",
        latest_revision_number: 1,
        latest_revision_status: Verified,
```

(Condensed — the CLI prints the full upload record. `verification:
"verified"` is the line that matters: the platform re-ran your bundle
against the live engine and accepted it, payload schema and all.)

Your agent joins the arena pool and shows up on the environment's
**Your agents** tab. Re-upload as a revision with
`task upload AGENT=walker AGENT_ID=<id>`.

If you are here for a **hiring assessment**: the upload that names it —
`task upload AGENT=walker ASSESSMENT=<id>`, the exact command is on the
assessment's Ship step — is your submission. It is final and later uploads
to it are refused, so `task match` until you are happy first. An upload
without `ASSESSMENT=` is a plain upload and never a submission. By default it also
sends the text files under `agents/<name>/` as an employer-visible source
archive — see the README's
[assessment recipe](README.md#recipes) for exactly what the employer
receives (replays, metrics, notes, source, an interview guide) and the two
options an invite may carry (hand-written-only, a follow-up round).

## The advanced track: Rust and C

The python paths above never leave ONNX. The other two languages ship a
component you wrote yourself — no ONNX, no Python at match time, and the
whole thing still starts from the engine's declaration:

```sh
task setup LANGS=rust                                    # once
task create-agent NAME=ferrous ENV=go1-beacon LANG=rust
task build AGENT=ferrous
task match AGENT=ferrous
```

The scaffold is a complete cargo project: the vendored WIT world, the
hand-written wire reader (`src/wire.rs`, ~530 dependency-free lines pinned
by the spec goldens under `reference/rust-wire/`), a generated
`src/interface.rs`, and `src/lib.rs` — yours — answering the neutral
action until you edit `on_tick`. The interface carries the same header
and constants as the Python one (every slice a `Range<usize>` with its
doc and unit, every column a `COL_<NAME>` index) and then typed access in
Rust's own terms: `Obs::read(&view)` decodes the view into a struct with
one method per value returning its natural type — `&[f32; 54]` here, a
`&[[f32; 8]; 16]` for a history window, `&[u8]` for an image — and
`Action { fr: [0.0, 0.9, -1.8], ..Action::neutral() }.encode()` builds
the input by name:

```rust
let Some(obs) = Obs::read(&view) else { return Vec::new() };
// `obs[trunk_quat]` — trunk orientation
let _first = &obs.obs()[interface::obs::obs::TRUNK_QUAT];
Action::neutral().encode()
```

`task build` compiles it to a `wasm32-wasip2` component and bundles it;
the same `task match`/`task upload` run it.

C is the same shape:

```sh
task setup LANGS=c        # finds /opt/wasi-sdk or $WASI_SDK, else downloads the pinned wasi-sdk
task create-agent NAME=clanger ENV=go1-beacon LANG=c
task build AGENT=clanger
task match AGENT=clanger
```

`agents/clanger/` holds `wire.h`/`wire.c` (the C99 twin of the Rust
reader, pinned by the same goldens under `reference/c-wire/`), a generated
`interface.h`, `agent.c` — yours — and a one-clang-line `build.sh`:
wit-bindgen generates the world bindings, wasi-sdk's clang links the
component directly. The header is the same declaration again as
`#define`s (`OBS_OBS_TRUNK_QUAT_START` / `_LEN`, `OBS_<VALUE>_ROWS` /
`_COLS` / `_COL_<NAME>`, the shape and bounds arrays) plus typed access:
`obs_obs(&view, out)` copies the value into a declaration-shaped array
(`float out[16][8]` for a window; a `u8` value is borrowed as bytes), and
`agent_action_t` with `agent_action_neutral` / `agent_action_encode`
builds the input by named field:

```c
float obs[54];
if (obs_obs(&view, obs)) {
    float first = obs[OBS_OBS_TRUNK_QUAT_START];   /* trunk orientation */
}
agent_action_t action;
agent_action_neutral(&action);
agent_action_encode(&action, &ret->ptr, &ret->len);
```

Bytes in, bytes out.

## Two seats learning at once

On a duel environment — two seats, one winner (`task info ENV=<slug>`
says how many seats a mode has) — plain `task train` trains seat 0 while
seat 1 plays neutral. `PARALLEL=1` trains BOTH seats with one shared policy
over the generic PettingZoo view — the classic first rung of self-play:

```sh
task create-agent NAME=duelist ENV=<slug>
task train AGENT=duelist PARALLEL=1 STEPS=128
```

```
── self-play training <slug> [<mode>] for 128 seat-steps (both seats learning, one shared policy)
→ onnx: agents/duelist/out/policy.onnx (1279116 bytes)
✓ torch/onnxruntime parity: max abs diff 7.451e-08
→ bundle: agents/duelist/out/bundle
```

Every seat's transition lands in the same PPO buffer; the exported
artifact is unchanged (one `policy.onnx`), so `task match` and `task
upload` work identically.

## Where to go from here

- Swap the training loop for your own stack — the env is plain
  `gymnasium.make("Lockstep/Env-v0", engine_source=...)`, your
  generated `interface.py` names every feature, and `model.py` is yours
  to rewrite; the README's [BYO section](README.md#commands) is the seam.
- On a data environment, start from `RECIPE=supervised` and compare it
  with PPO on the same seeds — `task match` scores both with the same
  engine.
- Try another environment: `task info ENV=panda-pick`, then
  `task create-agent NAME=picky ENV=panda-pick` — same commands, a MuJoCo
  arm, its own brief.
- Read the wire spec ([docs/wire.md](docs/wire.md)) and port the reader to
  your favorite language — `reference/rust-wire/` and `reference/c-wire/`
  are the worked examples, pinned by the spec's own goldens.
