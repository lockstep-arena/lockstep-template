# Tutorial: from a clean machine to the ladder

This walks the whole path with real captured output: what the environment
is, a **non-trained** agent through the full pipeline, then a **trained**
one. go1-beacon (a quadruped walking to a beacon) is the worked example
because its brief is short; every other slug is the same commands.

```sh
git clone https://github.com/lockstep-arena/lockstep-template
cd lockstep-template
task doctor
```

```
lockstep template doctor

  ✓ Python            3.14.6 at /opt/homebrew/opt/python@3.14/bin/python3.14
  ✓ Task              3.53.1 at /opt/homebrew/bin/task
  ✓ lockstep CLI      lockstep 0.1.4 at ~/.cargo/bin/lockstep
  ✗ venv              no .venv/ yet
    fix → task setup   (creates .venv and installs the training stack)
  ✓ LOCKSTEP_API_KEY  set (only `task upload` needs it)
  ✓ Rust toolchain    cargo 1.95.0 (f2d3ce0bd 2026-03-21), wasm32-wasip2 installed
  · engine            no out/engine.wasm yet — fetched on demand by task info / train / match
    fix → task engine ENV=<slug>

1 required item(s) missing — fix the ✗ lines above, then run `task doctor` again.
```

Every ✗ comes with its fix. The · lines are optional and say what they
unlock.

## Contents

- [Zero decisions: `task quickstart`](#zero-decisions-task-quickstart)
- [Part 0 — read the environment](#part-0--read-the-environment)
- [Part 1 — a non-trained agent](#part-1--a-non-trained-agent)
- [Part 1½ — the same idea in pure Rust](#part-1½--the-same-idea-in-pure-rust)
- [Part 2 — a trained agent](#part-2--a-trained-agent)
- [Part 3 — reading a match](#part-3--reading-a-match)
- [Part 4 — compete](#part-4--compete)
- [Two seats learning at once](#two-seats-learning-at-once)
- [Where to go from here](#where-to-go-from-here)

## Zero decisions: `task quickstart`

```sh
task quickstart ENV=go1-beacon
```

runs `setup` → `engine` → `scripted` → `match` back to back and ends with:

```
quickstart: done — out/archive.bin is a real go1-beacon match with the scripted agent in every seat.
Watch it:   drop out/archive.bin on https://lockstep.it/replay
Read it:    task info ENV=go1-beacon   (the brief and the wire, in this terminal)
Train one:  task train ENV=go1-beacon  then task match ENV=go1-beacon, then task upload ENV=go1-beacon
```

The rest of this tutorial is those same steps, one at a time, with what
each one prints.

## Part 0 — read the environment

The engine describes itself — nothing below was typed into the template:

```sh
task setup ENV=go1-beacon
task info  ENV=go1-beacon
```

```
go1-beacon · default  (out/engine.wasm)
=======================================
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

METADATA
  model           unitree_go1
  task            walk to the beacon and stay near it without falling
  mode            default
  control rate    50 Hz
  episode length  1,000 ticks
```

That is the whole contract: 54 floats in, 12 joint targets out, and the
reward you will train on spelled out by the engine that computes it. The
same declaration renders as the environment's
[Interface tab](https://lockstep.it/competitions/environment/go1-beacon/interface);
`python -m lockstep_train.info --env go1-beacon` prints it without
downloading anything.

Multi-mode environments take `MODE=`:

```sh
task info ENV=dance-off MODE=servo-assist
```

```
dance-off · servo-assist  (out/engine.wasm)
===========================================
  v0.8.0 · payload schema 8 · seat 0

THE BRIEF
  GOAL
    You control a 13-body ragdoll dancer facing an opponent on the same stage.
    A routine of 52 cards scrolls across the marquee; each card names a target
    pose (or a freestyle bar) and the beat it lands on. Match the pose as the
    card crosses the hit line and hold it through the beat.
  …

WHAT YOU SEE EACH TICK
  marquee  u8[1,64,256]  0 .. 255
      The scrolling card strip exactly as the audience sees it: one grayscale
      plane, 64 rows × 256 columns, row-major, y-down, 0 = black. …
      ONNX input: marquee f32[1,1,64,256] = pixels ÷ 255 (the shell normalizes
      u8 images; a policy trained on raw 0..255 reads noise).
  agent  f32[62]  unbounded
      …
  cues  i32[4,3]  -1 .. inf
      The next 4 cards as rows of (move_id, modifier, ticks_to_hit), nearest
      first. …
```

An image, a vector and a table — three shapes, one declaration.

## Part 1 — a non-trained agent

`examples/scripted_agent.py` builds the simplest correct ONNX graph — every
action element at its bounds midpoint (the wire's own "neutral") — and
stages it exactly like a trained one. The point: the platform does not care
where your graph came from.

```sh
task scripted ENV=go1-beacon
```

```
→ out/engine.wasm  (…/environments/go1-beacon/releases/0.2.0-…/default/engine.wasm)
→ out/agent-onnx.wasm  (…/environments/go1-beacon/releases/0.2.0-…/default/agent-onnx.wasm)
→ onnx: out/scripted-policy.onnx (573 bytes)
✓ torch/onnxruntime parity: max abs diff 0.000e+00
→ bundle: out/scripted-bundle

Run it:   task match ENV=go1-beacon BUNDLE=out/scripted-bundle
```

(torch's ONNX exporter prints a few opset warnings above these lines; they
are noise.)

```sh
task match ENV=go1-beacon BUNDLE=out/scripted-bundle
```

```
✓ out/engine.wasm up to date
✓ out/agent-onnx.wasm up to date
match finished: 13 frames, final tick 12, rankings [0], winner None — archive out/archive.bin (6437 bytes)
MatchRun(
    MatchRunOutput {
        slug: "go1-beacon",
        seats: 1,
        mode: "default",
        frames: 13,
        final_tick: 12,
        metrics: [ score 4.87, success 0, reached 0, max-tilt 0.196, energy 349, pushes 2, … ]
    },
)
✓ match archived → out/archive.bin
Watch it: drop out/archive.bin on https://lockstep.it/replay
```

(Condensed — the CLI prints every metric as a struct.) Thirteen frames:
the neutral action is each joint's range **midpoint**, which is not a
standing pose, so the Go1 folds and the brief's ENDS rule fires at tick 12
— "the moment the robot falls". Score 4.87 out of 100, all of it stability
points earned before the fall. That is the real engine scoring a real
policy; it is just a bad one. Open `examples/scripted_agent.py` and make it
yours — the docstring shows where a closed-form policy plugs in, `task
info` tells you which indices of `obs` to read, and its action doc tells
you the pose that stands (hip 0, thigh 0.9, calf −1.8).

## Part 1½ — the same idea in pure Rust

`examples/rust-agent/` is the same no-training bot with the training stack
deleted: ~200 lines that decode the tensor wire **by hand** — the brief and
every slice's doc included, so an agent can read its own instructions — and
answer every tick directly. The decoder is pinned by the spec's own golden
encodings (`cargo test` in that directory). On Dance-Off it recognizes the servo
declaration and sways; on any other environment it plays neutral.

```sh
task rust-agent
task match ENV=go1-beacon BUNDLE=examples/rust-agent/target/wasm32-wasip2/release/rust_agent.wasm
```

No ONNX, no inference host, no dependencies beyond the WIT world — bytes
in, bytes out.

## Part 2 — a trained agent

```sh
task train ENV=go1-beacon STEPS=64 NUM_ENVS=1     # tiny — proves the pipeline in ~a minute
```

```
✓ out/engine.wasm up to date
✓ out/agent-onnx.wasm up to date
── training go1-beacon [default] for 64 steps
  update device: mps   envs: 1 (SyncVectorEnv)
      128/64 steps  episodes=3    mean_return=    1.05     1.1s
→ weights: out/policy.pt
→ onnx: out/policy.onnx (174546 bytes)
✓ torch/onnxruntime parity: max abs diff 7.451e-08
→ bundle: out/agent-bundle

Run it:   task match
Compete:  task upload
```

The network was built from the declaration in Part 0: a LayerNorm+MLP
stream for `obs` into a 12-wide tanh head — and that tanh head is not a
style choice: the ONNX shell reads outputs in `[-1, 1]` and maps each one
onto its joint's declared range. The parity line is the load-bearing one:
the exported graph and the trained network are held to the same numbers
under onnxruntime — the exact runtime the platform's inference host uses.

For a policy that actually walks, `STEPS=64` becomes millions and
`NUM_ENVS=8`; the brief's REWARD paragraph tells you what the gradient will
look like before you spend the compute. See the README's
[reward landscape](README.md#the-reward-landscape-read-before-a-long-run)
note first.

```sh
task match ENV=go1-beacon
```

```
match finished: 10 frames, final tick 9, rankings [0], winner None — archive out/archive.bin (5042 bytes)
        metrics: [ score 3.33, success 0, reached 0, max-tilt 0.294, energy 569, pushes 2, … ]
✓ match archived → out/archive.bin
```

Sixty-four steps of PPO fell over faster than standing still did. That is
the honest starting line: the pipeline is proven, the policy is not.

## Part 3 — reading a match

The archive is the whole match, deterministically replayable. Drop
`out/archive.bin` on <https://lockstep.it/replay> to watch it rendered, or
read the outcome straight off the CLI output above: final tick, rankings
by seat, winner. Matches end when the engine says so — the brief's ENDS
paragraph — never on a wall clock.

## Part 4 — compete

```sh
task upload ENV=go1-beacon NAME=my-first-agent
```

```
uploaded bundle agents/…/8c7b581b54e499ae….wasm (1199629 bytes, sha256 7e1491b8…)
        mode: "create",
        verification: "verified",
        display_name: "my-first-agent",
        mode_key: "default",
        latest_revision_number: 1,
        latest_revision_status: Verified,
```

(Condensed — the CLI prints the full upload record. `verification:
"verified"` is the line that matters: the platform re-ran your bundle
against the live engine and accepted it, payload schema and all.)

Your agent joins the arena pool and shows up on the environment's
**Your agents** tab. Re-upload as a revision with
`task upload AGENT_ID=<id>`.

If you are here for a **hiring assessment**: this same upload is what you
seal on your invite page — see the README's
[assessment recipe](README.md#recipes).

## Two seats learning at once

Dance-Off is a duel: two seats, one winner. Plain `task train` trains seat
0 while seat 1 plays neutral. `PARALLEL=1` trains BOTH seats with one
shared policy over the generic PettingZoo view — the classic first rung of
self-play:

```sh
task train ENV=dance-off PARALLEL=1 STEPS=128
```

```
── self-play training dance-off [servo-assist] for 128 seat-steps (both seats learning, one shared policy)
→ onnx: out/policy.onnx (1279116 bytes)
✓ torch/onnxruntime parity: max abs diff 7.451e-08
→ bundle: out/agent-bundle
```

Every seat's transition lands in the same PPO buffer; the exported
artifact is unchanged (one `policy.onnx`), so `task match` and `task
upload` work identically.

## Where to go from here

- Swap the training loop for your own stack — the env is plain
  `gymnasium.make("Lockstep/Env-v0", engine_source=...)`; the README's
  [BYO section](README.md#commands) is the seam.
- Try another environment: `task info ENV=panda-pick` — same commands, a
  MuJoCo arm, its own brief.
- Read the wire spec and write an agent in your favorite language —
  `examples/rust-agent/` is the worked example.
