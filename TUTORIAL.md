# Tutorial: two agents for Dance-Off

A worked example, end to end, with real output — using **Dance-Off**, the
shipped reference game. (The template itself is game-agnostic: every command
below takes `GAME=<slug>` and defaults to dance-off; the flow is identical
for any installed game package.) You'll build **two** agents:

1. a **non-trained** one — a hand-written policy, no RL, ~30 lines — to see
   what an agent actually *is*;
2. a **trained** one — PPO against the real engine, exported to ONNX.

Both go through the identical export → bundle → match pipeline, because to
the platform an agent is just an ONNX graph with the right signature sitting
next to the mode's WASM shell. Where the policy came from is your business.

The observation and action types used throughout are documented on the
[dance-off interface page](https://lockstep.it/games/dance-off/interface?mode=servo-assist).

Everything below assumes you've done the [prerequisites](README.md#prerequisites)
and `task setup`.

## Contents

- [Poke the environment first](#poke-the-environment-first)
- [Part 1 — a non-trained agent](#part-1--a-non-trained-agent)
- [Part 1½ — the same idea in pure Rust](#part-1--the-same-idea-in-pure-rust)
- [Part 2 — a trained agent](#part-2--a-trained-agent)
- [Part 3 — reading a match](#part-3--reading-a-match)
- [Part 4 — compete](#part-4--compete)
- [Two seats learning at once](#two-seats-learning-at-once)
- [Where to go from here](#where-to-go-from-here)

## Poke the environment first

Before building anything, step the real engine by hand. `task engine`
downloads it (208 KB of WASM — the same binary ranked matches run), then:

```python
import gymnasium
import lockstep_dance_off  # registers Lockstep/DanceOff-v0

env = gymnasium.make("Lockstep/DanceOff-v0", engine_source="out/engine.wasm")
obs, info = env.reset(seed=42)
print("marquee:", obs["marquee"].shape, obs["marquee"].dtype)
print("agent:  ", obs["agent"].shape, obs["agent"].dtype)
print("action: ", env.action_space)

total = 0.0
for _ in range(300):  # 5 seconds of game time
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total += reward
    if terminated or truncated:
        break
print("300 random ticks → total reward:", round(total, 3))
env.close()
```

Run it with `.venv/bin/python` and you'll see:

```
marquee: (64, 256, 1) uint8
agent:   (62,) float32
action:  Box(-1.0, 1.0, (48,), float32)
300 random ticks → total reward: 0.0
```

Three things worth absorbing:

- **The marquee is an image** — the scrolling move cards, host-rasterized,
  byte-identical to what a submitted agent sees in a ranked match. It is the
  goal signal: what you're being asked to dance, and how soon.
- **The agent vector is body state** — root pose, 12 joint quaternions,
  fallen flag, combo, score, tick (full layout in `lockstep_dance_off.env`
  and on the [interface page](https://lockstep.it/games/dance-off/interface?mode=servo-assist)).
- **Random flailing earns exactly zero.** Reward is the per-step score
  delta, and score comes from hitting move cards with good posture. This
  game does not pay for enthusiasm.

## Part 1 — a non-trained agent

[`examples/scripted_agent.py`](examples/scripted_agent.py) is the whole
thing. The policy is a `torch.nn.Module` with no parameters worth training —
it ignores the marquee, reads the current tick out of the agent vector, and
sways two joints on a slow sine at even effort:

```python
class Sway(nn.Module):
    # The export seam reads these three attributes — that's the whole
    # interface between "your policy" and the pipeline:
    action_len = ACTION_LEN                    # 48: 36 pose + 12 effort shares
    input_names = ["marquee", "agent"]         # the ONNX signature, by name
    input_shapes = {"marquee": (1, 64, 256), "agent": (62,)}

    def forward(self, marquee, agent):
        tick_seconds = agent[:, -1:] / 60.0
        wave = 0.35 * torch.sin(tick_seconds * 3.0)
        pose = torch.cat([wave.expand(-1, 2), wave.new_zeros(wave.shape[0], 34)], dim=1)
        effort = wave.new_full((wave.shape[0], 12), 0.5)
        return torch.cat([pose, effort], dim=1)
```

Build and run it:

```sh
task scripted
task match BUNDLE=out/scripted-bundle
```

The build is instant — it's the same export + parity-check + bundle staging
the trained path uses, just with nothing to train:

```
→ onnx: out/scripted-policy.onnx (1672 bytes)
✓ torch/onnxruntime parity: max abs diff 0.000e+00
→ bundle: out/scripted-bundle
```

And the match (self-play, seed `0703`, the engine's own 2773-tick routine):

```
match finished: 2773 frames, final tick 2772, rankings [0, 1], winner Some(0)
top-score:          137.4
mean-card-quality:  0.12
moves-hit:          4
falls:              0
mean-posture:       0.99
```

A 1.6 KB ONNX file that does nothing but stand steady and sway scores **137
points with zero falls**. Remember that number.

## Part 1½ — the same idea in pure Rust

The sway bot again — but this time with **no Python, no ONNX, no prebuilt
shell**. [`examples/rust-agent/`](examples/rust-agent/) implements the
`lockstep:agent` WIT world directly (vendored under its `wit/`) and speaks
the published FlatBuffers contract (`contract.fbs` — the current one is on the
game's [Interface tab](https://lockstep.it/games/dance-off/interface),
vendored at `contract/dance-off.fbs`); `task contract` refreshes the vendored
copy from the CDN release the engine pin points at. The View is
read zero-copy — the 16 KB marquee raster is never even touched:

```rust
fn on_tick(view: Vec<u8>) -> Vec<u8> {
    let tick = ViewRef::read_as_root(&view).and_then(|v| v.tick()).unwrap_or(0);
    let wave = 0.35 * libm::sinf(tick as f32 / 60.0 * 3.0);
    // …two joints sway, even effort, planus-build a ServoInput…
}
```

```sh
rustup target add wasm32-wasip2
task rust-agent
task match BUNDLE=examples/rust-agent/target/wasm32-wasip2/release/rust_agent.wasm
```

```
match finished: 2773 frames, rankings [0, 1], winner Some(0)
top-score:      299.9
falls:          0
```

A 64 KB component, from public contracts only — and it out-scores both the
Python sway bot and the smoke-trained policy. Any language that compiles to
a wasm component can do exactly this from the same `.fbs`.

## Part 2 — a trained agent

```sh
task train STEPS=1024
```

`STEPS=1024` is a smoke-test run (a few seconds of training) so you can watch
the whole pipeline; the default is 8192, and a real run wants orders of
magnitude more — but read
[the reward landscape](README.md#the-reward-landscape-read-before-a-long-run)
before assuming steps alone will get you a scoring agent. Output:

```
── training dance-off [servo-assist] for 1024 steps
  update device: mps   envs: 8 (DanceOffVectorEnv)
     1024/1024 steps  episodes=0    mean_return=     nan     2.3s
→ weights: out/policy.pt
→ onnx: out/policy.onnx (1209048 bytes)
✓ torch/onnxruntime parity: max abs diff 4.470e-08
→ bundle: out/agent-bundle

Run it:   task match
Compete:  task upload
```

Reading it:

- `envs: 8 (DanceOffVectorEnv)` — collection steps 8 engine instances in
  parallel. Dance-off ships a NATIVE vector env (engines on Rust threads in
  this process), so the loop picked it automatically; games without one get
  the same parallelism from worker processes (`AsyncVectorEnv`), and the
  update pass runs on the best available accelerator (here Apple MPS)
  either way. None of it changes what trains — the same policy comes out
  of a CPU-only box, just slower.
- `episodes=0`, `mean_return=nan` — 1024 steps across 8 parallel envs is 128
  ticks each, nowhere near a full episode (the routine runs ~2773 ticks), so
  no episode ever finished. Expected for a smoke run.
- Two files the smoke run already produced that matter for real runs:
  `out/checkpoint.pt` (rewritten atomically every rollout — a crash loses at
  most one rollout, `task train RESUME=1` continues) and `out/metrics.csv`
  (one diagnostics row per rollout — see
  [the README](README.md#watch-a-run-metricscsv) for how to read it).
- The **parity check** re-runs the exported ONNX under onnxruntime — the
  exact runtime the platform's inference host uses — and compares against
  torch. `3.0e-08` means the export IS the network you trained. A real graph
  difference (a dropped activation, a transposed input) fails loudly here
  instead of silently shipping a broken agent.
- The bundle is the submittable artifact:

  ```
  out/agent-bundle/
    lockstep.toml          declares the `policy` artifact by name
    component.wasm         the mode's shell (from the wheel — not trained)
    artifacts/policy.onnx  your network
  ```

Now run it:

```sh
task match
```

```
match finished: 2773 frames, final tick 2772, rankings [0, 1], winner None
top-score:          0.0
mean-card-quality:  0.05
moves-hit:          2
falls:              13
mean-posture:       0.84
```

**The 30-second trained agent loses to the sway bot** — 0 points and 13
falls against 137 and none. That is the honest baseline of this template,
and it's the right lesson to start from: continuous control is hard, and a
policy that hasn't learned to stand yet spends the match on the floor. The
sway bot's score is the number your training run has to beat before it has
learned anything at all — and beating it takes more than steps: a real
2M-step run of this exact loop still ended at score 0 (dance-off pays only
for hitting cards, and random motion never does). What it takes is YOUR
work on top of honest machinery — see
[the reward landscape](README.md#the-reward-landscape-read-before-a-long-run)
for where to start.

## Part 3 — reading a match

`task match` is a REAL match — `lockstep match run` executes the engine and
both agent bundles exactly as a ranked match would, then writes
`out/archive.bin`: a full-frame archive, the same format the platform's
watch page replays. The metrics block at the end is per-match telemetry:

| Metric | Meaning |
|---|---|
| `top-score` / `top-rating` | best seat's score and its rating conversion |
| `mean-card-quality` | average hit quality across resolved cards (0..1) |
| `cards-resolved` / `moves-hit` | cards that crossed the line / cards actually hit |
| `falls` | times a dancer hit the floor |
| `mean-posture` | average uprightness (1.0 = never wobbled) |

Self-play note: both seats run YOUR bundle, so seat differences are the
seed, not skill. Local inference uses whatever your machine has
(CoreML / DirectML / CPU) — timings are indicative; the watch page's
per-tick charts are authoritative.

## Part 4 — compete

```sh
cp .env.example .env    # then fill in LOCKSTEP_API_KEY
task upload             # creates the agent, waits for verification
```

The platform's verifier re-validates the bundle against the *published*
engine and, on success, the agent joins its mode's ladder. Later, after more
training:

```sh
task upload AGENT_ID=<id-from-the-first-upload>   # new revision, same agent
```

To sanity-check a bundle without credentials:
`lockstep agent validate --bundle out/agent-bundle`.

## Two seats learning at once

Dance-off has no opponent: dancers score independently, so one seat is the
whole story. A **duel** is different — in jetpack-joust the score is
zero-sum (a landed hit is +1 for the attacker and −1 for the victim), and a
policy trained against a free-falling seat 1 has never met a lance. The
`OPPONENT=` option (below) freezes a previous export in seat 1; the step
beyond that is to let BOTH seats learn at the same time:

```sh
task setup GAME=jetpack-joust          # the wheel's PettingZoo env comes with it
task train GAME=jetpack-joust PARALLEL=1 STEPS=50000
```

What changes, and what does not:

- **The env.** The game's wheel declares a PettingZoo `ParallelEnv`
  (training contract v2, `parallel_env_id`) over the same engine: agents
  `seat_0` and `seat_1`, each observing from its own frame and acting every
  tick, rewards = each seat's own score delta (they sum to zero). Per seat it
  is exactly the Gymnasium env — same observation keys, same action box.
- **The loop.** [`train/core/self_play.py`](train/core/self_play.py) is the
  single-seat PPO with one variable changed: ONE network chooses both
  seats' actions (one batched forward pass), and both seats' transitions go
  into the same buffer. `STEPS` counts seat-steps, so a duel contributes two
  per tick. The opponent is never frozen — it is the policy itself, so the
  curriculum moves with you. League/population play is deliberately not
  here; this is the first rung.
- **The artifact.** Unchanged. Competition is one agent per seat answering
  `on_tick`, so the run still exports ONE `policy.onnx`, through the same
  parity check, into the same `out/agent-bundle/`. `task match` and `task
  upload` work as before — PettingZoo is a training-side view only.
- **Reading the numbers.** In a zero-sum game the pooled `mean_return` is
  ~0 by construction; the new `mean_abs_return` column in `metrics.csv`
  (`mean|return|` on the console) is the one that says whether any jousting
  happened. `episodes=0` for a while is
  normal: a duel lasts 1800 ticks (`--time-limit-ticks`) and the first
  rollouts are shorter than that.
- **What is refused.** `PARALLEL=1` on a game without a parallel env (dance-
  off) stops before training with a message naming the game; so do
  `PARALLEL=1` together with `OPPONENT=` (different self-play rungs) or
  `NUM_ENVS` (there is ONE parallel env; the engine is not the bottleneck).

## Where to go from here

- **Beat the sway bot** — the real work. Raw steps demonstrably aren't
  enough on this reward landscape; the promising directions (reward shaping
  against the match metrics, curriculum, bigger nets, imitation, your own
  algorithm) are laid out in
  [the reward landscape](README.md#the-reward-landscape-read-before-a-long-run).
  The machinery keeps every attempt cheap: crash-safe checkpoints,
  comparable `metrics.csv` runs, parity-checked bundles.
- **Bring your own training stack**: the env is standard Gymnasium (works
  with SB3 / CleanRL / torchrl out of the box) and the export/stage seam
  takes any policy that answers the derived signature — see
  [the README](README.md#bring-your-own-training-stack).
- **Try the research tier**: `task train MODE=raw-torque` — no servo
  assistance, your policy outputs raw joint torques (action becomes
  `(36,)`). Its own ladder; a bundle targets exactly one tier. A plain
  `task match` afterwards does the right thing: the staged bundle records
  the mode it was trained for, and the match fetches that mode's engine.
- **Fight a real opponent (games with an adversarial seat)**: on
  jetpack-joust, `task train GAME=jetpack-joust
  OPPONENT=out/agent-bundle/artifacts/policy.onnx` fills seat 1 with your
  PREVIOUS export instead of the free-fall baseline — train, re-point
  OPPONENT at the new bundle, repeat: a poor man's self-play ladder. The
  real one is `PARALLEL=1` — [two seats learning at
  once](#two-seats-learning-at-once). (dance-off has no opponent seat to
  fill — dancers score independently, and `task train` says so if you try.)
- **Replace the network**: `train/` is yours. Only the derived ONNX
  signature (obs keys → `action`, enforced by `train/core/export.py`) and
  the bundle layout are load-bearing — see
  [the contract](README.md#the-contract) and the
  [interface page](https://lockstep.it/games/dance-off/interface?mode=servo-assist)
  for the types.
- **Get smarter than a sine wave without RL**: `examples/scripted_agent.py`
  never looks at the marquee. A scripted policy that *reads the cards* is a
  perfectly legitimate agent — and a stronger baseline than you'd guess.
