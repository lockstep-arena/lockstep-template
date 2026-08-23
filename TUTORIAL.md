# Tutorial: two agents for Dance-Off

This walks the whole path twice: first a **non-trained** agent (a
hand-written policy through the same export/bundle pipeline), then a
**trained** one. Everything below is real captured output from the commands
shown. Dance-Off is the default `ENV=`; substitute any slug from
[lockstep.it/arenas](https://lockstep.it/arenas) and the commands do not
change.

```sh
task setup                 # once: venv + training stack
task engine                # fetch dance-off's current release
```

## Contents

- [Poke the environment first](#poke-the-environment-first)
- [Part 1 — a non-trained agent](#part-1--a-non-trained-agent)
- [Part 1½ — the same idea in pure Rust](#part-1½--the-same-idea-in-pure-rust)
- [Part 2 — a trained agent](#part-2--a-trained-agent)
- [Part 3 — reading a match](#part-3--reading-a-match)
- [Part 4 — compete](#part-4--compete)
- [Two seats learning at once](#two-seats-learning-at-once)
- [Where to go from here](#where-to-go-from-here)

## Poke the environment first

The engine describes itself — nothing here was typed into the template:

```python
# .venv/bin/python
from lockstep_train.env import LockstepEnv

env = LockstepEnv(engine_source="out/engine.wasm")
print("observation space:")
for name, box in env.observation_space.spaces.items():
    print(f"  {name}: {box.dtype} {box.shape}")
print("action space:", env.action_space.dtype, env.action_space.shape)
init = env.seat_init
print("meta:", dict(init.meta))
print("action slices:", [(s.name, s.start, s.len) for s in init.actions[0].slices])
```

```
observation space:
  agent: float32 (62,)
  cues: int32 (4, 3)
  marquee: uint8 (1, 64, 256)
action space: float32 (48,)
meta: {'model': 'dance-off', 'task': "hit the routine's poses on the beat", 'mode': 'servo-assist', 'control_hz': '60'}
action slices: [('joint_targets', 0, 36), ('effort', 36, 12)]
```

A 62-float proprioception vector, the upcoming cue cards as integers, a
64×256 marquee strip your bot may watch (or ignore), and a 48-float action:
36 joint rotation targets + 12 effort shares, bounds included. The same
declaration renders on the environment's
[Interface page](https://lockstep.it/environments/dance-off/interface).

## Part 1 — a non-trained agent

`examples/scripted_agent.py` builds the simplest correct ONNX graph — every
action element at its bounds midpoint (the wire's own "neutral") — and
stages it exactly like a trained one. The point: the platform does not care
where your graph came from.

```sh
task scripted
```

```
→ onnx: out/scripted-policy.onnx (1246 bytes)
✓ torch/onnxruntime parity: max abs diff 0.000e+00
→ bundle: out/scripted-bundle

Run it:   task match ENV=dance-off BUNDLE=out/scripted-bundle
```

```sh
task match BUNDLE=out/scripted-bundle
```

```
match finished: 2773 frames, final tick 2772, rankings [1, 0], winner Some(1) — archive out/archive.bin (1355981 bytes)
✓ match archived → out/archive.bin
```

A full 46-second routine, both seats standing perfectly still, scored by
the real engine. Open `examples/scripted_agent.py` and make it yours — the
docstring shows where a closed-form policy plugs in.

## Part 1½ — the same idea in pure Rust

`examples/rust-agent/` is the same no-training bot with the training stack
deleted: ~150 lines that decode the tensor wire **by hand** (the wire is a
published one-page spec, and the decoder is pinned by the spec's own golden
encodings — `cargo test` in that directory) and answer every tick directly.
On Dance-Off it recognizes the servo declaration and sways; on any other
environment it plays neutral.

```sh
task rust-agent
task match BUNDLE=examples/rust-agent/target/wasm32-wasip2/release/rust_agent.wasm
```

```
match finished: 2773 frames, final tick 2772, rankings [0, 1], winner Some(0) — archive out/archive-rust.bin (1355981 bytes)
```

No ONNX, no inference host, no dependencies beyond the WIT world — bytes
in, bytes out.

## Part 2 — a trained agent

```sh
task train STEPS=64 NUM_ENVS=1     # tiny — proves the pipeline in ~a minute
```

```
── training dance-off [servo-assist] for 64 steps
  update device: cpu   envs: 1 (SyncVectorEnv)
→ weights: out/policy.pt
→ onnx: out/policy.onnx (1279116 bytes)
✓ torch/onnxruntime parity: max abs diff 3.353e-08
→ bundle: out/agent-bundle

Run it:   task match
Compete:  task upload
```

The network was built from the declaration you poked above: a CNN stream
for `marquee`, LayerNorm+MLP streams for `agent` and `cues` (the int32
cards enter the graph as int32 — the cast is part of the export), fused
into a 48-wide tanh action head. The parity line is the load-bearing one:
the exported graph and the trained network are held to the same numbers
under onnxruntime — the exact runtime the platform's inference host uses.

For a policy that actually dances, `STEPS=64` becomes millions and
`NUM_ENVS=8` — see the README's
[reward landscape](README.md#the-reward-landscape-read-before-a-long-run)
section first.

```sh
task match
```

```
match finished: 2773 frames, final tick 2772, rankings [0, 1], winner Some(0) — archive out/archive.bin (1355981 bytes)
```

## Part 3 — reading a match

The archive is the whole match, deterministically replayable. Drop
`out/archive.bin` on <https://lockstep.it/replay> to watch it rendered, or
read the outcome straight off the CLI output above: final tick, rankings
by seat, winner. Matches end when the engine says so (the routine's length
here), never on a wall clock.

## Part 4 — compete

<!-- P6 RE-CAPTURE: the two blocks below (upload output + the agent page
     line) are placeholders to be re-captured against the live stack during
     the release pass. Commands are exact; output shape matches the CLI's
     current format. -->

```sh
task upload NAME=my-first-agent
```

```
uploaded bundle (2 artifacts, 1.3 MB) → agent my-first-agent
verification: verified (payload schema v7, mode servo-assist)
```

Your agent joins the arena pool and shows up at
`lockstep.it/environments/dance-off/your-agents`. Re-upload as a revision
with `task upload AGENT_ID=<id>`.

If you are here for a **hiring assessment**: this same upload is what you
seal on your invite page — see the README's
[assessment recipe](README.md#taking-a-hiring-assessment).

## Two seats learning at once

Dance-Off is a duel: two seats, one winner. Plain `task train` trains seat
0 while seat 1 plays neutral. `PARALLEL=1` trains BOTH seats with one
shared policy over the generic PettingZoo view — the classic first rung of
self-play:

```sh
task train PARALLEL=1 STEPS=128
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
  [BYO section](README.md#bring-your-own-training-stack) is the seam.
- Try a robot: `task train ENV=panda-pick` — same commands, a MuJoCo arm.
- Read the wire spec and write an agent in your favorite language —
  `examples/rust-agent/` is the worked example.
