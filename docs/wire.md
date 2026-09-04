# The Lockstep wire — version 1

This is the **normative spec** of the agent-facing payload every Lockstep
environment speaks. The Rust module `src/wire.rs` (`lockstep_interface::wire`)
is the reference implementation; `lockstep-train` (Python) is a second,
conforming decoder; `tests/fixtures/wire/*.bin` + `*.json` are the golden
encodings both are tested against.

## The mental model

An environment **declares its layout once, then sends near-raw bytes
forever**. Three byte-streams cross the engine↔agent boundary:

1. **`SeatInit`** — sent once per seat, before the first tick. It *declares*
   every named **observation** and every named **action** this seat will see
   and send: element type, shape, bounds, and plain-language documentation
   down to the individual element run. It also carries free-form metadata and
   the seat's **brief** — goal, reward, episode end — in the engine's own
   words.
2. **`View`** — sent every tick: a tick counter, that tick's `reward`, the
   `done` flag, then one length-prefixed blob per declared observation.
   **Blob i is the i-th declared observation.** Nothing per-tick is named or
   typed — the declaration already said what every byte means.
3. **`Input`** — returned every tick by the agent: one length-prefixed blob
   per declared action, in declared order.

Everything self-describing lives in the declaration; everything per-tick is
positional. Consequences worth stating plainly:

- **Decoding a `View` or `Input` requires the `SeatInit`.** A tick payload
  alone is just blobs; only the declaration says which bytes are a
  quaternion and which are a time-left fraction.
- **Counts and lengths must match the declaration exactly.** A `View` carries
  exactly `n_obs` blobs (one per declared observation, declared order) and
  an `Input` exactly `n_act`; every blob's byte length is exactly its
  declaration's `dtype size × product(shape)`.
- A training host, an ONNX shell or a hand-written agent reads the
  declaration and knows the whole interface — no codegen, no per-environment
  package. The per-tick `reward` is computed by the engine, so training and
  a live match are literally the same computation.
- A person reads the same declaration and knows what every element *means*:
  the engine is the single source of an environment's documentation, and
  every surface that explains one (the Interface page, `lockstep_train.info`
  in a terminal, an agent's own log) renders this declaration.

## Encoding rules

- All integers and floats are **little-endian**. `f32` is IEEE-754 binary32.
- `str` = `u16 len` + that many bytes of UTF-8 (no terminator). An empty
  string is the two bytes `00 00`.
- A value's bytes are its elements in row-major order, `dtype size ×
  product(shape)` bytes exactly. Rank 0 means one element.
- `dtype`: `0 = f32` (4 bytes), `1 = u8` (1 byte), `2 = i32` (4 bytes).
- Observations and actions appear in **declared order** with **exact byte
  lengths** in every `View` and `Input`.

```
SliceSpec  : name str | doc str | unit str | start u32 | len u32

ValueSpec  : name str | doc str | dtype u8 | rank u8 | shape u32[rank]
           | low f32 | high f32
           | has_elem_bounds u8 | [ low f32[n] | high f32[n] ]      -- n = product(shape)
           | n_slices u32 | SliceSpec[n_slices]

SeatInit   : magic "LSTI" | version u32 = 1 | seat u32
           | n_obs u32 | ValueSpec[n_obs]
           | n_act u32 | ValueSpec[n_act]
           | n_meta u32 | { key str, value str }[n_meta]
           | goal str | reward str | ends str

View       : magic "LSTV" | tick u32 | reward f32 | done u8 | pad u8[3]
           | n_obs u32 | { len u32 | bytes[len] }[n_obs]

Input      : magic "LSTA" | n_act u32 | { len u32 | bytes[len] }[n_act]
```

`n_obs` is the number of declared observations and `n_act` the number of
declared actions — the same counts, in the same order, in the `SeatInit` and
in every `View`/`Input` of that seat's match.

### `SliceSpec`

A named contiguous run of elements inside a **flat** value — documentation
for humans and tooling, never a second layout; decoders may ignore slices.
Slices cannot describe columns of a rank ≥ 2 value — that is what the
value's own `doc` is for.

| Field | Meaning |
|---|---|
| `name` | Unique within its value (`joint_pos`, `beacon_body`). |
| `doc` | One plain-language line: what the run IS. |
| `unit` | The physical unit every element carries, short and concrete — `"m"`, `"rad/s"`, `"rad/s, trunk frame"`, `"unit quaternion wxyz, world frame"`. Empty when dimensionless (a flag, a fraction, a count). |
| `start` / `len` | Elements `start .. start+len` of the flattened value. |

### `ValueSpec`

Each observation and action is a named **value**: dtype, shape, bounds,
documented slices.

| Field | Meaning |
|---|---|
| `name` | Unique within its list. ONNX input/output names match it (see *The ONNX signature* below). |
| `doc` | One plain-language line: what this observation or action is. For rank ≥ 2 it also states what the rows and columns mean, e.g. *"the next 4 cues as rows of (move_id, modifier, ticks_to_hit), nearest first, −1 past the end"*. |
| `dtype` | Element type. A `u8` value is conventionally a `0..=255` image; `i32` carries discrete codes. |
| `shape` | Row-major. `[64, 256]` is 64 rows of 256. |
| `low` / `high` | Scalar bounds every element satisfies (`±inf` allowed for "unbounded"). |
| `elem_bounds` | Optional per-element `(low[n], high[n])` that tighten the scalar bounds — joint ranges, actuator `ctrlrange`. When present both arrays are exactly `n` long. |
| `slices` | Documented runs of a FLAT value, in any order; when present they should tile it (see *Documenting your environment*). |

### `SeatInit`

Sent once per seat (the engine world's `init` returns it per seat; the agent
world's `init` receives it). `seat` is the seat index. `meta` is free-form
`(key, value)` string pairs documented per environment — `control_hz`,
`model`, `task`, `episode_ticks` are the conventional ones — and never
load-bearing for decoding.

The three trailing strings are the seat's **brief**, per seat by construction
so an adversarial second seat can state its own goal:

| Field | Meaning |
|---|---|
| `goal` | The task, as you would explain it to a person taking the seat. |
| `reward` | What earns reward each tick and what costs it — enough to predict the sign of the number before training on it. |
| `ends` | Every condition that ends an episode (time-out, fall, success …) and what the final tick looks like. |

**Readers ignore unknown trailing bytes.** A future revision of this spec may
append fields after `ends`; a decoder that has consumed everything it knows
about MUST accept (and ignore) any bytes that remain. Only the declaration is
extensible this way — `View` and `Input` decoders stay strict, and a writer
never emits trailing bytes it cannot name.

The platform captures the decoded seat-0 `SeatInit` at release time as the
mode's `declaration_json` (the same JSON shape `serde` gives
`wire::SeatInit`; see `tests/fixtures/wire/seat_init.json`). The Interface
page, the admin readiness check and `lockstep_train.info` all render from
that capture.

### `View`

Sent every tick for every seat. `reward` is that tick's score delta for this
seat — the number a training loop sums; `done` mirrors the engine's
`get-status` (`true` on the final view). Then one blob per declared
observation, in declared order, each with its exact declared byte length.

The three `pad` bytes after `done`: **writers zero them, readers ignore
them.** They are reserved for future per-tick flags.

### `Input`

Returned every tick by the agent: one blob per declared action, in declared
order, each with its exact declared byte length.

## Malformed input is never a trap

A world decodes an `Input` by validating **magic, blob count, every byte
length, and (for `f32` values) that every element is finite**. Any failure —
including an empty payload, which is what a seat that missed its time slice
sends — yields the **neutral action** and bumps the environment's
`bad-inputs` session metric. The match continues.

The neutral action is, per element, the **midpoint of its bounds** when both
are finite (per-element bounds first, else the scalar pair), **else 0**;
`u8`/`i32` values round the midpoint. Worlds additionally **clamp** every
`f32` action element into its bounds before use, so an agent can never drive
an actuator past `ctrlrange`.

`wire::Input::decode_for(bytes, &actions)` is this rule; `wire::
ValueSpec::neutral_f32` / `clamp_f32` are the pieces.

## The ONNX signature

The platform's generic ONNX shell (`agent-onnx`) runs an exported policy
against this wire with no per-environment code, so the model's signature is
fixed by the declaration:

- **One ONNX input per declared observation, named exactly as declared**,
  shape `[1, *shape]` (a batch of one). `f32` and `i32` observations are fed
  as declared. **`u8` image observations are fed as `f32` divided by 255** —
  a `marquee u8[1,64,256]` declaration is the ONNX input
  `marquee: f32[1,1,64,256]` holding values in `0..=1`. A policy trained on
  raw `0..=255` values runs without error and reads noise. The Gymnasium env
  in `lockstep-train` hands you the raw `uint8` observation; the template's
  policy divides by 255 on the way into the network and exports the graph
  with that input, so a policy trained and exported through the template
  matches. Bring your own stack and the division is yours to put inside the
  exported graph.
- **One ONNX output per declared action, named exactly as declared** (a
  single declared action may also be named `action`), shape `[1, *shape]`,
  `f32`, **in `[-1, 1]`**. The shell clamps each element to `[-1, 1]` and
  maps it affinely onto that element's declared bounds (per element when the
  declaration has per-element bounds); an element with an open bound passes
  through unscaled. This is what a tanh-headed policy trained with
  `lockstep-train` — whose action space is the same normalized box — needs:
  training and a live match are the same computation. A graph that emits
  raw joint angles instead of `[-1, 1]` is silently clamped to the ends of
  every range.

## Documenting your environment

The declaration is the documentation. Write it with the builders:

```rust
ValueSpec::f32_vec("obs", 54, -INF, INF)
    .with_doc("everything the policy sees this tick, trunk-relative")
    .with_documented_slices(&[
        ("trunk_quat", 4, "body orientation", "unit quaternion wxyz, world frame"),
        ("beacon_body", 3, "beacon minus trunk, rotated into the trunk frame", "m, trunk frame"),
        ("time_left", 1, "fraction of the episode remaining", ""),
    ]);

SeatInit::new(seat, obs, actions)
    .with_meta("control_hz", 50)
    .with_brief(
        "Walk to the beacon and stand on it.",
        "+ progress toward the beacon each tick; − a fall.",
        "The episode ends when the robot falls, or after 20 s.",
    )
```

`SeatInit::undocumented()` is the audit: it returns every `DocGap` — an empty
brief paragraph, an observation, action or slice with no `doc`, and, for
values that declare slices, any elements no slice covers, any two slices
that overlap, and any slice that runs past its value. Environment test
suites assert it is empty for seat 0; the platform's admin readiness check
renders the same list for the captured declaration. A value with no slices
(an image plane, a table) is explained by its `doc` alone, and rank ≥ 2
values should be documented that way rather than sliced.

`with_slices(&[(name, len)])` still exists for the undocumented case; every
slice it builds is reported by the audit.

## Versioning

`WIRE_VERSION` (in `SeatInit`) is the layout of THIS document. Bumping it is
a new wire (a new `docs/wire-v2.md`, new magic numbers if the layout changes
incompatibly), never an in-place mutation. Two escape valves make most bumps
unnecessary: unused `dtype` codes can be assigned, and `SeatInit` is
tail-extensible (readers ignore unknown trailing bytes).

An environment's `descriptor.payload-schema-version` is its own coordinate on
top of the wire: it bumps when the environment adds, removes or reshapes an
observation or action, renames one, or changes what a slice means. A bump
marks every existing agent for that mode stale.

## The complete data surface

Where every byte-stream and contract around a match is defined:

| Surface | What it is | Documented in |
|---|---|---|
| `seat-init` / `view` / `input` payloads | The agent-facing wire — this document | here; reference impl `src/wire.rs` |
| `seed` payload | Host-drawn randomness handed to `engine.init` as data; opaque, environment-folded | `wit/engine-v0.2/engine.wit` |
| archive frames (`frame` payload) | The omniscient per-tick record, the environment's OWN format, read only by its player | `wit/engine-v0.2/engine.wit`; each environment's player |
| archive container | `SessionArchive` header/frames/trailer (postcard), environment-agnostic | `src/archive.rs` |
| WIT contracts | The engine, agent, physics and inference worlds | `wit/engine-v0.2`, `wit/agent-v0.1`, `wit/physics-v0.1`, `wit/inference-v0.1` |
| captured `declaration_json` | The decoded seat-0 `SeatInit`, captured by the verifier at release; drives the Interface page, readiness and `lockstep_train.info` | platform `docs/registry.md` |
| iframe SDK | The shell↔player protocol (JSON), unrelated to the agent wire | `docs/frontend-sdk.md` |

## Golden fixtures

`tests/fixtures/wire/{seat_init,view,input}.bin` are the exact encodings of
the canonical messages built in `tests/wire_goldens.rs`; the `.json` twins
are their decoded forms (`seat_init.json` is the `serde` JSON of
`wire::SeatInit`; `view.json`/`input.json` list each value's name, dtype and
decoded elements). The seat-init golden deliberately leaves one slice and one
action undocumented so the empty-string encoding is pinned too. Both Rust and
the Python decoder in `lockstep-train` test against the same files.

## Appendix: history

Version 1 replaced the per-environment FlatBuffers world — a `contract.fbs`,
generated code on both sides, and a per-environment Python package whose only
job was decoding it — with the single self-describing layout above. The
documentation channel (`doc`, `unit`, the brief) was added to version 1 **in
place** (2026-08-26): the platform, every environment and every decoder
republished in the same change, so no `SeatInit` written under the old
layout survived to be misread, and every mode's `payload-schema-version`
bumped with it. Details live in the git log and the README's *in-place
exceptions on record*.
