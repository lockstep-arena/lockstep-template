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
   down to the individual element run or column. It also carries free-form
   metadata, the seat's **brief** — goal, reward, episode end — in the
   engine's own words, and the **metrics** the session will report, each with
   a unit and a direction.
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
- A person reads the same declaration and knows what every element *means*
  and how the episode is scored: the engine is the single source of an
  environment's documentation, and every surface that explains one (the
  Interface page, `task info` in a terminal, the hiring report's metrics
  table) renders this declaration.

## The declaration, systematically

A declaration answers a fixed set of questions about each value, and each
question is answered by either a **closed vocabulary** (something tooling
must branch on — every decoder implements it once) or **free text**
(something only a person reads). That split is the whole design rule: a
consumer never branches on a string convention, and an environment never
needs a new vocabulary entry to say something only a human needs to know.

### The facets of a value

One `ValueSpec` per observation and per action.

| Facet | Field(s) | Defined by | Constrained by |
|---|---|---|---|
| Identity | `name` | per value | unique within its list; equals the ONNX input/output name |
| Element type | `dtype` ∈ {f32, u8, i32} | closed vocabulary | — |
| Extent | `shape` = u32[rank] | per value | row-major; `numel = ∏shape`; bytes per tick = `numel × size(dtype)`; what the axes *mean* is documentation, not a vocabulary |
| Element domain | `low`/`high` (one pair for every element), optional `elem_bounds` (one pair per element) | per value | meaning fixed by `dtype` (next table); `elem_bounds` tighten the scalar pair |
| Human meaning | `doc`; `slices[]` (`name, doc, unit, start, len`); `columns[]` (`name, doc, unit`) | free text, structured | `slices` document a rank-1 value; `columns` document the last axis of a rank ≥ 2 value; `doc` states what every other axis is |

### What `dtype` decides

| dtype | code | bytes | `low`/`high` mean | neutral action (per element) | clamp | ONNX input (observation) | ONNX output (action) | `lockstep-train` space |
|---|---|---|---|---|---|---|---|---|
| f32 | 0 | 4 | a real interval; `±inf` = open | the midpoint when both bounds are finite, else 0 | into `[low, high]` | fed as declared f32 | the graph emits `[-1, 1]`, mapped affinely onto `[low, high]`; an open bound passes through unscaled | `Box(low, high)` |
| u8 | 1 | 1 | fixed `0..=255` | 128 | `0..=255` | fed as f32 ÷ 255, in `[0, 1]` | `[-1, 1]` mapped onto `0..=255`, rounded | `Box(0, 255, uint8)` |
| i32 | 2 | 4 | an integer interval; an open bound clamps to the dtype range | the rounded midpoint | into `[low, high]` | fed as int32 | `[-1, 1]` mapped onto the integer range, rounded | `Box(int32)` |
| — | 3 | — | RESERVED (f64) | | | | | |
| — | 4 | — | RESERVED (bool) | | | | | |

Codes 3 and 4 are reserved and unassigned: assigning one touches every
decoder, the ONNX shell, the inference host and the goldens, and nothing yet
needs it. Bounds are data rather than text because they are what makes the
neutral action, clamping and the ONNX map honest.

### Rank and documentation

There is deliberately no vocabulary for what an axis means. A consumer that
decodes bytes does not need one, and the one consumer that might want one — a
starter neural network — is the candidate's own code. Meaning is prose plus
structured column docs:

| rank | typical shape | what says what the axes mean | structured docs |
|---|---|---|---|
| 0 | `[]` | `doc` | — |
| 1 | `[D]` | `doc` + `slices[]` (named runs with a unit) | slices tile the value |
| 2 | `[T, F]`, `[N, F]` | `doc` states what the rows are (the 64 most recent flows, newest last; one row per host, index = host id; …) | `columns[]` names the F columns with a doc and a unit |
| 3 | `[C, H, W]`, `[N, T, F]` | `doc` (a channel-first grayscale crop; per-card histories, …) | `columns[]` on the last axis when it is a feature axis; empty for a spatial last axis |
| n | anything | `doc` | `columns[]` on the last axis when it is meaningful |

### `MetricSpec` — what a session reports

The same pattern for the numbers an episode ends with. The engine's
`get-session-metrics()` returns `(key, value)` pairs; the declaration says
what each key means so a report can render any environment without knowing
which one it is.

| Facet | Field | Defined by | Read by |
|---|---|---|---|
| Identity | `key` (the emitted metric key) | data | the report joins emitted values to specs; the `scenario-` prefix drives candidate redaction |
| Meaning class | `kind` ∈ {score, count, rate, duration, money, flag, scenario} | closed vocabulary | value formatting, chart kind, redaction |
| Better is | `direction` ∈ {higher, lower, neutral} | closed vocabulary | glyphs, compare colouring |
| Display range | `low`/`high` (`±inf` = auto) | data | strip-chart axis |
| Prominence | `headline` | data | score cards |
| Human meaning | `doc`, `unit` | free text | labels |

`kind` codes: `0 score · 1 count · 2 rate · 3 duration · 4 money · 5 flag ·
6 scenario`. `direction` codes: `0 higher · 1 lower · 2 neutral`. A `scenario`
metric describes the seed (a mass, a drift tick, an adversary tier) and is
shown to employers only; its key keeps the `scenario-` prefix so that
redaction works for engines that declare no specs at all. The canonical keys
every environment emits — `score`, `success`, `ticks`, `bad-inputs`,
`fail-tick`, `fail-<reason>` — are declared by the environment kit, not by
each environment.

### `SeatInit` as a whole

| Section | Contents | Extensibility |
|---|---|---|
| header | magic, `WIRE_VERSION`, `seat` | — |
| values | `obs[]`, `actions[]`, each a length-prefixed `ValueSpec` | a per-value field appends inside the region |
| meta | `(key, value)` string pairs | free; documented keys below |
| brief | `goal`, `reward`, `ends` | free |
| tail | `metrics[]` first; later sections append after it | readers stop at end of input |

### Who reads what

| Consumer | name | dtype | shape | bounds | doc / slices / columns | metrics |
|---|---|---|---|---|---|---|
| wire decoders (this crate, `lockstep-train`, the template's Rust and C readers) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| per-tick `View` / `Input` byte checks | | ✓ | ✓ | | | |
| the environment kit's neutral action + clamp | | ✓ | ✓ | ✓ | | |
| `lockstep-train` spaces | ✓ | ✓ | ✓ | ✓ | | |
| the ONNX shell (feed by name; map outputs onto bounds) | ✓ | ✓ | ✓ | ✓ | | |
| the verifier's ONNX preflight (signature) | ✓ | ✓ | ✓ | | | |
| the template scaffolder (`interface.{py,rs,h}`, `model.py`) | ✓ | ✓ | ✓ | ✓ | slices + columns → constants; docs → comments | ✓ |
| `task info`, `lockstep_train.info`, the Interface page | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ ("how you are scored") |
| the hiring report, compare, the candidate's result | | | | | | ✓ |
| admin readiness | | | | | ✓ (`DocGap`) | ✓ (coverage) |

## Encoding rules

- All integers and floats are **little-endian**. `f32` is IEEE-754 binary32.
- `str` = `u16 len` + that many bytes of UTF-8 (no terminator). An empty
  string is the two bytes `00 00`.
- A value's bytes are its elements in row-major order, `dtype size ×
  product(shape)` bytes exactly. Rank 0 means one element.
- Observations and actions appear in **declared order** with **exact byte
  lengths** in every `View` and `Input`.

```
SliceSpec  : name str | doc str | unit str | start u32 | len u32

ColumnSpec : name str | doc str | unit str

ValueSpec  : len u32                                    -- bytes that follow, this value only
           | name str | doc str | dtype u8 | rank u8 | shape u32[rank]
           | low f32 | high f32
           | has_elem_bounds u8 | [ low f32[n] | high f32[n] ]      -- n = product(shape)
           | n_slices u32 | SliceSpec[n_slices]
           | n_columns u32 | ColumnSpec[n_columns]
           | ...                                        -- a reader skips to len

MetricSpec : key str | doc str | unit str | kind u8 | direction u8
           | low f32 | high f32 | headline u8

SeatInit   : magic "LSTI" | version u32 = 1 | seat u32
           | n_obs u32 | ValueSpec[n_obs]
           | n_act u32 | ValueSpec[n_act]
           | n_meta u32 | { key str, value str }[n_meta]
           | goal str | reward str | ends str
           | n_metrics u32 | MetricSpec[n_metrics]      -- tail section (2026-09)

View       : magic "LSTV" | tick u32 | reward f32 | done u8 | pad u8[3]
           | n_obs u32 | { len u32 | bytes[len] }[n_obs]

Input      : magic "LSTA" | n_act u32 | { len u32 | bytes[len] }[n_act]
```

`n_obs` is the number of declared observations and `n_act` the number of
declared actions — the same counts, in the same order, in the `SeatInit` and
in every `View`/`Input` of that seat's match.

### Two extension rules, and what they buy

**Every `ValueSpec` is a length-prefixed region.** `len` counts every byte of
the value after itself. A reader consumes the fields it knows in the order
above and then **skips to the end of the region**, so a field appended to
`ValueSpec` in a later revision is invisible to an older reader and never a
wire bump. A region that ends before the fields the reader knows is a
truncation error, not a declaration; a `len` that runs past the input is the
same error.

**Readers ignore unknown trailing bytes after `ends`.** A future revision may
append further tail sections after `metrics`; a decoder that has consumed
everything it knows about MUST accept (and ignore) any bytes that remain. A
writer that has no metrics still writes `n_metrics = 0`, so an absent tail
(end of input right after `ends`) decodes as `metrics = []` while a tail that
starts and stops short is an error.

Only the declaration is extensible these two ways — `View` and `Input`
decoders stay strict, and a writer never emits trailing bytes it cannot name.

### `SliceSpec`

A named contiguous run of elements inside a **rank-1** value — documentation
for humans and tooling, never a second layout; decoders may ignore slices.

| Field | Meaning |
|---|---|
| `name` | Unique within its value (`joint_pos`, `beacon_body`). |
| `doc` | Plain language: what the run IS. One line, unless the run carries a code table (see *Documenting your environment*). |
| `unit` | The physical unit every element carries, short and concrete — `"m"`, `"rad/s"`, `"USD"`, `"unit quaternion wxyz, world frame"`. Empty when dimensionless. `"code"` marks an integer category (see below). |
| `start` / `len` | Elements `start .. start+len` of the value. |

### `ColumnSpec`

The per-column twin of a slice for a value of **rank ≥ 2**: one entry per
element of the last axis, in order, so a `history f32[16, 8]` declares eight
columns. Either exactly `shape[rank-1]` entries or none; a rank ≤ 1 value
declares none (it uses slices).

| Field | Meaning |
|---|---|
| `name` | Unique within its value (`amount_log`, `mcc_code`). |
| `doc` | What the column IS; carries the code table when `unit` is `"code"`. |
| `unit` | As for a slice. |

### `ValueSpec`

| Field | Meaning |
|---|---|
| `name` | Unique within its list. ONNX input/output names match it (see *The ONNX signature* below). |
| `doc` | Plain language: what this observation or action is. For rank ≥ 2 it also states what the rows (and any further axes) mean and in what order — *"the last 16 transactions on this card, newest first"*. |
| `dtype` | Element type. A `u8` value is conventionally a `0..=255` image; `i32` carries discrete codes. |
| `shape` | Row-major. `[64, 256]` is 64 rows of 256. |
| `low` / `high` | Scalar bounds every element satisfies (`±inf` allowed for "unbounded"). |
| `elem_bounds` | Optional per-element `(low[n], high[n])` that tighten the scalar bounds — joint ranges, actuator `ctrlrange`, a code column beside a probability. When present both arrays are exactly `n` long. |
| `slices` | Documented runs of a rank-1 value, in any order; when present they should tile it. |
| `columns` | Documented columns of the last axis of a rank ≥ 2 value; when present, exactly one per column. |

### `MetricSpec`

| Field | Meaning |
|---|---|
| `key` | The exact key the engine emits from `get-session-metrics()`. `scenario`-kind keys start with `scenario-`. |
| `doc` | What the number means, as a label a hiring manager can read. |
| `unit` | `"pts"`, `"USD"`, `"ticks"`, `"%"`, `"alerts"` … empty when dimensionless. |
| `kind` | See the table above. |
| `direction` | Whether higher or lower is better, or neither (a count that is just a fact). |
| `low` / `high` | The display range for charts; `±inf` = fit to the data. |
| `headline` | The two or three metrics the report should show as cards beside the score. |

### `SeatInit`

Sent once per seat (the engine world's `init` returns it per seat; the agent
world's `init` receives it). `seat` is the seat index.

`meta` is free-form `(key, value)` string pairs, never load-bearing for
decoding. The documented keys:

| key | value | who reads it |
|---|---|---|
| `task` | one line naming the task | `task info`, the Interface page |
| `mode` | the mode key this engine is | the template's scaffolder |
| `model` | the robot / simulator model, when there is one | humans |
| `control_hz` | the control rate | `task info`, budgets |
| `episode_ticks` | the nominal episode length | `task info`, budgets |
| `reward_lag_ticks` | how many ticks after a decision its truth reaches `reward` — a number, or a range like `60-150` | `task info`, the supervised recipe's documentation |
| `tags` | comma-separated `facet:value` assessment tags from the shared vocabulary (`lockstep_interface::tags`); free words without a colon are display-only | the platform seeds the mode's tags from it on first release |

The three trailing strings are the seat's **brief**, per seat by construction
so an adversarial second seat can state its own goal:

| Field | Meaning |
|---|---|
| `goal` | The task, as you would explain it to a person taking the seat. |
| `reward` | What earns reward each tick and what costs it — the score anatomy, enough to predict the sign of the number before training on it. |
| `ends` | Every condition that ends an episode (time-out, fall, a failure mode …) and what the final tick looks like. |

Then the tail: the declared metrics.

The platform captures the decoded seat-0 `SeatInit` at release time as the
mode's `declaration_json` (the same JSON shape `serde` gives
`wire::SeatInit`; see `tests/fixtures/wire/seat_init.json`). The Interface
page, the admin readiness check, the hiring report and `lockstep_train.info`
all render from that capture.

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
sends, and an oversize payload the host refused to forward — yields the
**neutral action** and bumps the environment's `bad-inputs` session metric.
The match continues.

The neutral action is, per element, the **midpoint of its bounds** when both
are finite (per-element bounds first, else the scalar pair), **else 0**;
`u8`/`i32` values round the midpoint. Worlds additionally **clamp** every
`f32` action element into its bounds before use, so an agent can never drive
an actuator past `ctrlrange`.

`wire::Input::decode_for(bytes, &actions)` is this rule; `wire::
ValueSpec::neutral_f32` / `clamp_f32` are the pieces.

Because the neutral action is what a missed or malformed tick does, an
environment chooses its bounds so that the neutral action is the **safe
default** for its domain — no alert, approve, send to review, stay flat — and
the `doc` of the action says so in words.

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
  scaffolded model divides by 255 on the way into the network and exports the
  graph with that input, so a policy trained and exported through the
  template matches. Bring your own stack and the division is yours to put
  inside the exported graph.
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
- **Shapes are exact.** An output whose element count matches but whose
  shape does not (a transposed `[40, 64]` for a declared `[64, 40]`) is
  refused by the shell and by `lockstep-train`'s encoder; it never becomes a
  silently-misread action.

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

ValueSpec::f32_value("history", vec![16, 8], -INF, INF)
    .with_doc("this card's last 16 transactions, newest first; rows past what exists are zero with valid = 0")
    .with_columns(&[
        ("amount_log", "ln(1 + amount)", ""),
        ("mcc_code", "merchant category: 0 grocery, 1 fuel, 2 restaurant, … 23 other", "code"),
        ("since_last", "minutes since the previous transaction, ln(1 + x)", "ln(min)"),
        ("your_decision", "0 approve, 1 decline, 2 step-up, -1 not yours", "code"),
        ("outcome", "1 confirmed fraud, 0 confirmed good, -1 not yet known", "code"),
        ("valid", "1 for a real row, 0 for padding", "flag"),
        // …
    ]);

SeatInit::new(seat, obs, actions)
    .with_meta("control_hz", 50)
    .with_meta("reward_lag_ticks", 40)
    .with_tags(&[Tag::skill(Skill::Classification), Tag::level(Level::Entry)])
    .with_brief(
        "Walk to the beacon and stand on it.",
        "+ progress toward the beacon each tick; − a fall. The episode score is 0–100: …",
        "The episode ends when the robot falls, or after 20 s.",
    )
    .with_metrics(vec![
        MetricSpec::score("score", "0–100 as described in the brief").headline(),
        MetricSpec::money("fraud-loss", "approved amount later charged back", "USD").lower_is_better().headline(),
        MetricSpec::rate("precision", "true alerts over all alerts", "%"),
        MetricSpec::scenario("scenario-drift-tick", "when the rings changed tactics", "ticks"),
    ])
```

`SeatInit::undocumented()` is the audit: it returns every `DocGap` — an empty
brief paragraph, an observation, action, slice, column or metric with no
`doc`, and, for values that declare slices, any elements no slice covers, any
two slices that overlap, and any slice that runs past its value. Environment
test suites assert it is empty for seat 0; the platform's admin readiness
check renders the same list for the captured declaration. A value with no
slices or columns (an image plane) is explained by its `doc` alone.

`with_slices(&[(name, len)])` still exists for the undocumented case; every
slice it builds is reported by the audit.

Rules that make one environment comfortable to work in (the checklist an
environment is held to, not a style guide across environments):

- **One padding mechanism per environment, distinguishable from real
  values.** A trailing `valid` column (1 real, 0 padding) with zero-filled
  padding rows is the recommendation. Never reuse as padding a sentinel that
  also means something real (`-1` = "unknown" in one column and "padding" in
  another is the trap).
- **Every ordered axis states its order in its own `doc`** — newest first or
  oldest first, whichever reads naturally for the domain, said in words.
- **`unit = "code"` marks an integer category, and the code table lives in
  that column's or slice's `doc`.** `task info` and the Interface page render
  such docs as tables. Values and slices otherwise aim for one line.
- **A whole value that is discrete is `i32`.** A mixed row may carry an
  integer category as f32 with unit `code`.
- **The neutral action is the documented safe default**, and the action's
  `doc` names the tie rule when scores tie.
- **Declared metrics are the report table.** Aim for two dozen at most —
  aggregates, per-class rates, the canonical keys. Per-pair and per-cell
  breakdowns (a confusion matrix) belong in the replay, not in metrics.
- **Names.** Value, slice and column names are `snake_case` (they become
  constants in three languages); metric keys are `kebab-case` (they are
  report keys, not identifiers).

### Data domains

Environments that stream records — a transaction per tick, a flow window, a
bar — share a few documented shapes. None of them is a rule; each is a
convention a candidate will recognise from one environment to the next when
the environment chooses it.

- **Delayed truth.** `View.reward` is that tick's score delta, and a world may
  score decision `t` at tick `t + lag`. `meta.reward_lag_ticks` states the lag.
- **A feedback window.** Where the domain allows it, the revealed truth of
  earlier decisions is also observable as a rank-2 value whose rows are the
  decisions that resolved this tick — conventionally columns named
  `age_ticks`, `your_decision`, `truth`, then anything domain-specific, then
  `valid`. `lockstep-train`'s `LabelledStream` joins such a window back to
  the observations it labels; it takes the value and column names as
  arguments and defaults to these.
- **Windows and tables.** A sliding window is `[W, F]` with its order in the
  `doc`; an entity table is `[N, F]` with "index = entity id" in the `doc`;
  both document their columns.
- **Tokens.** Free text never rides the wire. A fixed-length `i32[K]` of
  token ids from an engine-owned vocabulary (0 = padding) does, with the
  tokenizer stated in `task info`.

## Versioning

`WIRE_VERSION` (in `SeatInit`) is the layout of THIS document. Bumping it is
a new wire (a new `docs/wire-v2.md`, new magic numbers if the layout changes
incompatibly), never an in-place mutation. Three escape valves make most
bumps unnecessary: unused `dtype` and `kind` codes can be assigned, `ValueSpec`
is a length-prefixed region (per-value fields append), and `SeatInit` is
tail-extensible (sections append after `ends`).

An environment's `descriptor.payload-schema-version` is its own coordinate on
top of the wire: it bumps when the environment adds, removes or reshapes an
observation or action, renames one, or changes what a slice or column means.
A bump marks every existing agent for that mode stale.

## The complete data surface

Where every byte-stream and contract around a match is defined:

| Surface | What it is | Documented in |
|---|---|---|
| `seat-init` / `view` / `input` payloads | The agent-facing wire — this document | here; reference impl `src/wire.rs` |
| `seed` payload | Host-drawn randomness handed to `engine.init` as data; opaque, environment-folded | `wit/engine-v0.2/engine.wit` |
| archive frames (`frame` payload) | The omniscient per-tick record, the environment's OWN format, read only by its player | `wit/engine-v0.2/engine.wit`; each environment's player |
| archive container | `SessionArchive` header/frames/trailer (postcard), environment-agnostic | `src/archive.rs` |
| WIT contracts | The engine, agent, physics, inference and data worlds | `wit/engine-v0.2`, `wit/agent-v0.1`, `wit/physics-v0.1`, `wit/inference-v0.1`, `wit/data-v0.1` |
| captured `declaration_json` | The decoded seat-0 `SeatInit`, captured by the verifier at release; drives the Interface page, readiness, the report's metric vocabulary and `lockstep_train.info` | platform `docs/registry.md` |
| assessment tags | The `facet:value` vocabulary an environment declares in `meta.tags` and the platform curates | `src/tags.rs` |
| iframe SDK | The shell↔player protocol (JSON), unrelated to the agent wire | `docs/frontend-sdk.md` |

## Golden fixtures

`tests/fixtures/wire/{seat_init,view,input}.bin` are the exact encodings of
the canonical messages built in `tests/wire_goldens.rs`; the `.json` twins
are their decoded forms (`seat_init.json` is the `serde` JSON of
`wire::SeatInit`; `view.json`/`input.json` list each value's name, dtype and
decoded elements). The seat-init golden deliberately leaves one slice and one
action undocumented so the empty-string encoding is pinned too, declares one
rank-2 value with columns, and declares three metrics.
`seat_init_notail.bin` is the same declaration written without the metric
tail, frozen, and pins the rule that an absent tail decodes to `metrics = []`.
Both Rust and the Python decoder in `lockstep-train` test against the same
files.

## Appendix: history

Version 1 replaced the per-environment FlatBuffers world — a `contract.fbs`,
generated code on both sides, and a per-environment Python package whose only
job was decoding it — with the single self-describing layout above. The
documentation channel (`doc`, `unit`, the brief) was added to version 1 **in
place** (2026-08-26): the platform, every environment and every decoder
republished in the same change, so no `SeatInit` written under the old
layout survived to be misread, and every mode's `payload-schema-version`
bumped with it. The expansion to data assessments (2026-09) took the same
route once more for the length-prefixed `ValueSpec` region, `columns`, the
`MetricSpec` tail and the assessment-tag vocabulary: one cutover republished
every engine and wiped every agent, and the region and tail rules mean it is
the last time an addition needs it. Details live in the git log and the
README's *in-place exceptions on record*.
