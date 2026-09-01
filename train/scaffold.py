"""``task create-agent`` — scaffold ``agents/<name>/`` from the engine's own
declaration.

The engine is the single source of what an environment IS. This module
fetches the (env, mode) release into the keyed cache, decodes seat 0's
``SeatInit`` (every observation and action, every documented slice with its
doc/unit/bounds, the goal/reward/ends brief, the budgets) and writes a
ready-to-edit agent project for the language of your choice::

    python -m train.scaffold --name my-bot --env <slug> [--mode M] [--lang python|rust|c]

Two kinds of files, and the contract between them:

- GENERATED (refreshed on every re-run): the interface file
  (``interface.py`` / ``src/interface.rs`` / ``interface.h``) and
  ``agent.toml``. Regenerate freely — after a release bump, say — they
  carry no hand edits.
- YOURS (written once, NEVER touched again): the policy stub
  (``policy.py`` / ``src/lib.rs`` / ``agent.c``) plus the build files.
  The stub answers the neutral action out of the box and shows, in
  comments, exactly how to read every declared slice by name.

Re-running with the same name refuses a different env/mode/lang unless
``--force`` — a name is an identity, not a slot.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from .agents import LANGS, AgentConfig, load_agent, write_agent_toml
from .core import utf8_output
from .core.engine import EnginePaths, ensure_engine

ROOT = Path(__file__).resolve().parents[1]
#: The vendored agent WIT world (+ inference dep) every wasm agent targets.
WIT_DIR = ROOT / "wit"
#: Hand-written wire readers/writers, tested against the spec goldens in
#: reference/ — copied (not templated) into rust/c scaffolds.
REFERENCE = ROOT / "reference"


# ---------------------------------------------------------------------------
# Declaration plumbing
# ---------------------------------------------------------------------------


def read_declaration(paths: EnginePaths):
    """(SeatInit, Budgets, title) from the cached engine — the same call
    ``python -m lockstep_train.info --engine`` renders from."""
    from lockstep_train.info import from_engine

    return from_engine(str(paths.engine), seat=0)


def _ident(name: str) -> str:
    """A safe UPPER_SNAKE identifier fragment from a wire name."""
    frag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    if not frag:
        frag = "value"
    if frag[0].isdigit():
        frag = "_" + frag
    return frag.upper()


def _fmt_f(v: float) -> str:
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:g}"


def _bounds_line(t) -> str:
    if t.elem_bounds is not None:
        lo, hi = t.elem_bounds
        return (
            f"per-element bounds (see the slices): overall [{_fmt_f(float(min(lo)))}, "
            f"{_fmt_f(float(max(hi)))}]"
        )
    return f"bounds [{_fmt_f(t.low)}, {_fmt_f(t.high)}] every element"


def _slice_bounds(t, s) -> str:
    """The tightest per-element bounds statement for one slice."""
    if t.elem_bounds is None:
        return ""
    lo, hi = t.elem_bounds
    seg_lo = [float(v) for v in lo[s.start : s.stop]]
    seg_hi = [float(v) for v in hi[s.start : s.stop]]
    if not seg_lo:
        return ""
    if len(set(seg_lo)) == 1 and len(set(seg_hi)) == 1:
        return f"[{_fmt_f(seg_lo[0])}, {_fmt_f(seg_hi[0])}] per element"
    return f"per-element bounds [{_fmt_f(min(seg_lo))}..{_fmt_f(max(seg_hi))}], varying"


def _wrap(text: str, prefix: str, width: int = 76) -> list[str]:
    out: list[str] = []
    for para in text.splitlines() or [""]:
        line = prefix.rstrip() if not para.strip() else None
        if line is not None:
            out.append(line)
            continue
        words = para.split()
        cur = prefix
        for w in words:
            if len(cur) + len(w) + 1 > width and cur.strip() != prefix.strip():
                out.append(cur.rstrip())
                cur = prefix
            cur += w + " "
        out.append(cur.rstrip())
    return out


def _budget_lines(budgets) -> list[str]:
    pairs = [
        ("tick rate", f"{budgets.tick_rate_hz} Hz" if budgets.tick_rate_hz else None),
        (
            "agent time slice",
            f"{budgets.agent_time_slice_ms} ms per tick (wall clock; miss it and the tick is forfeit)"
            if budgets.agent_time_slice_ms
            else None,
        ),
        (
            "allowed missed ticks",
            str(budgets.allowed_missed_ticks)
            if budgets.allowed_missed_ticks is not None
            else None,
        ),
        (
            "episode ticks",
            f"{budgets.min_ticks}..{budgets.max_ticks}"
            if budgets.min_ticks is not None and budgets.max_ticks is not None
            else None,
        ),
        (
            "roster",
            f"{budgets.roster_min}..{budgets.roster_max} seats"
            if budgets.roster_min is not None and budgets.roster_max is not None
            else None,
        ),
    ]
    return [f"{k}: {v}" for k, v in pairs if v]


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def gen_interface_py(cfg: AgentConfig, init, budgets, title: str) -> str:
    L: list[str] = []
    L.append('"""GENERATED by `task create-agent` — regenerated on re-run; do not edit.')
    L.append("")
    L.append(f"{title}")
    L.append("")
    L.append("Everything below is the engine's OWN declaration (the same facts as")
    L.append("`task info` and the environment's Interface page), frozen into named")
    L.append("constants so your policy reads observations by name, not by magic")
    L.append("index.")
    L.append("")
    L.append("GOAL")
    L.extend(_wrap(init.brief.goal, "  "))
    L.append("REWARD")
    L.extend(_wrap(init.brief.reward, "  "))
    L.append("EPISODE END")
    L.extend(_wrap(init.brief.ends, "  "))
    bl = _budget_lines(budgets)
    if bl:
        L.append("")
        L.append("BUDGETS")
        for b in bl:
            L.append(f"  {b}")
    if init.meta:
        L.append("")
        L.append("META")
        for k, v in init.meta:
            L.append(f"  {k} = {v}")
    L.append('"""')
    L.append("")
    L.append(f'ENV = "{cfg.env}"')
    L.append(f'MODE = "{cfg.mode}"')
    L.append(f"PAYLOAD_SCHEMA_VERSION = {cfg.payload_schema_version}")
    L.append("")

    for kind, values in (("OBS", init.obs), ("ACT", init.actions)):
        header = "Observations" if kind == "OBS" else "Actions"
        arrive = (
            "one Gymnasium Dict entry per name below (this exact declared order)"
            if kind == "OBS"
            else "your policy's output, mapped onto the declared bounds"
        )
        L.append("#" + " " + "─" * 74)
        L.append(f"# {header} — {arrive}.")
        L.append("#" + " " + "─" * 74)
        for i, t in enumerate(values):
            base = f"{kind}_{_ident(t.name)}"
            L.append("")
            for line in _wrap(t.doc or "(the engine declares no doc)", "# "):
                L.append(line)
            L.append(f"# {t.dtype}{list(t.shape)} — {_bounds_line(t)}")
            if t.dtype == "u8" and kind == "OBS":
                L.append("# NOTE: u8 image — the Gymnasium env hands you raw 0..255 uint8;")
                L.append("# the export path divides by 255 inside the graph (ONNX rule).")
            L.append(f'{base} = "{t.name}"')
            L.append(f"{base}_INDEX = {i}")
            L.append(f"{base}_SHAPE = {tuple(int(d) for d in t.shape)}")
            for s in t.slices:
                sb = _slice_bounds(t, s)
                unit = f" [{s.unit}]" if s.unit else ""
                tail = f" — {sb}" if sb else ""
                L.extend(_wrap(f"{s.doc or '(no doc)'}{unit}{tail}", "# "))
                L.append(f"{base}_{_ident(s.name)} = slice({s.start}, {s.stop})")
        L.append("")

    neutral = []
    for t in init.actions:
        neutral.append([round(float(v), 6) for v in t.neutral_f32()])
    L.append("# The neutral action per declared action value — what the engine plays")
    L.append("# for a missing/malformed input (each element's bounds midpoint, else 0).")
    L.append(f"NEUTRAL_ACTION = {neutral!r}")
    L.append("")
    return "\n".join(L)


def gen_policy_py(cfg: AgentConfig, init) -> str:
    obs_reads = []
    for t in init.obs:
        base = f"OBS_{_ident(t.name)}"
        if t.slices:
            s = t.slices[0]
            obs_reads.append(
                f"        #   {t.name}[iface.{base}_{_ident(s.name)}]"
                f"  # {s.doc or s.name}"
            )
        else:
            obs_reads.append(f"        #   obs[iface.{base}]  # {t.doc or t.name}")
    act = init.actions[0] if init.actions else None
    act_comment = (
        f"the declared `{act.name}` value, {act.numel} element(s) in [-1, 1]"
        if act
        else "the declared action"
    )
    reads = "\n".join(obs_reads) if obs_reads else "        #   (no observations declared)"
    return f'''"""YOUR policy — written once by `task create-agent`, never overwritten.

Two ways to make this agent yours:

1. HAND-WRITE IT (no training): edit ``forward`` below — read observations
   by named slice via ``interface.py`` and emit whatever pose you like.
   ``task build AGENT={cfg.name}`` exports THIS module to ONNX, parity-checks
   it and stages the submittable bundle. Out of the box it plays the
   neutral action (every output 0 → each action element's bounds midpoint).

2. TRAIN IT: ``task train AGENT={cfg.name}`` runs PPO with the generic
   trainable policy over the same observations and stages the trained
   bundle instead. (This file is untouched by training.)

The exported graph's contract (the shell enforces it at match time): one
input per declared observation, by name; one output in [-1, 1] mapped
affinely onto the declared action bounds. u8 image observations arrive
here as raw 0..255 uint8 from Gymnasium; the export seam divides by 255
inside the graph, so train on what you see.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import interface as iface  # the generated sibling — regenerate with `task create-agent`


class ScriptedPolicy(nn.Module):
    """Hand-written policy: neutral until you edit ``forward``."""

    def __init__(
        self,
        input_shapes: dict[str, tuple[int, ...]],
        input_dtypes: dict[str, str],
        action_len: int,
    ):
        super().__init__()
        self.input_names = list(input_shapes)
        self.input_shapes = dict(input_shapes)
        self.input_dtypes = dict(input_dtypes)
        self.action_len = action_len

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Inputs arrive batched, one per declared observation, in
        ``self.input_names`` order ({act_comment} out).

        Read a named run of an observation with the generated slices, e.g.::

{reads}

        The zeros below are the neutral action; ``keep_alive`` only stops
        the exporter from pruning unused inputs — delete it once you
        actually read them.
        """
        batch = inputs[0].shape[0]
        keep_alive = sum(t.float().mean() for t in inputs) * 0.0
        return torch.zeros(batch, self.action_len) + keep_alive
'''


def scaffold_python(cfg: AgentConfig, init, budgets, title: str) -> None:
    d = cfg.dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "interface.py").write_text(gen_interface_py(cfg, init, budgets, title))
    policy = d / "policy.py"
    if not policy.exists():
        policy.write_text(gen_policy_py(cfg, init))
        print(f"→ {policy}  (yours — edit it)", file=sys.stderr)
    else:
        print(f"✓ {policy} untouched (yours)", file=sys.stderr)
    print(f"→ {d / 'interface.py'}  (generated)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------


def gen_interface_rs(cfg: AgentConfig, init, budgets, title: str) -> str:
    L: list[str] = []
    L.append("//! GENERATED by `task create-agent` — regenerated on re-run; do not edit.")
    L.append("//!")
    L.append(f"//! {title}")
    L.append("//!")
    L.append("//! The engine's OWN declaration as named constants: read observations")
    L.append("//! by slice range, never by magic index.")
    L.append("//!")
    L.append("//! # Goal")
    L.extend(_wrap(init.brief.goal, "//! "))
    L.append("//!")
    L.append("//! # Reward")
    L.extend(_wrap(init.brief.reward, "//! "))
    L.append("//!")
    L.append("//! # Episode end")
    L.extend(_wrap(init.brief.ends, "//! "))
    bl = _budget_lines(budgets)
    if bl:
        L.append("//!")
        L.append("//! # Budgets")
        for b in bl:
            L.append(f"//! - {b}")
    L.append("")
    L.append("#![allow(dead_code)]")
    L.append("")
    L.append(f'pub const ENV: &str = "{cfg.env}";')
    L.append(f'pub const MODE: &str = "{cfg.mode}";')
    L.append(f"pub const PAYLOAD_SCHEMA_VERSION: u32 = {cfg.payload_schema_version};")
    L.append("")
    for kind, values, modname in (("obs", init.obs, "obs"), ("act", init.actions, "act")):
        what = "observation" if kind == "obs" else "action"
        L.append(f"/// Every declared {what}, in wire order (blob i of a")
        if kind == "obs":
            L.append("/// `View` is the i-th module below).")
        else:
            L.append("/// an `Input` is the i-th module below).")
        L.append(f"pub mod {modname} {{")
        for i, t in enumerate(values):
            m = _ident(t.name).lower()
            L.append(f"    /// {t.doc or '(no doc declared)'}")
            L.append(f"    ///")
            L.append(f"    /// `{t.dtype}{list(t.shape)}` — {_bounds_line(t)}")
            L.append(f"    pub mod {m} {{")
            L.append(f'        pub const NAME: &str = "{t.name}";')
            L.append(f"        pub const INDEX: usize = {i};")
            L.append(f"        pub const LEN: usize = {t.numel};")
            shape = ", ".join(str(int(x)) for x in t.shape)
            L.append(f"        pub const SHAPE: [u32; {len(t.shape)}] = [{shape}];")
            for s in t.slices:
                sb = _slice_bounds(t, s)
                unit = f" [{s.unit}]" if s.unit else ""
                tail = f" — {sb}" if sb else ""
                L.extend(_wrap(f"{s.doc or '(no doc)'}{unit}{tail}", "        /// "))
                L.append(
                    f"        pub const {_ident(s.name)}: core::ops::Range<usize> = "
                    f"{s.start}..{s.stop};"
                )
            L.append("    }")
        L.append("}")
        L.append("")
    L.append("/// Little-endian f32s from a raw blob (a `View` value's bytes).")
    L.append("pub fn f32s(bytes: &[u8]) -> Vec<f32> {")
    L.append("    bytes")
    L.append("        .chunks_exact(4)")
    L.append("        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))")
    L.append("        .collect()")
    L.append("}")
    L.append("")
    return "\n".join(L)


def gen_lib_rs(cfg: AgentConfig, init) -> str:
    first_obs = init.obs[0] if init.obs else None
    obs_example = ""
    if first_obs is not None:
        m = _ident(first_obs.name).lower()
        if first_obs.slices:
            s = first_obs.slices[0]
            sm = _ident(s.name)
            obs_example = (
                f"            // let {m} = interface::f32s(view.values[interface::obs::{m}::INDEX]);\n"
                f"            // let {s.name} = &{m}[interface::obs::{m}::{sm}];"
            )
        else:
            obs_example = (
                f"            // let {m} = view.values[interface::obs::{m}::INDEX]; // raw bytes"
            )
    return f'''//! YOUR agent — written once by `task create-agent`, never overwritten.
//!
//! A hand-written wasm agent for {cfg.env} [{cfg.mode}]: no ONNX, no
//! training — `on-tick` decodes the view with the vendored wire reader
//! (`wire.rs`, tested against the spec goldens) and answers with whatever
//! action you compute. Out of the box it answers the NEUTRAL action.
//!
//! Build:  task build AGENT={cfg.name}   (cargo → wasm32-wasip2 component)
//! Match:  task match AGENT={cfg.name}
//!
//! `interface.rs` (generated) names every declared observation, slice and
//! action — read by name, never by magic index.

mod interface;
mod wire;

wit_bindgen::generate!({{
    path: "wit",
    world: "agent",
    generate_all,
}});

use wire::SeatInit;

struct Agent;

/// Everything worth precomputing once, at `init`.
struct Plan {{
    /// Pre-encoded neutral bytes, one blob per declared action.
    neutral: Vec<Vec<u8>>,
}}

static PLAN: std::sync::Mutex<Option<Plan>> = std::sync::Mutex::new(None);

impl Guest for Agent {{
    fn init(init_state: Vec<u8>) {{
        let plan = SeatInit::decode(&init_state).ok().map(|init| Plan {{
            neutral: init.actions.iter().map(wire::neutral_bytes).collect(),
        }});
        *PLAN.lock().unwrap() = plan;
    }}

    fn on_tick(view: Vec<u8>) -> Vec<u8> {{
        let guard = PLAN.lock().unwrap();
        let Some(plan) = guard.as_ref() else {{
            return Vec::new(); // engine substitutes the neutral action
        }};
        let Ok(view) = wire::View::decode(&view) else {{
            return Vec::new();
        }};

        // Read observations by name via the generated interface, e.g.:
{obs_example}

        // Neutral until you make it yours: replace a blob with
        // wire::f32_bytes(&your_values) at the right action INDEX.
        let values: Vec<Vec<u8>> = plan.neutral.clone();
        let _ = &view;
        wire::encode_input(&values)
    }}
}}

export!(Agent);
'''


def gen_cargo_toml(cfg: AgentConfig) -> str:
    crate = re.sub(r"[^a-z0-9_]+", "_", cfg.name.lower()).strip("_") or "agent"
    return f"""# YOUR build file — written once by `task create-agent`, never overwritten.
[package]
name = "{crate}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
# The one build dependency: the WIT world bindings generator. The wire
# codec is the vendored `src/wire.rs` — hand-written, no deps, tested
# against the spec goldens in the template's reference/ crate.
wit-bindgen = "0.58"
libm = "0.2"

[profile.release]
lto = true
opt-level = "s"
strip = true
"""


def scaffold_rust(cfg: AgentConfig, init, budgets, title: str) -> None:
    d = cfg.dir
    (d / "src").mkdir(parents=True, exist_ok=True)
    # Vendored, refreshed every run: the WIT world + the wire reader.
    if (d / "wit").exists():
        shutil.rmtree(d / "wit")
    shutil.copytree(WIT_DIR, d / "wit")
    shutil.copyfile(REFERENCE / "rust-wire" / "src" / "wire.rs", d / "src" / "wire.rs")
    (d / "src" / "interface.rs").write_text(gen_interface_rs(cfg, init, budgets, title))
    print(f"→ {d / 'src' / 'interface.rs'}  (generated)", file=sys.stderr)
    for path, content in (
        (d / "Cargo.toml", gen_cargo_toml(cfg)),
        (d / "src" / "lib.rs", gen_lib_rs(cfg, init)),
    ):
        if path.exists():
            print(f"✓ {path} untouched (yours)", file=sys.stderr)
        else:
            path.write_text(content)
            print(f"→ {path}  (yours — edit it)", file=sys.stderr)
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("target/\nout/\n")


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------


def gen_interface_h(cfg: AgentConfig, init, budgets, title: str) -> str:
    guard = f"AGENT_{_ident(cfg.name)}_INTERFACE_H"
    L: list[str] = []
    L.append("/* GENERATED by `task create-agent` — regenerated on re-run; do not edit.")
    L.append(" *")
    L.append(f" * {title}")
    L.append(" *")
    L.append(" * The engine's OWN declaration as named constants: read observations")
    L.append(" * by START/LEN, never by magic index.")
    L.append(" *")
    L.append(" * GOAL")
    L.extend(_wrap(init.brief.goal, " *   "))
    L.append(" * REWARD")
    L.extend(_wrap(init.brief.reward, " *   "))
    L.append(" * EPISODE END")
    L.extend(_wrap(init.brief.ends, " *   "))
    bl = _budget_lines(budgets)
    if bl:
        L.append(" *")
        L.append(" * BUDGETS")
        for b in bl:
            L.append(f" *   {b}")
    L.append(" */")
    L.append(f"#ifndef {guard}")
    L.append(f"#define {guard}")
    L.append("")
    L.append(f'#define AGENT_ENV "{cfg.env}"')
    L.append(f'#define AGENT_MODE "{cfg.mode}"')
    L.append(f"#define AGENT_PAYLOAD_SCHEMA_VERSION {cfg.payload_schema_version}")
    L.append("")
    for kind, values in (("OBS", init.obs), ("ACT", init.actions)):
        what = "observation" if kind == "OBS" else "action"
        L.append(f"/* ── Every declared {what}, in wire order ── */")
        for i, t in enumerate(values):
            base = f"{kind}_{_ident(t.name)}"
            L.append("")
            L.append("/*")
            L.extend(_wrap(t.doc or "(no doc declared)", " * "))
            L.append(f" * {t.dtype}{list(t.shape)} — {_bounds_line(t)}")
            L.append(" */")
            L.append(f'#define {base}_NAME "{t.name}"')
            L.append(f"#define {base}_INDEX {i}")
            L.append(f"#define {base}_LEN {t.numel}")
            for s in t.slices:
                sb = _slice_bounds(t, s)
                unit = f" [{s.unit}]" if s.unit else ""
                tail = f" -- {sb}" if sb else ""
                L.append("/*")
                L.extend(_wrap(f"{s.doc or '(no doc)'}{unit}{tail}", " * "))
                L.append(" */")
                L.append(f"#define {base}_{_ident(s.name)}_START {s.start}")
                L.append(f"#define {base}_{_ident(s.name)}_LEN {s.len}")
        L.append("")
    L.append(f"#endif /* {guard} */")
    return "\n".join(L) + "\n"


def gen_agent_c(cfg: AgentConfig, init) -> str:
    first_obs = None
    for t in init.obs:
        if t.slices and t.dtype == "f32":
            first_obs = t
            break
    if first_obs is None and init.obs:
        first_obs = init.obs[0]
    example = "     *   (no observations declared)"
    if first_obs is not None:
        base = f"OBS_{_ident(first_obs.name)}"
        if first_obs.slices:
            s = first_obs.slices[0]
            sn = f"{base}_{_ident(s.name)}"
            example = (
                f"     *   const uint8_t *blob = view.values[{base}_INDEX].ptr;\n"
                f"     *   float {s.name}[{sn}_LEN];\n"
                f"     *   wire_read_f32(blob + 4 * {sn}_START, {sn}_LEN, {s.name});"
            )
        else:
            example = f"     *   const wire_blob_t *blob = &view.values[{base}_INDEX];"
    return f'''/* YOUR agent — written once by `task create-agent`, never overwritten.
 *
 * A hand-written wasm agent for {cfg.env} [{cfg.mode}]: no ONNX, no
 * training — `on-tick` decodes the view with the vendored wire reader
 * (wire.c/wire.h, tested against the spec goldens) and answers with
 * whatever action you compute. Out of the box it answers the NEUTRAL
 * action.
 *
 * Build:  task build AGENT={cfg.name}   (wit-bindgen c + wasi-sdk clang
 *                                        → wasm32-wasip2 component)
 * Match:  task match AGENT={cfg.name}
 *
 * interface.h (generated) names every declared observation, slice and
 * action — read by name, never by magic index.
 */

#include <stdlib.h>
#include <string.h>

#include "gen/agent.h"
#include "interface.h"
#include "wire.h"

/* Everything worth precomputing once, at init. */
static wire_seat_init_t g_init;
static int g_ready = 0;

void exports_agent_init(agent_seat_init_payload_t *init_state) {{
    g_ready = wire_seat_init_decode(init_state->ptr, init_state->len, &g_init) == WIRE_OK;
    agent_seat_init_payload_free(init_state);
}}

void exports_agent_on_tick(agent_view_payload_t *view_bytes, agent_input_payload_t *ret) {{
    /* An empty input tells the engine to play the neutral action — the
     * safe answer whenever anything is off. */
    ret->ptr = NULL;
    ret->len = 0;

    if (!g_ready) {{
        agent_view_payload_free(view_bytes);
        return;
    }}
    wire_view_t view;
    if (wire_view_decode(view_bytes->ptr, view_bytes->len, &view) != WIRE_OK) {{
        agent_view_payload_free(view_bytes);
        return;
    }}

    /* Read observations by name via interface.h, e.g.:
{example}
     */

    /* Neutral until you make it yours: fill each action's f32s and
     * encode. wire_neutral_f32 gives the per-element bounds midpoint. */
    wire_input_builder_t b;
    wire_input_builder_start(&b, g_init.n_actions);
    for (uint32_t i = 0; i < g_init.n_actions; i++) {{
        const wire_value_spec_t *spec = &g_init.actions[i];
        float *vals = malloc(sizeof(float) * spec->numel);
        wire_neutral_f32(spec, vals);
        wire_input_builder_push_f32(&b, spec, vals);
        free(vals);
    }}
    wire_input_builder_finish(&b, &ret->ptr, &ret->len);

    wire_view_free(&view);
    agent_view_payload_free(view_bytes);
}}
'''


def gen_c_build_sh(cfg: AgentConfig) -> str:
    return f"""#!/usr/bin/env bash
# YOUR build script — written once by `task create-agent`. `task build
# AGENT={cfg.name}` runs it with WASI_SDK + WIT_BINDGEN on PATH resolved by
# the template (task setup LANGS=c provisions them).
set -euo pipefail
cd "$(dirname "$0")"

WASI_SDK="${{WASI_SDK:-/opt/wasi-sdk}}"
CLANG="$WASI_SDK/bin/clang"
[ -x "$CLANG" ] || {{ echo "no wasi-sdk at $WASI_SDK — run: task setup LANGS=c" >&2; exit 1; }}

# 1. World bindings (gen/agent.h, gen/agent.c, gen/agent_component_type.o).
WIT_BINDGEN="${{WIT_BINDGEN:-wit-bindgen}}"
"$WIT_BINDGEN" c wit --world agent --out-dir gen

# 2. One clang line: C → wasm32-wasip2 component (reactor: no main).
mkdir -p out
"$CLANG" --target=wasm32-wasip2 -mexec-model=reactor -O2 \\
  -o out/agent.wasm \\
  agent.c wire.c gen/agent.c gen/agent_component_type.o

echo "→ out/agent.wasm (wasm component)"
"""


def scaffold_c(cfg: AgentConfig, init, budgets, title: str) -> None:
    d = cfg.dir
    d.mkdir(parents=True, exist_ok=True)
    if (d / "wit").exists():
        shutil.rmtree(d / "wit")
    shutil.copytree(WIT_DIR, d / "wit")
    for name in ("wire.h", "wire.c"):
        shutil.copyfile(REFERENCE / "c-wire" / name, d / name)
    (d / "interface.h").write_text(gen_interface_h(cfg, init, budgets, title))
    print(f"→ {d / 'interface.h'}  (generated)", file=sys.stderr)
    for path, content, mode in (
        (d / "agent.c", gen_agent_c(cfg, init), 0o644),
        (d / "build.sh", gen_c_build_sh(cfg), 0o755),
    ):
        if path.exists():
            print(f"✓ {path} untouched (yours)", file=sys.stderr)
        else:
            path.write_text(content)
            path.chmod(mode)
            print(f"→ {path}  (yours — edit it)", file=sys.stderr)
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("gen/\nout/\n")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

GENERATORS = {
    "python": scaffold_python,
    "rust": scaffold_rust,
    "c": scaffold_c,
}


def check_toolchain(lang: str) -> None:
    """create-agent NEVER installs anything: fail fast with the exact fix."""
    if lang == "rust":
        import shutil as _sh
        import subprocess

        if not _sh.which("cargo"):
            raise SystemExit(
                "LANG=rust needs the Rust toolchain — run: task setup LANGS=rust"
            )
        rustup = _sh.which("rustup")
        if rustup:
            out = subprocess.run(
                [rustup, "target", "list", "--installed"], capture_output=True, text=True
            ).stdout
            if "wasm32-wasip2" not in out:
                raise SystemExit(
                    "LANG=rust needs the wasm32-wasip2 target — run: task setup LANGS=rust"
                )
    if lang == "c":
        from .toolchain import find_wasi_sdk, find_wit_bindgen

        if find_wasi_sdk() is None or find_wit_bindgen() is None:
            raise SystemExit("LANG=c needs wasi-sdk + wit-bindgen — run: task setup LANGS=c")


def main(argv: list[str] | None = None) -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="agent name (NAME= on the task line)")
    p.add_argument("--env", required=True, help="environment slug (ENV=)")
    p.add_argument("--mode", default=None, help="mode key (MODE=); default the release's")
    p.add_argument("--lang", default="python", choices=LANGS, help="LANG=; default python")
    p.add_argument(
        "--force",
        action="store_true",
        help="allow re-scaffolding under a different env/mode/lang (a name is "
        "an identity — without this, a mismatch refuses)",
    )
    args = p.parse_args(argv)

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name):
        raise SystemExit(f"NAME={args.name!r}: use letters, digits, - and _")

    existing_toml = Path("agents") / args.name / "agent.toml"
    if existing_toml.is_file():
        prev = load_agent(args.name)
        want_mode = args.mode or prev.mode
        if (prev.env, prev.lang) != (args.env, args.lang) or (
            args.mode and args.mode != prev.mode
        ):
            if not args.force:
                raise SystemExit(
                    f"agents/{args.name} already exists as {prev.env}/{prev.mode} "
                    f"[{prev.lang}]; asked for {args.env}/{want_mode} [{args.lang}]. "
                    "A name is an identity — pick a new NAME=, or pass FORCE=1 to "
                    "re-scaffold in place (generated files refresh; your policy "
                    "files are still never touched)."
                )

    check_toolchain(args.lang)

    paths = ensure_engine(args.env, args.mode)
    init, budgets, title = read_declaration(paths)
    cfg = AgentConfig(
        name=args.name,
        env=paths.env,
        mode=paths.mode,
        lang=args.lang,
        environment_version=budgets.environment_version or paths.version,
        payload_schema_version=budgets.payload_schema_version or 0,
    )

    GENERATORS[args.lang](cfg, init, budgets, title)
    write_agent_toml(cfg)
    print(f"→ {cfg.dir / 'agent.toml'}  (generated)", file=sys.stderr)
    print(
        f"\ncreate-agent: {cfg.name} ready — {cfg.env} [{cfg.mode}] in {cfg.lang}.",
        file=sys.stderr,
    )
    next_cmd = "train" if cfg.lang == "python" else "build"
    print(
        f"next: edit {cfg.dir}/  then  task {next_cmd} AGENT={cfg.name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
