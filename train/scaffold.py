"""``task create-agent`` — scaffold ``agents/<name>/`` from the engine's own
declaration.

The engine is the single source of what an environment IS. This module
fetches the (env, mode) release into the keyed cache, decodes seat 0's
``SeatInit`` (every observation and action, every documented slice and
column with its doc/unit/bounds, the goal/reward/ends brief, the metrics,
the tags, the budgets) and writes a ready-to-edit agent project for the
language of your choice::

    python -m train.scaffold --name my-bot --env <slug> [--mode M] [--lang python|rust|c]

Two kinds of files, and the contract between them:

- GENERATED (refreshed on every re-run): the interface file
  (``interface.py`` / ``src/interface.rs`` / ``interface.h``) and
  ``agent.toml``. Regenerate freely — after a release bump, say — they
  carry no hand edits.
- YOURS (written once, NEVER touched again): the policy stub
  (``policy.py`` / ``src/lib.rs`` / ``agent.c``), the network
  (``model.py``, python only) and the build files. The stub reads one
  observation through the typed accessor and answers the neutral action
  through the typed builder out of the box.

The three languages are at parity. Every interface file carries the same
header (goal / reward / ends, budgets, every metric with unit and
direction, tags, reward lag, the neutral-action rule) and, per value:
name, dtype, shape, element count, bounds (per element when declared) and
the doc; per slice and per column an index constant with its doc, unit
and any code table verbatim. Typed access is the language's own:
Python gets declaration-shaped numpy views and ``action(**by_name)``;
Rust a generated ``Obs<'a>`` and an ``Action`` struct with ``encode()``;
C typed accessor functions and an action struct with ``encode``.

Re-running with the same name refuses a different env/mode/lang unless
``--force`` — a name is an identity, not a slot.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
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

#: The documented feedback-window column names (docs/wire.md, *Data
#: domains*) — what the supervised block is scaffolded from.
FEEDBACK_AGE = "age_ticks"
FEEDBACK_DECISION = "your_decision"
FEEDBACK_TRUTH = "truth"
FEEDBACK_VALID = "valid"


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


def _lower(name: str) -> str:
    return _ident(name).lower()


def _fmt_f(v: float) -> str:
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:g}"


def _py_float(v: float) -> str:
    """A Python float literal (``inf`` spelled so the module executes)."""
    if v == float("inf"):
        return 'float("inf")'
    if v == float("-inf"):
        return 'float("-inf")'
    return repr(float(v))


def _rs_float(v: float) -> str:
    if v == float("inf"):
        return "f32::INFINITY"
    if v == float("-inf"):
        return "f32::NEG_INFINITY"
    s = repr(float(v))
    if "e" in s or "E" in s:
        return f"{float(v):e}".replace("e+", "e") + "_f32"
    return s if "." in s else s + ".0"


def _c_float(v: float) -> str:
    if v == float("inf"):
        return "INFINITY"
    if v == float("-inf"):
        return "(-INFINITY)"
    s = repr(float(v))
    if "e" in s or "E" in s:
        return f"{float(v):e}f"
    return (s if "." in s else s + ".0") + "f"


#: Comment text wraps at this many columns NOT counting the language's
#: comment prefix, so one doc breaks at the same words in every language.
TEXT_WIDTH = 72


def _wrap(text: str, prefix: str, width: int = TEXT_WIDTH) -> list[str]:
    out: list[str] = []
    for para in text.splitlines() or [""]:
        if not para.strip():
            out.append(prefix.rstrip())
            continue
        words = para.split()
        cur = ""
        for w in words:
            if cur and len(cur) + len(w) > width:
                out.append((prefix + cur).rstrip())
                cur = ""
            cur += w + " "
        out.append((prefix + cur).rstrip())
    return out


def _doc_lines(text: str, prefix: str) -> list[str]:
    """A doc as comment lines. A one-line doc is wrapped; a doc that
    carries its own line breaks — a code table — is kept VERBATIM, line
    for line, so the table reads exactly as the engine wrote it."""
    if not text:
        return [prefix + "(the engine declares no doc)"]
    if "\n" in text.strip():
        return [(prefix + line).rstrip() for line in text.strip().splitlines()]
    return _wrap(text, prefix)


def _with_unit(doc: str, unit: str, tail: str = "") -> str:
    """``doc [unit] — tail`` for a one-line doc; for a doc that carries its
    own lines (a code table) the unit and tail follow the FIRST line and
    the table stays verbatim below it."""
    doc = doc or "(no doc)"
    suffix = (f" [{unit}]" if unit else "") + (f" — {tail}" if tail else "")
    if "\n" in doc.strip():
        first, rest = doc.strip().split("\n", 1)
        return f"{first}{suffix}\n{rest}"
    return f"{doc}{suffix}"


def _bounds_line(t) -> str:
    if t.elem_bounds is not None:
        lo, hi = t.elem_bounds
        return (
            f"per-element bounds (see below): overall [{_fmt_f(float(min(lo)))}, "
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


def _shape_str(t) -> str:
    return f"{t.dtype}[{', '.join(str(int(d)) for d in t.shape)}]"


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
            "memory cap",
            f"{budgets.agent_memory_cap_bytes / (1024 * 1024):g} MB for the agent bundle"
            if getattr(budgets, "agent_memory_cap_bytes", None)
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


def _metric_lines(init) -> list[str]:
    """One line per declared metric: key — doc (unit, direction, headline)."""
    if not init.metrics:
        return ["(the engine declares no metrics; the report shows its raw keys)"]
    ordered = init.headline_metrics() + [m for m in init.metrics if not m.headline]
    direction = {"higher": "higher is better", "lower": "lower is better", "neutral": "neutral"}
    out = []
    for m in ordered:
        q = [m.unit] if m.unit else []
        q.append(direction.get(m.direction, m.direction))
        if m.headline:
            q.append("headline")
        if m.is_scenario:
            q.append("scenario: employers only")
        out.append(f"{m.key} — {m.doc or '(no doc)'} ({', '.join(q)})")
    return out


def _neutral_words(t) -> str:
    neutral = t.neutral_f32()
    if len(neutral) == 0:
        return ""
    vals = [round(float(v)) if t.dtype != "f32" else float(v) for v in neutral]
    if all(v == vals[0] for v in vals):
        return f"{t.name} → every element {_fmt_f(float(vals[0]))}"
    return f"{t.name} → per-element bounds midpoint"


def _reward_lag(init) -> str | None:
    return init.meta_get("reward_lag_ticks")


@dataclass
class _Section:
    """One header section as plain lines; each language prefixes them."""

    title: str
    lines: list[str]


def header_sections(init, budgets, title: str) -> list[_Section]:
    """The header every interface file carries, language-neutral."""
    sections = [
        _Section("GOAL", _doc_lines(init.brief.goal, "")),
        _Section("REWARD", _doc_lines(init.brief.reward, "")),
        _Section("EPISODE END", _doc_lines(init.brief.ends, "")),
        _Section("HOW YOU ARE SCORED", _metric_lines(init)),
    ]
    bl = _budget_lines(budgets)
    if bl:
        sections.append(_Section("BUDGETS", bl))
    tags = init.tags()
    if tags:
        sections.append(_Section("TAGS", [", ".join(tags)]))
    lag = _reward_lag(init)
    if lag:
        sections.append(
            _Section(
                "REWARD LAG",
                [f"{lag} ticks after a decision its truth reaches the reward (meta.reward_lag_ticks)"],
            )
        )
    neutral = [
        "Out-of-range values are clamped into bounds. A missing, malformed or",
        "non-finite input plays the NEUTRAL action — the midpoint of each",
        "element's bounds, or 0 when a bound is open (u8/i32 round it) — and",
        "counts as a missed tick. Ties: the action's own doc names the rule.",
    ]
    for t in init.actions:
        words = _neutral_words(t)
        if words:
            neutral.append(f"neutral: {words}")
    sections.append(_Section("NEUTRAL ACTION", neutral))
    meta = [(k, v) for k, v in init.meta if k not in ("tags", "reward_lag_ticks")]
    if meta:
        sections.append(_Section("META", [f"{k} = {v}" for k, v in meta]))
    return sections


# ---------------------------------------------------------------------------
# Action fields — the typed builder's named arguments, shared by all three
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionField:
    """One named argument of the typed action builder: a slice of an
    action (when it declares slices) or a whole action value."""

    name: str
    action_index: int
    start: int
    stop: int
    doc: str
    unit: str

    @property
    def len(self) -> int:
        return self.stop - self.start


def action_fields(init) -> list[ActionField]:
    fields: list[ActionField] = []
    for i, t in enumerate(init.actions):
        if t.slices:
            for s in sorted(t.slices, key=lambda x: x.start):
                fields.append(ActionField(s.name, i, s.start, s.stop, s.doc, s.unit))
        else:
            fields.append(ActionField(t.name, i, 0, t.numel, t.doc, ""))
    # Slice names may repeat across actions (two arms, each with `wrist`):
    # a colliding name gets its action's name in front.
    seen: dict[str, int] = {}
    for f in fields:
        seen[_lower(f.name)] = seen.get(_lower(f.name), 0) + 1
    out = []
    for f in fields:
        if seen[_lower(f.name)] > 1:
            f = ActionField(
                f"{init.actions[f.action_index].name}_{f.name}", f.action_index, f.start, f.stop, f.doc, f.unit
            )
        out.append(f)
    return out


def _elem_type(dtype: str, lang: str) -> str:
    return {
        "python": {"f32": "np.float32", "u8": "np.uint8", "i32": "np.int32"},
        "rust": {"f32": "f32", "u8": "u8", "i32": "i32"},
        "c": {"f32": "float", "u8": "uint8_t", "i32": "int32_t"},
    }[lang][dtype]


def _neutral_values(t) -> list:
    vals = t.neutral_f32()
    if t.dtype == "f32":
        return [float(v) for v in vals]
    return [int(round(float(v))) for v in vals]


def _lit(v, dtype: str, lang: str) -> str:
    if dtype == "f32":
        return {"python": repr(float(v)), "rust": _rs_float(v), "c": _c_float(v)}[lang]
    return str(int(v))


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
    L.append("constants and typed accessors so your policy reads observations by")
    L.append("name, never by magic index.")
    for sec in header_sections(init, budgets, title):
        L.append("")
        L.append(sec.title)
        for line in sec.lines:
            L.extend(_wrap(line, "  "))
    L.append('"""')
    L.append("")
    L.append("from __future__ import annotations")
    L.append("")
    L.append("import numpy as np")
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
            L.append(f"# ── {t.name} ── {_shape_str(t)} — {t.numel} element(s), {_bounds_line(t)}")
            L.extend(_doc_lines(t.doc, "# "))
            if t.dtype == "u8" and kind == "OBS":
                L.append("# u8: the Gymnasium env hands you raw 0..255 uint8; the graph input is")
                L.append("# f32 ÷ 255 (the shell normalizes at match time, training divides on the way in).")
            L.append(f'{base} = "{t.name}"')
            L.append(f"{base}_INDEX = {i}")
            L.append(f'{base}_DTYPE = "{t.dtype}"')
            L.append(f"{base}_SHAPE = {tuple(int(d) for d in t.shape)}")
            L.append(f"{base}_LEN = {t.numel}")
            L.append(f"{base}_LOW = {_py_float(t.low)}")
            L.append(f"{base}_HIGH = {_py_float(t.high)}")
            if t.elem_bounds is not None:
                lo, hi = t.elem_bounds
                L.append(f"{base}_ELEM_LOW = [{', '.join(_py_float(float(v)) for v in lo)}]")
                L.append(f"{base}_ELEM_HIGH = [{', '.join(_py_float(float(v)) for v in hi)}]")
            if len(t.shape) >= 2:
                L.append(f"{base}_ROWS = {int(t.shape[-2])}")
                L.append(f"{base}_COLS = {int(t.shape[-1])}")
            for s in sorted(t.slices, key=lambda x: x.start):
                L.extend(_doc_lines(_with_unit(s.doc, s.unit, _slice_bounds(t, s)), "# "))
                L.append(f"{base}_{_ident(s.name)} = slice({s.start}, {s.stop})")
            if t.columns:
                L.append(f"# columns of {t.name} — the index along the last axis:")
                for ci, c in enumerate(t.columns):
                    L.extend(_doc_lines(_with_unit(c.doc, c.unit), "# "))
                    L.append(f"{base}_COL_{_ident(c.name)} = {ci}")
        L.append("")

    # ── typed observation view ──
    L.append("#" + " " + "─" * 74)
    L.append("# Typed access — declaration-shaped numpy views of one tick's obs dict.")
    L.append("#" + " " + "─" * 74)
    L.append("")
    L.append("")
    L.append("class Obs:")
    L.append('    """One tick\'s observation dict (what the env returns) as named,')
    L.append("    declaration-shaped arrays: ``Obs(obs).<value>``, and per slice /")
    L.append('    column ``Obs(obs).<value>_<slice>`` / ``Obs(obs).<value>_<column>``."""')
    L.append("")
    L.append('    __slots__ = ("_d",)')
    L.append("")
    L.append("    def __init__(self, obs: dict):")
    L.append("        self._d = obs")
    for t in init.obs:
        base = f"OBS_{_ident(t.name)}"
        prop = _lower(t.name)
        L.append("")
        L.append("    @property")
        L.append(f"    def {prop}(self) -> np.ndarray:")
        L.append(f'        """{_shape_str(t)} — {(t.doc or t.name).splitlines()[0]}"""')
        L.append(
            f"        return np.asarray(self._d[{base}], dtype={_elem_type(t.dtype, 'python')})"
            f".reshape({base}_SHAPE)"
        )
        for s in sorted(t.slices, key=lambda x: x.start):
            L.append("")
            L.append("    @property")
            L.append(f"    def {prop}_{_lower(s.name)}(self) -> np.ndarray:")
            L.append(f'        """{(s.doc or s.name).splitlines()[0]}{f" [{s.unit}]" if s.unit else ""}"""')
            L.append(f"        return self.{prop}[{base}_{_ident(s.name)}]")
        for c in t.columns:
            L.append("")
            L.append("    @property")
            L.append(f"    def {prop}_{_lower(c.name)}(self) -> np.ndarray:")
            L.append(f'        """column {c.name} of every row — {(c.doc or c.name).splitlines()[0]}{f" [{c.unit}]" if c.unit else ""}"""')
            L.append(f"        return self.{prop}[..., {base}_COL_{_ident(c.name)}]")
    L.append("")
    L.append("")

    # ── typed action builder ──
    fields = action_fields(init)
    L.append("#" + " " + "─" * 74)
    L.append("# Typed action builder — declared units in, the env's action out.")
    L.append("#" + " " + "─" * 74)
    L.append("")
    neutral = [_neutral_values(t) for t in init.actions]
    L.append("# The neutral action per declared action value — what the engine plays")
    L.append("# for a missing/malformed input (each element's bounds midpoint, else 0).")
    L.append(f"NEUTRAL_ACTION = {neutral!r}")
    L.append("")
    L.append("_ACT_SPECS = [")
    for t in init.actions:
        lo, hi = t.bounds_arrays()
        L.append(
            f"    (\"{t.name}\", {_elem_type(t.dtype, 'python')}, {tuple(int(d) for d in t.shape)}, "
            f"[{', '.join(_py_float(float(v)) for v in lo)}], "
            f"[{', '.join(_py_float(float(v)) for v in hi)}]),"
        )
    L.append("]")
    L.append("_ACT_FIELDS = {")
    for f in fields:
        L.append(f'    "{_lower(f.name)}": ({f.action_index}, {f.start}, {f.stop}),')
    L.append("}")
    L.append("")
    L.append("")
    sig = ", ".join(f"{_lower(f.name)}=None" for f in fields)
    L.append(f"def action(*, {sig}):" if fields else "def action():")
    L.append('    """One action in DECLARED units, by name — every field defaults to')
    L.append("    its neutral value. Returns what ``env.step`` takes: the single")
    L.append('    declared action as an array, or a dict by action name when several."""')
    L.append("    given = {")
    for f in fields:
        L.append(f'        "{_lower(f.name)}": {_lower(f.name)},')
    L.append("    }")
    L.append("    arrays = [np.array(n, dtype=np.float32) for n in NEUTRAL_ACTION]")
    L.append("    for key, value in given.items():")
    L.append("        if value is None:")
    L.append("            continue")
    L.append("        i, start, stop = _ACT_FIELDS[key]")
    L.append("        arrays[i][start:stop] = np.asarray(value, dtype=np.float32).reshape(stop - start)")
    L.append("    out = [a.astype(dtype).reshape(shape) for a, (_, dtype, shape, _, _) in zip(arrays, _ACT_SPECS)]")
    L.append("    if len(out) == 1:")
    L.append("        return out[0]")
    L.append("    return {name: a for (name, _, _, _, _), a in zip(_ACT_SPECS, out)}")
    L.append("")
    L.append("")
    L.append(f"def normalized(*, {sig}) -> np.ndarray:" if fields else "def normalized() -> np.ndarray:")
    L.append('    """The same action as the ONNX OUTPUT convention: one flat float32')
    L.append("    array in [-1, 1] (every declared action's elements, in order), which")
    L.append("    the shell maps affinely back onto the bounds above — so a scripted")
    L.append('    policy can return exactly the values it names here. Missing = neutral."""')
    L.append("    given = {")
    for f in fields:
        L.append(f'        "{_lower(f.name)}": {_lower(f.name)},')
    L.append("    }")
    L.append("    parts = []")
    L.append("    for i, (_, _, _, lo, hi) in enumerate(_ACT_SPECS):")
    L.append("        v = np.array(NEUTRAL_ACTION[i], dtype=np.float32)")
    L.append("        for key, value in given.items():")
    L.append("            j, start, stop = _ACT_FIELDS[key]")
    L.append("            if j == i and value is not None:")
    L.append("                v[start:stop] = np.asarray(value, dtype=np.float32).reshape(stop - start)")
    L.append("        lo = np.array(lo, dtype=np.float32)")
    L.append("        hi = np.array(hi, dtype=np.float32)")
    L.append("        bounded = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)")
    L.append("        span = np.where(bounded, hi - lo, 1.0)")
    L.append("        parts.append(np.clip(np.where(bounded, (v - lo) / span * 2.0 - 1.0, v), -1.0, 1.0))")
    L.append("    return np.concatenate(parts).astype(np.float32)")
    L.append("")
    return "\n".join(L)


def _first_obs_read_py(init) -> tuple[str, str]:
    """(comment, expression) reading one observation by typed accessor —
    the batched torch tensor in ``forward`` and the numpy view outside it."""
    if not init.obs:
        return "(no observations declared)", "None"
    t = init.obs[0]
    base = f"OBS_{_ident(t.name)}"
    if t.columns:
        c = t.columns[0]
        return (
            f"column `{c.name}` of every `{t.name}` row: {(c.doc or c.name).splitlines()[0]}",
            f"inputs[iface.{base}_INDEX][..., iface.{base}_COL_{_ident(c.name)}]",
        )
    if t.slices:
        s = t.slices[0]
        return (
            f"`{t.name}[{s.name}]`: {(s.doc or s.name).splitlines()[0]}",
            f"inputs[iface.{base}_INDEX][:, iface.{base}_{_ident(s.name)}]",
        )
    return (
        f"`{t.name}`: {(t.doc or t.name).splitlines()[0]}",
        f"inputs[iface.{base}_INDEX]",
    )


def gen_policy_py(cfg: AgentConfig, init) -> str:
    comment, expr = _first_obs_read_py(init)
    fields = action_fields(init)
    example_field = _lower(fields[0].name) if fields else None
    act = init.actions[0] if init.actions else None
    act_comment = (
        f"the `{act.name}` action: {act.numel} value(s) per row, each in [-1, 1]"
        if act
        else "the declared action"
    )
    builder_example = (
        f"iface.normalized({example_field}=...)" if example_field else "iface.normalized()"
    )
    return f'''"""YOUR policy — written once by `task create-agent`, never overwritten.

Two ways to make this agent yours:

1. WRITE IT BY HAND (no training): edit ``forward`` below. ``interface.py``
   names every observation, so you read values by name, not by index.
   ``task build AGENT={cfg.name}`` turns this module into the ONNX file your
   agent ships, checks that the ONNX file gives the same answers as this
   code, and packs the bundle you upload. As created it plays the neutral
   action (what the environment does for a seat that sends nothing), so it
   runs but does not try.

2. TRAIN IT: ``task train AGENT={cfg.name}`` (PPO) or ``task train
   AGENT={cfg.name} RECIPE=supervised`` trains the network in ``model.py``
   on the same observations and packs that instead. Training never touches
   this file.

Then play it: ``task match AGENT={cfg.name}``. Add ``LOGS=1`` to see, each
tick, every observation by name and the action it sent (the shell that runs
the exported graph prints them, and says why whenever it plays the neutral
action), and ``STEP=1`` to pause after every tick. To look inside the network
itself, run the environment in your own Python loop
(``gymnasium.make("Lockstep/Env-v0", ...)``) and print or use pdb there.

What the network must look like (checked when it plays): one input per
observation, named as declared, at the declared shape; one output with every
value in [-1, 1], which the match stretches onto each action's declared
range. u8 values (images) arrive divided by 255; i32 values arrive as int32.
"""

from __future__ import annotations

import numpy as np
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
        # The neutral action in the graph's own units ([-1, 1] per element),
        # built through the typed builder; name a field to change it, e.g.
        # {builder_example}
        self.neutral = nn.Parameter(
            torch.from_numpy(np.asarray(iface.normalized(), dtype=np.float32)), requires_grad=False
        )

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Inputs arrive batched, one tensor per declared observation, in
        ``self.input_names`` order. Return {act_comment}.
        """
        batch = inputs[0].shape[0]
        # One observation, read through the generated interface —
        # {comment}
        first = {expr}
        # Every input must reach the output or the exporter prunes it;
        # replace this with your own use of `first` and the rest.
        keep_alive = sum(t.float().mean() for t in inputs) * 0.0 + first.float().mean() * 0.0
        return self.neutral.expand(batch, -1) + keep_alive
'''


def _conv_example(name: str, t) -> list[str]:
    """A commented-out convolution stream for a rank-3 u8 value, sized to
    fit the declared image (channel-first, as the wire declares)."""
    c, h, w = (int(d) for d in t.shape)
    if min(h, w) >= 32:
        body = [
            f"        #     nn.Conv2d({c}, 16, kernel_size=8, stride=4), nn.ReLU(),",
            "        #     nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),",
        ]
    else:
        body = [
            f"        #     nn.Conv2d({c}, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),",
            "        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),",
        ]
    return [
        f"        # A convolution instead — `{name}` is image-shaped (channel-first {c}×{h}×{w}):",
        f'        # "{name}": nn.Sequential(',
        *body,
        "        #     nn.Flatten(),",
        "        # ),",
    ]


def gen_model_py(cfg: AgentConfig, init) -> str:
    L: list[str] = []
    L.append('"""YOUR network — written once by `task create-agent`, never overwritten.')
    L.append("")
    L.append(f"`task train AGENT={cfg.name}` builds the policy HERE — PPO and the")
    L.append("supervised recipe alike. One named stream per declared observation")
    L.append("(each takes that value's batched float tensor at its declared shape and")
    L.append("returns `[batch, width]`), then the template core adds the shared trunk")
    L.append("and heads (`train.core.policy.Policy`). The starter stream is the same")
    L.append("for every value whatever its dtype or rank: flatten → LayerNorm →")
    L.append("Linear → ReLU. Replace one line to give a value a different stream.")
    L.append("")
    L.append("What the tensors are when they reach a stream: f32 as declared; u8")
    L.append("already divided by 255 (the shell feeds the graph the same); i32 cast to")
    L.append("float inside the graph. Shapes are the declared ones, batch first.")
    L.append("")
    L.append("One thing to know about the starter: LayerNorm normalizes each sample")
    L.append("across its own elements, so a value with only a few elements loses its")
    L.append("absolute scale and mean on the way in. When those carry the signal, drop")
    L.append("the LayerNorm from that value's stream (`flat_stream(shape, norm=False)`).")
    L.append('"""')
    L.append("")
    L.append("from __future__ import annotations")
    L.append("")
    L.append("import torch.nn as nn  # noqa: F401 — for the streams you write here")
    L.append("")
    L.append("from train.core.policy import Policy, flat_stream")
    L.append("")
    L.append("")
    L.append("def build_policy(observation_space, action_space) -> Policy:")
    L.append('    """The network `task train` trains and `task build`/`task train` export."""')
    L.append("    streams = {")
    for t in init.obs:
        shape = tuple(int(d) for d in t.shape)
        doc = (t.doc or "(no doc)").splitlines()[0]
        note = ", arrives as f32 in [0, 1]" if t.dtype == "u8" else (", arrives as int32" if t.dtype == "i32" else "")
        L.append(f"        # {t.name} — {_shape_str(t)}{note}: {doc}")
        L.append(f'        "{t.name}": flat_stream({shape!r}),')
        if t.dtype == "u8" and len(shape) == 3:
            L.extend(_conv_example(t.name, t))
    L.append("    }")
    L.append("    return Policy(observation_space, action_space, streams)")
    L.append("")
    return "\n".join(L)


def scaffold_python(cfg: AgentConfig, init, budgets, title: str) -> None:
    d = cfg.dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "interface.py").write_text(gen_interface_py(cfg, init, budgets, title), encoding="utf-8")
    print(f"→ {d / 'interface.py'}  (generated)", file=sys.stderr)
    for path, content in (
        (d / "policy.py", gen_policy_py(cfg, init)),
        (d / "model.py", gen_model_py(cfg, init)),
    ):
        if path.exists():
            print(f"✓ {path} untouched (yours)", file=sys.stderr)
        else:
            path.write_text(content, encoding="utf-8")
            print(f"→ {path}  (yours — edit it)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------


def _rs_nested_type(elem: str, shape: tuple[int, ...]) -> str:
    """``[[f32; F]; T]`` for a declared shape (innermost axis last)."""
    ty = elem
    for d in reversed(shape):
        ty = f"[{ty}; {d}]"
    return ty


def _rs_obs_field(t) -> tuple[str, str]:
    """(field type, method return type) for one observation value."""
    if t.dtype == "u8":
        return "&'a [u8]", "&'a [u8]"
    elem = _elem_type(t.dtype, "rust")
    shape = tuple(int(d) for d in t.shape)
    if len(shape) == 0:
        return elem, elem
    if len(shape) <= 3:
        ty = _rs_nested_type(elem, shape)
        return ty, f"&{ty}"
    return f"Vec<{elem}>", f"&[{elem}]"


def _rs_decode_lines(t, var: str) -> list[str]:
    """Lines that decode `bytes` (the value's blob) into `var`."""
    elem = _elem_type(t.dtype, "rust")
    read = "le_f32" if t.dtype == "f32" else "le_i32"
    shape = tuple(int(d) for d in t.shape)
    if len(shape) == 0:
        return [f"        let {var} = {read}(&bytes[0..4]);"]
    if len(shape) > 3:
        return [f"        let {var}: Vec<{elem}> = bytes.chunks_exact(4).map({read}).collect();"]
    zero = "0f32" if t.dtype == "f32" else "0i32"
    init = zero
    for d in reversed(shape):
        init = f"[{init}; {d}]"
    lines = [
        f"        let mut {var} = {init};",
        "        {",
        f"            let mut it = bytes.chunks_exact(4).map({read});",
    ]
    indent = "            "
    target = var
    loops = []
    for depth, _ in enumerate(shape[:-1]):
        lines.append(f"{indent}for a{depth} in {target}.iter_mut() {{")
        target = f"a{depth}"
        loops.append(indent)
        indent += "    "
    lines.append(f"{indent}for x in {target}.iter_mut() {{")
    lines.append(f"{indent}    *x = it.next()?;")
    lines.append(f"{indent}}}")
    for ind in reversed(loops):
        lines.append(f"{ind}}}")
    lines.append("        }")
    return lines


def gen_interface_rs(cfg: AgentConfig, init, budgets, title: str) -> str:
    L: list[str] = []
    L.append("//! GENERATED by `task create-agent` — regenerated on re-run; do not edit.")
    L.append("//!")
    L.append(f"//! {title}")
    L.append("//!")
    L.append("//! The engine's OWN declaration as named constants and typed access:")
    L.append("//! `Obs::read(&view)` gives every observation in its natural type,")
    L.append("//! `Action { .. }.encode()` builds the input — read and write by name,")
    L.append("//! never by magic index.")
    for sec in header_sections(init, budgets, title):
        L.append("//!")
        L.append(f"//! # {sec.title.title()}")
        for line in sec.lines:
            L.extend(_wrap(line, "//! "))
    L.append("")
    L.append("#![allow(dead_code)]")
    L.append("")
    L.append("use crate::wire;")
    L.append("")
    L.append(f'pub const ENV: &str = "{cfg.env}";')
    L.append(f'pub const MODE: &str = "{cfg.mode}";')
    L.append(f"pub const PAYLOAD_SCHEMA_VERSION: u32 = {cfg.payload_schema_version};")
    L.append("")
    for kind, values, modname in (("obs", init.obs, "obs"), ("act", init.actions, "act")):
        what = "observation" if kind == "obs" else "action"
        L.append(f"/// Every declared {what}, in wire order (blob i of")
        if kind == "obs":
            L.append("/// a `View` is the i-th module below).")
        else:
            L.append("/// an `Input` is the i-th module below).")
        L.append(f"pub mod {modname} {{")
        for i, t in enumerate(values):
            m = _lower(t.name)
            L.extend(_doc_lines(t.doc, "    /// "))
            L.append("    ///")
            L.append(f"    /// {_shape_str(t)} — {t.numel} element(s), {_bounds_line(t)}")
            L.append(f"    pub mod {m} {{")
            L.append(f'        pub const NAME: &str = "{t.name}";')
            L.append(f"        pub const INDEX: usize = {i};")
            L.append(f'        pub const DTYPE: &str = "{t.dtype}";')
            L.append(f"        pub const LEN: usize = {t.numel};")
            shape = ", ".join(str(int(x)) for x in t.shape)
            L.append(f"        pub const SHAPE: [u32; {len(t.shape)}] = [{shape}];")
            L.append(f"        pub const LOW: f32 = {_rs_float(t.low)};")
            L.append(f"        pub const HIGH: f32 = {_rs_float(t.high)};")
            if t.elem_bounds is not None:
                lo, hi = t.elem_bounds
                L.append(
                    f"        pub const ELEM_LOW: [f32; {t.numel}] = "
                    f"[{', '.join(_rs_float(float(v)) for v in lo)}];"
                )
                L.append(
                    f"        pub const ELEM_HIGH: [f32; {t.numel}] = "
                    f"[{', '.join(_rs_float(float(v)) for v in hi)}];"
                )
            if len(t.shape) >= 2:
                L.append(f"        pub const ROWS: usize = {int(t.shape[-2])};")
                L.append(f"        pub const COLS: usize = {int(t.shape[-1])};")
            for s in sorted(t.slices, key=lambda x: x.start):
                L.extend(_doc_lines(_with_unit(s.doc, s.unit, _slice_bounds(t, s)), "        /// "))
                L.append(
                    f"        pub const {_ident(s.name)}: core::ops::Range<usize> = "
                    f"{s.start}..{s.stop};"
                )
            for ci, c in enumerate(t.columns):
                L.extend(_doc_lines(f"column {c.name}: " + _with_unit(c.doc, c.unit), "        /// "))
                L.append(f"        pub const COL_{_ident(c.name)}: usize = {ci};")
            L.append("    }")
        L.append("}")
        L.append("")

    L.append("fn le_f32(b: &[u8]) -> f32 {")
    L.append("    f32::from_le_bytes([b[0], b[1], b[2], b[3]])")
    L.append("}")
    L.append("")
    L.append("fn le_i32(b: &[u8]) -> i32 {")
    L.append("    i32::from_le_bytes([b[0], b[1], b[2], b[3]])")
    L.append("}")
    L.append("")
    L.append("/// Little-endian f32s from a raw blob (a `View` value's bytes).")
    L.append("pub fn f32s(bytes: &[u8]) -> Vec<f32> {")
    L.append("    bytes.chunks_exact(4).map(le_f32).collect()")
    L.append("}")
    L.append("")
    L.append("/// Little-endian i32s from a raw blob (an i32 value's bytes).")
    L.append("pub fn i32s(bytes: &[u8]) -> Vec<i32> {")
    L.append("    bytes.chunks_exact(4).map(le_i32).collect()")
    L.append("}")
    L.append("")

    # ── Obs ──
    L.append("/// One tick's observation, every declared value in its natural type")
    L.append("/// (f32/i32 values decoded into declaration-shaped arrays; u8 values")
    L.append("/// borrowed from the view as bytes, row-major).")
    # `'a` is the borrow of the view's bytes; only u8 values borrow, so a
    # declaration without one needs the lifetime pinned to a marker field.
    borrows = any(t.dtype == "u8" for t in init.obs)
    L.append("pub struct Obs<'a> {")
    if not borrows:
        L.append("    _view: core::marker::PhantomData<&'a ()>,")
    for t in init.obs:
        field_ty, _ = _rs_obs_field(t)
        L.append(f"    {_lower(t.name)}: {field_ty},")
    L.append("}")
    L.append("")
    L.append("impl<'a> Obs<'a> {")
    L.append("    /// Decode a view; `None` when a value's byte length is not the")
    L.append("    /// declared one (the safe answer is then the neutral action).")
    L.append("    pub fn read(view: &wire::View<'a>) -> Option<Self> {")
    L.append(f"        if view.values.len() != {len(init.obs)} {{")
    L.append("            return None;")
    L.append("        }")
    for t in init.obs:
        var = _lower(t.name)
        L.append(f"        let bytes: &'a [u8] = view.values[obs::{var}::INDEX];")
        L.append(f"        if bytes.len() != {t.byte_len} {{")
        L.append("            return None;")
        L.append("        }")
        if t.dtype == "u8":
            L.append(f"        let {var} = bytes;")
        else:
            L.extend(_rs_decode_lines(t, var))
    L.append("        Some(Obs {")
    if not borrows:
        L.append("            _view: core::marker::PhantomData,")
    for t in init.obs:
        L.append(f"            {_lower(t.name)},")
    L.append("        })")
    L.append("    }")
    for t in init.obs:
        var = _lower(t.name)
        _, ret = _rs_obs_field(t)
        L.append("")
        L.extend(_doc_lines(f"`{t.name}` {_shape_str(t)} — {t.doc or '(no doc)'}", "    /// "))
        L.append(f"    pub fn {var}(&self) -> {ret} {{")
        if t.dtype == "u8":
            L.append(f"        self.{var}")
        elif len(t.shape) == 0:
            L.append(f"        self.{var}")
        else:
            L.append(f"        &self.{var}")
        L.append("    }")
    L.append("}")
    L.append("")

    # ── Action ──
    fields = action_fields(init)
    L.append("/// One action in DECLARED units, by name; `Action::neutral()` (also")
    L.append("/// `Default`) is what the engine plays for a missing input. `encode()`")
    L.append("/// is the bytes `on_tick` returns.")
    L.append("#[derive(Clone, Debug, PartialEq)]")
    L.append("pub struct Action {")
    for f in fields:
        t = init.actions[f.action_index]
        elem = _elem_type(t.dtype, "rust")
        ty = elem if f.len == 1 else f"[{elem}; {f.len}]"
        L.extend(_doc_lines(_with_unit(f.doc or f.name, f.unit), "    /// "))
        L.append(f"    pub {_lower(f.name)}: {ty},")
    L.append("}")
    L.append("")
    L.append("impl Action {")
    L.append("    pub fn neutral() -> Self {")
    L.append("        Action {")
    for f in fields:
        t = init.actions[f.action_index]
        vals = _neutral_values(t)[f.start : f.stop]
        lits = [_lit(v, t.dtype, "rust") for v in vals]
        L.append(f"            {_lower(f.name)}: {lits[0] if f.len == 1 else '[' + ', '.join(lits) + ']'},")
    L.append("        }")
    L.append("    }")
    L.append("")
    L.append("    /// The encoded `Input`: one blob per declared action, in wire order.")
    L.append("    pub fn encode(&self) -> Vec<u8> {")
    L.append("        let mut values: Vec<Vec<u8>> = Vec::new();")
    for i, t in enumerate(init.actions):
        elem = _elem_type(t.dtype, "rust")
        zero = {"f32": "0f32", "i32": "0i32", "u8": "0u8"}[t.dtype]
        L.append(f"        let mut v{i}: [{elem}; {t.numel}] = [{zero}; {t.numel}];")
        for f in fields:
            if f.action_index != i:
                continue
            if f.len == 1:
                L.append(f"        v{i}[{f.start}] = self.{_lower(f.name)};")
            else:
                L.append(f"        v{i}[{f.start}..{f.stop}].copy_from_slice(&self.{_lower(f.name)});")
        if t.dtype == "f32":
            L.append(f"        values.push(wire::f32_bytes(&v{i}));")
        elif t.dtype == "i32":
            L.append(f"        values.push(v{i}.iter().flat_map(|x| x.to_le_bytes()).collect());")
        else:
            L.append(f"        values.push(v{i}.to_vec());")
    L.append("        wire::encode_input(&values)")
    L.append("    }")
    L.append("}")
    L.append("")
    L.append("impl Default for Action {")
    L.append("    fn default() -> Self {")
    L.append("        Action::neutral()")
    L.append("    }")
    L.append("}")
    L.append("")
    return "\n".join(L)


def gen_lib_rs(cfg: AgentConfig, init) -> str:
    if init.obs:
        t = init.obs[0]
        m = _lower(t.name)
        if t.columns:
            c = t.columns[0]
            obs_example = (
                f"        // `{t.name}` row 0, column `{c.name}` — {(c.doc or c.name).splitlines()[0]}\n"
                f"        let _first = obs.{m}()[0][interface::obs::{m}::COL_{_ident(c.name)}];"
            )
        elif t.slices:
            s = t.slices[0]
            obs_example = (
                f"        // `{t.name}[{s.name}]` — {(s.doc or s.name).splitlines()[0]}\n"
                f"        let _first = &obs.{m}()[interface::obs::{m}::{_ident(s.name)}];"
            )
        else:
            obs_example = (
                f"        // `{t.name}` — {(t.doc or t.name).splitlines()[0]}\n"
                f"        let _first = obs.{m}();"
            )
    else:
        obs_example = "        // (no observations declared)"
    if init.obs:
        # One commented line showing how to log what the agent sees; the
        # CLI prefixes every line with the seat and tick.
        t0 = init.obs[0]
        # A u8 value is an image: print one byte, not the whole plane.
        whole_image = t0.dtype == "u8" and not t0.columns and not t0.slices
        label, expr = (f"{t0.name}[0]", "_first[0]") if whole_image else (_log_label(t0), "_first")
        obs_example += (
            f"\n        // eprintln!(\"{label} = {{:?}}\", {expr}); "
            f"// see it: task match AGENT={cfg.name} LOGS=1"
        )
    fields = action_fields(init)
    field_example = (
        f"        // Name a field to change it: Action {{ {_lower(fields[0].name)}: ..., ..Action::neutral() }}"
        if fields
        else "        // (no action fields declared)"
    )
    return f'''//! YOUR agent — written once by `task create-agent`, never overwritten.
//!
//! A hand-written agent for {cfg.env} [{cfg.mode}], compiled to
//! WebAssembly. Every tick `on_tick` receives what your seat observes and
//! returns your action: `wire.rs` unpacks the bytes, the generated
//! `interface.rs` gives every observation, slice, column and action its
//! name, and the code below decides. As created it plays the neutral action
//! (what the environment does for a seat that sends nothing), so it runs but
//! does not try.
//!
//! Build:  task build AGENT={cfg.name}
//! Match:  task match AGENT={cfg.name}
//! Debug:  task match AGENT={cfg.name} LOGS=1 STEP=1
//!         (prints whatever you `eprintln!`, tagged with seat and tick, and
//!         pauses after every tick; only on your machine — ranked matches
//!         discard it)

mod interface;
mod wire;

wit_bindgen::generate!({{
    path: "wit",
    world: "agent",
    generate_all,
}});

use interface::{{Action, Obs}};
use wire::SeatInit;

struct Agent;

/// Everything worth precomputing once, at `init`.
#[allow(dead_code)]
struct Plan {{
    /// The seat's declaration — bounds, docs, metrics — should you need
    /// it at tick time (the generated interface already knows the layout).
    init: SeatInit,
}}

static PLAN: std::sync::Mutex<Option<Plan>> = std::sync::Mutex::new(None);

impl Guest for Agent {{
    fn init(init_state: Vec<u8>) {{
        *PLAN.lock().unwrap() = SeatInit::decode(&init_state).ok().map(|init| Plan {{ init }});
    }}

    fn on_tick(view: Vec<u8>) -> Vec<u8> {{
        let guard = PLAN.lock().unwrap();
        let Some(_plan) = guard.as_ref() else {{
            return Vec::new(); // engine substitutes the neutral action
        }};
        let Ok(view) = wire::View::decode(&view) else {{
            return Vec::new();
        }};
        let Some(obs) = Obs::read(&view) else {{
            return Vec::new();
        }};

        // One observation, read through the typed accessor:
{obs_example}

        // The neutral action, through the typed builder — make it yours.
{field_example}
        Action::neutral().encode()
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
    (d / "src" / "interface.rs").write_text(gen_interface_rs(cfg, init, budgets, title), encoding="utf-8")
    print(f"→ {d / 'src' / 'interface.rs'}  (generated)", file=sys.stderr)
    for path, content in (
        (d / "Cargo.toml", gen_cargo_toml(cfg)),
        (d / "src" / "lib.rs", gen_lib_rs(cfg, init)),
    ):
        if path.exists():
            print(f"✓ {path} untouched (yours)", file=sys.stderr)
        else:
            path.write_text(content, encoding="utf-8")
            print(f"→ {path}  (yours — edit it)", file=sys.stderr)
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("target/\nout/\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------


def _c_array_dims(shape: tuple[int, ...]) -> str:
    return "".join(f"[{d}]" for d in shape)


def gen_interface_h(cfg: AgentConfig, init, budgets, title: str) -> str:
    guard = f"AGENT_{_ident(cfg.name)}_INTERFACE_H"
    L: list[str] = []
    L.append("/* GENERATED by `task create-agent` — regenerated on re-run; do not edit.")
    L.append(" *")
    L.append(f" * {title}")
    L.append(" *")
    L.append(" * The engine's OWN declaration as named constants and typed access:")
    L.append(" * obs_<value>() reads a value in its natural type, agent_action_t +")
    L.append(" * agent_action_encode() build the input — by name, never by magic index.")
    for sec in header_sections(init, budgets, title):
        L.append(" *")
        L.append(f" * {sec.title}")
        for line in sec.lines:
            L.extend(_wrap(line, " *   "))
    L.append(" */")
    L.append(f"#ifndef {guard}")
    L.append(f"#define {guard}")
    L.append("")
    L.append("#include <math.h>")
    L.append("#include <stddef.h>")
    L.append("#include <stdint.h>")
    L.append("#include <string.h>")
    L.append("")
    L.append('#include "wire.h"')
    L.append("")
    L.append("#if defined(__GNUC__) || defined(__clang__)")
    L.append("#define AGENT_UNUSED __attribute__((unused))")
    L.append("#else")
    L.append("#define AGENT_UNUSED")
    L.append("#endif")
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
            L.extend(_doc_lines(t.doc, " * "))
            L.append(f" * {_shape_str(t)} — {t.numel} element(s), {_bounds_line(t)}")
            L.append(" */")
            L.append(f'#define {base}_NAME "{t.name}"')
            L.append(f"#define {base}_INDEX {i}")
            c_dtype = {"f32": "WIRE_F32", "u8": "WIRE_U8", "i32": "WIRE_I32"}[t.dtype]
            L.append(f"#define {base}_DTYPE {c_dtype}")
            L.append(f"#define {base}_RANK {len(t.shape)}")
            L.append(f"#define {base}_LEN {t.numel}")
            shape = ", ".join(str(int(x)) for x in t.shape) or "0"
            L.append(
                f"static const uint32_t {base}_SHAPE[{max(1, len(t.shape))}] AGENT_UNUSED = {{{shape}}};"
            )
            L.append(f"#define {base}_LOW {_c_float(t.low)}")
            L.append(f"#define {base}_HIGH {_c_float(t.high)}")
            if t.elem_bounds is not None:
                lo, hi = t.elem_bounds
                L.append(
                    f"static const float {base}_ELEM_LOW[{t.numel}] AGENT_UNUSED = "
                    f"{{{', '.join(_c_float(float(v)) for v in lo)}}};"
                )
                L.append(
                    f"static const float {base}_ELEM_HIGH[{t.numel}] AGENT_UNUSED = "
                    f"{{{', '.join(_c_float(float(v)) for v in hi)}}};"
                )
            if len(t.shape) >= 2:
                L.append(f"#define {base}_ROWS {int(t.shape[-2])}")
                L.append(f"#define {base}_COLS {int(t.shape[-1])}")
            for s in sorted(t.slices, key=lambda x: x.start):
                L.append("/*")
                L.extend(_doc_lines(_with_unit(s.doc, s.unit, _slice_bounds(t, s)), " * "))
                L.append(" */")
                L.append(f"#define {base}_{_ident(s.name)}_START {s.start}")
                L.append(f"#define {base}_{_ident(s.name)}_LEN {s.len}")
            for ci, c in enumerate(t.columns):
                L.append("/*")
                L.extend(_doc_lines(f"column {c.name}: " + _with_unit(c.doc, c.unit), " * "))
                L.append(" */")
                L.append(f"#define {base}_COL_{_ident(c.name)} {ci}")
        L.append("")

    # ── typed accessors ──
    L.append("/* ── Typed access: one accessor per observation. Each checks the blob's")
    L.append(" * length against the declaration; a mismatch returns 0 / NULL (answer")
    L.append(" * the neutral action then). f32/i32 values are copied into a")
    L.append(" * declaration-shaped array; u8 values are returned as the borrowed")
    L.append(" * row-major bytes. ── */")
    for t in init.obs:
        base = f"OBS_{_ident(t.name)}"
        fn = f"obs_{_lower(t.name)}"
        L.append("")
        L.append("/*")
        L.extend(_doc_lines(f"{t.name} {_shape_str(t)} — {t.doc or '(no doc)'}", " * "))
        L.append(" */")
        if t.dtype == "u8":
            L.append(f"static inline const uint8_t *{fn}(const wire_view_t *v) {{")
            L.append(f"    if (v->n_values <= {base}_INDEX || v->values[{base}_INDEX].len != {t.byte_len}) return NULL;")
            L.append(f"    return v->values[{base}_INDEX].ptr;")
            L.append("}")
            continue
        elem = _elem_type(t.dtype, "c")
        reader = "wire_read_f32" if t.dtype == "f32" else "wire_read_i32"
        shape = tuple(int(d) for d in t.shape)
        if len(shape) == 0:
            L.append(f"static inline int {fn}(const wire_view_t *v, {elem} *out) {{")
            flat = "out"
        elif len(shape) <= 3:
            L.append(f"static inline int {fn}(const wire_view_t *v, {elem} out{_c_array_dims(shape)}) {{")
            flat = "&out" + "[0]" * len(shape)
        else:
            L.append(f"static inline int {fn}(const wire_view_t *v, {elem} *out /* {base}_LEN, row-major */) {{")
            flat = "out"
        L.append(f"    if (v->n_values <= {base}_INDEX || v->values[{base}_INDEX].len != {t.byte_len}) return 0;")
        L.append(f"    {reader}(v->values[{base}_INDEX].ptr, {base}_LEN, {flat});")
        L.append("    return 1;")
        L.append("}")
    L.append("")

    # ── action struct ──
    fields = action_fields(init)
    L.append("/* ── One action in DECLARED units, by name. agent_action_neutral() fills")
    L.append(" * what the engine plays for a missing input; agent_action_encode()")
    L.append(" * mallocs the encoded input into (*ptr, *len) — exactly what on_tick")
    L.append(" * returns (the canonical ABI frees it). ── */")
    L.append("typedef struct {")
    for f in fields:
        t = init.actions[f.action_index]
        elem = _elem_type(t.dtype, "c")
        unit = f" [{f.unit}]" if f.unit else ""
        doc = (f.doc or f.name).splitlines()[0]
        decl = f"{elem} {_lower(f.name)};" if f.len == 1 else f"{elem} {_lower(f.name)}[{f.len}];"
        L.append(f"    {decl:<32} /* {doc}{unit} */")
    if not fields:
        L.append("    int _unused;")
    L.append("} agent_action_t;")
    L.append("")
    L.append("static inline void agent_action_neutral(agent_action_t *a) {")
    for f in fields:
        t = init.actions[f.action_index]
        vals = _neutral_values(t)[f.start : f.stop]
        if f.len == 1:
            L.append(f"    a->{_lower(f.name)} = {_lit(vals[0], t.dtype, 'c')};")
        else:
            for k, v in enumerate(vals):
                L.append(f"    a->{_lower(f.name)}[{k}] = {_lit(v, t.dtype, 'c')};")
    if not fields:
        L.append("    a->_unused = 0;")
    L.append("}")
    L.append("")
    L.append("static inline void agent_put_le32(uint8_t *dst, uint32_t bits) {")
    L.append("    dst[0] = (uint8_t)bits;")
    L.append("    dst[1] = (uint8_t)(bits >> 8);")
    L.append("    dst[2] = (uint8_t)(bits >> 16);")
    L.append("    dst[3] = (uint8_t)(bits >> 24);")
    L.append("}")
    L.append("")
    L.append("static inline void agent_action_encode(const agent_action_t *a, uint8_t **ptr, size_t *len) {")
    L.append("    wire_input_builder_t b;")
    L.append(f"    wire_input_builder_start(&b, {len(init.actions)});")
    for i, t in enumerate(init.actions):
        elem = _elem_type(t.dtype, "c")
        L.append("    {")
        L.append(f"        {elem} v[{t.numel}];")
        L.append(f"        uint8_t raw[{t.byte_len}];")
        L.append(f"        memset(v, 0, sizeof v);")
        for f in fields:
            if f.action_index != i:
                continue
            if f.len == 1:
                L.append(f"        v[{f.start}] = a->{_lower(f.name)};")
            else:
                L.append(f"        memcpy(&v[{f.start}], a->{_lower(f.name)}, sizeof a->{_lower(f.name)});")
        if t.dtype == "u8":
            L.append(f"        memcpy(raw, v, {t.byte_len});")
        else:
            L.append(f"        for (size_t k = 0; k < {t.numel}; k++) {{")
            L.append("            uint32_t bits;")
            L.append("            memcpy(&bits, &v[k], 4);")
            L.append("            agent_put_le32(raw + 4 * k, bits);")
            L.append("        }")
        L.append(f"        wire_input_builder_push_raw(&b, raw, {t.byte_len});")
        L.append("    }")
    L.append("    wire_input_builder_finish(&b, ptr, len);")
    L.append("}")
    L.append("")
    L.append(f"#endif /* {guard} */")
    return "\n".join(L) + "\n"


def _log_label(t) -> str:
    """How the scaffolds' example log line names the observation it reads —
    the same element the typed-accessor example picks."""
    if t.columns:
        return f"{t.name}[0].{t.columns[0].name}"
    if t.slices:
        return f"{t.name}[{t.slices[0].name}]"
    return t.name


def _c_log(cfg: AgentConfig, label: str, expr: str, dtype: str, indent: str) -> str:
    """One commented fprintf line for the C stub. u8/i32 print as ints,
    f32 as a float."""
    fmt, cast = ("%f", "(double)") if dtype == "f32" else ("%d", "(int)")
    return (
        f"{indent}/* fprintf(stderr, \"{label} = {fmt}\\n\", {cast}{expr});\n"
        f"{indent}   see it: task match AGENT={cfg.name} LOGS=1 */"
    )


def gen_agent_c(cfg: AgentConfig, init) -> str:
    if init.obs:
        t = init.obs[0]
        base = f"OBS_{_ident(t.name)}"
        fn = f"obs_{_lower(t.name)}"
        shape = tuple(int(d) for d in t.shape)
        elem = _elem_type(t.dtype, "c")
        label = _log_label(t)
        if t.dtype == "u8":
            example = (
                f"    /* `{t.name}` — {(t.doc or t.name).splitlines()[0]} */\n"
                f"    const uint8_t *{_lower(t.name)} = {fn}(&view);\n"
                f"    (void){_lower(t.name)};\n"
                + _c_log(cfg, f"{t.name}[0]", f"({_lower(t.name)} ? {_lower(t.name)}[0] : 0)", "u8", "    ")
            )
        elif len(shape) == 0:
            example = (
                f"    /* `{t.name}` — {(t.doc or t.name).splitlines()[0]} */\n"
                f"    {elem} {_lower(t.name)};\n"
                f"    if ({fn}(&view, &{_lower(t.name)})) {{\n"
                f"        (void){_lower(t.name)};\n"
                + _c_log(cfg, label, _lower(t.name), t.dtype, "        ")
                + "\n    }"
            )
        elif len(shape) <= 3:
            if t.columns:
                c = t.columns[0]
                use = (
                    f"        /* row 0, column `{c.name}` — {(c.doc or c.name).splitlines()[0]} */\n"
                    f"        {elem} first = {_lower(t.name)}[0]{'[0]' * (len(shape) - 2)}[{base}_COL_{_ident(c.name)}];\n"
                    "        (void)first;\n"
                    + _c_log(cfg, label, "first", t.dtype, "        ")
                )
            elif t.slices:
                s = t.slices[0]
                use = (
                    f"        /* `{t.name}[{s.name}]` — {(s.doc or s.name).splitlines()[0]} */\n"
                    f"        {elem} first = {_lower(t.name)}[{base}_{_ident(s.name)}_START];\n"
                    "        (void)first;\n"
                    + _c_log(cfg, f"{t.name}[{s.name}][0]", "first", t.dtype, "        ")
                )
            else:
                first_elem = f"{_lower(t.name)}" + "[0]" * len(shape)
                use = f"        (void){_lower(t.name)};\n" + _c_log(
                    cfg, f"{t.name}[0]", first_elem, t.dtype, "        "
                )
            example = (
                f"    /* `{t.name}` — {(t.doc or t.name).splitlines()[0]} */\n"
                f"    {elem} {_lower(t.name)}{_c_array_dims(shape)};\n"
                f"    if ({fn}(&view, {_lower(t.name)})) {{\n"
                f"{use}\n"
                f"    }}"
            )
        else:
            example = (
                f"    /* `{t.name}` — {(t.doc or t.name).splitlines()[0]} */\n"
                f"    {elem} {_lower(t.name)}[{base}_LEN];\n"
                f"    if ({fn}(&view, {_lower(t.name)})) {{\n"
                f"        (void){_lower(t.name)};\n"
                + _c_log(cfg, f"{t.name}[0]", f"{_lower(t.name)}[0]", t.dtype, "        ")
                + "\n    }"
            )
    else:
        example = "    /* (no observations declared) */"
    fields = action_fields(init)
    field_example = (
        f"    /* Name a field to change it, e.g. action.{_lower(fields[0].name)} */"
        if fields
        else "    /* (no action fields declared) */"
    )
    return f'''/* YOUR agent — written once by `task create-agent`, never overwritten.
 *
 * A hand-written agent for {cfg.env} [{cfg.mode}], compiled to
 * WebAssembly. Every tick exports_agent_on_tick receives what your seat
 * observes and returns your action: wire.c unpacks the bytes, the
 * generated interface.h gives every observation, slice, column and action
 * its name, and the code below decides. As created it plays the neutral
 * action (what the environment does for a seat that sends nothing), so it
 * runs but does not try.
 *
 * Build:  task build AGENT={cfg.name}
 * Match:  task match AGENT={cfg.name}
 * Debug:  task match AGENT={cfg.name} LOGS=1 STEP=1
 *         (prints whatever you fprintf to stderr, tagged with seat and
 *         tick, and pauses after every tick; only on your machine — ranked
 *         matches discard it)
 */

#include <stdio.h>
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

    /* One observation, read through the typed accessor: */
{example}

    /* The neutral action, through the typed builder — make it yours. */
    agent_action_t action;
    agent_action_neutral(&action);
{field_example}
    agent_action_encode(&action, &ret->ptr, &ret->len);

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
    (d / "interface.h").write_text(gen_interface_h(cfg, init, budgets, title), encoding="utf-8")
    print(f"→ {d / 'interface.h'}  (generated)", file=sys.stderr)
    for path, content, mode in (
        (d / "agent.c", gen_agent_c(cfg, init), 0o644),
        (d / "build.sh", gen_c_build_sh(cfg), 0o755),
    ):
        if path.exists():
            print(f"✓ {path} untouched (yours)", file=sys.stderr)
        else:
            path.write_text(content, encoding="utf-8")
            path.chmod(mode)
            print(f"→ {path}  (yours — edit it)", file=sys.stderr)
    gi = d / ".gitignore"
    if not gi.exists():
        gi.write_text("gen/\nout/\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# The supervised block: where the truth of a decision shows up
# ---------------------------------------------------------------------------


def supervised_block(init) -> dict | None:
    """The ``[supervised]`` table for ``agent.toml``: the first observation
    whose columns carry the documented feedback names, else the reward
    lag from ``meta``, else nothing (the recipe then refuses with the
    reason)."""
    if not init.actions:
        return None
    action = init.actions[0].name
    for t in init.obs:
        names = {c.name for c in t.columns}
        if {FEEDBACK_AGE, FEEDBACK_DECISION, FEEDBACK_TRUTH} <= names:
            return {
                "value": t.name,
                "age_col": FEEDBACK_AGE,
                "decision_col": FEEDBACK_DECISION,
                "truth_col": FEEDBACK_TRUTH,
                "valid_col": FEEDBACK_VALID if FEEDBACK_VALID in names else None,
                "action": action,
            }
    lag = _reward_lag(init)
    if lag and lag.strip().isdigit():
        return {"reward_lag_ticks": int(lag.strip()), "action": action}
    return None


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
        supervised=supervised_block(init),
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
    if cfg.supervised and cfg.lang == "python":
        print(
            f"      this environment reveals the truth of your decisions — "
            f"task train AGENT={cfg.name} RECIPE=supervised fits a classifier to it",
            file=sys.stderr,
        )


if __name__ == "__main__":
    sys.exit(main())
