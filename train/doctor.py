"""`task doctor` — is this machine ready for the template?

Checks every prerequisite the README lists and, for each one that is
missing, prints the exact fix. Runs on the SYSTEM python with no
third-party imports, because the venv is one of the things it checks.

    python3 -m train.doctor            # human report; exit 1 if a required item is missing
    python3 -m train.doctor --json     # the same report as JSON, for tooling

Required: Python >= 3.11, Task, the lockstep CLI, the venv with
lockstep-train installed. Per-language toolchains are REQUIRED exactly when
an agent of that language exists under agents/ (else reported as optional):
rust = cargo + the wasm32-wasip2 target; c = wasi-sdk + wit-bindgen.
Optional always: LOCKSTEP_API_KEY (only `task upload` needs it), the keyed
engine cache (filled on demand).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

INSTALL_CLI = {
    "posix": "curl -fsSL https://dl.lockstep.it/install.sh | sh",
    "nt": 'powershell -ExecutionPolicy Bypass -c "irm https://dl.lockstep.it/install.ps1 | iex"',
}


@dataclass
class Check:
    name: str
    ok: bool
    required: bool
    detail: str
    fix: str = ""


def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)
    return p.returncode, (p.stdout or p.stderr).strip()


def check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return Check(
        "Python",
        ok,
        True,
        f"{v.major}.{v.minor}.{v.micro} at {sys.executable}",
        "" if ok else "install Python 3.11 or newer (https://www.python.org/downloads/) and re-run with it",
    )


def check_task() -> Check:
    path = shutil.which("task")
    if not path:
        return Check("Task", False, True, "not on PATH", "install Task: https://taskfile.dev/installation/")
    _, out = _run([path, "--version"])
    return Check("Task", True, True, f"{out or 'present'} at {path}")


#: The oldest CLI whose commands take every flag the template passes
#: (`--clamp-seats` on `match run` arrived in 0.1.5, `--assessment` on
#: `agent upload` and `archive report` in 0.1.10, `--agent-logs` / `--step`
#: / `--until-tick` in 0.1.11, `LOCKSTEP_AGENT_LOGS` — which makes a trained
#: policy show what it sees under `task match LOGS=1` — in 0.1.12). The
#: installer always fetches the newest, so "update" is the whole fix.
MIN_CLI = (0, 1, 12)


def _parse_version(text: str) -> tuple[int, ...] | None:
    """`lockstep 0.1.5` → (0, 1, 5); anything unparseable → None."""
    for word in text.split():
        parts = word.split(".")
        if len(parts) >= 2 and all(p.isdigit() for p in parts):
            return tuple(int(p) for p in parts)
    return None


def check_cli() -> Check:
    path = shutil.which("lockstep")
    install = INSTALL_CLI["nt" if os.name == "nt" else "posix"]
    if not path:
        return Check(
            "lockstep CLI",
            False,
            True,
            "not on PATH — `task match` and `task upload` refuse to run without it",
            install,
        )
    _, out = _run([path, "--version"])
    version = _parse_version(out)
    if version is not None and version < MIN_CLI:
        want = ".".join(str(n) for n in MIN_CLI)
        return Check(
            "lockstep CLI",
            False,
            True,
            f"{out} at {path} — older than {want}, so `task match` / `task upload` / `task report` fail on flags it does not know",
            f"update it (the installer fetches the newest): {install}",
        )
    return Check("lockstep CLI", True, True, f"{out or 'present'} at {path}")


def check_venv() -> Check:
    if not VENV_PY.exists():
        return Check(
            "venv",
            False,
            True,
            f"no {VENV.relative_to(ROOT)}/ yet",
            "task setup   (creates .venv and installs the training stack)",
        )
    code, out = _run(
        [
            str(VENV_PY),
            "-c",
            "import lockstep_train, gymnasium, torch, onnxruntime; "
            "print(getattr(lockstep_train, '__version__', '?'))",
        ],
        timeout=60.0,
    )
    if code != 0:
        return Check(
            "venv",
            False,
            True,
            f".venv exists but the training stack does not import: {out.splitlines()[-1] if out else 'unknown error'}",
            "task setup   (re-runs the install into the existing .venv)",
        )
    return Check("venv", True, True, f".venv ready — lockstep-train {out}")


def check_api_key() -> Check:
    key = os.environ.get("LOCKSTEP_API_KEY", "").strip()
    if not key:
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text(errors="replace").splitlines():
                if line.startswith("LOCKSTEP_API_KEY=") and line.split("=", 1)[1].strip():
                    key = line.split("=", 1)[1].strip()
                    break
    if key:
        return Check("LOCKSTEP_API_KEY", True, False, "set (only `task upload` needs it)")
    return Check(
        "LOCKSTEP_API_KEY",
        False,
        False,
        "not set — everything except `task upload` works without it",
        "cp .env.example .env, then paste a key from the Authorization page at https://lockstep.it into LOCKSTEP_API_KEY=",
    )


def _agent_langs() -> set[str]:
    """Languages of the agents that actually exist (drives which toolchains
    are REQUIRED). Read directly — no third-party imports here."""
    import tomllib

    langs: set[str] = set()
    agents = ROOT / "agents"
    if agents.is_dir():
        for toml in agents.glob("*/agent.toml"):
            try:
                langs.add(tomllib.loads(toml.read_text()).get("agent", {}).get("lang", "python"))
            except (OSError, tomllib.TOMLDecodeError):
                continue
    return langs


def check_rust(required: bool) -> Check:
    why = (
        "a rust agent exists under agents/"
        if required
        else "only needed for LANG=rust agents"
    )
    cargo = shutil.which("cargo")
    rustup = shutil.which("rustup")
    if not cargo:
        return Check(
            "Rust toolchain",
            False,
            required,
            f"not installed — {why}",
            "task setup LANGS=rust   (or https://rustup.rs then: rustup target add wasm32-wasip2)",
        )
    if rustup:
        _, out = _run([rustup, "target", "list", "--installed"])
        if "wasm32-wasip2" not in out:
            return Check(
                "Rust toolchain",
                False,
                required,
                f"cargo present, wasm32-wasip2 target missing — {why}",
                "task setup LANGS=rust   (runs: rustup target add wasm32-wasip2)",
            )
    _, out = _run([cargo, "--version"])
    return Check("Rust toolchain", True, required, f"{out or 'present'}, wasm32-wasip2 installed")


def check_c(required: bool) -> Check:
    # train.toolchain has the one detection order (env → /opt/wasi-sdk →
    # repo-local cache); reuse it rather than encoding it twice.
    sys.path.insert(0, str(ROOT))
    try:
        from train.toolchain import WASI_SDK_VERSION, find_wasi_sdk, find_wit_bindgen
    finally:
        sys.path.pop(0)

    why = "a c agent exists under agents/" if required else "only needed for LANG=c agents"
    sdk = find_wasi_sdk()
    wb = find_wit_bindgen()
    if sdk is None or wb is None:
        missing = " + ".join(
            n for n, present in (("wasi-sdk", sdk), ("wit-bindgen", wb)) if not present
        )
        return Check(
            "C toolchain",
            False,
            required,
            f"{missing} missing — {why}",
            f"task setup LANGS=c   (detects /opt/wasi-sdk or $WASI_SDK, else downloads "
            f"wasi-sdk {WASI_SDK_VERSION} into out/toolchains/)",
        )
    return Check("C toolchain", True, required, f"wasi-sdk at {sdk}; wit-bindgen at {wb}")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def check_engine() -> Check:
    """The keyed cache, one line per (environment, mode): its size on disk
    and the blobs / ONNX artifacts fetched beside the engine — so a data
    environment's tables are visibly there (or visibly not)."""
    cache = ROOT / "out" / "cache"
    engines = sorted(cache.glob("*/*/engine.wasm")) if cache.is_dir() else []
    if engines:
        lines = []
        for e in engines:
            d = e.parent
            files = [p for p in d.rglob("*") if p.is_file()]
            size = sum(p.stat().st_size for p in files)
            data = sorted(p.name for p in (d / "data").iterdir() if p.is_file()) if (d / "data").is_dir() else []
            arts = (
                sorted(p.name for p in (d / "artifacts").iterdir() if p.is_file())
                if (d / "artifacts").is_dir()
                else []
            )
            extras = []
            if data:
                extras.append(f"data: {', '.join(data)}")
            if arts:
                extras.append(f"artifacts: {', '.join(arts)}")
            if not (d / "agent-onnx.wasm").is_file():
                extras.append("shell missing — re-run task info")
            tail = f" ({'; '.join(extras)})" if extras else ""
            lines.append(f"{d.parent.name}/{d.name} {_human(size)}{tail}")
        return Check("engine cache", True, False, "out/cache holds: " + "; ".join(lines))
    return Check(
        "engine cache",
        False,
        False,
        "no cached engines yet — every task that needs one (info / train / "
        "build / match) fetches it on demand",
        "nothing to run by hand; task info ENV=<slug> warms the cache",
    )


#: The checks, in the order they are reported. Names are listed up front so
#: the report can be printed one line at a time — the column width is known
#: before the slow checks (the venv imports torch) have run.
def checks() -> list[tuple[str, Callable[[], Check]]]:
    langs = _agent_langs()
    return [
        ("Python", check_python),
        ("Task", check_task),
        ("lockstep CLI", check_cli),
        ("venv", check_venv),
        ("LOCKSTEP_API_KEY", check_api_key),
        ("Rust toolchain", lambda: check_rust(required="rust" in langs)),
        ("C toolchain", lambda: check_c(required="c" in langs)),
        ("engine cache", check_engine),
    ]


def run(progress: Callable[[Check], None] | None = None) -> list[Check]:
    """Run every check; `progress` is called with each one as it lands."""
    done: list[Check] = []
    for _, fn in checks():
        c = fn()
        done.append(c)
        if progress is not None:
            progress(c)
    return done


def render_line(c: Check, width: int) -> str:
    """One check as the report prints it (plus its fix line when it failed)."""
    mark = "✓" if c.ok else ("✗" if c.required else "·")
    line = f"  {mark} {c.name:<{width}}  {c.detail}"
    if not c.ok and c.fix:
        line += f"\n    fix → {c.fix}"
    return line


def render_summary(checks: list[Check]) -> str:
    missing = [c for c in checks if not c.ok and c.required]
    optional = [c for c in checks if not c.ok and not c.required]
    lines = [""]
    if missing:
        lines.append(f"{len(missing)} required item(s) missing — fix the ✗ lines above, then run `task doctor` again.")
    else:
        lines.append("Everything required is in place.")
        if optional:
            lines.append("The · lines are optional; each says what it unlocks.")
    return "\n".join(lines) + "\n"


def render(checks: list[Check]) -> str:
    """The whole report at once (the streaming path prints the same lines)."""
    width = max(len(c.name) for c in checks)
    body = "\n".join(render_line(c, width) for c in checks)
    return f"{HEADER}\n\n{body}\n{render_summary(checks)}"


HEADER = "lockstep template doctor"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    ap.add_argument(
        "--check-cli",
        action="store_true",
        help="exit 0 when the lockstep CLI is new enough, 1 otherwise, printing nothing "
        "(what task match's precondition runs)",
    )
    args = ap.parse_args(argv)
    if args.check_cli:
        return 0 if check_cli().ok else 1
    if args.json:
        # Progress goes to stderr so stdout stays a single JSON document.
        print("doctor: running checks…", file=sys.stderr, flush=True)
        report = run()
        print(json.dumps([asdict(c) for c in report], indent=2))
    else:
        # Say something before the first slow check (the venv one imports
        # the training stack, which can take several seconds cold), and
        # print every result the moment it lands rather than all at the end.
        width = max(len(name) for name, _ in checks())
        print(f"{HEADER}\nrunning checks… (the venv check imports the training stack; give it a moment)\n", flush=True)
        report = run(lambda c: print(render_line(c, width), flush=True))
        sys.stdout.write(render_summary(report))
    return 1 if any(not c.ok and c.required for c in report) else 0


if __name__ == "__main__":
    raise SystemExit(main())
