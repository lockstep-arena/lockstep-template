"""`task doctor` — is this machine ready for the template?

Checks every prerequisite the README lists and, for each one that is
missing, prints the exact fix. Runs on the SYSTEM python with no
third-party imports, because the venv is one of the things it checks.

    python3 -m train.doctor            # human report; exit 1 if a required item is missing
    python3 -m train.doctor --json     # the same report as JSON, for tooling

Required: Python >= 3.11, Task, the lockstep CLI, the venv with
lockstep-train installed. Optional (reported, never fatal): LOCKSTEP_API_KEY
(only `task upload` needs it), the Rust toolchain + wasm32-wasip2 target
(only the Rust example agent needs it), a fetched engine (`task engine`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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


def check_cli() -> Check:
    path = shutil.which("lockstep")
    if not path:
        return Check(
            "lockstep CLI",
            False,
            True,
            "not on PATH — `task match` and `task upload` refuse to run without it",
            INSTALL_CLI["nt" if os.name == "nt" else "posix"],
        )
    _, out = _run([path, "--version"])
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
        "cp .env.example .env, then paste a key from your account settings at https://lockstep.it into LOCKSTEP_API_KEY=",
    )


def check_rust() -> Check:
    cargo = shutil.which("cargo")
    rustup = shutil.which("rustup")
    if not cargo:
        return Check(
            "Rust toolchain",
            False,
            False,
            "not installed — only the pure-Rust example agent needs it",
            "https://rustup.rs then: rustup target add wasm32-wasip2",
        )
    if rustup:
        _, out = _run([rustup, "target", "list", "--installed"])
        if "wasm32-wasip2" not in out:
            return Check(
                "Rust toolchain",
                False,
                False,
                "cargo present, wasm32-wasip2 target missing — only the Rust example agent needs it",
                "rustup target add wasm32-wasip2",
            )
    _, out = _run([cargo, "--version"])
    return Check("Rust toolchain", True, False, f"{out or 'present'}, wasm32-wasip2 installed")


def check_engine() -> Check:
    cache = ROOT / "out" / "cache"
    engines = sorted(cache.glob("*/*/engine.wasm")) if cache.is_dir() else []
    if engines:
        keys = ", ".join(f"{e.parent.parent.name}/{e.parent.name}" for e in engines)
        return Check("engine cache", True, False, f"out/cache holds: {keys}")
    return Check(
        "engine cache",
        False,
        False,
        "no cached engines yet — every task that needs one (info / train / "
        "build / match) fetches it on demand",
        "nothing to run by hand; task info ENV=<slug> warms the cache",
    )


def run() -> list[Check]:
    return [
        check_python(),
        check_task(),
        check_cli(),
        check_venv(),
        check_api_key(),
        check_rust(),
        check_engine(),
    ]


def render(checks: list[Check]) -> str:
    lines = ["lockstep template doctor", ""]
    width = max(len(c.name) for c in checks)
    for c in checks:
        mark = "✓" if c.ok else ("✗" if c.required else "·")
        lines.append(f"  {mark} {c.name:<{width}}  {c.detail}")
        if not c.ok and c.fix:
            lines.append(f"    fix → {c.fix}")
    missing = [c for c in checks if not c.ok and c.required]
    optional = [c for c in checks if not c.ok and not c.required]
    lines.append("")
    if missing:
        lines.append(f"{len(missing)} required item(s) missing — fix the ✗ lines above, then run `task doctor` again.")
    else:
        lines.append("Everything required is in place.")
        if optional:
            lines.append("The · lines are optional; each says what it unlocks.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = ap.parse_args(argv)
    checks = run()
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        sys.stdout.write(render(checks))
    return 1 if any(not c.ok and c.required for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
