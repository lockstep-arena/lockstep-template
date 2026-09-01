"""Per-language toolchains: detect first, install only in ``task setup``.

The rule (and the reason this module exists): ``task setup LANGS=…`` is the
ONE place that provisions anything. ``task create-agent`` and ``task
build`` only *detect* — a missing toolchain fails fast with the exact
``task setup LANGS=…`` line to run, never a surprise download.

C toolchain = wasi-sdk (clang with a ``wasm32-wasip2`` target that links
components directly) + ``wit-bindgen`` (the world-bindings generator).
Detection order for wasi-sdk: ``$WASI_SDK`` / ``$WASI_SDK_PATH`` → the
conventional ``/opt/wasi-sdk`` → the repo-local cache this module installs
into (``out/toolchains/wasi-sdk-<ver>``). ``task setup LANGS=c`` downloads
the pinned release into that cache only when nothing is found.

    python -m train.toolchain check c        # detect + report (exit 1 if missing)
    python -m train.toolchain install c      # detect, else download into out/toolchains/
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLCHAINS = ROOT / "out" / "toolchains"

#: The pinned wasi-sdk release `task setup LANGS=c` installs when no
#: system install is found. ≥ 24 is required (the first with a
#: wasm32-wasip2 clang target that links components directly).
WASI_SDK_VERSION = "25"
WASI_SDK_BASE = "https://github.com/WebAssembly/wasi-sdk/releases/download"

#: wit-bindgen pin for the cargo fallback install (any recent works; the
#: scaffold's build script only uses `wit-bindgen c`).
WIT_BINDGEN_VERSION = "0.46.0"


def _wasi_sdk_asset() -> str:
    mach = platform.machine().lower()
    arch = "arm64" if mach in ("arm64", "aarch64") else "x86_64"
    osname = {"darwin": "macos", "linux": "linux", "windows": "windows"}[
        "windows" if os.name == "nt" else sys.platform if sys.platform in ("darwin",) else "linux"
    ]
    return f"wasi-sdk-{WASI_SDK_VERSION}.0-{arch}-{osname}"


def find_wasi_sdk() -> Path | None:
    """The first wasi-sdk whose clang exists: env override, /opt/wasi-sdk,
    then the repo-local cache."""
    exe = "clang.exe" if os.name == "nt" else "clang"
    candidates: list[Path] = []
    for var in ("WASI_SDK", "WASI_SDK_PATH"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.append(Path(v))
    candidates.append(Path("/opt/wasi-sdk"))
    if TOOLCHAINS.is_dir():
        candidates.extend(sorted(TOOLCHAINS.glob("wasi-sdk-*")))
    for c in candidates:
        if (c / "bin" / exe).exists():
            return c
    return None


def find_wit_bindgen() -> str | None:
    found = shutil.which("wit-bindgen")
    if found:
        return found
    exe = "wit-bindgen.exe" if os.name == "nt" else "wit-bindgen"
    if TOOLCHAINS.is_dir():
        for cand in sorted(TOOLCHAINS.glob(f"wit-bindgen-*/{exe}")):
            return str(cand)
    return None


def install_wasi_sdk() -> Path:
    """Download the pinned wasi-sdk into out/toolchains/ (idempotent)."""
    found = find_wasi_sdk()
    if found:
        return found
    asset = _wasi_sdk_asset()
    url = f"{WASI_SDK_BASE}/wasi-sdk-{WASI_SDK_VERSION}/{asset}.tar.gz"
    TOOLCHAINS.mkdir(parents=True, exist_ok=True)
    tarball = TOOLCHAINS / f"{asset}.tar.gz"
    print(f"→ downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as resp:
        tarball.write_bytes(resp.read())
    with tarfile.open(tarball) as tf:
        tf.extractall(TOOLCHAINS, filter="data")
    tarball.unlink()
    extracted = TOOLCHAINS / asset
    target = TOOLCHAINS / f"wasi-sdk-{WASI_SDK_VERSION}"
    if extracted != target:
        extracted.rename(target)
    print(f"→ wasi-sdk {WASI_SDK_VERSION} at {target}", file=sys.stderr)
    return target


def _wit_bindgen_asset() -> tuple[str, str]:
    """(asset filename, archive kind) for this machine's prebuilt release."""
    mach = platform.machine().lower()
    arch = "aarch64" if mach in ("arm64", "aarch64") else "x86_64"
    if os.name == "nt":
        return f"wit-bindgen-{WIT_BINDGEN_VERSION}-{arch}-windows.zip", "zip"
    osname = "macos" if sys.platform == "darwin" else "linux"
    return f"wit-bindgen-{WIT_BINDGEN_VERSION}-{arch}-{osname}.tar.gz", "tar"


def install_wit_bindgen() -> str:
    found = find_wit_bindgen()
    if found:
        return found
    asset, kind = _wit_bindgen_asset()
    url = (
        "https://github.com/bytecodealliance/wit-bindgen/releases/download/"
        f"v{WIT_BINDGEN_VERSION}/{asset}"
    )
    TOOLCHAINS.mkdir(parents=True, exist_ok=True)
    archive = TOOLCHAINS / asset
    print(f"→ downloading {url}", file=sys.stderr)
    try:
        with urllib.request.urlopen(url) as resp:
            archive.write_bytes(resp.read())
    except OSError:
        # No prebuilt for this machine (or offline registry) — cargo fallback.
        if not shutil.which("cargo"):
            raise SystemExit(
                f"could not download {url} and cargo is unavailable — install "
                "wit-bindgen yourself: https://github.com/bytecodealliance/wit-bindgen/releases"
            ) from None
        print(f"→ cargo install wit-bindgen-cli@{WIT_BINDGEN_VERSION}", file=sys.stderr)
        subprocess.run(
            ["cargo", "install", f"wit-bindgen-cli@{WIT_BINDGEN_VERSION}", "--locked"],
            check=True,
        )
        return shutil.which("wit-bindgen") or "wit-bindgen"
    if kind == "zip":
        import zipfile

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(TOOLCHAINS)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(TOOLCHAINS, filter="data")
    archive.unlink()
    found = find_wit_bindgen()
    if not found:
        raise SystemExit(f"extracted {asset} but found no wit-bindgen binary under {TOOLCHAINS}")
    # The release ships without the exec bit on some archives.
    if os.name != "nt":
        os.chmod(found, 0o755)
    print(f"→ wit-bindgen at {found}", file=sys.stderr)
    return found


def check(lang: str) -> list[tuple[str, str | None, str]]:
    """[(component, found-where-or-None, fix)] for one language."""
    if lang == "python":
        venv_py = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        return [
            (
                "venv + training stack",
                str(venv_py) if venv_py.exists() else None,
                "task setup",
            )
        ]
    if lang == "rust":
        cargo = shutil.which("cargo")
        target_ok = False
        if cargo and shutil.which("rustup"):
            out = subprocess.run(
                ["rustup", "target", "list", "--installed"], capture_output=True, text=True
            ).stdout
            target_ok = "wasm32-wasip2" in out
        elif cargo:
            target_ok = True  # non-rustup installs: trust the build to say
        return [
            ("cargo", cargo, "task setup LANGS=rust  (or https://rustup.rs)"),
            (
                "wasm32-wasip2 target",
                "installed" if target_ok else None,
                "task setup LANGS=rust  (runs: rustup target add wasm32-wasip2)",
            ),
        ]
    if lang == "c":
        sdk = find_wasi_sdk()
        wb = find_wit_bindgen()
        return [
            (
                "wasi-sdk (>= 24)",
                str(sdk) if sdk else None,
                "task setup LANGS=c  (detects /opt/wasi-sdk or $WASI_SDK, else "
                f"downloads wasi-sdk {WASI_SDK_VERSION} into out/toolchains/)",
            ),
            ("wit-bindgen", wb, "task setup LANGS=c"),
        ]
    raise SystemExit(f"unknown language {lang!r}")


def install(lang: str) -> None:
    if lang == "python":
        return  # task setup's pip steps own this
    if lang == "rust":
        if not shutil.which("cargo"):
            raise SystemExit(
                "no cargo on PATH — install Rust first: https://rustup.rs, then re-run"
            )
        if shutil.which("rustup"):
            subprocess.run(["rustup", "target", "add", "wasm32-wasip2"], check=True)
        return
    if lang == "c":
        install_wasi_sdk()
        install_wit_bindgen()
        return
    raise SystemExit(f"unknown language {lang!r}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] not in ("check", "install"):
        print(__doc__)
        return 2
    verb, lang = args
    if verb == "install":
        install(lang)
    rows = check(lang)
    missing = False
    for name, where, fix in rows:
        if where:
            print(f"  ✓ {name}: {where}", file=sys.stderr)
        else:
            missing = True
            print(f"  ✗ {name}: missing — fix: {fix}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
