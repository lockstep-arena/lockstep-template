"""Fetch an environment's released artifacts (``task engine`` runs this).

Downloads BOTH halves of the agent story next to each other::

    out/engine.wasm       the mode's engine — what you train against
    out/agent-onnx.wasm   the generic ONNX agent shell — what you ship

The shell is generic across every environment (it reads the tensor-wire
declaration and binds ONNX inputs by name), but it is versioned WITH the
release, so it is fetched from the same release directory rather than
pinned here. Each file gets a ``.url`` stamp so switching ``ENV``/``MODE``
re-downloads and unchanged re-runs are no-ops.

Usage::

    python -m train.core.engine --env <slug> [--version V] [--mode M] --out out
    python -m train.core.engine --env <slug> --print-url
    python -m train.core.engine --env <slug> --bundle out/agent-bundle --out out
"""

from __future__ import annotations

import argparse
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from . import utf8_output
from .discovery import EnvRelease, resolve


def bundle_manifest(bundle: Path) -> dict | None:
    """The ``lockstep.toml`` of a staged bundle (directory or .zip), or
    ``None`` for anything else — a bare component ``.wasm`` has no manifest
    and therefore no mode of its own."""
    if bundle.is_dir():
        manifest = bundle / "lockstep.toml"
        if manifest.is_file():
            return tomllib.loads(manifest.read_text())
        return None
    if bundle.is_file() and bundle.suffix == ".zip":
        with zipfile.ZipFile(bundle) as zf:
            try:
                with zf.open("lockstep.toml") as f:
                    return tomllib.loads(f.read().decode())
            except KeyError:
                return None
    return None


def resolve_mode(slug: str, mode: str | None, bundle: Path | None) -> str | None:
    """Which mode's engine to fetch: the explicit ``--mode`` wins, else the
    mode the bundle's manifest says it was staged for, else the release
    default (``None`` here → :func:`train.core.discovery.resolve` decides).

    A match of bundle-vs-engine across modes does not fail cleanly — it
    plays out as an agent that can't decode a single observation — so a
    disagreement between an explicit ``--mode`` and the bundle's declared
    mode is an error here, where it can still say what to do about it."""
    manifest = bundle_manifest(bundle) if bundle else None
    if manifest is None:
        return mode
    bundle_env = manifest.get("environment")
    if bundle_env and bundle_env != slug:
        raise SystemExit(
            f"bundle {bundle} was staged for environment {bundle_env!r}, not "
            f"{slug!r} — retrain (task train ENV={slug}) or pass the right ENV="
        )
    bundle_mode = manifest.get("mode")
    if mode and bundle_mode and mode != bundle_mode:
        raise SystemExit(
            f"bundle {bundle} was staged for mode {bundle_mode!r} but "
            f"MODE={mode} was asked for — drop MODE= to use the bundle's "
            f"own mode, or retrain with MODE={mode}"
        )
    return mode or bundle_mode


def fetch_file(url: str, out: Path) -> bool:
    """Download ``url`` to ``out``; returns False if the stamp says it's
    already there. Atomic (tmp then replace), like everything under out/."""
    stamp = out.with_suffix(out.suffix + ".url")
    if out.is_file() and stamp.is_file() and stamp.read_text().strip() == url:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "lockstep-template"})
    with urllib.request.urlopen(req) as resp:
        tmp.write_bytes(resp.read())
    tmp.replace(out)
    stamp.write_text(url + "\n")
    return True


def fetch_release(release: EnvRelease, out_dir: Path) -> dict[str, Path]:
    """Engine + generic agent shell into ``out_dir``; ``{name: path}``."""
    written = {}
    for name, url in (
        ("engine.wasm", release.engine_url),
        ("agent-onnx.wasm", release.agent_shell_url),
    ):
        path = out_dir / name
        if fetch_file(url, path):
            print(f"→ {path}  ({url})")
        else:
            print(f"✓ {path} up to date")
        written[name] = path
    return written


def main() -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, help="environment slug (ENV= on the task line)")
    p.add_argument("--version", default=None, help="release version; default latest")
    p.add_argument("--mode", default=None, help="mode key; default the release's default_mode")
    p.add_argument(
        "--bundle",
        default=None,
        help="an agent bundle (dir or .zip) whose lockstep.toml picks the "
        "mode when --mode is not given; a bare .wasm is fine and picks "
        "nothing",
    )
    p.add_argument("--out", default="out", help="output DIRECTORY")
    p.add_argument(
        "--print-url",
        action="store_true",
        help="print the resolved engine URL instead of downloading",
    )
    args = p.parse_args()

    mode = resolve_mode(args.env, args.mode or None, Path(args.bundle) if args.bundle else None)
    release = resolve(args.env, args.version, mode)
    if args.print_url:
        print(release.engine_url)
        return
    fetch_release(release, Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
