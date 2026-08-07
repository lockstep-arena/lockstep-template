"""Fetch a game's pinned engine wasm (``task engine`` runs this).

The engine URL is not template configuration — it comes from the installed
game package's GameSpec, which pins the exact published release its codec
was built against. This helper downloads it next to a ``.url`` stamp so
switching ``GAME``/``MODE`` re-downloads and unchanged re-runs are no-ops.

Usage::

    python -m train.core.engine --game <slug> --out out/engine.wasm
    python -m train.core.engine --game <slug> --mode <mode> --print-url
    python -m train.core.engine --game <slug> --bundle out/agent-bundle --out out/engine.wasm
"""

from __future__ import annotations

import argparse
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

from . import utf8_output
from .discovery import resolve_game


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


def resolve_mode(spec: dict, mode: str | None, bundle: Path | None) -> str | None:
    """Which mode's engine to fetch: the explicit ``--mode`` wins, else the
    mode the bundle's manifest says it was staged for, else the game default.

    A match of bundle-vs-engine across modes does not fail cleanly — it
    plays out as an agent that can't decode a single observation — so a
    disagreement between an explicit ``--mode`` and the bundle's declared
    mode is an error here, where it can still say what to do about it."""
    manifest = bundle_manifest(bundle) if bundle else None
    if manifest is None:
        return mode
    bundle_game = manifest.get("game")
    if bundle_game and bundle_game != spec["slug"]:
        raise SystemExit(
            f"bundle {bundle} was staged for game {bundle_game!r}, not "
            f"{spec['slug']!r} — retrain (task train GAME={spec['slug']}) "
            f"or pass the right GAME="
        )
    bundle_mode = manifest.get("mode")
    if mode and bundle_mode and mode != bundle_mode:
        raise SystemExit(
            f"bundle {bundle} was staged for mode {bundle_mode!r} but "
            f"MODE={mode} was asked for — drop MODE= to use the bundle's "
            f"own mode, or retrain with MODE={mode}"
        )
    return mode or bundle_mode


def engine_url(spec: dict, mode: str | None) -> str:
    mode = mode or spec["default_mode"]
    if mode not in spec["modes"]:
        raise SystemExit(
            f"game {spec['slug']!r} has no mode {mode!r} "
            f"(modes: {', '.join(sorted(spec['modes']))})"
        )
    return spec["modes"][mode]["engine_url"]


def fetch(url: str, out: Path) -> bool:
    """Download ``url`` to ``out``; returns False if the stamp says it's
    already there. Atomic (tmp then replace), like everything under out/."""
    stamp = out.with_suffix(out.suffix + ".url")
    if out.is_file() and stamp.is_file() and stamp.read_text().strip() == url:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    # A named User-Agent: the CDN rejects Python's default one.
    req = urllib.request.Request(
        url, headers={"User-Agent": "lockstep-template-engine-fetch"}
    )
    with urllib.request.urlopen(req) as resp:
        tmp.write_bytes(resp.read())
    tmp.replace(out)
    stamp.write_text(url + "\n")
    return True


def main() -> None:
    utf8_output()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None, help="defaults to the one installed game")
    p.add_argument("--mode", default=None, help="defaults to the game's default mode")
    p.add_argument(
        "--bundle",
        default=None,
        help="an agent bundle (dir or .zip) whose lockstep.toml picks the "
        "mode when --mode is not given; a bare .wasm is fine and picks "
        "nothing",
    )
    p.add_argument("--out", default="out/engine.wasm")
    p.add_argument(
        "--print-url",
        action="store_true",
        help="print the pinned engine URL instead of downloading",
    )
    args = p.parse_args()

    spec, _module = resolve_game(args.game)
    mode = resolve_mode(
        spec, args.mode or None, Path(args.bundle) if args.bundle else None
    )
    url = engine_url(spec, mode)
    if args.print_url:
        print(url)
        return
    out = Path(args.out)
    if fetch(url, out):
        print(f"→ {out}  ({url})")
    else:
        print(f"✓ {out} up to date")


if __name__ == "__main__":
    sys.exit(main())
