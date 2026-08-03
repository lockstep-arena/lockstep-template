"""Fetch a game's pinned engine wasm (``task engine`` runs this).

The engine URL is not template configuration — it comes from the installed
game package's GameSpec, which pins the exact published release its codec
was built against. This helper downloads it next to a ``.url`` stamp so
switching ``GAME``/``MODE`` re-downloads and unchanged re-runs are no-ops.

Usage::

    python -m train.core.engine --game <slug> --out out/engine.wasm
    python -m train.core.engine --game <slug> --mode <mode> --print-url
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from .discovery import resolve_game


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", default=None, help="defaults to the one installed game")
    p.add_argument("--mode", default=None, help="defaults to the game's default mode")
    p.add_argument("--out", default="out/engine.wasm")
    p.add_argument(
        "--print-url",
        action="store_true",
        help="print the pinned engine URL instead of downloading",
    )
    args = p.parse_args()

    spec, _module = resolve_game(args.game)
    url = engine_url(spec, args.mode or None)
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
