"""Game-agnostic training core.

Everything in here is parameterized by a discovered GameSpec (see
:mod:`train.core.discovery`) and the env's own Gymnasium spaces. Importing a
game package from this package is forbidden — games are reached ONLY through
entry-point discovery, which is what lets one template train every installed
game without ever naming one.
"""

from __future__ import annotations

import sys


def utf8_output() -> None:
    """Make the ``→``/``✓`` progress lines safe on Windows consoles.

    Windows' default stdout encoding (cp1252, the 'charmap' codec) cannot
    encode them and CRASHES the run mid-print — an OS-specific footgun this
    template refuses to ship. Every CLI entry point calls this first; it is
    a no-op where the streams are already UTF-8 (macOS, Linux, modern
    terminals).
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass
