"""Environment-agnostic training core.

Everything in here is parameterized by the engine wasm's own tensor-wire
declaration (surfaced as Gymnasium spaces by ``lockstep_train``) and the
CDN release metadata (see :mod:`train.core.discovery`). There are no
per-environment packages to import — naming an environment anywhere in this
package is forbidden, which is what lets one template train every published
environment without ever naming one.
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
