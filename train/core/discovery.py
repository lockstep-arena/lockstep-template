"""Resolve a published environment from the CDN — no packages, no codegen.

There are no per-environment Python packages and no entry points (platform
decision D3): an environment is fully described by its published release.
``environments/<slug>/latest.json`` on the CDN (served no-store, so it is
always the current release) names the version, the default mode and the
mode list; everything else an agent needs — the engine wasm, the generic
ONNX agent shell, the tensor-wire interface the engine self-describes —
lives at deterministic paths under that release:

    environments/<slug>/latest.json
    environments/<slug>/releases/<version>/<mode>/engine.wasm
    environments/<slug>/releases/<version>/<mode>/agent-onnx.wasm

The CDN base is ``$LOCKSTEP_CDN_URL`` (default ``https://dl.lockstep.it``)
— the same variable ``lockstep_train.fetch`` reads, so the template and the
library can never disagree about where "the CDN" is.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from lockstep_train.fetch import cdn_base


@dataclass(frozen=True)
class EnvRelease:
    """One resolved (environment, version, mode) release."""

    slug: str
    version: str
    mode: str
    #: Every mode key the release publishes (for error messages + docs).
    modes: tuple[str, ...]

    @property
    def release_url(self) -> str:
        return f"{cdn_base()}/environments/{self.slug}/releases/{self.version}/{self.mode}"

    @property
    def engine_url(self) -> str:
        return f"{self.release_url}/engine.wasm"

    @property
    def agent_shell_url(self) -> str:
        return f"{self.release_url}/agent-onnx.wasm"


def _get(url: str) -> bytes:
    # A named User-Agent: the CDN rejects Python's default one.
    req = urllib.request.Request(url, headers={"User-Agent": "lockstep-template"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SystemExit(
                f"{url} does not exist — is the environment slug right? "
                f"Browse the catalog at https://lockstep.it/arenas"
            ) from e
        raise SystemExit(f"GET {url}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"GET {url}: {e.reason}") from e


def resolve(slug: str, version: str | None = None, mode: str | None = None) -> EnvRelease:
    """``latest.json`` → an :class:`EnvRelease`.

    ``version``/``mode`` override the release's defaults (``VERSION=`` /
    ``MODE=`` on the task line). A multi-mode environment with no
    ``default_mode`` must be told which mode — same rule as
    ``python -m lockstep_train.fetch``.
    """
    if not slug:
        raise SystemExit("pass ENV=<slug>")
    latest = json.loads(_get(f"{cdn_base()}/environments/{slug}/latest.json"))
    version = version or latest["version"]
    modes = tuple(latest.get("modes", ()))
    if mode is None:
        mode = latest.get("default_mode")
        if not mode:
            if len(modes) == 1:
                mode = modes[0]
            else:
                raise SystemExit(
                    f"{slug} has modes {list(modes)} and no default_mode — "
                    "pass MODE=<key>"
                )
    elif modes and mode not in modes:
        raise SystemExit(
            f"{slug}@{version} has no mode {mode!r} (modes: {', '.join(modes)})"
        )
    return EnvRelease(slug=slug, version=version, mode=mode, modes=modes)
