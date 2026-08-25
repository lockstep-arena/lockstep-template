"""Resolve a published environment — no packages, no codegen.

There are no per-environment Python packages and no entry points (platform
decision D3): an environment is fully described by its published release.
The release lives under an UNGUESSABLE directory on the CDN
(``environments/<slug>/releases/<version>-<token>/<mode>/``), so its path
is never derived here — it comes from the platform API's ``environment/get``
(the pinned ``engine_object_key``), which ``lockstep_train.fetch.resolve``
wraps. Competitions answer publicly; an assessment-only environment answers
an invited candidate through ``LOCKSTEP_API_KEY`` (the key ``task upload``
already needs). Under that prefix:

    <prefix>/engine.wasm        the mode's engine — what you train against
    <prefix>/agent-onnx.wasm    the generic ONNX agent shell — what you ship

Endpoints come from the same variables the library reads —
``$LOCKSTEP_API_URL`` and ``$LOCKSTEP_CDN_URL`` — so the template and the
library can never disagree about where "the platform" is.
"""

from __future__ import annotations

from dataclasses import dataclass

from lockstep_train.fetch import cdn_base
from lockstep_train.fetch import resolve as resolve_release


@dataclass(frozen=True)
class EnvRelease:
    """One resolved (environment, version, mode) release."""

    slug: str
    version: str
    mode: str
    #: Every mode key the release publishes (for error messages + docs).
    modes: tuple[str, ...]
    #: CDN prefix of this mode's release directory (no trailing slash).
    prefix: str

    @property
    def release_url(self) -> str:
        return f"{cdn_base()}/{self.prefix}"

    @property
    def engine_url(self) -> str:
        return f"{self.release_url}/engine.wasm"

    @property
    def agent_shell_url(self) -> str:
        return f"{self.release_url}/agent-onnx.wasm"


def resolve(slug: str, version: str | None = None, mode: str | None = None) -> EnvRelease:
    """``environment/get`` → an :class:`EnvRelease`.

    ``mode`` overrides the release's ``default_mode`` (``MODE=`` on the task
    line); a multi-mode environment with no default must be told which —
    same rule as ``python -m lockstep_train.fetch``. ``version`` (``VERSION=``)
    is an assertion: only the current release is published, so a pin that
    disagrees with it stops here instead of training against the wrong
    engine.
    """
    if not slug:
        raise SystemExit("pass ENV=<slug>")
    try:
        release = resolve_release(slug, mode)
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    if version and version != release.version:
        raise SystemExit(
            f"{slug}/{release.mode} is at {release.version}; VERSION={version} is not "
            "published (only the current release is available)"
        )
    return EnvRelease(
        slug=slug,
        version=release.version,
        mode=release.mode,
        modes=release.modes,
        prefix=release.prefix,
    )
