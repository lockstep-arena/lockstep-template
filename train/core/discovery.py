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


@dataclass(frozen=True)
class PublishedEnvironment:
    """One environment the platform currently lists with a published mode."""

    slug: str
    #: Every PUBLISHED mode key (a mode with a release), in platform order.
    modes: tuple[str, ...]
    #: True when at least one published mode's engine is fetchable without an
    #: invite key (its release names a CDN ``engine_object_key``). False means
    #: assessment-only: ``LOCKSTEP_API_KEY`` for an invited candidate.
    public: bool


def published() -> list[PublishedEnvironment]:
    """``environment/list`` → every environment with at least one published mode.

    This is how the template learns what environments exist: nothing is
    named in this repo — not as a default, not in CI. ``task envs`` prints
    it; ``task create-agent ENV=<slug>`` takes one of these slugs.
    """
    # Same helper `resolve` uses — it already handles LOCKSTEP_API_URL,
    # LOCKSTEP_API_KEY and the User-Agent.
    from lockstep_train.fetch import _api_post

    try:
        payload = _api_post("environment/list", {"tag": None})
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    found: list[PublishedEnvironment] = []
    for env in payload.get("environments", []):
        releases = [
            (m["key"], m["release"]) for m in env.get("modes", []) if m.get("release") is not None
        ]
        if not releases:
            continue
        found.append(
            PublishedEnvironment(
                slug=env["id"],
                modes=tuple(key for key, _ in releases),
                public=any(bool(rel.get("engine_object_key")) for _, rel in releases),
            )
        )
    return found


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    p = argparse.ArgumentParser(
        prog="python -m train.core.discovery",
        description="List the environments the platform publishes — the slugs ENV= takes.",
    )
    p.add_argument(
        "--first",
        action="store_true",
        help="print only the first PUBLIC slug (exit 1 when there is none)",
    )
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    envs = published()
    if args.first:
        for env in envs:
            if env.public:
                print(env.slug)
                return 0
        print(
            "no publicly published environment on the platform"
            + (" (all listed environments need an invite key)" if envs else ""),
            file=sys.stderr,
        )
        return 1
    if args.json:
        print(json.dumps([env.__dict__ for env in envs], indent=2))
        return 0
    if not envs:
        print("no published environments on the platform", file=sys.stderr)
        return 1
    width = max(len(env.slug) for env in envs)
    for env in envs:
        line = f"{env.slug.ljust(width)}    modes: {', '.join(env.modes)}"
        if not env.public:
            line += "    (invite key required)"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
