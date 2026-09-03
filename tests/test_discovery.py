"""Release resolution against a local mock platform — no network, no packages.

Serves a fake ``environment/get`` + ``environment/list`` (the API) plus
``latest.json`` (the CDN's public pointer, read only for ``default_mode``)
from thread-local HTTP
servers and points ``$LOCKSTEP_API_URL`` / ``$LOCKSTEP_CDN_URL`` at them —
the same variables ``lockstep_train`` reads, so the template and the library
agree about where "the platform" is by construction. Release prefixes are
tokened, as on the real CDN: nothing here derives a path from slug + version.
Every slug and mode below is invented for the test: the template names no
real environment anywhere (tests/test_import_lint.py enforces it).
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from train.core.discovery import main as discovery_main
from train.core.discovery import published, resolve

TOKEN = "0123456789abcdef"
RELEASES = {
    "two-mode": {"version": "0.7.0", "default_mode": "alpha",
                 "modes": ["alpha", "beta"]},
    # A multi-mode release that (wrongly) omits default_mode: the resolver
    # must demand MODE= rather than guess a ladder.
    "no-default": {"version": "1.0.0", "modes": ["a", "b"]},
    # A single mode needs no default to be unambiguous.
    "solo": {"version": "2.0.0", "modes": ["default"]},
    # Assessment-only: published, but its engine has no public CDN key —
    # only an invited candidate (LOCKSTEP_API_KEY) can fetch it.
    "sealed": {"version": "3.0.0", "modes": ["default"], "public": False},
    # Annotated but never published: environment/list carries it with no
    # release on any mode, so `published()` must drop it.
    "unpublished": {"version": "0.0.0", "modes": ["default"], "published": False},
}


def prefix(slug: str, mode: str) -> str:
    return f"environments/{slug}/releases/{RELEASES[slug]['version']}-{TOKEN}/{mode}"


def _mode_entry(slug: str, mode: str) -> dict:
    """One `modes[]` element as `environment/get` + `environment/list` shape it."""
    env = RELEASES[slug]
    if not env.get("published", True):
        return {"key": mode, "release": None}
    key = f"{prefix(slug, mode)}/engine.wasm" if env.get("public", True) else ""
    return {"key": mode, "release": {"environment_version": env["version"],
                                     "engine_object_key": key}}


class Api(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 — http.server's spelling
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/environment/list":
            payload = json.dumps({"environments": [
                {"id": slug, "modes": [_mode_entry(slug, m) for m in env["modes"]]}
                for slug, env in RELEASES.items()
            ]}).encode()
            self.send_response(200)
        else:
            env = RELEASES.get(body.get("id")) if self.path == "/environment/get" else None
            if env is None:
                payload = b'{"error":"environment not found"}'
                self.send_response(400)
            else:
                payload = json.dumps({"environment": {
                    "id": body["id"],
                    "modes": [_mode_entry(body["id"], m) for m in env["modes"]],
                }}).encode()
                self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep pytest output clean
        pass


class Cdn(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "environments" and parts[2] == "latest.json":
            body = RELEASES.get(parts[1])
            if body is not None:
                payload = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


def _serve(handler):
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture(scope="module")
def platform():
    api, cdn = _serve(Api), _serve(Cdn)
    yield f"http://127.0.0.1:{api.server_port}", f"http://127.0.0.1:{cdn.server_port}"
    api.shutdown()
    cdn.shutdown()


@pytest.fixture(autouse=True)
def point_at_mock(platform, monkeypatch):
    api, cdn = platform
    monkeypatch.setenv("LOCKSTEP_API_URL", api)
    monkeypatch.setenv("LOCKSTEP_CDN_URL", cdn)
    monkeypatch.delenv("LOCKSTEP_API_KEY", raising=False)


def test_resolves_version_default_mode_and_tokened_prefix(platform):
    _, cdn = platform
    release = resolve("two-mode")
    assert release.version == "0.7.0"
    assert release.mode == "alpha"
    assert release.engine_url == f"{cdn}/{prefix('two-mode', 'alpha')}/engine.wasm"
    # The generic shell rides the SAME release directory.
    assert release.agent_shell_url.endswith(f"{TOKEN}/alpha/agent-onnx.wasm")


def test_mode_override():
    release = resolve("two-mode", mode="beta")
    assert release.mode == "beta"
    assert release.prefix.endswith("/beta")


def test_version_is_an_assertion_on_the_current_release():
    assert resolve("two-mode", version="0.7.0").version == "0.7.0"
    with pytest.raises(SystemExit, match="VERSION=0.6.0 is not published"):
        resolve("two-mode", version="0.6.0")


def test_unknown_mode_is_refused_with_the_mode_list():
    with pytest.raises(SystemExit, match="alpha, beta"):
        resolve("two-mode", mode="nope")


def test_multi_mode_without_default_demands_mode():
    with pytest.raises(SystemExit, match="pass --mode"):
        resolve("no-default")


def test_single_mode_needs_no_default():
    assert resolve("solo").mode == "default"


def test_unknown_environment_says_so():
    with pytest.raises(SystemExit, match="environment slug"):
        resolve("no-such-env")


def test_published_lists_only_environments_with_a_published_mode():
    envs = {e.slug: e for e in published()}
    assert "unpublished" not in envs, "a mode with no release is not published"
    assert set(envs) == {"two-mode", "no-default", "solo", "sealed"}
    assert envs["two-mode"].modes == ("alpha", "beta")
    assert envs["solo"].modes == ("default",)
    assert envs["two-mode"].public and envs["solo"].public
    # Published, but no public engine key: fetchable only with an invite key.
    assert envs["sealed"].modes == ("default",)
    assert not envs["sealed"].public
    # Platform order is preserved — `--first` depends on it.
    assert [e.slug for e in published()] == ["two-mode", "no-default", "solo", "sealed"]


def test_cli_lists_every_published_environment(capsys):
    assert discovery_main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [ln.split()[0] for ln in lines] == ["two-mode", "no-default", "solo", "sealed"]
    assert "modes: alpha, beta" in lines[0]
    assert "(invite key required)" in lines[3] and "(invite key required)" not in lines[0]


def test_cli_first_picks_the_first_public_slug(capsys):
    assert discovery_main(["--first"]) == 0
    assert capsys.readouterr().out.strip() == "two-mode"


def test_cli_first_fails_clearly_when_nothing_is_public(monkeypatch, capsys):
    from train.core import discovery

    sealed_only = [e for e in published() if not e.public]
    monkeypatch.setattr(discovery, "published", lambda: sealed_only)
    assert discovery.main(["--first"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no publicly published environment" in captured.err
