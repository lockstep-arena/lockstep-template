"""Release resolution against a local mock platform — no network, no packages.

Serves a fake ``environment/get`` (the API) plus ``latest.json`` (the CDN's
public pointer, read only for ``default_mode``) from thread-local HTTP
servers and points ``$LOCKSTEP_API_URL`` / ``$LOCKSTEP_CDN_URL`` at them —
the same variables ``lockstep_train`` reads, so the template and the library
agree about where "the platform" is by construction. Release prefixes are
tokened, as on the real CDN: nothing here derives a path from slug + version.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from train.core.discovery import resolve

TOKEN = "0123456789abcdef"
RELEASES = {
    "dance-off": {"version": "0.7.0", "default_mode": "servo-assist",
                  "modes": ["servo-assist", "raw-torque"]},
    # A multi-mode release that (wrongly) omits default_mode: the resolver
    # must demand MODE= rather than guess a ladder.
    "no-default": {"version": "1.0.0", "modes": ["a", "b"]},
    # A single mode needs no default to be unambiguous.
    "solo": {"version": "2.0.0", "modes": ["default"]},
}


def prefix(slug: str, mode: str) -> str:
    return f"environments/{slug}/releases/{RELEASES[slug]['version']}-{TOKEN}/{mode}"


class Api(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 — http.server's spelling
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        env = RELEASES.get(body.get("id")) if self.path == "/environment/get" else None
        if env is None:
            payload = b'{"error":"environment not found"}'
            self.send_response(400)
        else:
            payload = json.dumps({"environment": {"id": body["id"], "modes": [
                {"key": m, "release": {"environment_version": env["version"],
                                       "engine_object_key": f"{prefix(body['id'], m)}/engine.wasm"}}
                for m in env["modes"]
            ]}}).encode()
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
    release = resolve("dance-off")
    assert release.version == "0.7.0"
    assert release.mode == "servo-assist"
    assert release.engine_url == f"{cdn}/{prefix('dance-off', 'servo-assist')}/engine.wasm"
    # The generic shell rides the SAME release directory.
    assert release.agent_shell_url.endswith(f"{TOKEN}/servo-assist/agent-onnx.wasm")


def test_mode_override():
    release = resolve("dance-off", mode="raw-torque")
    assert release.mode == "raw-torque"
    assert release.prefix.endswith("/raw-torque")


def test_version_is_an_assertion_on_the_current_release():
    assert resolve("dance-off", version="0.7.0").version == "0.7.0"
    with pytest.raises(SystemExit, match="VERSION=0.6.0 is not published"):
        resolve("dance-off", version="0.6.0")


def test_unknown_mode_is_refused_with_the_mode_list():
    with pytest.raises(SystemExit, match="servo-assist, raw-torque"):
        resolve("dance-off", mode="nope")


def test_multi_mode_without_default_demands_mode():
    with pytest.raises(SystemExit, match="pass --mode"):
        resolve("no-default")


def test_single_mode_needs_no_default():
    assert resolve("solo").mode == "default"


def test_unknown_environment_says_so():
    with pytest.raises(SystemExit, match="environment slug"):
        resolve("no-such-env")
