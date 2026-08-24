"""CDN release resolution against a local mock — no network, no packages.

Serves a fake ``environments/<slug>/latest.json`` from a thread-local HTTP
server and points ``$LOCKSTEP_CDN_URL`` at it (the same variable
``lockstep_train`` reads, so the template and the library agree about where
"the CDN" is by construction).
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest

from train.core.discovery import resolve

LATEST = {
    "dance-off": {"version": "0.7.0", "default_mode": "servo-assist",
                  "modes": ["servo-assist", "raw-torque"]},
    # A multi-mode release that (wrongly) omits default_mode: the resolver
    # must demand MODE= rather than guess a ladder.
    "no-default": {"version": "1.0.0", "modes": ["a", "b"]},
    # A single mode needs no default to be unambiguous.
    "solo": {"version": "2.0.0", "modes": ["default"]},
}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — http.server's spelling
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "environments" and parts[2] == "latest.json":
            body = LATEST.get(parts[1])
            if body is not None:
                payload = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):  # keep pytest output clean
        pass


@pytest.fixture(scope="module")
def cdn(request):
    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture(autouse=True)
def point_at_mock(cdn, monkeypatch):
    monkeypatch.setenv("LOCKSTEP_CDN_URL", cdn)


def test_latest_resolves_version_and_default_mode(cdn):
    release = resolve("dance-off")
    assert release.version == "0.7.0"
    assert release.mode == "servo-assist"
    assert release.engine_url == (
        f"{cdn}/environments/dance-off/releases/0.7.0/servo-assist/engine.wasm"
    )
    # The generic shell rides the SAME release directory.
    assert release.agent_shell_url.endswith("/servo-assist/agent-onnx.wasm")


def test_version_and_mode_override():
    release = resolve("dance-off", version="0.6.0", mode="raw-torque")
    assert release.version == "0.6.0"
    assert release.mode == "raw-torque"


def test_unknown_mode_is_refused_with_the_mode_list():
    with pytest.raises(SystemExit, match="servo-assist, raw-torque"):
        resolve("dance-off", mode="nope")


def test_multi_mode_without_default_demands_mode():
    with pytest.raises(SystemExit, match="pass MODE="):
        resolve("no-default")


def test_single_mode_needs_no_default():
    assert resolve("solo").mode == "default"


def test_unknown_environment_says_so():
    with pytest.raises(SystemExit, match="does not exist"):
        resolve("no-such-env")
