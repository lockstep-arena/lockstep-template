"""`task doctor`'s API-key check: optional until an agent targets an
assessment-only environment or the API rejects the key — then required, with
the fix. The API is stubbed; an unreachable API never fails the check."""

import pytest

from train import doctor

GATED = {"release": {"engine_object_key": ""}}
PUBLIC = {"release": {"engine_object_key": "environments/x/releases/1/default/engine.wasm"}}


@pytest.fixture
def repo(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, "ROOT", tmp_path)
    monkeypatch.delenv("LOCKSTEP_API_KEY", raising=False)
    return tmp_path


def agent(root, name, env):
    (root / "agents" / name).mkdir(parents=True)
    (root / "agents" / name / "agent.toml").write_text(
        f'[agent]\nname = "{name}"\nenv = "{env}"\nmode = "default"\nlang = "python"\n'
    )


def api(monkeypatch, *, profile=200, modes=None):
    """Stub `_api_post`: `profile` is account/profile's status (0 = unreachable);
    `modes` maps an env slug to its anonymous environment/get modes."""

    def post(route, body, key=""):
        if route == "account/profile":
            return profile, {}
        if modes is None:
            return 0, {}
        found = modes.get(body["id"])
        return (200, {"environment": {"modes": found}}) if found is not None else (404, {})

    monkeypatch.setattr(doctor, "_api_post", post)


def test_no_key_and_no_gated_agent_is_optional(repo, monkeypatch):
    agent(repo, "open-bot", "open-env")
    api(monkeypatch, modes={"open-env": [PUBLIC]})
    c = doctor.check_api_key()
    assert (c.ok, c.required) == (False, False)
    assert "assessment-only" in c.detail and c.fix


def test_no_key_with_an_assessment_only_agent_is_required(repo, monkeypatch):
    agent(repo, "open-bot", "open-env")
    agent(repo, "gated-bot", "gated-env")
    api(monkeypatch, modes={"open-env": [PUBLIC], "gated-env": [PUBLIC, GATED]})
    c = doctor.check_api_key()
    assert (c.ok, c.required) == (False, True)
    assert "agents/gated-bot targets gated-env" in c.detail


def test_key_from_dotenv_is_verified(repo, monkeypatch):
    (repo / ".env").write_text("LOCKSTEP_API_KEY=abc\n")
    api(monkeypatch, profile=200)
    c = doctor.check_api_key()
    assert c.ok and "verified" in c.detail and "not verified" not in c.detail


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_is_required_with_a_fix(repo, monkeypatch, status):
    monkeypatch.setenv("LOCKSTEP_API_KEY", "abc")
    api(monkeypatch, profile=status)
    c = doctor.check_api_key()
    assert (c.ok, c.required) == (False, True)
    assert "rejects" in c.detail and "mint a new key" in c.fix


def test_unreachable_api_never_fails(repo, monkeypatch):
    monkeypatch.setenv("LOCKSTEP_API_KEY", "abc")
    api(monkeypatch, profile=0)
    assert doctor.check_api_key().ok

    monkeypatch.delenv("LOCKSTEP_API_KEY")
    agent(repo, "gated-bot", "gated-env")
    api(monkeypatch, profile=0, modes=None)
    c = doctor.check_api_key()
    assert (c.ok, c.required) == (False, False)
