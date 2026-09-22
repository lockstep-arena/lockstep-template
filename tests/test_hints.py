"""The closing hint of `task build` / `task train`: run, debug, submit — and
an opponents example sized to the roster, only when there is room for one."""

from train import hints


def test_opponents_follow_the_roster():
    assert hints.opponents_example(None) is None
    assert hints.opponents_example(1) is None
    assert hints.opponents_example(2) == 'OPPONENTS="other-bot"'
    assert hints.opponents_example(3) == 'OPPONENTS="bot-2 bot-3"'
    assert hints.opponents_example(4) == 'OPPONENTS="bot-2 bot-3 bot-4"'
    assert hints.opponents_example(8) == 'OPPONENTS="bot-2 bot-3 … bot-8"'


def test_next_steps_for_a_solo_and_a_duel_engine(monkeypatch, tmp_path):
    engine = tmp_path / "engine.wasm"
    monkeypatch.setattr(hints, "roster_max", lambda _engine: 1)
    solo = hints.next_steps("my-bot", engine)
    assert solo.splitlines() == [
        "Run it:               task match AGENT=my-bot",
        "Debug it:             task match AGENT=my-bot LOGS=1 STEP=1",
        "Submit final bundle:  task upload AGENT=my-bot",
    ]
    monkeypatch.setattr(hints, "roster_max", lambda _engine: 2)
    duel = hints.next_steps("my-bot", engine)
    assert duel.splitlines()[0] == (
        'Run it:               task match AGENT=my-bot   (vs. other bots: OPPONENTS="other-bot")'
    )
    assert "Compete" not in duel


def test_an_unreadable_engine_just_drops_the_opponents_hint(tmp_path):
    lines = hints.next_steps(None, tmp_path / "missing.wasm").splitlines()
    assert lines[0] == "Run it:               task match"
    assert lines[2] == "Submit final bundle:  task upload"
