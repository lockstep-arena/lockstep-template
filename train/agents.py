"""Named agents under ``agents/<name>/`` — discovery and ``agent.toml``.

An agent is a directory the scaffolder (``task create-agent``) laid out for
one (environment, mode, language). Its ``agent.toml`` is the identity every
other task reads — ``task train/build/match/upload AGENT=<name>`` resolve
the engine from it, so an agent can never be run against the wrong mode::

    [agent]
    name = "my-bot"
    env = "<slug>"
    mode = "<mode>"
    lang = "python"                    # python | rust | c

    [release]                          # what the scaffold was generated from
    environment_version = "0.8.0"
    payload_schema_version = 8

    [supervised]                       # only when the declaration reveals truth
    value = "feedback"                 # the feedback window and its columns …
    age_col = "age_ticks"
    decision_col = "your_decision"
    truth_col = "truth"
    valid_col = "valid"
    action = "decision"                # … label this action's first element
    # or, for an environment that scores through the lagged reward instead:
    # reward_lag_ticks = 40

The ``[supervised]`` block is written from the declaration (a value whose
columns carry the documented feedback names, or ``meta.reward_lag_ticks``)
and read by ``task train RECIPE=supervised`` — see ``train/core/supervised.py``.

``AGENT=`` may be omitted when exactly one agent exists — the common case —
and every error here says exactly what to type instead.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

AGENTS_ROOT = Path("agents")

LANGS = ("python", "rust", "c")


@dataclass(frozen=True)
class AgentConfig:
    """One ``agents/<name>/agent.toml``, parsed."""

    name: str
    env: str
    mode: str
    lang: str
    environment_version: str
    payload_schema_version: int
    #: The ``[supervised]`` block (label source for the supervised recipe),
    #: or ``None`` when the declaration reveals no truth to learn from.
    supervised: dict | None = field(default=None, compare=True)

    @property
    def dir(self) -> Path:
        return AGENTS_ROOT / self.name

    @property
    def out_dir(self) -> Path:
        """Per-agent build products (gitignored): checkpoints, onnx, bundle."""
        return self.dir / "out"

    @property
    def bundle_dir(self) -> Path:
        return self.out_dir / "bundle"


def _toml_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def supervised_toml(block: dict) -> str:
    """The ``[supervised]`` table: where the truth of a decision shows up,
    as the declaration names it."""
    lines = [
        "",
        "# The supervised recipe's label source (`task train RECIPE=supervised`):",
        "# generated from the declaration — a feedback window whose columns say",
        "# how old each resolved decision is, what you decided and what was true",
        "# (or the lag after which the reward scores a decision). `action` is",
        "# the declared action the truth trains, element 0.",
        "[supervised]",
    ]
    for k, v in block.items():
        if v is None:
            continue
        lines.append(f"{k} = {_toml_scalar(v)}")
    return "\n".join(lines) + "\n"


def agent_toml_text(cfg: AgentConfig) -> str:
    return (
        "# Written by `task create-agent` and refreshed on re-run — the one\n"
        "# record of what this agent IS. Every task resolves the engine from\n"
        "# here (task train/build/match/upload AGENT=" + cfg.name + "), so a\n"
        "# mode mismatch between your agent and the engine is impossible.\n"
        "\n"
        "[agent]\n"
        f'name = "{cfg.name}"\n'
        f'env = "{cfg.env}"\n'
        f'mode = "{cfg.mode}"\n'
        f'lang = "{cfg.lang}"\n'
        "\n"
        "# The release the interface files were generated from. `task\n"
        "# create-agent` refreshes them (and this stamp) against the current\n"
        "# release; a payload_schema_version bump on the platform marks\n"
        "# uploaded agents stale — regenerate and rebuild when that happens.\n"
        "[release]\n"
        f'environment_version = "{cfg.environment_version}"\n'
        f"payload_schema_version = {cfg.payload_schema_version}\n"
        + (supervised_toml(cfg.supervised) if cfg.supervised else "")
    )


def write_agent_toml(cfg: AgentConfig) -> Path:
    cfg.dir.mkdir(parents=True, exist_ok=True)
    path = cfg.dir / "agent.toml"
    path.write_text(agent_toml_text(cfg), encoding="utf-8")
    return path


def load_agent(name: str, root: Path = AGENTS_ROOT) -> AgentConfig:
    path = root / name / "agent.toml"
    if not path.is_file():
        known = ", ".join(a.name for a in list_agents(root)) or "none yet"
        raise SystemExit(
            f"no agent at {path} (agents present: {known}) — create one: "
            f"task create-agent NAME={name} ENV=<slug>"
        )
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    agent = raw.get("agent", {})
    release = raw.get("release", {})
    lang = agent.get("lang", "python")
    if lang not in LANGS:
        raise SystemExit(f"{path}: unknown lang {lang!r} (one of {', '.join(LANGS)})")
    return AgentConfig(
        name=agent.get("name", name),
        env=agent["env"],
        mode=agent.get("mode", ""),
        lang=lang,
        environment_version=release.get("environment_version", ""),
        payload_schema_version=int(release.get("payload_schema_version", 0)),
        supervised=dict(raw["supervised"]) if raw.get("supervised") else None,
    )


def list_agents(root: Path = AGENTS_ROOT) -> list[AgentConfig]:
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if (d / "agent.toml").is_file():
            out.append(load_agent(d.name, root))
    return out


def resolve_agent(name: str | None, root: Path = AGENTS_ROOT) -> AgentConfig:
    """``AGENT=`` resolution: the named agent, else the sole agent, else a
    helpful error listing what exists."""
    if name:
        return load_agent(name, root)
    agents = list_agents(root)
    if len(agents) == 1:
        return agents[0]
    if not agents:
        raise SystemExit(
            "no agents yet — create one: task create-agent NAME=my-bot ENV=<slug>"
        )
    names = ", ".join(a.name for a in agents)
    raise SystemExit(f"several agents exist ({names}) — say which: AGENT=<name>")


def import_agent_module(cfg: AgentConfig, name: str):
    """Import ``agents/<name>/<name>.py`` (``policy``, ``model``,
    ``interface``) as the agent's own top-level module: the agent dir goes
    on ``sys.path`` for the import (names like ``my-bot`` are not
    importable as packages) and any earlier agent's module of the same
    name is dropped first, so two agents never share one."""
    import importlib
    import sys

    for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
        del sys.modules[mod]
    sys.path.insert(0, str(cfg.dir.resolve()))
    try:
        importlib.invalidate_caches()
        return importlib.import_module(name)
    finally:
        sys.path.pop(0)


def policy_builder(cfg: AgentConfig):
    """The agent's network: ``model.build_policy`` from its ``model.py``
    when it has one (scaffolded by ``task create-agent``), else the stock
    :class:`train.core.policy.Policy` — one flat stream per value."""
    from .core.policy import Policy

    if (cfg.dir / "model.py").is_file():
        module = import_agent_module(cfg, "model")
        build = getattr(module, "build_policy", None)
        if build is None:
            raise SystemExit(f"{cfg.dir / 'model.py'} defines no build_policy(observation_space, action_space)")
        return build
    return Policy


def _main() -> None:
    """Taskfile plumbing: print one agent fact per line as shell vars.

    ``python -m train.agents shellvars [NAME]`` →
    ``AGENT_NAME=… AGENT_ENV=… AGENT_MODE=… AGENT_LANG=… AGENT_DIR=…
    AGENT_BUNDLE=…`` (eval-able)."""
    import sys

    args = sys.argv[1:]
    if not args or args[0] != "shellvars":
        raise SystemExit(__doc__)
    name = args[1] if len(args) > 1 and args[1] else None
    cfg = resolve_agent(name)
    print(f"AGENT_NAME={cfg.name}")
    print(f"AGENT_ENV={cfg.env}")
    print(f"AGENT_MODE={cfg.mode}")
    print(f"AGENT_LANG={cfg.lang}")
    print(f"AGENT_DIR={cfg.dir}")
    print(f"AGENT_BUNDLE={cfg.bundle_dir}")


if __name__ == "__main__":
    _main()
