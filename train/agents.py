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

``AGENT=`` may be omitted when exactly one agent exists — the common case —
and every error here says exactly what to type instead.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
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
    )


def write_agent_toml(cfg: AgentConfig) -> Path:
    cfg.dir.mkdir(parents=True, exist_ok=True)
    path = cfg.dir / "agent.toml"
    path.write_text(agent_toml_text(cfg))
    return path


def load_agent(name: str, root: Path = AGENTS_ROOT) -> AgentConfig:
    path = root / name / "agent.toml"
    if not path.is_file():
        known = ", ".join(a.name for a in list_agents(root)) or "none yet"
        raise SystemExit(
            f"no agent at {path} (agents present: {known}) — create one: "
            f"task create-agent NAME={name} ENV=<slug>"
        )
    raw = tomllib.loads(path.read_text())
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
