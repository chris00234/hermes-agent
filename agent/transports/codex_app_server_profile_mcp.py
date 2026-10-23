"""Per-profile ``HERMES_HOME`` for the MCP servers codex launches on the app-server runtime.

Hermes' own MCP client launches each enabled stdio server a profile declares with that
profile's ``HERMES_HOME`` (``tools/mcp_tool_config.py::_build_safe_env``). On
``codex_app_server`` those servers come from the shared codex ``config.toml``, launched with
codex's env allowlist plus the table's own env, so a server that keys its identity off
``HERMES_HOME`` acts as whichever profile last wrote the table (or as none). The per-server
``-c`` override that hermes-tools already gets restores that parity for every declared server
the codex config also defines.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path

from agent.transports.hermes_tools_mcp_server import HERMES_TOOLS_MCP_SERVER_NAME


def _declared_stdio_servers() -> set[str]:
    """Stdio servers the active profile declares and has not disabled."""
    from hermes_cli.config import load_config_readonly
    from tools.mcp_tool_common import mcp_server_enabled

    servers = load_config_readonly().get("mcp_servers") or {}
    if not isinstance(servers, dict):
        return set()
    return {
        name for name, spec in servers.items()
        if isinstance(spec, dict) and spec.get("command") and mcp_server_enabled(spec)
    }


def _codex_stdio_servers(spawn_env: Mapping[str, str]) -> set[str]:
    """Stdio servers the codex config the app-server will read defines (empty when unreadable)."""
    home = Path(spawn_env.get("HOME") or Path.home())
    config = Path(spawn_env.get("CODEX_HOME") or home / ".codex") / "config.toml"
    try:
        servers = tomllib.loads(config.read_text(encoding="utf-8-sig")).get("mcp_servers") or {}
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    if not isinstance(servers, dict):
        return set()
    return {name for name, spec in servers.items() if isinstance(spec, dict) and spec.get("command")}


def profile_mcp_home_overrides(spawn_env: Mapping[str, str], hermes_home: str) -> list[str]:
    """``-c mcp_servers.<name>.env.HERMES_HOME=...`` for each declared server codex also defines.

    Only names present in the codex config are targeted: an override for a missing name creates
    an env-only entry that codex refuses ("invalid transport")."""
    from hermes_cli.codex_runtime_plugin_migration import _quote_key

    shared = _codex_stdio_servers(spawn_env) - {HERMES_TOOLS_MCP_SERVER_NAME}
    if not shared:
        return []
    overrides: list[str] = []
    for name in sorted(shared & _declared_stdio_servers()):
        overrides += ["-c", f"mcp_servers.{_quote_key(name)}.env.HERMES_HOME={json.dumps(hermes_home)}"]
    return overrides
