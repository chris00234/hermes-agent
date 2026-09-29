"""Profile-isolated environments for detached Hermes actions and restart watchers."""

import os
from pathlib import Path

# Process-global listener settings survive the launch-profile residue strip, so a child for another
# profile must drop them or it binds the launcher's API port.
_API_SERVER_LISTENER_KEYS = ("API_SERVER_ENABLED", "API_SERVER_HOST", "API_SERVER_PORT", "API_SERVER_CORS_ORIGINS")


def profile_action_environment(
    target_home: Path | None, env_overrides: dict[str, str] | None = None, *, host: bool = False,
) -> dict[str, str]:
    """Build a child environment; its normal startup loads the target profile's own dotenv.

    ``host`` targets the default root's gateway: ``host_gateway_child_env`` supplies the default
    profile's secrets in place of the launcher's, and ``target_home`` is not consulted."""
    if host:
        from hermes_constants import get_default_hermes_root
        from tools.environments.local import _is_routed_home, host_gateway_child_env
        action_env = host_gateway_child_env()
        if _is_routed_home(get_default_hermes_root()):
            for key in _API_SERVER_LISTENER_KEYS:
                action_env.pop(key, None)
    elif target_home is None:
        action_env = dict(os.environ)
    else:
        from hermes_cli.env_loader import (
            _PROFILE_MANAGED_ENV_KEYS, _env_keys_defined_in_dotenv, get_secret_source_values,
        )
        from hermes_constants import (
            apply_subprocess_home_env, get_default_hermes_root, get_hermes_home, get_process_hermes_home,
        )
        from tools.environments.local import _is_routed_home, build_subprocess_env, strip_launch_profile_env
        action_env = build_subprocess_env(base=os.environ, scrub_secrets=True)
        profile_keys = set(_PROFILE_MANAGED_ENV_KEYS)
        source_homes = {str(home) for home in (get_default_hermes_root(), get_hermes_home(), get_process_hermes_home())}
        for source_home in source_homes:
            profile_keys.update(_env_keys_defined_in_dotenv(Path(source_home) / ".env"))
            profile_keys.update(get_secret_source_values(source_home).keys())
        for key in profile_keys:
            action_env.pop(key, None)
        strip_launch_profile_env(action_env, target_home)
        routed = _is_routed_home(target_home)
        for key in _API_SERVER_LISTENER_KEYS:
            if routed or key in profile_keys:
                action_env.pop(key, None)
            elif key in os.environ:
                action_env[key] = os.environ[key]
        action_env["HERMES_HOME"] = str(target_home)
        apply_subprocess_home_env(action_env)
    action_env["HERMES_NONINTERACTIVE"] = "1"
    from gateway.config_loader import drop_bridged_env
    drop_bridged_env(action_env)
    for marker in ("_HERMES_GATEWAY", "HERMES_GATEWAY", "HERMES_GATEWAY_MODE"):
        action_env.pop(marker, None)
    if env_overrides:
        action_env.update(env_overrides)
    return action_env
