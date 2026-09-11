"""Keep the native thread binding across Hermes agent eviction and process restart."""

from __future__ import annotations

_BINDING_KEY = "codex_app_server_thread"


def load_thread_id(agent) -> str | None:
    binding = getattr(agent, "_codex_thread_binding", None)
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    if db is not None and session_id:
        binding = db.get_session_model_config_value(session_id, _BINDING_KEY)
    if not isinstance(binding, dict) or binding.get("session_id") != session_id:
        return None
    thread_id = binding.get("thread_id")
    return thread_id if isinstance(thread_id, str) and thread_id else None


def save_thread_id(agent, thread_id: str) -> None:
    session_id = getattr(agent, "session_id", None)
    binding = {"session_id": session_id, "thread_id": thread_id}
    agent._codex_thread_binding = binding
    db = getattr(agent, "_session_db", None)
    if db is not None and session_id:
        db.patch_session_model_config(session_id, {_BINDING_KEY: binding})
