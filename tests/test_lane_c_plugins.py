"""Private lifecycle fences on the extracted plugin dispatch path."""

import contextvars
import sys
import threading
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from agent import relay_runtime
from gateway import private_execution_policy as policy_module
from hermes_cli import lifecycle, observability, plugins
from hermes_cli.plugins_dispatch import _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE


@contextmanager
def private_context():
    policy = policy_module.build_policy(
        run_id="lane-c", profile="isolated", execution_attempt_id="attempt-c",
        lease_expires_at="0", allowed_toolsets={"terminal"},
    )
    token = policy_module.bind_private_policy(policy)
    try:
        yield
    finally:
        policy_module.reset_private_policy(token)


@pytest.mark.parametrize("mode", ["private", "accessor-error", "accessor-missing"])
@pytest.mark.parametrize("entry", [
    "facade", "manager", "lifecycle", "finalize", "has", "lifecycle-has",
    "manager-has", "iter", "manager-iter", "discover", "manager-discover",
    "background-discover",
])
def test_private_fence_precedes_callbacks_and_discovery(monkeypatch, mode, entry):
    manager = plugins.PluginManager()
    callback = Mock(return_value="leak")
    observer = Mock()
    relay = Mock()
    discovery = Mock()
    manager._hooks["on_session_finalize"] = [callback]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(manager, "_discover_and_load_inner", discovery)
    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", Mock())
    monkeypatch.setattr(observability, "observe_lifecycle", observer)
    monkeypatch.setattr(observability, "handles_hook", observer)
    monkeypatch.setattr(relay_runtime.SESSION_COORDINATOR, "finalize_conversation", relay)
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0)
    calls = {
        "facade": lambda: plugins.invoke_hook("on_session_finalize", session_id="private"),
        "manager": lambda: manager.invoke_hook("on_session_finalize", session_id="private"),
        "lifecycle": lambda: lifecycle.invoke_hook("on_session_finalize", session_id="private"),
        "finalize": lambda: lifecycle.finalize_session(session_id="private"),
        "has": lambda: plugins.has_hook("on_session_finalize"),
        "lifecycle-has": lambda: lifecycle.has_hook("on_session_finalize"),
        "manager-has": lambda: manager.has_hook("on_session_finalize"),
        "iter": lambda: plugins.iter_hook_callbacks("on_session_finalize"),
        "manager-iter": lambda: manager.iter_hook_callbacks("on_session_finalize"),
        "discover": plugins.discover_plugins,
        "manager-discover": lambda: manager.discover_and_load(force=True),
        "background-discover": plugins.start_background_plugin_discovery,
    }
    with private_context():
        if mode == "accessor-error":
            monkeypatch.setattr(policy_module, "current_private_policy", Mock(side_effect=RuntimeError))
        elif mode == "accessor-missing":
            monkeypatch.delattr(policy_module, "current_private_policy")
        assert not calls[entry]()
    callback.assert_not_called()
    observer.assert_not_called()
    relay.assert_not_called()
    discovery.assert_not_called()
    assert not manager._discovered


@pytest.mark.parametrize("hook", ["on_session_finalize", "post_tool_call", "subagent_stop"])
def test_public_delivery_narrow_kwargs_and_context_isolation(monkeypatch, hook):
    manager = plugins.PluginManager()
    discovery = Mock()
    events = []
    marker = contextvars.ContextVar("lane-c-marker", default=None)
    token = marker.set("public-marker")

    def narrow(session_id, *, additive="default"):
        events.append((marker.get(), policy_module.current_private_policy(), threading.current_thread()))
        return (session_id, additive)

    def broken(session_id):
        events.append("broken-once")
        raise TypeError("callback body error, not a signature mismatch")

    wide = Mock(return_value="wide")
    manager._hooks[hook] = [narrow, broken, wide]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(manager, "_discover_and_load_inner", discovery)
    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", Mock())
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 2.0)
    monkeypatch.setattr(observability, "observe_lifecycle", lambda name, **kw: events.append("observer"))
    try:
        with private_context():
            assert manager.invoke_hook(hook, session_id="private") == []
            result = contextvars.Context().run(lifecycle.invoke_hook, hook, session_id="parallel-public")
            assert result == [("parallel-public", "default"), "wide"]
        events.clear()
        assert lifecycle.invoke_hook(hook, session_id="public", additive="new", unknown=True) == [
            ("public", "new"), "wide",
        ]
    finally:
        marker.reset(token)
    discovery.assert_called_once()
    assert events[:2] == ["observer", ("public-marker", None, events[1][2])]
    assert (events[1][2] is threading.current_thread()) == (hook != "post_tool_call")
    assert events[2:] == ["broken-once"]
    assert wide.call_args.kwargs["unknown"] is True
    assert "telemetry_schema_version" in wide.call_args.kwargs


def test_public_finalization_preserves_observer_relay_plugin_order(monkeypatch):
    events = []
    manager = plugins.PluginManager()
    manager._discovered = True
    manager._hooks["on_session_finalize"] = [lambda session_id: events.append(("plugin", session_id)) or "ok"]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(observability, "observe_lifecycle", lambda name, **kw: events.append((name, kw)))
    monkeypatch.setattr(relay_runtime, "current_profile_key", lambda: "isolated")
    monkeypatch.setattr(relay_runtime.SESSION_COORDINATOR, "finalize_conversation", lambda **kw: events.append(("relay", kw)))
    assert lifecycle.finalize_session(session_id="public", messages=["payload"]) == ["ok"]
    assert events == [
        ("on_session_finalize", {"session_id": "public", "messages": ["payload"]}),
        ("relay", {"profile_key": "isolated", "session_id": "public"}),
        ("plugin", "public"),
    ]


@pytest.mark.parametrize("hook", ["pre_tool_call", "post_tool_call"])
def test_timeout_suppression_and_worker_cleanup(monkeypatch, hook):
    manager = plugins.PluginManager()
    hold = threading.Event()
    started = threading.Event()
    workers = []
    calls = []

    def callback(tool_name):
        workers.append(threading.current_thread())
        calls.append(tool_name)
        started.set()
        hold.wait(timeout=5)
        return "late"

    manager._hooks[hook] = [callback]
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0.05)
    expected = [{"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE}] if hook == "pre_tool_call" else []
    try:
        assert manager.invoke_hook(hook, tool_name="terminal", additive=True) == expected
        assert started.wait(timeout=2)
        assert manager.invoke_hook(hook, tool_name="terminal") == expected
        assert calls == ["terminal"]
    finally:
        hold.set()
        for worker in workers:
            worker.join(timeout=2)
            assert not worker.is_alive()
    assert not manager._hook_running_callbacks
    manager._hook_timeout_suppressed_until.clear()
    assert manager.invoke_hook(hook, tool_name="terminal") == ["late"]


def test_worker_start_failure_is_fail_closed_and_recovers(monkeypatch):
    manager = plugins.PluginManager()
    callback = Mock(return_value="ok")
    manager._hooks["pre_tool_call"] = [callback]
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 2.0)
    with monkeypatch.context() as patcher:
        patcher.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("no thread")))
        assert manager.invoke_hook("pre_tool_call") == [
            {"action": "block", "message": _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE},
        ]
    callback.assert_not_called()
    assert not manager._hook_running_callbacks
    assert manager.invoke_hook("pre_tool_call") == ["ok"]


def test_absent_policy_module_preserves_public_hooks(monkeypatch):
    manager = plugins.PluginManager()
    manager._hooks["gateway_platform_event"] = [lambda **kw: kw]
    monkeypatch.delitem(sys.modules, "gateway.private_execution_policy")
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0)
    assert manager.invoke_hook("gateway_platform_event", payload={"event": "public"}) == [
        {"payload": {"event": "public"}},
    ]
