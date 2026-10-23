"""A source rebuilt from a native admission envelope names the bot that received it.

Native admission routes from ``restore_native_source`` (the envelope), not from the live source.
``restore_provenance`` re-verified the receiving adapter but never pinned the routing identity, so
``transport_profile_of`` returned None and a lane created by that source — the first message of a
new Telegram DM topic, on a multiplexed gateway where each profile has its own bot — persisted
``transport_profile`` NULL in the routing entry and the sessions row, for good.
"""
import weakref
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership
from gateway.session import SessionSource, SessionStore
from gateway.session_identity import identity_of, transport_profile_of
from gateway.session_ingress_context import (
    capture_provenance, native_callback, register_transport_home, restore_provenance)


class _FakeAdapter:
    """Weakref-able stand-in for a profile's own live Telegram bot."""


@pytest.mark.asyncio
async def test_restored_topic_source_creates_its_lane_with_the_receiving_bot(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    liz_home = root / "profiles" / "liz"
    liz_home.mkdir(parents=True, mode=0o700)
    (liz_home / "config.yaml").write_text("{}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [("default", root), ("liz", liz_home)]
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    process_ownership.reserve([home for _, home in homes])
    try:
        await initialize_gateway_runtime(runner)
        register_transport_home(runner, "liz", liz_home)
        bot = _FakeAdapter()
        runner.adapters = {}  # no primary Telegram bot: every profile receives on its own
        runner._profile_adapters = {"liz": {Platform.TELEGRAM: bot}}
        live = SessionSource(
            platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm", user_id="4242",
            thread_id="97317", profile="liz")
        live._transport_adapter_ref = weakref.ref(bot)
        event = MessageEvent(text="hi", message_type=MessageType.TEXT, source=live, message_id="m1")
        with native_callback(runner, event, liz_home, "liz"):
            provenance = capture_provenance(runner, event)

        restored = SessionSource.from_dict(live.to_dict())  # what restore_native_source rebuilds
        restore_provenance(runner, restored, provenance)
        assert transport_profile_of(restored) == "liz"
        assert identity_of(restored).adapter() is bot and not identity_of(restored).transport_inferred

        store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
        lane = store.get_or_create_session(restored)
        assert lane.session_key.endswith(":97317") and lane.transport_profile == "liz"
        assert store._db_for_key(lane.session_key).get_session(lane.session_id)["transport_profile"] == "liz"
    finally:
        for authority in list(getattr(runner, "session_authorities", []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
