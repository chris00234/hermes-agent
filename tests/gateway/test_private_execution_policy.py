"""Unit tests for the private-execution policy object and grant verification.

DE_DESIGN §8 step 1: frozen policy dataclass, ContextVar accessors, versioned
HMAC grant mint/verify, always-excluded toolsets, and fail-closed secret
resolution for the private /v1/runs lane.
"""

import dataclasses
import threading

import pytest

from gateway.private_execution_policy import (
    ALWAYS_EXCLUDED_TOOLSETS,
    GRANT_SECRET_ENV,
    GRANT_VERSION,
    GrantError,
    PrivateExecutionPolicy,
    bind_private_policy,
    build_policy,
    current_private_policy,
    mint_grant,
    reset_private_policy,
    resolve_private_grant_secret,
    verify_grant,
)
from tools.thread_context import propagate_context_to_thread

SECRET = "unit-test-grant-secret-0123456789abcdef"
PROFILE = "design"
NOW = 1_757_000_000


def _grant(**overrides):
    kwargs = {"attempt_id": "attempt_1", "expiry": NOW + 600, "profile": PROFILE}
    kwargs.update(overrides)
    return mint_grant(SECRET, **kwargs)


class TestVerifyGrant:
    def test_valid_grant_returns_attempt_and_expiry(self):
        attempt_id, expiry = verify_grant(
            _grant(), SECRET, profile=PROFILE, now=NOW
        )
        assert attempt_id == "attempt_1"
        assert expiry == NOW + 600

    def test_expired_grant_rejected(self):
        grant = _grant(expiry=NOW - 1)
        with pytest.raises(GrantError, match="expired"):
            verify_grant(grant, SECRET, profile=PROFILE, now=NOW)

    def test_expiry_equal_to_now_rejected(self):
        grant = _grant(expiry=NOW)
        with pytest.raises(GrantError, match="expired"):
            verify_grant(grant, SECRET, profile=PROFILE, now=NOW)

    @pytest.mark.parametrize(
        "header",
        [
            "",
            "v1",
            "v1.attempt_1",
            "v1.attempt_1.123",
            "v1.attempt_1.123.deadbeef.extra",
            "v1..123." + "0" * 64,
            "v1.attempt_1.notanint." + "0" * 64,
            "v1.attempt_1.٣٣٣." + "0" * 64,
            "v1.attempt_1.123.nothex",
            "v1.attempt_1.123." + "Z" * 64,
            None,
            42,
        ],
    )
    def test_malformed_grant_rejected(self, header):
        with pytest.raises(GrantError, match="malformed"):
            verify_grant(header, SECRET, profile=PROFILE, now=NOW)

    def test_unknown_version_rejected(self):
        tampered = "v2" + _grant()[2:]
        with pytest.raises(GrantError, match="version"):
            verify_grant(tampered, SECRET, profile=PROFILE, now=NOW)

    def test_wrong_profile_rejected(self):
        grant = _grant(profile="liz")
        with pytest.raises(GrantError, match="authentic"):
            verify_grant(grant, SECRET, profile=PROFILE, now=NOW)

    def test_tampered_mac_rejected(self):
        grant = _grant()
        head, mac = grant.rsplit(".", 1)
        flipped = ("0" if mac[0] != "0" else "1") + mac[1:]
        with pytest.raises(GrantError, match="authentic"):
            verify_grant(f"{head}.{flipped}", SECRET, profile=PROFILE, now=NOW)

    def test_wrong_secret_rejected(self):
        with pytest.raises(GrantError, match="authentic"):
            verify_grant(
                _grant(), "another-secret-value-0123456789ab", profile=PROFILE, now=NOW
            )

    def test_tampered_attempt_id_rejected(self):
        parts = _grant().split(".")
        parts[1] = "attempt_2"
        with pytest.raises(GrantError, match="authentic"):
            verify_grant(".".join(parts), SECRET, profile=PROFILE, now=NOW)

    @pytest.mark.parametrize("secret", [None, "", "short"])
    def test_missing_secret_fails_closed(self, secret):
        with pytest.raises(GrantError, match="disabled"):
            verify_grant(_grant(), secret, profile=PROFILE, now=NOW)


class TestBuildPolicy:
    def _policy(self, toolsets=("terminal", "web_search")):
        return build_policy(
            run_id="run_abc",
            profile=PROFILE,
            execution_attempt_id="attempt_1",
            lease_expires_at="2026-09-08T12:00:00+00:00",
            allowed_toolsets=toolsets,
        )

    def test_policy_is_frozen(self):
        policy = self._policy()
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.profile = "other"

    def test_grant_version_default(self):
        assert self._policy().grant_version == GRANT_VERSION

    def test_always_excluded_toolsets_stripped(self):
        policy = self._policy(
            toolsets=("terminal", "memory", "delegate_task", "message_agent")
        )
        assert policy.allowed_toolsets == frozenset({"terminal"})
        assert not policy.allowed_toolsets & ALWAYS_EXCLUDED_TOOLSETS

    def test_delegation_toolset_stripped(self):
        # delegate_task lives in the "delegation" TOOLSET — excluding only
        # the tool name would leave the toolset grantable.
        policy = self._policy(toolsets=("terminal", "delegation"))
        assert policy.allowed_toolsets == frozenset({"terminal"})

    def test_denied_tool_names_constant_covers_delegation_tools(self):
        from gateway.private_execution_policy import ALWAYS_DENIED_TOOLS

        assert {"delegate_task", "message_agent"} <= ALWAYS_DENIED_TOOLS

    def test_digest_stable_across_toolset_ordering(self):
        first = self._policy(toolsets=("web_search", "terminal"))
        second = self._policy(toolsets=("terminal", "web_search"))
        assert first == second
        assert first.policy_digest == second.policy_digest

    def test_digest_unaffected_by_stripped_toolsets(self):
        with_excluded = self._policy(toolsets=("terminal", "web_search", "memory"))
        without = self._policy(toolsets=("terminal", "web_search"))
        assert with_excluded.policy_digest == without.policy_digest

    def test_digest_changes_when_field_changes(self):
        base = self._policy()
        other = build_policy(
            run_id="run_xyz",
            profile=PROFILE,
            execution_attempt_id="attempt_1",
            lease_expires_at="2026-09-08T12:00:00+00:00",
            allowed_toolsets=("terminal", "web_search"),
        )
        assert base.policy_digest != other.policy_digest


class TestContextVarAccessors:
    def _policy(self):
        return build_policy(
            run_id="run_ctx",
            profile=PROFILE,
            execution_attempt_id="attempt_1",
            lease_expires_at="2026-09-08T12:00:00+00:00",
            allowed_toolsets=("terminal",),
        )

    def test_default_is_none(self):
        assert current_private_policy() is None

    def test_bind_reset_roundtrip(self):
        policy = self._policy()
        token = bind_private_policy(policy)
        try:
            assert current_private_policy() is policy
        finally:
            reset_private_policy(token)
        assert current_private_policy() is None

    def test_nested_reset_restores_outer(self):
        outer = self._policy()
        inner = self._policy()
        outer_token = bind_private_policy(outer)
        inner_token = bind_private_policy(inner)
        try:
            assert current_private_policy() is inner
            reset_private_policy(inner_token)
            assert current_private_policy() is outer
        finally:
            reset_private_policy(outer_token)

    def test_bare_thread_does_not_inherit_policy(self):
        policy = self._policy()
        seen = []
        token = bind_private_policy(policy)
        try:
            thread = threading.Thread(
                target=lambda: seen.append(current_private_policy())
            )
            thread.start()
            thread.join()
        finally:
            reset_private_policy(token)
        assert seen == [None]

    def test_propagated_thread_sees_policy(self):
        policy = self._policy()
        seen = []
        token = bind_private_policy(policy)
        try:
            thread = threading.Thread(
                target=propagate_context_to_thread(
                    lambda: seen.append(current_private_policy())
                )
            )
            thread.start()
            thread.join()
        finally:
            reset_private_policy(token)
        assert seen == [policy]


class TestResolveSecret:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv(GRANT_SECRET_ENV, raising=False)
        assert resolve_private_grant_secret(PROFILE) is None

    def test_short_secret_returns_none(self, monkeypatch):
        monkeypatch.setenv(GRANT_SECRET_ENV, "short")
        assert resolve_private_grant_secret(PROFILE) is None

    def test_usable_secret_returned(self, monkeypatch):
        monkeypatch.setenv(GRANT_SECRET_ENV, SECRET)
        assert resolve_private_grant_secret(PROFILE) == SECRET

    def test_resolution_error_fails_closed(self, monkeypatch):
        def _boom(name, default=None):
            raise RuntimeError("scope failure")

        monkeypatch.setattr("agent.secret_scope.get_secret", _boom)
        assert resolve_private_grant_secret(PROFILE) is None


def test_policy_dataclass_field_names():
    assert [f.name for f in dataclasses.fields(PrivateExecutionPolicy)] == [
        "run_id",
        "profile",
        "execution_attempt_id",
        "lease_expires_at",
        "allowed_toolsets",
        "policy_digest",
        "grant_version",
    ]
