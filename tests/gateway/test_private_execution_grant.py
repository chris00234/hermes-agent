"""Unit tests for the versioned v2 HMAC grant (mint/verify).

Grant format: ``v2.{attempt_id}.{expiry}.{toolsets}.{mac}`` where toolsets is
the comma-joined, sorted, deduplicated toolset list carried by the grant —
the sole source of a private run's allowed_toolsets. v1 grants are rejected
everywhere.
"""

import pytest

from gateway.private_execution_policy import (
    GRANT_VERSION,
    GrantError,
    mint_grant,
    verify_grant,
)

SECRET = "unit-test-grant-secret-0123456789abcdef"
PROFILE = "design"
NOW = 1_757_000_000
TOOLSETS = ("shell", "web_search")


def _grant(**overrides):
    kwargs = {
        "attempt_id": "attempt_1",
        "expiry": NOW + 600,
        "profile": PROFILE,
        "toolsets": TOOLSETS,
    }
    kwargs.update(overrides)
    return mint_grant(SECRET, **kwargs)


class TestVerifyGrant:
    def test_grant_version_is_v2(self):
        assert GRANT_VERSION == "v2"

    def test_valid_grant_returns_attempt_expiry_toolsets(self):
        attempt_id, expiry, toolsets = verify_grant(
            _grant(), SECRET, profile=PROFILE, now=NOW
        )
        assert attempt_id == "attempt_1"
        assert expiry == NOW + 600
        assert toolsets == TOOLSETS

    def test_dotted_attempt_id_roundtrips(self):
        grant = _grant(attempt_id="att.2026.09")
        attempt_id, _expiry, toolsets = verify_grant(
            grant, SECRET, profile=PROFILE, now=NOW
        )
        assert attempt_id == "att.2026.09"
        assert toolsets == TOOLSETS

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
            "v2",
            "v2.attempt_1",
            "v2.attempt_1.123",
            "v2.attempt_1.123.shell",
            "v2..123.shell." + "0" * 64,
            "v2.attempt_1.notanint.shell." + "0" * 64,
            "v2.attempt_1.٣٣٣.shell." + "0" * 64,
            "v2.attempt_1.123.shell.nothex",
            "v2.attempt_1.123.shell." + "Z" * 64,
            None,
            42,
        ],
    )
    def test_malformed_grant_rejected(self, header):
        with pytest.raises(GrantError, match="malformed"):
            verify_grant(header, SECRET, profile=PROFILE, now=NOW)

    @pytest.mark.parametrize(
        "segment",
        [
            "",  # empty toolsets segment: a grant must name >= 1 toolset
            "Shell",
            "9shell",
            "a" * 33,
            "shell,shell",  # duplicates: not the canonical dedup form
            "web_search,shell",  # unsorted: not the canonical sorted form
            ",".join(f"t{i:02d}" for i in range(17)),  # > 16 names
            "memory",
            "shell,delegation",
            "delegate_task",
            "message_agent",
        ],
    )
    def test_defective_toolsets_segment_rejected(self, segment):
        header = f"v2.attempt_1.{NOW + 600}.{segment}." + "0" * 64
        with pytest.raises(GrantError, match="toolsets"):
            verify_grant(header, SECRET, profile=PROFILE, now=NOW)

    @pytest.mark.parametrize(
        "header",
        [
            "v1.attempt_1.123." + "0" * 64,  # well-formed v1: lane retired
            "v1.attempt_1.123.shell." + "0" * 64,
            "v3.attempt_1.123.shell." + "0" * 64,
        ],
    )
    def test_non_v2_version_rejected(self, header):
        with pytest.raises(GrantError, match="version"):
            verify_grant(header, SECRET, profile=PROFILE, now=NOW)

    def test_tampered_toolsets_rejected(self):
        parts = _grant().rsplit(".", 2)
        tampered = f"{parts[0]}.terminal.{parts[2]}"
        with pytest.raises(GrantError, match="authentic"):
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


class TestMintGrant:
    def test_toolsets_segment_sorted_and_deduplicated(self):
        grant = _grant(toolsets=("web_search", "shell", "web_search"))
        assert grant.rsplit(".", 2)[1] == "shell,web_search"

    @pytest.mark.parametrize(
        "toolsets",
        [
            (),
            ("Shell",),
            ("shell", "memory"),
            ("delegation",),
            tuple(f"t{i:02d}" for i in range(17)),
        ],
    )
    def test_unmintable_toolsets_rejected(self, toolsets):
        with pytest.raises(GrantError, match="toolsets"):
            _grant(toolsets=toolsets)

    def test_minted_grant_verifies(self):
        grant = _grant(toolsets=("web_search", "shell"))
        _attempt, _expiry, toolsets = verify_grant(
            grant, SECRET, profile=PROFILE, now=NOW
        )
        assert toolsets == ("shell", "web_search")
