"""Admission-minted immutable execution policy for the private /v1/runs lane.

Modeled on ``gateway/hosted_room_execution_policy.py`` (frozen dataclass +
ContextVar bind/reset/current). Authenticity comes from the versioned HMAC
grant (``verify_grant``); ``policy_digest`` is an integrity check only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone, UTC
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

GRANT_VERSION = "v2"
GRANT_SECRET_ENV = "PRIVATE_EXEC_GRANT_KEY"
MAX_GRANT_TOOLSETS = 16

# A delegated child agent or cross-agent message would not inherit the policy
# fences, and the memory toolset is a durable sink — always stripped.
# TOOLSET names ("delegation" holds delegate_task; message_agent is injected,
# not registered under any toolset) plus the tool names themselves in case a
# caller passes tool names as toolsets.
ALWAYS_EXCLUDED_TOOLSETS = frozenset(
    {"memory", "delegation", "delegate_task", "message_agent"}
)

# Per-TOOL denylist enforced at the tool gate (_authorized_dispatch and the
# invoke_tool seam) regardless of which toolset the registry maps them to.
ALWAYS_DENIED_TOOLS = frozenset({"delegate_task", "message_agent"})

_ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_EXPIRY_RE = re.compile(r"^[0-9]{1,20}$")
_MAC_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_TOOLSET_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class GrantError(ValueError):
    """A private execution grant is missing, malformed, or not authentic."""


@dataclass
class _LeaseExtension:
    # Written only by the gateway event loop; shared with already-running tool threads.
    expires_at: str = ""


@dataclass(frozen=True)
class PrivateExecutionPolicy:
    """Immutable per-run policy applied at the agent and tool boundaries."""

    run_id: str
    profile: str
    execution_attempt_id: str
    lease_expires_at: str
    allowed_toolsets: frozenset[str]
    policy_digest: str
    grant_version: str
    _lease: _LeaseExtension = field(default_factory=_LeaseExtension, compare=False, repr=False)

    @property
    def effective_lease_expires_at(self) -> str:
        return self._lease.expires_at or self.lease_expires_at


def renew_private_lease(
    policy: PrivateExecutionPolicy, header_value: str, secret: str | None, *, profile: str, now: float,
) -> str:
    """Extend a live lease without changing the admission identity, digest or toolsets."""
    attempt, expiry, toolsets = verify_grant(header_value, secret, profile=profile, now=now)
    if (profile, attempt, frozenset(toolsets)) != (
        policy.profile, policy.execution_attempt_id, policy.allowed_toolsets,
    ):
        raise GrantError("renewal must preserve the execution identity and toolsets")
    current = datetime.fromisoformat(policy.effective_lease_expires_at).timestamp()
    if current <= now or expiry < current or expiry > now + 1800:
        raise GrantError("renewal must extend a live lease by at most 30 minutes")
    policy._lease.expires_at = datetime.fromtimestamp(expiry, tz=UTC).isoformat()
    return policy._lease.expires_at


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def build_policy(
    *,
    run_id: str,
    profile: str,
    execution_attempt_id: str,
    lease_expires_at: str,
    allowed_toolsets: Iterable[str],
    grant_version: str = GRANT_VERSION,
) -> PrivateExecutionPolicy:
    """Derive the immutable policy server-side; excluded toolsets never enter."""

    allowed = frozenset(str(item) for item in allowed_toolsets)
    allowed -= ALWAYS_EXCLUDED_TOOLSETS
    unsigned = {
        "run_id": str(run_id),
        "profile": str(profile),
        "execution_attempt_id": str(execution_attempt_id),
        "lease_expires_at": str(lease_expires_at),
        "allowed_toolsets": sorted(allowed),
        "grant_version": str(grant_version),
    }
    digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    return PrivateExecutionPolicy(
        run_id=unsigned["run_id"],
        profile=unsigned["profile"],
        execution_attempt_id=unsigned["execution_attempt_id"],
        lease_expires_at=unsigned["lease_expires_at"],
        allowed_toolsets=allowed,
        policy_digest=digest,
        grant_version=unsigned["grant_version"],
    )


def _normalize_grant_toolsets(toolsets: Iterable[str]) -> tuple[str, ...]:
    """Canonical (sorted, deduplicated) toolset names for the grant segment."""

    names = sorted({str(item) for item in toolsets})
    if not 1 <= len(names) <= MAX_GRANT_TOOLSETS:
        raise GrantError("grant toolsets are invalid")
    for name in names:
        if (
            _TOOLSET_NAME_RE.fullmatch(name) is None
            or name in ALWAYS_EXCLUDED_TOOLSETS
        ):
            raise GrantError("grant toolsets are invalid")
    return tuple(names)


def _parse_grant_toolsets_segment(segment: str) -> tuple[str, ...]:
    """Parse a grant's toolsets segment; only the canonical form verifies."""

    names = tuple(segment.split(","))
    if _normalize_grant_toolsets(names) != names:
        raise GrantError("grant toolsets are invalid")
    return names


def _grant_mac(
    secret: str, *, attempt_id: str, expiry: int, profile: str, toolsets_segment: str
) -> str:
    message = (
        f"{GRANT_VERSION}|{attempt_id}|{expiry}|{profile}|{toolsets_segment}".encode()
    )
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def mint_grant(
    secret: str,
    *,
    attempt_id: str,
    expiry: int,
    profile: str,
    toolsets: Iterable[str],
) -> str:
    """Mint a ``v2`` grant header value (tests and the Brain bridge)."""

    expiry = int(expiry)
    segment = ",".join(_normalize_grant_toolsets(toolsets))
    mac = _grant_mac(
        secret,
        attempt_id=attempt_id,
        expiry=expiry,
        profile=profile,
        toolsets_segment=segment,
    )
    return f"{GRANT_VERSION}.{attempt_id}.{expiry}.{segment}.{mac}"


def _split_grant(header_value: Any) -> tuple[str, str, str, str]:
    """Structural parse: ``(attempt_id, raw_expiry, toolsets_segment, mac)``."""

    if not isinstance(header_value, str):
        raise GrantError("grant is malformed")
    version, sep, rest = header_value.strip().partition(".")
    if not sep:
        raise GrantError("grant is malformed")
    if version != GRANT_VERSION:
        raise GrantError("grant version is unsupported")
    # attempt_id may contain dots; expiry/toolsets/mac never do, so parse
    # the fixed tail from the right.
    parts = rest.rsplit(".", 3)
    if len(parts) != 4:
        raise GrantError("grant is malformed")
    attempt_id, raw_expiry, toolsets_segment, supplied_mac = parts
    if (
        _ATTEMPT_ID_RE.fullmatch(attempt_id) is None
        or _EXPIRY_RE.fullmatch(raw_expiry) is None
        or _MAC_HEX_RE.fullmatch(supplied_mac) is None
    ):
        raise GrantError("grant is malformed")
    return attempt_id, raw_expiry, toolsets_segment, supplied_mac


def verify_grant(
    header_value: Any,
    secret: str | None,
    *,
    profile: str,
    now: float,
) -> tuple[str, int, tuple[str, ...]]:
    """Validate a grant header; return ``(attempt_id, expiry, toolsets)``.

    Fail-closed: no usable secret means the lane is disabled and every grant
    is rejected; ``v1`` (and any non-``v2``) grants are rejected everywhere.
    Raises :class:`GrantError` on any defect.
    """

    if not isinstance(secret, str) or len(secret) < 16:
        raise GrantError("private execution lane is disabled")
    attempt_id, raw_expiry, toolsets_segment, supplied_mac = _split_grant(
        header_value
    )
    toolsets = _parse_grant_toolsets_segment(toolsets_segment)
    expiry = int(raw_expiry)
    expected = _grant_mac(
        secret,
        attempt_id=attempt_id,
        expiry=expiry,
        profile=profile,
        toolsets_segment=toolsets_segment,
    )
    if not hmac.compare_digest(supplied_mac, expected):
        raise GrantError("grant is not authentic")
    if expiry <= now:
        raise GrantError("grant is expired")
    return attempt_id, expiry, toolsets


def resolve_private_grant_secret(profile: str | None) -> str | None:
    """Resolve the profile-scoped grant secret; ``None`` disables the lane."""

    try:
        from agent.secret_scope import get_secret
        from hermes_cli.auth import has_usable_secret

        key = get_secret(GRANT_SECRET_ENV, "") or ""
        if not has_usable_secret(key, min_length=16):
            return None
        return key
    except Exception as exc:
        # Fail closed if the profile scope or strength guard cannot resolve
        # the credential. Do not log the key or exception text.
        logger.warning(
            "Failed to resolve a usable profile-scoped %s for %r: %s",
            GRANT_SECRET_ENV,
            profile,
            type(exc).__name__,
        )
        return None


_CURRENT_PRIVATE_POLICY: ContextVar[PrivateExecutionPolicy | None] = ContextVar(
    "private_execution_policy",
    default=None,
)


def bind_private_policy(policy: PrivateExecutionPolicy | None) -> Token:
    return _CURRENT_PRIVATE_POLICY.set(policy)


def reset_private_policy(token: Token) -> None:
    _CURRENT_PRIVATE_POLICY.reset(token)


def current_private_policy() -> PrivateExecutionPolicy | None:
    return _CURRENT_PRIVATE_POLICY.get()


__all__ = [
    "ALWAYS_DENIED_TOOLS",
    "ALWAYS_EXCLUDED_TOOLSETS",
    "GRANT_SECRET_ENV",
    "GRANT_VERSION",
    "GrantError",
    "PrivateExecutionPolicy",
    "bind_private_policy",
    "build_policy",
    "current_private_policy",
    "mint_grant",
    "reset_private_policy",
    "resolve_private_grant_secret",
    "verify_grant",
]
