"""Canonical policy names and backward-compatible aliases."""

# Compatibility only: these names existed during development and may still be
# present in older checkpoints/configs. Runtime code should use the WSA names.
from logging import getLogger

logger = getLogger(__name__)

WSA_BASE = "WSA_Base"
WSA_LARGE = "WSA_Large"

TBOT_SA1 = WSA_BASE
TBOT_SA1_WAN = WSA_LARGE

WSA_BASE_LEGACY_ALIASES = frozenset({"TBot_SA1", "tbot_sa1", "cubev2", "magicbot"})
WSA_LARGE_LEGACY_ALIASES = frozenset(
    {"TBot_SA1_Wan", "tbot_sa1_wan", "MagicBot_R0", "magicbot_r0", "magicbot-r0"}
)

TBOT_SA1_LEGACY_ALIASES = WSA_BASE_LEGACY_ALIASES
TBOT_SA1_WAN_LEGACY_ALIASES = WSA_LARGE_LEGACY_ALIASES

WSA_BASE_ALIASES = frozenset({WSA_BASE, "wsa_base", *WSA_BASE_LEGACY_ALIASES})
WSA_LARGE_ALIASES = frozenset(
    {WSA_LARGE, "wsa_large", *WSA_LARGE_LEGACY_ALIASES}
)
TBOT_SA1_ALIASES = WSA_BASE_ALIASES
TBOT_SA1_WAN_ALIASES = WSA_LARGE_ALIASES

_LOGGED_LEGACY_POLICY_TYPES: set[str] = set()


def is_wsa_base(policy_type: str | None) -> bool:
    return policy_type in WSA_BASE_ALIASES


def is_wsa_large(policy_type: str | None) -> bool:
    return policy_type in WSA_LARGE_ALIASES


def is_tbot_sa1(policy_type: str | None) -> bool:
    return is_wsa_base(policy_type)


def is_tbot_sa1_wan(policy_type: str | None) -> bool:
    return is_wsa_large(policy_type)


def legacy_policy_target(policy_type: str | None) -> str | None:
    if policy_type in WSA_BASE_LEGACY_ALIASES:
        return WSA_BASE
    if policy_type in WSA_LARGE_LEGACY_ALIASES:
        return WSA_LARGE
    return None


def log_legacy_policy_name(policy_type: str | None) -> None:
    target = legacy_policy_target(policy_type)
    if target is None or policy_type is None or policy_type in _LOGGED_LEGACY_POLICY_TYPES:
        return
    _LOGGED_LEGACY_POLICY_TYPES.add(policy_type)
    logger.info(
        "%s is the old development name of %s; using %s as the canonical policy name.",
        policy_type,
        target,
        target,
    )


def canonical_policy_type(policy_type: str | None) -> str | None:
    if is_wsa_base(policy_type):
        log_legacy_policy_name(policy_type)
        return WSA_BASE
    if is_wsa_large(policy_type):
        log_legacy_policy_name(policy_type)
        return WSA_LARGE
    return policy_type
