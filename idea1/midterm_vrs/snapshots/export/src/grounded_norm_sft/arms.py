"""Arm registry guard (plan section 7, 10.2)."""

from __future__ import annotations

from .constants import ARMS, DIAGNOSTIC_ONLY_ARMS, TRAINABLE_ARMS


class ArmNotExecutable(RuntimeError):
    pass


def require_trainable_arm(arm: str) -> str:
    if arm in DIAGNOSTIC_ONLY_ARMS:
        raise ArmNotExecutable(
            f"{arm} training is stopped under plan v1.2 (10.2): a separate amendment must define shared "
            "eligibility/fallback, per-family feasibility coverage and drift handling first")
    if arm not in ARMS:
        raise ArmNotExecutable(f"unknown arm {arm!r}")
    if arm not in TRAINABLE_ARMS:
        raise ArmNotExecutable(f"{arm} is not a training arm")
    return arm
