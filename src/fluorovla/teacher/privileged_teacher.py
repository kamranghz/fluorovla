"""Privileged Teacher (anatomy-aware demonstration generator): rule-based, insertion-only, patient independent.

TEACHER INPUT (privileged simulator state; never a policy input):
    tip_extremity_mm   physical distal extremity arclength along the centerline
    zone_near_mm, zone_far_mm   landing-zone bounds (extremity coordinate)
    (tip velocity, inserted length, active bodies and lumen state are logged but the rule does not need them)

POLICY INPUT (the only things the future GR00T policy may see) = the existing observation contract.

Rule (one decision per 0.1 s, no future state):
    remaining = zone_near - tip_extremity
    CRUISE while remaining > CRUISE_MARGIN, SLOW while remaining > 0, HOLD once the extremity is inside the zone.
CRUISE_MARGIN = 2 * V_MAX * DT = 2.4 mm: two CRUISE decisions of lag, so the last approach is made at SLOW.
"""
from __future__ import annotations

from dataclasses import dataclass

from fluorovla.task_contract import DECISION_DT_S, V_MAX_MPS

HOLD_MPS, SLOW_MPS, CRUISE_MPS = 0.0, 0.006, 0.012
CRUISE_MARGIN_MM = 2.0 * V_MAX_MPS * DECISION_DT_S * 1000.0

POLICY_OBSERVATION_KEYS = ("observation.images.fluoro", "inserted_length_m", "previous_insert_velocity_mps", "annotation.human.task_description")
TEACHER_ONLY_KEYS = ("tip_extremity_mm", "zone_near_mm", "zone_far_mm", "tip_speed_mps", "n_active", "axis_violation_mm", "centerline_s_mm")
assert not set(POLICY_OBSERVATION_KEYS) & set(TEACHER_ONLY_KEYS)


@dataclass(frozen=True)
class TeacherState:
    tip_extremity_mm: float
    zone_near_mm: float
    zone_far_mm: float


def teacher_action(st: TeacherState) -> tuple[str, float]:
    remaining = st.zone_near_mm - st.tip_extremity_mm
    if remaining > CRUISE_MARGIN_MM:
        return "CRUISE", CRUISE_MPS
    if remaining > 0.0:
        return "SLOW", SLOW_MPS
    return "HOLD", HOLD_MPS
