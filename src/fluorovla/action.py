"""Backend-independent catheter progression Action API (frozen interface).

Semantic command only: what the catheter should do.
Backend-specific feed mechanisms (e.g. portal-edge rest length) must not appear here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CatheterAction:
    """Minimal progression action shared by controller, logger, VLA, and backends."""

    insert_velocity_mps: float

    def __post_init__(self) -> None:
        v = float(self.insert_velocity_mps)
        object.__setattr__(self, "insert_velocity_mps", v)


# Canonical named progression commands (SI). Validation criterion is physical
# command-dependent geometry, not merely different controller L_in values.
HOLD = CatheterAction(insert_velocity_mps=0.000)
SLOW = CatheterAction(insert_velocity_mps=0.006)
CRUISE = CatheterAction(insert_velocity_mps=0.012)

NAMED_ACTIONS: dict[str, CatheterAction] = {
    "HOLD": HOLD,
    "SLOW": SLOW,
    "CRUISE": CRUISE,
}


def action_to_dict(action: CatheterAction) -> dict[str, float]:
    """JSON-serializable demo/logger payload."""
    return {"insert_velocity_mps": float(action.insert_velocity_mps)}


def action_from_dict(data: Mapping[str, Any]) -> CatheterAction:
    """Rebuild action from a logged dict. Rejects backend-specific keys."""
    if "insert_velocity_mps" not in data:
        raise KeyError("CatheterAction requires insert_velocity_mps")
    banned = (
        "rest_lengths",
        "baseline_rest_lengths",
        "n_active_nodes",
        "L_in",
        "L_in_m",
        "backend",
        "torque",
        "steering",
    )
    for key in banned:
        if key in data:
            raise ValueError(f"backend-specific or out-of-scope field in action payload: {key}")
    return CatheterAction(insert_velocity_mps=float(data["insert_velocity_mps"]))


def named_action(name: str) -> CatheterAction:
    key = str(name).strip().upper()
    if key not in NAMED_ACTIONS:
        raise KeyError(f"unknown progression action {name!r}; expected one of {sorted(NAMED_ACTIONS)}")
    return NAMED_ACTIONS[key]


def as_loggable(action: CatheterAction) -> dict[str, float]:
    """Alias for demonstration logging."""
    return action_to_dict(action)
