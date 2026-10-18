"""Newton adapter: map CatheterAction → shared Newton material-feed engine.

Does not reimplement portal feed, activation, or remainder bookkeeping.
Those remain in ``fluorovla.physics.engine``.

Supports explicit feed modes:
  - free_space_portal_slide (validated)
  - patient_distal_joint (validated)

Public ProgressionBackend surface unchanged:
  apply_action(CatheterAction) -> None

Mapping (SI, no conversion):
    CatheterAction.insert_velocity_mps → engine insert velocity (m/s)
"""

from __future__ import annotations

from typing import Any

from fluorovla.action import HOLD, CatheterAction, action_to_dict
from fluorovla.physics.engine import (
    FEED_MODE_FREE_SPACE_PORTAL_SLIDE,
    FEED_MODE_PATIENT_DISTAL_JOINT,
    NUM_FRAMES,
    active_feed_mode,
    clear_feed_mode,
    configure_feed_mode,
    run_episode,
)


class NewtonProgressionAdapter:
    """Thin boundary from common Action API to shared Newton progression.

    Protocol surface (ProgressionBackend):
        apply_action(CatheterAction) -> None

    Episode execution (backend-private):
        run_episode(...) invokes the shared engine with the current action
        velocity under the configured feed mode. Physics equations are not
        duplicated here.
    """

    def __init__(
        self,
        action: CatheterAction | None = None,
        *,
        feed_mode: str = FEED_MODE_FREE_SPACE_PORTAL_SLIDE,
        bend_stiffness: float | None = None,
        twist_stiffness: float | None = None,
    ) -> None:
        self._action = HOLD if action is None else action
        self._feed_mode = str(feed_mode)
        self._bend_stiffness = bend_stiffness
        self._twist_stiffness = twist_stiffness
        if self._feed_mode not in (
            FEED_MODE_FREE_SPACE_PORTAL_SLIDE,
            FEED_MODE_PATIENT_DISTAL_JOINT,
        ):
            raise ValueError(f"unknown feed_mode {self._feed_mode!r}")

    def apply_action(self, action: CatheterAction) -> None:
        if not isinstance(action, CatheterAction):
            raise TypeError(f"expected CatheterAction, got {type(action)!r}")
        self._action = action

    @property
    def action(self) -> CatheterAction:
        return self._action

    @property
    def insert_velocity_mps(self) -> float:
        """Insertion velocity consumed by the Newton progression path (SI m/s)."""
        return float(self._action.insert_velocity_mps)

    @property
    def feed_mode(self) -> str:
        return self._feed_mode

    def configure_backend_feed(self) -> None:
        """Apply this adapter's feed mode into the shared engine."""
        configure_feed_mode(
            self._feed_mode,
            bend_stiffness=self._bend_stiffness,
            twist_stiffness=self._twist_stiffness,
        )

    def run_episode(self, *, num_frames: int = NUM_FRAMES, action_name: str | None = None) -> dict[str, Any]:
        """Run one fresh free-space episode using the current CatheterAction velocity.

        Patient-contact episodes use a dedicated contact harness; this method
        always exercises the free-space portal-slide path (free-space regression).
        """
        name = action_name
        if name is None:
            v = self.insert_velocity_mps
            if abs(v) < 1e-15:
                name = "HOLD"
            elif abs(v - 0.006) < 1e-15:
                name = "SLOW"
            elif abs(v - 0.012) < 1e-15:
                name = "CRUISE"
            else:
                name = f"v={v}"
        # Free-space regression always uses portal-slide semantics.
        result = run_episode(
            velocity_mps=self.insert_velocity_mps,
            num_frames=num_frames,
            action_name=name,
        )
        result["action_input_velocity_mps"] = float(self._action.insert_velocity_mps)
        result["velocity_passthrough_match"] = (
            abs(float(result["internal_velocity_mps"]) - float(self._action.insert_velocity_mps)) < 1e-15
        )
        result["adapter_feed_mode"] = self._feed_mode
        result["engine_feed_mode_during_episode"] = result.get("feed_mode")
        return result

    def loggable_action(self) -> dict[str, float]:
        return action_to_dict(self._action)


__all__ = [
    "NewtonProgressionAdapter",
    "FEED_MODE_FREE_SPACE_PORTAL_SLIDE",
    "FEED_MODE_PATIENT_DISTAL_JOINT",
    "configure_feed_mode",
    "clear_feed_mode",
    "active_feed_mode",
]
