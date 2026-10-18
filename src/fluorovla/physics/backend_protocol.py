"""Minimal backend contract for progression commands."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fluorovla.action import CatheterAction


@runtime_checkable
class ProgressionBackend(Protocol):
    """Backend-independent consumer of CatheterAction.

    Callers issue semantic progression commands only:
        backend.apply_action(CatheterAction(insert_velocity_mps=0.006))
    The backend decides how insertion is realized physically.
    """

    def apply_action(self, action: CatheterAction) -> None:
        """Accept the current progression command for subsequent simulation steps."""
        ...
