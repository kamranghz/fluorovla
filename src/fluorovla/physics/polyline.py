"""Catheter polyline (solver-local metres) from Newton rigid-body poses."""
from __future__ import annotations

import numpy as np

from fluorovla.physics.engine import L_SEG, quat_rotate


def body_q_to_polyline_m(body_q: np.ndarray, bodies: list[int], n_active: int, *, fallback_axis: np.ndarray, L_seg: float = L_SEG) -> np.ndarray:
    """Nodes of the active rod: proximal end of the first body, then the distal end of every active body. ``fallback_axis`` is used only for a degenerate quaternion."""
    if n_active < 1:
        raise ValueError("n_active must be >= 1")
    pts: list[np.ndarray] = []
    half = 0.5 * float(L_seg)
    for i, b in enumerate(bodies[: int(n_active)]):
        row = np.asarray(body_q[int(b)], dtype=np.float64)
        com = row[:3]
        axis = quat_rotate(row[3:7], np.array([0.0, 0.0, 1.0], dtype=np.float64))
        n = float(np.linalg.norm(axis))
        axis = (axis / n) if n >= 1e-12 else np.asarray(fallback_axis, dtype=np.float64).copy()
        p0 = com - half * axis
        p1 = com + half * axis
        if i == 0:
            pts.append(p0)
        pts.append(p1)
    return np.stack(pts, axis=0).astype(np.float64)
