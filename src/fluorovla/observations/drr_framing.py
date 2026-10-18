"""Patient-specific C-arm framing for SlangDiffDRR (pose.translation / isocenter).

Shader geometry (diffdrr_slang.slang):
  isocenter = volumeCenter + pose.translation
  source/detector orbit that isocenter with Pose.rotation

Viewport historically passes translation = (0,0,0), so LAO-45 rotates about the
CT volume geometric center only — which can miss a catheter near the volume edge.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def volume_center_mm(app) -> np.ndarray:
    nx, ny, nz = app.sim["vol_shape_xyz"]
    sx, sy, sz = app.sim["spacing_xyz_mm"]
    return np.array([nx * sx * 0.5, ny * sy * 0.5, nz * sz * 0.5], dtype=np.float64)


def active_catheter_centroid_vol_mm(app, n_active: int) -> np.ndarray:
    """Centroid of currently active catheter nodes in volume-mm coordinates."""
    pos_m = app.sim["solver"].positions.cpu().numpy()[: int(n_active)]
    pos_vol_mm, _ = app._pos_to_vol_mm(pos_m)
    return np.asarray(pos_vol_mm, dtype=np.float64).mean(axis=0)


def carm_translation_for_isocenter(target_vol_mm: np.ndarray, vol_center_mm: np.ndarray) -> np.ndarray:
    """pose.translation (mm) that places C-arm isocenter at target_vol_mm."""
    return (np.asarray(target_vol_mm, dtype=np.float64) - np.asarray(vol_center_mm, dtype=np.float64)).astype(
        np.float32
    )


def apply_carm_isocenter(app, target_vol_mm: np.ndarray, *, source: str) -> dict[str, Any]:
    """Store framing on app.sim; does not mutate physics state."""
    center = volume_center_mm(app)
    target = np.asarray(target_vol_mm, dtype=np.float64).reshape(3)
    translation = carm_translation_for_isocenter(target, center)
    app.sim["carm_translation_mm"] = translation
    app.sim["carm_isocenter_vol_mm"] = target.astype(np.float32)
    app.sim["carm_isocenter_source"] = str(source)
    app.sim["carm_volume_center_mm"] = center.astype(np.float32)
    # Vessel overlays were precomputed at translation=0; disable to avoid wrong cyan mask.
    app.vessel_overlay_on = False
    return {
        "volume_center_mm": center.tolist(),
        "target_isocenter_vol_mm": target.tolist(),
        "carm_translation_mm": translation.tolist(),
        "source": source,
    }


def install_carm_translation_render_patch(SlangViewportApp) -> None:
    """Make SlangViewportApp._render_fluoro honor app.sim['carm_translation_mm'].

    Preserves PROJECTIONS rotations (e.g. LAO-45). Only swaps the translation
    argument that was previously hard-wired to _TRANS_ZERO.
    """
    if getattr(SlangViewportApp, "_fluorovla_carm_translation_patched", False):
        return

    orig = SlangViewportApp._render_fluoro
    viewport_globals = orig.__globals__
    if "_TRANS_ZERO" not in viewport_globals:
        raise RuntimeError(
            "cannot locate viewport _TRANS_ZERO for C-arm framing patch "
            f"(globals keys sample={list(viewport_globals)[:12]})"
        )

    def _render_fluoro_framed(self, projection: str):
        trans = self.sim.get("carm_translation_mm")
        if trans is None:
            return orig(self, projection)
        t = np.asarray(trans, dtype=np.float32).reshape(1, 3)
        old = viewport_globals.get("_TRANS_ZERO")
        viewport_globals["_TRANS_ZERO"] = t
        try:
            return orig(self, projection)
        finally:
            if old is not None:
                viewport_globals["_TRANS_ZERO"] = old

    SlangViewportApp._render_fluoro = _render_fluoro_framed
    SlangViewportApp._fluorovla_carm_translation_patched = True


def project_points_to_detector_px(
    pts_vol_mm: np.ndarray,
    *,
    rotation_xyz_rad: np.ndarray,
    translation_mm: np.ndarray,
    vol_center_mm: np.ndarray,
    det_size: int = 320,
    pixel_spacing_mm: float = 0.6,
    sid_mm: float = 510.0,
    sdd_mm: float = 1020.0,
) -> np.ndarray:
    """Match slang computePixelIntensityBatched projection (ZXY Euler)."""
    import math

    rx, ry, rz = [float(x) for x in np.asarray(rotation_xyz_rad, dtype=np.float64).reshape(3)]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    R = np.array(
        [
            [cz * cy - sz * sx * sy, -sz * cx, cz * sy + sz * sx * cy],
            [sz * cy + cz * sx * sy, cz * cx, sz * sy - cz * sx * cy],
            [-cx * sy, sx, cx * cy],
        ],
        dtype=np.float64,
    )
    iso = np.asarray(vol_center_mm, dtype=np.float64) + np.asarray(translation_mm, dtype=np.float64)
    out = []
    for P in np.asarray(pts_vol_mm, dtype=np.float64).reshape(-1, 3):
        cam = R.T @ (P - iso)
        src = np.array([0.0, 0.0, -sid_mm], dtype=np.float64)
        d = cam - src
        t = ((sdd_mm - sid_mm) - src[2]) / d[2]
        hit = src + t * d
        u = hit[0] / pixel_spacing_mm + det_size * 0.5 - 0.5
        v = hit[1] / pixel_spacing_mm + det_size * 0.5 - 0.5
        out.append([u, v])
    return np.asarray(out, dtype=np.float64)
