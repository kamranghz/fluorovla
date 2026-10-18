"""Per start-condition episode configuration for the canonical patient physics.

For (patient, access side, start depth F):
  portal   = point on the RAW centerline polyline whose projection onto the smoothed twin path has arclength s_p = z_tip - 73.5 - F, with
             z_tip = S_contact - 4.72 - 3.0 (polyline tip at the near edge of the landing window); clipped to the path start
  tangent  = 5-mm chord of the smoothed path at the portal (the rod is laid straight along it)
  landing zone (tip extremity) = [S - 4.72, S - 2.32]
  T_MAX    = ceil(1.25 * ((F_c - 6) / 12 + 6 / 6) / 0.1) + 3 decisions with F_c = F + EPS (start extremity to window centre)
Start depths: F at 0 %, 50 % and 100 % of [60 mm, F_hi] with F_hi = min(100 mm, S - 81.2 mm - 1 mm).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import fluorovla.task_contract as tc

R_C, ZONE_NEAR, ZONE_FAR, ROD_BEHIND = 3.0, 4.72, 2.32, 73.5
F_LO, F_CAP = 60.0, 100.0


def f_range(S: float) -> tuple[float, float]:
    return F_LO, min(F_CAP, S - 81.2 - 1.0)


def start_depths(S: float) -> list[float]:
    lo, hi = f_range(S)
    if hi < lo:
        return []
    return [lo, 0.5 * (lo + hi), hi]


def raw_point_for_smoothed_s(proj: tc.PathProjector, pts: np.ndarray, s_p: float):
    """Portal on the RAW centerline polyline at the position whose projection onto the smoothed path has arclength s_p (clipped to the path start)."""
    s_nodes = np.array([proj(x)[0] for x in pts])
    s_c = max(float(s_p), 0.0)
    k = int(np.searchsorted(s_nodes, s_c, side="right") - 1)
    k = int(np.clip(k, 0, len(pts) - 2))
    if s_nodes[k + 1] <= s_nodes[k] + 1e-9:
        return pts[k].copy(), s_c
    a = float(np.clip((s_c - s_nodes[k]) / (s_nodes[k + 1] - s_nodes[k]), 0.0, 1.0))
    return pts[k] + a * (pts[k + 1] - pts[k]), s_c


def build_episode_config(pid: str, side: str, k: int, S_contact_mm: float, ext_pts: np.ndarray, ext_rad: np.ndarray, out_dir: str | Path,
                         F_override: float | None = None) -> Path:
    """Write config.json, landing_zone.json and path_ext.npz into ``out_dir``; returns the directory."""
    depths = start_depths(S_contact_mm)
    F = depths[k] if F_override is None else float(F_override)
    proj = tc.PathProjector(ext_pts, ext_rad)
    z_tip = S_contact_mm - ZONE_NEAR - R_C
    s_p = z_tip - ROD_BEHIND - F
    portal, s_pc = raw_point_for_smoothed_s(proj, ext_pts, s_p)
    t = proj.point_at(s_pc + 5.0) - proj.point_at(s_pc)
    t = t / np.linalg.norm(t)
    zone = [S_contact_mm - ZONE_NEAR, S_contact_mm - ZONE_FAR]
    F_c = F + tc.EPS_MM
    t_max = int(math.ceil(1.25 * ((F_c - 6.0) / (tc.V_MAX_MPS * 1000.0) + 6.0 / (tc.V_SLOW_MPS * 1000.0)) / tc.DECISION_DT_S)) + 3
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"patient_id": pid, "access_side": side, "start_index": k, "F_mm": F, "S_contact_mm": S_contact_mm, "s_portal_mm": s_p,
           "ct_dir": f"cache/twin/{pid}", "mesh_cache": f"cache/twin/{pid}/lumen_collision_mesh_ct_frame.npz", "mesh_frame": "ct",
           "portal_ct_mm": portal.tolist(), "tangent": t.tolist(), "sign_scale": 1.0,
           "landing_zone": "landing_zone.json", "path_npz": "path_ext.npz"}
    json.dump(cfg, open(d / "config.json", "w"), indent=1)
    json.dump({"landing_zone_extremity_mm": zone, "timeout": {"T_MAX_decisions": t_max, "formula": "ceil(1.25*((F_c-6)/12+6/6)/0.1)+3", "F_c_mm": F_c},
               "S_contact_mm": S_contact_mm, "F_mm": F}, open(d / "landing_zone.json", "w"), indent=1)
    np.savez(d / "path_ext.npz", pts=ext_pts, rad=ext_rad)
    return d
