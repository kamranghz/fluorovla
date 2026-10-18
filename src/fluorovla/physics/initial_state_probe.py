"""Diagnostic of the initial rod placement for one episode configuration (does not influence the episode)."""
from __future__ import annotations

import numpy as np

import fluorovla.task_contract as tc
from fluorovla.physics.lumen_sdf import LumenSDF
from fluorovla.physics.patient_config import PatientConfig


def probe_initial_state(cfg: PatientConfig, sdf: LumenSDF) -> dict:
    """SDF (mm) along the access path from the portal and along the 73.5-mm straight initial rod."""
    z = np.load(cfg.path_npz_path)
    proj = tc.PathProjector(z["pts"], z["rad"])
    off = cfg.ct_offset * 1000.0
    s_p = cfg.s_portal_mm if cfg.s_portal_mm is not None else 0.0
    ss = np.arange(s_p, s_p + 120.0, 2.0)
    path_loc = (np.array([proj.point_at(x) for x in ss]) - off) / 1000.0
    t = cfg.tangent
    rod = np.array([k * 1e-3 * t for k in np.arange(0.0, 73.51, 1.5)])
    out = {}
    for nm, pts in (("path", path_loc), ("straight_rod", rod)):
        ph, _, _ = sdf.query(pts.astype(np.float32))
        ph = np.asarray(ph) * 1000.0
        out[nm] = {"n": len(ph), "phi_mm_min": float(ph.min()), "phi_mm_max": float(ph.max()), "frac_outside": float(np.mean(ph > 0)),
                   "frac_phi_gt_-3mm(catheter radius not clear)": float(np.mean(ph > -3.0))}
    out["mesh_vertices"] = sdf.meta.get("n_vertices")
    return out
