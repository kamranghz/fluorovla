"""Patient-specific episode configuration (the only place where patient geometry enters the physics episode).

A configuration is a JSON file in the episode directory, next to ``landing_zone.json`` and ``path_ext.npz``:

  patient_id, access_side
  ct_dir          twin cache directory, relative to the data root (contains vessel_mask.npy, metadata.json, centerline_*.npy)
  mesh_cache      lumen-collision mesh cache, relative to the data root
  mesh_frame      "ct" (cache holds ``vertices_ct_m``; the CT offset is subtracted at load) or "solver_local" (cache holds ``vertices_m``)
  portal_m        portal position in solver-local metres (always the origin: the solver frame is the CT frame shifted to the portal)
  tangent_raw     insertion direction (normalized when used), or ``tangent`` (normalized once more, see below)
  ct_offset_m     CT position of the portal in metres, or ``portal_ct_mm`` (divided by 1000)
  sign_scale      sign convention of the lumen SDF (phi < 0 inside)

Numerical note (kept for bit-exact reproduction of existing results): a configuration written with the key ``tangent`` is normalized when read and
normalized again when used (unit/|unit| can differ from unit in the last bit); a configuration with ``tangent_raw`` is normalized once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fluorovla.common.paths import resolve_data


@dataclass(frozen=True)
class PatientConfig:
    patient_id: str
    access_side: str
    ct_dir: Path
    mesh_cache: Path
    mesh_frame: str
    portal_m: tuple
    tangent_raw: tuple
    ct_offset_m: tuple
    sign_scale: float
    landing_zone_path: Path
    path_npz_path: Path
    config_path: Path | None = None
    start_index: int | None = None
    F_mm: float | None = None
    s_portal_mm: float | None = None

    @property
    def portal(self) -> np.ndarray:
        return np.array(self.portal_m, dtype=np.float64)

    @property
    def tangent(self) -> np.ndarray:
        t = np.array(self.tangent_raw, dtype=np.float64)
        return t / np.linalg.norm(t)

    @property
    def ct_offset(self) -> np.ndarray:
        return np.array(self.ct_offset_m, dtype=np.float64)

    @classmethod
    def from_json(cls, path: str | Path) -> "PatientConfig":
        path = Path(path)
        c = json.load(open(path))
        if "tangent_raw" in c:
            raw = list(c["tangent_raw"])
        else:
            t = np.asarray(c["tangent"], dtype=np.float64)
            raw = (t / np.linalg.norm(t)).tolist()
        off = c["ct_offset_m"] if "ct_offset_m" in c else (np.asarray(c["portal_ct_mm"], dtype=np.float64) / 1000.0).tolist()
        base = path.parent
        return cls(patient_id=c["patient_id"], access_side=c.get("access_side", ""), ct_dir=resolve_data(c["ct_dir"]), mesh_cache=resolve_data(c["mesh_cache"]),
                   mesh_frame=c["mesh_frame"], portal_m=tuple(c.get("portal_m", [0.0, 0.0, 0.0])), tangent_raw=tuple(raw), ct_offset_m=tuple(off),
                   sign_scale=float(c.get("sign_scale", 1.0)), landing_zone_path=base / c.get("landing_zone", "landing_zone.json"),
                   path_npz_path=base / c.get("path_npz", "path_ext.npz"), config_path=path, start_index=c.get("start_index"), F_mm=c.get("F_mm"),
                   s_portal_mm=c.get("s_portal_mm"))
