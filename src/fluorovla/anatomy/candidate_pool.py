"""Side assignment, descriptor rows and the anatomical candidate pool from twin access-path geometries."""
from __future__ import annotations

from pathlib import Path

import numpy as np

GATE_MM = 150.0            # minimum access-path length (S_contact) for a candidate path
LABEL_DIST_MM = 6.0        # an entry endpoint belongs to the iliac label whose mask is nearer than this


def label_distance(raw_patient_dir: str | Path, point_mm: np.ndarray) -> dict[str, float]:
    """Distance (mm) from a world point to the nearest voxel of the left / right iliac-artery label (TotalSegmentator)."""
    import nibabel as nib
    from scipy.spatial import cKDTree

    out = {}
    for side in ("left", "right"):
        f = Path(raw_patient_dir) / "segmentations" / f"iliac_artery_{side}.nii.gz"
        if not f.exists():
            out[side] = 1e9
            continue
        img = nib.load(str(f))
        idx = np.argwhere(np.asarray(img.dataobj) > 0)
        if len(idx) == 0:
            out[side] = 1e9
            continue
        w = idx @ img.affine[:3, :3].T + img.affine[:3, 3]
        out[side] = float(cKDTree(w).query(point_mm)[0])
    return out


def assign_sides(geometry: dict, raw_patient_dir: str | Path) -> dict:
    """{'left': entry record, 'right': entry record}: for each side the lowest-z endpoint whose nearest iliac label is that side and lies within LABEL_DIST_MM."""
    by_side: dict = {}
    for e in geometry["entries"]:
        ld = label_distance(raw_patient_dir, np.array(e["entry_ct_mm"]))
        side = min(ld, key=ld.get)
        if ld[side] > LABEL_DIST_MM:
            continue
        e = dict(e, label_dist_mm=ld[side])
        if side not in by_side or e["entry_ct_mm"][2] < by_side[side]["entry_ct_mm"][2]:
            by_side[side] = e
    return by_side


def is_candidate_path(entry: dict | None) -> bool:
    return bool(entry and entry.get("valid") and entry["S_contact_mm"] >= GATE_MM)


def descriptor_rows(sides_by_patient: dict[str, dict], metadata: dict[str, dict]) -> list[dict]:
    """One row per candidate path with the seven descriptors and the study metadata (input of the cohort selection)."""
    rows = []
    for pid, sides in sides_by_patient.items():
        for side, s in sides.items():
            if not is_candidate_path(s):
                continue
            m = metadata.get(pid, {})
            rows.append({"pid": pid, "side": side, **{k: s[k] for k in ("S_contact_mm", "window_mm", "r_med_mm", "r_p05_mm", "c_z_mm", "tort_w", "e30_p95_mm", "e30_max_mm", "flare", "ang_deg", "jag")},
                         "cluster_xyz": s["centroid_ct_mm"], "cta": "angiography" in (m.get("study_type") or ""), "study_type": m.get("study_type"), "pathology": m.get("pathology"),
                         "sex": m.get("gender") or "unk", "age": m.get("age"), "scanner": m.get("scanner_model")})
    return rows


def candidate_pool(sides_by_patient: dict[str, dict]) -> list[str]:
    return sorted(p for p, sides in sides_by_patient.items() if any(is_candidate_path(s) for s in sides.values()))
