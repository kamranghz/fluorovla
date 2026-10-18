"""Extract and verify patient access-path geometry from the twins; define the anatomical candidate pool and the descriptor table.

usage: verify_patient_geometry.py <out_dir> [patient_id ...] [--cache DIR] [--raw DIR] [--metadata study_metadata.json]
Writes into <out_dir>: geometry.json (all entries per patient), geometry.npz (extended paths), sides.json (left/right access paths),
candidate_pool.json, descriptor_rows.json, MANIFEST.json. Without patient ids every twin in the cache is processed.
"""
import _bootstrap  # noqa: F401  isort:skip
import argparse
import json
from pathlib import Path

import numpy as np

from fluorovla.anatomy.access_geometry import extract_patient_geometry, split_arrays
from fluorovla.anatomy.candidate_pool import assign_sides, candidate_pool, descriptor_rows
from fluorovla.common.paths import data_root
from fluorovla.common.provenance import write_run_manifest

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("patients", nargs="*")
ap.add_argument("--cache", default=str(data_root() / "cache/twin"))
ap.add_argument("--raw", default=str(data_root() / "raw/totalsegmentator"))
ap.add_argument("--metadata", default=None, help="JSON list of study records (id, study_type, pathology, gender, age, scanner_model)")
a = ap.parse_args()
out = Path(a.out_dir)
pids = a.patients or sorted(p.name for p in Path(a.cache).iterdir() if (p / "centerline_points_mm.npy").is_file())
geoms, errors = [], {}
for pid in pids:
    try:
        geoms.append(extract_patient_geometry(Path(a.cache) / pid, pid))
    except Exception as ex:  # noqa: BLE001  (an extraction failure is reported, never silently dropped)
        errors[pid] = repr(ex)
geoms, arrays = split_arrays(geoms)
sides = {g["pid"]: assign_sides(g, Path(a.raw) / g["pid"]) for g in geoms}
meta = {r["id"]: r for r in json.load(open(a.metadata))} if a.metadata else {}
rows = descriptor_rows(sides, meta)
out.mkdir(parents=True, exist_ok=True)
write_run_manifest(out, "patient geometry verification", pids, {"cache": a.cache, "raw": a.raw}, "COMPLETED", {"errors": errors})
json.dump(geoms, open(out / "geometry.json", "w"), indent=1)
np.savez_compressed(out / "geometry.npz", **arrays)
json.dump(sides, open(out / "sides.json", "w"), indent=1, default=float)
json.dump(candidate_pool(sides), open(out / "candidate_pool.json", "w"), indent=1)
json.dump(rows, open(out / "descriptor_rows.json", "w"), indent=0, default=float)
print(f"patients {len(pids)}  extraction errors {len(errors)}  candidate pool {len(candidate_pool(sides))}  candidate paths {len(rows)}")
