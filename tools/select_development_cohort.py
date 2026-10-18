"""Deterministic development-cohort selection from a descriptor table (descriptor_rows.json of verify_patient_geometry.py).

usage: select_development_cohort.py <descriptor_rows.json> <out_json> [--config configs/anatomy/cohort_selection.json] [--lock cohort_lock.json]
With --lock the result must equal the locked cohort exactly (exit 1 otherwise).
"""
import _bootstrap  # noqa: F401  isort:skip
import argparse
import json
import sys
from pathlib import Path

from fluorovla.anatomy.cohort_selection import select_development_cohort

ap = argparse.ArgumentParser()
ap.add_argument("rows")
ap.add_argument("out")
ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/anatomy/cohort_selection.json"))
ap.add_argument("--lock", default=None)
a = ap.parse_args()
cfg = json.load(open(a.config))
res = select_development_cohort(json.load(open(a.rows)), cfg["cohort_size"], cfg["reference_patient"])
json.dump(res, open(a.out, "w"), indent=1, default=float)
print(json.dumps(res["cohort"]))
if a.lock:
    locked = json.load(open(a.lock))["development_cohort"]
    ok = locked == res["cohort"]
    print("matches lock:", ok)
    sys.exit(0 if ok else 1)
