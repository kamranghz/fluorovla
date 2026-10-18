"""Build patient twins (mu volume, vessel mask, centerline graph) for the given patients.

usage: build_patient_twin.py <patient_id> [...] [--raw DIR] [--cache DIR] [--vdt PATH]
defaults: raw = <data>/raw/totalsegmentator, cache = <data>/cache/twin; --vdt (or env VDT_PATH) = directory containing ``vasculature_digital_twin``.
"""
import _bootstrap  # noqa: F401  isort:skip
import argparse
import json

from fluorovla.anatomy.twin_build import build_patient_twin
from fluorovla.common.paths import data_root

ap = argparse.ArgumentParser()
ap.add_argument("patients", nargs="+")
ap.add_argument("--raw", default=str(data_root() / "raw/totalsegmentator"))
ap.add_argument("--cache", default=str(data_root() / "cache/twin"))
ap.add_argument("--vdt", default=None)
a = ap.parse_args()
for pid in a.patients:
    try:
        print(json.dumps(build_patient_twin(a.raw, a.cache, pid, a.vdt)), flush=True)
    except Exception as ex:  # noqa: BLE001  (a failed build is reported, not hidden)
        print(json.dumps({"pid": pid, "status": "ERROR", "error": repr(ex)}), flush=True)
