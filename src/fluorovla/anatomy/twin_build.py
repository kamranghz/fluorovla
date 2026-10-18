"""Build the patient twin (mu volume, vessel mask, centerline graph) from a TotalSegmentator CT with the unmodified i4h ``vasculature_digital_twin`` package.

The package is imported from ``VDT_PATH`` (directory that contains ``vasculature_digital_twin``, e.g. a copy of the i4h site-packages entry). Runs on CPU and does not
need Docker or WSL (outputs were verified bit-identical to the i4h container build).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REQUIRED = ("centerline_points_mm.npy", "centerline_edges.npy", "centerline_radii_mm.npy", "vessel_mask.npy", "metadata.json", "mu_volume.npy", "hu_volume.npy")


def build_patient_twin(raw_dir: str | Path, cache_dir: str | Path, pid: str, vdt_path: str | Path | None = None) -> dict:
    vdt = vdt_path or os.environ.get("VDT_PATH")
    if vdt and str(vdt) not in sys.path:
        sys.path.insert(0, str(vdt))
    from vasculature_digital_twin.cli import preprocess_ct, segment_vessels   # noqa: PLC0415

    raw_dir, out = Path(raw_dir) / pid, Path(cache_dir) / pid
    if all((out / f).is_file() for f in REQUIRED):
        return {"pid": pid, "status": "EXISTS"}
    t0 = time.time()
    r1 = preprocess_ct.main(["--nifti", str(raw_dir / "ct.nii.gz"), "--output-dir", str(out), "--save-hu"])
    r2 = segment_vessels.main(["--ct-dir", str(out), "--ts-gt-dir", str(raw_dir / "segmentations"), "--device", "cpu"])
    return {"pid": pid, "status": "BUILT" if (r1 == 0 and r2 == 0) else "FAILED", "preprocess_rc": r1, "segment_rc": r2, "seconds": time.time() - t0}
