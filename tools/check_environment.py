"""Capture the runtime environment and compare it with the reference stack (docs/reproducibility.md).

usage: check_environment.py [out.json]     exit 0 when every pinned component matches, 1 otherwise (the capture is written either way)
"""
import _bootstrap  # noqa: F401  isort:skip
import json
import sys

from fluorovla.common.provenance import capture_environment

REFERENCE = {"python": "3.12", "warp": "1.17.0", "numpy": "2.5.0", "scipy": "1.17.1", "skimage": "0.26.0"}
env = capture_environment()
diffs = {k: {"reference": v, "found": env.get(k)} for k, v in REFERENCE.items() if not str(env.get(k, "")).startswith(v)}
try:
    import newton  # noqa: F401
    import subprocess
    import pathlib
    root = pathlib.Path(newton.__file__).resolve().parents[1]
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env["newton_commit"] = head or "unknown (no git metadata; compare env/NEWTON_COMMIT.txt)"
    if head and head != "6d2ece7c314248960a249ebc46b0dafbb5a89232":
        diffs["newton_commit"] = {"reference": "6d2ece7c314248960a249ebc46b0dafbb5a89232", "found": head}
except Exception as ex:  # noqa: BLE001
    diffs["newton"] = {"reference": "importable", "found": repr(ex)}
env["differences_from_reference"] = diffs
if len(sys.argv) > 1:
    json.dump(env, open(sys.argv[1], "w"), indent=1)
print(json.dumps(env, indent=1))
sys.exit(0 if not diffs else 1)
