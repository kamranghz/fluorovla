"""Package the results of a cloud (or any remote) physics-mapping run for return to the local canonical project.

usage: package_cloud_return.py <bundle_root> <runs_dir> <out_dir> [--commands commands.txt]
  <bundle_root>  the uploaded bundle (contains MANIFEST_SHA256.txt and project/)
  <runs_dir>     attempt-addressable run directory (physics-mapping runs, reproduction runs, logs)
Writes into <out_dir> (a copy of the runs plus): ENVIRONMENT_CLOUD.json, CHANGED_FILES.json (ADDED / MODIFIED / REMOVED / UNCHANGED relative to the uploaded
manifest, with before/after SHA-256), STAGE3_FINAL_SUMMARY.json, RETURN_MANIFEST.json and RETURN_SHA256.txt.
"""
import _bootstrap  # noqa: F401  isort:skip
import json
import shutil
import sys
from pathlib import Path

from fluorovla.common.provenance import capture_environment, sha256_file, write_file_manifest
from fluorovla.evaluation.physics_mapping_summary import summarize

bundle, runs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
commands = Path(sys.argv[sys.argv.index("--commands") + 1]) if "--commands" in sys.argv else None
if out.exists():
    sys.exit(f"{out} exists; evidence is never overwritten")
shutil.copytree(runs, out / "runs")
json.dump(capture_environment(), open(out / "ENVIRONMENT_CLOUD.json", "w"), indent=1)
# code / config changes relative to the uploaded manifest (project/ tree only; data and runs are not code)
want = {}
for line in open(bundle / "MANIFEST_SHA256.txt", encoding="utf-8"):
    if line.strip():
        h, r = line.rstrip("\n").split("  ", 1)
        want[r] = h
proj = bundle / "project"
have = {p.relative_to(bundle).as_posix(): p for p in proj.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and not any(x.endswith(".egg-info") for x in p.parts)}
changed = {"ADDED": [], "MODIFIED": [], "REMOVED": [], "UNCHANGED": 0, "REGENERATED_CACHE_FILES": []}
for r, p in have.items():
    if r not in want and r.startswith("project/data/cache/"):            # regenerable caches (e.g. the lumen collision mesh) are not code or config
        changed["REGENERATED_CACHE_FILES"].append({"path": r, "sha256": sha256_file(p)})
    elif r not in want:
        changed["ADDED"].append({"path": r, "sha256_after": sha256_file(p)})
    elif sha256_file(p) != want[r]:
        changed["MODIFIED"].append({"path": r, "sha256_before": want[r], "sha256_after": sha256_file(p)})
    else:
        changed["UNCHANGED"] += 1
changed["REMOVED"] = [r for r in want if r.startswith("project/") and r not in have and not r.startswith("project/runs/")]
json.dump(changed, open(out / "CHANGED_FILES.json", "w"), indent=1)
for item in changed["ADDED"] + changed["MODIFIED"]:                      # every changed file travels back
    dst = out / "changed_files" / item["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle / item["path"], dst)
if commands and commands.is_file():
    shutil.copy2(commands, out / "COMMANDS_EXECUTED.txt")
wl_path = bundle / "workload/workload.json"
if wl_path.is_file():
    wl = json.load(open(wl_path))
    res = summarize(runs / "physics_mapping", wl["episodes"], write_accepted=False) if (runs / "physics_mapping").is_dir() else {"summary": "no physics_mapping runs"}
    json.dump(res, open(out / "STAGE3_FINAL_SUMMARY.json", "w"), indent=1)
json.dump({"bundle_commit": json.load(open(bundle / "BUNDLE_INFO.json")).get("git_commit") if (bundle / "BUNDLE_INFO.json").is_file() else None,
           "code_changes": {k: (len(v) if isinstance(v, list) else v) for k, v in changed.items()}, "note": "verify RETURN_SHA256.txt before accepting these artifacts"},
          open(out / "RETURN_MANIFEST.json", "w"), indent=1)
n = write_file_manifest(out, out / "RETURN_SHA256.txt")
print(f"return package: {n} files hashed; code changes: ADDED {len(changed['ADDED'])}, MODIFIED {len(changed['MODIFIED'])}, REMOVED {len(changed['REMOVED'])}, UNCHANGED {changed['UNCHANGED']}")
