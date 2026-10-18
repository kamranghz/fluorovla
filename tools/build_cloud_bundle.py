"""Build a self-contained native-Linux physics-mapping bundle FROM THE CLEAN GIT REPOSITORY (no working-tree files are copied).

usage: build_cloud_bundle.py <out_dir> --newton-src DIR --data-root DIR [--local-results DIR] [--environment-json FILE]
  - requires a clean git working tree; the bundle contains `git archive HEAD` (the exact committed code, configs, tests, docs, artifacts)
  - twins (vessel mask, metadata, centerline, reference mesh) of the locked development cohort + sentinel, from <data-root>/cache/twin
  - the physics-mapping workload: episode configurations for every candidate access side of those patients, the list of episodes that have a local (preliminary)
    result, the pending list, and the local results as comparison references
  - the exact Newton source (without .git) and its commit, BUNDLE_INFO.json, ENVIRONMENT_LOCAL.json, RUNBOOK_NATIVE_LINUX.md and MANIFEST_SHA256.txt
"""
import _bootstrap  # noqa: F401  isort:skip
import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

from fluorovla.anatomy.candidate_pool import is_candidate_path
from fluorovla.anatomy.episode_config import build_episode_config, start_depths
from fluorovla.common.provenance import write_file_manifest

REPO = Path(__file__).resolve().parents[1]
TWIN_FILES = ("vessel_mask.npy", "metadata.json", "centerline_points_mm.npy", "centerline_edges.npy", "centerline_radii_mm.npy")
def reference_mesh_files() -> dict:
    """Mesh caches named by the reference configurations (e.g. the solver-local mesh of the accepted reference) -> {patient_dir: [file names]}."""
    out = {}
    for cfg in (REPO / "artifacts/references").rglob("config.json"):
        c = json.load(open(cfg))
        parts = Path(c["mesh_cache"]).parts                      # cache/twin/<patient>/<file>
        if len(parts) == 4 and (Path(a.data_root) / c["mesh_cache"]).is_file():
            out.setdefault(parts[2], []).append(parts[3])
    return out



ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--newton-src", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--local-results", default=None, help="directory of local per-episode result folders <name>/run/episode_physics_ep.json (comparison only)")
ap.add_argument("--environment-json", default=None)
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"{out} exists; bundles are never overwritten")
dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
if dirty:
    sys.exit("working tree is not clean; commit first:\n" + dirty)
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
(out / "project").mkdir(parents=True)
tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", "HEAD"], capture_output=True).stdout
tarfile.open(fileobj=io.BytesIO(tar)).extractall(out / "project")

MESH = reference_mesh_files()
lock = json.load(open(REPO / "artifacts/provenance/cohort_lock.json"))
patients = lock["development_cohort"] + [lock["stress_sentinel"]]
data = Path(a.data_root)
for pid in patients:
    dst = out / "project/data/cache/twin" / pid
    dst.mkdir(parents=True, exist_ok=True)
    for f in TWIN_FILES + tuple(MESH.get(pid, ())):
        shutil.copy2(data / "cache/twin" / pid / f, dst / f)

sides = json.load(open(REPO / "artifacts/references/candidate_pool_geometry/sides.json"))
arr = np.load(REPO / "artifacts/references/candidate_pool_geometry/geometry.npz")
ecfg = out / "workload/episode_configs"
names = []
for pid in patients:
    for side, s in sorted(sides[pid].items()):
        if is_candidate_path(s):
            for k in range(len(start_depths(s["S_contact_mm"]))):
                name = f"{pid}_{side}_F{k}"
                build_episode_config(pid, side, k, s["S_contact_mm"], arr[f"{pid}__{s['entry_node']}__ext_pts"], arr[f"{pid}__{s['entry_node']}__ext_rad"], ecfg / name)
                names.append(name)
completed = []
if a.local_results:
    ref = out / "workload/reference_local_results"
    ref.mkdir(parents=True)
    for n in names:
        f = Path(a.local_results) / n / "run/episode_physics_ep.json"
        if f.is_file():
            shutil.copy2(f, ref / f"{n}.json")
            completed.append(n)
json.dump({"purpose": "physics mapping (Privileged Teacher, legacy rule) on the locked development cohort + stress sentinel", "episodes_configs_note": "episode_configs/<name>/{config,landing_zone,path_ext}",
           "episode_configs_dir": "episode_configs", "episodes": names, "completed_local": completed, "pending": [n for n in names if n not in completed],
           "local_results_note": "completed_local = preliminary results from the unstable local WSL2 stack, kept for comparison; they are re-run on the stable stack for the final mapping"},
          open(out / "workload/workload.json", "w"), indent=1)

nsrc = Path(a.newton_src)
shutil.copytree(nsrc, out / "env/newton-src", ignore=shutil.ignore_patterns(".git", "__pycache__"))
(out / "env/NEWTON_COMMIT.txt").write_text(subprocess.run(["git", "-C", str(nsrc), "rev-parse", "HEAD"], capture_output=True, text=True).stdout)
shutil.copy2(REPO / "constraints.txt", out / "env/constraints.txt")
if a.environment_json:
    shutil.copy2(a.environment_json, out / "ENVIRONMENT_LOCAL.json")
shutil.copy2(REPO / "docs/cloud_workflow.md", out / "RUNBOOK_NATIVE_LINUX.md")
json.dump({"git_commit": commit, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "patients": patients, "n_episodes": len(names), "n_completed_local": len(completed),
           "n_pending": len(names) - len(completed), "source": "git archive HEAD of a clean working tree"}, open(out / "BUNDLE_INFO.json", "w"), indent=1)
n = write_file_manifest(out, out / "MANIFEST_SHA256.txt")
print(f"bundle {out}: {n} files, commit {commit}, {len(names)} episodes ({len(completed)} with local result, {len(names) - len(completed)} pending)")
