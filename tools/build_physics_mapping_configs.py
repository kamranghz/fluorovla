"""Build the episode configurations (3 start conditions per access side) for a list of patients.

usage: build_physics_mapping_configs.py <geometry_dir> <out_dir> <patient_id> [...]
<geometry_dir> = output of verify_patient_geometry.py (sides.json, geometry.npz). Writes <out_dir>/<patient>_<side>_F<k>/{config.json, landing_zone.json, path_ext.npz}
and <out_dir>/episode_list.json. Only candidate paths (valid, S_contact >= 150 mm) are configured.
"""
import _bootstrap  # noqa: F401  isort:skip
import json
import sys
from pathlib import Path

import numpy as np

from fluorovla.anatomy.candidate_pool import is_candidate_path
from fluorovla.anatomy.episode_config import build_episode_config, start_depths

geo, out = Path(sys.argv[1]), Path(sys.argv[2])
patients = sys.argv[3:]
sides = json.load(open(geo / "sides.json"))
arr = np.load(geo / "geometry.npz")
names = []
for pid in patients:
    for side, s in sorted(sides[pid].items()):
        if not is_candidate_path(s):
            continue
        for k in range(len(start_depths(s["S_contact_mm"]))):
            name = f"{pid}_{side}_F{k}"
            build_episode_config(pid, side, k, s["S_contact_mm"], arr[f"{pid}__{s['entry_node']}__ext_pts"], arr[f"{pid}__{s['entry_node']}__ext_rad"], out / name)
            names.append(name)
json.dump(names, open(out / "episode_list.json", "w"), indent=1)
print(len(names), "episode configurations")
