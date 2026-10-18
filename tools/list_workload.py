"""List episode-configuration directories of a physics-mapping workload.
usage: list_workload.py <workload.json> {all|pending|completed_local}      (prints one directory per line)"""
import json
import sys
from pathlib import Path

wl = json.load(open(sys.argv[1]))
base = Path(sys.argv[1]).resolve().parent / wl["episode_configs_dir"]
names = wl["episodes"] if sys.argv[2] == "all" else wl[sys.argv[2]]
for n in names:
    print(base / n)
