"""Summarize attempt-addressable physics-mapping runs.
usage: summarize_physics_mapping.py <runs_dir> <episode_list.json | workload.json> <out_json>"""
import _bootstrap  # noqa: F401  isort:skip
import json
import sys

from fluorovla.evaluation.physics_mapping_summary import summarize

names = json.load(open(sys.argv[2]))
names = names["episodes"] if isinstance(names, dict) else names
res = summarize(sys.argv[1], names)
json.dump(res, open(sys.argv[3], "w"), indent=1)
print(json.dumps(res["summary"], indent=1))
