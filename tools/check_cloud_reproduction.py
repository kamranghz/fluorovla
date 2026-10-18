"""Gate of the native-Linux physics-mapping workflow: reproduce the accepted reference, one completed SUCCESS and one completed LUMEN_VIOLATION episode.

usage: check_cloud_reproduction.py <repro_runs_dir> <out_verdict.json>
Expects <repro_runs_dir>/<run_name>/attempt_NNN (run names in configs/evaluation/reproduction_cases.json)/run/episode_physics_ep.json (the LAST attempt with a result is used).
Exit 0 = PASS (every case PASS_BIT_IDENTICAL or PASS_TOLERANCE), 1 = BLOCK (any status or action-sequence difference, or tolerance exceeded, or a missing case).
"""
import _bootstrap  # noqa: F401  isort:skip
import json
import sys
from pathlib import Path

from fluorovla.evaluation.episode_comparison import load_episode, reproduction_verdict
from fluorovla.evaluation.physics_mapping_summary import accepted_attempt

REPO = Path(__file__).resolve().parents[1]
CASES = {c["label"]: (c["run_name"], REPO / c["reference_episode"]) for c in json.load(open(REPO / "configs/evaluation/reproduction_cases.json"))["cases"]}
runs = Path(sys.argv[1])
out, ok = {}, True
for label, (name, ref) in CASES.items():
    acc, states = accepted_attempt(runs / name)
    if acc is None:
        out[label] = {"verdict": "BLOCK", "reason": "no completed attempt", "attempt_states": states}
        ok = False
        continue
    v = reproduction_verdict(load_episode(ref), load_episode(acc / "run/episode_physics_ep.json"))
    v["attempt"] = acc.name
    out[label] = v
    ok &= v["verdict"] != "BLOCK"
out["DECISION"] = "PASS" if ok else "BLOCK"
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: (v["verdict"] if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))
sys.exit(0 if ok else 1)
