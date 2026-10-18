"""Reproduction / regression check of a candidate physics episode against a reference episode.

usage: check_reproduction.py <reference_episode.json> <candidate_episode.json> [--regression]
Default: native-Linux reproduction verdict (PASS_BIT_IDENTICAL / PASS_TOLERANCE / BLOCK, exit 1 on BLOCK).
--regression: strict row-level regression criterion (exit 1 unless it passes).
"""
import _bootstrap  # noqa: F401  isort:skip
import json
import sys

from fluorovla.evaluation.episode_comparison import compare_regression, load_episode, reproduction_verdict

ref, cand = load_episode(sys.argv[1]), load_episode(sys.argv[2])
if "--regression" in sys.argv:
    r = compare_regression(ref, cand)
    print(json.dumps({k: (v if k != "mismatches" else v[:8]) for k, v in r.items()}, indent=1, default=str))
    sys.exit(0 if r["PASS"] else 1)
v = reproduction_verdict(ref, cand)
print(json.dumps(v, indent=1))
sys.exit(0 if v["verdict"] != "BLOCK" else 1)
