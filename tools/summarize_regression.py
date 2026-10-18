"""Write a regression summary for a set of (reference episode, candidate episode) pairs, with SHA-256 of both files and the row-level result.
usage: summarize_regression.py <out.json> <label>=<reference.json>|<candidate.json> [...]"""
import _bootstrap  # noqa: F401  isort:skip
import json
import sys

from fluorovla.common.provenance import sha256_file
from fluorovla.evaluation.episode_comparison import compare_regression, load_episode

out = {}
for arg in sys.argv[2:]:
    label, pair = arg.split("=", 1)
    ref, cand = pair.split("|")
    r = compare_regression(load_episode(ref), load_episode(cand))
    out[label] = {"reference_sha256": sha256_file(ref), "candidate_sha256": sha256_file(cand), "status": r["status_cand"], "n_rows": r["n_rows_cand"], "n_compared": r["n_compared"],
                  "bit_identical_rows": r["bit_identical_rows"], "PASS": r["PASS"], "max_abs_diff": r["max_abs_diff"]}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps({k: (v["PASS"], v["bit_identical_rows"]) for k, v in out.items()}))
sys.exit(0 if all(v["PASS"] for v in out.values()) else 1)
