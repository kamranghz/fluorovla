"""CPU-only analysis: how does the geometry of the start condition (curvature of the vessel over the 73.5-mm straight initial rod) relate to the physics-mapping outcome?

usage: analyze_start_geometry.py <configs_root> <physics_mapping_runs_dir> <out_json>
For every episode configuration: tangent-versus-chord angles, straight-rod end deviation, arc/chord ratio, total turning, and the original terminal status / decision /
tip distance to the centerline at decision 0. Also reports rank correlations. Descriptive statistics only; no threshold is fitted.
"""
import _bootstrap  # noqa: F401  isort:skip
import json
import sys
from pathlib import Path

import numpy as np

from fluorovla.anatomy.registration_audit import start_geometry_features
from fluorovla.evaluation.physics_mapping_summary import accepted_attempt
from fluorovla.physics.patient_config import PatientConfig

root, runs, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = []
for d in sorted(p for p in root.iterdir() if (p / "config.json").is_file()):
    cfg = PatientConfig.from_json(d / "config.json")
    z = np.load(cfg.path_npz_path)
    f = start_geometry_features(cfg, z["pts"], z["rad"])
    acc, _ = accepted_attempt(runs / d.name)
    r0 = {}
    if acc is not None:
        e = json.load(open(acc / "run/episode_physics_ep.json"))
        r0 = {"status": e["summary"]["status"], "decisions": e["summary"]["decisions"], "tip_dist_decision0_mm": e["rows"][0]["priv_tip_dist_centerline_mm"],
              "violation_decision0_mm": e["rows"][0]["axis_violation_mm"]}
    rows.append({"name": d.name, "patient": cfg.patient_id, "F_mm": cfg.F_mm, **f, **r0})
from scipy.stats import spearmanr
res = {"n": len(rows), "rows": rows, "spearman_vs_tip_dist_decision0": {}, "spearman_vs_episode_length": {}}
for k in ("angle_30mm_deg", "angle_73.5mm_deg", "rod_end_deviation_mm", "arc_over_chord_73.5mm", "total_turning_over_rod_deg", "median_twin_radius_over_rod_mm", "F_mm"):
    x = [r[k] for r in rows if "tip_dist_decision0_mm" in r]
    res["spearman_vs_tip_dist_decision0"][k] = float(spearmanr(x, [r["tip_dist_decision0_mm"] for r in rows if "tip_dist_decision0_mm" in r])[0])
    res["spearman_vs_episode_length"][k] = float(spearmanr(x, [r["decisions"] for r in rows if "tip_dist_decision0_mm" in r])[0])
succ = [r for r in rows if r.get("status") == "SUCCESS"]
res["success_rows"] = [{k: r[k] for k in ("name", "angle_30mm_deg", "rod_end_deviation_mm", "arc_over_chord_73.5mm", "total_turning_over_rod_deg")} for r in succ]
json.dump(res, open(out, "w"), indent=1, default=float)
print(json.dumps({"n": res["n"], "spearman_vs_tip_dist_decision0": res["spearman_vs_tip_dist_decision0"], "spearman_vs_episode_length": res["spearman_vs_episode_length"]}, indent=1))
for r in succ:
    print("SUCCESS", r["name"], {k: round(r[k], 2) for k in ("angle_30mm_deg", "rod_end_deviation_mm", "arc_over_chord_73.5mm", "total_turning_over_rod_deg")})
