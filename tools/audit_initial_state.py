"""Initial-state / registration audit of physics-mapping episodes (diagnostic only; no physics constant, Teacher rule or task definition is changed).

usage: audit_initial_state.py <out_dir> --configs-root DIR --episodes NAME [NAME ...] [--reference-runs DIR] [--no-counterfactuals] [--variant-cap N]

  --configs-root     directory containing <name>/{config.json, landing_zone.json, path_ext.npz} (workload/episode_configs of a bundle)
  --reference-runs   physics_mapping run directory of the original run (<name>/attempt_NNN/run/episode_physics_ep.json); used for the failure timing and to check that
                     the instrumented re-run reproduces the original episode bit-identically (the observer must not change the physics)
Writes <out_dir>/{cases/<name>.json, audit_summary.json, counterfactual_comparison.json, COMPARISON.md, STAGE3A_VERDICT.json, MANIFEST.json}.
Needs CUDA + Warp + the pinned Newton checkout (run it on native Linux; WSL results are development only).
"""
import _bootstrap  # noqa: F401  isort:skip
import argparse
import json
from pathlib import Path

import numpy as np

import fluorovla.task_contract as tc
from fluorovla.anatomy import registration_audit as ra
from fluorovla.common.provenance import write_run_manifest
from fluorovla.evaluation.episode_comparison import compare_regression
from fluorovla.evaluation.physics_mapping_summary import accepted_attempt
from fluorovla.evaluation.initial_state_verdict import initial_state_verdict
from fluorovla.physics.episode_runner import run_physics_episode
from fluorovla.physics.initial_state_audit import R_MM, InitialStateObserver
from fluorovla.physics.lumen_sdf import LumenSDF
from fluorovla.physics.patient_config import PatientConfig

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("--configs-root", required=True)
ap.add_argument("--episodes", nargs="+", required=True)
ap.add_argument("--reference-runs", default=None)
ap.add_argument("--no-counterfactuals", action="store_true")
ap.add_argument("--resume", action="store_true", help="reuse cases/<name>.json that already exist in the output directory (interrupted runs)")
ap.add_argument("--variant-cap", type=int, default=20, help="decision cap of the counterfactual rollouts (diagnostic; status DECISION_CAP when reached)")
a = ap.parse_args()
out = Path(a.out_dir)
(out / "cases").mkdir(parents=True, exist_ok=True)


def run_variant(cfg, proj, tag, run_dir, cap):
    obs = InitialStateObserver(proj)
    s = run_physics_episode(cfg, run_dir, tag, observer=obs, decision_cap=cap)
    ps = obs.summary()
    return s, obs, ps


def metrics_of(s, ps):
    p = ps["post_settle"]
    return {"post_settle_max_violation_mm": p["max_axis_violation_mm"], "post_settle_min_clearance_mm": p["min_clearance_mm"], "post_settle_max_body_to_path_distance_mm": p["max_body_to_path_distance_mm"],
            "post_settle_tip_dist_to_path_mm": p["tip_dist_to_path_mm"], "settle_max_violation_mm": ps["settle_max_violation_mm"], "settle_max_speed_mps": ps["settle_max_speed_mps"],
            "status": s["status"], "decisions": s["decisions"], "rollout_max_axis_violation_mm": s["max_axis_violation_mm"]}


cases, cf_table = [], {}
for name in a.episodes:
    done = out / "cases" / f"{name}.json"
    if a.resume and done.is_file():
        case = json.load(open(done))
        cases.append(case)
        cf_table[name] = case["counterfactuals"]
        print(f"{name}: reused existing case result", flush=True)
        continue
    cdir = Path(a.configs_root) / name
    cfg = PatientConfig.from_json(cdir / "config.json")
    z = np.load(cfg.path_npz_path)
    pts, rad = z["pts"], z["rad"]
    proj = tc.PathProjector(pts, rad)
    twin = ra.load_twin(cfg)
    sdf = LumenSDF(cfg)
    phi_fn = lambda p, _s=sdf: _s.query(np.asarray(p, dtype=np.float32).reshape(-1, 3))[0]          # noqa: E731  (solver-local metres -> phi metres)
    ref_ep = None
    if a.reference_runs:
        acc, _ = accepted_attempt(Path(a.reference_runs) / name)
        if acc is not None:
            ref_ep = json.load(open(acc / "run/episode_physics_ep.json"))
    import time as _time
    _t0 = _time.time()
    print(f"[{name}] registration / frame checks ...", flush=True)
    case = {"name": name, "config": json.load(open(cdir / "config.json")),
            "registration": ra.portal_registration(cfg, pts, rad, twin, phi_fn), "roundtrip": ra.roundtrip_report(cfg, pts, rad, twin),
            "frame_consistency": ra.centerline_mask_consistency(twin), "mesh_consistency": ra.mesh_mask_consistency(cfg, twin),
            "tangent": ra.tangent_comparison(cfg, pts, rad), "sdf_sign": ra.sdf_sign_consistency(cfg, twin, phi_fn, pts, rad), "portal_probe": ra.portal_neighborhood_probe(cfg, phi_fn), "start_geometry": ra.start_geometry_features(cfg, pts, rad), "path_clearance": ra.path_clearance_profile(cfg, pts, rad, phi_fn, R_MM)}
    print(f"[{name}] geometric checks done in {_time.time() - _t0:.0f} s; instrumented production episode ...", flush=True)
    # production configuration, instrumented
    s, obs, ps = run_variant(cfg, proj, "production", out / "runs" / name / "production", None)
    case["initial_state"] = ps
    tr = ps["violation_track"] + ps["last_60_substeps"]
    case["observer_consistency_max_abs_diff_mm"] = max([abs(r["engine_violation_mm"] - r["audit_max_axis_violation_mm"]) for r in tr], default=None)      # same state, same samples
    case["production"] = metrics_of(s, ps)
    inst = json.load(open(out / "runs" / name / "production" / "episode_physics_production.json"))
    case["instrumented_matches_original_bit_identically"] = (compare_regression(ref_ep, inst)["bit_identical_rows"] if ref_ep else None)
    st = s["status"]
    case["original_status"] = ref_ep["summary"]["status"] if ref_ep else st
    case["original_terminal_decision"] = (ref_ep["summary"]["decisions"] - 1) if ref_ep else (s["decisions"] - 1)
    case["failure_timing"] = {"terminal_decision_index": s["decisions"] - 1, "status": st, "max_axis_violation_mm": s["max_axis_violation_mm"],
                              "unsafe_immediately_after_settle": ps["post_settle"]["max_axis_violation_mm"] > tc.LUMEN_DELTA_MM,
                              "first_decision_engine_violation_mm": max([r["engine_violation_mm"] for r in ps["violation_track"] if r["decision"] == 0], default=None),
                              "worst_point_at_termination": (ps["last_60_substeps"][-1] if ps["last_60_substeps"] else None)}
    cf = {"production": case["production"]}
    print(f"[{name}] production episode done at {_time.time() - _t0:.0f} s", flush=True)
    if not a.no_counterfactuals:
        for vname, vcfg in ra.counterfactual_configs(cfg, pts, rad).items():
            vs, vobs, vps = run_variant(vcfg, proj, "variant", out / "runs" / name / "variants" / vname.replace("|", "_").replace("=", "-"), a.variant_cap)
            m = metrics_of(vs, vps)
            m["tangent_vs_chords"] = {k: v["angle_deg"] for k, v in ra.tangent_comparison(vcfg, pts, rad).items() if isinstance(v, dict) and "angle_deg" in v}
            m["straight_rod_end_deviation_73p5_mm"] = ra.tangent_comparison(vcfg, pts, rad)["73.5mm"]["rod_end_to_nearest_path_mm"]
            m["portal_ct_mm"] = (vcfg.ct_offset * 1000.0).tolist()
            cf[vname] = m
    case["counterfactuals"] = cf
    cf_table[name] = cf
    json.dump(case, open(out / "cases" / f"{name}.json", "w"), indent=1, default=float)
    cases.append(case)
    print(f"{name}: original {case['original_status']} at decision {case['original_terminal_decision']}; post-settle max violation {case['production']['post_settle_max_violation_mm']:.3f} mm", flush=True)

verdict = initial_state_verdict(cases)
json.dump(cf_table, open(out / "counterfactual_comparison.json", "w"), indent=1, default=float)
json.dump(verdict, open(out / "STAGE3A_VERDICT.json", "w"), indent=1, default=float)

rows = []
for c in cases:
    t = c["tangent"]
    rows.append({"case": c["name"], "original": f"{c['original_status']}@{c['original_terminal_decision']}", "portal_to_raw_path_mm": c["registration"]["distance_to_raw_polyline_mm"],
                 "portal_in_mask": c["registration"].get("portal_voxel_in_vessel_mask"), "portal_phi_exact_mm": c["registration"].get("portal_phi_mm"), "portal_phi_neighborhood_mm": c["registration"].get("portal_phi_neighborhood_median_mm"),
                 "tangent_angle_5/15/30/73.5_deg": [round(t[f"{w:g}mm"]["angle_deg"], 2) for w in ra.TANGENT_WINDOWS_MM], "rod_end_dev_73.5_mm": round(t["73.5mm"]["rod_end_to_nearest_path_mm"], 2),
                 "post_settle_max_body_to_path_mm": c["production"]["post_settle_max_body_to_path_distance_mm"], "post_settle_min_clearance_mm": c["production"]["post_settle_min_clearance_mm"],
                 "post_settle_max_violation_mm": c["production"]["post_settle_max_violation_mm"], "terminal_decision": c["failure_timing"]["terminal_decision_index"],
                 "settle_max_speed_mps": c["production"]["settle_max_speed_mps"], "max_roundtrip_error_mm": max(max(v["ct_solver_ct_error_mm"], v["ct_voxel_ct_error_mm"]) for v in c["roundtrip"].values()),
                 "sdf_deep_sign_mismatch_fraction": c["sdf_sign"]["deep_mismatch_fraction"], "arc_over_chord_73.5mm": t["path_arc_over_chord_73.5mm"], "path_wall_distance_below_R_fraction": c["path_clearance"]["fraction_wall_distance_below_catheter_radius"], "min_path_wall_distance_mm": c["path_clearance"]["min_wall_distance_mm"],
                 "instrumented_bit_identical_to_original": c["instrumented_matches_original_bit_identically"]})
json.dump({"cases": rows}, open(out / "audit_summary.json", "w"), indent=1, default=float)
with open(out / "COMPARISON.md", "w", encoding="utf-8") as f:
    f.write("| " + " | ".join(rows[0].keys()) + " |\n|" + "---|" * len(rows[0]) + "\n")
    for r in rows:
        f.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")
write_run_manifest(out, "initial-state / registration audit", a.episodes, {"configs_root": a.configs_root, "variant_cap": a.variant_cap, "reference_runs": a.reference_runs}, "COMPLETED",
                   {"verdict": verdict["verdict"]})
print(json.dumps({"verdict": verdict["verdict"], "reasons": verdict["reasons"]}, indent=1))
