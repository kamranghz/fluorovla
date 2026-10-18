"""Comparison of two physics-episode result files: row-level regression and the native-Linux reproduction verdict."""
from __future__ import annotations

import json

COMPARED_NUM = ("insert_velocity_mps", "inserted_length_m", "previous_insert_velocity_mps", "n_active_start", "n_active_end", "physics_frame_start", "priv_tip_extremity_mm",
                "priv_tip_extremity_end_mm", "priv_tip_polyline_s_mm", "priv_tip_dist_centerline_mm", "priv_local_radius_mm", "axis_violation_mm", "outside_frac",
                "tip_speed_mps", "root_disp_m", "dwell_run")
COMPARED_STR = ("action", "phase")
SUMMARY_KEYS = ("success_decision_index", "post_success_hold_decisions", "final_tip_extremity_mm", "final_inserted_length_m", "max_axis_violation_mm",
                "max_root_displacement_m", "active_bodies_final", "all_finite")
TOL_MM = 0.05          # pre-declared reproduction tolerance (priv_tip_extremity_end_mm and axis_violation_mm) when bit identity is impossible


def load_episode(path) -> dict:
    return json.load(open(path))


def compare_regression(ref: dict, cand: dict, rel_tol: float = 1e-9) -> dict:
    """Row-by-row regression criterion: status, action sequence, summary keys and numeric rows equal (bit-identical, or within ``rel_tol``)."""
    R, C = ref["rows"], cand["rows"]
    out = {"n_rows_ref": len(R), "n_rows_cand": len(C), "status_ref": ref["summary"]["status"], "status_cand": cand["summary"]["status"], "mismatches": [],
           "max_rel_diff": 0.0, "max_abs_diff": 0.0, "n_compared": 0}
    if len(R) == len(C):
        for i, (a, b) in enumerate(zip(R, C)):
            for k in COMPARED_STR + ("in_zone_end",):
                if a[k] != b[k]:
                    out["mismatches"].append((i, k, a[k], b[k]))
            for k in COMPARED_NUM:
                x, y = float(a[k]), float(b[k])
                d = abs(x - y)
                rel = d / max(abs(x), 1e-300) if x != 0 else d
                out["max_abs_diff"] = max(out["max_abs_diff"], d)
                out["max_rel_diff"] = max(out["max_rel_diff"], rel)
                out["n_compared"] += 1
                if rel > rel_tol and d > 1e-15:
                    out["mismatches"].append((i, k, x, y))
    for k in SUMMARY_KEYS:
        if ref["summary"].get(k) != cand["summary"].get(k):
            out["mismatches"].append(("summary", k, ref["summary"].get(k), cand["summary"].get(k)))
    out["bit_identical_rows"] = out["max_abs_diff"] == 0.0 and not out["mismatches"]
    out["PASS"] = len(R) == len(C) and not out["mismatches"] and out["status_ref"] == out["status_cand"]
    return out


def reproduction_verdict(ref: dict, cand: dict) -> dict:
    """PASS_BIT_IDENTICAL / PASS_TOLERANCE / BLOCK under the pre-declared reproduction criteria (identical status, action sequence and decision count;
    <= 0.05 mm in priv_tip_extremity_end_mm and axis_violation_mm)."""
    Ra, Rb = ref["rows"], cand["rows"]
    v = {"status_ref": ref["summary"]["status"], "status_cand": cand["summary"]["status"], "n_ref": len(Ra), "n_cand": len(Rb)}
    v["same_status"] = v["status_ref"] == v["status_cand"]
    v["same_n"] = len(Ra) == len(Rb)
    n = min(len(Ra), len(Rb))
    v["same_actions"] = [r["action"] for r in Ra[:n]] == [r["action"] for r in Rb[:n]] and v["same_n"]
    num = [k for k in Ra[0] if isinstance(Ra[0][k], (int, float)) and not isinstance(Ra[0][k], bool) and k != "elapsed_wall_s"]
    v["max_abs_diff_all_numeric"] = max((abs(float(Ra[i][k]) - float(Rb[i][k])) for i in range(n) for k in num if k in Rb[i]), default=0.0)
    v["max_abs_diff_tip_ext_end_mm"] = max((abs(Ra[i]["priv_tip_extremity_end_mm"] - Rb[i]["priv_tip_extremity_end_mm"]) for i in range(n)), default=0.0)
    v["max_abs_diff_axis_violation_mm"] = max((abs(Ra[i]["axis_violation_mm"] - Rb[i]["axis_violation_mm"]) for i in range(n)), default=0.0)
    bit = v["same_status"] and v["same_actions"] and v["max_abs_diff_all_numeric"] == 0.0
    tol = v["same_status"] and v["same_actions"] and v["max_abs_diff_tip_ext_end_mm"] <= TOL_MM and v["max_abs_diff_axis_violation_mm"] <= TOL_MM
    v["verdict"] = "PASS_BIT_IDENTICAL" if bit else ("PASS_TOLERANCE" if tol else "BLOCK")
    return v
