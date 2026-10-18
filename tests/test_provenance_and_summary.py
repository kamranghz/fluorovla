"""Provenance manifests, attempt-addressable summaries (infrastructure interruption is never a physics result) and the reproduction verdict."""
import copy
import json

import pytest

from fluorovla.common.provenance import verify_file_manifest, write_file_manifest, write_run_manifest
from fluorovla.evaluation.episode_comparison import compare_regression, reproduction_verdict
from fluorovla.evaluation.physics_mapping_summary import summarize


def _episode(status="SUCCESS", n=3, viol=0.5):
    rows = [{"action": "CRUISE", "phase": "approach", "in_zone_end": False, "insert_velocity_mps": 0.012, "inserted_length_m": 0.0012 * i, "previous_insert_velocity_mps": 0.012,
             "n_active_start": 3, "n_active_end": 3, "physics_frame_start": 6 * i, "priv_tip_extremity_mm": 80.0 + i, "priv_tip_extremity_end_mm": 81.0 + i,
             "priv_tip_polyline_s_mm": 77.0 + i, "priv_tip_dist_centerline_mm": 1.0, "priv_local_radius_mm": 4.0, "axis_violation_mm": viol, "outside_frac": 0.0,
             "tip_speed_mps": 0.01, "root_disp_m": 0.0, "dwell_run": 0} for i in range(n)]
    return {"summary": {"status": status, "decisions": n, "success_decision_index": None, "post_success_hold_decisions": 0, "final_tip_extremity_mm": 83.0,
                        "final_inserted_length_m": 0.0, "max_axis_violation_mm": viol, "max_root_displacement_m": 0.0, "active_bodies_final": 3, "all_finite": True,
                        "timeout_decisions": 50}, "rows": rows}


def test_manifest_roundtrip_and_detection(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/b.txt").write_text("y")
    mf = tmp_path / "MANIFEST.txt"
    assert write_file_manifest(tmp_path, mf) == 2
    assert verify_file_manifest(tmp_path, mf)["bad"] == []
    (tmp_path / "a.txt").write_text("tampered")
    assert verify_file_manifest(tmp_path, mf)["bad"] == ["a.txt"]


def test_run_manifest_never_overwrites(tmp_path):
    write_run_manifest(tmp_path / "run", "test", ["s0000"], {}, "COMPLETED")
    with pytest.raises(FileExistsError):
        write_run_manifest(tmp_path / "run", "test", ["s0000"], {}, "COMPLETED")


def test_reproduction_verdicts():
    a = _episode()
    assert reproduction_verdict(a, copy.deepcopy(a))["verdict"] == "PASS_BIT_IDENTICAL"
    b = copy.deepcopy(a)
    b["rows"][1]["priv_tip_extremity_end_mm"] += 0.04
    assert reproduction_verdict(a, b)["verdict"] == "PASS_TOLERANCE"
    b["rows"][1]["priv_tip_extremity_end_mm"] += 0.04
    assert reproduction_verdict(a, b)["verdict"] == "BLOCK"
    c = copy.deepcopy(a)
    c["rows"][2]["action"] = "SLOW"
    assert reproduction_verdict(a, c)["verdict"] == "BLOCK"
    d = copy.deepcopy(a)
    d["summary"]["status"] = "LUMEN_VIOLATION"
    assert reproduction_verdict(a, d)["verdict"] == "BLOCK"
    assert compare_regression(a, copy.deepcopy(a))["PASS"]


def test_interrupted_attempts_are_not_physics_results(tmp_path):
    for k in range(3):
        d = tmp_path / f"s0000_left_F{k}"
        (d / "attempt_001").mkdir(parents=True)                                   # interrupted: no DONE.json
        if k < 2:
            a2 = d / "attempt_002"
            (a2 / "run").mkdir(parents=True)
            json.dump(_episode("SUCCESS"), open(a2 / "run/episode_physics_ep.json", "w"))
            json.dump({"state": "COMPLETED", "has_result": True}, open(a2 / "DONE.json", "w"))
    res = summarize(tmp_path, [f"s0000_left_F{k}" for k in range(3)], write_accepted=False)
    assert res["summary"]["attempt_states"]["INFRASTRUCTURE_INTERRUPTED"] == 3
    assert res["episodes"][2]["status"] == "NO_ACCEPTED_ATTEMPT"                   # never converted to a physics failure
    assert res["summary"]["sides_admitted"] == 0 and res["summary"]["status_counts"].get("NO_ACCEPTED_ATTEMPT") == 1
