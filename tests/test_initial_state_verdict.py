"""Verdict logic of the initial-state audit: every verdict must be reachable and none may be granted by a success rate."""
import copy

from fluorovla.evaluation.initial_state_verdict import case_role, initial_state_verdict


def _case(name, status, terminal, post_viol, rod_dev, variants=None, frames_ok=True):
    rt = {"portal": {"ct_solver_ct_error_mm": 1e-13, "ct_voxel_ct_error_mm": 1e-13}}
    return {"name": name, "original_status": status, "original_terminal_decision": terminal,
            "registration": {"portal_voxel_in_vessel_mask": True, "portal_phi_mm": -4.0, "portal_phi_neighborhood_median_mm": -4.0, "distance_to_raw_polyline_mm": 0.0},
            "roundtrip": rt if frames_ok else {"portal": {"ct_solver_ct_error_mm": 0.8, "ct_voxel_ct_error_mm": 1e-13}},
            "frame_consistency": {"round": {"fraction_inside_mask": 1.0}}, "mesh_consistency": {"median_abs_mask_value_minus_0p5": 1e-4},
            "tangent": {"73.5mm": {"rod_end_to_nearest_path_mm": rod_dev}},
            "production": {"post_settle_max_violation_mm": post_viol, "status": status},
            "counterfactuals": {"production": {"post_settle_max_violation_mm": post_viol, "status": status}, **(variants or {})}}


def _fix(v=0.2, status="DECISION_CAP"):
    return {"post_settle_max_violation_mm": v, "status": status}


def test_roles():
    assert case_role(_case("a", "SUCCESS", 50, 0.1, 1.0)) == "control"
    assert case_role(_case("b", "LUMEN_VIOLATION", 1, 9.0, 20.0)) == "early_failure"
    assert case_role(_case("c", "LUMEN_VIOLATION", 36, 0.1, 2.0)) == "late_failure"


def test_bug_confirmed_when_initialization_only_variant_resolves_and_mechanism_separates():
    cases = [_case("ctl", "SUCCESS", 50, 0.2, 1.0, {"v1": _fix()}), _case("bad", "LUMEN_VIOLATION", 1, 27.0, 20.0, {"v1": _fix()})]
    v = initial_state_verdict(cases)
    assert v["verdict"] == "BUG_CONFIRMED" and v["detail"]["resolving_initialization_variants"] == ["v1"]


def test_variant_that_breaks_a_control_does_not_resolve():
    cases = [_case("ctl", "SUCCESS", 50, 0.2, 1.0, {"v1": _fix(5.0)}), _case("bad", "LUMEN_VIOLATION", 1, 27.0, 20.0, {"v1": _fix()})]
    v = initial_state_verdict(cases)
    assert v["verdict"] == "INCONCLUSIVE" and v["detail"]["resolving_initialization_variants"] == []


def test_pass_when_frames_ok_and_early_failures_are_valid_after_settle():
    cases = [_case("ctl", "SUCCESS", 50, 0.2, 1.0), _case("bad", "LUMEN_VIOLATION", 1, 0.9, 1.5)]
    assert initial_state_verdict(cases)["verdict"] == "PASS_INITIALIZATION_VALID"


def test_frame_inconsistency_blocks_pass_and_bug_without_counterfactual():
    cases = [_case("ctl", "SUCCESS", 50, 0.2, 1.0, frames_ok=False), _case("bad", "LUMEN_VIOLATION", 1, 0.9, 1.5, frames_ok=False)]
    v = initial_state_verdict(cases)
    assert v["verdict"] == "INCONCLUSIVE" and any("round trip" in r for r in v["reasons"])


def test_invalid_control_is_inconclusive():
    cases = [_case("ctl", "SUCCESS", 50, 4.0, 1.0), _case("bad", "LUMEN_VIOLATION", 1, 27.0, 20.0)]
    assert initial_state_verdict(cases)["verdict"] == "INCONCLUSIVE"


def test_no_variant_and_invalid_after_settle_points_to_next_subsystem():
    cases = [_case("ctl", "SUCCESS", 50, 0.2, 1.0), _case("bad", "LUMEN_VIOLATION", 1, 27.0, 20.0)]
    v = initial_state_verdict(cases)
    assert v["verdict"] == "INCONCLUSIVE" and any("collision-mesh" in r for r in v["reasons"])
    assert copy.deepcopy(v)["detail"]["early_failures_invalid_after_settle"] == ["bad"]


def test_exact_origin_sign_anomaly_does_not_count_as_portal_outside_but_a_wrong_neighbourhood_does():
    ok = _case("ctl", "SUCCESS", 50, 0.2, 1.0)
    ok["registration"].update({"portal_phi_mm": 3.0, "portal_phi_neighborhood_median_mm": -3.0, "portal_exact_query_sign_anomaly": True})
    bad = _case("bad", "LUMEN_VIOLATION", 1, 0.9, 1.5)
    assert initial_state_verdict([ok, bad])["detail"]["frame_and_registration_ok"]["ctl"] is True
    bad["registration"].update({"portal_phi_mm": 3.0, "portal_phi_neighborhood_median_mm": 3.0})
    assert initial_state_verdict([ok, bad])["detail"]["frame_and_registration_ok"]["bad"] is False
