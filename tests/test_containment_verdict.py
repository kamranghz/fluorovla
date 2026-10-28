"""Findings logic of the containment audit: calibration lock, provenance classes, narrow labels, no outcome input, specificity against pre-declared positive controls."""
import copy
import json

import pytest

from fluorovla.evaluation import containment_verdict as cv

BINS = [{"id": "r2-4", "lo_mm": 2.0, "hi_mm": 4.0, "value_mm": 1.0, "n": 10, "median_signed_mm": 0.2, "p05_signed_mm": -0.3, "p95_signed_mm": 0.6, "p99_abs_mm": 0.6},
        {"id": "r4-8", "lo_mm": 4.0, "hi_mm": 8.0, "value_mm": 0.5, "n": 10, "median_signed_mm": 0.1, "p05_signed_mm": -0.2, "p95_signed_mm": 0.4, "p99_abs_mm": 0.3}]
BIAS = {"median_signed_mm": 0.2, "p05_signed_mm": -0.3, "p95_signed_mm": 0.6}
CAL = {"schema": 1, "locked": True, "environment": {"platform": "Linux-6.8-x86_64"}, "checks": {"fullres_error_within_voxel_diagonal": True, "engine_path_calibrated": True},
       "constants": {"lumen_delta_mm": 1.5, "target_phi_mm": -3.0}, "primary_reference": {"name": "fullres", "rule": "x"},
       "effects": {"reference_variants_pooled": {"fullres": BIAS}},
       "tolerances": {"fullres_reference_vs_analytic": {"value_mm": 1.0, "class": "SYNTHETIC_VALIDATION", "variant": "fullres", "bins": BINS},
                      "engine_query_vs_cpu_mesh_same_triangles": {"value_mm": 1e-3, "class": "IMPLEMENTATION"},
                      "collision_mesh_vertex_deviation_bound": {"value_mm": 3.0, "class": "SYNTHETIC_VALIDATION"}}}


def _write(tmp_path, cal):
    cal = dict(cal)
    cal["calibration_sha256"] = cv.calibration_hash(cal)
    p = tmp_path / "numerical_tolerances.json"
    p.write_text(json.dumps(cal))
    (tmp_path / "numerical_tolerances.json.sha256").write_text(f"{cal['calibration_sha256']}  numerical_tolerances.json\n")
    return p, cal


def test_locked_calibration_loads_and_tampering_is_detected(tmp_path):
    p, cal = _write(tmp_path, CAL)
    assert cv.load_locked_calibration(p)["calibration_sha256"] == cal["calibration_sha256"]
    tampered = json.loads(p.read_text())
    tampered["tolerances"]["fullres_reference_vs_analytic"]["bins"][0]["value_mm"] = 99.0
    p.write_text(json.dumps(tampered))
    with pytest.raises(cv.CalibrationError):
        cv.load_locked_calibration(p)


@pytest.mark.parametrize("change", [lambda c: c.update(locked=False), lambda c: c["environment"].update(platform="Windows-11"), lambda c: c["checks"].update(engine_path_calibrated=False),
                                    lambda c: c["tolerances"].pop("engine_query_vs_cpu_mesh_same_triangles"), lambda c: c["tolerances"]["fullres_reference_vs_analytic"].update(bins=[]),
                                    lambda c: c.update(primary_reference={"name": "other"})])
def test_unlocked_non_linux_or_incomplete_calibration_is_refused(tmp_path, change):
    cal = copy.deepcopy(CAL)
    change(cal)
    p, _ = _write(tmp_path, cal)
    with pytest.raises(cv.CalibrationError):
        cv.load_locked_calibration(p)


def test_missing_sidecar_is_refused(tmp_path):
    p, _ = _write(tmp_path, CAL)
    (tmp_path / "numerical_tolerances.json.sha256").unlink()
    with pytest.raises(cv.CalibrationError):
        cv.load_locked_calibration(p)


def test_every_threshold_has_a_provenance_class_and_no_engine_effect_is_a_tolerance():
    thr = cv.build_thresholds(CAL)
    assert thr and all(t["class"] in cv.CLASSES and t["source"] for t in thr.values())
    assert thr["lumen_delta_mm"]["class"] == "PROTOCOL" and thr["tol_fullres_reference_by_bin"]["class"] == "SYNTHETIC_VALIDATION"
    assert thr["engine_query_agreement_tol_mm"]["class"] == "IMPLEMENTATION" and thr["engine_query_agreement_tol_mm"]["value"] == pytest.approx(1e-3)
    assert not any("engine_vs" in k or "ds" in k.split("_") for k in thr)
    assert {k for k, t in thr.items() if t["class"] == "HEURISTIC"} == {"tol_fullres_reference_p99_by_bin"}      # only the descriptive secondary analysis; no status or label reads it


def test_heuristic_threshold_can_never_support_a_confirmed_status():
    with pytest.raises(ValueError):
        cv.Finding("X", "CONFIRMED", ["h"], {}, [], ["HEURISTIC"])
    with pytest.raises(ValueError):
        cv.Finding("X", "CONFIRMED", ["h"], {}, [], ["NOT_A_CLASS"])
    assert cv.Finding("X", "QUANTITATIVE_ONLY", ["h"], {}, [], ["HEURISTIC"]).status == "QUANTITATIVE_ONLY"      # descriptive use is allowed


def test_bin_tolerance_function_selects_the_bin_of_the_local_radius():
    import numpy as np
    tol, ok = cv.bin_tolerance_fn(cv.build_thresholds(CAL))(np.array([3.0, 4.0, 7.9, 8.0, 1.0, 9.0]))
    assert ok.tolist() == [True, True, True, True, False, False] and tol[:4].tolist() == [1.0, 0.5, 0.5, 0.5]


# --------------------------------------------------------------------------- synthetic findings
def _sd(v_full, v_ds=0.0, outside=0, sign=1.0):
    d = {"axis_violation_mm": v_full, "capsule_penetration_mm": v_full + 3.0, "first_axis_exceedance_arclength_mm": 40.0, "first_contact_arclength_mm": 30.0,
         "worst_sample": {"rod_index": 2, "fraction": 1.0, "point_ct_mm": [0, 0, 0], "phi_mm": v_full}, "n_samples_outside_grid": outside, "n_dense_outside_grid": 0}
    return {"sign_scale": sign, "straight_deployment": {"fullres": d, "ds": dict(d, axis_violation_mm=v_ds)},
            "path_clearance": {"fullres": {"min_wall_distance_mm": 2.0}, "ds": {"min_wall_distance_mm": 1.0}},
            "mesh_fidelity": {"vertex_signed_distance_to_fullres_boundary_mm": {"median": 1.0}, "cpu_rebuild_matches_cache": True}}


def _event(engine, lo, up, decision=3, sign_bad=0, p99=None):
    return {"stage": "decision_substep", "decision": decision, "frame": 18, "sub": 4, "n_active": 3, "engine_violation_mm": engine, "fullres_violation_mm": 0.5 * (lo + up),
            "fullres_violation_lower_mm": lo, "fullres_violation_upper_mm": up, "n_samples_outside_calibrated_bins": 0, "ds_violation_mm": 1.0, "worst_sample_engine": {},
            "n_engine_sign_disagreements_beyond_bound": sign_bad, "criterion_flip_fullres_vs_engine": False,
            "fullres_violation_lower_p99_mm": None if p99 is None else p99[0], "fullres_violation_upper_p99_mm": None if p99 is None else p99[1], "engine_minus_fullres_over_samples_mm": {}, "engine_minus_ds_over_samples_mm": {}}


def _ta(trigger=None, peak=None, ready=None, ok=True):
    """Trace analysis; ``trigger``/``peak``/``ready`` are (engine, lower, upper) violation triples in mm."""
    ke = {k: (_event(*v) if v else None) for k, v in (("trigger", trigger), ("peak", peak or trigger), ("ready", ready))}
    return {"observer_consistency_max_abs_diff_mm": 0.0, "start_vs_analytic_straight_rod_max_mm": 0.0 if ok else 1.0, "constants_match_trace": True, "reference_inside_grid": True,
            "key_events": ke, "same_point_all_events": {"engine_minus_fullres_mm": {"n": 5}, "contact_active_engine_minus_fullres_mm": {"n": 4, "median": 0.3}}, "sign_disagreements_all_events": {"n": 0}}


META_OK = {"instrumented_matches_original_bit_identically": True, "status": "LUMEN_VIOLATION"}
THR = cv.build_thresholds(CAL)
SAFE_READY = (0.0, 0.0, 0.4)
FLIP_TRIGGER = (2.4, 0.2, 1.2)                      # engine 2.4 > 1.5 while the reference cannot exceed 1.2 <= 1.5


def _case(name, ta, role="unspecified", v_full=0.2, outcome="LUMEN_VIOLATION", meta=META_OK):
    return {"name": name, "role": role, "descriptive_outcome": outcome, "config_analysis": _sd(v_full), "trace_analysis": ta, "trace_meta": meta}


def test_collision_discretization_decision_flip_is_judged_on_the_reference_interval_only():
    f = lambda tr: cv.decision_flip_finding(_ta(trigger=tr, ready=SAFE_READY), META_OK, THR, BIAS)         # noqa: E731
    assert f((2.4, 0.2, 1.2)).status == "CONFIRMED"                            # reference upper bound 1.2 <= 1.5 < engine 2.4
    assert f((2.4, 0.2, 1.5)).status == "CONFIRMED"                            # boundary: not strictly greater
    assert f((2.4, 0.2, 1.6)).status == "INDETERMINATE"                        # the reference interval straddles the limit
    assert f((2.4, 1.6, 2.2)).status == "NOT_SUPPORTED"                        # the reference exceeds the limit beyond its own error
    assert f((1.4, 0.0, 0.6)).status == "NOT_SUPPORTED"                        # the engine did not exceed the limit
    fl = f(FLIP_TRIGGER)
    assert set(fl.thresholds_used) == {"lumen_delta_mm", "tol_fullres_reference_by_bin", "observer_consistency_tol_mm", "start_match_tol_mm"} and set(fl.classes) <= {"PROTOCOL", "SYNTHETIC_VALIDATION", "IMPLEMENTATION"}
    assert fl.evidence["reference_bias_context"]["median_inside_calibrated_reference_error_band"] is True


def test_decision_event_is_the_trigger_when_the_engine_exceeded_the_limit_there_else_the_peak():
    ta = _ta(trigger=(1.0, 0.0, 0.5), peak=(2.4, 0.2, 1.2))
    ta["key_events"]["peak"]["decision"] = 2
    fnd = cv.decision_flip_finding(ta, META_OK, THR, BIAS)
    assert fnd.evidence["event_kind"] == "peak" and fnd.status == "CONFIRMED"
    assert cv.decision_flip_finding(_ta(trigger=FLIP_TRIGGER), META_OK, THR, BIAS).evidence["event_kind"] == "trigger"


def test_engine_query_sign_disagreement_or_uncovered_radius_prevents_confirmation():
    ta = _ta(trigger=FLIP_TRIGGER)
    ta["key_events"]["trigger"]["n_engine_sign_disagreements_beyond_bound"] = 1
    assert cv.decision_flip_finding(ta, META_OK, THR).reason_codes == ["ENGINE_QUERY_SIGN_DISAGREEMENT_AT_EVENT"]
    ta2 = _ta(trigger=FLIP_TRIGGER)
    ta2["key_events"]["trigger"]["n_samples_outside_calibrated_bins"] = 2
    f = cv.decision_flip_finding(ta2, META_OK, THR)
    assert f.status == "INDETERMINATE" and f.reason_codes == ["RADIUS_OUTSIDE_CALIBRATED_BINS"]


def test_pre_action_invalidity_uses_only_the_post_settle_ready_state_and_reports_the_straight_rod_as_evidence():
    f = lambda rd, v=0.0: cv.pre_action_invalidity_finding(_sd(v), _ta(trigger=FLIP_TRIGGER, ready=rd), META_OK, THR)          # noqa: E731
    assert f((2.8, 1.8, 3.2)).status == "CONFIRMED"                            # lower bound 1.8 > 1.5
    assert f((2.8, 1.5, 3.2)).status == "INDETERMINATE"
    assert f((0.2, 0.0, 1.5)).status == "NOT_SUPPORTED"
    assert f((0.2, 0.0, 0.4), 9.0).status == "NOT_SUPPORTED"                  # a violating straight rod (9 mm) does not decide anything
    assert f((0.2, 0.0, 0.4), 9.0).evidence["straight_rod_geometry"]["axis_violation_mm"] == 9.0
    assert cv.pre_action_invalidity_finding(_sd(9.0), _ta(trigger=FLIP_TRIGGER, ready=None), META_OK, THR).reason_codes == ["NO_READY_EVENT"]


def test_straight_deployment_is_quantitative_only_for_every_configuration_and_never_confirmed():
    for v in (0.0, 1.5, 5.0, 40.0):
        o = cv.straight_deployment_observation(_sd(v))
        assert o.status == "QUANTITATIVE_ONLY" and o.thresholds_used == [] and o.evidence["axis_violation_mm"] == v
    out = cv.containment_verdict([_case("a", _ta(trigger=(0.5, 0.0, 0.4), ready=SAFE_READY))], CAL | {"calibration_sha256": "h"}, {"a": _sd(30.0), "b": _sd(0.1)})
    assert {r["status"] for r in out["all_configurations_straight_deployment"].values()} == {"QUANTITATIVE_ONLY"}
    case = out["traced_cases"][0]
    assert "STRAIGHT" not in case["label"] and not any("STRAIGHT" in m for m in case["findings"])           # no mechanism is derived from the straight geometry


@pytest.mark.parametrize("ta,meta,code", [(None, None, "NO_TRACE"), (_ta(trigger=FLIP_TRIGGER), {"instrumented_matches_original_bit_identically": None}, "TRACE_REPRODUCTION_NOT_CHECKED"),
                                          (_ta(trigger=FLIP_TRIGGER), {"instrumented_matches_original_bit_identically": False}, "TRACE_NOT_BIT_IDENTICAL"),
                                          (_ta(trigger=FLIP_TRIGGER, ok=False), META_OK, "START_NOT_ANALYTIC_STRAIGHT")])
def test_untrustworthy_trace_is_not_evaluable_and_never_confirms(ta, meta, code):
    for f in (cv.decision_flip_finding(ta, meta, THR), cv.pre_action_invalidity_finding(_sd(1.0), ta, meta, THR)):
        assert f.status == "NOT_EVALUABLE" and code in f.reason_codes


# --------------------------------------------------------------------------- specificity and labels
CONTROL_OK = _ta(trigger=(1.0, 0.0, 0.5), ready=SAFE_READY)


def test_a_positive_control_meeting_a_criterion_makes_the_mechanism_non_discriminating_everywhere():
    control = _case("ctl", _ta(trigger=FLIP_TRIGGER, ready=SAFE_READY), role="positive_control", outcome="SUCCESS")       # fabricated: the flip criterion fires on a control
    flip_case = _case("fail", _ta(trigger=FLIP_TRIGGER, ready=SAFE_READY), role="early_failure")
    out = cv.containment_verdict([control, flip_case, _case("ctl2", CONTROL_OK, role="positive_control")], CAL | {"calibration_sha256": "h"})
    by = {c["name"]: c for c in out["traced_cases"]}
    for n in ("ctl", "fail", "ctl2"):
        f = by[n]["findings"][cv.FLIP]
        assert f["status"] == "NON_DISCRIMINATING" and f["evidence"]["positive_controls_meeting_criterion"] == ["ctl"] and "CONFIRMED" not in by[n]["label"]
    assert by["fail"]["findings"][cv.FLIP]["evidence"]["raw_status"] == "CONFIRMED"
    assert by["fail"]["label"] == "INCONCLUSIVE"
    assert out["specificity"][cv.FLIP]["non_discriminating"] and not out["specificity"][cv.PRE_ACTION]["non_discriminating"]
    assert by["ctl"]["findings"][cv.PRE_ACTION]["status"] == "NOT_SUPPORTED"                                              # the other mechanism is unaffected


def test_a_flip_case_is_confirmed_when_the_positive_controls_do_not_meet_the_criterion():
    cases = [_case("ctl1", CONTROL_OK, role="positive_control", outcome="SUCCESS"), _case("ctl2", CONTROL_OK, role="positive_control", outcome="SUCCESS"),
             _case("fail", _ta(trigger=FLIP_TRIGGER, ready=SAFE_READY), role="late_failure")]
    out = cv.containment_verdict(cases, CAL | {"calibration_sha256": "h"})
    by = {c["name"]: c for c in out["traced_cases"]}
    assert by["fail"]["label"] == "COLLISION_DISCRETIZATION_DECISION_FLIP_CONFIRMED"
    assert by["ctl1"]["label"] == "NO_CONTAINMENT_AUDIT_MECHANISM_IDENTIFIED"
    assert out["specificity"][cv.FLIP]["specificity_established"] and out["specificity"][cv.FLIP]["positive_controls"] == {"ctl1": "NOT_SUPPORTED", "ctl2": "NOT_SUPPORTED"}


def test_pre_action_invalidity_label_and_multiple_mechanisms():
    controls = [_case(f"ctl{i}", CONTROL_OK, role="positive_control") for i in range(2)]
    pre = _case("pre", _ta(trigger=(2.0, 1.6, 2.4), ready=(2.8, 1.8, 3.2)), role="early_failure")
    both = _case("both", _ta(trigger=FLIP_TRIGGER, ready=(2.8, 1.8, 3.2)), role="early_failure")
    by = {c["name"]: c for c in cv.containment_verdict(controls + [pre, both], CAL | {"calibration_sha256": "h"})["traced_cases"]}
    assert by["pre"]["label"] == "PRE_ACTION_INVALIDITY_CONFIRMED" and by["both"]["label"] == "MULTIPLE_MECHANISMS_IDENTIFIED"


def test_no_trace_cannot_yield_the_no_mechanism_label():
    c = _case("c", None)
    assert cv.containment_verdict([c], CAL | {"calibration_sha256": "h"})["traced_cases"][0]["label"] == "INCONCLUSIVE"


def test_labels_do_not_depend_on_outcomes_or_on_other_cases():
    base = lambda out: cv.containment_verdict([_case("a", _ta(trigger=FLIP_TRIGGER, ready=SAFE_READY), role="early_failure", outcome=out)], CAL | {"calibration_sha256": "h"})["traced_cases"][0]["label"]     # noqa: E731
    assert base("SUCCESS") == base("LUMEN_VIOLATION") == "COLLISION_DISCRETIZATION_DECISION_FLIP_CONFIRMED"
    many = [_case(f"x{i}", CONTROL_OK, role="early_failure", outcome="SUCCESS") for i in range(50)] + [_case("a", _ta(trigger=FLIP_TRIGGER, ready=SAFE_READY), role="early_failure")]
    out = cv.containment_verdict(many, CAL | {"calibration_sha256": "h"})
    assert {c["name"]: c["label"] for c in out["traced_cases"]}["a"] == "COLLISION_DISCRETIZATION_DECISION_FLIP_CONFIRMED"
    assert "success" not in json.dumps(out["scope"]).lower()


def test_verdict_output_carries_provenance_and_has_no_global_verdict():
    out = cv.containment_verdict([_case("a", CONTROL_OK, role="positive_control")], CAL | {"calibration_sha256": "h"}, {"a": _sd(5.0), "b": _sd(0.1)})
    assert out["calibration_sha256"] == "h" and set(out["threshold_provenance"]) == set(THR) and out["primary_reference"] == "fullres"
    assert set(out["all_configurations_straight_deployment"]) == {"a", "b"}
    assert not {"verdict", "NO_DEFECT_FOUND", "SIMULATOR_VALID"} & (set(out) | set(json.dumps(out).split('"')))
    labels = {f["status"] for c in out["traced_cases"] for f in c["findings"].values()}
    assert labels <= set(cv.STATUSES)
    assert {q["mechanism"] for q in out["traced_cases"][0]["quantitative_observations"]} == {"MESH_DISCRETIZATION_OFFSET", "CENTERLINE_CLEARANCE_BELOW_CATHETER_RADIUS", "AXIS_TO_BODY_GAP",
                                                                                              "ENGINE_QUERY_INTEGRITY", "SECONDARY_P99_DESCRIPTIVE"}
    assert all(q["status"] == "QUANTITATIVE_ONLY" for q in out["traced_cases"][0]["quantitative_observations"])


def test_radius_outside_every_bin_makes_the_event_indeterminate_and_never_confirms_or_aborts():
    import numpy as np
    tol, ok = cv.bin_tolerance_fn(THR)(np.array([1.0, 9.0]))                                  # below / above the declared bins
    assert not ok.any() and np.isnan(tol).all()                                               # tolerance UNAVAILABLE, not extrapolated
    ta = _ta(trigger=FLIP_TRIGGER, ready=(2.8, 1.8, 3.2))
    for k in ("trigger", "peak", "ready"):
        ta["key_events"][k].update(n_samples_outside_calibrated_bins=1, fullres_violation_lower_mm=None, fullres_violation_upper_mm=None)
    fl, pre = cv.decision_flip_finding(ta, META_OK, THR), cv.pre_action_invalidity_finding(_sd(0.0), ta, META_OK, THR)
    assert fl.status == pre.status == "INDETERMINATE" and fl.reason_codes == pre.reason_codes == ["RADIUS_OUTSIDE_CALIBRATED_BINS"]
    out = cv.containment_verdict([_case("c", ta, role="early_failure")], CAL | {"calibration_sha256": "h"})
    assert out["traced_cases"][0]["label"] == "INCONCLUSIVE"


def test_out_of_bin_radius_with_engine_at_or_below_the_limit_is_not_supported_not_indeterminate():
    """eng <= delta needs no reference tolerance at all, so an out-of-bin-radius sample must not block that
    decision: this is checked before the bin-coverage/interval-availability check, not after it."""
    ta = _ta(trigger=(1.2, 0.0, 0.0), ready=(1.2, 0.0, 0.0))                                  # engine 1.2 <= delta 1.5 at every key event
    for k in ("trigger", "peak", "ready"):
        ta["key_events"][k].update(n_samples_outside_calibrated_bins=1, fullres_violation_lower_mm=None, fullres_violation_upper_mm=None)
    fl = cv.decision_flip_finding(ta, META_OK, THR)
    assert fl.status == "NOT_SUPPORTED" and fl.reason_codes == ["ENGINE_DID_NOT_EXCEED_LIMIT_AT_EVENT"]
    # the specificity rule needs a positive control to actually be NOT_SUPPORTED (not INDETERMINATE) to be established
    control = _case("ctl", ta, role="positive_control", outcome="SUCCESS")
    flip_case = _case("fail", _ta(trigger=FLIP_TRIGGER, ready=SAFE_READY), role="early_failure")
    out = cv.containment_verdict([control, flip_case], CAL | {"calibration_sha256": "h"})
    assert out["specificity"][cv.FLIP]["positive_controls"]["ctl"] == "NOT_SUPPORTED"
    assert out["specificity"][cv.FLIP]["specificity_established"] is True
    by = {c["name"]: c for c in out["traced_cases"]}
    assert by["fail"]["findings"][cv.FLIP]["status"] == "CONFIRMED"                            # not NON_DISCRIMINATING: the control never met the criterion


def test_secondary_p99_analysis_is_descriptive_heuristic_and_cannot_change_anything():
    ta = _ta(trigger=(2.4, 0.2, 1.6), ready=(2.8, 1.2, 3.2))                                  # max-error interval straddles the limit: INDETERMINATE for both mechanisms
    for k, p99 in (("trigger", (0.2, 1.3)), ("peak", (0.2, 1.3)), ("ready", (1.8, 2.5))):
        ta["key_events"][k].update(fullres_violation_lower_p99_mm=p99[0], fullres_violation_upper_p99_mm=p99[1])
    thr = cv.build_thresholds(CAL)
    assert thr["tol_fullres_reference_p99_by_bin"]["class"] == "HEURISTIC" and thr["tol_fullres_reference_p99_by_bin"]["value"][0]["value_mm"] == 0.6
    obs = cv.secondary_p99_observation(ta, META_OK, thr)
    assert obs.mechanism == "SECONDARY_P99_DESCRIPTIVE" and obs.status == "QUANTITATIVE_ONLY"
    assert obs.evidence[cv.FLIP]["p99_criterion"] == "CRITERION_MET" and obs.evidence[cv.PRE_ACTION]["p99_criterion"] == "CRITERION_MET" and obs.classes == ["PROTOCOL", "HEURISTIC"]
    with pytest.raises(ValueError):                                                           # it can never be promoted: a HEURISTIC class refuses CONFIRMED
        cv.Finding("SECONDARY_P99_DESCRIPTIVE", "CONFIRMED", obs.thresholds_used, obs.evidence, [], obs.classes)
    case = _case("c", ta, role="early_failure")
    with_p99 = cv.containment_verdict([case], CAL | {"calibration_sha256": "h"})
    for e in ta["key_events"].values():                                                       # remove the secondary inputs entirely
        if e:
            e.update(fullres_violation_lower_p99_mm=None, fullres_violation_upper_p99_mm=None)
    without = cv.containment_verdict([_case("c", ta, role="early_failure")], CAL | {"calibration_sha256": "h"})
    a, b = with_p99["traced_cases"][0], without["traced_cases"][0]
    assert a["label"] == b["label"] == "INCONCLUSIVE" and {k: v["status"] for k, v in a["findings"].items()} == {k: v["status"] for k, v in b["findings"].items()}
    assert all(v["status"] == "INDETERMINATE" for v in a["findings"].values())
    sec = [q for q in a["quantitative_observations"] if q["mechanism"] == "SECONDARY_P99_DESCRIPTIVE"][0]
    assert sec["status"] == "QUANTITATIVE_ONLY" and "CONFIRMED" not in json.dumps(sec["status"])
    assert [q for q in b["quantitative_observations"] if q["mechanism"] == "SECONDARY_P99_DESCRIPTIVE"][0]["evidence"][cv.FLIP]["p99_criterion"] == "UNAVAILABLE"
