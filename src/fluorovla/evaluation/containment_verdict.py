"""Findings logic of the containment audit. CPU-only and unit-tested.

Inputs are the per-case analysis dictionaries produced by ``fluorovla.anatomy.containment_audit`` and the locked numerical calibration. The audit examines a BOUNDED set of
mechanisms; its findings say nothing about any mechanism outside that set, and no global "defect-free" or "simulator valid" verdict exists.

Every value that drives a conclusion carries one provenance class:
  PROTOCOL              fixed by the project protocol / task contract (``LUMEN_DELTA_MM``)
  IMPLEMENTATION        fixed by the simulator source or by floating-point representation (catheter radius, float32 rounding)
  SYNTHETIC_VALIDATION  measured on analytic synthetic geometry by ``calibrate`` on native Linux and locked by hash before any patient analysis
  HEURISTIC             judgement without validation; may be reported descriptively but can never support a status containing CONFIRMED

Mechanisms (status per traced case), each decided only with PROTOCOL and SYNTHETIC_VALIDATION thresholds:
  COLLISION_DISCRETIZATION_DECISION_FLIP  at the terminating event (the trigger event if the engine exceeded the limit there, else the peak event) the engine's axis violation exceeds
                                          the limit while the full-resolution reference violation, increased by the calibrated reference error of each sample's radius bin, does not
  PRE_ACTION_INVALIDITY                   at the post-settle `ready` state the reference violation, decreased by the calibrated reference error, already exceeds the limit
The analytic straight-deployment metrics of ALL configurations are QUANTITATIVE_ONLY: the protocol test does not apply before the settle phases. Mesh offset, centerline clearance, the
axis-versus-body gap and engine query integrity are measured and reported, never promoted to a mechanism claim here. Numerical error (the reference error) is separated from the
effect under test (the collision-surface discretization): the effect is never used as a tolerance.

Specificity: a mechanism whose criterion is met by any case with the pre-declared role `positive_control` is NON_DISCRIMINATING for the whole audit and cannot be CONFIRMED in any case.
Roles come from the configuration. No outcome statistic (success rate, number of failures) is an input to any status or label.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASSES = ("PROTOCOL", "IMPLEMENTATION", "SYNTHETIC_VALIDATION", "HEURISTIC")
STATUSES = ("CONFIRMED", "NOT_SUPPORTED", "INDETERMINATE", "NOT_EVALUABLE", "NON_DISCRIMINATING", "QUANTITATIVE_ONLY")
FLIP = "COLLISION_DISCRETIZATION_DECISION_FLIP"
PRE_ACTION = "PRE_ACTION_INVALIDITY"
MECHANISMS = (FLIP, PRE_ACTION)
POSITIVE_CONTROL = "positive_control"
OBSERVER_CONSISTENCY_TOL_MM = 1e-6        # IMPLEMENTATION: the probe recomputes the engine measure from the same float64 values
START_MATCH_TOL_MM = 1e-3                 # IMPLEMENTATION: float32 body poses against the analytic straight placement (the tolerance used when the observer was validated)
REQUIRED_TOLERANCES = ("fullres_reference_vs_analytic", "engine_query_vs_cpu_mesh_same_triangles")


class CalibrationError(RuntimeError):
    """The numerical calibration is missing, unlocked, not native-Linux, or does not match its recorded hash."""


# --------------------------------------------------------------------------- calibration lock
def canonical_json_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def calibration_hash(cal: dict) -> str:
    return canonical_json_sha256({k: v for k, v in cal.items() if k != "calibration_sha256"})


def load_locked_calibration(path: str | Path) -> dict:
    """Load numerical_tolerances.json and verify its hash (inside the file and in the sidecar), the native-Linux lock and the presence of the required tolerances."""
    path = Path(path)
    if not path.is_file():
        raise CalibrationError(f"{path} does not exist; run `calibrate` first")
    cal = json.load(open(path, encoding="utf-8"))
    if cal.get("calibration_sha256") != calibration_hash(cal):
        raise CalibrationError("calibration content does not match its recorded hash")
    side = path.with_name(path.name + ".sha256")
    if not side.is_file() or side.read_text().split()[0] != cal["calibration_sha256"]:
        raise CalibrationError("hash sidecar missing or different from the calibration")
    if cal.get("locked") is not True or "Linux" not in str(cal.get("environment", {}).get("platform", "")):
        raise CalibrationError("calibration is not a locked native-Linux calibration")
    miss = [t for t in REQUIRED_TOLERANCES if t not in cal["tolerances"]]
    if miss:
        raise CalibrationError(f"required tolerances missing: {miss}")
    if not cal["tolerances"]["fullres_reference_vs_analytic"].get("bins") or cal.get("primary_reference", {}).get("name") not in ("fullres", "fullres_hc"):
        raise CalibrationError("the radius-binned reference tolerance or the primary reference variant is missing")
    if not (cal["checks"].get("fullres_error_within_voxel_diagonal") and cal["checks"].get("engine_path_calibrated")):
        raise CalibrationError(f"calibration checks failed: {cal['checks']}")
    return cal


def primary_reference_name(cal: dict) -> str:
    return cal["primary_reference"]["name"]


# --------------------------------------------------------------------------- threshold registry
def build_thresholds(cal: dict) -> dict:
    c, t = cal["constants"], cal["tolerances"]
    ref = t["fullres_reference_vs_analytic"]
    reg = {"lumen_delta_mm": (c["lumen_delta_mm"], "PROTOCOL", "task_contract.LUMEN_DELTA_MM: the lumen-violation limit of the episode"),
           "contact_threshold_phi_mm": (c["target_phi_mm"], "IMPLEMENTATION", "lumen_sdf.TARGET_PHI_M = -float(engine.RADIUS)"),
           "observer_consistency_tol_mm": (OBSERVER_CONSISTENCY_TOL_MM, "IMPLEMENTATION", "same float64 expression recomputed from the recorded samples"),
           "start_match_tol_mm": (START_MATCH_TOL_MM, "IMPLEMENTATION", "float32 body poses against the analytic straight placement"),
           "engine_query_agreement_tol_mm": (t["engine_query_vs_cpu_mesh_same_triangles"]["value_mm"], t["engine_query_vs_cpu_mesh_same_triangles"]["class"],
                                             "calibration: float32 engine query against a CPU query on the same triangles"),
           "tol_fullres_reference_by_bin": ([dict(b) for b in ref["bins"]], ref["class"], f"calibration: {ref['variant']} reference error per twin-radius bin (analytic tubes)")}
    if all("p99_abs_mm" in b for b in ref["bins"]):
        reg["tol_fullres_reference_p99_by_bin"] = ([dict(b, value_mm=b["p99_abs_mm"]) for b in ref["bins"]], "HEURISTIC",
                                                    "choice of the p99 instead of the max of the calibrated reference error: descriptive only, can never support CONFIRMED")
    if "collision_mesh_vertex_deviation_bound" in t:
        reg["engine_sign_guard_bound_mm"] = (t["collision_mesh_vertex_deviation_bound"]["value_mm"], t["collision_mesh_vertex_deviation_bound"]["class"],
                                             "calibration: largest deviation of a collision-mesh vertex from the analytic surface")
    return {k: {"value": v, "class": cl, "source": src} for k, (v, cl, src) in reg.items()}


def bin_tolerance_fn(thr: dict, key: str = "tol_fullres_reference_by_bin"):
    """points' twin radius array -> (tolerance_mm array, in_bin bool array) from the locked per-bin reference error. Used by the analysis step.
    A radius outside every bin has no tolerance (NaN, in_bin False): UNAVAILABLE, never extrapolated."""
    from fluorovla.anatomy.containment_audit import radius_bin_lookup
    bins = thr[key]["value"]
    look = radius_bin_lookup(bins)
    vals = np.array([b["value_mm"] for b in bins] + [np.nan])

    def f(radius):
        idx = look(radius)
        tol = vals[idx]                                           # idx -1 selects the trailing NaN
        return tol, idx >= 0
    return f


@dataclass
class Finding:
    mechanism: str
    status: str
    thresholds_used: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    reason_codes: list = field(default_factory=list)
    classes: list = field(default_factory=list)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(self.status)
        if "CONFIRMED" in self.status and "HEURISTIC" in self.classes:
            raise ValueError("a HEURISTIC threshold can never support a CONFIRMED status")
        if "CONFIRMED" in self.status and any(c not in CLASSES for c in self.classes):
            raise ValueError("every threshold behind a CONFIRMED status needs a provenance class")

    def to_dict(self) -> dict:
        return {"mechanism": self.mechanism, "status": self.status, "reason_codes": self.reason_codes, "thresholds_used": self.thresholds_used, "provenance_classes": sorted(set(self.classes)),
                "evidence": self.evidence}


def _finding(mech, status, thr, used, evidence, reasons=()):
    return Finding(mech, status, list(used), evidence, list(reasons), [thr[u]["class"] for u in used])


# --------------------------------------------------------------------------- quantities that are reported, not judged
def straight_deployment_observation(config_analysis: dict, primary: str = "fullres") -> Finding:
    """Exact straight pre-deployment against the references (all configurations). QUANTITATIVE_ONLY: the protocol test never applies before the settle phases, so the straight rod
    is evidence about the geometry, never a criterion."""
    sd = config_analysis["straight_deployment"]
    full, ds = sd[primary], sd["ds"]
    ev = {"reference": primary, "axis_violation_mm": full["axis_violation_mm"], "ds_axis_violation_mm": ds["axis_violation_mm"], "capsule_penetration_mm": full["capsule_penetration_mm"],
          "first_axis_exceedance_arclength_mm": full["first_axis_exceedance_arclength_mm"], "first_contact_arclength_mm": full["first_contact_arclength_mm"], "worst_sample": full["worst_sample"],
          "n_samples_outside_grid": full["n_samples_outside_grid"] + full["n_dense_outside_grid"], "sign_scale": config_analysis.get("sign_scale")}
    return Finding("STRAIGHT_DEPLOYMENT_GEOMETRY", "QUANTITATIVE_ONLY", [], ev, ["NOT_A_CRITERION_BEFORE_SETTLE"], [])


def trace_integrity(trace_analysis: dict | None, trace_meta: dict | None, thr: dict) -> list[str]:
    """Reason codes that make the trace unusable (empty list = usable)."""
    if trace_analysis is None:
        return ["NO_TRACE"]
    bad = []
    rep = (trace_meta or {}).get("instrumented_matches_original_bit_identically")
    if rep is None:
        bad.append("TRACE_REPRODUCTION_NOT_CHECKED")
    elif not rep:
        bad.append("TRACE_NOT_BIT_IDENTICAL")
    c = trace_analysis.get("observer_consistency_max_abs_diff_mm")
    if c is None:
        bad.append("NO_DECISION_EVENTS")
    elif c > thr["observer_consistency_tol_mm"]["value"]:
        bad.append("OBSERVER_INCONSISTENT")
    s = trace_analysis.get("start_vs_analytic_straight_rod_max_mm")
    if s is None or s > thr["start_match_tol_mm"]["value"]:
        bad.append("START_NOT_ANALYTIC_STRAIGHT")
    if not trace_analysis.get("constants_match_trace", False):
        bad.append("CONSTANTS_MISMATCH")
    if not trace_analysis.get("reference_inside_grid", False):
        bad.append("REFERENCE_GRID_INSUFFICIENT")
    return bad


def _interval_problem(ev: dict) -> list[str]:
    if ev.get("n_samples_outside_calibrated_bins"):
        return ["RADIUS_OUTSIDE_CALIBRATED_BINS"]
    if ev.get("fullres_violation_lower_mm") is None or ev.get("fullres_violation_upper_mm") is None:
        return ["REFERENCE_INTERVAL_UNAVAILABLE"]
    return []


def terminating_event(trace_analysis: dict) -> tuple[dict | None, dict | None]:
    """(trigger event, peak event). The decision event is the trigger event if the engine's violation exceeds the protocol limit there (the engine's own termination test), else the
    peak event. Chosen from the trace and the protocol limit only; the episode outcome label is not read."""
    ke = trace_analysis["key_events"]
    return ke.get("trigger"), ke.get("peak")


def _reference_bias_context(trace_analysis: dict, cal_bias: dict | None) -> dict:
    ca_ = trace_analysis["same_point_all_events"].get("contact_active_engine_minus_fullres_mm", {})
    med = ca_.get("median")
    ctx = {"contact_active_engine_minus_reference_median_mm": med, "n_contact_active_samples": ca_.get("n"), "calibrated_reference_error_band_mm": cal_bias}
    if med is not None and cal_bias:
        ctx["median_inside_calibrated_reference_error_band"] = bool(cal_bias["p05_signed_mm"] <= med <= cal_bias["p95_signed_mm"])
    return ctx


def decision_flip_finding(trace_analysis: dict | None, trace_meta: dict | None, thr: dict, cal_bias: dict | None = None) -> Finding:
    m = FLIP
    used = ["lumen_delta_mm", "tol_fullres_reference_by_bin", "observer_consistency_tol_mm", "start_match_tol_mm"]
    bad = trace_integrity(trace_analysis, trace_meta, thr)
    if bad:
        return _finding(m, "NOT_EVALUABLE", thr, used, {}, bad)
    trig, peak = terminating_event(trace_analysis)
    delta = thr["lumen_delta_mm"]["value"]
    if trig is not None and trig["engine_violation_mm"] > delta:
        ev, which = trig, "trigger"
    else:
        ev, which = peak, "peak"
    if ev is None:
        return _finding(m, "NOT_EVALUABLE", thr, used, {}, ["NO_DECISION_EVENTS"])
    evidence = {"event_kind": which, "event": {k: ev[k] for k in ("stage", "decision", "frame", "sub", "n_active")}, "engine_violation_mm": ev["engine_violation_mm"],
                "fullres_violation_mm": ev["fullres_violation_mm"], "fullres_violation_lower_mm": ev.get("fullres_violation_lower_mm"), "fullres_violation_upper_mm": ev.get("fullres_violation_upper_mm"),
                "ds_violation_mm": ev["ds_violation_mm"], "alt_reference_violation_mm": ev.get("alt_violation_mm"), "tolerance_samples_mm": ev.get("tolerance_samples_mm"),
                "worst_sample_engine": ev["worst_sample_engine"], "n_engine_sign_disagreements_beyond_bound": ev.get("n_engine_sign_disagreements_beyond_bound"),
                "sign_disagreements_all_events": trace_analysis.get("sign_disagreements_all_events"), "reference_bias_context": _reference_bias_context(trace_analysis, cal_bias)}
    eng = ev["engine_violation_mm"]
    if eng <= delta:
        # No reference tolerance is needed to decide this branch, so it is evaluated before the
        # bin-coverage / interval-availability check below: an out-of-bin-radius sample whose engine
        # violation already did not exceed the limit is NOT_SUPPORTED, never INDETERMINATE.
        return _finding(m, "NOT_SUPPORTED", thr, used, evidence, ["ENGINE_DID_NOT_EXCEED_LIMIT_AT_EVENT"])
    probs = _interval_problem(ev)
    if probs:
        return _finding(m, "INDETERMINATE", thr, used, evidence, probs)                      # tolerance UNAVAILABLE for a traced sample: no CONFIRMED, no abort
    lo, up = ev["fullres_violation_lower_mm"], ev["fullres_violation_upper_mm"]
    if eng > delta and up <= delta:
        if (ev.get("n_engine_sign_disagreements_beyond_bound") or 0) > 0:
            return _finding(m, "INDETERMINATE", thr, used, evidence, ["ENGINE_QUERY_SIGN_DISAGREEMENT_AT_EVENT"])
        return _finding(m, "CONFIRMED", thr, used, evidence, ["ENGINE_EXCEEDS_LIMIT_WHILE_REFERENCE_UPPER_BOUND_DOES_NOT"])
    if lo > delta:
        return _finding(m, "NOT_SUPPORTED", thr, used, evidence, ["REFERENCE_ALSO_EXCEEDS_LIMIT_BEYOND_REFERENCE_ERROR"])
    return _finding(m, "INDETERMINATE", thr, used, evidence, ["REFERENCE_VIOLATION_INTERVAL_STRADDLES_LIMIT"])


def pre_action_invalidity_finding(config_analysis: dict, trace_analysis: dict | None, trace_meta: dict | None, thr: dict, primary: str = "fullres") -> Finding:
    m = PRE_ACTION
    used = ["lumen_delta_mm", "tol_fullres_reference_by_bin", "observer_consistency_tol_mm", "start_match_tol_mm"]
    straight = straight_deployment_observation(config_analysis, primary).evidence                 # evidence only, never the criterion
    bad = trace_integrity(trace_analysis, trace_meta, thr)
    if bad:
        return _finding(m, "NOT_EVALUABLE", thr, used, {"straight_rod_geometry": straight}, bad)
    ev = trace_analysis["key_events"].get("ready")
    if ev is None:
        return _finding(m, "NOT_EVALUABLE", thr, used, {"straight_rod_geometry": straight}, ["NO_READY_EVENT"])
    delta = thr["lumen_delta_mm"]["value"]
    evidence = {"event": {k: ev[k] for k in ("stage", "decision", "frame", "sub", "n_active")}, "engine_violation_mm": ev["engine_violation_mm"], "fullres_violation_mm": ev["fullres_violation_mm"],
                "fullres_violation_lower_mm": ev.get("fullres_violation_lower_mm"), "fullres_violation_upper_mm": ev.get("fullres_violation_upper_mm"), "ds_violation_mm": ev["ds_violation_mm"],
                "worst_sample_engine": ev["worst_sample_engine"], "straight_rod_geometry": straight}
    probs = _interval_problem(ev)
    if probs:
        return _finding(m, "INDETERMINATE", thr, used, evidence, probs)
    lo, up = ev["fullres_violation_lower_mm"], ev["fullres_violation_upper_mm"]
    if lo > delta:
        return _finding(m, "CONFIRMED", thr, used, evidence, ["READY_STATE_VIOLATES_LIMIT_BEYOND_REFERENCE_ERROR"])
    if up <= delta:
        return _finding(m, "NOT_SUPPORTED", thr, used, evidence, ["READY_STATE_WITHIN_LIMIT_BEYOND_REFERENCE_ERROR"])
    return _finding(m, "INDETERMINATE", thr, used, evidence, ["REFERENCE_VIOLATION_INTERVAL_STRADDLES_LIMIT"])


def secondary_p99_observation(trace_analysis: dict | None, trace_meta: dict | None, thr: dict) -> Finding:
    """SECONDARY_P99_DESCRIPTIVE: the flip / pre-action criteria evaluated with the per-bin p99 reference error instead of the maximum. The threshold is HEURISTIC, the status is
    always QUANTITATIVE_ONLY (a CONFIRMED status would be refused by ``Finding``), and nothing here is read by the statuses, the specificity rule or the labels."""
    name = "SECONDARY_P99_DESCRIPTIVE"
    if "tol_fullres_reference_p99_by_bin" not in thr or trace_analysis is None or trace_integrity(trace_analysis, trace_meta, thr):
        return Finding(name, "QUANTITATIVE_ONLY", [], {"available": False})
    used = ["lumen_delta_mm", "tol_fullres_reference_p99_by_bin"]
    delta = thr["lumen_delta_mm"]["value"]
    trig, peak = terminating_event(trace_analysis)
    ev = trig if (trig is not None and trig["engine_violation_mm"] > delta) else peak
    ready = trace_analysis["key_events"].get("ready")

    def bounds(e):
        return (None, None) if e is None else (e.get("fullres_violation_lower_p99_mm"), e.get("fullres_violation_upper_p99_mm"))
    lo_f, up_f = bounds(ev)
    lo_r, up_r = bounds(ready)
    flip = "UNAVAILABLE" if up_f is None else ("CRITERION_MET" if ev["engine_violation_mm"] > delta and up_f <= delta else "CRITERION_NOT_MET")
    pre = "UNAVAILABLE" if lo_r is None else ("CRITERION_MET" if lo_r > delta else "CRITERION_NOT_MET")
    return Finding(name, "QUANTITATIVE_ONLY", used, {"descriptive_only": True, FLIP: {"p99_criterion": flip, "p99_reference_bounds_mm": [lo_f, up_f]},
                                                     PRE_ACTION: {"p99_criterion": pre, "p99_reference_bounds_mm": [lo_r, up_r]}}, ["NEVER_CHANGES_A_STATUS_OR_LABEL"], [thr[u]["class"] for u in used])


def quantitative_observations(config_analysis: dict, trace_analysis: dict | None, thr: dict, trace_meta: dict | None = None) -> list[dict]:
    """Measured quantities that are reported but not promoted to mechanism claims (no threshold is applied)."""
    obs = []
    mf = config_analysis.get("mesh_fidelity")
    peak = (trace_analysis or {}).get("key_events", {}).get("peak") if trace_analysis else None
    obs.append(Finding("MESH_DISCRETIZATION_OFFSET", "QUANTITATIVE_ONLY", [], {
        "mesh_vertex_signed_distance_to_fullres_boundary_mm": mf["vertex_signed_distance_to_fullres_boundary_mm"] if mf else None,
        "cpu_rebuild_matches_cache": mf["cpu_rebuild_matches_cache"] if mf else None,
        "peak_event_engine_minus_ds_over_samples_mm": peak["engine_minus_ds_over_samples_mm"] if peak else None,
        "peak_event_engine_minus_fullres_over_samples_mm": peak["engine_minus_fullres_over_samples_mm"] if peak else None,
        "peak_event_engine_violation_mm": peak["engine_violation_mm"] if peak else None, "peak_event_fullres_violation_mm": peak["fullres_violation_mm"] if peak else None,
        "peak_event_criterion_flip_fullres_vs_engine": peak["criterion_flip_fullres_vs_engine"] if peak else None}))
    obs.append(Finding("CENTERLINE_CLEARANCE_BELOW_CATHETER_RADIUS", "QUANTITATIVE_ONLY", [], {k: config_analysis["path_clearance"][k] for k in config_analysis["path_clearance"]}))
    obs.append(Finding("AXIS_TO_BODY_GAP", "QUANTITATIVE_ONLY", [], {
        "peak_event_axis_violation_engine_mm": peak["engine_violation_mm"] if peak else None, "peak_event_dense_axis_violation_fullres_mm": peak.get("dense_axis_violation_fullres_mm") if peak else None,
        "peak_event_dense_body_penetration_fullres_mm": peak.get("dense_body_penetration_fullres_mm") if peak else None, "peak_event_axis_to_body_gap_fullres_mm": peak.get("axis_to_body_gap_fullres_mm") if peak else None}))
    obs.append(Finding("ENGINE_QUERY_INTEGRITY", "QUANTITATIVE_ONLY", [], {
        "sign_disagreements_all_events": (trace_analysis or {}).get("sign_disagreements_all_events"), "peak_event_max_abs_engine_minus_mesh_mm": peak.get("max_abs_engine_minus_mesh_mm") if peak else None,
        "samples_without_engine_sdf_hit": (trace_analysis or {}).get("samples_without_engine_sdf_hit"), "query_agreement_tolerance_mm": thr["engine_query_agreement_tol_mm"]["value"]}))
    obs.append(secondary_p99_observation(trace_analysis, trace_meta, thr))
    return [o.to_dict() for o in obs]


# --------------------------------------------------------------------------- per-case label
def case_label(findings: dict[str, Finding]) -> tuple[str, list[str]]:
    confirmed = [m for m, f in findings.items() if f.status == "CONFIRMED"]
    open_ = {m: f for m, f in findings.items() if f.status in ("INDETERMINATE", "NOT_EVALUABLE", "NON_DISCRIMINATING")}
    reasons = [f"{m}:{c}" for m, f in findings.items() for c in f.reason_codes]
    if len(confirmed) >= 2:
        return "MULTIPLE_MECHANISMS_IDENTIFIED", reasons
    if len(confirmed) == 1:
        return f"{confirmed[0]}_CONFIRMED", reasons
    if not open_:
        return "NO_CONTAINMENT_AUDIT_MECHANISM_IDENTIFIED", reasons
    return "INCONCLUSIVE", reasons


def raw_case_findings(case: dict, thr: dict, cal_bias: dict | None = None, primary: str = "fullres") -> dict[str, Finding]:
    """Mechanism findings of one case before the specificity rule is applied."""
    ta, tm = case.get("trace_analysis"), case.get("trace_meta")
    return {FLIP: decision_flip_finding(ta, tm, thr, cal_bias), PRE_ACTION: pre_action_invalidity_finding(case["config_analysis"], ta, tm, thr, primary)}


def apply_specificity(raw: dict[str, dict[str, Finding]], roles: dict[str, str]) -> tuple[dict[str, dict[str, Finding]], dict]:
    """Roles are pre-declared in the configuration. A mechanism whose criterion (raw status CONFIRMED) is met by any positive control is NON_DISCRIMINATING in every case."""
    table, out = {}, {n: dict(f) for n, f in raw.items()}
    for mech in MECHANISMS:
        controls = {n: raw[n][mech].status for n in raw if roles.get(n) == POSITIVE_CONTROL}
        firing = sorted(n for n, st in controls.items() if st == "CONFIRMED")
        table[mech] = {"positive_controls": controls, "criterion_met_by_positive_controls": firing, "non_discriminating": bool(firing),
                       "specificity_established": bool(controls) and not firing and any(st == "NOT_SUPPORTED" for st in controls.values())}
        if firing:
            for n in raw:
                f = raw[n][mech]
                out[n][mech] = Finding(mech, "NON_DISCRIMINATING", f.thresholds_used, dict(f.evidence, raw_status=f.status, positive_controls_meeting_criterion=firing),
                                       ["POSITIVE_CONTROL_MEETS_MECHANISM_CRITERION"], f.classes)
    return out, table


def case_verdict(case: dict, thr: dict, findings: dict[str, Finding] | None = None, cal_bias: dict | None = None, primary: str = "fullres") -> dict:
    """``case``: {name, descriptive_outcome, config_analysis, trace_analysis (optional), trace_meta (optional)}. The descriptive outcome is carried through and never used."""
    findings = findings if findings is not None else raw_case_findings(case, thr, cal_bias, primary)
    label, reasons = case_label(findings)
    return {"name": case["name"], "role": case.get("role"), "descriptive_outcome": case.get("descriptive_outcome"), "label": label, "reason_codes": reasons,
            "findings": {k: v.to_dict() for k, v in findings.items()}, "quantitative_observations": quantitative_observations(case["config_analysis"], case.get("trace_analysis"), thr, case.get("trace_meta"))}


def containment_verdict(cases: list[dict], cal: dict, all_config_analyses: dict | None = None) -> dict:
    thr = build_thresholds(cal)
    primary = primary_reference_name(cal) if "primary_reference" in cal else "fullres"
    bias = cal.get("effects", {}).get("reference_variants_pooled", {}).get(primary)
    raw = {c["name"]: raw_case_findings(c, thr, bias, primary) for c in cases}
    final, spec = apply_specificity(raw, {c["name"]: c.get("role") for c in cases})
    out_cases = [case_verdict(c, thr, final[c["name"]], bias, primary) for c in cases]
    table = None
    if all_config_analyses is not None:
        table = {n: straight_deployment_observation(a, primary).to_dict() for n, a in all_config_analyses.items()}
    return {"scope": f"bounded mechanism set: {', '.join(MECHANISMS)}; other mechanisms are not examined and no global verdict is given",
            "calibration_sha256": cal["calibration_sha256"], "primary_reference": primary, "calibration_engine_vs_own_mesh_disagreements": cal.get("engine_vs_own_mesh_disagreements"),
            "collision_discretization_effect_by_radius_bin": {k: cal.get("effects", {}).get(k) for k in ("ds_vs_analytic_by_bin", "mesh_vs_analytic_by_bin", "engine_vs_analytic_by_bin_without_query_failures")},
            "threshold_provenance": thr, "specificity": spec, "traced_cases": out_cases, "all_configurations_straight_deployment": table}
