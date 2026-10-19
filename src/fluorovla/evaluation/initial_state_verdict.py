"""Explicit verdict logic of the initial-state / registration audit. CPU-only and unit-tested.

Inputs are per-case metric dictionaries produced by tools/audit_initial_state.py. Nothing here is tuned to a success rate.

Case roles:  control        original episode SUCCESS
             early_failure  original LUMEN_VIOLATION at decision <= 1
             late_failure   original LUMEN_VIOLATION after decision 1 (contrast case; not part of the 'systematically invalid after settle' test)
Pre-declared thresholds (before any audit result was seen):
  frame / registration consistency:  SDF sign agreement with the mask deeper than 2 mm (mismatch <= 1 %); CT<->solver and CT<->voxel round trips <= 1e-6 mm; centerline nodes inside the mask >= 99 %; portal inside the mask and phi(portal) < 0;
                                      portal-to-raw-polyline distance <= 1e-3 mm; mesh vertices on the 0.5 iso-surface (median |v - 0.5| <= 0.01)
  post-settle validity:               max axis violation of the settled rod <= LUMEN_DELTA_MM (1.5 mm, the existing terminal tolerance)
"""
from __future__ import annotations

LUMEN_DELTA_MM = 1.5
ROUNDTRIP_TOL_MM = 1e-6
INSIDE_FRACTION = 0.99
PORTAL_DIST_TOL_MM = 1e-3
MESH_ISO_TOL = 0.01
SDF_DEEP_MISMATCH_TOL = 0.01      # fraction of points deeper than 2 mm inside/outside the mask whose SDF sign disagrees with the mask


def case_role(c: dict) -> str:
    if c["original_status"] == "SUCCESS":
        return "control"
    if c["original_status"] == "LUMEN_VIOLATION":
        return "early_failure" if c["original_terminal_decision"] <= 1 else "late_failure"
    return "other"


def frames_and_registration_ok(c: dict) -> tuple[bool, list[str]]:
    bad = []
    r = c["registration"]
    for name, rt in c["roundtrip"].items():
        if rt["ct_solver_ct_error_mm"] > ROUNDTRIP_TOL_MM or rt["ct_voxel_ct_error_mm"] > ROUNDTRIP_TOL_MM:
            bad.append(f"round trip error at {name}")
    if c["frame_consistency"]["round"]["fraction_inside_mask"] < INSIDE_FRACTION:
        bad.append("centerline nodes outside the vessel mask")
    if not r.get("portal_voxel_in_vessel_mask", False):
        bad.append("portal voxel not in vessel mask")
    phi_portal = r.get("portal_phi_neighborhood_median_mm", r.get("portal_phi_mm"))
    if phi_portal is not None and not phi_portal < 0:
        bad.append("portal outside the lumen mesh")
    if r["distance_to_raw_polyline_mm"] > PORTAL_DIST_TOL_MM:
        bad.append("portal not on the raw path polyline")
    if c["mesh_consistency"]["median_abs_mask_value_minus_0p5"] > MESH_ISO_TOL:
        bad.append("mesh vertices not on the mask iso-surface")
    if c.get("sdf_sign", {}).get("deep_mismatch_fraction", 0.0) > SDF_DEEP_MISMATCH_TOL:
        bad.append("SDF sign disagrees with the vessel mask deep inside/outside")
    return (not bad), bad


def post_settle_valid(m: dict) -> bool:
    return m["post_settle_max_violation_mm"] <= LUMEN_DELTA_MM


def initial_state_verdict(cases: list[dict]) -> dict:
    roles = {c["name"]: case_role(c) for c in cases}
    controls = [c for c in cases if roles[c["name"]] == "control"]
    early = [c for c in cases if roles[c["name"]] == "early_failure"]
    reasons: list[str] = []
    frame_ok = {}
    for c in cases:
        ok, bad = frames_and_registration_ok(c)
        frame_ok[c["name"]] = (ok, bad)
        if not ok:
            reasons.append(f"{c['name']}: frame/registration inconsistency: {bad}")
    all_frames_ok = all(v[0] for v in frame_ok.values())
    settle_ok = {c["name"]: post_settle_valid(c["production"]) for c in cases}
    early_invalid = [c["name"] for c in early if not settle_ok[c["name"]]]
    controls_invalid = [c["name"] for c in controls if not settle_ok[c["name"]]]
    variants = sorted({v for c in cases for v in c["counterfactuals"]})
    resolving = []
    for v in variants:
        if v == "production":
            continue
        try:
            fixes_failures = all(post_settle_valid(c["counterfactuals"][v]) for c in early) if early else False
            keeps_controls = all(post_settle_valid(c["counterfactuals"][v]) and c["counterfactuals"][v]["status"] not in ("LUMEN_VIOLATION", "PHYSICS_FAILURE") for c in controls)
        except KeyError:
            continue
        if fixes_failures and keeps_controls:
            resolving.append(v)
    mech = {}
    for c in cases:
        mech[c["name"]] = c["tangent"]["73.5mm"]["rod_end_to_nearest_path_mm"]
    mechanism_separates = bool(early) and bool(controls) and min(mech[c["name"]] for c in early) > max(mech[c["name"]] for c in controls)
    detail = {"roles": roles, "frame_and_registration_ok": {k: v[0] for k, v in frame_ok.items()}, "post_settle_valid": settle_ok, "early_failures_invalid_after_settle": early_invalid,
              "controls_invalid_after_settle": controls_invalid, "resolving_initialization_variants": resolving, "straight_rod_end_deviation_mm": mech,
              "mechanism_separates_failures_from_controls": mechanism_separates}
    if controls_invalid:
        reasons.append(f"positive controls invalid after settle: {controls_invalid}")
        return {"verdict": "INCONCLUSIVE", "reasons": reasons, "detail": detail}
    if resolving and mechanism_separates and all_frames_ok:
        reasons.append(f"initialization-only variants {resolving} make every early failure valid after settle and keep every control valid; the straight-rod end deviation separates failures from controls in the production convention")
        return {"verdict": "BUG_CONFIRMED", "reasons": reasons, "detail": detail}
    if all_frames_ok and not early_invalid:
        reasons.append("coordinate frames and registration are numerically consistent, and no early failure is already invalid after the settle phase: the violations arise during insertion")
        return {"verdict": "PASS_INITIALIZATION_VALID", "reasons": reasons, "detail": detail}
    if early_invalid:
        reasons.append(f"early failures already violate the lumen tolerance after settle ({early_invalid}) but no pre-declared initialization-only variant resolves them with the controls intact "
                       f"(resolving variants: {resolving or 'none'}); next subsystems to audit: collision-mesh discretization / clearance versus catheter radius, axis-sample semantics")
    return {"verdict": "INCONCLUSIVE", "reasons": reasons, "detail": detail}
