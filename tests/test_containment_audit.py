"""CPU tests of the containment-audit measurements on synthetic geometry with known answers (no Warp, no GPU, no patient data)."""
import json
import math
import shutil

import numpy as np
import pytest

from fluorovla.anatomy import containment_audit as ca
from fluorovla.anatomy import synthetic_geometry as sg
from fluorovla.physics.patient_config import PatientConfig
from conftest import ROOT

C = ca.canonical_constants()
STAGES = ["build_scene", "after_settle1", "after_body_enable", "after_settle2", "ready", "settle1_frame", "settle2_frame", "decision_substep"]


def test_canonical_constants_come_from_source_and_are_self_consistent():
    assert C["radius_mm"] == pytest.approx(3.0) and C["l_seg_mm"] == pytest.approx(30.0) and C["num_segments"] == 7
    assert C["target_phi_mm"] == -C["radius_mm"] and len(C["sample_frac"]) == 5 and C["lumen_delta_mm"] == 1.5
    assert C["mask_downsample"] == 2 and C["predeploy_active"] == 3


def test_contact_semantics_drift_is_detected(tmp_path):
    pkg = tmp_path / "pkg"
    for m in ("task_contract.py", "teacher/privileged_teacher.py", "physics/engine.py", "physics/lumen_sdf.py", "physics/patient_episode.py", "anatomy/access_geometry.py", "anatomy/episode_config.py"):
        (pkg / m).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "src/fluorovla" / m, pkg / m)
    assert ca.canonical_constants(pkg)["radius_mm"] == pytest.approx(3.0)
    f = pkg / "physics/lumen_sdf.py"
    f.write_text(f.read_text(encoding="utf-8").replace("TARGET_PHI_M = -CATHETER_RADIUS_M", "TARGET_PHI_M = -0.5 * CATHETER_RADIUS_M"), encoding="utf-8")
    with pytest.raises(ca.CanonicalConstantsDrift):
        ca.canonical_constants(pkg)


def test_full_selftest_passes():
    bad = [r for r in ca.run_selftest(C) if not r["passed"]]
    assert not bad, bad


def test_analytic_tube_distance_matches_brute_force_and_rejects_invalid_radius():
    t = sg.TubeSpec(3.0, 30.0, 30.0, 25.0, 70.0, (20.0, 10.0, 5.0), (3.0, 2.0, 1.0))
    pts = sg.random_points_near_surface(t, 200, 6.0, np.random.default_rng(3))
    cl = t.centerline_world(0.02)
    brute = np.array([np.linalg.norm(cl - p, axis=1).min() for p in pts])
    assert np.abs((3.0 - brute) - t.sd_mm(pts)).max() < 0.05
    with pytest.raises(ValueError):
        sg.TubeSpec(30.0, 10.0, 10.0, 25.0, 70.0)


def test_grid_reference_error_is_bounded_by_voxel_diagonal_for_both_strides():
    tube = sg.TubeSpec(3.5, 40.0, 40.0, 30.0, 90.0, (70.0, 10.0, -40.0))
    for sp in ((1.5, 1.5, 1.5), (3.0, 0.9, 1.2)):
        mask, origin, spv = sg.voxelize(tube, sp, 15.0, (0.3, 0.7, 0.1))
        pts = sg.random_points_near_surface(tube, 800, 6.0, np.random.default_rng(4))
        for stride in (1, C["mask_downsample"]):
            f = ca.SignedDistanceField.from_mask(mask, origin, spv, stride)
            assert f.inside_grid(pts).all()
            assert np.abs(f.phi_mm(pts) + tube.sd_mm(pts)).max() <= np.linalg.norm(sp) * stride


def test_straight_deployment_on_bent_tube_matches_independent_geometry():
    # tube heading +z for 40 mm, then a 90 degree turn of radius 30 mm toward +x; a straight rod along +z leaves the tube at the bend
    tube = sg.TubeSpec(5.0, 40.0, 100.0, 30.0, 90.0)
    af = sg.AnalyticField(tube)
    r = ca.straight_deployment(af, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], C)
    far_z = (C["predeploy_active"] - 1 + C["sample_frac"][-1] - 0.5) * C["l_seg_mm"]           # 75 mm along the rod
    # independent: the rod point (0, 0, z) with z > 40 is at distance |hypot(30, z - 40) - 30| from the arc circle
    expected = abs(math.hypot(30.0, far_z - 40.0) - 30.0) - 5.0
    assert r["samples_phi_mm"][-1] == pytest.approx(expected, abs=1e-6)
    assert r["axis_violation_mm"] == pytest.approx(max(0.0, expected), abs=1e-6)
    assert r["axis_feasible_protocol"] == (max(0.0, expected) <= C["lumen_delta_mm"])
    assert r["first_axis_exceedance_arclength_mm"] > 40.0
    # the grid reference reproduces the analytic metric within its voxel-diagonal error
    mask, origin, spv = sg.voxelize(tube, (1.5, 1.5, 1.5), 15.0)
    ff = ca.SignedDistanceField.from_mask(mask, origin, spv, 1)
    rg = ca.straight_deployment(ff, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], C)
    assert abs(rg["axis_violation_mm"] - r["axis_violation_mm"]) <= 1.5 * math.sqrt(3)


def test_path_clearance_counts_wall_distance_below_catheter_radius():
    tube = sg.TubeSpec(2.5, 300.0)                                            # radius below the catheter radius: the centerline is never clear of the contact criterion
    z = np.arange(0.0, 250.0, 1.5)
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    res = ca.path_clearance(sg.AnalyticField(tube), pts, np.full(len(z), 2.5), [0.0, 0.0, 10.0], C)
    assert res["fraction_wall_distance_below_catheter_radius"] == 1.0 and res["min_wall_distance_mm"] == pytest.approx(2.5, abs=1e-6)
    assert res["max_clearance_deficit_mm"] == pytest.approx(C["radius_mm"] - 2.5, abs=1e-6)


def test_config_analysis_and_mesh_fidelity_on_synthetic_twin(tmp_path):
    tube = sg.TubeSpec(4.0, 60.0, 60.0, 30.0, 70.0, start_mm=(5.0, 6.0, 7.0))
    mask, origin, spv = sg.voxelize(tube, (1.5, 1.5, 1.5), 15.0)
    twin = {"mask": mask, "origin_xyz_mm": origin, "spacing_zyx_mm": spv}
    verts, _ = ca.collision_mesh_ct_mm(mask, origin, spv, C["mask_downsample"])
    geom = ca.PatientGeometry(twin, ca.SignedDistanceField.from_twin(twin, 1), ca.SignedDistanceField.from_twin(twin, C["mask_downsample"]), verts.astype(np.float32).astype(np.float64))
    fid = ca.mesh_fidelity(geom, C)
    assert fid["cpu_rebuild_matches_cache"] and abs(fid["vertex_signed_distance_to_fullres_boundary_mm"]["median"]) <= 1.5 * C["mask_downsample"]
    cl = tube.centerline_world(1.5)
    np.savez(tmp_path / "p.npz", pts=cl, rad=np.full(len(cl), 4.0))
    cfg = PatientConfig(patient_id="synthetic", access_side="x", ct_dir=tmp_path, mesh_cache=tmp_path / "m.npz", mesh_frame="ct", portal_m=(0, 0, 0), tangent_raw=tuple(tube.R[:, 2].tolist()),
                        ct_offset_m=tuple((np.array([5.0, 6.0, 7.0]) / 1000.0).tolist()), sign_scale=1.0, landing_zone_path=tmp_path / "l.json", path_npz_path=tmp_path / "p.npz", s_portal_mm=0.0)
    res = ca.config_analysis(cfg, geom, C)
    assert set(res["straight_deployment"]) == {"fullres", "ds"} and res["straight_deployment"]["fullres"]["n_samples"] == 2 * 5
    assert res["straight_deployment"]["fullres"]["axis_violation_mm"] == 0.0                          # rod starts on the tube axis, the straight part is 60 mm
    assert res["path_clearance"]["fullres"]["n_outside_grid"] == 0


def _trace(field, lateral_by_decision, bias_mm=0.0, off_m=(0.05, 0.1, 0.15)):
    """Synthetic engine trace: a straight rod along +z, laterally displaced by up to ``lateral_by_decision[d]`` mm in x during decision d; engine phi = reference phi + bias."""
    L = C["l_seg_mm"] / 1000.0
    off = np.array(off_m)
    rows = []

    def row(stage, n_active, dec, lat_mm, sub=0):
        q = np.zeros((7, 7))
        q[:, 6] = 1.0
        for i in range(7):
            q[i, :3] = [lat_mm / 1000.0, 0.0, i * L]
        pts = np.array([q[i, :3] + (f - 0.5) * L * np.array([0, 0, 1.0]) for i in range(1, n_active) for f in C["sample_frac"]])
        phi = (field.phi_mm((pts + off) * 1000.0) + bias_mm) / 1000.0
        viol = max(0.0, phi.max()) if dec >= 0 else float("nan")
        return dict(stage=STAGES.index(stage), decision=dec, frame=dec * 6, sub=sub, n_active=n_active, engine_violation_m=viol, body_q=q,
                    sample_body=np.repeat(np.arange(1, n_active), 5).astype(np.int16), sample_fraction=np.tile(C["sample_frac"], n_active - 1), sample_points_m=pts, phi_engine_m=phi,
                    sample_hit=np.ones(len(pts), np.int8))

    rows.append(row("build_scene", 2, -1, 0.0))
    rows.append(row("ready", 3, -1, 0.0))
    for d, lat in enumerate(lateral_by_decision):
        for s in range(30):
            rows.append(row("decision_substep", 3, d, lat * (s + 1) / 30.0, s))
    E, S = len(rows), 10
    a = {"stage_code": np.array([r["stage"] for r in rows], np.int16), "decision": np.array([r["decision"] for r in rows], np.int32), "frame": np.array([r["frame"] for r in rows], np.int32),
         "sub": np.array([r["sub"] for r in rows], np.int16), "n_active": np.array([r["n_active"] for r in rows], np.int16), "engine_violation_m": np.array([r["engine_violation_m"] for r in rows]),
         "contact_count": np.zeros(E, np.int32), "body_q": np.stack([r["body_q"] for r in rows]), "sample_body": np.full((E, S), -1, np.int16), "sample_fraction": np.full((E, S), np.nan),
         "sample_points_m": np.full((E, S, 3), np.nan), "phi_engine_m": np.full((E, S), np.nan), "sample_hit": np.zeros((E, S), np.int8)}
    for i, r in enumerate(rows):
        n = len(r["sample_body"])
        a["sample_body"][i, :n], a["sample_fraction"][i, :n], a["sample_points_m"][i, :n], a["phi_engine_m"][i, :n], a["sample_hit"][i, :n] = (
            r["sample_body"], r["sample_fraction"], r["sample_points_m"], r["phi_engine_m"], r["sample_hit"])
    meta = {"stage_names": STAGES, "ct_offset_m": off.tolist(), "tangent": [0.0, 0.0, 1.0], "l_seg_m": L, "radius_m": C["radius_mm"] / 1000.0, "sample_frac": list(C["sample_frac"]),
            "moving_rod_indices": [1, 2, 3, 4, 5, 6]}
    return a, meta


def _field():
    """Tube along +z whose axis passes through the rod (portal at CT (50, 100, 150) mm), starting 20 mm before the portal."""
    return sg.AnalyticField(sg.TubeSpec(5.0, 400.0, start_mm=(50.0, 100.0, 130.0)))


def test_trace_analysis_reports_same_point_discrepancy_and_events():
    f = _field()
    tr, meta = _trace(f, [0.0, 6.5])                       # decision 1 pushes the rod 6.5 mm sideways: phi = 6.5 - 5 = 1.5 mm
    r = ca.analyze_trace(tr, meta, {"fullres": f, "ds": f}, C)
    assert r["observer_consistency_max_abs_diff_mm"] < 1e-9 and r["reference_inside_grid"] and r["constants_match_trace"]
    assert r["start_vs_analytic_straight_rod_max_mm"] < 1e-9
    assert r["key_events"]["trigger"]["decision"] == 1 and r["key_events"]["peak"]["decision"] == 1
    assert r["key_events"]["peak"]["max_abs_engine_minus_fullres_mm"] < 1e-9 and r["key_events"]["peak"]["engine_violation_mm"] == pytest.approx(1.5, abs=1e-6)
    assert [s["decision"] for s in r["decision_series"]] == [0, 1]


def test_trace_analysis_detects_engine_bias_and_criterion_flip():
    f = _field()
    tr, meta = _trace(f, [0.0, 6.0], bias_mm=0.8)          # reference: 1.0 mm beyond the wall; the engine reports 1.8 mm > 1.5
    r = ca.analyze_trace(tr, meta, {"fullres": f, "ds": f}, C)
    pk = r["key_events"]["peak"]
    assert pk["max_abs_engine_minus_fullres_mm"] == pytest.approx(0.8, abs=1e-6)
    assert pk["engine_violation_mm"] == pytest.approx(1.8, abs=1e-6) and pk["fullres_violation_mm"] == pytest.approx(1.0, abs=1e-6)
    assert pk["criterion_flip_fullres_vs_engine"] is True
    assert pk["dense_body_penetration_fullres_mm"] == pytest.approx(pk["dense_axis_violation_fullres_mm"] + C["radius_mm"], abs=1e-6)       # capsule semantics: phi + R
    assert r["same_point_all_events"]["engine_minus_fullres_mm"]["median"] == pytest.approx(0.8, abs=1e-6)


def test_trace_analysis_flags_inconsistent_observer_and_wrong_start():
    f = _field()
    tr, meta = _trace(f, [0.0, 6.0])
    tr["engine_violation_m"][tr["decision"] >= 0] += 0.004                       # recorded engine value no longer matches the recomputation
    assert ca.analyze_trace(tr, meta, {"fullres": f, "ds": f}, C)["observer_consistency_max_abs_diff_mm"] == pytest.approx(4.0, abs=1e-6)
    tr2, meta2 = _trace(f, [0.0, 6.0])
    tr2["sample_points_m"][0] += 0.001                                           # the engine start is not the analytic straight rod
    assert ca.analyze_trace(tr2, meta2, {"fullres": f, "ds": f}, C)["start_vs_analytic_straight_rod_max_mm"] == pytest.approx(1.0 * math.sqrt(3), abs=1e-6)


def _bin(r):
    return {"id": f"r{r}", "lo_mm": r - 0.5, "hi_mm": r + 0.5, "tube_radius_mm": [r]}


def test_calibration_separates_engine_sign_failures_from_the_continuous_error():
    design = dict(ca.SELFTEST_DESIGN, radius_bins=[_bin(3.0)], rotations_deg=[[0.0, 0.0, 0.0]], voxel_offset_fractions_xyz=[[0.0, 0.0, 0.0]])

    def own_mesh(mask, origin, spacing):
        return ca.MeshReference(*ca.collision_mesh_ct_mm(mask, origin, spacing, C["mask_downsample"]))

    def factory(mask, origin, spacing):                                           # stand-in for the engine path: the CPU mesh query with one sign failure per design
        mesh = own_mesh(mask, origin, spacing)

        def phi(p):
            v = mesh.phi_mm(p)
            v[7] = -v[7] if v[7] != 0 else 1.0
            return v
        return phi

    clean = ca.calibrate(design, [(1.5, 1.5, 1.5)], C, engine_factory=lambda m, o, s: own_mesh(m, o, s).phi_mm)
    assert clean["engine_vs_own_mesh_disagreements"]["detection"] == "own_cpu_mesh"
    cal = ca.calibrate(design, [(1.5, 1.5, 1.5)], C, engine_factory=factory)
    d = cal["engine_vs_own_mesh_disagreements"]
    assert clean["engine_vs_own_mesh_disagreements"]["n_disagreements"] == 0
    assert d["n_disagreements"] == len(design["bends"]) and d["n_sign_differs"] == d["n_disagreements"] and d["rate"] > 0
    assert cal["error_summary"]["engine"]["max_abs_mm"] == pytest.approx(clean["error_summary"]["engine"]["max_abs_mm"], abs=1e-9)      # the failure does not enter the continuous error
    t = cal["tolerances"]
    assert set(t) >= {"fullres_reference_vs_analytic", "engine_query_vs_cpu_mesh_same_triangles"} and not {"engine_vs_analytic", "engine_vs_fullres", "ds_reference_vs_analytic"} & set(t)
    assert t["engine_query_vs_cpu_mesh_same_triangles"]["class"] == "IMPLEMENTATION" and t["fullres_reference_vs_analytic"]["class"] == "SYNTHETIC_VALIDATION"
    assert cal["checks"]["engine_path_calibrated"] and cal["checks"]["engine_query_bounded_by_own_mesh"]
    assert cal["design"] == design and cal["constants"]["radius_mm"] == C["radius_mm"]


def test_trace_analysis_lists_engine_sign_disagreements():
    f = _field()
    tr, meta = _trace(f, [0.0, 3.0])
    k = 40
    tr["phi_engine_m"][k, 2] = -tr["phi_engine_m"][k, 2]                          # one engine sample on the wrong side of the wall
    r = ca.analyze_trace(tr, meta, {"fullres": f, "ds": f}, C)
    assert r["sign_disagreements_all_events"]["n"] == 1
    top = r["sign_disagreements_all_events"]["largest_by_abs_fullres_phi"][0]
    assert top["event_index"] == k and top["engine_phi_mm"] == pytest.approx(-top["fullres_phi_mm"])


def test_engine_sign_failures_are_separated_without_a_cpu_mesh(monkeypatch):
    monkeypatch.setattr(ca, "mesh_reference_or_none", lambda *a, **k: None)          # secondary reference UNAVAILABLE (package not installed)
    design = dict(ca.SELFTEST_DESIGN, radius_bins=[_bin(4.0)], rotations_deg=[[0.0, 0.0, 0.0]], voxel_offset_fractions_xyz=[[0.0, 0.0, 0.0]])
    flip = {"on": False}

    def factory(mask, origin, spacing):
        f = ca.SignedDistanceField.from_mask(mask, origin, spacing, C["mask_downsample"])

        def phi(p):
            v = f.phi_mm(p)
            if flip["on"]:
                k = int(np.argmin(v))                                                  # a point deep inside the lumen reported as outside
                v[k] = -v[k]
            return v
        return phi

    clean = ca.calibrate(design, [(1.5, 1.5, 1.5)], C, engine_factory=factory)
    flip["on"] = True
    cal = ca.calibrate(design, [(1.5, 1.5, 1.5)], C, engine_factory=factory)
    assert cal["engine_vs_own_mesh_disagreements"]["detection"] == "analytic_sign_beyond_mesh_deviation"
    assert cal["engine_vs_own_mesh_disagreements"]["n_sign_differs"] == len(design["bends"]) and clean["engine_vs_own_mesh_disagreements"]["n_disagreements"] == 0
    assert cal["error_summary"]["engine"]["max_abs_mm"] == pytest.approx(clean["error_summary"]["engine"]["max_abs_mm"], abs=1e-9)
    assert not cal["checks"]["engine_query_bounded_by_own_mesh"]


def test_reference_error_is_binned_by_radius_and_the_discretization_effect_is_never_a_tolerance():
    design = dict(ca.SELFTEST_DESIGN, radius_bins=[{"id": "thin", "lo_mm": 2.0, "hi_mm": 3.5, "tube_radius_mm": [2.5]}, {"id": "wide", "lo_mm": 3.5, "hi_mm": 6.0, "tube_radius_mm": [5.0]}],
                  bends=[ca.SELFTEST_DESIGN["bends"][0]], rotations_deg=[[0.0, 0.0, 0.0]], voxel_offset_fractions_xyz=[[0.31, 0.17, 0.43]])
    cal = ca.calibrate(design, [(1.5, 1.5, 1.5)], C)
    t = cal["tolerances"]["fullres_reference_vs_analytic"]
    assert [b["id"] for b in t["bins"]] == ["thin", "wide"] and all(b["value_mm"] > 0 and b["n"] > 0 for b in t["bins"]) and t["value_mm"] == max(b["value_mm"] for b in t["bins"])
    assert t["variant"] == cal["primary_reference"]["name"]
    eff = cal["effects"]
    assert set(eff["ds_vs_analytic_by_bin"]) == {"thin", "wide"} and "ds" not in json.dumps(cal["tolerances"])
    assert eff["ds_vs_analytic_by_bin"]["thin"]["max_abs_mm"] > t["bins"][0]["value_mm"]                 # the collision-resolution effect exceeds the reference error (and is not mixed into it)
    with pytest.raises(ValueError):
        ca.calibrate(dict(design, variant_selection_rule="anything_else"), [(1.5, 1.5, 1.5)], C)
    for bad in ([{"id": "a", "lo_mm": 2.0, "hi_mm": 3.0, "tube_radius_mm": [4.0]}], [{"id": "a", "lo_mm": 2.0, "hi_mm": 3.0, "tube_radius_mm": [2.5]}, {"id": "b", "lo_mm": 3.5, "hi_mm": 5.0, "tube_radius_mm": [4.0]}]):
        with pytest.raises(ValueError):
            ca.design_bins(dict(design, radius_bins=bad))


def test_primary_variant_follows_the_declared_rule_on_the_pooled_synthetic_errors():
    cal = ca.calibrate(dict(ca.SELFTEST_DESIGN, radius_bins=[_bin(3.5)], rotations_deg=[[0.0, 0.0, 0.0]], voxel_offset_fractions_xyz=[[0.31, 0.17, 0.43]]), [(1.5, 1.5, 1.5)], C)
    pooled = cal["primary_reference"]["pooled_abs_median_and_p99_mm"]
    hc_better = pooled["fullres_hc"][0] < pooled["fullres"][0] and pooled["fullres_hc"][1] < pooled["fullres"][1]
    assert cal["primary_reference"]["name"] == ("fullres_hc" if hc_better else "fullres") and cal["primary_reference"]["rule"] == ca.VARIANT_SELECTION_RULE


def test_radius_bin_lookup_and_local_radius():
    bins = [{"lo_mm": 2.0, "hi_mm": 3.0}, {"lo_mm": 3.0, "hi_mm": 5.0}]
    assert ca.radius_bin_lookup(bins)(np.array([1.9, 2.0, 2.99, 3.0, 5.0, 5.01])).tolist() == [-1, 0, 0, 1, 1, -1]
    z = np.linspace(0, 100, 51)
    path = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    assert ca.local_radius_fn(path, np.linspace(3.0, 5.0, 51))(np.array([[0.0, 0.0, 50.0]]))[0] == pytest.approx(4.0, abs=0.05)


def test_trace_analysis_gives_a_rigorous_reference_interval_per_sample_tolerance_and_a_sign_guard():
    f = _field()
    tr, meta = _trace(f, [0.0, 6.0])                                                                  # reference: 1.0 mm beyond the wall
    tol = lambda pts: (np.full(len(pts), 0.3), np.ones(len(pts), bool))                                # noqa: E731
    r = ca.analyze_trace(tr, meta, {"fullres": f, "ds": f, "alt": f}, C, tol_fn=tol, sign_bound_mm=1.0)
    pk = r["key_events"]["peak"]
    assert pk["fullres_violation_lower_mm"] == pytest.approx(0.7, abs=1e-6) and pk["fullres_violation_upper_mm"] == pytest.approx(1.3, abs=1e-6)
    assert pk["n_engine_sign_disagreements_beyond_bound"] == 0 and pk["alt_violation_mm"] == pytest.approx(1.0, abs=1e-6)
    out = lambda pts: (np.full(len(pts), np.nan), np.zeros(len(pts), bool))                           # noqa: E731
    r2 = ca.analyze_trace(tr, meta, {"fullres": f, "ds": f}, C, tol_fn=out)
    assert r2["key_events"]["peak"]["fullres_violation_upper_mm"] is None and r2["key_events"]["peak"]["n_samples_outside_calibrated_bins"] > 0
    # an engine sample on the wrong side of the wall, farther from it than the guard bound, is counted at the event that contains it
    k = r["key_events"]["peak"]["event_index"]
    sb = tr["sample_body"][k]
    j = int(np.flatnonzero(sb >= 0)[0])
    tr["phi_engine_m"][k, j] = -tr["phi_engine_m"][k, j]
    r3 = ca.analyze_trace(tr, meta, {"fullres": f, "ds": f}, C, tol_fn=tol, sign_bound_mm=0.5)
    assert r3["key_events"]["peak"]["n_engine_sign_disagreements_beyond_bound"] == 1


def test_secondary_p99_interval_and_unavailable_tolerance_for_a_radius_outside_the_bins():
    f = _field()
    tr, meta = _trace(f, [0.0, 6.0])                                                                  # reference: 1.0 mm beyond the wall
    mx = lambda pts: (np.full(len(pts), 0.9), np.ones(len(pts), bool))                                 # noqa: E731
    p99 = lambda pts: (np.full(len(pts), 0.4), np.ones(len(pts), bool))                                # noqa: E731
    pk = ca.analyze_trace(tr, meta, {"fullres": f, "ds": f}, C, tol_fn=mx, tol_p99_fn=p99)["key_events"]["peak"]
    assert pk["fullres_violation_upper_mm"] == pytest.approx(1.9, abs=1e-6) and pk["fullres_violation_upper_p99_mm"] == pytest.approx(1.4, abs=1e-6)
    assert pk["fullres_violation_lower_p99_mm"] == pytest.approx(0.6, abs=1e-6)
    none = lambda pts: (np.full(len(pts), np.nan), np.zeros(len(pts), bool))                           # noqa: E731
    pk2 = ca.analyze_trace(tr, meta, {"fullres": f, "ds": f}, C, tol_fn=none, tol_p99_fn=none)["key_events"]["peak"]
    assert pk2["fullres_violation_lower_mm"] is pk2["fullres_violation_upper_mm"] is pk2["fullres_violation_upper_p99_mm"] is None and pk2["n_samples_outside_calibrated_bins"] > 0


def test_calibration_records_the_p99_error_per_bin():
    cal = ca.calibrate(dict(ca.SELFTEST_DESIGN, radius_bins=[_bin(3.0), {"id": "w", "lo_mm": 3.5, "hi_mm": 5.0, "tube_radius_mm": [4.0]}], rotations_deg=[[0.0, 0.0, 0.0]],
                            voxel_offset_fractions_xyz=[[0.31, 0.17, 0.43]]), [(1.5, 1.5, 1.5)], C)
    for b in cal["tolerances"]["fullres_reference_vs_analytic"]["bins"]:
        assert 0 < b["p99_abs_mm"] <= b["value_mm"]
