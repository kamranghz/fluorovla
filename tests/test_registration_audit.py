"""CPU tests of the initial-state / registration audit calculations (synthetic geometry with known answers)."""
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

import fluorovla.task_contract as tc
from fluorovla.anatomy import registration_audit as ra
from fluorovla.physics.patient_config import PatientConfig


def _cfg(portal_ct_mm, tangent, s_portal=0.0):
    t = np.asarray(tangent, float)
    return PatientConfig(patient_id="synthetic", access_side="left", ct_dir=Path("."), mesh_cache=Path("."), mesh_frame="ct", portal_m=(0.0, 0.0, 0.0),
                         tangent_raw=tuple((t / np.linalg.norm(t)).tolist()), ct_offset_m=tuple((np.asarray(portal_ct_mm, float) / 1000.0).tolist()), sign_scale=1.0,
                         landing_zone_path=Path("lz.json"), path_npz_path=Path("p.npz"), s_portal_mm=s_portal)


def _straight_path(n=200, step=1.5):
    z = np.arange(n) * step
    return np.stack([np.zeros(n), np.zeros(n), z], axis=1) + np.array([10.0, 20.0, 0.0]), np.full(n, 4.0)


def _arc_path(radius=100.0, n=200, step=1.5):
    th = np.arange(n) * step / radius
    pts = np.stack([radius * (1 - np.cos(th)), np.zeros(n), radius * np.sin(th)], axis=1) + np.array([10.0, 20.0, 0.0])
    return pts, np.full(n, 4.0)


def test_roundtrip_is_exact_to_float_precision():
    off = np.array([0.061955078125, 0.201955078125, -0.0085])
    assert ra.ct_to_solver_to_ct_error_mm(np.array([[61.9, 201.9, -8.0], [0.0, 1.0, 2.0]]), off) < 1e-9
    twin = {"origin_xyz_mm": np.array([-236.5, -45.5, -61.0]), "spacing_zyx_mm": np.array([1.5, 1.5, 1.5])}
    assert ra.voxel_roundtrip_error_mm(np.array([[10.2, 20.3, 30.4]]), twin) < 1e-9
    p = np.array([[3.0, 4.5, 6.0]])
    idx = ra.world_to_voxel(p, twin["origin_xyz_mm"], twin["spacing_zyx_mm"])
    assert idx.shape == (1, 3) and np.allclose(idx[0], (p[0, ::-1] - twin["origin_xyz_mm"][::-1]) / 1.5)


def test_point_to_polyline_distance_and_arclength():
    poly = np.array([[0.0, 0, 0], [10.0, 0, 0], [10.0, 10.0, 0]])
    d, s = ra.point_to_polyline_distance_mm(np.array([5.0, 3.0, 0.0]), poly)
    assert d == pytest.approx(3.0) and s == pytest.approx(5.0)
    d, s = ra.point_to_polyline_distance_mm(np.array([10.0, 4.0, 0.0]), poly)
    assert d == pytest.approx(0.0) and s == pytest.approx(14.0)


def test_tangent_comparison_straight_path_aligned():
    pts, rad = _straight_path()
    cfg = _cfg(pts[0], [0, 0, 1])
    t = ra.tangent_comparison(cfg, pts, rad)
    for w in ra.TANGENT_WINDOWS_MM:
        assert t[f"{w:g}mm"]["angle_deg"] == pytest.approx(0.0, abs=1e-3)
    assert t["straight_rod_max_lateral_deviation_mm"] < 0.05
    assert t["path_arc_over_chord_73.5mm"] == pytest.approx(1.0, abs=1e-3)


def test_tangent_comparison_detects_misaligned_rod_on_curved_path():
    R = 60.0
    pts, rad = _arc_path(radius=R)
    k = 40                                                            # 60 mm into the path, away from the smoothed ends
    th = k * 1.5 / R
    cfg = _cfg(pts[k], [math.sin(th), 0.0, math.cos(th)])             # straight rod along the exact local tangent, the vessel bends away
    t = ra.tangent_comparison(cfg, pts, rad)
    # chord direction over L on a circle of radius R is rotated by L / (2R); angle grows with the window
    for w in ra.TANGENT_WINDOWS_MM:
        assert t[f"{w:g}mm"]["angle_deg"] == pytest.approx(math.degrees(w / (2 * R)), rel=0.05, abs=0.2)
    assert t["73.5mm"]["rod_end_to_nearest_path_mm"] > 5.0
    assert t["path_arc_over_chord_73.5mm"] > 1.0


def test_portal_registration_on_polyline():
    pts, rad = _straight_path()
    portal = pts[10] + (pts[11] - pts[10]) * 0.3
    cfg = _cfg(portal, [0, 0, 1], s_portal=15.0)
    r = ra.portal_registration(cfg, pts, rad, twin=None, phi_fn=lambda p: np.full(len(p), -0.004))
    assert r["distance_to_raw_polyline_mm"] < 1e-9
    assert r["raw_polyline_arclength_mm"] == pytest.approx(10 * 1.5 + 0.3 * 1.5)
    assert r["portal_phi_mm"] == pytest.approx(-4.0)
    off_cfg = _cfg(portal + np.array([0.0, 2.0, 0.0]), [0, 0, 1])
    assert ra.portal_registration(off_cfg, pts, rad)["distance_to_raw_polyline_mm"] == pytest.approx(2.0)


def test_counterfactual_configs_change_only_portal_and_tangent():
    pts, rad = _arc_path(radius=60.0)
    portal = pts[20] + 0.4 * (pts[21] - pts[20])
    proj = tc.PathProjector(pts, rad)
    cfg = _cfg(portal, [0, 0, 1], s_portal=proj(portal)[0])
    v = ra.counterfactual_configs(cfg, pts, rad)
    assert len(v) == len(ra.PORTAL_CONVENTIONS) * len(ra.TANGENT_CONVENTIONS_MM)
    for name, c in v.items():
        assert c.patient_id == cfg.patient_id and c.mesh_cache == cfg.mesh_cache and c.landing_zone_path == cfg.landing_zone_path and c.sign_scale == cfg.sign_scale
        assert np.linalg.norm(c.tangent) == pytest.approx(1.0)
    base = v["portal=configured|tangent=5mm_chord"]
    assert np.allclose(base.ct_offset, cfg.ct_offset)
    t5, t30 = v["portal=configured|tangent=5mm_chord"].tangent, v["portal=configured|tangent=30mm_chord"].tangent
    assert np.degrees(np.arccos(np.clip(t5 @ t30, -1, 1))) > 1.0                      # chords of different length differ on a curved path
    assert np.allclose(v["portal=path_start_node|tangent=5mm_chord"].ct_offset * 1000.0, pts[0])
    assert dataclasses.replace(cfg).tangent_raw == cfg.tangent_raw


def test_centerline_mask_consistency_on_synthetic_cylinder():
    shape = (40, 40, 60)
    zz, yy, xx = np.indices(shape)
    mask = ((yy - 20) ** 2 + (xx - 20) ** 2) <= 8 ** 2
    spacing = np.array([1.5, 1.5, 1.5])
    origin = np.array([-30.0, -30.0, -20.0])
    z = np.arange(5, 35)
    idx = np.stack([z, np.full(len(z), 20), np.full(len(z), 20)], axis=1).astype(float)
    cl = ra.voxel_to_world(idx, origin, spacing)
    twin = {"mask": mask, "origin_xyz_mm": origin, "spacing_zyx_mm": spacing, "centerline_mm": cl, "centerline_radii_mm": np.full(len(z), 8.0 * 1.5)}
    r = ra.centerline_mask_consistency(twin)
    assert r["round"]["fraction_inside_mask"] == 1.0
    assert r["best_convention_by_radius_agreement"] == "round"
    assert r["round"]["median_abs_edt_minus_twin_radius_mm"] < 1.6


def test_mesh_and_mask_distance_semantics_with_uint8_mask(tmp_path):
    """vessel_mask.npy is stored as uint8; the signed distance of a surface point must be ~0 (a bitwise ~ on uint8 once produced hundreds of mm)."""
    import json
    shape = (30, 30, 30)
    zz, yy, xx = np.indices(shape)
    mask = (((yy - 15) ** 2 + (xx - 15) ** 2) <= 6 ** 2).astype(np.uint8)
    (tmp_path / "metadata.json").write_text(json.dumps({"origin_xyz_mm": [0.0, 0.0, 0.0], "spacing_zyx_mm": [1.5, 1.5, 1.5]}))
    np.save(tmp_path / "vessel_mask.npy", mask)
    np.save(tmp_path / "centerline_points_mm.npy", np.array([[22.5, 22.5, 22.5]]))
    np.save(tmp_path / "centerline_radii_mm.npy", np.array([9.0]))
    from skimage import measure
    v, f, _, _ = measure.marching_cubes(mask[::2, ::2, ::2].astype(np.float32), level=0.5, spacing=(3.0, 3.0, 3.0))
    vct = (np.column_stack([v[:, 2], v[:, 1], v[:, 0]]) / 1000.0).astype(np.float32)
    np.savez(tmp_path / "mesh.npz", vertices_ct_m=vct, faces=f.astype(np.int32))
    cfg = dataclasses.replace(_cfg([22.5, 22.5, 22.5], [0, 0, 1]), ct_dir=tmp_path, mesh_cache=tmp_path / "mesh.npz")
    twin = ra.load_twin(cfg)
    assert twin["mask"].dtype == bool
    m = ra.mesh_mask_consistency(cfg, twin)
    assert m["median_abs_mask_value_minus_0p5"] < 1e-5
    assert abs(m["vertex_signed_distance_to_full_res_mask_boundary_mm"]["median"]) < 2.5


def test_sdf_sign_consistency_agrees_for_an_exact_sdf_and_flags_a_flipped_one():
    shape = (60, 60, 80)
    zz, yy, xx = np.indices(shape)
    R_vox = 6
    mask = (((yy - 30) ** 2 + (xx - 30) ** 2) <= R_vox ** 2)
    spacing = np.array([1.5, 1.5, 1.5])
    origin = np.array([0.0, 0.0, 0.0])
    axis_xy = np.array([30 * 1.5, 30 * 1.5])
    z = np.arange(10, 70) * 1.5
    pts = np.stack([np.full(len(z), axis_xy[0]), np.full(len(z), axis_xy[1]), z], axis=1)
    rad = np.full(len(z), R_vox * 1.5)
    twin = {"mask": mask, "origin_xyz_mm": origin, "spacing_zyx_mm": spacing, "centerline_mm": pts, "centerline_radii_mm": rad}
    cfg = _cfg(pts[5], [0, 0, 1], s_portal=5 * 1.5)

    def exact_phi(p_local_m):                                   # solver-local metres -> phi metres (negative inside the cylinder)
        p_ct = (np.asarray(p_local_m, float) + cfg.ct_offset) * 1000.0
        return (np.hypot(p_ct[:, 0] - axis_xy[0], p_ct[:, 1] - axis_xy[1]) - R_vox * 1.5) / 1000.0

    good = ra.sdf_sign_consistency(cfg, twin, exact_phi, pts, rad, n=4000, reach_mm=8.0)
    assert good["deep_mismatch_fraction"] < 0.01
    flipped = ra.sdf_sign_consistency(cfg, twin, lambda p: -exact_phi(p), pts, rad, n=4000, reach_mm=8.0)
    assert flipped["deep_mismatch_fraction"] > 0.9
    probe = ra.portal_neighborhood_probe(cfg, exact_phi)
    assert len(probe["x"]) == len(probe["offsets_mm"]) and probe["z"][6] == pytest.approx(-9.0, abs=0.2)
