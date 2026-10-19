"""CPU-only geometric checks for the initial-state / registration audit (no Warp, no GPU).

Coordinate conventions (as used by the pipeline):
  CT frame         millimetres, the frame of the twin centerline, the vessel mask and the extended path
  solver-local     metres; solver = (CT - ct_offset), the portal sits at the origin
  voxel            vessel_mask.npy is indexed [z, y, x]; world = origin_xyz + index * spacing (index = voxel centre), metadata.json holds origin and spacing
The checks below measure numbers on actual points; they never only compare shapes or metadata.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

import fluorovla.task_contract as tc
from fluorovla.physics.patient_config import PatientConfig

TANGENT_WINDOWS_MM = (5.0, 15.0, 30.0, 73.5)
ROD_LENGTH_MM = 73.5


# --------------------------------------------------------------------------- twin / mask access
def load_twin(cfg: PatientConfig) -> dict:
    md = json.loads((cfg.ct_dir / "metadata.json").read_text())
    return {"mask": np.load(cfg.ct_dir / "vessel_mask.npy").astype(bool),   # the file is uint8 0/1; boolean semantics are required for ~mask
            "origin_xyz_mm": np.asarray(md["origin_xyz_mm"], float), "spacing_zyx_mm": np.asarray(md["spacing_zyx_mm"], float),
            "centerline_mm": np.load(cfg.ct_dir / "centerline_points_mm.npy").astype(float), "centerline_radii_mm": np.load(cfg.ct_dir / "centerline_radii_mm.npy").astype(float)}


def world_to_voxel(p_xyz_mm: np.ndarray, origin_xyz_mm: np.ndarray, spacing_zyx_mm: np.ndarray) -> np.ndarray:
    """Continuous [z, y, x] voxel coordinates of world points (N, 3) in x, y, z order."""
    p = np.atleast_2d(np.asarray(p_xyz_mm, float))
    return (p[:, ::-1] - origin_xyz_mm[::-1]) / spacing_zyx_mm


def voxel_to_world(idx_zyx: np.ndarray, origin_xyz_mm: np.ndarray, spacing_zyx_mm: np.ndarray) -> np.ndarray:
    idx = np.atleast_2d(np.asarray(idx_zyx, float))
    return (idx * spacing_zyx_mm)[:, ::-1] + origin_xyz_mm


def ct_to_solver_to_ct_error_mm(p_ct_mm: np.ndarray, ct_offset_m: np.ndarray) -> float:
    """Round trip CT-mm -> solver-local (m) -> CT-mm, the exact arithmetic of the pipeline; returns the max abs error in mm."""
    p = np.atleast_2d(np.asarray(p_ct_mm, float))
    local_m = p / 1000.0 - ct_offset_m
    back = (local_m + ct_offset_m) * 1000.0
    return float(np.max(np.abs(back - p)))


def voxel_roundtrip_error_mm(p_ct_mm: np.ndarray, twin: dict) -> float:
    idx = world_to_voxel(p_ct_mm, twin["origin_xyz_mm"], twin["spacing_zyx_mm"])
    back = voxel_to_world(idx, twin["origin_xyz_mm"], twin["spacing_zyx_mm"])
    return float(np.max(np.abs(back - np.atleast_2d(p_ct_mm))))


def signed_distance_mm(twin: dict) -> np.ndarray:
    """Signed distance to the full-resolution mask boundary in mm (positive inside), computed once per twin (the two distance transforms are slow on large volumes)."""
    if "_sd" not in twin:
        from scipy import ndimage as ndi
        twin["_sd"] = (ndi.distance_transform_edt(twin["mask"], sampling=twin["spacing_zyx_mm"])
                       - ndi.distance_transform_edt(~twin["mask"], sampling=twin["spacing_zyx_mm"])).astype(np.float32)
    return twin["_sd"]


def edt_mm(twin: dict) -> np.ndarray:
    """Distance of every mask voxel to the nearest non-mask voxel (0 outside the mask)."""
    return np.maximum(signed_distance_mm(twin), 0.0)


def mask_value_at(twin: dict, p_ct_mm: np.ndarray, convention: str = "round") -> np.ndarray:
    idx = world_to_voxel(p_ct_mm, twin["origin_xyz_mm"], twin["spacing_zyx_mm"])
    if convention == "round":
        ii = np.rint(idx).astype(int)
    elif convention == "floor":
        ii = np.floor(idx).astype(int)
    elif convention == "round_minus_half":
        ii = np.rint(idx - 0.5).astype(int)
    else:
        raise ValueError(convention)
    ii = np.clip(ii, 0, np.array(twin["mask"].shape) - 1)
    return ii


# --------------------------------------------------------------------------- frame consistency
def centerline_mask_consistency(twin: dict) -> dict:
    """Do the twin centerline nodes lie inside the vessel mask, and does the EDT of the mask at the node reproduce the twin radius? Compared for three index conventions
    so that a half-voxel frame offset between centerline and mask would show up as a clear preference for another convention."""
    edt = edt_mm(twin)
    out = {"n_nodes": int(len(twin["centerline_mm"]))}
    for conv in ("round", "floor", "round_minus_half"):
        ii = mask_value_at(twin, twin["centerline_mm"], conv)
        inside = twin["mask"][ii[:, 0], ii[:, 1], ii[:, 2]]
        e = edt[ii[:, 0], ii[:, 1], ii[:, 2]]
        out[conv] = {"fraction_inside_mask": float(np.mean(inside)), "median_abs_edt_minus_twin_radius_mm": float(np.median(np.abs(e - twin["centerline_radii_mm"]))),
                     "p90_abs_edt_minus_twin_radius_mm": float(np.percentile(np.abs(e - twin["centerline_radii_mm"]), 90))}
    out["best_convention_by_radius_agreement"] = min(("round", "floor", "round_minus_half"), key=lambda c: out[c]["median_abs_edt_minus_twin_radius_mm"])
    return out


def mesh_vertices_ct_mm(cfg: PatientConfig) -> np.ndarray:
    d = np.load(cfg.mesh_cache)
    if cfg.mesh_frame == "ct":
        return d["vertices_ct_m"].astype(np.float64) * 1000.0
    return (d["vertices_m"].astype(np.float64) + cfg.ct_offset) * 1000.0


def mesh_mask_consistency(cfg: PatientConfig, twin: dict, downsample: int = 2, n: int = 5000) -> dict:
    """Marching-cubes vertices must sit on the 0.5 iso-surface of the (downsampled) mask when mapped back through the same frame. Also reports how much narrower or wider
    the mesh is than the mask: signed distance statistics of the vertices to the full-resolution mask boundary."""
    from scipy import ndimage as ndi
    v = mesh_vertices_ct_mm(cfg)
    rng = np.random.default_rng(0)
    sel = v[rng.choice(len(v), size=min(n, len(v)), replace=False)]
    mask_ds = twin["mask"][::downsample, ::downsample, ::downsample].astype(np.float32)
    idx = (sel[:, ::-1] - twin["origin_xyz_mm"][::-1]) / (twin["spacing_zyx_mm"] * downsample)
    val = ndi.map_coordinates(mask_ds, idx.T, order=1, mode="nearest")
    # signed distance of the vertices to the full-resolution mask boundary (positive inside the mask)
    sd = signed_distance_mm(twin)
    fi = np.clip(np.rint(world_to_voxel(sel, twin["origin_xyz_mm"], twin["spacing_zyx_mm"])).astype(int), 0, np.array(twin["mask"].shape) - 1)
    sdv = sd[fi[:, 0], fi[:, 1], fi[:, 2]]
    return {"n_vertices_sampled": int(len(sel)), "median_abs_mask_value_minus_0p5": float(np.median(np.abs(val - 0.5))), "p95_abs_mask_value_minus_0p5": float(np.percentile(np.abs(val - 0.5), 95)),
            "vertex_signed_distance_to_full_res_mask_boundary_mm": {"median": float(np.median(sdv)), "p05": float(np.percentile(sdv, 5)), "p95": float(np.percentile(sdv, 95))}}


# --------------------------------------------------------------------------- portal / path registration
def point_to_polyline_distance_mm(p: np.ndarray, poly: np.ndarray) -> tuple[float, float]:
    """(distance, arclength of the foot point along the polyline) of a point to a raw polyline."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-18), 0.0, 1.0)
    foot = a + t[:, None] * ab
    d = np.linalg.norm(p - foot, axis=1)
    k = int(np.argmin(d))
    seg = np.linalg.norm(ab, axis=1)
    return float(d[k]), float(seg[:k].sum() + t[k] * seg[k])


def portal_registration(cfg: PatientConfig, path_pts: np.ndarray, path_rad: np.ndarray, twin: dict | None = None, phi_fn=None) -> dict:
    """Where is the portal relative to the path, the mask and the lumen SDF? ``phi_fn(points_solver_local_m) -> phi in metres`` is optional (GPU)."""
    proj = tc.PathProjector(path_pts, path_rad)
    portal_ct = cfg.ct_offset * 1000.0
    d_raw, s_raw = point_to_polyline_distance_mm(portal_ct, path_pts)
    s_sm, d_sm, r_sm = proj(portal_ct)
    out = {"portal_ct_mm": portal_ct.tolist(), "ct_offset_m": cfg.ct_offset.tolist(), "portal_solver_local_m": cfg.portal.tolist(),
           "distance_to_raw_polyline_mm": d_raw, "raw_polyline_arclength_mm": s_raw, "nearest_raw_node_distance_mm": float(np.min(np.linalg.norm(path_pts - portal_ct, axis=1))),
           "distance_to_smoothed_path_mm": d_sm, "smoothed_path_arclength_mm": s_sm, "local_twin_radius_mm": r_sm, "configured_s_portal_mm": cfg.s_portal_mm}
    if twin is not None:
        ii = mask_value_at(twin, portal_ct, "round")[0]
        out["portal_voxel_in_vessel_mask"] = bool(twin["mask"][tuple(ii)])
        out["portal_edt_mm"] = float(edt_mm(twin)[tuple(ii)])
    if phi_fn is not None:
        out["portal_phi_mm"] = float(phi_fn(np.zeros((1, 3)))[0] * 1000.0)
        # The exact origin can be a degenerate query for the voxel-aligned mesh (several axis-aligned faces at exactly the same distance, sign taken from one of them);
        # the neighbourhood median (+-1 and +-1.5 mm along the solver axes) is the robust portal-inside-lumen test.
        offs = np.array([s * k * np.eye(3)[a] for a in range(3) for k in (1.0, 1.5) for s in (-1.0, 1.0)]) / 1000.0
        near = np.asarray(phi_fn(offs.astype(np.float32)), float) * 1000.0
        out["portal_phi_neighborhood_median_mm"] = float(np.median(near))
        out["portal_exact_query_sign_anomaly"] = bool(out["portal_phi_mm"] > 0 and out["portal_phi_neighborhood_median_mm"] < 0)
    return out


def roundtrip_report(cfg: PatientConfig, path_pts: np.ndarray, path_rad: np.ndarray, twin: dict) -> dict:
    """CT -> solver-local -> CT and CT -> voxel -> CT on the portal, the first path point and points 15 mm and 30 mm downstream of the portal (smoothed path)."""
    proj = tc.PathProjector(path_pts, path_rad)
    portal_ct = cfg.ct_offset * 1000.0
    s0 = max(proj(portal_ct)[0], 0.0)
    pts = {"portal": portal_ct, "first_path_point": np.asarray(path_pts[0], float), "portal_plus_15mm": proj.point_at(s0 + 15.0), "portal_plus_30mm": proj.point_at(s0 + 30.0)}
    out = {}
    for k, p in pts.items():
        out[k] = {"point_ct_mm": np.asarray(p).tolist(), "ct_solver_ct_error_mm": ct_to_solver_to_ct_error_mm(p, cfg.ct_offset), "ct_voxel_ct_error_mm": voxel_roundtrip_error_mm(p, twin),
                  "voxel_in_mask": bool(twin["mask"][tuple(mask_value_at(twin, p, "round")[0])])}
    return out


# --------------------------------------------------------------------------- initial tangent versus the vessel path
def tangent_comparison(cfg: PatientConfig, path_pts: np.ndarray, path_rad: np.ndarray, windows=TANGENT_WINDOWS_MM) -> dict:
    """Angle between the initialization tangent and the smoothed-path chord over each window (dot product, endpoint deviation of the straight rod from the path)."""
    proj = tc.PathProjector(path_pts, path_rad)
    portal_ct = cfg.ct_offset * 1000.0
    s0 = max(proj(portal_ct)[0], 0.0)
    t = cfg.tangent
    res = {"s_portal_on_path_mm": float(s0), "initial_tangent": t.tolist()}
    for L in windows:
        chord = proj.point_at(s0 + L) - proj.point_at(s0)
        c = chord / np.linalg.norm(chord)
        dot = float(np.clip(np.dot(t, c), -1.0, 1.0))
        end_rod = portal_ct + L * t
        res[f"{L:g}mm"] = {"chord_direction": c.tolist(), "dot": dot, "angle_deg": float(math.degrees(math.acos(dot))),
                           "rod_end_to_path_point_mm": float(np.linalg.norm(end_rod - proj.point_at(s0 + L))), "rod_end_to_nearest_path_mm": float(proj(end_rod)[1])}
    # lateral deviation of the whole straight rod from the path
    ss = np.arange(0.0, ROD_LENGTH_MM + 1e-9, 1.5)
    dev = []
    for s in ss:
        s_foot, d, _ = proj(portal_ct + s * t)
        if s_foot > 1e-9:                           # a point whose foot is clamped at the (smoothed) path start has no lateral meaning
            dev.append(d)
    dev = np.array(dev)
    res["straight_rod_max_lateral_deviation_mm"] = float(dev.max())
    res["straight_rod_mean_lateral_deviation_mm"] = float(dev.mean())
    res["path_arc_over_chord_73.5mm"] = float(ROD_LENGTH_MM / np.linalg.norm(proj.point_at(s0 + ROD_LENGTH_MM) - proj.point_at(s0)))
    return res


def path_clearance_profile(cfg: PatientConfig, path_pts: np.ndarray, path_rad: np.ndarray, phi_fn, catheter_radius_mm: float, length_mm: float = 120.0, step_mm: float = 1.0) -> dict:
    """Wall distance (-phi) of the lumen mesh along the smoothed centerline starting at the portal, against the twin radius and the catheter radius. A rod centred on the
    centerline fits the contact criterion (phi <= -radius) only where the wall distance is at least the catheter radius."""
    proj = tc.PathProjector(path_pts, path_rad)
    s0 = max(proj(cfg.ct_offset * 1000.0)[0], 0.0)
    ss = np.arange(s0, min(s0 + length_mm, proj.s[-1]), step_mm)
    pts_ct = np.array([proj.point_at(s) for s in ss])
    phi_mm = phi_fn((pts_ct / 1000.0 - cfg.ct_offset).astype(np.float32)) * 1000.0
    wall = -phi_mm
    rad = np.array([proj(p)[2] for p in pts_ct])
    below = wall < catheter_radius_mm
    return {"s_start_mm": float(s0), "n": int(len(ss)), "min_wall_distance_mm": float(wall.min()), "median_wall_distance_mm": float(np.median(wall)),
            "fraction_wall_distance_below_catheter_radius": float(np.mean(below)), "fraction_centerline_outside_mesh": float(np.mean(wall < 0)),
            "median_twin_radius_mm": float(np.median(rad)), "median_wall_minus_twin_radius_mm": float(np.median(wall - rad)),
            "arrays": {"s_mm": ss.tolist(), "wall_distance_mm": wall.tolist(), "twin_radius_mm": rad.tolist()}}


# --------------------------------------------------------------------------- initialization-only counterfactuals
PORTAL_CONVENTIONS = ("configured", "smoothed_path_point", "path_start_node")
TANGENT_CONVENTIONS_MM = (5.0, 15.0, 30.0, 73.5)


def _config_with(cfg: PatientConfig, portal_ct_mm: np.ndarray, tangent: np.ndarray, path_pts, path_rad) -> PatientConfig:
    """Same patient, mesh, path and landing zone; only the portal position and the insertion direction differ. Tangent numerics follow PatientConfig.from_json for a
    configuration written with the key 'tangent' (normalized once here, once on use)."""
    proj = tc.PathProjector(path_pts, path_rad)
    t = np.asarray(tangent, float)
    raw = t / np.linalg.norm(t)
    s_p = float(max(proj(portal_ct_mm)[0], 0.0))
    return dataclasses.replace(cfg, ct_offset_m=tuple((np.asarray(portal_ct_mm, float) / 1000.0).tolist()), tangent_raw=tuple(raw.tolist()), s_portal_mm=s_p)


def counterfactual_configs(cfg: PatientConfig, path_pts: np.ndarray, path_rad: np.ndarray) -> dict[str, PatientConfig]:
    """Initialization-only variants (no physics constant is touched): portal convention x tangent chord window. 'configured' + 5 mm is the production configuration."""
    proj = tc.PathProjector(path_pts, path_rad)
    s_cfg = cfg.s_portal_mm if cfg.s_portal_mm is not None else proj(cfg.ct_offset * 1000.0)[0]
    portals = {"configured": cfg.ct_offset * 1000.0, "smoothed_path_point": proj.point_at(max(s_cfg, 0.0)), "path_start_node": np.asarray(path_pts[0], float)}
    out = {}
    for pname, p in portals.items():
        s0 = max(proj(p)[0], 0.0)
        for L in TANGENT_CONVENTIONS_MM:
            t = proj.point_at(s0 + L) - proj.point_at(s0)
            out[f"portal={pname}|tangent={L:g}mm_chord"] = _config_with(cfg, p, t, path_pts, path_rad)
    return out


# --------------------------------------------------------------------------- start-geometry features (CPU only; computable for every configured episode)
def start_geometry_features(cfg: PatientConfig, path_pts: np.ndarray, path_rad: np.ndarray) -> dict:
    """Geometry of the start condition that the straight-rod initialization depends on: curvature of the path over the rod length, tangent-versus-chord angles and the end
    deviation of the straight 73.5-mm rod from the path. Uses only the configuration and the extended path."""
    t = tangent_comparison(cfg, path_pts, path_rad)
    proj = tc.PathProjector(path_pts, path_rad)
    s0 = t["s_portal_on_path_mm"]
    ss = np.arange(s0, min(s0 + ROD_LENGTH_MM, proj.s[-1]), 1.0)
    p = np.array([proj.point_at(s) for s in ss])
    turn = np.degrees(np.arccos(np.clip((np.diff(p, axis=0)[0] / np.linalg.norm(np.diff(p, axis=0)[0])) @ (np.diff(p, axis=0)[-1] / np.linalg.norm(np.diff(p, axis=0)[-1])), -1, 1)))
    return {"s_portal_mm": s0, "angle_5mm_deg": t["5mm"]["angle_deg"], "angle_15mm_deg": t["15mm"]["angle_deg"], "angle_30mm_deg": t["30mm"]["angle_deg"], "angle_73.5mm_deg": t["73.5mm"]["angle_deg"],
            "rod_end_deviation_mm": t["73.5mm"]["rod_end_to_nearest_path_mm"], "straight_rod_max_lateral_deviation_mm": t["straight_rod_max_lateral_deviation_mm"],
            "arc_over_chord_73.5mm": t["path_arc_over_chord_73.5mm"], "total_turning_over_rod_deg": float(turn), "median_twin_radius_over_rod_mm": float(np.median([proj(q)[2] for q in p]))}


# --------------------------------------------------------------------------- SDF sign / depth consistency with the vessel mask
def sdf_sign_consistency(cfg: PatientConfig, twin: dict, phi_fn, path_pts: np.ndarray, path_rad: np.ndarray, n: int = 20000, reach_mm: float = 10.0, seed: int = 0,
                         deep_mm: float = 2.0) -> dict:
    """Does the lumen SDF (phi < 0 inside) agree with the vessel mask it was built from? Random points within ``reach_mm`` of the first 120 mm of the path are compared:
    sign of phi against mask membership, and phi against the negative signed depth to the full-resolution mask boundary. Mismatches deeper than ``deep_mm`` inside or outside
    the mask cannot be explained by the 3-mm mesh discretization and indicate a sign / frame problem."""
    from scipy import ndimage as ndi
    proj = tc.PathProjector(path_pts, path_rad)
    s0 = max(proj(cfg.ct_offset * 1000.0)[0], 0.0)
    rng = np.random.default_rng(seed)
    ss = rng.uniform(s0, min(s0 + 120.0, proj.s[-1]), n)
    base = np.array([proj.point_at(s) for s in ss])
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = base + dirs * (reach_mm * rng.uniform(0, 1, n) ** (1 / 3))[:, None]
    sd = signed_distance_mm(twin)
    ii = mask_value_at(twin, pts, "round")
    depth = sd[ii[:, 0], ii[:, 1], ii[:, 2]]
    in_mask = twin["mask"][ii[:, 0], ii[:, 1], ii[:, 2]]
    phi = np.asarray(phi_fn((pts / 1000.0 - cfg.ct_offset).astype(np.float32)), float) * 1000.0
    inside_sdf = phi < 0
    deep_in = in_mask & (depth >= deep_mm)
    deep_out = (~in_mask) & (depth <= -deep_mm)
    diff = phi + depth
    return {"n": int(n), "sign_mismatch_fraction": float(np.mean(inside_sdf != in_mask)), "n_deep_inside_mask": int(deep_in.sum()), "n_deep_outside_mask": int(deep_out.sum()),
            "deep_inside_mask_but_phi_positive_fraction": float(np.mean(phi[deep_in] > 0)) if deep_in.any() else None,
            "deep_outside_mask_but_phi_negative_fraction": float(np.mean(phi[deep_out] < 0)) if deep_out.any() else None,
            "deep_mismatch_fraction": float((np.sum(phi[deep_in] > 0) + np.sum(phi[deep_out] < 0)) / max(1, deep_in.sum() + deep_out.sum())),
            "phi_plus_depth_mm": {"median": float(np.median(diff)), "p05": float(np.percentile(diff, 5)), "p95": float(np.percentile(diff, 95))},
            "mesh_narrower_than_mask_median_mm": float(np.median(diff[in_mask & (depth > 0.5)])) if (in_mask & (depth > 0.5)).any() else None}


def portal_neighborhood_probe(cfg: PatientConfig, phi_fn, half_extent_mm: float = 6.0, step_mm: float = 1.0) -> dict:
    """phi (mm) along the three solver axes and along the insertion tangent through the portal (solver-local origin)."""
    ks = np.arange(-half_extent_mm, half_extent_mm + 1e-9, step_mm)
    t = cfg.tangent
    out = {"offsets_mm": ks.tolist()}
    for name, d in (("x", np.array([1.0, 0, 0])), ("y", np.array([0, 1.0, 0])), ("z", np.array([0, 0, 1.0])), ("tangent", t)):
        p = (ks[:, None] * d[None, :]) / 1000.0
        out[name] = (np.asarray(phi_fn(p.astype(np.float32)), float) * 1000.0).tolist()
    return out
