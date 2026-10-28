"""CPU geometry / measurement analysis of the lumen-containment audit (numpy, scipy, scikit-image; trimesh is imported lazily for the optional secondary mesh reference).

Questions this module can answer (it never runs the physics and never imports Warp or Newton):
  * what is the signed distance of any CT-frame point to the vessel wall according to independent references
      - ``fullres``  spacing-aware signed EDT of the full-resolution vessel mask, trilinear interpolation (PRIMARY reference),
      - ``ds``       the same construction on the mask subsampled by the collision-mesh stride (what the 3-mm collision mesh is built from),
      - ``mesh``     signed distance to a collision mesh rebuilt on the CPU from the mask (SECONDARY, optional; UNAVAILABLE when not validated);
  * is the straight pre-deployment of the rod compatible with the anatomy (exact geometry, no physics);
  * how do the engine's recorded SDF values (read-only probe trace) compare with the references at the same physical points;
  * how large is the gap between the axis-sample violation test and the physical-body (capsule) penetration.

Conventions (as in ``registration_audit``): CT frame in millimetres, ``vessel_mask`` indexed [z, y, x], world = origin_xyz + index * spacing, phi < 0 inside the lumen.

Semantics read from the canonical source (``canonical_constants``; the values are never copied into this module):
  * contact activation (``physics/lumen_sdf.generate_lumen_contacts``): a rod axis sample generates a contact when ``phi > TARGET_PHI_M`` with ``TARGET_PHI_M = -CATHETER_RADIUS_M``
    and ``CATHETER_RADIUS_M = float(RADIUS)``; so the radius-R sphere centred on the sample overlaps the wall exactly when ``phi - TARGET_PHI_M > 0``. That excess is the body
    (capsule-surface) penetration of a sample: ``penetration = max(0, phi - TARGET_PHI)``  (derivation in docs/provenance/stage3b_containment_audit.md).
  * lumen-violation test (``physics/patient_episode.measure_outside``): ``max(0, max phi over the axis samples of the non-root bodies)`` compared with ``LUMEN_DELTA_MM`` (strictly greater
    terminates the episode). It is an axis measure: it ignores the catheter radius.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fluorovla.task_contract as tc
from fluorovla.anatomy import registration_audit as ra
from fluorovla.anatomy import synthetic_geometry as sg
from fluorovla.common.constants_snapshot import snapshot

FLOAT_SLACK_M = 1e-9           # IMPLEMENTATION: recomputation of the same float64 expression from the recorded float32 state
ENGINE_MESH_AGREEMENT_TOL_MM = 1e-3   # IMPLEMENTATION: engine (float32 Warp query) and CPU mesh query use the same triangles; larger differences are disagreements, not discretization


class CanonicalConstantsDrift(RuntimeError):
    """The canonical source no longer has the form this audit was derived for; the derivation must be redone before any result is used."""


# --------------------------------------------------------------------------- canonical constants (read from source with ast; no Warp import)
def canonical_constants(package_dir: str | Path | None = None) -> dict:
    snap = snapshot(package_dir)
    eng, lum, pe, task = snap["physics/engine.py"], snap["physics/lumen_sdf.py"], snap["physics/patient_episode.py"], snap["task_contract.py"]
    if lum["CATHETER_RADIUS_M"] != "expr: float(RADIUS)" or lum["TARGET_PHI_M"] != "expr: -CATHETER_RADIUS_M":
        raise CanonicalConstantsDrift("lumen_sdf contact threshold is no longer TARGET_PHI_M = -float(RADIUS)")
    if list(lum["SAMPLE_FRAC"]) != list(pe["SAMPLE_FRAC"]) or lum["N_SAMPLES"] != len(lum["SAMPLE_FRAC"]):
        raise CanonicalConstantsDrift("axis-sample fractions of lumen_sdf and patient_episode differ")
    radius_mm = float(eng["RADIUS"]) * 1000.0
    return {"radius_mm": radius_mm, "l_seg_mm": float(eng["L_SEG"]) * 1000.0, "num_segments": int(eng["NUM_SEGMENTS"]), "sample_frac": tuple(float(f) for f in lum["SAMPLE_FRAC"]),
            "mask_downsample": int(lum["MASK_DOWNSAMPLE"]), "target_phi_mm": -radius_mm, "sdf_query_radius_mm": float(lum["MAX_DIST_M"]) * 1000.0,
            "predeploy_active": int(pe["PREDEPLOY_ACTIVE"]), "lumen_delta_mm": float(task["LUMEN_DELTA_MM"])}


def live_constants_match(constants: dict) -> dict:
    """Compare the source-derived constants with the values the running physics modules actually use (needs the physics stack)."""
    from fluorovla.physics import engine as eng
    from fluorovla.physics import lumen_sdf as lum
    from fluorovla.physics import patient_episode as pe
    live = {"radius_mm": eng.RADIUS * 1000.0, "l_seg_mm": eng.L_SEG * 1000.0, "num_segments": eng.NUM_SEGMENTS, "sample_frac": tuple(lum.SAMPLE_FRAC), "mask_downsample": lum.MASK_DOWNSAMPLE,
            "target_phi_mm": lum.TARGET_PHI_M * 1000.0, "sdf_query_radius_mm": lum.MAX_DIST_M * 1000.0, "predeploy_active": pe.PREDEPLOY_ACTIVE, "lumen_delta_mm": tc.LUMEN_DELTA_MM}
    return {k: bool(np.allclose(np.asarray(live[k], float), np.asarray(constants[k], float), rtol=0, atol=1e-9)) for k in live}


# --------------------------------------------------------------------------- reference signed-distance fields
def _signed_edt_mm(mask: np.ndarray, spacing_zyx_mm: np.ndarray, half_voxel: bool = False) -> np.ndarray:
    """Signed distance (positive inside) between voxel centres. ``half_voxel``: face-centred variant. The boundary lies on the face half a step from the nearest centre of the other
    class, so each side is shortened by half the finest spacing (the first shell of either class is at least that far): the inside value next to the wall becomes +h/2 and the
    outside value -h/2, and trilinear interpolation then has unit slope across the wall instead of the slope 2 of the centre-to-centre distance."""
    from scipy import ndimage as ndi
    mask = np.asarray(mask, bool)
    if not mask.any() or mask.all():
        raise ValueError("the mask must contain both inside and outside voxels")
    d_in = ndi.distance_transform_edt(mask, sampling=spacing_zyx_mm)
    d_out = ndi.distance_transform_edt(~mask, sampling=spacing_zyx_mm)
    if half_voxel:
        h2 = 0.5 * float(np.min(spacing_zyx_mm))
        return np.where(mask, d_in - h2, -(d_out - h2)).astype(np.float32)
    return (d_in - d_out).astype(np.float32)


class SignedDistanceField:
    """Signed distance (positive inside) of a boolean mask on its grid, queried in the CT frame. ``phi_mm`` follows the simulator convention (phi < 0 inside)."""

    def __init__(self, sd_mm: np.ndarray, origin_xyz_mm, spacing_zyx_mm, stride: int, name: str):
        self.sd, self.origin, self.spacing, self.stride, self.name = sd_mm, np.asarray(origin_xyz_mm, float), np.asarray(spacing_zyx_mm, float), int(stride), name

    @classmethod
    def from_mask(cls, mask, origin_xyz_mm, spacing_zyx_mm, stride: int = 1, name: str | None = None, sd: np.ndarray | None = None, half_voxel: bool = False) -> "SignedDistanceField":
        sp = np.asarray(spacing_zyx_mm, float) * stride
        if sd is None:
            sd = _signed_edt_mm(np.asarray(mask, bool)[::stride, ::stride, ::stride], sp, half_voxel)
        return cls(sd, origin_xyz_mm, sp, stride, name or (("fullres_hc" if half_voxel else "fullres") if stride == 1 else f"ds{stride}"))

    @classmethod
    def from_twin(cls, twin: dict, stride: int = 1, half_voxel: bool = False) -> "SignedDistanceField":
        sd = ra.signed_distance_mm(twin) if (stride == 1 and not half_voxel) else None          # reuses the validated (cached) full-resolution computation
        return cls.from_mask(twin["mask"], twin["origin_xyz_mm"], twin["spacing_zyx_mm"], stride, sd=sd, half_voxel=half_voxel)

    def _index(self, pts_ct_mm) -> np.ndarray:
        return ra.world_to_voxel(np.atleast_2d(np.asarray(pts_ct_mm, float)), self.origin, self.spacing)

    def inside_grid(self, pts_ct_mm) -> np.ndarray:
        idx = self._index(pts_ct_mm)
        return np.all((idx >= 0) & (idx <= np.array(self.sd.shape) - 1), axis=1)

    def phi_mm(self, pts_ct_mm) -> np.ndarray:
        from scipy import ndimage as ndi
        idx = self._index(pts_ct_mm)
        return -ndi.map_coordinates(self.sd, idx.T, order=1, mode="nearest").astype(np.float64)


def collision_mesh_ct_mm(mask: np.ndarray, origin_xyz_mm, spacing_zyx_mm, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent CPU rebuild of the collision-mesh construction (marching cubes level 0.5 on the stride-subsampled mask). Vertices in CT millimetres (float64)."""
    from skimage import measure
    sp = np.asarray(spacing_zyx_mm, float) * stride
    m = np.asarray(mask)[::stride, ::stride, ::stride].astype(np.float32)
    v_zyx, faces, _, _ = measure.marching_cubes(m, level=0.5, spacing=tuple(sp))
    return np.column_stack([v_zyx[:, 2], v_zyx[:, 1], v_zyx[:, 0]]) + np.asarray(origin_xyz_mm, float), faces.astype(np.int32)


class MeshReference:
    """Secondary signed-distance reference: trimesh signed distance to a closed triangle mesh (positive inside in trimesh; converted to phi < 0 inside)."""
    name = "mesh"

    def __init__(self, verts_mm: np.ndarray, faces: np.ndarray):
        import trimesh                                                     # lazy and optional: it was NOT in the previous cloud pip freeze; no package is ever installed for it, so the reference is then UNAVAILABLE
        self.mesh = trimesh.Trimesh(vertices=np.asarray(verts_mm, float), faces=np.asarray(faces), process=False)
        self._sd = trimesh.proximity.signed_distance

    def phi_mm(self, pts_ct_mm) -> np.ndarray:
        return -np.asarray(self._sd(self.mesh, np.atleast_2d(np.asarray(pts_ct_mm, float))), float)

    def inside_grid(self, pts_ct_mm) -> np.ndarray:
        return np.ones(len(np.atleast_2d(pts_ct_mm)), bool)


def mesh_reference_or_none(verts_mm, faces) -> "MeshReference | None":
    try:
        return MeshReference(verts_mm, faces)
    except Exception:  # noqa: BLE001  (missing package or unusable mesh: the reference is UNAVAILABLE, never guessed)
        return None


def validate_mesh_reference_on_sphere(radius_mm: float = 10.0, subdivisions: int = 3) -> dict:
    """Analytic validation of the secondary mesh reference on an icosphere: the signed distance may deviate from the exact sphere by at most the facet sag, and the sign must be
    right wherever the exact distance exceeds the sag. Returns {'available': bool, ...}; any failure means UNAVAILABLE."""
    try:
        import trimesh
        m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius_mm)
        ref = MeshReference(m.vertices, m.faces)
        rng = np.random.default_rng(0)
        d = rng.normal(size=(400, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts = d * rng.uniform(0.5 * radius_mm, 1.5 * radius_mm, 400)[:, None]
        exact = np.linalg.norm(pts, axis=1) - radius_mm                    # phi of the exact sphere (negative inside)
        sag = radius_mm - float(np.abs(np.einsum("ij,ij->i", m.triangles_center, m.face_normals)).min())
        err = np.abs(ref.phi_mm(pts) - exact)
        sign_bad = int(np.sum((np.abs(exact) > 2 * sag) & (np.sign(ref.phi_mm(pts)) != np.sign(exact))))
        ok = bool(err.max() <= 2 * sag + 1e-9 and sign_bad == 0)
        return {"available": ok, "max_abs_error_mm": float(err.max()), "facet_sag_mm": sag, "sign_errors_outside_sag_band": sign_bad}
    except Exception as ex:  # noqa: BLE001
        return {"available": False, "reason": f"{type(ex).__name__}: {ex}"}


# --------------------------------------------------------------------------- patient geometry (real data; CPU reads only)
@dataclass
class PatientGeometry:
    twin: dict
    fullres: SignedDistanceField
    ds: SignedDistanceField
    mesh_vertices_ct_mm: np.ndarray            # the vertices of the collision mesh the engine actually loads (cache)
    fullres_hc: "SignedDistanceField | None" = None      # half-voxel face-centred variant of the full-resolution reference

    def primary(self, name: str) -> "SignedDistanceField":
        """The full-resolution reference variant selected by the locked calibration (``fullres`` or ``fullres_hc``)."""
        if name == "fullres_hc":
            if self.fullres_hc is None:
                raise ValueError("the calibration selected fullres_hc but the geometry was loaded without it")
            return self.fullres_hc
        if name != "fullres":
            raise ValueError(f"unknown primary reference {name!r}")
        return self.fullres


def load_patient_geometry(cfg, constants: dict) -> PatientGeometry:
    twin = ra.load_twin(cfg)
    return PatientGeometry(twin, SignedDistanceField.from_twin(twin, 1), SignedDistanceField.from_twin(twin, constants["mask_downsample"]), ra.mesh_vertices_ct_mm(cfg),
                           SignedDistanceField.from_twin(twin, 1, half_voxel=True))


def mesh_fidelity(geom: PatientGeometry, constants: dict) -> dict:
    """Engine collision mesh versus the full-resolution mask: signed distance of every cached mesh vertex to the mask boundary (positive = vertex inside the mask), and the
    maximum difference between the cached vertices and an independent CPU rebuild (float32 storage bounds it)."""
    v = geom.mesh_vertices_ct_mm
    sd = -geom.fullres.phi_mm(v)
    tw = geom.twin
    rv, _ = collision_mesh_ct_mm(tw["mask"], tw["origin_xyz_mm"], tw["spacing_zyx_mm"], constants["mask_downsample"])
    same_topology = rv.shape == v.shape
    f32_bound = float(np.finfo(np.float32).eps * max(1.0, float(np.abs(v).max())) * 4.0)                      # IMPLEMENTATION: float32 rounding of the stored coordinates (+ offset subtraction)
    dev = float(np.abs(rv - v).max()) if same_topology else None
    return {"n_vertices": int(len(v)), "vertex_signed_distance_to_fullres_boundary_mm": {"median": float(np.median(sd)), "p05": float(np.percentile(sd, 5)), "p95": float(np.percentile(sd, 95)),
                                                                                      "min": float(sd.min()), "max": float(sd.max())},
            "cpu_rebuild_same_vertex_count": bool(same_topology), "cpu_rebuild_max_vertex_difference_mm": dev, "float32_representation_bound_mm": f32_bound,
            "cpu_rebuild_matches_cache": bool(same_topology and dev <= f32_bound)}


# --------------------------------------------------------------------------- straight deployment (exact geometry, no physics)
def straight_rod_samples(portal_ct_mm, tangent, constants: dict, n_active: int | None = None) -> list[tuple[int, float, np.ndarray]]:
    """(rod index, fraction, CT point) of the canonical axis samples of the straight rod: the root body's centre sits on the portal, body k centred at portal + k*L_SEG*tangent,
    samples at centre + (fraction - 0.5)*L_SEG*tangent for every non-root active body (the order of ``patient_episode.axis_sample_points``)."""
    n_active = constants["predeploy_active"] if n_active is None else n_active
    p0, t = np.asarray(portal_ct_mm, float), np.asarray(tangent, float)
    t = t / np.linalg.norm(t)
    L = constants["l_seg_mm"]
    return [(k, f, p0 + (k + f - 0.5) * L * t) for k in range(1, n_active) for f in constants["sample_frac"]]


def straight_deployment(field, portal_ct_mm, tangent, constants: dict, n_active: int | None = None, step_mm: float = 0.25) -> dict:
    """Exact feasibility of the straight pre-deployment against one reference field: the canonical samples and the continuous axis."""
    n_active = constants["predeploy_active"] if n_active is None else n_active
    L, tgt, delta = constants["l_seg_mm"], constants["target_phi_mm"], constants["lumen_delta_mm"]
    samp = straight_rod_samples(portal_ct_mm, tangent, constants, n_active)
    pts = np.array([p for _k, _f, p in samp])
    phi = field.phi_mm(pts)
    t = np.asarray(tangent, float) / np.linalg.norm(tangent)
    s = np.arange(0.5 * L, (n_active - 0.5) * L + 1e-9, step_mm)               # moving bodies only (the root body is excluded from the canonical test)
    dense_pts = np.asarray(portal_ct_mm, float) + s[:, None] * t
    dphi = field.phi_mm(dense_pts)
    scan_s = np.arange(0.5 * L, constants["num_segments"] * L, step_mm)
    sphi = field.phi_mm(np.asarray(portal_ct_mm, float) + scan_s[:, None] * t)
    first = lambda mask: (float(scan_s[int(np.argmax(mask))]) if mask.any() else None)       # noqa: E731
    k = int(np.argmax(phi))
    return {"reference": field.name, "n_active": int(n_active), "n_samples": int(len(phi)), "samples_phi_mm": phi.tolist(),
            "axis_violation_mm": float(max(0.0, phi.max())), "capsule_penetration_mm": float(max(0.0, (phi - tgt).max())), "n_samples_beyond_wall": int(np.sum(phi > 0)),
            "n_samples_in_contact": int(np.sum(phi > tgt)), "worst_sample": {"rod_index": samp[k][0], "fraction": samp[k][1], "point_ct_mm": samp[k][2].tolist(), "phi_mm": float(phi[k])},
            "dense_axis_violation_mm": float(max(0.0, dphi.max())), "dense_capsule_penetration_mm": float(max(0.0, (dphi - tgt).max())),
            "axis_feasible_protocol": bool(max(0.0, phi.max()) <= delta), "contact_free": bool(phi.max() <= tgt),
            "first_axis_exceedance_arclength_mm": first(sphi > delta), "first_contact_arclength_mm": first(sphi > tgt), "first_beyond_wall_arclength_mm": first(sphi > 0),
            "n_samples_outside_grid": int(np.sum(~field.inside_grid(pts))), "n_dense_outside_grid": int(np.sum(~field.inside_grid(dense_pts)))}


def audited_path_points(path_pts_mm, path_rad_mm, portal_ct_mm, constants: dict, length_mm: float | None = None, step_mm: float = 1.0):
    """(start arclength, points, twin radii) of the centerline stretch starting at the portal that the audit examines (anatomy only: no outcome, no physics)."""
    length_mm = constants["num_segments"] * constants["l_seg_mm"] if length_mm is None else length_mm
    proj = tc.PathProjector(path_pts_mm, path_rad_mm)
    s0 = max(proj(np.asarray(portal_ct_mm, float))[0], 0.0)
    ss = np.arange(s0, min(s0 + length_mm, proj.s[-1]), step_mm)
    pts = np.array([proj.point_at(x) for x in ss])
    return s0, pts, np.array([proj(p)[2] for p in pts])


def local_radius_fn(path_pts_mm, path_rad_mm):
    """points (mm) -> twin lumen radius (mm) at their projection on the centerline."""
    proj = tc.PathProjector(path_pts_mm, path_rad_mm)
    return lambda pts: np.array([proj(p)[2] for p in np.atleast_2d(np.asarray(pts, float))], float)


def radius_bin_lookup(bins: list[dict]):
    """bins: [{'lo_mm','hi_mm', ...}] -> function radius array -> bin index (-1 outside every bin); lo inclusive, hi exclusive (the last bin includes its upper edge)."""
    def f(r):
        r = np.asarray(r, float)
        idx = np.full(r.shape, -1, int)
        for i, b in enumerate(bins):
            last = i == len(bins) - 1
            idx[(r >= b["lo_mm"]) & ((r <= b["hi_mm"]) if last else (r < b["hi_mm"])) & (idx < 0)] = i
        return idx
    return f


def path_clearance(field, path_pts_mm, path_rad_mm, portal_ct_mm, constants: dict, length_mm: float | None = None, step_mm: float = 1.0) -> dict:
    """Wall distance (-phi) of one reference along the smoothed centerline starting at the portal, against the catheter radius."""
    s0, pts, rad = audited_path_points(path_pts_mm, path_rad_mm, portal_ct_mm, constants, length_mm, step_mm)
    ss = np.arange(len(pts))
    wall = -field.phi_mm(pts)
    R = constants["radius_mm"]
    return {"reference": field.name, "n": int(len(ss)), "s_start_mm": float(s0), "min_wall_distance_mm": float(wall.min()), "p05_wall_distance_mm": float(np.percentile(wall, 5)),
            "median_wall_distance_mm": float(np.median(wall)), "fraction_wall_distance_below_catheter_radius": float(np.mean(wall < R)), "fraction_centerline_outside_wall": float(np.mean(wall < 0)),
            "max_clearance_deficit_mm": float(max(0.0, R - wall.min())), "median_twin_radius_mm": float(np.median(rad)), "twin_radius_range_mm": [float(rad.min()), float(rad.max())], "median_wall_minus_twin_radius_mm": float(np.median(wall - rad)),
            "n_outside_grid": int(np.sum(~field.inside_grid(pts)))}


def config_analysis(cfg, geom: PatientGeometry, constants: dict, mesh_ref=None) -> dict:
    """Everything that needs only the configuration and the twin (CPU): straight deployment and path clearance under every available reference."""
    z = np.load(cfg.path_npz_path)
    portal = cfg.ct_offset * 1000.0
    refs = {"fullres": geom.fullres, "ds": geom.ds}
    if geom.fullres_hc is not None:
        refs["fullres_hc"] = geom.fullres_hc
    if mesh_ref is not None:
        refs["mesh"] = mesh_ref
    out = {"patient_id": cfg.patient_id, "access_side": cfg.access_side, "portal_ct_mm": portal.tolist(), "tangent": cfg.tangent.tolist(), "sign_scale": cfg.sign_scale,
           "straight_deployment": {}, "path_clearance": {}}
    for name, f in refs.items():
        out["straight_deployment"][name] = straight_deployment(f, portal, cfg.tangent, constants)
        if name != "mesh":
            out["path_clearance"][name] = path_clearance(f, z["pts"], z["rad"], portal, constants)
    return out


# --------------------------------------------------------------------------- read-only probe trace analysis
def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:3], q[3]
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def body_axis_points_m(body_q_row: np.ndarray, l_seg_m: float, n: int, fallback_axis) -> np.ndarray:
    """n points along the axis of one body (solver-local metres) from its centre -L/2 to +L/2."""
    ax = _quat_rotate_xyzw(np.asarray(body_q_row[3:7], float), np.array([0.0, 0.0, 1.0]))
    nrm = float(np.linalg.norm(ax))
    ax = ax / nrm if nrm > 1e-12 else np.asarray(fallback_axis, float)
    return np.asarray(body_q_row[:3], float) + np.linspace(-0.5, 0.5, n)[:, None] * l_seg_m * ax


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p05": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)), "min": float(x.min()), "max": float(x.max()),
            "max_abs": float(np.abs(x).max())}


def _event_max_violation(phi_mm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per event: max(0, max phi over the valid samples) (the engine's test); events without samples give 0."""
    masked = np.where(valid, phi_mm, -np.inf)
    return np.maximum(0.0, masked.max(axis=1, initial=-np.inf))


def analyze_trace(tr: dict, meta: dict, fields: dict, constants: dict, mesh_ref=None, dense_points: int = 61, tol_fn=None, sign_bound_mm: float | None = None, tol_p99_fn=None) -> dict:
    """Compare a recorded engine trace with the reference fields at the same physical points. ``tr`` holds the arrays written by ``ContainmentProbe`` and ``meta`` its metadata.

    ``fields['fullres']`` is the PRIMARY full-resolution reference (the variant selected by the locked calibration); an optional ``fields['alt']`` is reported alongside.
    ``tol_fn(points_mm) -> (tolerance_mm, in_bin)`` gives the calibrated reference error of the radius bin of each sample; with it every key event carries a rigorous interval for the
    reference violation, [max(0, max(phi - tol)), max(0, max(phi + tol))]. ``sign_bound_mm``: engine/reference sign disagreements farther than this from the wall are counted per event.
    A sample whose radius is outside every calibrated bin has no tolerance (UNAVAILABLE): the interval of its event is then None, never a guess.
    ``tol_p99_fn``: the same with the per-bin p99 error; it only feeds the descriptive SECONDARY_P99_DESCRIPTIVE analysis."""
    off_m = np.asarray(meta["ct_offset_m"], float)
    tgt, delta, R = constants["target_phi_mm"], constants["lumen_delta_mm"], constants["radius_mm"]
    sb, spm, phi_e = tr["sample_body"], tr["sample_points_m"], tr["phi_engine_m"] * 1000.0
    valid = sb >= 0
    hit = valid & (tr["sample_hit"] > 0)
    E, S = sb.shape
    pts_mm = (spm + off_m) * 1000.0
    phis = {"engine": phi_e}
    for name, f in fields.items():
        arr = np.full((E, S), np.nan)
        if valid.any():
            arr[valid] = f.phi_mm(pts_mm[valid])
        phis[name] = arr
    in_grid = np.ones((E, S), bool)
    if valid.any():
        in_grid[valid] = fields["fullres"].inside_grid(pts_mm[valid])
    viol = {k: _event_max_violation(v, valid) for k, v in phis.items()}
    pen = {k: np.maximum(0.0, np.where(valid, v - tgt, -np.inf).max(axis=1, initial=-np.inf)) for k, v in phis.items()}
    dec = tr["decision"]
    is_dec = dec >= 0
    engine_rec = tr["engine_violation_m"] * 1000.0
    consistency = float(np.nanmax(np.abs(engine_rec[is_dec] - viol["engine"][is_dec]))) if is_dec.any() else None
    out: dict = {"n_events": int(E), "n_decision_events": int(is_dec.sum()), "observer_consistency_max_abs_diff_mm": consistency,
                 "samples_outside_reference_grid": int(np.sum(valid & ~in_grid)), "reference_inside_grid": bool(not np.any(valid & ~in_grid)),
                 "samples_without_engine_sdf_hit": int(np.sum(valid & ~hit))}

    # same physical point, all events (samples with an engine hit only: without a hit the engine reports the query radius, not a distance)
    d_full, d_ds = (phis["engine"] - phis["fullres"])[hit], (phis["engine"] - phis["ds"])[hit]
    contact = hit & (phi_e > tgt)
    out["same_point_all_events"] = {"engine_minus_fullres_mm": _stats(d_full), "engine_minus_ds_mm": _stats(d_ds),
                                    "contact_active_engine_minus_fullres_mm": _stats((phis["engine"] - phis["fullres"])[contact]), "contact_active_engine_minus_ds_mm": _stats((phis["engine"] - phis["ds"])[contact])}

    # engine and reference on opposite sides of the wall: the largest such samples over the whole trace (raw record; the judgement belongs to the verdict)
    opp = hit & (np.sign(phi_e) != np.sign(phis["fullres"])) & (phis["fullres"] != 0)
    ei, si = np.nonzero(opp)
    order = np.argsort(-np.abs(phis["fullres"][ei, si]))[:10]
    out["sign_disagreements_all_events"] = {"n": int(opp.sum()), "largest_by_abs_fullres_phi": [
        {"event_index": int(ei[k]), "decision": int(dec[ei[k]]), "frame": int(tr["frame"][ei[k]]), "sub": int(tr["sub"][ei[k]]), "rod_index": int(sb[ei[k], si[k]]),
         "fraction": float(tr["sample_fraction"][ei[k], si[k]]), "engine_phi_mm": float(phi_e[ei[k], si[k]]), "fullres_phi_mm": float(phis["fullres"][ei[k], si[k]])} for k in order]}

    # time series per decision (max over the substeps of a decision, as the engine does)
    series = []
    for d in sorted(set(int(x) for x in dec[is_dec])):
        sel = dec == d
        series.append({"decision": d, "engine_violation_mm": float(viol["engine"][sel].max()), "fullres_violation_mm": float(viol["fullres"][sel].max()), "ds_violation_mm": float(viol["ds"][sel].max()),
                       "engine_penetration_mm": float(pen["engine"][sel].max()), "fullres_penetration_mm": float(pen["fullres"][sel].max()), "n_active_end": int(tr["n_active"][sel][-1])})
    out["decision_series"] = series

    # key events
    names = list(meta["stage_names"])
    def first_stage(stage):
        c = names.index(stage) if stage in names else -1
        w = np.flatnonzero(tr["stage_code"] == c)
        return int(w[0]) if len(w) else None

    keys: dict[str, int | None] = {"build_scene": first_stage("build_scene"), "ready": first_stage("ready"), "trigger": None, "peak": None}
    if is_dec.any():
        last = int(dec.max())
        cand = np.flatnonzero(dec == last)
        keys["trigger"] = int(cand[np.argmax(engine_rec[cand])])          # the substep that carries the maximum of the last decision (the one the engine compares with the limit)
        di = np.flatnonzero(is_dec)
        keys["peak"] = int(di[np.argmax(engine_rec[di])])
    moving_idx = meta["moving_rod_indices"]
    out["key_events"] = {}
    for label, i in keys.items():
        if i is None:
            out["key_events"][label] = None
            continue
        v = valid[i]
        ev = {"event_index": i, "stage": names[int(tr["stage_code"][i])], "decision": int(dec[i]), "frame": int(tr["frame"][i]), "sub": int(tr["sub"][i]), "n_active": int(tr["n_active"][i]),
              "engine_violation_mm": float(viol["engine"][i]), "fullres_violation_mm": float(viol["fullres"][i]), "ds_violation_mm": float(viol["ds"][i]),
              "engine_penetration_mm": float(pen["engine"][i]), "fullres_penetration_mm": float(pen["fullres"][i]), "recorded_contact_count": int(tr["contact_count"][i]),
              "n_samples_contact_active_engine": int(np.sum(v & (phi_e[i] > tgt))), "n_samples_contact_active_fullres": int(np.sum(v & (phis["fullres"][i] > tgt)))}
        k = int(np.argmax(np.where(v, phi_e[i], -np.inf)))
        ev["worst_sample_engine"] = {"rod_index": int(sb[i, k]), "fraction": float(tr["sample_fraction"][i, k]), "point_ct_mm": pts_mm[i, k].tolist(), "engine_phi_mm": float(phi_e[i, k]),
                                     "fullres_phi_mm": float(phis["fullres"][i, k]), "ds_phi_mm": float(phis["ds"][i, k]), "engine_has_sdf_hit": bool(hit[i, k])}
        hv = hit[i]
        diff = (phi_e[i] - phis["fullres"][i])[hv]
        ev["engine_minus_fullres_over_samples_mm"] = _stats(diff)
        ev["max_abs_engine_minus_fullres_mm"] = float(np.abs(diff).max()) if len(diff) else None
        ev["engine_minus_ds_over_samples_mm"] = _stats((phi_e[i] - phis["ds"][i])[hv])
        ev["criterion_flip_fullres_vs_engine"] = bool((viol["engine"][i] > delta) != (viol["fullres"][i] > delta))
        if "alt" in phis:
            ev["alt_violation_mm"] = float(viol["alt"][i])
        pf = phis["fullres"][i][v]
        ev["fullres_violation_lower_mm"] = ev["fullres_violation_upper_mm"] = ev["n_samples_outside_calibrated_bins"] = None
        if tol_fn is not None and v.any():
            tol, in_bin = tol_fn(pts_mm[i][v])
            ev["n_samples_outside_calibrated_bins"] = int(np.sum(~in_bin))
            ev["tolerance_samples_mm"] = {"min": float(np.min(tol[in_bin])) if in_bin.any() else None, "max": float(np.max(tol[in_bin])) if in_bin.any() else None}
            if in_bin.all():
                ev["fullres_violation_lower_mm"] = float(max(0.0, np.max(pf - tol)))
                ev["fullres_violation_upper_mm"] = float(max(0.0, np.max(pf + tol)))
        ev["fullres_violation_lower_p99_mm"] = ev["fullres_violation_upper_p99_mm"] = None
        if tol_p99_fn is not None and v.any():
            tol99, in_bin99 = tol_p99_fn(pts_mm[i][v])
            if in_bin99.all():
                ev["fullres_violation_lower_p99_mm"] = float(max(0.0, np.max(pf - tol99)))
                ev["fullres_violation_upper_p99_mm"] = float(max(0.0, np.max(pf + tol99)))
        ev["n_engine_sign_disagreements_beyond_bound"] = None
        if sign_bound_mm is not None:
            far = v & hit[i] & (np.sign(phi_e[i]) != np.sign(phis["fullres"][i])) & (np.abs(phis["fullres"][i]) > sign_bound_mm)
            ev["n_engine_sign_disagreements_beyond_bound"] = int(far.sum())
        # axis versus physical body: continuous axis of every active moving body against the full-resolution reference
        dense = []
        for ri in moving_idx:
            if ri < ev["n_active"]:
                dense.append(body_axis_points_m(tr["body_q"][i, ri], meta["l_seg_m"], dense_points, meta["tangent"]))
        if dense:
            dp = (np.vstack(dense) + off_m) * 1000.0
            dphi = fields["fullres"].phi_mm(dp)
            ev["dense_axis_violation_fullres_mm"] = float(max(0.0, dphi.max()))
            ev["dense_body_penetration_fullres_mm"] = float(max(0.0, (dphi - tgt).max()))
            ev["axis_to_body_gap_fullres_mm"] = ev["dense_body_penetration_fullres_mm"] - ev["dense_axis_violation_fullres_mm"]
        if mesh_ref is not None:
            pm = mesh_ref.phi_mm(pts_mm[i][hv])
            ev["engine_minus_mesh_over_samples_mm"] = _stats(phi_e[i][hv] - pm)
            ev["max_abs_engine_minus_mesh_mm"] = float(np.abs(phi_e[i][hv] - pm).max()) if len(pm) else None
        out["key_events"][label] = ev

    # the engine's start state against the analytic straight placement
    bs = keys["build_scene"]
    if bs is not None:
        exp = straight_rod_samples(np.asarray(off_m) * 1000.0, meta["tangent"], constants, int(tr["n_active"][bs]))
        got = pts_mm[bs][valid[bs]]
        out["start_vs_analytic_straight_rod_max_mm"] = float(np.linalg.norm(got - np.array([p for _k, _f, p in exp]), axis=1).max()) if len(got) == len(exp) else None
    out["constants_match_trace"] = bool(abs(meta["l_seg_m"] * 1000.0 - constants["l_seg_mm"]) < 1e-9 and abs(meta["radius_m"] * 1000.0 - constants["radius_mm"]) < 1e-9
                                        and list(meta["sample_frac"]) == list(constants["sample_frac"]))
    return out


# --------------------------------------------------------------------------- numerical calibration on analytic synthetic geometry
def _error_summary(errs: list[np.ndarray], tags: list[str]) -> dict | None:
    keep = [(e, t) for e, t in zip(errs, tags) if len(e)]
    if not keep:
        return None
    allv = np.concatenate([e for e, _ in keep])
    worst = int(np.argmax([np.abs(e).max() for e, _ in keep]))
    return {"n": int(len(allv)), "max_abs_mm": float(np.abs(allv).max()), "p99_abs_mm": float(np.percentile(np.abs(allv), 99)), "median_signed_mm": float(np.median(allv)),
            "p05_signed_mm": float(np.percentile(allv, 5)), "p95_signed_mm": float(np.percentile(allv, 95)), "worst_design": keep[worst][1]}


VARIANT_SELECTION_RULE = "half_voxel_if_lower_abs_median_and_lower_p99_abs"


def design_bins(design: dict) -> list[dict]:
    """The declared radius bins: contiguous, ascending, every synthetic tube radius inside its bin."""
    bins = design["radius_bins"]
    for i, b in enumerate(bins):
        if not b["lo_mm"] < b["hi_mm"] or not b["tube_radius_mm"] or not all(b["lo_mm"] <= r <= b["hi_mm"] for r in b["tube_radius_mm"]):
            raise ValueError(f"invalid radius bin {b}")
        if i and b["lo_mm"] != bins[i - 1]["hi_mm"]:
            raise ValueError("radius bins must be contiguous and ascending")
    return bins


def calibrate(design: dict, spacings_zyx_mm: list, constants: dict, engine_factory=None) -> dict:
    """Measure the numerical error of the full-resolution reference variants against exact analytic tubes, binned by the local tube radius, and report the discretization EFFECT
    of the collision surface (ds, CPU mesh, engine) separately. Only the reference error enters the tolerances.

    Tolerance of a bin = max |phi_reference - phi_analytic| over the declared design inside that bin; the primary variant is chosen by ``VARIANT_SELECTION_RULE`` on the pooled errors.
    The engine query is bounded by the engine against a CPU mesh of the SAME triangles (``ENGINE_MESH_AGREEMENT_TOL_MM``, IMPLEMENTATION); queries beyond it, or - without a CPU mesh -
    wrong-sign queries farther from the true surface than any collision-mesh vertex, are counted as rare query failures (rate, examples) and excluded from the continuous errors."""
    if design.get("variant_selection_rule") != VARIANT_SELECTION_RULE:
        raise ValueError(f"the design must declare variant_selection_rule = {VARIANT_SELECTION_RULE!r}")
    bins = design_bins(design)
    rng = np.random.default_rng(design["seed"])
    stride = constants["mask_downsample"]
    names = ("fullres", "fullres_hc", "ds", "mesh", "engine", "engine_vs_mesh")
    errs: dict[str, list] = {k: [] for k in names}
    tags: dict[str, list] = {k: [] for k in names}
    binids: dict[str, list] = {k: [] for k in names}
    vertex_dev = []
    diag_ok = {"fullres": True, "fullres_hc": True}
    engine_runs: list = []
    sphere = validate_mesh_reference_on_sphere()

    def add(name, bid, tag, e):
        errs[name].append(e)
        tags[name].append(tag)
        binids[name].append(bid)

    for sp in spacings_zyx_mm:
        diag = float(np.linalg.norm(sp))
        for b in bins:
            for r in b["tube_radius_mm"]:
                for bend in design["bends"]:
                    for rot in design["rotations_deg"]:
                        for off in design["voxel_offset_fractions_xyz"]:
                            tube = sg.TubeSpec(radius_mm=r, straight_in_mm=design["straight_in_mm"], straight_out_mm=design["straight_out_mm"], bend_radius_mm=bend["bend_radius_mm"],
                                               turn_deg=bend["turn_deg"], rotation_deg=tuple(rot), start_mm=tuple(design["start_mm"]))
                            tag = f"spacing={list(map(float, sp))}|r={r}|bend={bend}|rot={rot}|off={off}"
                            mask, origin, spv = sg.voxelize(tube, sp, design["margin_mm"], off)
                            fields = {"fullres": SignedDistanceField.from_mask(mask, origin, spv, 1), "fullres_hc": SignedDistanceField.from_mask(mask, origin, spv, 1, half_voxel=True),
                                      "ds": SignedDistanceField.from_mask(mask, origin, spv, stride)}
                            pts = sg.random_points_near_surface(tube, design["points_per_case"], design["evaluation_band_mm"], rng)
                            phi_true = -tube.sd_mm(pts)
                            for name, f in fields.items():
                                ok = f.inside_grid(pts)
                                e = f.phi_mm(pts[ok]) - phi_true[ok]
                                add(name, b["id"], tag, e)
                                if name in diag_ok and len(e) and np.abs(e).max() > diag:
                                    diag_ok[name] = False
                            mv, mf = collision_mesh_ct_mm(mask, origin, spv, stride)
                            vertex_dev.append(float(np.abs(tube.sd_mm(mv)).max()))
                            mesh = mesh_reference_or_none(mv, mf) if sphere["available"] else None
                            pm = None
                            if mesh is not None:
                                pm = mesh.phi_mm(pts)
                                add("mesh", b["id"], tag, pm - phi_true)
                            if engine_factory is not None:
                                engine_runs.append((b["id"], tag, np.asarray(engine_factory(mask, origin, spv)(pts), float), phi_true, pm))
    # engine queries: separate rare failures from the continuous error. With a CPU mesh of the same triangles a failure is |engine - mesh| > ENGINE_MESH_AGREEMENT_TOL_MM. Without it, a
    # failure is a wrong sign at a point farther from the true surface than the largest measured deviation of the collision-mesh vertices from the true surface (a point that far from
    # the surface cannot be on the other side of any facet).
    h_bound = float(max(vertex_dev)) if vertex_dev else None
    disagreements, n_compared, mode = [], 0, ("own_cpu_mesh" if engine_runs and all(r[4] is not None for r in engine_runs) else "analytic_sign_beyond_mesh_deviation")
    for bid, tag, pe, ptrue, pm in engine_runs:
        n_compared += len(pe)
        if mode == "own_cpu_mesh":
            dis = np.abs(pe - pm) > ENGINE_MESH_AGREEMENT_TOL_MM
            add("engine_vs_mesh", bid, tag, (pe - pm)[~dis])
        else:
            dis = (np.sign(pe) != np.sign(ptrue)) & (np.abs(ptrue) > h_bound)
        for j in np.flatnonzero(dis):
            disagreements.append({"design": tag, "bin": bid, "phi_true_mm": float(ptrue[j]), "phi_engine_mm": float(pe[j]), "phi_cpu_mesh_mm": (float(pm[j]) if pm is not None else None),
                                  "sign_differs": bool(np.sign(pe[j]) != np.sign(ptrue[j]))})
        add("engine", bid, tag, (pe - ptrue)[~dis])
    summ = {k: _error_summary(errs[k], tags[k]) for k in names}

    def by_bin(name):
        out = {}
        for b in bins:
            idx = [i for i, x in enumerate(binids[name]) if x == b["id"]]
            out[b["id"]] = _error_summary([errs[name][i] for i in idx], [tags[name][i] for i in idx])
        return out

    # reference variant: the half-voxel face-centred field replaces the centre-to-centre one only when it has the lower |median| and the lower p99 on the pooled synthetic errors
    pooled = {k: (abs(summ[k]["median_signed_mm"]), summ[k]["p99_abs_mm"]) for k in ("fullres", "fullres_hc")}
    primary = "fullres_hc" if (pooled["fullres_hc"][0] < pooled["fullres"][0] and pooled["fullres_hc"][1] < pooled["fullres"][1]) else "fullres"
    per_bin = by_bin(primary)
    tol_bins = [{"id": b["id"], "lo_mm": b["lo_mm"], "hi_mm": b["hi_mm"], "value_mm": per_bin[b["id"]]["max_abs_mm"], "n": per_bin[b["id"]]["n"],
                 "median_signed_mm": per_bin[b["id"]]["median_signed_mm"], "p05_signed_mm": per_bin[b["id"]]["p05_signed_mm"], "p95_signed_mm": per_bin[b["id"]]["p95_signed_mm"], "p99_abs_mm": per_bin[b["id"]]["p99_abs_mm"]} for b in bins]
    tol: dict = {"fullres_reference_vs_analytic": {
        "value_mm": max(t["value_mm"] for t in tol_bins), "class": "SYNTHETIC_VALIDATION", "variant": primary, "bins": tol_bins,
        "derivation": "per radius bin: max |phi_reference - phi_analytic| over the declared design; 'value_mm' is the maximum over the bins (informational)"},
        "engine_query_vs_cpu_mesh_same_triangles": {"value_mm": ENGINE_MESH_AGREEMENT_TOL_MM, "class": "IMPLEMENTATION",
                                                    "derivation": "float32 Warp query and CPU query on the same triangles; larger differences are counted as query failures"}}
    if h_bound is not None:
        tol["collision_mesh_vertex_deviation_bound"] = {"value_mm": h_bound, "class": "SYNTHETIC_VALIDATION",
                                                       "derivation": "max distance of any collision-mesh vertex from the analytic surface over the design; wrong-sign engine queries beyond it are query failures, not discretization"}
    effects = {"note": "discretization EFFECTS of the collision surface against the analytic truth, per radius bin; reported, never used as a tolerance",
               "ds_vs_analytic_by_bin": by_bin("ds"), "mesh_vs_analytic_by_bin": by_bin("mesh") if summ["mesh"] is not None else None,
               "engine_vs_analytic_by_bin_without_query_failures": by_bin("engine") if summ["engine"] is not None else None,
               "reference_variants_pooled": {k: summ[k] for k in ("fullres", "fullres_hc")}}
    checks = {"fullres_error_within_voxel_diagonal": bool(diag_ok[primary]), "mesh_reference_sphere_validation": bool(sphere["available"]),
              "mesh_reference_available": bool(summ["mesh"] is not None), "engine_path_calibrated": bool(summ["engine"] is not None),
              "engine_failures_separated": bool(engine_factory is not None and n_compared > 0), "engine_query_bounded_by_own_mesh": bool(mode == "own_cpu_mesh" and n_compared > 0)}
    dis_summary = None if not n_compared else {
        "n_queries_compared": int(n_compared), "n_disagreements": len(disagreements), "rate": len(disagreements) / n_compared,
        "n_sign_differs": int(sum(d["sign_differs"] for d in disagreements)), "detection": mode, "agreement_tolerance_mm": ENGINE_MESH_AGREEMENT_TOL_MM, "mesh_deviation_bound_mm": h_bound,
        "examples": sorted(disagreements, key=lambda d: -abs(d["phi_engine_mm"] - d["phi_true_mm"]))[:10]}
    return {"error_summary": summ, "effects": effects, "tolerances": tol, "primary_reference": {"name": primary, "rule": VARIANT_SELECTION_RULE, "pooled_abs_median_and_p99_mm": pooled},
            "checks": checks, "engine_vs_own_mesh_disagreements": dis_summary, "mesh_reference_sphere_validation": sphere,
            "mesh_vertex_deviation_from_analytic_surface_max_mm": h_bound, "design": design, "spacings_zyx_mm": [list(map(float, s)) for s in spacings_zyx_mm],
            "constants": {k: (list(v) if isinstance(v, tuple) else v) for k, v in constants.items()}}


# --------------------------------------------------------------------------- self-test (engineering validation only; not the locked calibration)
SELFTEST_DESIGN = {"seed": 7, "variant_selection_rule": VARIANT_SELECTION_RULE,
                   "radius_bins": [{"id": "r3.0-4.0", "lo_mm": 3.0, "hi_mm": 4.0, "tube_radius_mm": [3.0]}, {"id": "r4.0-5.0", "lo_mm": 4.0, "hi_mm": 5.0, "tube_radius_mm": [4.5]}],
                   "bends": [{"bend_radius_mm": None, "turn_deg": 0.0}, {"bend_radius_mm": 25.0, "turn_deg": 70.0}],
                   "rotations_deg": [[0.0, 0.0, 0.0], [25.0, 40.0, 15.0]], "voxel_offset_fractions_xyz": [[0.0, 0.0, 0.0], [0.31, 0.17, 0.43]], "straight_in_mm": 30.0, "straight_out_mm": 30.0,
                   "start_mm": [10.0, -20.0, 5.0], "margin_mm": 12.0, "evaluation_band_mm": 6.0, "points_per_case": 400}


def run_selftest(constants: dict | None = None) -> list[dict]:
    """Synthetic analytic checks of the measurement implementation. Each result: {'check', 'passed', 'detail'}."""
    constants = constants or canonical_constants()
    res: list[dict] = []
    def add(name, passed, detail=None):
        res.append({"check": name, "passed": bool(passed), "detail": detail})

    # 1 analytic tube signed distance against brute force on a dense centerline
    rng = np.random.default_rng(1)
    for t in (sg.TubeSpec(3.0, 30.0, 30.0, rotation_deg=(20.0, 10.0, 5.0)), sg.TubeSpec(3.0, 30.0, 30.0, 25.0, 70.0, (20.0, 10.0, 5.0), (3.0, 2.0, 1.0))):
        pts = sg.random_points_near_surface(t, 300, 6.0, rng)
        cl = t.centerline_world(0.02)
        brute = np.array([np.linalg.norm(cl - p, axis=1).min() for p in pts])
        add("analytic tube signed distance equals brute force", np.abs((t.radius_mm - brute) - t.sd_mm(pts)).max() < 0.05, float(np.abs((t.radius_mm - brute) - t.sd_mm(pts)).max()))

    # 2 grid references against the analytic tube, isotropic and anisotropic spacing, sub-voxel offsets
    for sp in ((1.5, 1.5, 1.5), (2.0, 1.0, 1.5)):
        tube = sg.TubeSpec(4.0, 30.0, 30.0, 25.0, 70.0, (20.0, 10.0, 5.0), (3.0, 2.0, 1.0))
        mask, origin, spv = sg.voxelize(tube, sp, 12.0, (0.3, 0.2, 0.4))
        pts = sg.random_points_near_surface(tube, 600, 6.0, np.random.default_rng(2))
        diag = float(np.linalg.norm(sp))
        for stride in (1, constants["mask_downsample"]):
            f = SignedDistanceField.from_mask(mask, origin, spv, stride)
            err = f.phi_mm(pts) + tube.sd_mm(pts)
            add(f"{f.name} reference error within one voxel diagonal (spacing {sp})", np.abs(err).max() <= diag * stride, float(np.abs(err).max()))
            add(f"{f.name} reference sign matches analytic outside one voxel diagonal", not np.any((np.abs(tube.sd_mm(pts)) > diag * stride) & (np.sign(f.phi_mm(pts)) == np.sign(tube.sd_mm(pts)))), None)

    # 3 voxel convention: a mask voxel centre must lie inside, its non-mask neighbour outside
    m = np.zeros((9, 9, 9), bool); m[3:6, 3:6, 3:6] = True
    f = SignedDistanceField.from_mask(m, (10.0, 20.0, 30.0), (2.0, 1.0, 1.5), 1)
    c_in = np.array([[10.0 + 4 * 1.5, 20.0 + 4 * 1.0, 30.0 + 4 * 2.0]])       # xyz of voxel (z=4, y=4, x=4): x uses spacing[2]=1.5, y spacing[1]=1.0, z spacing[0]=2.0
    c_out = np.array([[10.0 + 7 * 1.5, 20.0 + 4 * 1.0, 30.0 + 4 * 2.0]])
    add("voxel convention (z, y, x) with world = origin + index * spacing", f.phi_mm(c_in)[0] < 0 < f.phi_mm(c_out)[0], [float(f.phi_mm(c_in)[0]), float(f.phi_mm(c_out)[0])])

    # 4 straight deployment: a rod along a straight tube centred on its axis is contained; tilted rods leave the tube at a known distance
    tube = sg.TubeSpec(5.0, 200.0)
    af = sg.AnalyticField(tube)
    r = straight_deployment(af, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], constants)
    add("straight rod on the tube axis: phi = -tube radius, contained", abs(r["axis_violation_mm"]) < 1e-9 and r["contact_free"] and abs(max(r["samples_phi_mm"]) + 5.0) < 1e-9, r["samples_phi_mm"][:3])
    tilt = math.radians(10.0)
    r = straight_deployment(af, [0.0, 0.0, 0.0], [math.sin(tilt), 0.0, math.cos(tilt)], constants)
    n = constants["predeploy_active"]
    far = (n - 1 + constants["sample_frac"][-1] - 0.5) * constants["l_seg_mm"]
    exp_phi = far * math.sin(tilt) - 5.0
    add("tilted straight rod: farthest axis sample at the analytic lateral offset", abs(max(r["samples_phi_mm"]) - exp_phi) < 1e-9 and abs(r["axis_violation_mm"] - max(0.0, exp_phi)) < 1e-9, [max(r["samples_phi_mm"]), exp_phi])
    add("contact activation criterion equals phi > -R", r["n_samples_in_contact"] == int(np.sum(np.array(r["samples_phi_mm"]) > -constants["radius_mm"])), r["n_samples_in_contact"])
    add("capsule penetration is phi + R (contact semantics) and exceeds the axis violation by R", abs(r["capsule_penetration_mm"] - (max(r["samples_phi_mm"]) + constants["radius_mm"])) < 1e-9, r["capsule_penetration_mm"])

    # 5 quaternion helper against a known rotation
    q = np.array([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)])             # +90 deg about y: z-axis -> x-axis
    pts = body_axis_points_m(np.r_[[0.1, 0.2, 0.3], q], 0.03, 3, [0, 0, 1])
    add("body axis points follow the quaternion (x, y, z, w)", np.allclose(pts[1], [0.1, 0.2, 0.3]) and np.allclose(pts[2] - pts[0], [0.03, 0.0, 0.0], atol=1e-12), pts.tolist())

    # 6 mesh reference: validated or UNAVAILABLE
    ms = validate_mesh_reference_on_sphere()
    add("secondary mesh reference validated on an analytic sphere, or reported UNAVAILABLE", True, {"status": "VALIDATED" if ms["available"] else "UNAVAILABLE", **ms})

    # 7 small calibration run with the self-test design (CPU; exercises the whole calibration code path)
    cal = calibrate(SELFTEST_DESIGN, [(1.5, 1.5, 1.5)], constants)
    add("calibration code path runs and the reference errors are bounded by one voxel diagonal", cal["checks"]["fullres_error_within_voxel_diagonal"]
        and cal["effects"]["reference_variants_pooled"][cal["primary_reference"]["name"]]["max_abs_mm"] <= math.sqrt(3) * 1.5, {k: v["value_mm"] for k, v in cal["tolerances"].items()})
    # 8 half-voxel variant: on a straight wall (axis-aligned slab) the centre-to-centre field has slope 2 across the wall, the face-centred one slope 1
    slab = np.zeros((12, 12, 12), bool); slab[:, :, :6] = True                                  # wall between x index 5 and 6, i.e. at x = 5.5 * 1.5 mm
    wall_x = 5.5 * 1.5
    pt = np.array([[wall_x - 0.5, 8.0, 8.0]])                                                    # 0.5 mm inside
    f0, f1 = SignedDistanceField.from_mask(slab, (0.0, 0.0, 0.0), (1.5, 1.5, 1.5), 1), SignedDistanceField.from_mask(slab, (0.0, 0.0, 0.0), (1.5, 1.5, 1.5), 1, half_voxel=True)
    add("face-centred variant reproduces the distance to the voxel face; the centre-to-centre field doubles it near the wall", abs(f1.phi_mm(pt)[0] + 0.5) < 1e-6 and abs(f0.phi_mm(pt)[0] + 1.0) < 1e-6,
        [float(f0.phi_mm(pt)[0]), float(f1.phi_mm(pt)[0])])
    return res
