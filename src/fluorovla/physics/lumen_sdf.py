"""Lumen signed-distance contact: collision mesh from the twin vessel mask, SDF queries, and the contact generator used by the patient episode.

Semantics (unchanged from the validated implementation): phi = sign_scale * mesh_query_sign * distance, phi < 0 inside the lumen. A rod sample whose phi exceeds
``TARGET_PHI_M`` (= minus the catheter radius) generates a contact against the wall point obtained by moving the sample along the outward normal by -phi.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import warp as wp
from newton import Contacts
from skimage import measure

from fluorovla.physics.engine import CONTACT_MAX, DEVICE, L_SEG, RADIUS, quat_rotate
from fluorovla.physics.patient_config import PatientConfig

# ---- fixed contact parameters (patient independent; unchanged from the validated implementation) ----
MASK_DOWNSAMPLE = 2            # lumen mask is subsampled by 2 before marching cubes (3 mm collision-mesh voxel at 1.5 mm CT spacing)
CATHETER_RADIUS_M = float(RADIUS)
MAX_DIST_M = 0.05              # SDF query radius
TARGET_PHI_M = -CATHETER_RADIUS_M
SAMPLE_FRAC = (0.0, 0.25, 0.5, 0.75, 1.0)   # axis samples per rod body, proximal -> distal
N_SAMPLES = 5


@wp.kernel
def query_sdf_at_points(
    mesh_id: wp.uint64,
    pts: wp.array(dtype=wp.vec3),
    max_dist: float,
    sign_scale: float,
    phi_out: wp.array(dtype=float),
    nx_out: wp.array(dtype=float),
    ny_out: wp.array(dtype=float),
    nz_out: wp.array(dtype=float),
    hit_out: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    p = pts[i]
    res = wp.mesh_query_point_sign_normal(mesh_id, p, max_dist)
    if not res.result:
        phi_out[i] = max_dist
        nx_out[i] = 0.0
        ny_out[i] = 0.0
        nz_out[i] = 0.0
        hit_out[i] = 0
        return
    closest = wp.mesh_eval_position(mesh_id, res.face, res.u, res.v)
    delta = p - closest
    dist = wp.length(delta)
    phi_out[i] = sign_scale * res.sign * dist
    hit_out[i] = 1
    if dist < 1.0e-12:
        nx_out[i] = 0.0
        ny_out[i] = 0.0
        nz_out[i] = 0.0
        return
    # outward normal ~ sign_scale * sign * normalize(p - closest)
    g = (sign_scale * res.sign) * (delta / dist)
    gn = wp.length(g)
    if gn < 1.0e-12:
        nx_out[i] = 0.0
        ny_out[i] = 0.0
        nz_out[i] = 0.0
    else:
        n = g / gn
        nx_out[i] = n[0]
        ny_out[i] = n[1]
        nz_out[i] = n[2]


def _marching_cubes_ct_m(cfg: PatientConfig):
    """Marching cubes on the twin vessel mask; returns (vertices in CT metres float32, faces int32, downsample, spacing mm, origin mm)."""
    md = json.loads((cfg.ct_dir / "metadata.json").read_text())
    mask = np.load(cfg.ct_dir / "vessel_mask.npy")
    ds = MASK_DOWNSAMPLE
    mask_ds = mask[::ds, ::ds, ::ds]
    spacing = np.asarray(md["spacing_zyx_mm"], dtype=np.float64) * ds
    origin = np.asarray(md["origin_xyz_mm"], dtype=np.float64)
    verts_zyx, faces, _, _ = measure.marching_cubes(mask_ds.astype(np.float32), level=0.5, spacing=tuple(spacing))
    verts_xyz_mm = np.column_stack([verts_zyx[:, 2], verts_zyx[:, 1], verts_zyx[:, 0]]) + origin
    verts_ct_m = (verts_xyz_mm / 1000.0).astype(np.float32)
    return verts_ct_m, faces.astype(np.int32), ds, spacing, origin


def load_lumen_mesh(cfg: PatientConfig) -> tuple[wp.Mesh, dict[str, Any]]:
    """Collision mesh of the lumen in solver-local metres (CT absolute minus the CT offset). Builds and caches it when the cache file does not exist."""
    off32 = cfg.ct_offset.astype(np.float32)
    meta: dict[str, Any] = {"downsample": MASK_DOWNSAMPLE, "extractor": "skimage.measure.marching_cubes level=0.5", "mesh_frame": cfg.mesh_frame,
                            "ct_offset_m": cfg.ct_offset.tolist(), "sign_scale": cfg.sign_scale}
    cache = cfg.mesh_cache
    if not cache.is_file():
        verts_ct_m, faces, ds, spacing, origin = _marching_cubes_ct_m(cfg)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cfg.mesh_frame == "ct":
            np.savez_compressed(cache, vertices_ct_m=verts_ct_m, faces=faces, downsample=np.int32(ds))
        else:
            np.savez_compressed(cache, vertices_m=(verts_ct_m - off32).astype(np.float32), faces=faces, ct_offset_m=off32, downsample=np.int32(ds),
                                spacing_zyx_mm=spacing.astype(np.float32), origin_xyz_mm=origin.astype(np.float32))
        meta["cache_written"] = str(cache)
    data = np.load(cache)
    if cfg.mesh_frame == "ct":
        verts = (data["vertices_ct_m"].astype(np.float32) - off32).astype(np.float32)
    elif cfg.mesh_frame == "solver_local":
        verts = data["vertices_m"].astype(np.float32)
    else:
        raise ValueError(f"unknown mesh_frame {cfg.mesh_frame!r}")
    faces = data["faces"].astype(np.int32)
    meta["cache"] = str(cache)
    meta["n_vertices"] = int(verts.shape[0])
    meta["n_faces"] = int(faces.shape[0])
    mesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device=DEVICE), indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=DEVICE))
    return mesh, meta


class LumenSDF:
    """A lumen mesh together with its SDF sign convention."""

    def __init__(self, cfg: PatientConfig):
        self.cfg = cfg
        self.mesh, self.meta = load_lumen_mesh(cfg)
        self.sign_scale = float(cfg.sign_scale)

    def query(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(phi [m], outward normals, hit flags) at world points."""
        return query_points_sdf(self.mesh, pts, self.sign_scale)


def query_points_sdf(mesh: wp.Mesh, pts: np.ndarray, sign_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    n = len(pts)
    if n == 0:
        return np.zeros(0), np.zeros((0, 3)), np.zeros(0, dtype=np.int32)
    wp_pts = wp.array(pts, dtype=wp.vec3, device=DEVICE)
    phi = wp.zeros(n, dtype=float, device=DEVICE)
    nx = wp.zeros(n, dtype=float, device=DEVICE)
    ny = wp.zeros(n, dtype=float, device=DEVICE)
    nz = wp.zeros(n, dtype=float, device=DEVICE)
    hit = wp.zeros(n, dtype=wp.int32, device=DEVICE)
    wp.launch(query_sdf_at_points, dim=n, inputs=[mesh.id, wp_pts, float(MAX_DIST_M), float(sign_scale), phi, nx, ny, nz, hit], device=DEVICE)
    ph = phi.numpy().astype(np.float64)
    normals = np.stack([nx.numpy(), ny.numpy(), nz.numpy()], axis=1).astype(np.float64)
    return ph, normals, hit.numpy()


def quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = map(float, q[:4])
    return quat_rotate(np.array([-x, -y, -z, w], dtype=np.float64), v)


def transform_point_world(X_row: np.ndarray, p_local: np.ndarray) -> np.ndarray:
    return X_row[:3].astype(np.float64) + quat_rotate(X_row[3:7], np.asarray(p_local, dtype=np.float64))


def generate_lumen_contacts(
    *,
    sdf: LumenSDF,
    contacts: Contacts,
    state,
    bodies: list[int],
    n_active: int,
    root: int,
    body_to_shape: dict[int, int],
    wall_shape_id: int,
) -> dict[str, Any]:
    contacts.clear()
    q = state.body_q.numpy()
    max_n = CONTACT_MAX
    sh0 = np.full(max_n, -1, dtype=np.int32)
    sh1 = np.full(max_n, -1, dtype=np.int32)
    p0 = np.zeros((max_n, 3), dtype=np.float32)
    p1 = np.zeros((max_n, 3), dtype=np.float32)
    nrm = np.zeros((max_n, 3), dtype=np.float32)
    m0 = np.zeros(max_n, dtype=np.float32)
    m1 = np.zeros(max_n, dtype=np.float32)
    o0 = np.zeros((max_n, 3), dtype=np.float32)
    o1 = np.zeros((max_n, 3), dtype=np.float32)
    count = 0
    movable = [b for b in bodies[:n_active] if b != root]
    for b in movable:
        shape0 = body_to_shape[b]
        X = q[b]
        locals_pts = [np.array([0.0, 0.0, (f - 0.5) * L_SEG], dtype=np.float64) for f in SAMPLE_FRAC]
        worlds = np.stack([transform_point_world(X, lp) for lp in locals_pts], axis=0).astype(np.float32)
        ph, normals, hits = sdf.query(worlds)
        for i in range(N_SAMPLES):
            if hits[i] == 0:
                continue
            phi = float(ph[i])
            if phi <= TARGET_PHI_M:
                continue
            n_out = normals[i]
            nn = float(np.linalg.norm(n_out))
            if nn < 1e-12:
                continue
            n_out = n_out / nn
            p_local = locals_pts[i]
            p_world = worlds[i].astype(np.float64)
            wall_world = p_world + n_out * (-phi)
            if count >= max_n:
                break
            sh0[count] = shape0
            sh1[count] = wall_shape_id
            p0[count] = p_local.astype(np.float32)
            p1[count] = wall_world.astype(np.float32)
            nrm[count] = n_out.astype(np.float32)
            m0[count] = float(RADIUS)
            m1[count] = 0.0
            o0[count] = quat_rotate_inv(X[3:7], n_out * RADIUS).astype(np.float32)
            o1[count] = 0.0
            count += 1
        if count >= max_n:
            break
    contacts.rigid_contact_shape0.assign(sh0)
    contacts.rigid_contact_shape1.assign(sh1)
    contacts.rigid_contact_point0.assign(p0)
    contacts.rigid_contact_point1.assign(p1)
    contacts.rigid_contact_normal.assign(nrm)
    contacts.rigid_contact_margin0.assign(m0)
    contacts.rigid_contact_margin1.assign(m1)
    contacts.rigid_contact_offset0.assign(o0)
    contacts.rigid_contact_offset1.assign(o1)
    cc = contacts.contact_counters.numpy()
    cc[0] = count
    cc[1] = 0
    contacts.contact_counters.assign(cc)
    return {"count": count}
