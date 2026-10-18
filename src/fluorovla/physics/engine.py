#!/usr/bin/env python3
"""Shared Newton material-feed engine (free-space mode + patient-contact mode).

Free-space mode: validated portal-slide feed / activation.

Patient-contact mode: validated portal-fixed / distal-joint
feed, tip-preserving activation, and SolverVBD body_q_prev rebaseline.

Do not change either mode's equations without a protocol erratum (see docs/reproducibility.md).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

import newton
from newton import Contacts, ModelFlags

DEVICE = "cuda:0"
NUM_SEGMENTS = 7
L_SEG = 0.03
RADIUS = 0.003
DT_FRAME = 1.0 / 60.0
SIM_SUBSTEPS = 5
SIM_DT = DT_FRAME / SIM_SUBSTEPS
NUM_FRAMES = 360
ELAPSED_S = NUM_FRAMES * DT_FRAME
PORTAL = np.array([0.0, 0.0, 0.0], dtype=np.float64)
AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# Explicit feed modes (do not silently interchange).
FEED_MODE_FREE_SPACE_PORTAL_SLIDE = "free_space_portal_slide"
FEED_MODE_PATIENT_DISTAL_JOINT = "patient_distal_joint"

# free-space bend (validated).
BEND_STIFFNESS_FREE_SPACE = 1.0e2
TWIST_STIFFNESS_FREE_SPACE = 1.0e2
# patient-contact mode patient-contact bend (validated value).
BEND_STIFFNESS_PATIENT = 1.0e1
TWIST_STIFFNESS_PATIENT = 1.0e1

# Canonical patient-feed physics (validated for the single-patient reachability + teacher task, 2026-10-13/14).
# Used by every patient_distal_joint scene by default. Free-space keeps its own 8 iterations.
PATIENT_VBD_ITERATIONS = 16            # was 8 in the pre-canonical engine (contact residual converged: 16 == 32)
FREE_SPACE_VBD_ITERATIONS = 8
PATIENT_TIP_BEND_TWIST_STIFFNESS = 15.0   # soft distal tip: bend/twist ke of the joint joining the CURRENT tip body to its parent
                                          # (every other rod joint keeps the configured bend/twist stiffness, e.g. 1e2)

# patient-contact mode contact materials when lumen-contact assets are requested.
SHAPE_KE = 1.0e5
SHAPE_KD = 1.0e2
SHAPE_MU = 0.3
CONTACT_MAX = 256

# Optional world pose override (patient mapping). None → free-space mode defaults.
_WORLD_POSE: dict[str, Any] | None = None
# Optional feed-mode override. None → free-space.
_FEED_CFG: dict[str, Any] | None = None

_WP_INIT = False


def configure_world_pose(
    *,
    portal_m: np.ndarray | list[float],
    axis: np.ndarray | list[float],
    root_at_portal: bool = False,
) -> None:
    """Set Newton world portal/axis for subsequent ``run_episode`` / ``build_scene`` calls.

    ``root_at_portal=False`` preserves free-space mode layout (polyline start at portal,
    root COM at portal + 0.5*L_seg*axis). ``root_at_portal=True`` places root COM
    exactly at the portal (patient mapping patient init).
    """
    global _WORLD_POSE
    p = np.asarray(portal_m, dtype=np.float64).reshape(3)
    a = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError("axis must be non-zero")
    _WORLD_POSE = {
        "portal": p.copy(),
        "axis": (a / n).copy(),
        "root_at_portal": bool(root_at_portal),
    }


def clear_world_pose() -> None:
    """Restore free-space mode default portal/axis for subsequent episodes."""
    global _WORLD_POSE
    _WORLD_POSE = None


def active_world_pose() -> tuple[np.ndarray, np.ndarray, bool]:
    if _WORLD_POSE is None:
        return PORTAL.copy(), AXIS.copy(), False
    return (
        _WORLD_POSE["portal"].copy(),
        _WORLD_POSE["axis"].copy(),
        bool(_WORLD_POSE["root_at_portal"]),
    )


def configure_feed_mode(
    mode: str,
    *,
    bend_stiffness: float | None = None,
    twist_stiffness: float | None = None,
) -> None:
    """Select free-space or patient-contact feed/activation semantics.

    Parameters
    ----------
    mode:
        ``free_space_portal_slide`` or ``patient_distal_joint``.
    bend_stiffness / twist_stiffness:
        Optional overrides. Defaults: free-space 1e2 / patient 1e1 (patient-contact mode).
    """
    global _FEED_CFG
    if mode not in (FEED_MODE_FREE_SPACE_PORTAL_SLIDE, FEED_MODE_PATIENT_DISTAL_JOINT):
        raise ValueError(f"unknown feed_mode {mode!r}")
    if mode == FEED_MODE_FREE_SPACE_PORTAL_SLIDE:
        bend = BEND_STIFFNESS_FREE_SPACE if bend_stiffness is None else float(bend_stiffness)
        twist = TWIST_STIFFNESS_FREE_SPACE if twist_stiffness is None else float(twist_stiffness)
    else:
        bend = BEND_STIFFNESS_PATIENT if bend_stiffness is None else float(bend_stiffness)
        twist = TWIST_STIFFNESS_PATIENT if twist_stiffness is None else float(twist_stiffness)
    _FEED_CFG = {
        "mode": mode,
        "bend_stiffness": bend,
        "twist_stiffness": twist,
    }


def clear_feed_mode() -> None:
    """Restore default free-space feed mode."""
    global _FEED_CFG
    _FEED_CFG = None


def active_feed_mode() -> str:
    if _FEED_CFG is None:
        return FEED_MODE_FREE_SPACE_PORTAL_SLIDE
    return str(_FEED_CFG["mode"])


def active_feed_stiffness() -> tuple[float, float]:
    if _FEED_CFG is None:
        return BEND_STIFFNESS_FREE_SPACE, TWIST_STIFFNESS_FREE_SPACE
    return float(_FEED_CFG["bend_stiffness"]), float(_FEED_CFG["twist_stiffness"])


@wp.kernel
def set_portal_joint_parent_z(
    joint_index: int,
    parent_z: float,
    joint_X_p: wp.array[wp.transform],
):
    tid = wp.tid()
    if tid > 0:
        return
    X = joint_X_p[joint_index]
    q = wp.transform_get_rotation(X)
    joint_X_p[joint_index] = wp.transform(wp.vec3(0.0, 0.0, parent_z), q)


def ensure_warp(device: str = DEVICE) -> None:
    global _WP_INIT
    if not _WP_INIT:
        wp.init()
        _WP_INIT = True
    wp.set_device(device)


def _finite(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).all())


def _pos(q: np.ndarray, i: int) -> np.ndarray:
    return np.asarray(q[i, :3], dtype=np.float64)


def _axis_proj(p: np.ndarray, portal: np.ndarray, axis: np.ndarray) -> float:
    return float(np.dot(p - portal, axis))


def distal_endpoint(q: np.ndarray, body_idx: int, axis: np.ndarray) -> np.ndarray:
    return _pos(q, body_idx) + axis * (0.5 * L_SEG)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = map(float, q[:4])
    qv = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (w * uv + uuv)


def body_axis(q: np.ndarray, body: int, fallback: np.ndarray) -> np.ndarray:
    a = quat_rotate(q[body, 3:7], np.array([0.0, 0.0, 1.0], dtype=np.float64))
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        fb = np.asarray(fallback, dtype=np.float64)
        return fb / (np.linalg.norm(fb) + 1e-15)
    return a / n


def apply_fraction_feed(model, portal_joint: int, baseline_z: float, L_fraction: float) -> None:
    """Free-space: slide portal joint_X_p by L_fraction."""
    wp.launch(
        kernel=set_portal_joint_parent_z,
        dim=1,
        inputs=[portal_joint, float(baseline_z + L_fraction), model.joint_X_p],
        device=DEVICE,
    )


def apply_portal_fixed(model, portal_joint: int, baseline_z: float) -> None:
    """Patient mode: keep portal passage fixed at baseline."""
    wp.launch(
        kernel=set_portal_joint_parent_z,
        dim=1,
        inputs=[portal_joint, float(baseline_z), model.joint_X_p],
        device=DEVICE,
    )


def apply_distal_fraction_feed(
    model,
    joints: list[int],
    portal_joint: int,
    baseline_z: float,
    n_active: int,
    L_fraction: float,
) -> int:
    """Patient mode: apply L_fraction on distal-most enabled rod joint; portal fixed."""
    apply_portal_fixed(model, portal_joint, baseline_z)
    if n_active < 3:
        return portal_joint
    feed_joint = joints[n_active - 2]
    wp.launch(
        kernel=set_portal_joint_parent_z,
        dim=1,
        inputs=[feed_joint, float(baseline_z + max(0.0, L_fraction)), model.joint_X_p],
        device=DEVICE,
    )
    return feed_joint


def apply_feed(
    model,
    portal_joint: int,
    baseline_z: float,
    L_fraction: float,
    *,
    joints: list[int] | None = None,
    n_active: int | None = None,
) -> None:
    """Dispatch fractional feed by active feed mode."""
    if active_feed_mode() == FEED_MODE_PATIENT_DISTAL_JOINT:
        if joints is None or n_active is None:
            raise ValueError("patient_distal_joint feed requires joints and n_active")
        apply_distal_fraction_feed(
            model, joints, portal_joint, baseline_z, int(n_active), float(L_fraction)
        )
    else:
        apply_fraction_feed(model, portal_joint, baseline_z, float(L_fraction))


def rebaseline_solver_poses(solver, state) -> None:
    """Prevent BDF velocity spike after discrete activation teleports (patient-contact mode)."""
    wp.copy(dest=solver.body_q_prev, src=state.body_q)
    snap = getattr(solver, "_coupling_body_q_prev_snapshot", None)
    if snap is not None:
        wp.copy(dest=snap, src=state.body_q)
    mask = getattr(solver, "_rigid_pose_rebaseline_mask", None)
    if mask is not None:
        mask.fill_(True)


def _restore_body_inertia(model, body: int, stored: dict) -> None:
    inv_m = model.body_inv_mass.numpy().copy()
    inv_I = model.body_inv_inertia.numpy().copy()
    mass = model.body_mass.numpy().copy()
    inertia = model.body_inertia.numpy().copy()
    s = stored[body]
    inv_m[body] = s["inv_mass"]
    mass[body] = s["mass"]
    inv_I[body] = s["inv_inertia"]
    inertia[body] = s["inertia"]
    model.body_inv_mass.assign(inv_m)
    model.body_inv_inertia.assign(inv_I)
    model.body_mass.assign(mass)
    model.body_inertia.assign(inertia)


def enable_body_along_parent(
    model,
    solver,
    state,
    bodies: list[int],
    joints: list[int],
    stored: dict,
    n_active: int,
    axis_fallback: np.ndarray,
    baseline_z: float,
) -> None:
    """Patient CRUISE pre-deploy: enable next body at parent + parent_axis * L_seg."""
    parent = bodies[n_active - 1]
    child = bodies[n_active]
    joint_act = joints[n_active - 1]
    q = state.body_q.numpy().copy()
    qd = state.body_qd.numpy().copy()
    parent_ax = body_axis(q, parent, axis_fallback)
    q[child, :3] = _pos(q, parent) + parent_ax * L_SEG
    q[child, 3:7] = q[parent, 3:7]
    qd[child] = 0.0
    state.body_q.assign(q)
    state.body_qd.assign(qd)
    _restore_body_inertia(model, child, stored)
    wp.launch(
        kernel=set_portal_joint_parent_z,
        dim=1,
        inputs=[joint_act, float(baseline_z), model.joint_X_p],
        device=DEVICE,
    )
    enabled = model.joint_enabled.numpy().copy()
    enabled[joint_act] = True
    model.joint_enabled.assign(enabled)
    solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)
    rebaseline_solver_poses(solver, state)


def apply_distal_tip_compliance(model, solver, joints: list[int], n_active: int) -> None:
    """Canonical soft distal tip (patient mode): the joint joining the current tip body (n_active-1) to its parent
    gets PATIENT_TIP_BEND_TWIST_STIFFNESS for its bend and twist DOFs; all other rod joints get the configured
    bend/twist stiffness. Stateless; call after the pre-deploy settle and after every activation."""
    if PATIENT_TIP_BEND_TWIST_STIFFNESS <= 0.0 or n_active < 2:
        return
    bend, twist = active_feed_stiffness()
    ke = model.joint_target_ke.numpy().copy()
    qs = model.joint_qd_start.numpy()
    tip_joint = joints[n_active - 2]
    for j in joints:
        a = int(qs[j])
        if j == tip_joint:
            ke[a + 2] = ke[a + 3] = PATIENT_TIP_BEND_TWIST_STIFFNESS      # DOF order: stretch, shear, bend, twist
        else:
            ke[a + 2], ke[a + 3] = bend, twist
    model.joint_target_ke.assign(ke)
    solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES)


def activate_next_tip_preserving(
    model,
    solver,
    state,
    bodies: list[int],
    joints: list[int],
    stored: dict,
    n_active: int,
    axis_fallback: np.ndarray,
    baseline_z: float,
) -> None:
    """Patient activation: tip-preserving discrete conversion + body_q_prev rebaseline."""
    parent = bodies[n_active - 2] if n_active >= 2 else bodies[0]
    tip = bodies[n_active - 1]
    child = bodies[n_active]
    feed_joint = joints[n_active - 2] if n_active >= 2 else joints[0]
    joint_act = joints[n_active - 1]

    q = state.body_q.numpy().copy()
    qd = state.body_qd.numpy().copy()
    tip_pos = _pos(q, tip).copy()
    tip_quat = q[tip, 3:7].copy()
    parent_ax = body_axis(q, parent, axis_fallback)

    q[tip, :3] = _pos(q, parent) + parent_ax * L_SEG
    q[tip, 3:7] = q[parent, 3:7]
    q[child, :3] = tip_pos
    q[child, 3:7] = tip_quat
    qd[tip] = 0.0
    qd[child] = 0.0
    state.body_q.assign(q)
    state.body_qd.assign(qd)

    _restore_body_inertia(model, child, stored)

    wp.launch(
        kernel=set_portal_joint_parent_z,
        dim=1,
        inputs=[feed_joint, float(baseline_z), model.joint_X_p],
        device=DEVICE,
    )
    wp.launch(
        kernel=set_portal_joint_parent_z,
        dim=1,
        inputs=[joint_act, float(baseline_z), model.joint_X_p],
        device=DEVICE,
    )
    enabled = model.joint_enabled.numpy().copy()
    enabled[joint_act] = True
    model.joint_enabled.assign(enabled)
    solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)
    rebaseline_solver_poses(solver, state)
    apply_distal_tip_compliance(model, solver, joints, n_active + 1)   # the activated child is the new tip


def activate_next(model, solver, state, bodies, joints, stored, n_active: int, axis: np.ndarray) -> None:
    """Free-space activation: place child along world insertion axis."""
    parent = bodies[n_active - 1]
    child = bodies[n_active]
    joint_act = joints[n_active - 1]

    q = state.body_q.numpy().copy()
    qd = state.body_qd.numpy().copy()
    p_parent = _pos(q, parent)
    quat_parent = np.asarray(q[parent, 3:7], dtype=np.float64)
    p_child = p_parent + axis * L_SEG
    q[child, :3] = p_child
    q[child, 3:7] = quat_parent
    qd[child] = 0.0
    state.body_q.assign(q)
    state.body_qd.assign(qd)

    inv_m = model.body_inv_mass.numpy().copy()
    inv_I = model.body_inv_inertia.numpy().copy()
    mass = model.body_mass.numpy().copy()
    inertia = model.body_inertia.numpy().copy()
    s = stored[child]
    inv_m[child] = s["inv_mass"]
    mass[child] = s["mass"]
    inv_I[child] = s["inv_inertia"]
    inertia[child] = s["inertia"]
    model.body_inv_mass.assign(inv_m)
    model.body_inv_inertia.assign(inv_I)
    model.body_mass.assign(mass)
    model.body_inertia.assign(inertia)

    enabled = model.joint_enabled.numpy().copy()
    enabled[joint_act] = True
    model.joint_enabled.assign(enabled)
    solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)


def activate_next_for_mode(
    model,
    solver,
    state,
    bodies,
    joints,
    stored,
    n_active: int,
    axis: np.ndarray,
    baseline_z: float,
) -> None:
    """Dispatch activation by active feed mode."""
    if active_feed_mode() == FEED_MODE_PATIENT_DISTAL_JOINT:
        activate_next_tip_preserving(
            model, solver, state, bodies, joints, stored, n_active, axis, baseline_z
        )
    else:
        activate_next(model, solver, state, bodies, joints, stored, n_active, axis)


def build_scene(
    *,
    portal: np.ndarray | None = None,
    axis: np.ndarray | None = None,
    root_at_portal: bool | None = None,
    enable_lumen_contact_assets: bool = False,
):
    """Build rod scene. Free-space return tuple is unchanged.

    If ``enable_lumen_contact_assets=True`` (patient contact regression), also
    returns ``(body_to_shape, wall_shape_id, contacts)``.
    """
    portal_v, axis_v, root_at = active_world_pose()
    if portal is not None:
        portal_v = np.asarray(portal, dtype=np.float64).reshape(3)
    if axis is not None:
        a = np.asarray(axis, dtype=np.float64).reshape(3)
        axis_v = a / float(np.linalg.norm(a))
    if root_at_portal is not None:
        root_at = bool(root_at_portal)

    bend_ke, twist_ke = active_feed_stiffness()
    patient_mode = active_feed_mode() == FEED_MODE_PATIENT_DISTAL_JOINT

    # free-space mode default: polyline nodes at portal + i*L_seg*axis (root COM at +0.5 L_seg).
    # Patient init: shift by -0.5 L_seg so root COM sits exactly on the portal.
    origin_shift = -0.5 * L_SEG if root_at else 0.0
    pts_np = [
        portal_v + (float(i) * L_SEG + origin_shift) * axis_v for i in range(NUM_SEGMENTS + 1)
    ]
    pts = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in pts_np]
    q_seg = wp.quat_between_vectors(
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(float(axis_v[0]), float(axis_v[1]), float(axis_v[2])),
    )
    quats = [q_seg] * NUM_SEGMENTS

    builder = newton.ModelBuilder()
    if enable_lumen_contact_assets:
        builder.default_shape_cfg.has_shape_collision = True
        builder.default_shape_cfg.has_particle_collision = False
        builder.default_shape_cfg.ke = SHAPE_KE
        builder.default_shape_cfg.kd = SHAPE_KD
        builder.default_shape_cfg.mu = SHAPE_MU
    else:
        builder.default_shape_cfg.has_shape_collision = False
        builder.default_shape_cfg.has_particle_collision = False

    bodies, joints = builder.add_rod(
        positions=pts,
        quaternions=quats,
        radius=RADIUS,
        stretch_stiffness=1.0e6,
        bend_stiffness=bend_ke,
        twist_stiffness=twist_ke,
        bend_damping=1.0,
        twist_damping=1.0,
        stretch_damping=10.0,
        label="stage2d" if not patient_mode else "patient_feed",
        body_frame_origin="com",
        wrap_in_articulation=True,
    )

    wall_shape_id = -1
    if enable_lumen_contact_assets:
        wall_shape_id = int(
            builder.add_shape_sphere(
                body=-1, radius=0.001, cfg=builder.default_shape_cfg, label="lumen_wall_proxy"
            )
        )

    root = int(bodies[0])
    builder.body_mass[root] = 0.0
    builder.body_inv_mass[root] = 0.0
    builder.body_inertia[root] = wp.mat33(0.0)
    builder.body_inv_inertia[root] = wp.mat33(0.0)

    builder.color(balance_colors=False)
    model = builder.finalize(device=DEVICE)
    model.set_gravity((0.0, 0.0, 0.0))
    if enable_lumen_contact_assets:
        model.rigid_contact_max = CONTACT_MAX

    bodies = [int(b) for b in bodies]
    joints = [int(j) for j in joints]
    portal_joint = joints[0]
    baseline_z = 0.5 * L_SEG

    inv_m = model.body_inv_mass.numpy().copy()
    inv_I = model.body_inv_inertia.numpy().copy()
    mass = model.body_mass.numpy().copy()
    inertia = model.body_inertia.numpy().copy()

    stored = {}
    for b in bodies[2:]:
        stored[b] = {
            "inv_mass": float(inv_m[b]),
            "mass": float(mass[b]),
            "inv_inertia": inv_I[b].copy(),
            "inertia": inertia[b].copy(),
        }

    for b in bodies[2:]:
        inv_m[b] = 0.0
        inv_I[b] = np.zeros((3, 3), dtype=np.float32)
        mass[b] = 0.0
        inertia[b] = np.zeros((3, 3), dtype=np.float32)
    model.body_inv_mass.assign(inv_m)
    model.body_inv_inertia.assign(inv_I)
    model.body_mass.assign(mass)
    model.body_inertia.assign(inertia)

    enabled = model.joint_enabled.numpy().copy()
    for j in joints[1:]:
        enabled[j] = False
    model.joint_enabled.assign(enabled)

    if patient_mode:
        # Portal and all joints start at baseline (no portal slide).
        for j in joints:
            wp.launch(
                kernel=set_portal_joint_parent_z,
                dim=1,
                inputs=[j, float(baseline_z), model.joint_X_p],
                device=DEVICE,
            )

    solver = newton.solvers.SolverVBD(
        model,
        iterations=PATIENT_VBD_ITERATIONS if patient_mode else FREE_SPACE_VBD_ITERATIONS,
        rigid_compliant_alm=True,
    )
    solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES)

    base = (model, solver, bodies, joints, root, portal_joint, baseline_z, stored, portal_v, axis_v)
    if not enable_lumen_contact_assets:
        return base

    shape_body = model.shape_body.numpy()
    body_to_shape: dict[int, int] = {}
    for si, bi in enumerate(shape_body):
        if int(bi) >= 0 and int(bi) not in body_to_shape:
            body_to_shape[int(bi)] = int(si)
    contacts = Contacts(CONTACT_MAX, 0, device=DEVICE, requested_attributes={"force"})
    return (*base, body_to_shape, wall_shape_id, contacts)


def run_episode(
    *,
    velocity_mps: float,
    num_frames: int = NUM_FRAMES,
    action_name: str | None = None,
) -> dict[str, Any]:
    """Run one fresh free-space episode at the given insertion velocity (SI m/s).

    Always uses portal-slide feed/activation regardless of ``configure_feed_mode``,
    matching validated free-space semantics. Patient episodes use a dedicated path.
    """
    ensure_warp(DEVICE)
    # Isolate free-space regression from any ambient patient feed_mode.
    prev_feed = None if _FEED_CFG is None else dict(_FEED_CFG)
    configure_feed_mode(FEED_MODE_FREE_SPACE_PORTAL_SLIDE)
    try:
        (
            model,
            solver,
            bodies,
            joints,
            root,
            portal_joint,
            baseline_z,
            stored,
            portal,
            axis,
        ) = build_scene()
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        q_init = state_0.body_q.numpy().copy()
        root0 = _pos(q_init, root).copy()
        tip0 = distal_endpoint(q_init, bodies[1], axis)
        geo0 = _axis_proj(tip0, portal, axis)
        rod_axis = _pos(q_init, bodies[1]) - _pos(q_init, bodies[0])
        rod_axis_n = float(np.linalg.norm(rod_axis))
        rod_axis_u = (rod_axis / rod_axis_n) if rod_axis_n > 1e-12 else axis.copy()
        reservoir_starts = {b: _pos(q_init, b).copy() for b in bodies[2:]}
        reservoir_max = {b: 0.0 for b in bodies[2:]}

        init_fingerprint = {
            "root_pos": root0.tolist(),
            "body_positions": [_pos(q_init, b).tolist() for b in bodies],
            "n_active": 2,
            "L_total": 0.0,
            "joint0_parent_z": float(model.joint_X_p.numpy()[portal_joint][2]),
            "portal_m": portal.tolist(),
            "axis": axis.tolist(),
            "rod_axis_unit": rod_axis_u.tolist(),
            "active_tip0_m": tip0.tolist(),
            "feed_mode": FEED_MODE_FREE_SPACE_PORTAL_SLIDE,
        }

        L_total = 0.0
        n_completed = 0
        n_active = 2
        activation_frames: list[int] = []
        max_old_jump = 0.0
        max_inv_err = 0.0
        double_counting = False
        root_max = 0.0
        all_finite = True
        elapsed_s = num_frames * DT_FRAME

        for frame in range(num_frames):
            for _sub in range(SIM_SUBSTEPS):
                L_total += float(velocity_mps) * SIM_DT
                L_fraction = L_total - n_completed * L_SEG
                max_inv_err = max(max_inv_err, abs(L_total - (n_completed * L_SEG + L_fraction)))

                while L_fraction >= L_SEG - 1e-12 and n_active < len(bodies):
                    q_pre = state_0.body_q.numpy().copy()
                    old_active = bodies[:n_active]
                    old_pos = {b: _pos(q_pre, b).copy() for b in old_active}

                    n_completed += 1
                    L_fraction = L_total - n_completed * L_SEG
                    apply_fraction_feed(model, portal_joint, baseline_z, max(0.0, L_fraction))
                    activate_next(model, solver, state_0, bodies, joints, stored, n_active, axis)
                    state_1.body_q.assign(state_0.body_q)
                    state_1.body_qd.assign(state_0.body_qd)

                    q_post = state_0.body_q.numpy().copy()
                    jump = max(float(np.linalg.norm(_pos(q_post, b) - old_pos[b])) for b in old_active)
                    max_old_jump = max(max_old_jump, jump)

                    Xp_z = float(model.joint_X_p.numpy()[portal_joint][2])
                    expected_z = baseline_z + max(0.0, L_fraction)
                    if abs(Xp_z - expected_z) > 1e-5:
                        double_counting = True
                    if abs(Xp_z - (baseline_z + L_total)) < 1e-6 and L_total > L_SEG:
                        double_counting = True

                    n_active += 1
                    activation_frames.append(frame)

                L_fraction = L_total - n_completed * L_SEG
                max_inv_err = max(max_inv_err, abs(L_total - (n_completed * L_SEG + L_fraction)))
                apply_fraction_feed(model, portal_joint, baseline_z, max(0.0, L_fraction))

                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, SIM_DT)
                state_0, state_1 = state_1, state_0

            q = state_0.body_q.numpy()
            qd = state_0.body_qd.numpy()
            if not (_finite(q) and _finite(qd)):
                all_finite = False
            root_max = max(root_max, float(np.linalg.norm(_pos(q, root) - root0)))
            for b in bodies[n_active:]:
                reservoir_max[b] = max(
                    reservoir_max[b], float(np.linalg.norm(_pos(q, b) - reservoir_starts[b]))
                )

        q_final = state_0.body_q.numpy()
        tip = bodies[n_active - 1]
        tip_final = distal_endpoint(q_final, tip, axis)
        physical_prog = _axis_proj(tip_final, portal, axis) - geo0
        tip_disp = tip_final - tip0
        tip_disp_n = float(np.linalg.norm(tip_disp))
        tip_disp_u = (tip_disp / tip_disp_n) if tip_disp_n > 1e-12 else np.zeros(3)
        direction_cosine = float(np.dot(tip_disp_u, axis)) if tip_disp_n > 1e-12 else 1.0
        direction_error_deg = float(np.degrees(np.arccos(np.clip(direction_cosine, -1.0, 1.0))))
        L_fraction_final = L_total - n_completed * L_SEG
        expected_dL = float(velocity_mps) * elapsed_s
        length_error = abs(L_total - expected_dL)
        expected_activations = min(int(np.floor(L_total / L_SEG + 1e-12)), len(bodies) - 2)
        reservoir_pre = float(max(reservoir_max.values())) if reservoir_max else 0.0
        stable = all_finite and float(np.nanmax(np.abs(q_final[:, :3]))) < 10.0
        remainder_ok = (-1e-9 <= L_fraction_final < L_SEG + 1e-9) and max_inv_err < 1e-12
        activation_ok = len(activation_frames) == expected_activations and n_active == 2 + len(activation_frames)

        name = action_name or f"v={velocity_mps}"
        local_pass = bool(
            length_error < 1e-9
            and remainder_ok
            and activation_ok
            and not double_counting
            and root_max < 1e-6
            and reservoir_pre < 1e-6
            and max_old_jump < 1e-6
            and all_finite
            and stable
        )
        if abs(float(velocity_mps)) < 1e-15:
            local_pass = local_pass and abs(L_total) < 1e-15 and len(activation_frames) == 0

        return {
            "action": name,
            "velocity_mps": float(velocity_mps),
            "internal_velocity_mps": float(velocity_mps),
            "elapsed_s": elapsed_s,
            "expected_delta_L_m": expected_dL,
            "measured_delta_L_m": float(L_total),
            "length_error_m": float(length_error),
            "activation_count": len(activation_frames),
            "activation_frames": activation_frames,
            "expected_activation_count_from_L_total": expected_activations,
            "initial_active_count": 2,
            "final_active_count": n_active,
            "n_completed_segments": n_completed,
            "final_L_fraction_m": float(L_fraction_final),
            "max_length_invariant_error_m": float(max_inv_err),
            "physical_progression_m": float(physical_prog),
            "tip_distal_progression_m": float(physical_prog),
            "progression_vector_m": tip_disp.tolist(),
            "direction_cosine_similarity": direction_cosine,
            "direction_error_deg": direction_error_deg,
            "bookkeeping_progression_m": float(L_SEG + L_total),
            "root_max_delta_m": float(root_max),
            "root_position_m": root0.tolist(),
            "rod_axis_unit": rod_axis_u.tolist(),
            "world_portal_m": portal.tolist(),
            "world_axis": axis.tolist(),
            "max_reservoir_pre_activation_displacement_m": reservoir_pre,
            "max_old_active_handoff_jump_m": float(max_old_jump),
            "double_counting": double_counting,
            "remainder_semantics_valid": remainder_ok,
            "activation_count_consistent": activation_ok,
            "finite": all_finite,
            "stable": stable,
            "topology_rebuilt": False,
            "init_fingerprint": init_fingerprint,
            "feed_mode": FEED_MODE_FREE_SPACE_PORTAL_SLIDE,
            "pass_local": local_pass,
        }
    finally:
        if prev_feed is None:
            clear_feed_mode()
        else:
            configure_feed_mode(
                prev_feed["mode"],
                bend_stiffness=prev_feed["bend_stiffness"],
                twist_stiffness=prev_feed["twist_stiffness"],
            )
