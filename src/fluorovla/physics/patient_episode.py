"""Canonical single-patient Newton episode (portal-fed insertion + SDF lumen contact), decision-stepped.

This is the one code path for Teacher rollouts and demonstration generation. It uses the canonical engine settings
(fluorovla.physics.engine: PATIENT_VBD_ITERATIONS, PATIENT_TIP_BEND_TWIST_STIFFNESS, NUM_SEGMENTS) with NO rollout-side overrides.
All patient geometry enters through ``PatientConfig``.

Timing: 60 Hz physics, 5 substeps/frame, 6 frames (0.1 s) per policy decision.
inserted_length_m  = L_total = integral of the EXECUTED insert velocity over the past substeps (portal feed encoder);
                     read at the START of a decision it contains only past executed actions.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from fluorovla.physics.engine import (
    FEED_MODE_PATIENT_DISTAL_JOINT, L_SEG, NUM_SEGMENTS, RADIUS, SIM_DT, SIM_SUBSTEPS, _finite, _pos,
    activate_next_tip_preserving, apply_distal_fraction_feed, apply_distal_tip_compliance, apply_portal_fixed,
    build_scene, configure_feed_mode, configure_world_pose, enable_body_along_parent, ensure_warp, quat_rotate)
from fluorovla.physics.lumen_sdf import LumenSDF, generate_lumen_contacts
from fluorovla.physics.patient_config import PatientConfig
from fluorovla.physics.polyline import body_q_to_polyline_m


BEND_STIFFNESS = 1.0e2            # validated target-contract bend/twist stiffness of the patient rod (unchanged)
SETTLE_FRAMES = 30
PREDEPLOY_ACTIVE = 3
FRAMES_PER_DECISION = 6
SAMPLE_FRAC = (0.0, 0.25, 0.5, 0.75, 1.0)
CATASTROPHIC_VEL_MPS = 2.0
_N_ACTIVE_INIT = 2


def measure_outside(sdf, q, bodies, n_active, root, axis):
    """(fraction of axis samples beyond the wall surface, max SDF excess [m]); identical to the audited function."""
    pts = []
    for b in bodies[:n_active]:
        if b == root:
            continue
        com = _pos(q, b)
        ax_b = quat_rotate(q[b, 3:7], np.array([0.0, 0.0, 1.0]))
        n = float(np.linalg.norm(ax_b))
        ax = (ax_b / n) if n > 1e-12 else np.asarray(axis, dtype=np.float64)
        for f in SAMPLE_FRAC:
            pts.append(com + (f - 0.5) * L_SEG * ax)
    if not pts:
        return 0.0, 0.0
    ph, _, _ = sdf.query(np.stack(pts, axis=0).astype(np.float32))
    return float(np.mean(ph > 0.0)), float(max(0.0, float(np.max(ph))))


class PatientEpisode:
    def __init__(self, cfg: PatientConfig) -> None:
        self.cfg = cfg
        assert NUM_SEGMENTS == 7
        ensure_warp()
        self.sdf = LumenSDF(cfg)
        self.mesh = self.sdf.mesh
        configure_world_pose(portal_m=cfg.portal, axis=cfg.tangent, root_at_portal=True)
        configure_feed_mode(FEED_MODE_PATIENT_DISTAL_JOINT, bend_stiffness=BEND_STIFFNESS, twist_stiffness=BEND_STIFFNESS)
        (self.model, self.solver, self.bodies, self.joints, self.root, self.portal_joint, self.baseline_z, self.stored,
         self.portal, self.axis, self.body_to_shape, self.wall_shape_id, self.contacts) = build_scene(enable_lumen_contact_assets=True)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.n_active = _N_ACTIVE_INIT
        for _ in range(SETTLE_FRAMES * SIM_SUBSTEPS):
            apply_portal_fixed(self.model, self.portal_joint, self.baseline_z)
            self._contact_step()
            self._swap()
        while self.n_active < PREDEPLOY_ACTIVE:
            enable_body_along_parent(self.model, self.solver, self.state_0, self.bodies, self.joints, self.stored,
                                     self.n_active, self.axis, self.baseline_z)
            self.n_active += 1
            self.state_1.body_q.assign(self.state_0.body_q)
            self.state_1.body_qd.assign(self.state_0.body_qd)
        for _ in range(SETTLE_FRAMES * SIM_SUBSTEPS):
            apply_portal_fixed(self.model, self.portal_joint, self.baseline_z)
            apply_distal_fraction_feed(self.model, self.joints, self.portal_joint, self.baseline_z, self.n_active, 0.0)
            self._contact_step()
            self._swap()
        apply_distal_tip_compliance(self.model, self.solver, self.joints, self.n_active)    # canonical soft tip
        self.L_total = 0.0
        self._n_completed = 0
        self.frame = 0
        self.activation_events: list[dict[str, Any]] = []
        self._root0 = _pos(self.state_0.body_q.numpy(), self.root).copy()
        self.root_max = 0.0
        self.all_finite = True
        self.max_viol_m = 0.0
        self.max_outside_frac = 0.0

    # -- internals ---------------------------------------------------------------------------------------------
    def _swap(self):
        self.state_0, self.state_1 = self.state_1, self.state_0

    def _contact_step(self):
        generate_lumen_contacts(sdf=self.sdf, contacts=self.contacts, state=self.state_0, bodies=self.bodies,
                                       n_active=self.n_active, root=self.root, body_to_shape=self.body_to_shape,
                                       wall_shape_id=self.wall_shape_id)
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, SIM_DT)

    # -- accepted physical state (what observations/state are built from) ---------------------------------------
    def polyline_solver_m(self) -> np.ndarray:
        return body_q_to_polyline_m(self.state_0.body_q.numpy(), self.bodies, self.n_active, fallback_axis=self.cfg.tangent)

    @property
    def inserted_length_m(self) -> float:
        return float(self.L_total)

    def tip_ct_mm(self) -> np.ndarray:
        return (self.polyline_solver_m()[-1] + self.cfg.ct_offset) * 1000.0

    # -- one policy decision ------------------------------------------------------------------------------------
    def step_decision(self, vel_mps: float) -> dict[str, Any]:
        viol = out = lin = 0.0
        for _f in range(FRAMES_PER_DECISION):
            for _sub in range(SIM_SUBSTEPS):
                self.L_total += float(vel_mps) * SIM_DT
                L_fraction = self.L_total - self._n_completed * L_SEG
                while L_fraction >= L_SEG - 1e-12 and self.n_active < len(self.bodies):
                    self._n_completed += 1
                    L_fraction = self.L_total - self._n_completed * L_SEG
                    activate_next_tip_preserving(self.model, self.solver, self.state_0, self.bodies, self.joints, self.stored,
                                                 self.n_active, self.axis, self.baseline_z)
                    apply_distal_fraction_feed(self.model, self.joints, self.portal_joint, self.baseline_z,
                                               self.n_active + 1, max(0.0, L_fraction))
                    self.state_1.body_q.assign(self.state_0.body_q)
                    self.state_1.body_qd.assign(self.state_0.body_qd)
                    self.n_active += 1
                    self.activation_events.append({"frame": self.frame, "n_active_after": self.n_active,
                                                   "inserted_length_m": self.L_total})
                    break
                L_fraction = self.L_total - self._n_completed * L_SEG
                apply_distal_fraction_feed(self.model, self.joints, self.portal_joint, self.baseline_z, self.n_active,
                                           max(0.0, L_fraction))
                self._contact_step()
                self._swap()
                qd = self.state_0.body_qd.numpy()
                q = self.state_0.body_q.numpy()
                for b in self.bodies[:self.n_active]:
                    if b == self.root:
                        continue
                    lin = max(lin, float(np.linalg.norm(qd[b, :3])))
                    if not (_finite(q[b]) and _finite(qd[b])):
                        self.all_finite = False
                of, vi = measure_outside(self.sdf, q, self.bodies, self.n_active, self.root, self.axis)
                out = max(out, of)
                viol = max(viol, vi)
            self.frame += 1
        q = self.state_0.body_q.numpy()
        root_d = float(np.linalg.norm(_pos(q, self.root) - self._root0))
        self.root_max = max(self.root_max, root_d)
        self.max_viol_m = max(self.max_viol_m, viol)
        self.max_outside_frac = max(self.max_outside_frac, out)
        tip_speed = float(np.linalg.norm(self.state_0.body_qd.numpy()[self.bodies[self.n_active - 1], :3]))
        return {"physics_frame": self.frame, "n_active": self.n_active, "axis_violation_mm": viol * 1000.0,
                "outside_frac": out, "max_body_speed_mps": lin, "tip_speed_mps": tip_speed, "root_disp_m": root_d,
                "finite": bool(self.all_finite and np.isfinite(self.polyline_solver_m()).all())}
