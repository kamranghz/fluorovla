"""Read-only observer for the initial-state audit: records the physical state of the catheter at the initialization checkpoints of ``PatientEpisode`` and tracks which
axis sample triggers the lumen-violation test. It reads Newton state and queries the lumen SDF; it never writes simulation state.

Checkpoints (see PatientEpisode):
  build_scene        rod as created by the engine, before any step (2 active bodies)
  after_settle1      after the first settle phase (30 frames, 2 active bodies)
  after_body_enable  third body enabled (before the second settle phase)
  after_settle2      after the second settle phase (3 active bodies)
  ready              start of decision 0 (identical to after_settle2 plus the soft-tip compliance setting)
  decision_substep   every substep of the decisions (violation tracking)
Not directly available in the Newton rod API: strain energy and per-joint twist. The closest reliable measures recorded are the relative rotation angle between
consecutive bodies (bend + twist combined) and the body speeds.
"""
from __future__ import annotations

import math

import numpy as np

import fluorovla.task_contract as tc
from fluorovla.physics.engine import L_SEG, RADIUS
from fluorovla.physics.patient_episode import axis_sample_points

R_MM = RADIUS * 1000.0


def _rel_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def snapshot(ep, proj: tc.PathProjector) -> dict:
    """Full state description of the active rod at the current instant."""
    q = ep.state_0.body_q.numpy()
    qd = ep.state_0.body_qd.numpy()
    off = ep.cfg.ct_offset
    bodies = ep.bodies[: ep.n_active]
    pts_c = np.array([q[b, :3] for b in bodies], float)
    phi_c, _, _ = ep.sdf.query(pts_c.astype(np.float32))
    out_bodies = []
    prev_q = None
    for i, b in enumerate(bodies):
        com = q[b, :3].astype(float)
        com_ct = (com + off) * 1000.0
        s, d, rad = proj(com_ct)
        rec = {"index": i, "body": int(b), "is_root": bool(b == ep.root), "com_solver_m": com.tolist(), "com_ct_mm": com_ct.tolist(), "dist_to_path_mm": d, "path_arclength_mm": s,
               "local_twin_radius_mm": rad, "phi_com_mm": float(phi_c[i] * 1000.0), "clearance_mm": float(-phi_c[i] * 1000.0 - R_MM),
               "speed_first3_mps": float(np.linalg.norm(qd[b, :3])), "speed_last3": float(np.linalg.norm(qd[b, 3:6])),
               "rel_rotation_to_previous_body_deg": None if prev_q is None else _rel_angle_deg(prev_q, q[b, 3:7].astype(float))}
        prev_q = q[b, 3:7].astype(float)
        out_bodies.append(rec)
    samples = axis_sample_points(q, ep.bodies, ep.n_active, ep.root, ep.axis)
    sp = np.array([p for _b, _f, p in samples], float)
    sphi = ep.sdf.query(sp.astype(np.float32))[0] * 1000.0 if len(sp) else np.zeros(0)
    srecs = []
    for (b, f, p), ph in zip(samples, sphi):
        c = (p + off) * 1000.0
        srecs.append({"body": int(b), "fraction": float(f), "point_ct_mm": c.tolist(), "phi_mm": float(ph), "excess_mm": float(max(0.0, ph)), "dist_to_path_mm": proj(c)[1]})
    k = int(np.argmax(sphi)) if len(sphi) else -1
    # the tip is the distal end of the last body (polyline end)
    tip_ct = np.asarray(ep.tip_ct_mm(), float)
    ts, td, tr = proj(tip_ct)
    moving = [r for r in out_bodies if not r["is_root"]]
    return {"n_active": int(ep.n_active), "bodies": out_bodies, "axis_samples": srecs,
            "max_axis_violation_mm": float(max(0.0, sphi.max())) if len(sphi) else 0.0, "n_samples_outside": int(np.sum(sphi > 0)), "n_samples": int(len(sphi)),
            "worst_sample": (srecs[k] if k >= 0 else None), "min_clearance_mm": float(min(r["clearance_mm"] for r in moving)) if moving else None,
            "max_body_to_path_distance_mm": float(max(r["dist_to_path_mm"] for r in moving)) if moving else None,
            "max_body_speed_first3_mps": float(max(r["speed_first3_mps"] for r in moving)) if moving else None,
            "tip_ct_mm": tip_ct.tolist(), "tip_dist_to_path_mm": td, "tip_path_arclength_mm": ts, "tip_extremity_mm": ts + R_MM,
            "root_position_solver_m": q[ep.root, :3].astype(float).tolist(), "max_rel_rotation_deg": max([r["rel_rotation_to_previous_body_deg"] for r in out_bodies if r["rel_rotation_to_previous_body_deg"] is not None], default=0.0)}


class InitialStateObserver:
    """Callable passed as ``observer`` to PatientEpisode / run_physics_episode."""

    CHECKPOINTS = ("build_scene", "after_settle1", "after_body_enable", "after_settle2", "ready")

    def __init__(self, proj: tc.PathProjector, keep_first_decisions: int = 3):
        self.proj = proj
        self.snapshots: dict[str, dict] = {}
        self.settle_trace: dict[str, list] = {"settle1": [], "settle2": []}
        self.violation_track: list[dict] = []
        self.keep_first_decisions = keep_first_decisions
        self._ring: list[dict] = []
        self._decision_substeps = 0
        self.root0 = None
        self.analytic_init: dict | None = None

    def __call__(self, stage: str, ep, info: dict) -> None:
        if stage in self.CHECKPOINTS:
            snap = snapshot(ep, self.proj)
            if stage == "build_scene":
                self.root0 = snap["root_position_solver_m"]
                self.analytic_init = self._analytic(ep, snap)
            snap["root_displacement_mm"] = float(np.linalg.norm(np.array(snap["root_position_solver_m"]) - np.array(self.root0)) * 1000.0)
            self.snapshots[stage] = snap
        elif stage in ("settle1_substep", "settle2_substep") and info["k"] % 5 == 4:       # one record per physics frame
            s = snapshot(ep, self.proj)
            self.settle_trace["settle1" if stage.startswith("settle1") else "settle2"].append(
                {"frame": info["k"] // 5, "n_active": s["n_active"], "max_axis_violation_mm": s["max_axis_violation_mm"], "max_body_speed_first3_mps": s["max_body_speed_first3_mps"],
                 "max_body_to_path_distance_mm": s["max_body_to_path_distance_mm"], "min_clearance_mm": s["min_clearance_mm"]})
        elif stage == "decision_substep":
            self._decision_substeps += 1
            dec = (self._decision_substeps - 1) // 30
            s = snapshot(ep, self.proj)
            w = s["worst_sample"] or {}
            rec = {"decision": dec, "frame": info["frame"], "sub": info["sub"], "engine_violation_mm": info["violation_m"] * 1000.0,
                   "audit_max_axis_violation_mm": s["max_axis_violation_mm"], "worst_body": w.get("body"), "worst_fraction": w.get("fraction"),
                   "worst_point_ct_mm": w.get("point_ct_mm"), "worst_phi_mm": w.get("phi_mm"), "tip_dist_to_path_mm": s["tip_dist_to_path_mm"], "n_active": s["n_active"]}
            if dec < self.keep_first_decisions or rec["engine_violation_mm"] > 0.5:
                self.violation_track.append(rec)
            self._ring.append(rec)
            if len(self._ring) > 60:
                self._ring.pop(0)

    @staticmethod
    def _analytic(ep, snap: dict) -> dict:
        """Compare the engine's initial body placement with the analytic straight rod portal + k * L_SEG * tangent."""
        t = ep.cfg.tangent
        com = np.array([b["com_solver_m"] for b in snap["bodies"]], float)
        k = np.arange(len(com))
        proj_len = com @ t
        lateral = np.linalg.norm(com - np.outer(proj_len, t), axis=1)
        return {"axial_offset_of_each_body_com_m": proj_len.tolist(), "expected_spacing_m": L_SEG, "lateral_offset_from_tangent_line_mm": (lateral * 1000.0).tolist(),
                "max_lateral_offset_mm": float(lateral.max() * 1000.0)}

    def summary(self) -> dict:
        ready = self.snapshots.get("ready", {})
        return {"checkpoints": self.snapshots, "settle_trace": self.settle_trace, "violation_track": self.violation_track, "last_60_substeps": self._ring, "analytic_initialization": self.analytic_init,
                "post_settle": {"max_axis_violation_mm": ready.get("max_axis_violation_mm"), "min_clearance_mm": ready.get("min_clearance_mm"),
                                "max_body_to_path_distance_mm": ready.get("max_body_to_path_distance_mm"), "tip_dist_to_path_mm": ready.get("tip_dist_to_path_mm"),
                                "max_body_speed_first3_mps": ready.get("max_body_speed_first3_mps"), "root_displacement_mm": ready.get("root_displacement_mm"),
                                "max_rel_rotation_deg": ready.get("max_rel_rotation_deg")},
                "settle_max_violation_mm": max([r["max_axis_violation_mm"] for ph in self.settle_trace.values() for r in ph], default=None),
                "settle_max_speed_mps": max([r["max_body_speed_first3_mps"] for ph in self.settle_trace.values() for r in ph if r["max_body_speed_first3_mps"] is not None], default=None)}
