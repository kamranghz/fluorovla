"""Run one closed-loop Privileged-Teacher episode on the canonical patient physics (one decision = 0.1 s = 6 physics frames).

Per decision t (causal order):
   1. the physical state accepted after decision t-1 (polyline, inserted length) is recorded  -> image_t / state_t source
   2. previous_insert_velocity_mps_t = action executed at t-1 (0.0 at t=0)
   3. the Privileged Teacher picks action_t from the privileged tip extremity of THIS accepted state
   4. action_t is executed for 6 physics frames (0.1 s)
The episode ends when the success condition (landing-zone window + dwell) is met, or on overshoot, lumen violation (> 1.5 mm), physics failure or the
timeout ``T_MAX`` stored with the episode configuration. After the success condition is first satisfied the SAME simulation continues for
``post_success_hold`` more decisions (the Teacher rule is unchanged and holds inside the zone).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

import fluorovla.task_contract as tc
from fluorovla.action import CRUISE, HOLD, SLOW
from fluorovla.physics import engine as eng
from fluorovla.physics.patient_config import PatientConfig
from fluorovla.physics.patient_episode import PatientEpisode
from fluorovla.teacher.privileged_teacher import TeacherState, teacher_action

NAMED = {"HOLD": HOLD, "SLOW": SLOW, "CRUISE": CRUISE}
R_C = eng.RADIUS * 1000.0


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    files = ["physics/episode_runner.py", "physics/patient_episode.py", "physics/engine.py", "physics/lumen_sdf.py", "physics/polyline.py", "physics/patient_config.py",
             "teacher/privileged_teacher.py", "task_contract.py", "action.py"]
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}


def run_physics_episode(cfg: PatientConfig, out_dir: str | Path, tag: str, post_success_hold: int = 0,
                        action_override: Callable[[str, float, TeacherState], tuple[str, float]] | None = None) -> dict:
    """Run the episode and write ``episode_physics_<tag>.json`` / ``.npz`` into ``out_dir``. ``action_override`` (optional) may replace the Teacher decision
    (used for paired-speed experiments); it is None for the canonical Teacher."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    land = json.load(open(cfg.landing_zone_path))
    zc = np.load(cfg.path_npz_path)
    proj = tc.PathProjector(zc["pts"], zc["rad"])
    ZONE = land["landing_zone_extremity_mm"]
    T_MAX = land["timeout"]["T_MAX_decisions"]
    POST_HOLD = int(post_success_hold)

    ep = PatientEpisode(cfg)
    t0 = time.time()

    def privileged(ep):
        poly = ep.polyline_solver_m()
        ct = ep.tip_ct_mm()
        s, d, rad = proj(ct)
        return poly, s, d, rad

    rows, polys, status = [], [], None
    success_idx = None
    prev_v = 0.0
    dwell = 0
    poly, s, d, rad = privileged(ep)
    for t in range(T_MAX + POST_HOLD):
        if success_idx is None and t >= T_MAX:                           # the timeout applies to the approach only
            break
        poly, s, d, rad = privileged(ep)                                  # accepted state at the start of decision t
        s_ext = s + R_C
        polys.append(poly)
        state = TeacherState(tip_extremity_mm=s_ext, zone_near_mm=ZONE[0], zone_far_mm=ZONE[1])
        name, v = teacher_action(state)
        if action_override is not None:
            name, v = action_override(name, v, state)
        action = NAMED[name]
        assert abs(action.insert_velocity_mps - v) < 1e-12
        row = {"decision": t, "time_s": t * 0.1, "inserted_length_m": ep.inserted_length_m, "previous_insert_velocity_mps": prev_v,
               "action": name, "insert_velocity_mps": action.insert_velocity_mps, "n_active_start": ep.n_active,
               "physics_frame_start": ep.frame, "phase": "post_success_hold" if success_idx is not None else "approach", "priv_tip_extremity_mm": s_ext,
               "priv_tip_polyline_s_mm": s, "priv_tip_dist_centerline_mm": d, "priv_local_radius_mm": rad}
        info = ep.step_decision(action.insert_velocity_mps)               # execute action_t
        prev_v = action.insert_velocity_mps                               # becomes previous_insert_velocity at t+1
        poly_n, s_n, d_n, rad_n = privileged(ep)
        s_ext_n = s_n + R_C
        ds = abs(s_ext_n - s_ext)
        in_win = abs(s_ext_n - (ZONE[0] + ZONE[1]) / 2) <= tc.EPS_MM
        dwell = dwell + 1 if (in_win and ds < tc.DWELL_STEP_MM) else 0
        row.update({"axis_violation_mm": info["axis_violation_mm"], "outside_frac": info["outside_frac"],
                    "tip_speed_mps": info["tip_speed_mps"], "root_disp_m": info["root_disp_m"], "priv_tip_extremity_end_mm": s_ext_n,
                    "n_active_end": info["n_active"], "in_zone_end": bool(ZONE[0] <= s_ext_n <= ZONE[1]), "dwell_run": dwell})
        rows.append(row)
        if (not info["finite"]) or info["max_body_speed_mps"] > tc.CATASTROPHIC_VEL_MPS or ep.root_max > tc.ROOT_MOVE_TOL_M:
            status = "PHYSICS_FAILURE"
            break
        if info["axis_violation_mm"] > tc.LUMEN_DELTA_MM:
            status = "LUMEN_VIOLATION"
            break
        if s_ext_n > ZONE[1]:
            status = "OVERSHOOT"
            break
        if success_idx is None and dwell >= tc.DWELL_STEPS:
            success_idx = t                                               # success condition first satisfied after decision t
            status = "SUCCESS"
        if success_idx is not None:
            if t > success_idx and action_override is None:
                assert name == "HOLD", name                              # inside the zone the unchanged Teacher rule holds
            if t - success_idx >= POST_HOLD:                             # exactly POST_HOLD decisions after the success decision
                break
    if status is None:
        status = "TIMEOUT"
    poly_final, s_f, d_f, rad_f = privileged(ep)
    summary = {
        "tag": tag, "status": status, "decisions": len(rows), "success_decision_index": success_idx,
        "post_success_hold_decisions": len(rows) - 1 - success_idx if success_idx is not None else 0, "post_success_hold_requested": POST_HOLD,
        "elapsed_wall_s": time.time() - t0,
        "final_tip_extremity_mm": rows[-1]["priv_tip_extremity_end_mm"], "landing_zone_extremity_mm": ZONE, "timeout_decisions": T_MAX,
        "max_axis_violation_mm": max(r["axis_violation_mm"] for r in rows), "max_outside_fraction": max(r["outside_frac"] for r in rows),
        "max_root_displacement_m": ep.root_max, "all_finite": bool(ep.all_finite), "active_bodies_final": ep.n_active,
        "activation_events": ep.activation_events, "final_inserted_length_m": ep.inserted_length_m,
        "engine_settings": {"NUM_SEGMENTS": eng.NUM_SEGMENTS, "PATIENT_VBD_ITERATIONS": eng.PATIENT_VBD_ITERATIONS,
                            "PATIENT_TIP_BEND_TWIST_STIFFNESS": eng.PATIENT_TIP_BEND_TWIST_STIFFNESS,
                            "FREE_SPACE_VBD_ITERATIONS": eng.FREE_SPACE_VBD_ITERATIONS, "rod_bend_twist_stiffness": 1e2,
                            "L_SEG_m": eng.L_SEG, "RADIUS_m": eng.RADIUS, "SHAPE_KE": eng.SHAPE_KE, "SHAPE_KD": eng.SHAPE_KD, "SHAPE_MU": eng.SHAPE_MU},
        "rollout_side_overrides": "none" if action_override is None else "action_override supplied",
        "code_sha256": _code_hashes(),
        "patient_id": cfg.patient_id, "access_side": cfg.access_side,
    }
    json.dump({"summary": summary, "rows": rows}, open(out / f"episode_physics_{tag}.json", "w"), indent=1, default=float)
    np.savez(out / f"episode_physics_{tag}.npz", polylines_solver_m=np.array(polys, dtype=object), final_polyline_solver_m=poly_final)
    return summary
