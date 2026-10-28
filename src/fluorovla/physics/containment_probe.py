"""Read-only runtime probe for the containment audit. It records, at the checkpoints of ``PatientEpisode`` and at every decision substep, the active body poses, the canonical
axis samples, the engine SDF values at those samples and the contact count of the step; it reads Newton state and queries the lumen SDF, it never writes simulation state.

Observer protocol (see PatientEpisode): ``probe(stage, episode, info)``. At ``decision_substep`` the state is the one AFTER the substep, ``info['violation_m']`` is the engine's own
violation measure of that state, and ``episode.contacts`` still holds the contacts that were generated from the state BEFORE the substep (recorded as ``contact_count``).
``EngineSdfEvaluator`` evaluates the engine SDF path on an arbitrary (for example synthetic, temporary) configuration; it only builds a mesh in the cache location of that configuration.
"""
from __future__ import annotations

import numpy as np

from fluorovla.physics.engine import L_SEG, RADIUS, SIM_SUBSTEPS
from fluorovla.physics.lumen_sdf import LumenSDF
from fluorovla.physics.patient_config import PatientConfig
from fluorovla.physics.patient_episode import FRAMES_PER_DECISION, SAMPLE_FRAC, axis_sample_points

CHECKPOINTS = ("build_scene", "after_settle1", "after_body_enable", "after_settle2", "ready")
STAGE_NAMES = CHECKPOINTS + ("settle1_frame", "settle2_frame", "decision_substep")
FRAME_SUBSTEPS = SIM_SUBSTEPS          # settle phases are recorded once per physics frame (its last substep)


class ContainmentProbe:
    """Callable passed as ``observer`` to PatientEpisode / run_physics_episode."""

    def __init__(self, record_settle_frames: bool = True):
        self.record_settle_frames = record_settle_frames
        self._rows: list[dict] = []
        self._meta: dict | None = None
        self._decision_substeps = 0

    # -- observer ------------------------------------------------------------------------------------------------
    def __call__(self, stage: str, ep, info: dict) -> None:
        if stage in CHECKPOINTS:
            self._record(stage, ep, decision=-1, frame=-1, sub=-1, violation_m=float("nan"))
        elif stage in ("settle1_substep", "settle2_substep") and self.record_settle_frames and info["k"] % FRAME_SUBSTEPS == FRAME_SUBSTEPS - 1:
            self._record("settle1_frame" if stage.startswith("settle1") else "settle2_frame", ep, decision=-1, frame=info["k"] // FRAME_SUBSTEPS, sub=info["k"] % FRAME_SUBSTEPS,
                         violation_m=float("nan"))
        elif stage == "decision_substep":
            self._decision_substeps += 1
            self._record("decision_substep", ep, decision=(self._decision_substeps - 1) // (FRAME_SUBSTEPS * FRAMES_PER_DECISION), frame=info["frame"], sub=info["sub"], violation_m=float(info["violation_m"]))

    def _record(self, stage: str, ep, decision: int, frame: int, sub: int, violation_m: float) -> None:
        if self._meta is None:
            self._meta = {"patient_id": ep.cfg.patient_id, "access_side": ep.cfg.access_side, "ct_offset_m": ep.cfg.ct_offset.tolist(), "tangent": ep.cfg.tangent.tolist(),
                          "sign_scale": float(ep.cfg.sign_scale), "l_seg_m": float(L_SEG), "radius_m": float(RADIUS), "sample_frac": list(SAMPLE_FRAC), "body_ids": [int(b) for b in ep.bodies],
                          "root_rod_index": int(list(ep.bodies).index(ep.root)), "moving_rod_indices": [i for i, b in enumerate(ep.bodies) if b != ep.root],
                          "stage_names": list(STAGE_NAMES), "sdf_mesh": {k: ep.sdf.meta.get(k) for k in ("n_vertices", "n_faces", "downsample", "mesh_frame", "extractor")}}
        q = ep.state_0.body_q.numpy()
        rod_q = q[np.asarray(ep.bodies, dtype=int)].astype(np.float64)                    # row i = rod body i (root first)
        samples = axis_sample_points(q, ep.bodies, ep.n_active, ep.root, ep.axis)
        rod_index = {int(b): i for i, b in enumerate(ep.bodies)}
        if samples:
            pts = np.stack([p for _b, _f, p in samples], axis=0)
            phi, _n, hit = ep.sdf.query(pts.astype(np.float32))
        else:
            pts, phi, hit = np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int32)
        self._rows.append({"stage": STAGE_NAMES.index(stage), "decision": decision, "frame": frame, "sub": sub, "n_active": int(ep.n_active), "engine_violation_m": violation_m,
                           "contact_count": int(ep.contacts.contact_counters.numpy()[0]), "body_q": rod_q,
                           "sample_body": np.array([rod_index[int(b)] for b, _f, _p in samples], dtype=np.int16), "sample_fraction": np.array([f for _b, f, _p in samples], float),
                           "sample_points_m": np.asarray(pts, float), "phi_engine_m": np.asarray(phi, float), "sample_hit": np.asarray(hit, dtype=np.int8)})

    # -- output --------------------------------------------------------------------------------------------------
    def arrays(self) -> tuple[dict, dict]:
        """(arrays for ``np.savez``, JSON-able metadata). Per-event sample arrays are padded with -1 / NaN to the maximum sample count."""
        if self._meta is None:
            raise RuntimeError("the probe recorded nothing")
        E = len(self._rows)
        S = max(len(r["sample_body"]) for r in self._rows)
        NB = self._rows[0]["body_q"].shape[0]
        a = {"stage_code": np.array([r["stage"] for r in self._rows], np.int16), "decision": np.array([r["decision"] for r in self._rows], np.int32),
             "frame": np.array([r["frame"] for r in self._rows], np.int32), "sub": np.array([r["sub"] for r in self._rows], np.int16),
             "n_active": np.array([r["n_active"] for r in self._rows], np.int16), "engine_violation_m": np.array([r["engine_violation_m"] for r in self._rows], float),
             "contact_count": np.array([r["contact_count"] for r in self._rows], np.int32), "body_q": np.stack([r["body_q"] for r in self._rows]),
             "sample_body": np.full((E, S), -1, np.int16), "sample_fraction": np.full((E, S), np.nan), "sample_points_m": np.full((E, S, 3), np.nan),
             "phi_engine_m": np.full((E, S), np.nan), "sample_hit": np.zeros((E, S), np.int8)}
        assert a["body_q"].shape == (E, NB, 7)
        for i, r in enumerate(self._rows):
            n = len(r["sample_body"])
            a["sample_body"][i, :n], a["sample_fraction"][i, :n], a["sample_points_m"][i, :n], a["phi_engine_m"][i, :n], a["sample_hit"][i, :n] = (
                r["sample_body"], r["sample_fraction"], r["sample_points_m"], r["phi_engine_m"], r["sample_hit"])
        return a, dict(self._meta, n_events=E)


class EngineSdfEvaluator:
    """Engine SDF path (Warp mesh query) on an arbitrary configuration: ``phi_mm(points_ct_mm)`` with the engine's float32 solver-local query. Builds the collision mesh at the
    configuration's own ``mesh_cache`` location only."""

    def __init__(self, cfg):
        from fluorovla.physics.engine import ensure_warp
        ensure_warp()
        self.cfg = cfg
        self.sdf = LumenSDF(cfg)

    def phi_mm(self, pts_ct_mm: np.ndarray) -> np.ndarray:
        local = (np.atleast_2d(np.asarray(pts_ct_mm, float)) / 1000.0 - self.cfg.ct_offset).astype(np.float32)
        return self.sdf.query(local)[0] * 1000.0


def synthetic_engine_factory(workdir):
    """Factory for the numerical calibration: ``factory(mask, origin_xyz_mm, spacing_zyx_mm) -> phi_mm(points_ct_mm)`` that runs the engine SDF path on a synthetic mask.
    Every synthetic twin is written below ``workdir`` (never into the canonical data cache); the solver frame is shifted to the centre of the volume like a portal offset."""
    import json
    from pathlib import Path
    root = Path(workdir)
    counter = [0]

    def factory(mask, origin_xyz_mm, spacing_zyx_mm):
        d = root / f"synthetic_twin_{counter[0]:05d}"
        counter[0] += 1
        d.mkdir(parents=True, exist_ok=False)
        np.save(d / "vessel_mask.npy", np.asarray(mask, np.uint8))
        (d / "metadata.json").write_text(json.dumps({"spacing_zyx_mm": list(map(float, spacing_zyx_mm)), "origin_xyz_mm": list(map(float, origin_xyz_mm))}))
        centre_mm = np.asarray(origin_xyz_mm, float) + (np.array(mask.shape)[::-1] - 1) / 2.0 * np.asarray(spacing_zyx_mm, float)[::-1]
        cfg = PatientConfig(patient_id="synthetic", access_side="synthetic", ct_dir=d, mesh_cache=d / "collision_mesh.npz", mesh_frame="solver_local", portal_m=(0.0, 0.0, 0.0),
                            tangent_raw=(0.0, 0.0, 1.0), ct_offset_m=tuple((centre_mm / 1000.0).tolist()), sign_scale=1.0, landing_zone_path=d / "unused.json", path_npz_path=d / "unused.npz")
        return EngineSdfEvaluator(cfg).phi_mm

    return factory
