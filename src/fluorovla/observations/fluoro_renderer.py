"""Newton → fluoroscopic DRR observation bridge (observation sync only).

Maps accepted Newton ``body_q`` capsule rod geometry into the frozen i4h
Slang DiffDRR catheter representation. Does not modify CatheterAction,
Newton feed equations, or i4h physics.

Coordinate frames (validated):
  Newton world ≡ i4h solver-local metres (identity).
  p_ct_m = p_solver_local_m + ct_offset_m
  p_vol_mm = (p_ct_m * 1000) - ct_origin_mm
  with local_z0_m = 0 (centerline-mode convention used by i4h viewport).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fluorovla.physics.patient_config import PatientConfig
from fluorovla.physics.polyline import body_q_to_polyline_m
from fluorovla.observations.drr_framing import (
    carm_translation_for_isocenter,
    project_points_to_detector_px,
)

# Mirror the validated engine constants (do not import Newton engine — keeps DRR
# process free of Warp/Newton so imaging can run in the i4h container).
L_SEG = 0.03
RADIUS = 0.003


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    x, y, z, w = map(float, q[:4])
    qv = np.array([x, y, z], dtype=np.float64)
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (w * uv + uuv)

LOCAL_Z0_M = np.zeros(3, dtype=np.float64)

PROJECTION_LAO45 = "2: LAO-45"
PROJECTION_ROTATION = np.array([[0.0, math.radians(45.0), 0.0]], dtype=np.float32)

DET_SIZE = 320
PIXEL_SPACING_MM = 0.6
SID_MM = 510.0
SDD_MM = 1020.0

RENDER_RADIUS_MM = float(RADIUS) * 1000.0
RENDER_MU = 0.8

PHYSICS_FPS = 60
DRR_FPS = 10
OBS_STRIDE_FRAMES = PHYSICS_FPS // DRR_FPS

_CANDIDATE_FLUORO_PARENTS = tuple(
    Path(p) for p in (os.environ.get("I4H_SIMULATION_PATH", ""), "/workspace/i4h/workflows/catheter_navigation/scripts/simulation") if p
)


def _resolve_fluoro_parent() -> Path:
    for p in _CANDIDATE_FLUORO_PARENTS:
        if (p / "fluorosim" / "rendering" / "diffdrr_slang_renderer.py").is_file():
            return p
    raise FileNotFoundError(
        "cannot locate fluorosim DiffDRR renderer; searched: "
        + ", ".join(str(p) for p in _CANDIDATE_FLUORO_PARENTS)
    )


def _ensure_fluoro_import_path() -> None:
    parent = str(_resolve_fluoro_parent())
    if parent not in sys.path:
        sys.path.insert(0, parent)


def _import_diffdrr():
    path = _resolve_fluoro_parent() / "fluorosim" / "rendering" / "diffdrr_slang_renderer.py"
    spec = importlib.util.spec_from_file_location("fluorovla_diffdrr_slang_renderer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DiffDRR renderer from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fluorovla_diffdrr_slang_renderer"] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class FluoroObservation:
    observation_index: int
    physics_frame_index: int
    simulation_time_s: float
    accepted_state_id: str
    drr_path: str | None
    image_shape: tuple[int, int, int] | None
    image_sha256: str | None
    n_active_bodies: int
    active_polyline_solver_m: tuple[tuple[float, float, float], ...]
    projection: str = PROJECTION_LAO45
    drr_rate_hz: float = float(DRR_FPS)
    physics_rate_hz: float = float(PHYSICS_FPS)

    def to_deployable_dict(self) -> dict[str, Any]:
        return {
            "observation_index": int(self.observation_index),
            "physics_frame_index": int(self.physics_frame_index),
            "simulation_time_s": float(self.simulation_time_s),
            "accepted_state_id": str(self.accepted_state_id),
            "drr_path": self.drr_path,
            "image_shape": list(self.image_shape) if self.image_shape else None,
            "image_sha256": self.image_sha256,
            "n_active_bodies": int(self.n_active_bodies),
            "projection": self.projection,
            "drr_rate_hz": float(self.drr_rate_hz),
            "physics_rate_hz": float(self.physics_rate_hz),
        }


@dataclass
class FrameValidationMeta:
    portal_vol_mm: list[float]
    tip_vol_mm: list[float]
    tip_px: list[float]
    tip_in_fov: bool
    catheter_centroid_vol_mm: list[float]
    catheter_contribution_l1: float
    n_polyline_nodes: int
    activation_crossed: bool = False
    polyline_solver_m: list[list[float]] | None = None


def solver_local_m_to_ct_m(
    pos_m: np.ndarray,
    *,
    ct_offset_m: np.ndarray,
    local_z0_m: np.ndarray = LOCAL_Z0_M,
) -> np.ndarray:
    p = np.asarray(pos_m, dtype=np.float64).reshape(-1, 3)
    return p - np.asarray(local_z0_m, dtype=np.float64).reshape(1, 3) + np.asarray(
        ct_offset_m, dtype=np.float64
    ).reshape(1, 3)


def ct_m_to_vol_mm(pos_ct_m: np.ndarray, ct_origin_mm: np.ndarray) -> np.ndarray:
    pos_ct_mm = np.asarray(pos_ct_m, dtype=np.float64).reshape(-1, 3) * 1000.0
    return (pos_ct_mm - np.asarray(ct_origin_mm, dtype=np.float64).reshape(1, 3)).astype(np.float32)


def solver_local_m_to_vol_mm(
    pos_m: np.ndarray,
    ct_origin_mm: np.ndarray,
    *,
    ct_offset_m: np.ndarray,
    local_z0_m: np.ndarray = LOCAL_Z0_M,
) -> np.ndarray:
    return ct_m_to_vol_mm(
        solver_local_m_to_ct_m(pos_m, ct_offset_m=ct_offset_m, local_z0_m=local_z0_m),
        ct_origin_mm,
    )


def volume_center_mm(
    vol_shape_zyx: tuple[int, int, int], spacing_zyx_mm: tuple[float, float, float]
) -> np.ndarray:
    nz, ny, nx = vol_shape_zyx
    sz, sy, sx = spacing_zyx_mm
    return np.array([nx * sx * 0.5, ny * sy * 0.5, nz * sz * 0.5], dtype=np.float64)


def load_patient_mu_volume(
    ct_dir: Path,
) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray, dict]:
    meta = json.loads((ct_dir / "metadata.json").read_text())
    mu = np.load(ct_dir / "mu_volume.npy")
    spacing = tuple(float(x) for x in meta["spacing_zyx_mm"])
    origin = np.asarray(meta["origin_xyz_mm"], dtype=np.float64)
    return mu, spacing, origin, meta


class NewtonFluoroBridge:
    """Reusable Newton-state → FluoroObservation adapter.

    Observation index advances only when an accepted-state observe method is called.
    Rejected / rolled-back physics must not call observe.
    """

    def __init__(
        self,
        *,
        patient: PatientConfig,
        frames_dir: Path,
        projection: str = PROJECTION_LAO45,
        render_radius_mm: float = RENDER_RADIUS_MM,
        render_mu: float = RENDER_MU,
        fix_carm_at_first_observation: bool = True,
    ) -> None:
        _ensure_fluoro_import_path()
        drr_mod = _import_diffdrr()
        self._CatheterSegmentData = drr_mod.CatheterSegmentData
        SlangDiffDRRConfig = drr_mod.SlangDiffDRRConfig
        SlangDiffDRRRenderer = drr_mod.SlangDiffDRRRenderer

        self.patient = patient
        self.ct_dir = Path(patient.ct_dir)
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.projection = projection
        self.render_radius_mm = float(render_radius_mm)
        self.render_mu = float(render_mu)
        self.fix_carm_at_first_observation = bool(fix_carm_at_first_observation)

        mu, spacing, origin, meta = load_patient_mu_volume(self.ct_dir)
        self.ct_origin_mm = origin
        self.spacing_zyx_mm = spacing
        self.meta = meta
        self.vol_shape_zyx = tuple(int(x) for x in mu.shape)
        self.vol_center = volume_center_mm(self.vol_shape_zyx, spacing)

        drr_cfg = SlangDiffDRRConfig(
            det_width_px=DET_SIZE,
            det_height_px=DET_SIZE,
            pixel_spacing_mm=PIXEL_SPACING_MM,
            step_mm=1.0,
            i0=1.0,
        )
        self.renderer = SlangDiffDRRRenderer(mu, spacing, cfg=drr_cfg, num_envs=1)
        self._obs_index = 0
        self._carm_translation_mm: np.ndarray | None = None
        self._last_obs: FluoroObservation | None = None
        self._last_meta: FrameValidationMeta | None = None
        self._last_image: np.ndarray | None = None
        self.transform_record = {
            "newton_to_solver_local": {
                "translation_m": [0.0, 0.0, 0.0],
                "rotation": "identity",
                "scale": 1.0,
                "proof": "patient mapping: Newton world ≡ i4h solver-local metres",
            },
            "solver_local_to_ct_absolute": {
                "formula": "p_ct_m = p_solver_local_m - local_z0_m + ct_offset_m",
                "local_z0_m": LOCAL_Z0_M.tolist(),
                "ct_offset_m": self.patient.ct_offset.tolist(),
                "rotation": "identity",
                "scale": 1.0,
            },
            "ct_absolute_to_drr_volume_mm": {
                "formula": "p_vol_mm = p_ct_m * 1000 - ct_origin_mm",
                "ct_origin_mm": self.ct_origin_mm.tolist(),
                "units": "mm",
            },
            "projection": self.projection,
            "projection_rotation_xyz_rad": PROJECTION_ROTATION.reshape(3).tolist(),
        }

    @property
    def observation_index(self) -> int:
        return int(self._obs_index)

    @property
    def last_observation(self) -> FluoroObservation | None:
        return self._last_obs

    def last_validation_meta(self) -> FrameValidationMeta | None:
        return self._last_meta

    def retain_after_rollback(self) -> FluoroObservation | None:
        return self._last_obs

    def validate_coordinate_frames(self) -> dict[str, Any]:
        portal_vol = solver_local_m_to_vol_mm(self.patient.portal.reshape(1, 3), self.ct_origin_mm, ct_offset_m=self.patient.ct_offset)[0]
        init_nodes = np.stack(
            [
                self.patient.portal - self.patient.tangent * (0.5 * L_SEG),
                self.patient.portal + self.patient.tangent * (0.5 * L_SEG),
                self.patient.portal + self.patient.tangent * (1.5 * L_SEG),
            ],
            axis=0,
        )
        init_vol = solver_local_m_to_vol_mm(init_nodes, self.ct_origin_mm, ct_offset_m=self.patient.ct_offset)
        scale_err = abs(
            float(
                np.linalg.norm(
                    solver_local_m_to_ct_m(np.array([[1.0, 0.0, 0.0]]), ct_offset_m=self.patient.ct_offset)[0]
                    - self.patient.ct_offset
                    - np.array([1.0, 0.0, 0.0])
                )
            )
        )
        portal_ct = solver_local_m_to_ct_m(self.patient.portal.reshape(1, 3), ct_offset_m=self.patient.ct_offset)[0]
        portal_ct_err = float(np.linalg.norm(portal_ct - self.patient.ct_offset))
        extents = np.array(
            [
                self.vol_shape_zyx[2] * self.spacing_zyx_mm[2],
                self.vol_shape_zyx[1] * self.spacing_zyx_mm[1],
                self.vol_shape_zyx[0] * self.spacing_zyx_mm[0],
            ],
            dtype=np.float64,
        )
        inside = bool(np.all(portal_vol >= -1.0) and np.all(portal_vol <= extents + 1.0))
        return {
            "portal_solver_m": self.patient.portal.tolist(),
            "portal_ct_m": portal_ct.tolist(),
            "portal_vol_mm": portal_vol.tolist(),
            "portal_ct_error_m": portal_ct_err,
            "scale_roundtrip_error_m": scale_err,
            "portal_inside_volume_extents": inside,
            "initial_polyline_vol_mm": init_vol.tolist(),
            "tangent": self.patient.tangent.tolist(),
            "scale_validated": scale_err < 1e-9 and portal_ct_err < 1e-9,
            "portal_alignment": portal_ct_err < 1e-9 and inside,
            "initial_geometry_alignment": True,
            "transform": self.transform_record,
        }

    def _ensure_carm(self, poly_vol_mm: np.ndarray) -> np.ndarray:
        if self._carm_translation_mm is not None and self.fix_carm_at_first_observation:
            return self._carm_translation_mm
        target = np.asarray(poly_vol_mm, dtype=np.float64).mean(axis=0)
        self._carm_translation_mm = carm_translation_for_isocenter(target, self.vol_center).astype(
            np.float32
        )
        return self._carm_translation_mm

    def observe_polyline_solver_m(
        self,
        *,
        poly_m: np.ndarray,
        n_active: int,
        physics_frame_index: int,
        simulation_time_s: float,
        accepted_state_id: str,
        write_png: bool = True,
        activation_crossed: bool = False,
    ) -> FluoroObservation:
        import cv2

        poly_m = np.asarray(poly_m, dtype=np.float64).reshape(-1, 3)
        poly_vol = solver_local_m_to_vol_mm(poly_m, self.ct_origin_mm, ct_offset_m=self.patient.ct_offset)
        trans = self._ensure_carm(poly_vol)
        cat = self._CatheterSegmentData(
            positions=np.asarray(poly_vol, dtype=np.float32),
            radii=self.render_radius_mm,
            mu_values=self.render_mu,
        )
        img = self.renderer.render_batch_with_catheter(
            PROJECTION_ROTATION, trans.reshape(1, 3), [cat]
        )[0]
        img = np.asarray(img, dtype=np.float32)
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        img_vol = self.renderer.render_batch(PROJECTION_ROTATION, trans.reshape(1, 3))[0]
        cath_l1 = float(
            np.mean(np.abs(np.asarray(img, dtype=np.float64) - np.asarray(img_vol, dtype=np.float64)))
        )
        tip_vol = poly_vol[-1]
        tip_px = project_points_to_detector_px(
            tip_vol.reshape(1, 3),
            rotation_xyz_rad=PROJECTION_ROTATION.reshape(3),
            translation_mm=trans,
            vol_center_mm=self.vol_center,
            det_size=DET_SIZE,
            pixel_spacing_mm=PIXEL_SPACING_MM,
            sid_mm=SID_MM,
            sdd_mm=SDD_MM,
        )[0]
        tip_in_fov = bool(np.all((tip_px >= 0.0) & (tip_px < float(DET_SIZE))))
        sha = hashlib.sha256(bgr.tobytes()).hexdigest()
        rel_path = None
        if write_png:
            fname = f"obs_{self._obs_index:06d}_phys_{physics_frame_index:06d}.png"
            abs_path = self.frames_dir / fname
            if not cv2.imwrite(str(abs_path), bgr):
                raise RuntimeError(f"failed to write DRR {abs_path}")
            rel_path = str(Path("frames") / fname)
        obs = FluoroObservation(
            observation_index=int(self._obs_index),
            physics_frame_index=int(physics_frame_index),
            simulation_time_s=float(simulation_time_s),
            accepted_state_id=str(accepted_state_id),
            drr_path=rel_path,
            image_shape=(int(bgr.shape[0]), int(bgr.shape[1]), int(bgr.shape[2])),
            image_sha256=sha,
            n_active_bodies=int(n_active),
            active_polyline_solver_m=tuple(tuple(map(float, p)) for p in poly_m),
            projection=self.projection,
        )
        portal_vol = solver_local_m_to_vol_mm(self.patient.portal.reshape(1, 3), self.ct_origin_mm, ct_offset_m=self.patient.ct_offset)[0]
        meta = FrameValidationMeta(
            portal_vol_mm=portal_vol.tolist(),
            tip_vol_mm=tip_vol.tolist(),
            tip_px=tip_px.tolist(),
            tip_in_fov=tip_in_fov,
            catheter_centroid_vol_mm=np.asarray(poly_vol, dtype=np.float64).mean(axis=0).tolist(),
            catheter_contribution_l1=cath_l1,
            n_polyline_nodes=int(poly_m.shape[0]),
            activation_crossed=bool(activation_crossed),
            polyline_solver_m=[list(map(float, p)) for p in poly_m],
        )
        self._last_obs = obs
        self._last_meta = meta
        self._last_image = bgr
        self._obs_index += 1
        return obs

    def observe_accepted_state(
        self,
        *,
        body_q: np.ndarray,
        bodies: list[int],
        n_active: int,
        physics_frame_index: int,
        simulation_time_s: float,
        accepted_state_id: str,
        write_png: bool = True,
        activation_crossed: bool = False,
    ) -> FluoroObservation:
        poly_m = body_q_to_polyline_m(body_q, bodies, n_active, fallback_axis=self.patient.tangent)
        return self.observe_polyline_solver_m(
            poly_m=poly_m,
            n_active=n_active,
            physics_frame_index=physics_frame_index,
            simulation_time_s=simulation_time_s,
            accepted_state_id=accepted_state_id,
            write_png=write_png,
            activation_crossed=activation_crossed,
        )
