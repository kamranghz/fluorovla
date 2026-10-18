# SPDX-FileCopyrightText: FluoroVLA
# ---------------------------------------------------------------------------
# Typed per-step records, episode metadata, and LeRobot v2.1 serialization
# for the GR00T catheter NEW_EMBODIMENT dataset.
#
# === Policy-visible state contract (FROZEN) ===
#   state[0] = inserted_length_m             (deployable)
#   state[1] = previous_insert_velocity_mps  (deployable, causal)
#
# === Causality ===
#   At decision step 0:  previous_insert_velocity_mps = INITIAL_PREVIOUS_VELOCITY (0.000)
#   At decision step t>0: previous_insert_velocity_mps = executed velocity at step t-1
#   The *executed* CatheterAction is used, not a predicted or requested value.
#
# === Privileged fields (evaluation-only, never in parquet) ===
#   n_active_bodies, activation_flag, accepted_state_id, physics_frame_index,
#   polyline_solver_m, tip_px, gt_delta_L_m, cath_contribution_L1, body_q
#
# === Serialization layout (GR00T-flavored LeRobot v2.1) ===
#
#   <output_dir>/
#     meta/
#       info.json           — LeRobot dataset header
#       modality.json       — GR00T modality mapping
#       tasks.jsonl         — task index -> task string
#       episodes.jsonl      — episode index -> {tasks, length}
#       stats.json          — per-field min/max/mean/std stubs
#       episode_schema.json — FluoroVLA privileged metadata
#     data/
#       chunk-000/
#         episode_000000.parquet  — policy-visible tabular data
#     videos/
#       chunk-000/
#         observation.images.fluoro/
#           episode_000000.mp4 — fluoro video decoded by the GR00T loader
#         fluoro/
#           episode_000000/    — DRR PNG frames (lossless source of the mp4)
#
# Parquet columns (policy-visible only):
#   action                  float32 [1]  — insert_velocity_mps
#   observation.state       float32 [2]  — [inserted_length_m, previous_insert_velocity_mps]
#   timestamp               float32 scalar — simulation_time_s
#   frame_index             int64   scalar — step index within episode
#   episode_index           int64   scalar — episode number
#   index                   int64   scalar — global frame index
#   task_index              int64   scalar — always 0
#   (scalar columns are stored as scalars, not length-1 arrays, because the
#    GR00T LeRobot loader hashes task_index; info.json still declares shape [1])
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Schema version and canonical strings.
# ---------------------------------------------------------------------------
CATHETER_SCHEMA_VERSION: str = "2.0.0"   # bumped from 1.0.0 for state contract change
#: LeRobot video feature key; the GR00T loader substitutes it for {video_key}.
_VIDEO_FEATURE_KEY: str = "observation.images.fluoro"
GROOT_EMBODIMENT_TAG: str = "new_embodiment"
LEROBOT_CODEBASE_VERSION: str = "v2.1"

#: Canonical task instruction — single source of truth.
#: Must match catheter_modality.CATHETER_TASK exactly.
CATHETER_TASK: str = (
    "Advance the catheter tip to the distal vessel target while remaining inside the lumen."
)

#: Previous-action sentinel at step 0.
INITIAL_PREVIOUS_VELOCITY: float = 0.000

# Modality.json mirrors catheter_modality.py.
# Stored here so episode_schema.py is importable without gr00t on the path.
MODALITY_JSON: dict = {
    "state": {
        "catheter": {"start": 0, "end": 2},
    },
    "action": {
        "insert_velocity": {"start": 0, "end": 1},
    },
    "video": {
        "fluoro": {"original_key": "observation.images.fluoro"},
    },
    "annotation": {
        "human.task_description": {"original_key": "task_index"},
    },
}


# ---------------------------------------------------------------------------
# Per-step record.
# ---------------------------------------------------------------------------

@dataclass
class CatheterEpisodeStep:
    """One per-step record for the catheter insertion VLA episode.

    Policy-visible fields are written to the LeRobot parquet file.
    Privileged fields are written only to episode_schema.json.
    """

    # ----------------------------------------------------------------
    # Policy-visible — observation
    # ----------------------------------------------------------------
    simulation_time_s: float
    """Wall time within the episode (seconds)."""

    inserted_length_m: float
    """Metres of catheter deployed from portal.
    Canonical field name: inserted_length_m.
    Do NOT use insertion_depth_m (deprecated alias removed in schema v2).
    Deployable: computable from catheter kinematics without imaging targets.
    """

    previous_insert_velocity_mps: float
    """Insert velocity executed at the immediately preceding step.
    At step 0: INITIAL_PREVIOUS_VELOCITY (0.000).
    At step t>0: the velocity from the executed CatheterAction at step t-1.
    Deployable: the policy itself issued the prior action.
    MUST NOT be set from predicted, requested, or episode-label velocity.
    """

    fluoro_png_path: str
    """Absolute path to the 320x320 DRR PNG for this step."""

    task_description: str
    """Fixed language instruction — always CATHETER_TASK."""

    # ----------------------------------------------------------------
    # Policy-visible — action taken at this step
    # ----------------------------------------------------------------
    insert_velocity_mps: float
    """Target insertion velocity executed at this step (m/s):
    HOLD=0.000, SLOW=0.006, CRUISE=0.012."""

    # ----------------------------------------------------------------
    # Privileged — evaluation / debugging only, never fed to policy
    # ----------------------------------------------------------------
    n_active_bodies: int
    """Raw Newton body activation count.  Privileged."""

    activation_flag: float
    """1.0 if n_active_bodies grew beyond episode-start count.
    Privileged — must NOT appear in policy-visible state.
    """

    accepted_state_id: str
    """Newton accepted-state ID for cross-validation sync checks.  Privileged."""

    physics_frame_index: int
    """Physics simulation frame number.  Privileged."""


# ---------------------------------------------------------------------------
# Episode-level metadata.
# ---------------------------------------------------------------------------

@dataclass
class CatheterEpisodeMeta:
    """Episode-level metadata stored in episode_schema.json."""

    episode_id: str
    patient_id: str
    feed_mode: str
    bend_stiffness: float
    action_sequence: str
    fps_drr: float
    fps_physics: float
    obs_stride_frames: int
    total_steps: int
    n_initial_bodies: int
    task_description: str
    embodiment_tag: str
    gt_delta_L_m: float
    groot_schema_version: str = CATHETER_SCHEMA_VERSION
    lerobot_codebase_version: str = LEROBOT_CODEBASE_VERSION
    initial_previous_velocity: float = INITIAL_PREVIOUS_VELOCITY


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------

def _validate_step(step: CatheterEpisodeStep, step_index: int) -> list[str]:
    """Return list of violation strings (empty = valid)."""
    errors: list[str] = []
    if step.inserted_length_m < 0.0:
        errors.append(f"inserted_length_m={step.inserted_length_m:.4f} < 0")
    valid_vels = {0.000, 0.006, 0.012}
    for attr, val in [("insert_velocity_mps", step.insert_velocity_mps),
                      ("previous_insert_velocity_mps", step.previous_insert_velocity_mps)]:
        nearest = min(valid_vels, key=lambda v: abs(v - val))
        if abs(val - nearest) > 1e-6:
            errors.append(f"step[{step_index}].{attr}={val} not in {{0.000, 0.006, 0.012}}")
    if step.task_description != CATHETER_TASK:
        errors.append(f"step[{step_index}].task_description mismatch")
    if step.n_active_bodies < 2:
        errors.append(f"step[{step_index}].n_active_bodies={step.n_active_bodies} < 2")
    return errors


def _validate_causality(steps: list[CatheterEpisodeStep]) -> list[str]:
    """Verify previous_insert_velocity_mps is causally derived from the prior step."""
    errors: list[str] = []
    # Step 0: must be sentinel
    if steps and abs(steps[0].previous_insert_velocity_mps - INITIAL_PREVIOUS_VELOCITY) > 1e-9:
        errors.append(
            f"step[0].previous_insert_velocity_mps={steps[0].previous_insert_velocity_mps} "
            f"!= INITIAL_PREVIOUS_VELOCITY={INITIAL_PREVIOUS_VELOCITY}"
        )
    # Step t>0: must equal insert_velocity_mps of step t-1
    for t in range(1, len(steps)):
        expected = steps[t - 1].insert_velocity_mps
        actual = steps[t].previous_insert_velocity_mps
        if abs(actual - expected) > 1e-9:
            errors.append(
                f"step[{t}].previous_insert_velocity_mps={actual} "
                f"!= step[{t-1}].insert_velocity_mps={expected}"
            )
    return errors


def validate_episode_steps(steps: list[CatheterEpisodeStep]) -> dict:
    """Validate step fields and causal consistency."""
    all_errors: list[str] = []
    for i, step in enumerate(steps):
        all_errors.extend(_validate_step(step, i))
    all_errors.extend(_validate_causality(steps))
    return {
        "pass": len(all_errors) == 0,
        "n_steps": len(steps),
        "n_errors": len(all_errors),
        "errors": all_errors,
    }


# ---------------------------------------------------------------------------
# Episode Logger.
# ---------------------------------------------------------------------------

class CatheterEpisodeLogger:
    """Collects per-step records and writes a GR00T-compatible LeRobot dataset.

    Causality contract:
        add_step is called once per decision step.
        The first call records previous_insert_velocity_mps = INITIAL_PREVIOUS_VELOCITY.
        Each subsequent call uses the insert_velocity_mps from the prior call.
        This ensures state[1] is always the *executed* prior action.

    Usage::

        logger = CatheterEpisodeLogger(output_dir=..., ...)
        for obs in observations:
            logger.add_step(
                simulation_time_s=obs.time,
                n_active_bodies=obs.n_active,
                l_deployed_m=obs.l_deployed,
                fluoro_png_path=obs.png,
                insert_velocity_mps=action.insert_velocity_mps,
                accepted_state_id=obs.state_id,
                physics_frame_index=obs.frame,
            )
        report = logger.finalize()
    """

    def __init__(
        self,
        output_dir: Path,
        patient_id: str,
        feed_mode: str,
        action_sequence: str,
        bend_stiffness: float,
        fps_drr: float = 10.0,
        fps_physics: float = 60.0,
        obs_stride_frames: int = 6,
        episode_index: int = 0,
        projection_label: str = "LAO-45",
    ):
        self.projection_label = projection_label    # info.json camera label; default keeps the pre-existing value
        self.output_dir = Path(output_dir)
        self.patient_id = patient_id
        self.feed_mode = feed_mode
        self.action_sequence = action_sequence
        self.bend_stiffness = bend_stiffness
        self.fps_drr = fps_drr
        self.fps_physics = fps_physics
        self.obs_stride_frames = obs_stride_frames
        self.episode_index = episode_index

        self._steps: list[CatheterEpisodeStep] = []
        self._n_initial_bodies: Optional[int] = None
        # Causal tracking: previous_insert_velocity_mps for next step.
        self._prev_velocity_mps: float = INITIAL_PREVIOUS_VELOCITY

        # Paths
        self._meta_dir = self.output_dir / "meta"
        self._data_dir = self.output_dir / "data" / "chunk-000"
        self._frames_dst = (
            self.output_dir / "videos" / "chunk-000" / "fluoro"
            / f"episode_{episode_index:06d}"
        )

    # ------------------------------------------------------------------
    # Step accumulation
    # ------------------------------------------------------------------

    def add_step(
        self,
        simulation_time_s: float,
        n_active_bodies: int,
        l_deployed_m: float,
        fluoro_png_path: str,
        insert_velocity_mps: float,
        accepted_state_id: str,
        physics_frame_index: int,
    ) -> CatheterEpisodeStep:
        """Record one decision step.

        Args:
            simulation_time_s:   Episode wall time (s).
            n_active_bodies:     Raw Newton active body count.
            l_deployed_m:        Total catheter material deployed from portal (m).
            fluoro_png_path:     Absolute path to the 320x320 DRR PNG.
            insert_velocity_mps: CatheterAction velocity EXECUTED at this step (m/s).
                                 This becomes previous_insert_velocity_mps for the next step.
            accepted_state_id:   Newton accepted-state ID string.
            physics_frame_index: Physics simulation frame index.

        Returns:
            The created CatheterEpisodeStep.
        """
        if self._n_initial_bodies is None:
            self._n_initial_bodies = n_active_bodies

        # activation_flag is privileged (not policy-visible state)
        activation_flag = float(n_active_bodies > self._n_initial_bodies)

        step = CatheterEpisodeStep(
            simulation_time_s=simulation_time_s,
            inserted_length_m=float(l_deployed_m),
            # Causal: state[1] = velocity from the *preceding* step.
            # At step 0: INITIAL_PREVIOUS_VELOCITY (0.000).
            previous_insert_velocity_mps=self._prev_velocity_mps,
            fluoro_png_path=fluoro_png_path,
            task_description=CATHETER_TASK,
            insert_velocity_mps=float(insert_velocity_mps),
            n_active_bodies=n_active_bodies,
            activation_flag=activation_flag,
            accepted_state_id=accepted_state_id,
            physics_frame_index=physics_frame_index,
        )
        self._steps.append(step)
        # Update causal tracker for the next step.
        self._prev_velocity_mps = float(insert_velocity_mps)
        return step

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> dict:
        """Write LeRobot v2.1 dataset to output_dir.

        Returns a report dict with 'pass', 'n_steps', 'validation', 'files'.
        """
        if not self._steps:
            return {"pass": False, "error": "no steps recorded"}

        # Validate (includes causality check)
        val = validate_episode_steps(self._steps)

        # Create dirs
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dst.mkdir(parents=True, exist_ok=True)

        # Write
        self._copy_frames()
        self._encode_video()
        self._write_parquet()
        self._write_info_json()
        self._write_modality_json()
        self._write_tasks_jsonl()
        self._write_episodes_jsonl()
        self._write_episode_schema_json()

        files = sorted(str(p) for p in self.output_dir.rglob("*") if p.is_file())

        return {
            "pass": val["pass"],
            "n_steps": len(self._steps),
            "validation": val,
            "output_dir": str(self.output_dir),
            "files_written": len(files),
        }

    # ------------------------------------------------------------------
    # Internal writers
    # ------------------------------------------------------------------

    def _copy_frames(self):
        for i, step in enumerate(self._steps):
            src = Path(step.fluoro_png_path)
            dst = self._frames_dst / f"frame_{i:06d}.png"
            if src.exists():
                shutil.copy2(str(src), str(dst))

    def _encode_video(self):
        """Encode the copied PNG frames to the mp4 the GR00T loader decodes.

        Lossless libx264 (crf 0, yuv444p). Fails loudly if ffmpeg is missing or
        a frame is missing: a dataset without a decodable video is not loadable.
        """
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found on PATH; cannot write fluoro mp4")
        missing = [i for i in range(len(self._steps))
                   if not (self._frames_dst / f"frame_{i:06d}.png").exists()]
        if missing:
            raise RuntimeError(f"missing PNG frames for steps {missing[:5]}")
        out_dir = self.output_dir / "videos" / "chunk-000" / _VIDEO_FEATURE_KEY
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"episode_{self.episode_index:06d}.mp4"
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-framerate", f"{self.fps_drr:g}",
            "-i", str(self._frames_dst / "frame_%06d.png"),
            "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p",
            str(out),
        ]
        subprocess.run(cmd, check=True)

    def _write_parquet(self):
        """Write policy-visible columns only (no privileged fields)."""
        import pandas as pd

        global_offset = self.episode_index * len(self._steps)
        rows = []
        for i, step in enumerate(self._steps):
            row = {
                # action: insert_velocity_mps at this step
                "action": np.array([step.insert_velocity_mps], dtype=np.float32),
                # state: [inserted_length_m, previous_insert_velocity_mps]
                # activation_flag is privileged and NOT included here
                "observation.state": np.array(
                    [step.inserted_length_m, step.previous_insert_velocity_mps],
                    dtype=np.float32,
                ),
                "timestamp": np.float32(step.simulation_time_s),
                "frame_index": np.int64(i),
                "episode_index": np.int64(self.episode_index),
                "index": np.int64(global_offset + i),
                "task_index": np.int64(0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        parquet_path = self._data_dir / f"episode_{self.episode_index:06d}.parquet"
        df.to_parquet(str(parquet_path), index=False, engine="pyarrow")

    def _write_info_json(self):
        info = {
            "codebase_version": LEROBOT_CODEBASE_VERSION,
            "robot_type": "catheter_endovascular",
            "total_episodes": 1,
            "total_frames": len(self._steps),
            "total_tasks": 1,
            "chunks_size": 1000,
            "fps": self.fps_drr,
            "splits": {"train": f"0:{len(self._steps)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            ),
            "features": {
                "action": {
                    "dtype": "float32",
                    "names": ["insert_velocity_mps"],
                    "shape": [1],
                },
                "observation.state": {
                    "dtype": "float32",
                    "names": ["inserted_length_m", "previous_insert_velocity_mps"],
                    "shape": [2],
                },
                "observation.images.fluoro": {
                    "dtype": "image",
                    "shape": [320, 320, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "image.height": 320,
                        "image.width": 320,
                        "image.channels": 3,
                        "image.is_depth_map": False,
                        "image.fps": self.fps_drr,
                        "image.pixel_spacing_mm": 0.6,
                        "image.projection": self.projection_label,
                        "image.sid_mm": 510.0,
                        "image.sdd_mm": 1020.0,
                    },
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
            "total_chunks": 1,
            "total_images": len(self._steps),
            "groot_schema_version": CATHETER_SCHEMA_VERSION,
            "embodiment_tag": GROOT_EMBODIMENT_TAG,
        }
        with open(self._meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def _write_modality_json(self):
        with open(self._meta_dir / "modality.json", "w") as f:
            json.dump(MODALITY_JSON, f, indent=2)

    def _write_tasks_jsonl(self):
        with open(self._meta_dir / "tasks.jsonl", "w") as f:
            f.write(json.dumps({"task_index": 0, "task": CATHETER_TASK}) + "\n")

    def _write_episodes_jsonl(self):
        with open(self._meta_dir / "episodes.jsonl", "w") as f:
            f.write(
                json.dumps({
                    "episode_index": self.episode_index,
                    "tasks": [CATHETER_TASK],
                    "length": len(self._steps),
                }) + "\n"
            )

    def _write_episode_schema_json(self):
        """FluoroVLA-specific metadata: privileged fields + provenance."""
        gt_delta_l = (
            self._steps[-1].inserted_length_m - self._steps[0].inserted_length_m
            if self._steps else 0.0
        )

        meta = CatheterEpisodeMeta(
            episode_id=f"{self.patient_id}_{self.action_sequence}_ep{self.episode_index:06d}",
            patient_id=self.patient_id,
            feed_mode=self.feed_mode,
            bend_stiffness=self.bend_stiffness,
            action_sequence=self.action_sequence,
            fps_drr=self.fps_drr,
            fps_physics=self.fps_physics,
            obs_stride_frames=self.obs_stride_frames,
            total_steps=len(self._steps),
            n_initial_bodies=self._n_initial_bodies or 0,
            task_description=CATHETER_TASK,
            embodiment_tag=GROOT_EMBODIMENT_TAG,
            gt_delta_L_m=gt_delta_l,
            groot_schema_version=CATHETER_SCHEMA_VERSION,
            lerobot_codebase_version=LEROBOT_CODEBASE_VERSION,
            initial_previous_velocity=INITIAL_PREVIOUS_VELOCITY,
        )

        # Privileged per-step data (NOT written to parquet)
        privileged_steps = [
            {
                "step": i,
                "n_active_bodies": s.n_active_bodies,
                "activation_flag": s.activation_flag,  # privileged
                "accepted_state_id": s.accepted_state_id,
                "physics_frame_index": s.physics_frame_index,
                "simulation_time_s": s.simulation_time_s,
            }
            for i, s in enumerate(self._steps)
        ]

        schema_doc = {
            "episode_meta": asdict(meta),
            "privileged_steps": privileged_steps,
            "policy_contract": {
                "deployable_state_fields": [
                    "inserted_length_m",
                    "previous_insert_velocity_mps",
                ],
                "causality": {
                    "step_0_previous_velocity": INITIAL_PREVIOUS_VELOCITY,
                    "step_t_previous_velocity": "= executed insert_velocity_mps at step t-1",
                    "source": "executed CatheterAction, not predicted or requested",
                },
                "privileged_fields": [
                    "n_active_bodies",
                    "activation_flag",
                    "accepted_state_id",
                    "physics_frame_index",
                    "polyline_solver_m",
                    "tip_px",
                    "gt_delta_L_m",
                    "cath_contribution_L1",
                    "body_q",
                ],
                "must_not_include_in_policy_obs": [
                    "target_xyz",
                    "tip_target_distance",
                    "centerline_coordinates",
                    "sdf_values",
                    "contact_forces",
                    "full_body_q",
                    "activation_flag",
                    "n_active_bodies",
                    "active_body_count",
                    "activation_event",
                ],
            },
        }

        with open(self._meta_dir / "episode_schema.json", "w") as f:
            json.dump(schema_doc, f, indent=2)

    # ------------------------------------------------------------------
    # Stats stub
    # ------------------------------------------------------------------

    def write_stats_stub(self):
        """Write meta/stats.json with observed min/max from this episode."""
        lengths = [s.inserted_length_m for s in self._steps]
        prev_vels = [s.previous_insert_velocity_mps for s in self._steps]
        velocities = [s.insert_velocity_mps for s in self._steps]

        stats = {
            "observation.state": {
                "min": [min(lengths), min(prev_vels)],
                "max": [max(lengths) if lengths else 0.0, max(prev_vels)],
                "mean": [
                    sum(lengths) / len(lengths) if lengths else 0.0,
                    sum(prev_vels) / len(prev_vels) if prev_vels else 0.0,
                ],
                "std": [0.0, 0.0],  # stub; requires full dataset aggregation
            },
            "action": {
                "min": [min(velocities)],
                "max": [max(velocities)],
                "mean": [sum(velocities) / len(velocities) if velocities else 0.0],
                "std": [0.0],  # stub
            },
        }
        with open(self._meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
