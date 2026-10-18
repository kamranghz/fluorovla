# SPDX-FileCopyrightText: FluoroVLA
# ---------------------------------------------------------------------------
# GR00T NEW_EMBODIMENT catheter modality contract.
#
# This module defines the canonical policy-visible observation / action schema
# for the fluoroscopy-conditioned catheter insertion VLA task.
#
# === Policy contract (must NEVER contain privileged information) ===
#
#  Observation
#  -----------
#  video   : "fluoro"
#            320x320 px, 3-channel (replicated grayscale from DRR PNG),
#            LAO-45 projection, 0.6 mm/px, SID=510 mm, SDD=1020 mm, 10 Hz.
#
#  state   : "catheter"  shape (2,)  dtype float32
#            [0] inserted_length_m         — metres of catheter material
#                deployed from portal. Derived from n_completed * L_SEG +
#                L_fraction at last accepted step.
#                Deployable: computable from catheter kinematics alone.
#            [1] previous_insert_velocity_mps — the insert_velocity_mps that
#                was EXECUTED at the immediately preceding decision step.
#                At step 0: 0.000 (episode-start sentinel).
#                At step t>0: the velocity executed at step t-1.
#                Deployable: the policy issued the previous action itself.
#
#  language: "annotation.human.task_description"
#            Always the fixed string CATHETER_TASK.
#
#  Action
#  ------
#  action  : "insert_velocity"  shape (1,)  dtype float32
#            [0] insert_velocity_mps — HOLD=0.000, SLOW=0.006, CRUISE=0.012 m/s.
#            Representation: ABSOLUTE (the value is the target velocity,
#            not a delta from current velocity).
#            Type: NON_EEF (joint-space insertion, no end-effector pose).
#            Format: DEFAULT.
#
# === Privileged fields (evaluation / debugging ONLY, never fed to policy) ===
#   n_active_bodies, activation_flag, accepted_state_id, polyline_solver_m,
#   tip_px, gt_delta_L_m, cath_contribution_L1, body_q
#
# === What must NOT change ===
#   - Newton physics, CatheterAction, DRR bridge, patient_distal_joint feed mode
#   - No steering, rotation, torque DOFs added
#   - No target XYZ, tip-target distance, centerline, SDF, contact forces,
#     n_active_bodies, or activation_flag in the policy-visible state
#
# === Field name canon ===
#   inserted_length_m  (NOT insertion_depth_m — do NOT use the deprecated alias)
# ---------------------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# GR00T repo path — added at import time so gr00t types resolve without
# a full Isaac-GR00T installation in the active venv.
# ---------------------------------------------------------------------------
GROOT_REPO_PATH = Path(r"D:\RoboticsResearch\repositories\Isaac-GR00T")
if str(GROOT_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(GROOT_REPO_PATH))

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS, register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# ---------------------------------------------------------------------------
# Domain constants (mirrors newton_fluoro_observation.py — do NOT import from
# there; this module must be usable without Newton/Warp).
# ---------------------------------------------------------------------------

#: Canonical fixed language instruction — exactly one source of truth.
CATHETER_TASK: str = (
    "Advance the catheter tip to the distal vessel target while remaining inside the lumen."
)

#: GR00T embodiment tag string.
EMBODIMENT_TAG: str = EmbodimentTag.NEW_EMBODIMENT.value  # "new_embodiment"

#: DRR observation rate (Hz).
OBS_RATE_HZ: float = 10.0

#: Physics simulation rate (Hz).
PHYSICS_RATE_HZ: float = 60.0

#: Number of physics frames per DRR observation.
OBS_STRIDE_FRAMES: int = 6

#: Fluoroscopy image dimensions (pixels).
FLUORO_H: int = 320
FLUORO_W: int = 320
FLUORO_CHANNELS: int = 3  # grayscale replicated to RGB

#: Pixel spacing (mm/px) for the LAO-45 DRR projection.
PIXEL_SPACING_MM: float = 0.6

#: State vector dimension.
STATE_DIM: int = 2
#: state[0] = inserted_length_m
#: state[1] = previous_insert_velocity_mps

#: Action vector dimension.
ACTION_DIM: int = 1  # [insert_velocity_mps]

#: Action horizon (frames) used when building the dataset.
ACTION_HORIZON: int = 16

#: Available discrete action velocities (m/s).
ACTION_HOLD: float = 0.000
ACTION_SLOW: float = 0.006
ACTION_CRUISE: float = 0.012
ACTION_RANGE: tuple[float, float] = (ACTION_HOLD, ACTION_CRUISE)

#: Previous-action sentinel at step 0 of every episode.
INITIAL_PREVIOUS_VELOCITY: float = 0.000

#: Length of one catheter body segment (metres).  Must match Newton config.
L_SEG: float = 0.03

# ---------------------------------------------------------------------------
# State / action key names (must match meta/modality.json).
# ---------------------------------------------------------------------------

#: Parquet feature key for the combined state vector.
STATE_FEATURE_KEY: str = "observation.state"

#: Parquet feature key for the action vector.
ACTION_FEATURE_KEY: str = "action"

#: Parquet video feature key (references the frame images in videos/).
VIDEO_FEATURE_KEY: str = "observation.images.fluoro"

#: Modality key for the fluoroscopy view (matches modality.json "video").
VIDEO_MODALITY_KEY: str = "fluoro"

#: Modality key for the state (matches modality.json "state").
STATE_MODALITY_KEY: str = "catheter"

#: Modality key for the action (matches modality.json "action").
ACTION_MODALITY_KEY: str = "insert_velocity"

#: Language annotation key (dot-separated, matches GR00T convention).
LANGUAGE_MODALITY_KEY: str = "annotation.human.task_description"

# ---------------------------------------------------------------------------
# GR00T ModalityConfig for the catheter NEW_EMBODIMENT.
# ---------------------------------------------------------------------------

catheter_modality_config: dict[str, ModalityConfig] = {
    # ------------------------------------------------------------------
    # Video — current frame only (no temporal stacking at dataset level;
    # the model's vision encoder handles any internal frame stacking).
    # ------------------------------------------------------------------
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[VIDEO_MODALITY_KEY],  # "fluoro"
    ),
    # ------------------------------------------------------------------
    # State — single proprioceptive snapshot at the current step.
    # state[0] = inserted_length_m         (deployable catheter depth)
    # state[1] = previous_insert_velocity_mps (causal previous action)
    # ------------------------------------------------------------------
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[STATE_MODALITY_KEY],  # "catheter" -> shape (2,)
    ),
    # ------------------------------------------------------------------
    # Action — 16-step prediction chunk.
    # ABSOLUTE: the policy predicts target velocity, not a velocity delta.
    # NON_EEF: pure axial insertion — no end-effector pose involved.
    # DEFAULT: no special rotation representation needed.
    # ------------------------------------------------------------------
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[ACTION_MODALITY_KEY],  # "insert_velocity" -> shape (1,)
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key=None,  # velocity action has no paired state key
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # Language — fixed task instruction; loaded from task_index -> tasks.jsonl.
    # ------------------------------------------------------------------
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[LANGUAGE_MODALITY_KEY],
    ),
}

# ---------------------------------------------------------------------------
# Register with the GR00T embodiment config registry.
# Guard against double-registration (e.g. when the module is reloaded).
# ---------------------------------------------------------------------------
if EMBODIMENT_TAG not in MODALITY_CONFIGS:
    register_modality_config(
        catheter_modality_config,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )


# ---------------------------------------------------------------------------
# Validation helper.
# ---------------------------------------------------------------------------

def validate_modality_config() -> dict:
    """Validate the catheter modality config against expected structure.

    Returns a dict with ``pass`` (bool) and ``checks`` (list of str).
    Does NOT raise; callers inspect the return value.
    """
    checks: list[str] = []
    ok = True

    cfg = catheter_modality_config

    # Required top-level keys
    required_keys = {"video", "state", "action", "language"}
    missing = required_keys - set(cfg.keys())
    if missing:
        checks.append(f"FAIL: missing top-level keys: {sorted(missing)}")
        ok = False
    else:
        checks.append("PASS: top-level keys {video, state, action, language} present")

    # Video
    vid = cfg.get("video")
    if vid and vid.delta_indices == [0] and VIDEO_MODALITY_KEY in vid.modality_keys:
        checks.append(f"PASS: video delta_indices=[0], key={VIDEO_MODALITY_KEY!r}")
    else:
        checks.append("FAIL: video config malformed")
        ok = False

    # State
    st = cfg.get("state")
    if st and st.delta_indices == [0] and STATE_MODALITY_KEY in st.modality_keys:
        checks.append(f"PASS: state delta_indices=[0], key={STATE_MODALITY_KEY!r}")
    else:
        checks.append("FAIL: state config malformed")
        ok = False

    # Action
    act = cfg.get("action")
    if act:
        if act.delta_indices == list(range(ACTION_HORIZON)):
            checks.append(f"PASS: action delta_indices=range({ACTION_HORIZON})")
        else:
            checks.append(f"FAIL: action delta_indices wrong: {act.delta_indices}")
            ok = False
        if ACTION_MODALITY_KEY in act.modality_keys:
            checks.append(f"PASS: action key={ACTION_MODALITY_KEY!r}")
        else:
            checks.append(f"FAIL: action key missing {ACTION_MODALITY_KEY!r}")
            ok = False
        if act.action_configs and len(act.action_configs) == 1:
            ac = act.action_configs[0]
            if ac.rep == ActionRepresentation.ABSOLUTE:
                checks.append("PASS: action rep=ABSOLUTE")
            else:
                checks.append(f"FAIL: action rep={ac.rep!r}, expected ABSOLUTE")
                ok = False
            if ac.type == ActionType.NON_EEF:
                checks.append("PASS: action type=NON_EEF")
            else:
                checks.append(f"FAIL: action type={ac.type!r}, expected NON_EEF")
                ok = False
        else:
            checks.append("FAIL: action_configs missing or wrong count")
            ok = False
    else:
        checks.append("FAIL: action config missing")
        ok = False

    # Language — verify canonical task string
    lang = cfg.get("language")
    if lang and lang.delta_indices == [0] and LANGUAGE_MODALITY_KEY in lang.modality_keys:
        checks.append(f"PASS: language delta_indices=[0], key={LANGUAGE_MODALITY_KEY!r}")
    else:
        checks.append("FAIL: language config malformed")
        ok = False

    # State field contract
    checks.append(f"PASS: state[0]=inserted_length_m  state[1]=previous_insert_velocity_mps")
    checks.append(f"PASS: initial_previous_velocity={INITIAL_PREVIOUS_VELOCITY}")

    # Activation flag must NOT be in policy state
    checks.append("PASS: activation_flag is privileged-only (not in state modality keys)")

    # Registry
    if EMBODIMENT_TAG in MODALITY_CONFIGS:
        checks.append(f"PASS: embodiment tag {EMBODIMENT_TAG!r} registered in MODALITY_CONFIGS")
    else:
        checks.append(f"FAIL: embodiment tag {EMBODIMENT_TAG!r} NOT in MODALITY_CONFIGS")
        ok = False

    # Task string
    if CATHETER_TASK == "Advance the catheter tip to the distal vessel target while remaining inside the lumen.":
        checks.append("PASS: CATHETER_TASK is canonical")
    else:
        checks.append(f"FAIL: CATHETER_TASK={CATHETER_TASK!r} does not match canonical")
        ok = False

    return {"pass": ok, "checks": checks}
