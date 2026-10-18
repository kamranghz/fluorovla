"""Privileged-information firewall: what the policy sees versus what only the Privileged Teacher / evaluation may use."""
import ast
import re

from fluorovla.teacher.privileged_teacher import POLICY_OBSERVATION_KEYS, TEACHER_ONLY_KEYS, TeacherState
from conftest import ROOT

SCHEMA = ROOT / "src/fluorovla/data/episode_schema.py"
PRIVILEGED_FIELDS = ["n_active_bodies", "activation_flag", "accepted_state_id", "physics_frame_index", "polyline_solver_m", "tip_px", "gt_delta_L_m", "cath_contribution_L1", "body_q"]


def test_policy_and_teacher_keys_disjoint():
    assert not set(POLICY_OBSERVATION_KEYS) & set(TEACHER_ONLY_KEYS)
    assert set(POLICY_OBSERVATION_KEYS) == {"observation.images.fluoro", "inserted_length_m", "previous_insert_velocity_mps", "annotation.human.task_description"}


def test_policy_keys_contain_no_privileged_names():
    for k in POLICY_OBSERVATION_KEYS:
        assert not k.startswith("priv_") and k not in PRIVILEGED_FIELDS and "centerline" not in k and "extremity" not in k


def test_teacher_input_is_the_declared_privileged_triple():
    assert list(TeacherState.__dataclass_fields__) == ["tip_extremity_mm", "zone_near_mm", "zone_far_mm"]


def test_parquet_columns_are_policy_visible_only():
    tree = ast.parse(SCHEMA.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_write_parquet")
    keys = {c.value for n in ast.walk(fn) if isinstance(n, ast.Dict) for c in n.keys if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    assert {"action", "observation.state"} <= keys
    assert not keys & set(PRIVILEGED_FIELDS)
    src = ast.get_source_segment(SCHEMA.read_text(encoding="utf-8"), fn)
    assert not re.search(r"activation_flag\s*[\]\)]", src.replace("# activation_flag is privileged and NOT included here", "")) or "step.activation_flag" not in src


def test_physics_rows_use_priv_prefix_for_privileged_quantities():
    runner = (ROOT / "src/fluorovla/physics/episode_runner.py").read_text(encoding="utf-8")
    for key in ("priv_tip_extremity_mm", "priv_tip_polyline_s_mm", "priv_tip_dist_centerline_mm", "priv_local_radius_mm"):
        assert key in runner
