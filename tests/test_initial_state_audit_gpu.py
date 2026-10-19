"""GPU tests of the audit observer: it must be read-only (bit-identical episodes) and consistent with the engine's own violation measure."""
import json
from pathlib import Path

import numpy as np
import pytest

from fluorovla.common.paths import data_root
from conftest import ARTIFACTS

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def stack():
    try:
        import warp as wp
        import newton  # noqa: F401
        wp.init()
        ok = wp.get_cuda_device_count() > 0 and (data_root() / "cache/twin").is_dir()
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        pytest.skip("CUDA/Warp/Newton or local data not available")


def _run(cfg_dir, tmp_path, hold=0, cap=None):
    import fluorovla.task_contract as tc
    from fluorovla.physics.episode_runner import run_physics_episode
    from fluorovla.physics.initial_state_audit import InitialStateObserver
    from fluorovla.physics.patient_config import PatientConfig
    cfg = PatientConfig.from_json(Path(cfg_dir) / "config.json")
    z = np.load(cfg.path_npz_path)
    obs = InitialStateObserver(tc.PathProjector(z["pts"], z["rad"]))
    s = run_physics_episode(cfg, tmp_path, "t", post_success_hold=hold, observer=obs, decision_cap=cap)
    return s, obs, json.load(open(tmp_path / "episode_physics_t.json"))


def test_axis_sample_points_match_measure_outside():
    from fluorovla.physics.patient_episode import axis_sample_points, measure_outside
    q = np.zeros((4, 7))
    q[:, 6] = 1.0
    q[:, 2] = np.arange(4) * 0.03
    q[:, 0] = np.array([0.0, 0.0, 0.002, 0.004])

    class Stub:
        def query(self, pts):
            return pts[:, 0] - 0.003, None, None

    pts = [p for _b, _f, p in axis_sample_points(q, [0, 1, 2, 3], 4, 0, np.array([0.0, 0.0, 1.0]))]
    assert len(pts) == 3 * 5                                                     # root excluded
    frac, viol = measure_outside(Stub(), q, [0, 1, 2, 3], 4, 0, np.array([0.0, 0.0, 1.0]))
    assert viol == pytest.approx(max(0.0, max(float(p[0]) - 0.003 for p in pts)), abs=1e-7)
    assert frac == pytest.approx(np.mean([float(p[0]) - 0.003 > 0 for p in pts]))


def test_observer_is_read_only_and_consistent(stack, tmp_path):
    from fluorovla.evaluation.episode_comparison import compare_regression
    ref = json.load(open(ARTIFACTS / "references/accepted_reference/accepted_episode.json"))
    s, obs, cand = _run(ARTIFACTS / "references/accepted_reference", tmp_path, hold=15)
    r = compare_regression(ref, cand)
    assert r["PASS"] and r["bit_identical_rows"]                                  # the observer does not change the physics
    assert set(obs.snapshots) == set(obs.CHECKPOINTS)
    assert obs.snapshots["ready"]["root_displacement_mm"] < 1e-3
    assert obs.analytic_init["max_lateral_offset_mm"] < 1e-3                      # engine placement equals the analytic straight rod
    assert len(obs.settle_trace["settle1"]) == 30 and len(obs.settle_trace["settle2"]) == 30


def test_violation_track_matches_engine(stack, tmp_path):
    d = ARTIFACTS / "references/reproduction/completed_lumen_violation"
    s, obs, cand = _run(d, tmp_path)
    assert s["status"] == "LUMEN_VIOLATION"
    ring = obs.summary()["last_60_substeps"]
    assert all(abs(r["engine_violation_mm"] - r["audit_max_axis_violation_mm"]) < 1e-3 for r in ring)        # same state, same samples, same SDF
    last_decision = ring[-1]["decision"]
    assert max(r["engine_violation_mm"] for r in ring if r["decision"] == last_decision) > 1.5
    worst = max(ring, key=lambda r: r["audit_max_axis_violation_mm"])
    assert worst["worst_body"] is not None and worst["worst_point_ct_mm"] is not None
