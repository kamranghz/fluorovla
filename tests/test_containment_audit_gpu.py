"""GPU / native-Linux tests of the containment probe and of the engine SDF path on synthetic geometry. Written on Windows, NOT executed there (no CUDA)."""
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
        ok = wp.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        pytest.skip("CUDA/Warp/Newton not available")


@pytest.fixture(scope="module")
def local_data(stack):
    if not (data_root() / "cache/twin").is_dir():
        pytest.skip("local data directory not available")


def _run(cfg_dir, tmp_path, hold=0):
    from fluorovla.physics.containment_probe import ContainmentProbe
    from fluorovla.physics.episode_runner import run_physics_episode
    from fluorovla.physics.patient_config import PatientConfig
    cfg = PatientConfig.from_json(Path(cfg_dir) / "config.json")
    probe = ContainmentProbe()
    s = run_physics_episode(cfg, tmp_path, "t", post_success_hold=hold, observer=probe)
    arrays, meta = probe.arrays()
    return cfg, s, arrays, meta, json.load(open(tmp_path / "episode_physics_t.json"))


def test_source_derived_constants_equal_the_running_physics_modules(stack):
    from fluorovla.anatomy import containment_audit as ca
    assert all(ca.live_constants_match(ca.canonical_constants()).values())


def test_probe_is_read_only_and_reproduces_the_engine_measure(local_data, tmp_path):
    from fluorovla.anatomy import containment_audit as ca
    from fluorovla.evaluation.episode_comparison import compare_regression
    cfg, s, arrays, meta, cand = _run(ARTIFACTS / "references/accepted_reference", tmp_path, hold=15)
    ref = json.load(open(ARTIFACTS / "references/accepted_reference/accepted_episode.json"))
    assert compare_regression(ref, cand)["bit_identical_rows"]                                   # the probe does not change the physics
    C = ca.canonical_constants()
    geom = ca.load_patient_geometry(cfg, C)
    r = ca.analyze_trace(arrays, meta, {"fullres": geom.fullres, "ds": geom.ds}, C)
    assert r["observer_consistency_max_abs_diff_mm"] < 1e-6                                      # same state, same samples, same SDF
    assert r["start_vs_analytic_straight_rod_max_mm"] < 1e-3 and r["constants_match_trace"]      # engine start equals the analytic straight placement
    assert r["key_events"]["peak"]["worst_sample_engine"]["engine_has_sdf_hit"]


def test_probe_trigger_event_of_a_completed_lumen_violation(local_data, tmp_path):
    from fluorovla.anatomy import containment_audit as ca
    cfg, s, arrays, meta, _ = _run(ARTIFACTS / "references/reproduction/completed_lumen_violation", tmp_path)
    assert s["status"] == "LUMEN_VIOLATION"
    C = ca.canonical_constants()
    geom = ca.load_patient_geometry(cfg, C)
    r = ca.analyze_trace(arrays, meta, {"fullres": geom.fullres, "ds": geom.ds}, C)
    assert r["key_events"]["trigger"]["engine_violation_mm"] > C["lumen_delta_mm"]


def test_engine_sdf_path_on_a_synthetic_twin_is_bounded_and_leaves_the_data_cache_alone(stack, tmp_path):
    from fluorovla.anatomy import containment_audit as ca
    from fluorovla.anatomy import synthetic_geometry as sg
    from fluorovla.physics.containment_probe import synthetic_engine_factory
    C = ca.canonical_constants()
    cache = data_root() / "cache"
    before = sorted(str(p) for p in cache.rglob("*")) if cache.is_dir() else []
    tube = sg.TubeSpec(3.5, 40.0, 40.0, 30.0, 70.0, (25.0, 40.0, 15.0), (10.0, -20.0, 5.0))
    sp = (1.5, 1.5, 1.5)
    mask, origin, spv = sg.voxelize(tube, sp, 15.0, (0.31, 0.17, 0.43))
    phi_fn = synthetic_engine_factory(tmp_path)(mask, origin, spv)
    pts = sg.random_points_near_surface(tube, 500, 6.0, np.random.default_rng(0))
    err = phi_fn(pts) + tube.sd_mm(pts)
    assert np.abs(err).max() <= np.linalg.norm(sp) * C["mask_downsample"]                       # stride-subsampled surface: within one stride-voxel diagonal
    assert (tmp_path / "synthetic_twin_00000" / "collision_mesh.npz").is_file()
    assert (sorted(str(p) for p in cache.rglob("*")) if cache.is_dir() else []) == before
