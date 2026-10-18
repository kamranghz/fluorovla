"""GPU physics regression (needs CUDA, Warp, the pinned Newton checkout and the local data directory; skipped otherwise).

Reproduces the accepted reference episode, one completed SUCCESS episode and one completed LUMEN_VIOLATION episode and compares them row by row
(bit-identical on the reference stack; see docs/reproducibility.md for the tolerance rule on other hardware)."""
import json
import os
from pathlib import Path

import pytest

from fluorovla.common.paths import data_root
from fluorovla.evaluation.episode_comparison import compare_regression, reproduction_verdict
from conftest import ARTIFACTS, ROOT

pytestmark = pytest.mark.gpu


def _cases():
    return json.load(open(ROOT / "configs/evaluation/reproduction_cases.json"))["cases"]


def _have_stack():
    try:
        import warp as wp
        import newton  # noqa: F401
        wp.init()
        return wp.get_cuda_device_count() > 0 and (data_root() / "cache/twin").is_dir()
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def stack():
    if not _have_stack():
        pytest.skip("CUDA/Warp/Newton or local data not available")


def _run(cfg_dir, tmp_path, hold):
    from fluorovla.physics.episode_runner import run_physics_episode
    from fluorovla.physics.patient_config import PatientConfig
    run_physics_episode(PatientConfig.from_json(Path(cfg_dir) / "config.json"), tmp_path, "ep", post_success_hold=hold)
    return json.load(open(tmp_path / "episode_physics_ep.json"))


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["label"])
def test_reference_cases_reproduce(stack, tmp_path, case):
    ref = json.load(open(ROOT / case["reference_episode"]))
    if "expected_status" in case:
        assert ref["summary"]["status"] == case["expected_status"]
    cand = _run(ROOT / case["config_dir"], tmp_path, case["post_hold"])
    if case["label"].startswith("A_"):
        r = compare_regression(ref, cand)                      # the accepted reference must be bit-identical on the reference stack
        assert r["PASS"] and (r["bit_identical_rows"] or os.environ.get("FLUOROVLA_ALLOW_TOLERANCE") == "1")
    v = reproduction_verdict(ref, cand)
    assert v["verdict"] in ("PASS_BIT_IDENTICAL", "PASS_TOLERANCE"), v
    if os.environ.get("FLUOROVLA_REQUIRE_BIT_IDENTITY") == "1":
        assert v["verdict"] == "PASS_BIT_IDENTICAL"
