"""CLI contract of tools/audit_containment.py: local selftest works, evidence modes refuse non-Linux, engineering calibration can never be a locked calibration."""
import importlib.util
import json
import sys

import numpy as np
import pytest

from conftest import ROOT


@pytest.fixture(scope="module")
def cli():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("audit_containment", ROOT / "tools/audit_containment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_selftest_runs_anywhere_and_labels_itself_engineering_only(cli, tmp_path, capsys):
    assert cli.main(["selftest", "--out", str(tmp_path / "s.json")]) == 0
    assert "ENGINEERING VALIDATION ONLY" in capsys.readouterr().out
    assert json.load(open(tmp_path / "s.json"))["engineering_validation_only"] is True


@pytest.mark.parametrize("argv", [["preflight", "--configs-root", "c", "--out", "o"], ["calibrate", "--configs-root", "c", "--out", "o"],
                                  ["trace", "--configs-root", "c", "--calibration", "k", "--out", "o"], ["analyze", "--configs-root", "c", "--trace-dir", "t", "--calibration", "k", "--out", "o"],
                                  ["verdict", "--analysis-dir", "a", "--calibration", "k", "--out", "o"]])
def test_evidence_modes_refuse_non_linux(cli, monkeypatch, tmp_path, argv):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2 and not (tmp_path / "o").exists()


def test_design_config_declares_cases_and_synthetic_design_before_any_run(cli):
    d = cli.load_design(None)
    prim = [c["name"] for c in d["trace_cases"] if c["tier"] == "primary"]
    assert prim == ["s0011_right_F0", "s0963_left_F1", "s0963_left_F2", "s0344_left_F1"]
    supp = [c for c in d["trace_cases"] if c["tier"] == "supplementary"]
    assert len(supp) == 1 and "1cb5008" in supp[0]["provenance"] and "outcome" not in supp[0]["reason"].replace("any containment-audit statistic", "")
    cal = d["numerical_calibration_design"]
    assert {"seed", "radius_bins", "variant_selection_rule", "bends", "rotations_deg", "voxel_offset_fractions_xyz", "evaluation_band_mm"} <= set(cal) and "tube_radius_mm" not in cal
    from fluorovla.anatomy import containment_audit as ca
    assert cal["variant_selection_rule"] == ca.VARIANT_SELECTION_RULE and ca.design_bins(cal)                    # rule and contiguous radius bins are declared in the file, before any run
    bins = ca.design_bins(cal)
    assert bins[0]["lo_mm"] == 1.5 and bins[-1]["hi_mm"] == 12.0 and {"r1.5-2.0", "r9.0-12.0"} <= {b["id"] for b in bins}          # widened coverage declared before any real-data run
    assert d["secondary_descriptive_analysis"]["provenance_class"] == "HEURISTIC" and d["secondary_descriptive_analysis"]["name"] == "SECONDARY_P99_DESCRIPTIVE"
    assert d["declared_on"] == "2026-10-28" and set(d["mechanisms"]) == {"COLLISION_DISCRETIZATION_DECISION_FLIP", "PRE_ACTION_INVALIDITY", "specificity"}
    assert [c["name"] for c in d["trace_cases"] if c["stage3a_role"] == "positive_control"] == ["s0011_right_F0", "s0963_left_F1"]          # roles are declared, not derived
    assert not any(k in cal for k in ("radius_mm", "l_seg_mm", "num_segments", "sample_frac", "mask_downsample", "lumen_delta_mm"))      # system constants are read from source, not declared here


def test_engineering_calibration_is_never_locked_and_is_rejected_by_the_lock(cli, tmp_path, monkeypatch):
    from fluorovla.evaluation.containment_verdict import CalibrationError, load_locked_calibration
    from fluorovla.anatomy import synthetic_geometry as sg
    data = tmp_path / "data"
    (data / "twin").mkdir(parents=True)
    json.dump({"spacing_zyx_mm": [1.5, 1.5, 1.5], "origin_xyz_mm": [0, 0, 0]}, open(data / "twin/metadata.json", "w"))
    cfgd = tmp_path / "configs" / "case_a"
    cfgd.mkdir(parents=True)
    json.dump({"patient_id": "synthetic", "ct_dir": "twin", "mesh_cache": "twin/m.npz", "mesh_frame": "ct", "tangent_raw": [0, 0, 1], "ct_offset_m": [0, 0, 0]}, open(cfgd / "config.json", "w"))
    monkeypatch.setenv("FLUOROVLA_DATA", str(data))
    design = json.load(open(ROOT / "configs/evaluation/containment_audit.json"))
    design["numerical_calibration_design"].update(radius_bins=[{"id": "r", "lo_mm": 2.5, "hi_mm": 3.5, "tube_radius_mm": [3.0]}], bends=[{"bend_radius_mm": None, "turn_deg": 0.0}], rotations_deg=[[0.0, 0.0, 0.0]],
                                                  voxel_offset_fractions_xyz=[[0.0, 0.0, 0.0]], points_per_case=200)
    json.dump(design, open(tmp_path / "design.json", "w"))
    out = tmp_path / "cal"
    assert cli.main(["calibrate", "--engineering-only", "--design", str(tmp_path / "design.json"), "--configs-root", str(tmp_path / "configs"), "--out", str(out)]) == 0
    f = out / "ENGINEERING_ONLY_numerical_tolerances.json"
    cal = json.load(open(f))
    assert cal["locked"] is False and cal["engineering_only"] is True and cal["spacings_zyx_mm"] == [[1.5, 1.5, 1.5]]
    assert not (out / "numerical_tolerances.json").exists()
    with pytest.raises(CalibrationError):
        load_locked_calibration(f)
    with pytest.raises(SystemExit):                                                    # evidence is never overwritten
        cli.main(["calibrate", "--engineering-only", "--design", str(tmp_path / "design.json"), "--configs-root", str(tmp_path / "configs"), "--out", str(out)])
