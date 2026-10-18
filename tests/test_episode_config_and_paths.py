"""Episode-configuration builder, PatientConfig numerics, and configuration/path handling."""
import json
import math

import numpy as np
import pytest

import fluorovla.task_contract as tc
from fluorovla.anatomy.episode_config import build_episode_config, f_range, start_depths
from fluorovla.physics.patient_config import PatientConfig
from conftest import ARTIFACTS, ROOT

REF = ARTIFACTS / "references/reproduction/completed_success"


def test_start_depth_rule():
    lo, hi = f_range(183.58)
    assert lo == 60.0 and hi == 100.0
    assert start_depths(165.26) == pytest.approx([60.0, 71.5, 83.06], abs=0.05)
    assert start_depths(100.0) == []                      # too short for the 60 mm minimum feed


def test_builder_reproduces_stored_reference_configuration(tmp_path):
    cfg_ref = json.load(open(REF / "config.json"))
    z = np.load(REF / "path_ext.npz")
    d = build_episode_config("s0963", "left", 1, cfg_ref["S_contact_mm"], z["pts"], z["rad"], tmp_path / "ep")
    got = json.load(open(d / "config.json"))
    for k in ("patient_id", "access_side", "start_index", "F_mm", "S_contact_mm", "s_portal_mm", "portal_ct_mm", "tangent", "sign_scale", "ct_dir", "mesh_frame"):
        assert got[k] == cfg_ref[k], k
    assert json.load(open(d / "landing_zone.json")) == json.load(open(REF / "landing_zone.json"))


def test_timeout_formula():
    lz = json.load(open(REF / "landing_zone.json"))
    F_c = lz["F_mm"] + tc.EPS_MM
    expect = int(math.ceil(1.25 * ((F_c - 6.0) / 12.0 + 6.0 / 6.0) / 0.1)) + 3
    assert lz["timeout"]["T_MAX_decisions"] == expect
    S = lz["S_contact_mm"]
    assert lz["landing_zone_extremity_mm"] == pytest.approx([S - 4.72, S - 2.32])


def test_patient_config_numerics_and_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUOROVLA_DATA", str(tmp_path / "data"))
    cfg = PatientConfig.from_json(ARTIFACTS / "references/accepted_reference/config.json")
    assert cfg.ct_dir == tmp_path / "data" / "cache/twin/s0011"
    t = np.array([-0.1975291222333908, 0.0, 0.9802970290184021])
    assert np.array_equal(cfg.tangent, t / np.linalg.norm(t))                      # reference constants of the accepted episode
    assert np.array_equal(cfg.ct_offset, np.array([0.061955078125, 0.201955078125, -0.0085]))
    assert cfg.landing_zone_path.parent == ARTIFACTS / "references/accepted_reference"
    c2 = PatientConfig.from_json(REF / "config.json")                              # patient configuration: portal_ct_mm / 1000, tangent normalized twice
    raw = json.load(open(REF / "config.json"))
    assert np.array_equal(c2.ct_offset, np.asarray(raw["portal_ct_mm"]) / 1000.0)


def test_no_absolute_or_platform_paths_in_source():
    bad = []
    for p in (ROOT / "src").rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if "/mnt/d" in s or "D:/" in s or "D:\\\\" in s or "C:/Users" in s:
                bad.append(f"{p.relative_to(ROOT)}:{i}")
    assert not bad, bad
