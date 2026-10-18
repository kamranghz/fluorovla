"""Cohort-lock integrity and exact reproduction of the locked development cohort from the stored descriptor table."""
import json

from fluorovla.anatomy.cohort_selection import select_development_cohort
from conftest import ARTIFACTS, ROOT

LOCKED = ["s0011", "s0963", "s1400", "s1364", "s0950", "s0344", "s1388", "s1348", "s1397", "s0513", "s1369", "s0573"]


def test_lock_contents(cohort_lock):
    assert cohort_lock["status"] == "LOCKED"
    assert cohort_lock["development_cohort"] == LOCKED
    assert cohort_lock["stress_sentinel"] == "s1082"
    assert cohort_lock["candidate_pool_n"] == 65 == len(cohort_lock["candidate_pool"])
    pool = set(cohort_lock["candidate_pool"])
    assert set(LOCKED) <= pool and "s1082" in pool and "s1082" not in LOCKED
    assert set(cohort_lock["returned_to_ordinary_pool"]) == {"s1391", "s0058"}
    assert not set(cohort_lock["returned_to_ordinary_pool"]) & set(LOCKED)
    assert "s1395" in cohort_lock["excluded_from_69_build"]


def test_selection_reproduces_lock(cohort_lock, descriptor_rows):
    cfg = json.load(open(ROOT / "configs/anatomy/cohort_selection.json"))
    res = select_development_cohort(descriptor_rows, cfg["cohort_size"], cfg["reference_patient"])
    assert res["cohort"] == cohort_lock["development_cohort"]


def test_pool_matches_geometry_artifact(cohort_lock):
    pool = json.load(open(ARTIFACTS / "references/candidate_pool_geometry/candidate_pool.json"))
    assert pool == cohort_lock["candidate_pool"]


def test_stored_selection_artifact_matches_lock(cohort_lock):
    sel = json.load(open(ARTIFACTS / "summaries/cohort_selection.json"))
    assert sel["curve"]["12"]["cohort"] == cohort_lock["development_cohort"]
