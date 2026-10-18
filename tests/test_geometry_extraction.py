"""Twin access-path geometry: synthetic-graph unit tests and agreement with the authoritative geometry artifact."""
import json

import numpy as np
import pytest

from fluorovla.anatomy import access_geometry as ag
from conftest import ARTIFACTS


def _y_graph():
    """Aorta (z up from the bifurcation), two iliac arms leaving at +-x; 1.5 mm node spacing."""
    pts, edges, rad = [], [], []

    def add(p, r):
        pts.append(p)
        rad.append(r)
        return len(pts) - 1

    c = add(np.array([0.0, 0.0, 0.0]), 8.0)                       # junction node
    prev = c
    for i in range(1, 120):                                       # aorta upward
        n = add(np.array([0.0, 0.0, 1.5 * i]), 9.0)
        edges.append((prev, n))
        prev = n
    for sign in (-1.0, 1.0):                                      # two iliac arms downward, diverging
        prev = c
        for i in range(1, 140):
            n = add(np.array([sign * 0.5 * 1.5 * i, 0.0, -1.5 * i]), 4.5)
            edges.append((prev, n))
            prev = n
    return np.array(pts), np.array(edges), np.array(rad)


def _entries(pts, edges, rad):
    import fluorovla.task_contract as tc
    adj = tc.build_adjacency(edges, len(pts))
    adj, _ = ag.prune_spurs(adj, pts)
    ends = [n for n, v in adj.items() if len(v) == 1]
    return {e: ag.access_from_entry(pts, rad, adj, e) for e in ends}


def test_clean_y_has_two_valid_iliac_paths():
    pts, edges, rad = _y_graph()
    ent = _entries(pts, edges, rad)
    iliac = [g for e, g in ent.items() if pts[e][2] < 0 and g["valid"] and g["S_contact_mm"] > 150]    # the aorta-top endpoint is a (non-access) path too
    valid = iliac
    assert len(valid) == 2
    assert all(abs(g["S_contact_mm"] - valid[0]["S_contact_mm"]) < 1.0 for g in valid)


def test_prune_spurs_removes_short_leaf_only():
    import fluorovla.task_contract as tc
    pts = np.array([[0, 0, 1.5 * i] for i in range(60)] + [[1.5 * i, 0, 45.0] for i in range(1, 5)], dtype=float)
    edges = [(i, i + 1) for i in range(59)] + [(30, 60)] + [(60 + i, 61 + i) for i in range(3)]
    adj = tc.build_adjacency(np.array(edges), len(pts))
    adj2, removed = ag.prune_spurs(adj, pts)
    assert removed == 4 and all(len(adj2[n]) <= 2 for n in adj2)


def test_authoritative_geometry_artifact_is_consistent(cohort_lock):
    sides = json.load(open(ARTIFACTS / "references/candidate_pool_geometry/sides.json"))
    for pid in cohort_lock["candidate_pool"]:
        ok = [s for s in sides[pid].values() if s.get("valid") and s["S_contact_mm"] >= 150.0]
        assert ok, pid
    assert sides["s0011"]["right"]["S_contact_mm"] == pytest.approx(183.5844, abs=0.01)        # reference construction


def test_recovered_topology_cases_have_valid_paths():
    """s0815 / s1369 (skeleton-cycle stubs) and s1379 (empty-window bug) must yield valid access paths; s1395 stays without one under the current contract."""
    sides = json.load(open(ARTIFACTS / "references/candidate_pool_geometry/sides.json"))
    for pid in ("s0815", "s1369", "s1379"):
        assert any(s.get("valid") and s["S_contact_mm"] >= 150.0 for s in sides[pid].values()), pid
    assert not any(s.get("valid") and s["S_contact_mm"] >= 150.0 for s in sides.get("s1395", {}).values())
