"""CPU test of the read-only containment probe with a stub episode (the real episode is exercised by the GPU test)."""
import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("newton")
from fluorovla.physics import containment_probe as cp  # noqa: E402
from fluorovla.physics.patient_episode import axis_sample_points  # noqa: E402


class _Arr:
    def __init__(self, a):
        self.a = a

    def numpy(self):
        return self.a


class _Sdf:
    meta = {"n_vertices": 10, "n_faces": 20, "downsample": 2, "mesh_frame": "solver_local", "extractor": "stub"}

    def query(self, pts):
        phi = pts[:, 0] - 0.003                                   # wall at x = 3 mm
        return phi, np.zeros((len(pts), 3)), np.ones(len(pts), np.int32)


class _Cfg:
    patient_id, access_side, sign_scale = "synthetic", "left", 1.0
    ct_offset = np.array([0.1, 0.2, 0.3])
    tangent = np.array([0.0, 0.0, 1.0])


class _Ep:
    def __init__(self):
        self.cfg, self.sdf = _Cfg(), _Sdf()
        self.bodies, self.root, self.n_active, self.axis = [10, 11, 12, 13, 14, 15, 16], 10, 3, np.array([0.0, 0.0, 1.0])
        q = np.zeros((20, 7))
        q[:, 6] = 1.0
        for i, b in enumerate(self.bodies):
            q[b, :3] = [0.001 * i, 0.0, 0.03 * i]
        self.state_0 = type("S", (), {"body_q": _Arr(q)})()
        self.contacts = type("C", (), {"contact_counters": _Arr(np.array([4, 0]))})()


def test_probe_records_canonical_samples_without_touching_the_state():
    ep = _Ep()
    before = ep.state_0.body_q.a.copy()
    p = cp.ContainmentProbe()
    p("build_scene", ep, {})
    p("settle1_substep", ep, {"k": 4})
    p("settle1_substep", ep, {"k": 3})                           # not the last substep of a frame: not recorded
    p("ready", ep, {})
    for sub in range(61):
        p("decision_substep", ep, {"frame": sub // 5, "sub": sub % 5, "violation_m": 0.0005})
    arrays, meta = p.arrays()
    assert np.array_equal(ep.state_0.body_q.a, before)
    assert arrays["stage_code"].tolist()[:3] == [cp.STAGE_NAMES.index(s) for s in ("build_scene", "settle1_frame", "ready")] and meta["n_events"] == 3 + 61
    assert arrays["decision"][3:].tolist() == [0] * 30 + [1] * 30 + [2]                     # 30 substeps per decision
    expected = axis_sample_points(ep.state_0.body_q.a, ep.bodies, 3, ep.root, ep.axis)
    n = len(expected)
    assert n == 10 and np.allclose(arrays["sample_points_m"][0, :n], np.stack([pt for _b, _f, pt in expected]))
    assert arrays["sample_body"][0, :n].tolist() == [1] * 5 + [2] * 5 and np.all(arrays["sample_body"][0, n:] == -1)
    assert np.allclose(arrays["phi_engine_m"][0, :n], arrays["sample_points_m"][0, :n, 0] - 0.003)
    assert arrays["body_q"].shape == (64, 7, 7) and arrays["contact_count"][0] == 4
    assert meta["root_rod_index"] == 0 and meta["moving_rod_indices"] == [1, 2, 3, 4, 5, 6] and meta["l_seg_m"] == pytest.approx(0.03)
