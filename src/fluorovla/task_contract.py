"""Target contract v1 for the insertion-only catheter task (task metadata; no physics).

Everything here is computed from the TASK-CONSTRUCTION privilege (patient centerline graph,
physical catheter state). None of it is a policy input.

Units: millimetres for geometry, seconds for time, unless a name ends in ``_mps``.

Constants are derived from validated quantities, not chosen freely:

  V_MAX_MPS, V_SLOW_MPS   executable action magnitudes (CatheterAction: 0.012 / 0.006 m/s)
  DECISION_DT_S           1 / 10 Hz fluoroscopy / decision rate (6 physics frames at 60 Hz)
  L_SEG_MM                material activation segment (30 mm)
  EPS_MM                  success half-window = V_MAX * DECISION_DT  (1.2 mm): the largest tip
                          advance in one decision, so a velocity policy cannot step over the window.
  DWELL_STEP_MM           a step counts as stationary if |ds| < V_SLOW*DT/2 (0.3 mm): midpoint
                          between 0 and the smallest non-zero commanded advance (0.6 mm).
  DWELL_STEPS             smallest integer K with K*DT > window crossing time at V_MAX
                          (2*EPS/V_MAX = 0.2 s)  ->  K = 3.
  T_MAX_STEPS             30 decisions = 3.0 s = the validated CRUISE episode (36 mm of feed).
  LUMEN_DELTA_MM          1.5 mm = half the 3 mm vessel collision-mesh voxel (surface uncertainty).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

V_MAX_MPS = 0.012
V_SLOW_MPS = 0.006
DECISION_DT_S = 0.1
PHYSICS_FRAMES_PER_DECISION = 6
L_SEG_MM = 30.0
EPS_MM = V_MAX_MPS * DECISION_DT_S * 1000.0                                  # 1.2
DWELL_STEP_MM = 0.5 * V_SLOW_MPS * DECISION_DT_S * 1000.0                    # 0.3
DWELL_STEPS = int(math.floor(2.0 * EPS_MM / (V_MAX_MPS * 1000.0) / DECISION_DT_S)) + 1   # 3
T_MAX_STEPS = 30
LUMEN_DELTA_MM = 1.5
CATASTROPHIC_VEL_MPS = 2.0
ROOT_MOVE_TOL_M = 1e-6
SMOOTH_SIGMA_MM = 3.0                                                         # = collision-mesh voxel
RESAMPLE_STEP_MM = 0.25

STATUSES = ("SUCCESS", "STOPPED_SHORT", "OVERSHOOT", "LUMEN_VIOLATION", "TIMEOUT", "PHYSICS_FAILURE")


# ── centerline graph helpers (cross-patient: topology only, no coordinates, no tangent constants) ──
def build_adjacency(edges: np.ndarray, n: int) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b in np.asarray(edges, dtype=np.int64):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def bfs_path(adj: dict[int, set[int]], s: int, t: int) -> list[int]:
    prev: dict[int, int | None] = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    out, u = [], t
    while u is not None:
        out.append(u)
        u = prev[u]
    return out[::-1]


def first_junction(adj: dict[int, set[int]], entry: int) -> int | None:
    seen, q = {entry}, deque([entry])
    while q:
        u = q.popleft()
        if len(adj[u]) >= 3:
            return u
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return None


def _gauss_smooth(a: np.ndarray, sigma_samples: float) -> np.ndarray:
    r = int(math.ceil(4 * sigma_samples))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    pad = np.pad(a, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(a.shape[1])], axis=1)


class PathProjector:
    """Smoothed centerline path. __call__(x_mm) -> (s_mm, distance_mm, local_radius_mm)."""

    def __init__(self, nodes_mm: np.ndarray, radii_mm: np.ndarray,
                 sigma_mm: float = SMOOTH_SIGMA_MM, step_mm: float = RESAMPLE_STEP_MM):
        nodes_mm = np.asarray(nodes_mm, dtype=np.float64)
        s_raw = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(nodes_mm, axis=0), axis=1))]
        grid = np.arange(0.0, s_raw[-1], step_mm)
        res = np.stack([np.interp(grid, s_raw, nodes_mm[:, i]) for i in range(3)], axis=1)
        rr = np.interp(grid, s_raw, np.asarray(radii_mm, dtype=np.float64))
        self.p = _gauss_smooth(res, sigma_mm / step_mm)
        self.r = rr
        d = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.s = np.r_[0.0, np.cumsum(d)]
        self.a, self.ab = self.p[:-1], np.diff(self.p, axis=0)
        self.ab2 = (self.ab ** 2).sum(1)
        self.raw_arclength_mm = float(s_raw[-1])

    def __call__(self, x_mm: Sequence[float]) -> tuple[float, float, float]:
        x = np.asarray(x_mm, dtype=np.float64).reshape(3)
        t = np.clip(((x - self.a) * self.ab).sum(1) / self.ab2, 0.0, 1.0)
        proj = self.a + t[:, None] * self.ab
        dist = np.linalg.norm(x - proj, axis=1)
        k = int(dist.argmin())
        return (float(self.s[k] + t[k] * math.sqrt(self.ab2[k])), float(dist[k]),
                float(self.r[k] + t[k] * (self.r[min(k + 1, len(self.r) - 1)] - self.r[k])))

    def point_at(self, s_mm: float) -> np.ndarray:
        k = int(np.clip(np.searchsorted(self.s, s_mm) - 1, 0, len(self.ab) - 1))
        t = float(np.clip((s_mm - self.s[k]) / math.sqrt(self.ab2[k]), 0.0, 1.0))
        return self.a[k] + t * self.ab[k]


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    s_target_mm: float
    target_ct_mm: tuple[float, float, float]
    s_start_mm: float
    d_target_mm: float
    eps_mm: float = EPS_MM
    dwell_steps: int = DWELL_STEPS
    dwell_step_mm: float = DWELL_STEP_MM
    t_max_steps: int = T_MAX_STEPS
    lumen_delta_mm: float = LUMEN_DELTA_MM
    path_length_to_first_junction_mm: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_path(points_mm: np.ndarray, edges: np.ndarray, radii_mm: np.ndarray,
                portal_ct_mm: np.ndarray) -> tuple[PathProjector, dict[str, Any]]:
    """Entry -> first junction path, topology only. Direction is unambiguous when the entry node is a
    graph endpoint; a non-endpoint entry would need the portal axis as a tie-break."""
    adj = build_adjacency(edges, len(points_mm))
    entry = int(np.linalg.norm(points_mm - np.asarray(portal_ct_mm), axis=1).argmin())
    junction = first_junction(adj, entry)
    path = bfs_path(adj, entry, junction)[:-1]                    # stop before the first degree>=3 node
    proj = PathProjector(points_mm[path], radii_mm[path])
    return proj, {"entry_node": entry, "entry_degree": len(adj[entry]), "first_junction_node": junction,
                  "path_nodes": path, "smoothed_path_length_mm": float(proj.s[-1]),
                  "skeleton_path_length_mm": proj.raw_arclength_mm}


def build_target(proj: PathProjector, s_start_mm: float, d_target_mm: float = L_SEG_MM,
                 target_id: str = "pre_junction_path_waypoint_1seg") -> TargetSpec:
    s_t = s_start_mm + d_target_mm
    tgt = proj.point_at(s_t)
    return TargetSpec(target_id=target_id, s_target_mm=float(s_t), target_ct_mm=tuple(float(x) for x in tgt),
                      s_start_mm=float(s_start_mm), d_target_mm=float(d_target_mm),
                      path_length_to_first_junction_mm=float(proj.s[-1]))


def evaluate(spec: TargetSpec, s_end_mm: Sequence[float], s_start_mm: float,
             violation_mm: Sequence[float], finite: Sequence[bool], max_speed_mps: Sequence[float],
             root_delta_m: float) -> dict[str, Any]:
    """Evaluate a trajectory from PHYSICAL state only, one entry per decision step.

    s_end_mm[k]        tip arc length on the smoothed centerline at the end of decision step k
    violation_mm[k]    max SDF excess of catheter axis samples over the substeps of step k
    Not used: commanded velocity / insertion, action history, target label.
    Terminal priority: PHYSICS_FAILURE > LUMEN_VIOLATION > OVERSHOOT > SUCCESS > (T_max) STOPPED_SHORT/TIMEOUT.
    """
    s = np.asarray(s_end_mm, dtype=float)
    prev = np.r_[s_start_mm, s[:-1]]
    ds = np.abs(s - prev)
    dwell_run = 0
    for k in range(min(len(s), spec.t_max_steps)):
        if (not finite[k]) or max_speed_mps[k] > CATASTROPHIC_VEL_MPS or root_delta_m > ROOT_MOVE_TOL_M:
            return {"status": "PHYSICS_FAILURE", "step": k}
        if violation_mm[k] > spec.lumen_delta_mm:
            return {"status": "LUMEN_VIOLATION", "step": k}
        if s[k] > spec.s_target_mm + spec.eps_mm:
            return {"status": "OVERSHOOT", "step": k, "s_mm": float(s[k])}
        in_win = abs(s[k] - spec.s_target_mm) <= spec.eps_mm
        dwell_run = dwell_run + 1 if (in_win and ds[k] < spec.dwell_step_mm) else 0
        if dwell_run >= spec.dwell_steps:
            return {"status": "SUCCESS", "step": k, "s_mm": float(s[k]),
                    "error_mm": float(s[k] - spec.s_target_mm)}
    n = min(len(s), spec.t_max_steps)
    if n < spec.t_max_steps:
        return {"status": "INCOMPLETE", "step": n - 1}
    k = n - 1
    stationary = bool(all(ds[max(0, k - spec.dwell_steps + 1):k + 1] < spec.dwell_step_mm))
    if s[k] < spec.s_target_mm - spec.eps_mm and stationary:
        return {"status": "STOPPED_SHORT", "step": k, "s_mm": float(s[k]),
                "short_by_mm": float(spec.s_target_mm - s[k])}
    return {"status": "TIMEOUT", "step": k, "s_mm": float(s[k])}
