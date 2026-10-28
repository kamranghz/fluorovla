"""Analytic synthetic geometry with a known signed distance (CPU-only), used to validate and calibrate the containment-audit measurements.

A ``TubeSpec`` is a circular tube around a centerline made of a straight segment, an optional circular arc in a plane and a second straight segment. Its signed distance
(positive inside, millimetres) is exact as long as the tube radius is smaller than the bend radius. All numbers in a ``TubeSpec`` are numerical-experiment design
parameters, not simulator constants.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def rotation_matrix_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    """Rz @ Ry @ Rx for angles in degrees."""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@dataclass(frozen=True)
class TubeSpec:
    """Tube starting at ``start_mm`` heading along the rotated +z axis; the arc (if any) turns toward the rotated +x axis."""
    radius_mm: float
    straight_in_mm: float
    straight_out_mm: float = 0.0
    bend_radius_mm: float | None = None      # None: straight tube of length straight_in_mm + straight_out_mm
    turn_deg: float = 0.0
    rotation_deg: tuple = (0.0, 0.0, 0.0)
    start_mm: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.bend_radius_mm is not None and not self.radius_mm < self.bend_radius_mm:
            raise ValueError("the analytic signed distance requires tube radius < bend radius")

    @property
    def R(self) -> np.ndarray:
        return rotation_matrix_deg(*self.rotation_deg)

    def to_local(self, pts_mm: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(np.asarray(pts_mm, float)) - np.asarray(self.start_mm, float)) @ self.R

    def to_world(self, local: np.ndarray) -> np.ndarray:
        return np.atleast_2d(local) @ self.R.T + np.asarray(self.start_mm, float)

    # -- local-frame centerline -----------------------------------------------------------------------------------
    def _arc_end(self):
        Rb, th = self.bend_radius_mm, math.radians(self.turn_deg)
        p = np.array([Rb * (1 - math.cos(th)), 0.0, self.straight_in_mm + Rb * math.sin(th)])
        h = np.array([math.sin(th), 0.0, math.cos(th)])
        return p, h

    def centerline_local(self, step_mm: float = 0.05) -> np.ndarray:
        """Dense centerline points (local frame) for brute-force checks."""
        if self.bend_radius_mm is None:
            n = max(2, int(math.ceil((self.straight_in_mm + self.straight_out_mm) / step_mm)) + 1)
            z = np.linspace(0.0, self.straight_in_mm + self.straight_out_mm, n)
            return np.stack([np.zeros(n), np.zeros(n), z], axis=1)
        n1 = max(2, int(math.ceil(self.straight_in_mm / step_mm)) + 1)
        seg1 = np.stack([np.zeros(n1), np.zeros(n1), np.linspace(0.0, self.straight_in_mm, n1)], axis=1)
        na = max(2, int(math.ceil(self.bend_radius_mm * math.radians(self.turn_deg) / step_mm)) + 1)
        th = np.linspace(0.0, math.radians(self.turn_deg), na)
        arc = np.stack([self.bend_radius_mm * (1 - np.cos(th)), np.zeros(na), self.straight_in_mm + self.bend_radius_mm * np.sin(th)], axis=1)
        p, h = self._arc_end()
        n2 = max(2, int(math.ceil(self.straight_out_mm / step_mm)) + 1)
        seg2 = p + np.linspace(0.0, self.straight_out_mm, n2)[:, None] * h
        return np.vstack([seg1, arc, seg2])

    def centerline_world(self, step_mm: float = 0.05) -> np.ndarray:
        return self.to_world(self.centerline_local(step_mm))

    @staticmethod
    def _dist_segment(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        t = np.clip(((q - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        return np.linalg.norm(q - (a + t[:, None] * ab), axis=1)

    def distance_to_centerline_mm(self, pts_mm: np.ndarray) -> np.ndarray:
        q = self.to_local(pts_mm)
        if self.bend_radius_mm is None:
            return self._dist_segment(q, np.zeros(3), np.array([0.0, 0.0, self.straight_in_mm + self.straight_out_mm]))
        Rb, th_end = self.bend_radius_mm, math.radians(self.turn_deg)
        d = self._dist_segment(q, np.zeros(3), np.array([0.0, 0.0, self.straight_in_mm]))
        p_end, h = self._arc_end()
        d = np.minimum(d, self._dist_segment(q, p_end, p_end + self.straight_out_mm * h))
        c = np.array([Rb, 0.0, self.straight_in_mm])
        rel = q - c
        theta = np.arctan2(rel[:, 2], -rel[:, 0])
        on_arc = (theta >= 0.0) & (theta <= th_end)
        th = np.clip(theta, 0.0, th_end)
        foot = c + Rb * np.stack([-np.cos(th), np.zeros(len(th)), np.sin(th)], axis=1)
        d_arc = np.where(on_arc, np.linalg.norm(q - foot, axis=1), np.inf)
        return np.minimum(d, d_arc)

    def sd_mm(self, pts_mm: np.ndarray) -> np.ndarray:
        """Signed distance to the tube surface, positive inside. Exact except within the end caps (they are flat discs in this model only via the centerline distance)."""
        return self.radius_mm - self.distance_to_centerline_mm(pts_mm)

    def bounds_world(self) -> tuple[np.ndarray, np.ndarray]:
        cl = self.centerline_world(0.5)
        return cl.min(axis=0) - self.radius_mm, cl.max(axis=0) + self.radius_mm


class AnalyticField:
    """Same ``phi_mm`` interface as the grid reference (phi < 0 inside), backed by the exact tube signed distance."""
    name = "analytic"

    def __init__(self, tube: TubeSpec):
        self.tube = tube

    def phi_mm(self, pts_ct_mm: np.ndarray) -> np.ndarray:
        return -self.tube.sd_mm(pts_ct_mm)

    def inside_grid(self, pts_ct_mm: np.ndarray) -> np.ndarray:
        return np.ones(len(np.atleast_2d(pts_ct_mm)), dtype=bool)


def voxelize(tube: TubeSpec, spacing_zyx_mm, margin_mm: float, offset_fraction=(0.0, 0.0, 0.0)):
    """(mask [z, y, x] bool, origin_xyz_mm, spacing_zyx_mm): voxel centres inside the tube. ``offset_fraction`` (x, y, z) shifts the grid by that fraction of a voxel
    (sub-voxel registration of the shape against the grid)."""
    sp = np.asarray(spacing_zyx_mm, float)
    lo, hi = tube.bounds_world()
    origin = (lo - margin_mm) + np.asarray(offset_fraction, float) * sp[::-1]
    shape_zyx = np.ceil((hi + margin_mm - origin)[::-1] / sp).astype(int) + 1
    xs = origin[0] + np.arange(shape_zyx[2]) * sp[2]
    ys = origin[1] + np.arange(shape_zyx[1]) * sp[1]
    mask = np.zeros(tuple(shape_zyx), dtype=bool)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    for k in range(shape_zyx[0]):
        z = origin[2] + k * sp[0]
        pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)], axis=1)
        mask[k] = (tube.sd_mm(pts) >= 0.0).reshape(xx.shape)
    return mask, origin, sp


def random_points_near_surface(tube: TubeSpec, n: int, band_mm: float, rng: np.random.Generator, interior_margin_mm: float = 0.0) -> np.ndarray:
    """Uniform random points with |signed distance| <= band that are at least ``interior_margin_mm`` away from the tube end planes along the centerline."""
    lo, hi = tube.bounds_world()
    out = np.zeros((0, 3))
    while len(out) < n:
        cand = rng.uniform(lo - band_mm, hi + band_mm, size=(max(4 * n, 4096), 3))
        sd = tube.sd_mm(cand)
        q = tube.to_local(cand)
        total = tube.straight_in_mm + tube.straight_out_mm if tube.bend_radius_mm is None else None
        keep = np.abs(sd) <= band_mm
        if tube.bend_radius_mm is None:
            keep &= (q[:, 2] >= interior_margin_mm) & (q[:, 2] <= total - interior_margin_mm)
        out = np.vstack([out, cand[keep]])
    return out[:n]
