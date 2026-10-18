"""Deterministic representative-patient selection in anatomy-descriptor space.

Input: one row per access path (patient x side) with the seven descriptors (see ``DESC``) plus ``r_p05_mm``, ``jag`` and ``cluster_xyz`` for the quality flags.
Algorithm (fixed, no random component):
  1. quality flags F1-F5 and EXTREME/BOUNDARY status per path; robust-z (median/MAD) winsorised at +-3.5 over the flagged-free paths;
  2. reference patient;
  3. greedy set cover of nine edge categories at the 5 % tail (widened to 7.5 %/10 % only when a category has no verified candidate);
  4. two compound-hard patients (>= 3 of 6 challenge dimensions in the hardest 20 %), chosen to cover the most still-uncovered hard-combination pairs;
  5. greedy facility-location fill (minimal mean nearest-selected distance) up to the requested size.
"""
from __future__ import annotations

import collections
import itertools

import numpy as np

# the seven descriptors (equal weight in the distance); 'hard' = direction in which the case is anatomically more challenging
DESC = [("r_med_mm", "vessel radius (window median)", "low"), ("S_contact_mm", "access-path length", None), ("tort_w", "tortuosity (window)", "high"),
        ("e30_p95_mm", "bend severity (p95 of 30-mm sagitta)", "high"), ("c_z_mm", "landing-zone clearance", "low"), ("flare", "lumen flare before bifurcation", "high"),
        ("ang_deg", "daughter-vessel angle", "high")]
KEYS = [d[0] for d in DESC]
HARD = [d for d in DESC if d[2] is not None]          # six challenge dimensions (path length is an edge axis in both directions)
WIN = 3.5                                              # robust-z winsorisation / outlier bound
S_BOUNDARY = 158.0                                     # feasibility gate 150 mm +- 5 %

EDGE = [("narrow vessel", "r_med_mm", "low"), ("wide vessel", "r_med_mm", "high"), ("short path", "S_contact_mm", "low"), ("long path", "S_contact_mm", "high"),
        ("high tortuosity", "tort_w", "high"), ("large bend severity", "e30_p95_mm", "high"), ("large bifurcation angle", "ang_deg", "high"),
        ("strong flare", "flare", "high"), ("reduced landing clearance", "c_z_mm", "low")]


def robust(v):
    v = np.asarray(v, float)
    med = float(np.median(v))
    mad = float(1.4826 * np.median(np.abs(v - med)))
    return med, mad


def add_flags(rows):
    """Adds id, flags, extremes, boundary, status (FLAGGED / EXTREME / BOUNDARY / OK), z and zraw to every row (in place); returns (rows, robust statistics)."""
    for r in rows:
        r["id"] = f"{r['pid']}_{r['side']}"
    comp = [r for r in rows if r["ang_deg"] is not None]
    st = {k: robust([r[k] for r in comp]) for k in KEYS + ["jag"]}
    by = {}
    for r in rows:
        by.setdefault(r["pid"], {})[r["side"]] = r
    mismatch = set()
    for pid, v in by.items():
        if len(v) == 2:
            a, b = np.array(v["left"]["cluster_xyz"]), np.array(v["right"]["cluster_xyz"])
            if np.linalg.norm(a - b) > 10.0:
                mismatch.add(pid)
    for r in rows:
        f, x = [], []
        if r["ang_deg"] is None:
            f.append("F1_no_angle")
        if r["pid"] in mismatch:
            f.append("F2_cluster_mismatch_between_sides")
        if (r["jag"] - st["jag"][0]) / st["jag"][1] > WIN:
            f.append("F3_skeleton_jagged")
        if r["flare"] < 0.6 or r["flare"] > 1.6:
            f.append("F4_flare_implausible")
        if r["r_p05_mm"] < 2.25:
            f.append("F5_radius_at_resolution_limit")
        if r["ang_deg"] is not None:
            for k in KEYS:
                z = (r[k] - st[k][0]) / st[k][1]
                if abs(z) > WIN:
                    x.append(f"{k}:{z:+.1f}")
        b = r["S_contact_mm"] < S_BOUNDARY
        r["flags"], r["extremes"], r["boundary"] = f, x, b
        r["status"] = "FLAGGED" if f else ("EXTREME" if x else ("BOUNDARY" if b else "OK"))
        if r["ang_deg"] is not None:
            r["z"] = [float(np.clip((r[k] - st[k][0]) / st[k][1], -WIN, WIN)) for k in KEYS]
            r["zraw"] = [float((r[k] - st[k][0]) / st[k][1]) for k in KEYS]
        else:
            r["z"] = r["zraw"] = None
    return rows, st


def eligible_patients(rows):
    return sorted({r["pid"] for r in rows if r["status"] in ("OK", "BOUNDARY")})


class Cover:
    """Coverage metrics of a set of selected patients over the verified (OK + BOUNDARY) access paths."""

    def __init__(self, rows, statuses=("OK", "BOUNDARY")):
        self.rows = rows
        self.T = [r for r in rows if r["status"] in statuses and r["z"] is not None]
        self.Z = np.array([r["z"] for r in self.T])
        self.pid = np.array([r["pid"] for r in self.T])
        X = np.array([[r[k] for k in KEYS] for r in self.T])
        self.X = X
        self.p5 = np.percentile(X, 5, axis=0)
        self.p95 = np.percentile(X, 95, axis=0)
        self.p10 = np.percentile(X, 10, axis=0)
        self.p90 = np.percentile(X, 90, axis=0)
        self.tail_names = [f"{k}:{d}" for k in KEYS for d in ("low", "high")]
        self.tail_mask = np.array([(X[:, j] <= self.p10[j]) if d == "low" else (X[:, j] >= self.p90[j]) for j, k in enumerate(KEYS) for d in ("low", "high")]).T
        hard = []
        self.hard_names = []
        for k, _n, direction in HARD:
            j = KEYS.index(k)
            hard.append((X[:, j] >= np.percentile(X[:, j], 75)) if direction == "high" else (X[:, j] <= np.percentile(X[:, j], 25)))
            self.hard_names.append(f"{k}:{direction}")
        self.hard = np.array(hard).T
        self.pairs = list(itertools.combinations(range(len(HARD)), 2))
        self.pair_pop = np.array([(self.hard[:, a] & self.hard[:, b]).any() for a, b in self.pairs])

    def idx(self, pids):
        return np.isin(self.pid, list(pids))

    def nn(self, pids):
        sel = self.Z[self.idx(pids)]
        if len(sel) == 0:
            return np.full(len(self.Z), np.inf)
        d = np.linalg.norm(self.Z[:, None, :] - sel[None, :, :], axis=2)
        return d.min(1)

    def metrics(self, pids):
        m = self.idx(pids)
        nn = self.nn(pids)
        out = {"mean_nn": float(nn.mean()), "p90_nn": float(np.percentile(nn, 90)), "max_nn": float(nn.max())}
        out["tails_covered"] = int((self.tail_mask[m].any(0)).sum()) if m.any() else 0
        span = []
        for j in range(len(KEYS)):
            lo, hi = self.p5[j], self.p95[j]
            if m.any():
                a, b = self.X[m, j].min(), self.X[m, j].max()
                span.append(max(0.0, min(hi, b) - max(lo, a)) / (hi - lo))
            else:
                span.append(0.0)
        out["range_cov_mean"] = float(np.mean(span))
        out["range_cov"] = [float(s) for s in span]
        cov_pairs = 0
        for pi, (a, b) in enumerate(self.pairs):
            if self.pair_pop[pi] and m.any() and (self.hard[m, a] & self.hard[m, b]).any():
                cov_pairs += 1
        out["pairs_covered"] = cov_pairs
        out["pairs_populated"] = int(self.pair_pop.sum())
        return out


def select_development_cohort(rows: list[dict], size: int, reference: str) -> dict:
    """Deterministic cohort of ``size`` patients (``reference`` = the protocol's benign reference patient, first). Returns the cohort and the intermediate sets (forced, edges, compound, tails, metrics)."""
    rows, _st = add_flags(rows)
    cov = Cover(rows)
    T, X = cov.T, cov.X
    elig = eligible_patients(rows)
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r["pid"]].append(r)
    usable = lambda p: [r for r in byp[p] if r["status"] in ("OK", "BOUNDARY")]       # noqa: E731
    pct = np.zeros_like(X)
    for j in range(X.shape[1]):
        pct[:, j] = (np.argsort(np.argsort(X[:, j])) + 0.5) / len(X)
    Tid = {r["id"]: i for i, r in enumerate(T)}
    HARD_IDX = [(KEYS.index(k), d) for k, _n, d in HARD]
    Hm = np.array([[pct[i, j] if d == "high" else 1.0 - pct[i, j] for j, d in HARD_IDX] for i in range(len(T))])
    c80 = (Hm >= 0.8).sum(1)

    def ext(name, i):
        k, d = [(e[1], e[2]) for e in EDGE if e[0] == name][0]
        j = KEYS.index(k)
        return pct[i, j] if d == "high" else 1.0 - pct[i, j]

    def edge_cover(forced):
        tails, cand = {}, {}
        for name, _k, _d in EDGE:
            for tail in (0.05, 0.075, 0.10):
                c = {}
                for i, r in enumerate(T):
                    if ext(name, i) >= 1 - tail:
                        c[r["pid"]] = max(c.get(r["pid"], 0), ext(name, i))
                if c:
                    cand[name], tails[name] = c, tail
                    break
        covered = {n for n in cand if any(p in cand[n] for p in forced)}
        chosen = []
        while len(covered) < len(cand):
            best = max((p for p in elig if p not in chosen and p not in forced),
                       key=lambda p: 1000 * len([n for n in cand if n not in covered and p in cand[n]]) + sum(cand[n][p] for n in cand if n not in covered and p in cand[n]))
            chosen.append(best)
            covered |= {n for n in cand if best in cand[n]}
        return chosen, cand, tails

    def compound(n_pick, exclude, base):
        cands = sorted({T[i]["pid"] for i in range(len(T)) if T[i]["pid"] not in exclude and c80[i] >= 3})
        picks = []
        for _ in range(n_pick):
            def score(p):
                m = cov.metrics(base + picks + [p])
                hard = np.mean([Hm[Tid[r["id"]]].mean() for r in usable(p)])
                return (m["pairs_covered"], hard)
            picks.append(max((p for p in cands if p not in picks), key=score))
        return picks

    def fl_fill(forced, n):
        sel = list(forced)
        while len(sel) < n:
            sel.append(min((p for p in elig if p not in sel), key=lambda p: cov.metrics(sel + [p])["mean_nn"]))
        return sel

    edges, cand, tails = edge_cover([reference])
    comp = compound(2, {reference} | set(edges), [reference] + edges)
    forced = [reference] + edges + comp
    cohort = fl_fill(forced, size)
    return {"cohort": cohort, "forced": forced, "edges": edges, "compound": comp, "tails": tails, "metrics": {k: v for k, v in cov.metrics(cohort).items() if k != "range_cov"},
            "n_verified_paths": len(T), "n_eligible_patients": len(elig)}
