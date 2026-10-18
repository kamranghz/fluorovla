"""Access-path geometry and anatomy descriptors from the patient twin (centerline graph).

Construction (reference: the accepted s0011 episode, whose S_contact is reproduced to 0.01 mm):
  entry = graph endpoint of an iliac artery; the portal arm runs from the entry to the first branch-point cluster (connected degree>=3 nodes);
  extended path = portal-arm nodes + cluster centroid + 30 mm of the straighter daughter; S_contact = arclength of the last portal-arm node (cluster entry)
  on the smoothed extended path; landing zone (tip-extremity coordinates) = [S_contact - 4.72, S_contact - 2.32].
Skeleton cleaning (patient independent): leaf branches shorter than SPUR_MM hanging on a junction are pruned before junction finding; two or more arms
shorter than SPUR_MM that leave a cluster and end on the same node (a tiny skeleton cycle) are not counted (LOOP_COLLAPSE).
Descriptors (seven): window-median radius, access-path length, window tortuosity, p95 of the 30-mm chord sagitta, landing clearance, flare, daughter angle;
window = the last 173.5 mm before the landing tip."""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np

import fluorovla.task_contract as tc

SPUR_MM = 12.0
R_C, W_LEN, ZONE_OFF = 3.0, 173.5, 4.72
C_CLEAR = 2.3132999999999995           # s0011 measured clearance (landing_zone.json): zone far edge = S_contact - C_CLEAR
EPS = tc.EPS_MM
STEP = tc.RESAMPLE_STEP_MM


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def prune_spurs(adj, P, spur_mm=SPUR_MM):
    """Iteratively remove leaf branches (endpoint .. junction) shorter than spur_mm. Returns pruned adjacency (copy) and #removed nodes."""
    adj = {k: set(v) for k, v in adj.items()}
    removed = 0
    changed = True
    while changed:
        changed = False
        leaves = [n for n, v in adj.items() if len(v) == 1]
        for leaf in leaves:
            if leaf not in adj or len(adj[leaf]) != 1:
                continue
            path, prev, cur, length = [leaf], None, leaf, 0.0
            while True:
                nxt = [n for n in adj[cur] if n != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
                length += float(np.linalg.norm(P[cur] - P[prev]))
                if len(adj[cur]) >= 3:
                    break
                path.append(cur)
                if length > spur_mm:
                    break
            if len(adj[cur]) >= 3 and length <= spur_mm:
                for n in path:
                    for m in adj[n]:
                        adj[m].discard(n)
                    del adj[n]
                    removed += 1
                changed = True
    return adj, removed


LOOP_COLLAPSE = True       # v2 (Stage 1 attempt 2): drop skeleton-cycle stubs (see access_from_entry)


def arm_walk(adj, start, cluster):
    nodes, prev, cur = [start], next(iter(set(adj[start]) & cluster)) if set(adj[start]) & cluster else None, start
    while True:
        nxt = [n for n in adj[cur] if n != prev and n not in cluster]
        if len(nxt) != 1 or len(adj[nxt[0]]) >= 3:
            break
        prev, cur = cur, nxt[0]
        nodes.append(cur)
    return nodes


def sagitta_profile(p, s):
    w_n = int(tc.L_SEG_MM / STEP)
    e = np.zeros(len(s))
    for i in range(len(s)):
        j = max(0, i - w_n)
        if i - j < 4:
            continue
        u = p[i] - p[j]
        n = np.linalg.norm(u)
        if n < 1e-9:
            continue
        u = u / n
        seg = p[j:i + 1] - p[j]
        e[i] = float(np.linalg.norm(seg - np.outer(seg @ u, u), axis=1).max())
    return e


def access_from_entry(P, R, adj, entry):
    """Geometry for one entry endpoint. Returns dict (with arrays under keys starting with '_') or {'valid': False, 'reason': ...}."""
    deg = {k: len(v) for k, v in adj.items()}
    j0 = tc.first_junction(adj, entry)
    if j0 is None:
        return {"valid": False, "reason": "NO_JUNCTION"}
    cluster, q = {j0}, deque([j0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if deg[v] >= 3 and v not in cluster:
                cluster.add(v)
                q.append(v)
    cl = sorted(cluster)
    arms = []
    for c in cl:
        for nb in adj[c]:
            if nb not in cluster:
                nodes = arm_walk(adj, nb, cluster)
                arms.append({"nodes": nodes, "has_entry": entry in nodes})
    if LOOP_COLLAPSE:
        # Patient-independent topology cleaning: two or more arms shorter than SPUR_MM that leave the cluster and END ON THE SAME NODE close a tiny skeleton cycle
        # (not an anatomical branch); they are removed from the arm list. Long arms and endpoint arms are never removed.
        by_end = {}
        for a in arms:
            L = float(np.linalg.norm(np.diff(P[[next(iter(set(adj[a["nodes"][0]]) & cluster))] + a["nodes"]], axis=0), axis=1).sum())
            a["len"] = L
            by_end.setdefault(a["nodes"][-1], []).append(a)
        drop = [a for grp in by_end.values() if len(grp) >= 2 and all(x["len"] < SPUR_MM and len(adj[x["nodes"][-1]]) == 2 for x in grp) for a in grp]
        arms = [a for a in arms if not any(a is d for d in drop)]
    if len(arms) != 3 or sum(a["has_entry"] for a in arms) != 1:
        return {"valid": False, "reason": f"CLUSTER_ARMS_{len(arms)}"}
    centroid = P[cl].mean(axis=0)
    nodes_to_cluster = tc.bfs_path(adj, entry, j0)
    portal_side = [n for n in nodes_to_cluster if n not in cluster]
    if len(portal_side) < 3:
        return {"valid": False, "reason": "PORTAL_ARM_TOO_SHORT"}

    def dirs(a, lo=12.0, hi=22.0):
        far = [P[n] for n in a["nodes"] if lo <= np.linalg.norm(P[n] - centroid) <= hi]
        return unit(np.mean(far, 0) - centroid) if far else None

    daughters = [a for a in arms if not a["has_entry"]]
    din = -unit(P[portal_side[-1]] - P[portal_side[max(0, len(portal_side) - 6)]])
    for a in daughters:
        a["dir"] = dirs(a, 0.0, 12.0)
    cont = min(daughters, key=lambda a: float(np.degrees(np.arccos(np.clip(np.dot(-din, a["dir"]), -1, 1)))) if a["dir"] is not None else 1e9)
    cn = cont["nodes"]
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[[cl[0]] + cn], axis=0), axis=1))]
    n_ext = int(min(np.searchsorted(cum, 30.0) + 1, len(cn)))
    ext_pts = np.vstack([P[portal_side], centroid[None, :], P[cn[:n_ext]]])
    ext_rad = np.r_[R[portal_side], R[cl].mean(), R[cn[:n_ext]]]
    proj = tc.PathProjector(ext_pts, ext_rad)
    s_carina, _, _ = proj(centroid)
    S, _, _ = proj(P[portal_side[-1]])
    s, p = proj.s, proj.p
    k = max(1, int(3.0 / STEP))
    ker = np.exp(-0.5 * (np.arange(-4 * k, 4 * k + 1) / k) ** 2)
    ker /= ker.sum()
    rs = np.convolve(np.pad(proj.r, (4 * k, 4 * k), mode="edge"), ker, mode="valid")[: len(s)]
    e30 = sagitta_profile(p, s)
    z_tip = S - ZONE_OFF - R_C
    w0 = max(0.0, z_tip - W_LEN)
    m = (s >= w0) & (s <= z_tip)
    if m.sum() < 4:                                    # endpoint far too close to the cluster (not an access path); recorded, never fatal for the patient
        return {"valid": False, "reason": "WINDOW_EMPTY_PATH_TOO_SHORT", "S_contact_mm": float(S)}
    chord = float(np.linalg.norm(p[np.searchsorted(s, z_tip) - 1] - p[np.searchsorted(s, w0)]))
    c_z = float(np.mean(rs[(s >= z_tip - 3.0) & (s <= z_tip)]) - R_C)
    i20 = int(np.searchsorted(s, S - 20.0))
    iS = int(min(np.searchsorted(s, S), len(s) - 1))
    # daughter angle: the two branches pointing away from the access arm (design convention: mean position 12-22 mm from the centroid)
    dd = [dirs(a) for a in daughters]
    ang = float(np.degrees(math.acos(np.clip(dd[0] @ dd[1], -1, 1)))) if all(d is not None for d in dd) else None
    return {"valid": True, "entry_node": int(entry), "cluster_nodes": cl, "centroid_ct_mm": centroid.tolist(),
            "S_contact_mm": float(S), "S_carina_mm": float(s_carina), "window_mm": float(z_tip - w0),
            "r_med_mm": float(np.median(rs[m])), "r_p05_mm": float(np.percentile(rs[m], 5)), "c_z_mm": c_z,
            "tort_w": float((z_tip - w0) / max(chord, 1e-6)), "e30_p95_mm": float(np.percentile(e30[m], 95)), "e30_max_mm": float(e30[m].max()),
            "flare": float(rs[iS] / rs[i20]), "ang_deg": ang, "jag": float(proj.raw_arclength_mm / max(S, 1e-6)),
            "entry_ct_mm": P[entry].tolist(), "n_portal_arm_nodes": len(portal_side),
            "landing_zone_extremity_mm": [float(S - ZONE_OFF), float(S - ZONE_OFF + 2 * EPS)],
            "F_max_mm": float(S - 81.2),
            "_ext_pts": ext_pts, "_ext_rad": ext_rad, "_s_carina": float(s_carina), "_centroid": centroid}


def extract_patient_geometry(twin_dir: str | Path, pid: str) -> dict:
    """All entry endpoints of one patient twin (``twin_dir`` = directory containing the centerline_* files); arrays are kept under keys starting with an underscore."""
    d = Path(twin_dir)
    P = np.load(d / "centerline_points_mm.npy").astype(np.float64)
    E = np.load(d / "centerline_edges.npy").astype(np.int64)
    R = np.load(d / "centerline_radii_mm.npy").astype(np.float64)
    adj0 = tc.build_adjacency(E, len(P))
    adj, n_pruned = prune_spurs(adj0, P)
    ends = [n for n, v in adj.items() if len(v) == 1]
    # candidate entries = endpoints whose first junction is a real 3-armed cluster; lowest-z endpoints are the iliac ends
    out = {"pid": pid, "n_nodes": int(len(P)), "n_pruned": n_pruned, "n_endpoints": len(ends), "entries": []}
    for e in ends:
        try:
            g = access_from_entry(P, R, adj, e)
        except Exception as ex:  # noqa: BLE001  (a failing entry is recorded as invalid, it must not discard the patient)
            g = {"valid": False, "reason": "ENTRY_EXCEPTION " + repr(ex)[:80]}
        g["entry_node"] = int(e)
        g["entry_ct_mm"] = P[e].tolist()
        out["entries"].append(g)
    return out


def split_arrays(geometries: list[dict]) -> tuple[list[dict], dict]:
    """Separate the extended-path arrays (``_ext_pts``/``_ext_rad``) from the JSON-serializable geometry records. Returns (records, arrays keyed ``pid__entry__name``)."""
    arrays: dict = {}
    for g in geometries:
        for e in g.get("entries", []):
            for k in [k for k in e if k.startswith("_")]:
                if k in ("_ext_pts", "_ext_rad"):
                    arrays[f"{g['pid']}__{e['entry_node']}__{k[1:]}"] = e[k]
                del e[k]
    return geometries, arrays
