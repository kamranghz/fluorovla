"""Physics-mapping summary: episode, side (all three start conditions) and patient (>= 1 admitted side) admission from attempt-addressable run directories.

Run layout: <runs_dir>/<patient>_<side>_F<k>/attempt_NNN/{STARTED.json, DONE.json, run/episode_physics_ep.json}. An attempt without DONE.json is an
infrastructure interruption and never a physics result; the accepted attempt of an episode is the LAST attempt that has a result.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path


def accepted_attempt(episode_dir: Path):
    """(attempt_dir | None, list of attempt states)."""
    acc, states = None, []
    for a in sorted(episode_dir.glob("attempt_*")):
        done = json.load(open(a / "DONE.json")) if (a / "DONE.json").is_file() else None
        states.append(done["state"] if done else "INFRASTRUCTURE_INTERRUPTED")
        if done and done.get("has_result"):
            acc = a
    return acc, states


def summarize(runs_dir: str | Path, episode_names: list[str], write_accepted: bool = True) -> dict:
    root = Path(runs_dir)
    eps, attempts = [], []
    for nm in episode_names:
        d = root / nm
        acc, states = accepted_attempt(d) if d.is_dir() else (None, [])
        attempts += [{"episode": nm, "attempt": f"attempt_{i + 1:03d}", "state": s} for i, s in enumerate(states)]
        if acc is None:
            eps.append({"name": nm, "status": "NO_ACCEPTED_ATTEMPT"})
            continue
        if write_accepted:
            json.dump({"episode": nm, "accepted": acc.name, "reason": "last completed attempt", "all_attempt_states": states}, open(d / "ACCEPTED_ATTEMPT.json", "w"), indent=1)
        e = json.load(open(acc / "run" / "episode_physics_ep.json"))
        s = e["summary"]
        pid, side, k = nm.split("_")[0], nm.split("_")[1], int(nm.split("_F")[1])
        acts = collections.Counter(r["action"] for r in e["rows"])
        eps.append({"name": nm, "pid": pid, "side": side, "k": k, "status": s["status"], "decisions": s["decisions"], "T_MAX": s["timeout_decisions"],
                    "max_axis_violation_mm": s["max_axis_violation_mm"], "n_CRUISE": acts["CRUISE"], "n_SLOW": acts["SLOW"], "n_HOLD": acts["HOLD"],
                    "accepted_attempt": acc.name})
    sides = collections.defaultdict(list)
    for e in eps:
        if "pid" in e:
            sides[(e["pid"], e["side"])].append(e)
    side_rows = {f"{p}_{s}": {"admitted": len(L) == 3 and all(x["status"] == "SUCCESS" for x in L), "statuses": [x["status"] for x in sorted(L, key=lambda x: x["k"])]}
                 for (p, s), L in sides.items()}
    pat = collections.defaultdict(list)
    for k, v in side_rows.items():
        pat[k.split("_")[0]].append(v["admitted"])
    patient_rows = {p: {"admitted": any(v), "sides_admitted": sum(v), "sides_tested": len(v)} for p, v in pat.items()}
    summ = {"episodes": len(eps), "status_counts": dict(collections.Counter(e["status"] for e in eps)), "sides_tested": len(side_rows),
            "sides_admitted": sum(v["admitted"] for v in side_rows.values()), "patients_tested": len(patient_rows),
            "patients_admitted": sum(v["admitted"] for v in patient_rows.values()), "attempt_states": dict(collections.Counter(a["state"] for a in attempts))}
    return {"summary": summ, "patients": patient_rows, "sides": side_rows, "episodes": eps, "attempts": attempts}
