"""Run Privileged-Teacher physics episodes (physics mapping) from episode-configuration directories; attempt-addressable, never overwrites.

usage: run_physics_mapping.py <runs_dir> <episode_config_dir> [...]
  Each episode runs in ONE process sequence (one CUDA context). For every episode a NEW directory <runs_dir>/<name>/attempt_NNN/ is created with
  STARTED.json (before), run/ (results, probe.json), stdout in the process log, and DONE.json (after a finished run, python errors included).
  An attempt without DONE.json is an infrastructure interruption, never a physics result.
env: FLUOROVLA_DATA (data root), MEDPHYSAI_POST_HOLD-equivalent: --post-hold N
"""
import _bootstrap  # noqa: F401  isort:skip
import argparse
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

from fluorovla.common.provenance import code_hash
from fluorovla.physics.episode_runner import run_physics_episode
from fluorovla.physics.initial_state_probe import probe_initial_state
from fluorovla.physics.lumen_sdf import LumenSDF
from fluorovla.physics.patient_config import PatientConfig

ap = argparse.ArgumentParser()
ap.add_argument("runs_dir")
ap.add_argument("episode_dirs", nargs="+")
ap.add_argument("--post-hold", type=int, default=0)
ap.add_argument("--no-probe", action="store_true")
a = ap.parse_args()
runs = Path(a.runs_dir)
runs.mkdir(parents=True, exist_ok=True)
for ed in a.episode_dirs:
    d = Path(ed)
    name = d.name
    base = runs / name
    base.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[1]) for p in base.glob("attempt_*")] or [0])
    att = base / f"attempt_{n:03d}"
    att.mkdir()
    cfgfile = d / "config.json"
    (att / "STARTED.json").write_text(json.dumps({"name": name, "attempt": n, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
                                                  "config_sha256": hashlib.sha256(cfgfile.read_bytes()).hexdigest(), "code_sha256": code_hash()["combined"]}))
    t0 = time.time()
    state = "COMPLETED"
    try:
        cfg = PatientConfig.from_json(cfgfile)
        if not a.no_probe:
            (att / "run").mkdir(exist_ok=True)
            json.dump(probe_initial_state(cfg, LumenSDF(cfg)), open(att / "run" / "probe.json", "w"), indent=1)
        run_physics_episode(cfg, att / "run", "ep", post_success_hold=a.post_hold)
    except BaseException as ex:  # noqa: BLE001  (a python error is a recorded result, not an infrastructure event)
        state = "PYTHON_ERROR"
        (att / "error.txt").write_text(traceback.format_exc())
        print(f"{name} attempt {n} PYTHON_ERROR {ex!r}", flush=True)
    has = (att / "run" / "episode_physics_ep.json").is_file()
    if state == "COMPLETED" and not has:
        state = "PYTHON_ERROR"
    (att / "DONE.json").write_text(json.dumps({"state": state, "seconds": time.time() - t0, "has_result": has}))
    print(f"{name} attempt {n} {state} {time.time() - t0:.0f}s", flush=True)
