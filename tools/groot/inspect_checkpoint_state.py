#!/usr/bin/env python3
"""Inspect a full-state HF/GR00T checkpoint: optimizer (Adam step + moments), scheduler, trainer_state, RNG. usage: <checkpoint_dir>"""
import json
import sys
from pathlib import Path

import torch

ck = Path(sys.argv[1])
files = {p.name: p.stat().st_size for p in ck.iterdir() if p.is_file()}
opt = torch.load(ck / "optimizer.pt", map_location="cpu", weights_only=False)
st = opt["state"]
steps = sorted({int(float(v["step"])) for v in st.values()})
nz = sum(float(v["exp_avg"].abs().max()) > 0 for v in st.values())
sch = torch.load(ck / "scheduler.pt", map_location="cpu", weights_only=False)
ts = json.load(open(ck / "trainer_state.json"))
rng = torch.load(ck / "rng_state.pth", map_location="cpu", weights_only=False)
print(json.dumps({"checkpoint": str(ck), "files_MB": {k: round(v / 2**20, 1) for k, v in files.items()},
                  "adam_param_tensors_with_state": len(st), "adam_step_values": steps, "adam_exp_avg_nonzero_tensors": nz,
                  "param_groups_lr": [g["lr"] for g in opt["param_groups"]][:3],
                  "scheduler_last_epoch": sch.get("last_epoch"), "scheduler_last_lr": sch.get("_last_lr"), "scheduler_step_count": sch.get("_step_count"),
                  "trainer_global_step": ts["global_step"], "trainer_log_history_len": len(ts["log_history"]),
                  "rng_keys": sorted(rng.keys()) if isinstance(rng, dict) else str(type(rng))}, indent=1))
