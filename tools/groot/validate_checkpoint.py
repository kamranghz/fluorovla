#!/usr/bin/env python3
"""Checkpoint completeness check (stdlib only). usage: _ckpt_valid.py <run_dir> [--delete-invalid]
Prints one line per checkpoint-N: 'N VALID|INVALID reason'. A checkpoint is VALID only if every safetensors shard in the index exists and has the exact
size its header implies, and optimizer.pt/scheduler.pt/rng_state.pth/trainer_state.json exist (optimizer.pt >= 90% of 2x trainable bytes is not assumed;
we require size > 1 GB and equal to the sibling checkpoints when several exist is NOT required) and trainer_state.json parses."""
import json, os, shutil, struct, sys
from pathlib import Path

def shard_ok(p):
    try:
        sz = p.stat().st_size
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default=0)
        return sz == 8 + n + end
    except Exception:
        return False

def check(ck):
    for f in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors.index.json", "config.json"):
        if not (ck / f).is_file():
            return False, f"missing {f}"
    if (ck / "optimizer.pt").stat().st_size < 2**30:
        return False, "optimizer.pt too small"
    try:
        ts = json.load(open(ck / "trainer_state.json"))
        idx = json.load(open(ck / "model.safetensors.index.json"))["weight_map"]
    except Exception as e:
        return False, f"unparseable json {e!r}"
    if ts.get("global_step") != int(ck.name.split("-")[1]):
        return False, "global_step mismatch"
    for s in sorted(set(idx.values())):
        if not shard_ok(ck / s):
            return False, f"bad shard {s}"
    return True, "ok"

run, delete = Path(sys.argv[1]), "--delete-invalid" in sys.argv
for ck in sorted((p for p in run.glob("checkpoint-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1])):
    ok, why = check(ck)
    print(f"{ck.name.split('-')[1]} {'VALID' if ok else 'INVALID'} {why}")
    if not ok and delete:
        shutil.rmtree(ck, ignore_errors=True)
        print(f"{ck.name.split('-')[1]} DELETED")
