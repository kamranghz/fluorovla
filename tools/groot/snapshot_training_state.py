#!/usr/bin/env python3
"""Offline trainable-only snapshot from a SAVED checkpoint (no live GPU training state, streaming, low host RAM) + exact reload validation.
usage: _snapshot_offline.py <checkpoint_dir> <snapshot_out.safetensors>
1) per tensor: checkpoint vs original base -> keep only changed tensors  2) save snapshot  3) load clean base on GPU, apply snapshot,
4) every tensor of the reloaded model must be byte-identical to the checkpoint tensor."""
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, "/repo/Isaac-GR00T"); sys.path.insert(0, "/proj/src")
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from fluorovla.policy import finetune_launcher as L

ck, out = Path(sys.argv[1]), Path(sys.argv[2])
t0 = time.time()
def sha(t): return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
base_dir = Path(L.prepare_local_base())
bidx = json.load(open(base_dir / "model.safetensors.index.json"))["weight_map"]
cidx = json.load(open(ck / "model.safetensors.index.json"))["weight_map"]
bh = {}
def bget(n):
    f = bidx[n]
    if f not in bh: bh[f] = safe_open(str(base_dir / f), framework="pt", device="cpu")
    return bh[f].get_tensor(n)
snap, same, missing = {}, 0, []
for f in sorted(set(cidx.values())):
    with safe_open(str(ck / f), framework="pt", device="cpu") as h:
        for n in h.keys():
            t = h.get_tensor(n)
            if n not in bidx: missing.append(n); snap[n] = t.clone(); continue
            if torch.equal(bget(n).to(t.dtype), t): same += 1
            else: snap[n] = t.clone()
save_file(snap, str(out))
res = {"checkpoint": str(ck), "ckpt_tensors": len(cidx), "changed_in_snapshot": len(snap), "unchanged_equal_base": same, "missing_in_base": missing[:5],
       "snapshot_GB": round(out.stat().st_size / 2**30, 2), "t_build_s": round(time.time() - t0)}
del snap
print(res, flush=True)
Path("/tmp/diag_reload/cfg").mkdir(parents=True, exist_ok=True)
cfg, ft = L.build_config("/out/dataset", "/tmp/diag_reload", "reload", 1, 8, 1e-4, 1)
cfg.validate()
from gr00t.model import MODEL_REGISTRY
pipe = MODEL_REGISTRY.get(type(cfg.model))(cfg, Path("/tmp/diag_reload/cfg")); pipe.setup()
model = pipe.return_model().cuda()
sn = load_file(str(out))
miss, unexp = model.load_state_dict(sn, strict=False)
res["load_unexpected_keys"] = list(unexp)
sd = model.state_dict(); mism = []; checked = 0
for f in sorted(set(cidx.values())):
    with safe_open(str(ck / f), framework="pt", device="cpu") as h:
        for n in h.keys():
            checked += 1
            if n not in sd or sha(sd[n].detach().cpu()) != sha(h.get_tensor(n)): mism.append(n)
res.update({"tensors_checked": checked, "mismatches": len(mism), "mismatch_examples": mism[:5], "EVERY_TENSOR_EXACT": len(mism) == 0 and not unexp, "total_s": round(time.time() - t0)})
json.dump(res, open(str(out) + ".validation.json", "w"), indent=1); print(json.dumps(res, indent=1))
