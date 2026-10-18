#!/usr/bin/env python3
"""Export loss history from a checkpoint's trainer_state.json. usage: _export_loss_history.py <checkpoint_dir> <out.json>"""
import json, sys
t = json.load(open(sys.argv[1] + "/trainer_state.json"))
h = [x for x in t["log_history"] if "loss" in x]
json.dump({"global_step": t["global_step"], "n_logged": len(h),
           "history": [{"step": x.get("step"), "loss": x.get("loss"), "grad_norm": x.get("grad_norm"), "learning_rate": x.get("learning_rate")} for x in h]},
          open(sys.argv[2], "w"))
print("LOSS_OK", t["global_step"], len(h))
