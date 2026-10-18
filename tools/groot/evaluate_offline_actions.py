#!/usr/bin/env python3
"""Offline evaluation of the tiny-overfit GR00T N1.7 checkpoint on its OWN training episode (no closed loop, no generalisation claim).
Runs in the gr00t:ecosystem container with a GPU.  usage: _groot_overfit_eval.py <checkpoint_dir> <dataset_dir> <out_json> [n_seeds]
Observations are exactly what the policy sees at deployment: decoded fluoro video frame t, state_t = [inserted_length_m, previous_insert_velocity_mps],
language. The model's 16-step normalised chunk is decoded with the processor saved in the checkpoint."""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/repo/Isaac-GR00T")
sys.path.insert(0, "/proj/src")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fluorovla.policy.finetune_launcher import load_modality_config, MODALITY_FILE, prepare_local_base  # noqa: E402

prepare_local_base()                                   # recreates the offline symlink the checkpoint's config.json points to
load_modality_config(MODALITY_FILE)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402
from fluorovla.policy.catheter_modality import catheter_modality_config as MC  # noqa: E402

CKPT, DS, OUTJ = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
N_SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
H = 16
loader = LeRobotEpisodeLoader(DS, MC)
df = loader[0]
N = len(df)
ST = np.stack(df["state.catheter"].to_numpy()).astype(np.float32)
AC = np.stack(df["action.insert_velocity"].to_numpy()).astype(np.float32)[:, 0]
VID = [np.asarray(f) for f in df["video.fluoro"]]
LANG = df["language.annotation.human.task_description"].iloc[0]
policy = Gr00tPolicy(EmbodimentTag.NEW_EMBODIMENT, CKPT, device="cuda:0", strict=True)
CANON = np.array([0.0, 0.006, 0.012], dtype=np.float64)
NAMES = ["HOLD", "SLOW", "CRUISE"]
n_valid = N - H + 1
preds = np.zeros((N_SEEDS, n_valid, H), dtype=np.float64)
for sd in range(N_SEEDS):
    torch.manual_seed(1000 + sd)
    for t in range(n_valid):
        obs = {"video": {"fluoro": VID[t][None, None].astype(np.uint8)},
               "state": {"catheter": ST[t][None, None].astype(np.float32)},
               "language": {"annotation.human.task_description": [[LANG]]}}
        act, _ = policy.get_action(obs)
        a = np.asarray(act["insert_velocity"]).reshape(-1)
        assert a.shape[0] == H, a.shape
        preds[sd, t] = a
labels = np.stack([AC[t:t + H] for t in range(n_valid)])                # (n_valid, 16) ground-truth chunks
cls_of = lambda v: np.abs(np.asarray(v)[..., None] - CANON).argmin(-1)   # noqa: E731
lab0 = cls_of(labels[:, 0])
res = {"checkpoint": CKPT, "n_valid_base_samples": int(n_valid), "n_seeds": N_SEEDS, "chunk_horizon": H}
mae_first = np.abs(preds[:, :, 0] - labels[None, :, 0])
mae_chunk = np.abs(preds - labels[None])
res["action_mae_first_step_mps"] = {"all": float(mae_first.mean()), **{NAMES[c]: float(mae_first[:, lab0 == c].mean()) for c in range(3)}}
res["action_mae_chunk16_mps"] = {"all": float(mae_chunk.mean()), **{NAMES[c]: float(mae_chunk[:, lab0 == c].mean()) for c in range(3)}}
pc = cls_of(preds[:, :, 0])
res["canonical_accuracy_first_step"] = {"all": float((pc == lab0[None]).mean()), **{NAMES[c]: float((pc[:, lab0 == c] == c).mean()) for c in range(3)}}
res["canonical_accuracy_chunk16"] = float((cls_of(preds) == cls_of(labels)[None]).mean())
conf = np.zeros((3, 3), dtype=int)
for sd in range(N_SEEDS):
    for t in range(n_valid):
        conf[lab0[t], pc[sd, t]] += 1
res["confusion_first_step_rows_true_HOLD_SLOW_CRUISE_cols_pred"] = conf.tolist()
res["label_counts_first_step"] = {NAMES[c]: int((lab0 == c).sum()) for c in range(3)}
# baselines for context
res["baseline_always_CRUISE"] = {"mae_first_mps": float(np.abs(0.012 - labels[:, 0]).mean()), "canonical_accuracy": float((lab0 == 2).mean())}
# transition region: labels CRUISE -> SLOW -> HOLD
first_slow = int(np.argmax(lab0 == 1))
first_hold = int(np.argmax(lab0 == 0))
lo, hi = max(0, first_slow - 8), min(n_valid, first_hold + 6)
res["transition_region"] = {"first_SLOW_step": first_slow, "first_HOLD_step": first_hold,
                            "rows": [{"t": int(t), "label": float(labels[t, 0]), "pred_mean": float(preds[:, t, 0].mean()),
                                      "pred_std": float(preds[:, t, 0].std()), "pred_class_seed0": NAMES[int(pc[0, t])],
                                      "label_class": NAMES[int(lab0[t])]} for t in range(lo, hi)]}
tr = np.arange(lo, hi)
res["transition_region"]["canonical_accuracy"] = float((pc[:, tr] == lab0[None, tr]).mean())
res["transition_region"]["mae_first_mps"] = float(mae_first[:, tr].mean())
# full-chunk example from the last base sample before the SLOW run
tt = max(0, first_slow - 9)
res["chunk_example"] = {"t": int(tt), "label_chunk": labels[tt].tolist(), "pred_chunk_seed0": preds[0, tt].round(5).tolist()}
res["state_dropout_used_at_eval"] = False
res["preds_first_step_seed0"] = preds[0, :, 0].round(6).tolist()
res["labels_first_step"] = labels[:, 0].round(6).tolist()
json.dump(res, open(OUTJ, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("action_mae_first_step_mps", "canonical_accuracy_first_step", "canonical_accuracy_chunk16",
                                       "confusion_first_step_rows_true_HOLD_SLOW_CRUISE_cols_pred")}, indent=1))
