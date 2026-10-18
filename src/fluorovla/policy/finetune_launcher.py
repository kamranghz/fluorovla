"""FluoroVLA GR00T N1.7 fine-tuning launch configuration for the catheter NEW_EMBODIMENT.

Mirrors gr00t/experiment/launch_finetune.py assignment-by-assignment (same FinetuneConfig defaults, same config fields), with
ONLY these FluoroVLA overrides:
    state_dropout_prob = 0.0          the policy state is 2-dim (inserted_length_m, previous_insert_velocity_mps); the GR00T
                                      default of 0.2 would remove the whole state in 20% of samples
    model.model_name   = local snapshot of nvidia/Cosmos-Reason2-2B   (offline container; same weights/tokenizer as the hub id)
    modality config    = src/fluorovla/policy/catheter_modality.py (16-step ABSOLUTE insert_velocity chunk)
The GR00T repository is not modified. Runs inside the gr00t:ecosystem container.

usage (container):  launch_catheter_finetune.py train <dataset_dir> <output_dir> <experiment_name> <max_steps> <global_batch_size> <lr> [save_steps]
"""
from __future__ import annotations

import importlib
import json
import time
import os
import sys
from pathlib import Path

sys.path.insert(0, "/repo/Isaac-GR00T")

from gr00t.configs.base_config import get_default_config  # noqa: E402
from gr00t.configs.finetune_config import FinetuneConfig  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402

MODALITY_FILE = "/proj/src/fluorovla/policy/catheter_modality.py"
BASE_MODEL = "/models/GR00T-N1.7-3B"
COSMOS_SNAPSHOT = "/hf/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_LOCAL = "/tmp/hub_ids/nvidia/Cosmos-Reason2-2B"          # symlink to COSMOS_SNAPSHOT (created by prepare_local_base)
FLUOROVLA_STATE_DROPOUT_PROB = 0.0


def prepare_local_base(dst: str = "/tmp/gr00t_n1d7_base_local") -> str:
    """Offline container only: AutoModel.from_pretrained reads model_name = 'nvidia/Cosmos-Reason2-2B' from the checkpoint's
    config.json and then calls the hub (model_info). Build a directory of symlinks to the untouched base checkpoint files with ONE
    change: config.json model_name -> the local snapshot of that same model. Weights, statistics, processor config are the originals."""
    link = Path("/tmp/hub_ids/nvidia/Cosmos-Reason2-2B")       # get_backbone_cls requires 'nvidia/Cosmos-Reason2' in the name
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(COSMOS_SNAPSHOT)
    d = Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    for f in Path(BASE_MODEL).iterdir():
        if f.name == "config.json" or (d / f.name).exists():
            continue
        (d / f.name).symlink_to(f)
    cfg = json.load(open(Path(BASE_MODEL) / "config.json"))
    cfg["model_name"] = COSMOS_LOCAL
    json.dump(cfg, open(d / "config.json", "w"), indent=2)
    return str(d)


def load_modality_config(modality_config_path: str) -> None:       # same route as launch_finetune.py
    path = Path(modality_config_path)
    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)


def build_config(dataset_path: str, output_dir: str, experiment_name: str, max_steps: int = 10000, global_batch_size: int = 64,
                 learning_rate: float = 1e-4, save_steps: int = 1000, dataloader_num_workers: int = 0):
    base_local = prepare_local_base()                                  # also creates the offline Cosmos symlink
    start = os.environ.get("FLUOROVLA_START_CKPT") or base_local   # continuation: weights-only start from a saved checkpoint
    ft = FinetuneConfig(base_model_path=start, dataset_path=dataset_path, embodiment_tag="new_embodiment",
                        modality_config_path=MODALITY_FILE, state_dropout_prob=FLUOROVLA_STATE_DROPOUT_PROB,
                        global_batch_size=global_batch_size, dataloader_num_workers=dataloader_num_workers,
                        learning_rate=learning_rate, output_dir=output_dir, experiment_name=experiment_name,
                        save_steps=save_steps, save_total_limit=int(os.environ.get("FLUOROVLA_SAVE_TOTAL_LIMIT", "1")), num_gpus=1, use_wandb=False, max_steps=max_steps,
                        save_only_model=os.environ.get("FLUOROVLA_SAVE_ONLY_MODEL", "0") == "1")      # default: FULL-state checkpoints (optimizer, scheduler, RNG, trainer_state)
    ft.embodiment_tag = EmbodimentTag.resolve(ft.embodiment_tag)
    embodiment_tag = ft.embodiment_tag.value
    load_modality_config(ft.modality_config_path)
    dataset_paths = [p for p in ft.dataset_path.split(os.pathsep) if p]
    config = get_default_config().load_dict({"data": {"download_cache": False, "datasets": [
        {"dataset_paths": dataset_paths, "mix_ratio": 1.0, "embodiment_tag": embodiment_tag}]}})
    config.load_config_path = None
    m = config.model
    m.tune_llm, m.tune_visual, m.tune_projector, m.tune_diffusion_model = ft.tune_llm, ft.tune_visual, ft.tune_projector, ft.tune_diffusion_model
    m.state_dropout_prob, m.random_rotation_angle, m.color_jitter_params, m.use_percentiles = (
        ft.state_dropout_prob, ft.random_rotation_angle, ft.color_jitter_params, ft.use_percentiles)
    m.extra_augmentation_config = None
    m.load_bf16, m.reproject_vision, m.backbone_trainable_params_fp32, m.use_relative_action = False, False, True, True
    m.model_name = COSMOS_LOCAL
    t = config.training
    t.experiment_name, t.start_from_checkpoint, t.optim = ft.experiment_name, ft.base_model_path, "adamw_torch"
    t.global_batch_size, t.dataloader_num_workers, t.learning_rate = ft.global_batch_size, ft.dataloader_num_workers, ft.learning_rate
    t.gradient_accumulation_steps, t.output_dir, t.save_steps, t.save_total_limit = (
        ft.gradient_accumulation_steps, ft.output_dir, ft.save_steps, ft.save_total_limit)
    t.num_gpus, t.use_wandb, t.max_steps, t.weight_decay, t.warmup_ratio = ft.num_gpus, ft.use_wandb, ft.max_steps, ft.weight_decay, ft.warmup_ratio
    t.wandb_project, t.save_only_model, t.resume_from_checkpoint, t.skip_weight_loading = (
        ft.wandb_project, ft.save_only_model, ft.resume_from_checkpoint, ft.skip_weight_loading)
    t.logging_steps = 1                                            # per-step loss curve (logging only)
    t.transformers_local_files_only = True                         # offline container
    d = config.data
    d.shard_size, d.episode_sampling_rate, d.num_shards_per_epoch, d.ds_weights_alpha = (
        ft.shard_size, ft.episode_sampling_rate, ft.num_shards_per_epoch, ft.ds_weights_alpha)
    return config, ft


def effective_summary(config, ft) -> dict:
    return {"base_model": config.training.start_from_checkpoint, "embodiment_tag": "new_embodiment",
            "state_dropout_prob": float(config.model.state_dropout_prob), "tune_llm": config.model.tune_llm,
            "tune_visual": config.model.tune_visual, "tune_projector": config.model.tune_projector,
            "tune_diffusion_model": config.model.tune_diffusion_model, "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay, "warmup_ratio": config.training.warmup_ratio,
            "global_batch_size": config.training.global_batch_size, "max_steps": config.training.max_steps,
            "optim": config.training.optim, "num_gpus": config.training.num_gpus, "use_percentiles": config.model.use_percentiles,
            "use_relative_action": config.model.use_relative_action, "model_action_horizon": config.model.action_horizon,
            "random_rotation_angle": config.model.random_rotation_angle, "color_jitter_params": config.model.color_jitter_params,
            "extra_augmentation_config": config.model.extra_augmentation_config, "shard_size": config.data.shard_size,
            "episode_sampling_rate": config.data.episode_sampling_rate, "model_name_tokenizer_source": config.model.model_name,
            "load_bf16": config.model.load_bf16, "seed": config.data.seed}


if __name__ == "__main__":
    if sys.argv[1] != "train":
        raise SystemExit("usage: launch_catheter_finetune.py train <dataset_dir> <output_dir> <experiment_name> <max_steps> <global_batch_size> <lr> [save_steps]")
    ds, out, name, steps, bs, lr = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
    save_steps = int(sys.argv[8]) if len(sys.argv) > 8 else steps
    cfg, ft = build_config(ds, out, name, steps, bs, lr, save_steps)
    if os.environ.get("FLUOROVLA_RESUME") == "1":
        cfg.training.resume_from_checkpoint = True
    cfg.validate()
    Path(out).mkdir(parents=True, exist_ok=True)
    json.dump(effective_summary(cfg, ft), open(Path(out) / f"{name}_effective_config.json", "w"), indent=2, default=str)
    min_save = int(os.environ.get("FLUOROVLA_MIN_SAVE_STEP", "0"))
    if min_save > 0:       # evaluation snapshots only from this step on (no save before it); training itself is untouched
        from gr00t.experiment.trainer import Gr00tTrainer  # noqa: E402
        _orig_save = Gr00tTrainer._save_checkpoint

        def _save_checkpoint(self, model, trial, *a, **k):
            if self.state.global_step < min_save:
                return None
            return _orig_save(self, model, trial, *a, **k)
        Gr00tTrainer._save_checkpoint = _save_checkpoint
    hb_path = os.environ.get("FLUOROVLA_HEARTBEAT")
    if hb_path:     # progress heartbeat for the external supervisor; read-only w.r.t. training (no model/optimizer/scheduler/data/RNG access)
        from transformers import TrainerCallback  # noqa: E402
        from gr00t.experiment.trainer import Gr00tTrainer as _T  # noqa: E402

        class _Heartbeat(TrainerCallback):
            def __init__(self, path, every=2):
                self.path, self.every, self.seq = path, every, 0

            def _write(self, phase, step):
                self.seq += 1
                tmp = self.path + ".tmp"
                with open(tmp, "w") as f:
                    f.write(json.dumps({"seq": self.seq, "global_step": int(step), "phase": phase, "pid": os.getpid(), "t_container": time.time()}))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)

            def on_train_begin(self, args, state, control, **kw):
                self._write("train_begin", state.global_step)

            def on_step_end(self, args, state, control, **kw):
                saving = bool(control.should_save)     # DefaultFlowCallback ran first: the save follows immediately
                if saving or state.global_step % self.every == 0:
                    self._write("saving" if saving else "train", state.global_step)

            def on_save(self, args, state, control, **kw):
                self._write("saved", state.global_step)

            def on_train_end(self, args, state, control, **kw):
                self._write("train_end", state.global_step)

        _orig_init = _T.__init__

        def _init(self, *a, **k):
            _orig_init(self, *a, **k)
            self.add_callback(_Heartbeat(hb_path))
        _T.__init__ = _init
    from gr00t.experiment.experiment import run  # noqa: E402
    run(cfg)
