# Refactor map (supersession map)

Every renamed, moved, consolidated or deleted scientific-looking path of the legacy `MedPhysAI` tree, with its replacement. The complete machine-readable list is `artifacts/provenance/refactor_map.json` (old_path, new_path_or_replacement, reason, scientific_status, category). Hashes of every legacy file before any change are in `PRE_REFACTOR_SHA256.txt` (outside the repository). The legacy tree is kept as an archive; only the entries marked `deleted` were removed.

Categories: DELETE 1, DUPLICATE 1, KEEP_CANONICAL 46, KEEP_EVIDENCE 154, KEEP_REPRODUCTION 3, OBSOLETE 121, REGENERABLE_CACHE 2, TEMP_DEBUG 102

## Canonical code (ported or consolidated)

| old_path | new_path_or_replacement | reason |
|---|---|---|
| `scripts/newton_shared_patient_feed_backend.py` | `src/fluorovla/physics/lumen_sdf.py` | generate_lumen_contacts consolidated; validation main() archived |
| `scripts/newton_stage4a_vessel_contact.py` | `src/fluorovla/physics/lumen_sdf.py` | library part (mesh, constants) consolidated; Stage-4A validation main() archived |
| `scripts/newton_stage4d_segment_contact.py` | `src/fluorovla/physics/lumen_sdf.py` | library part (SDF kernel/query) consolidated; Stage-4D validation main() archived |
| `scripts/supervise_train.ps1` | `tools/infra/supervise_groot_training.ps1` | ported (local WSL training supervision; training moves to cloud) |
| `scripts/_canonical_episode_physics.py` | `src/fluorovla/physics/episode_runner.py` | converted from a script to run_physics_episode(); bit-identical |
| `scripts/_ckpt_valid.py` | `tools/groot/validate_checkpoint.py` | ported |
| `scripts/_cuda_warmup.py` | `tools/groot/cuda_warmup.py` | ported |
| `scripts/_export_loss_history.py` | `tools/groot/export_loss_history.py` | ported |
| `scripts/_groot_overfit_eval.py` | `tools/groot/evaluate_offline_actions.py` | ported |
| `scripts/_inspect_ckpt_state.py` | `tools/groot/inspect_checkpoint_state.py` | ported |
| `scripts/_snapshot_offline.py` | `tools/groot/snapshot_training_state.py` | ported |
| `scripts/design_v3/_sel_common.py` | `src/fluorovla/anatomy/cohort_selection.py` | consolidated |
| `scripts/design_v3/_sel_step3_final.py` | `src/fluorovla/anatomy/cohort_selection.py` | consolidated (select_development_cohort) |
| `scripts/medphysai/action.py` | `src/fluorovla/action.py` | ported; AST identical |
| `scripts/medphysai/drr_framing.py` | `src/fluorovla/observations/drr_framing.py` | ported; AST identical (patched-attribute name renamed) |
| `scripts/medphysai/patient_config.py` | `src/fluorovla/physics/patient_config.py` | replaced: explicit PatientConfig object instead of import-time global CFG; s0011 constants moved to artifacts/references/s0011/config.json |
| `scripts/medphysai/target_contract.py` | `src/fluorovla/task_contract.py` | ported; AST identical |
| `scripts/medphysai/teacher.py` | `src/fluorovla/teacher/privileged_teacher.py` | ported; AST identical |
| `scripts/medphysai/backends/newton_fluoro_observation.py` | `src/fluorovla/observations/fluoro_renderer.py` | ported with PatientConfig injection; NOT regression-tested (renderer stage deferred) |
| `scripts/medphysai/backends/newton_patient_episode.py` | `src/fluorovla/physics/patient_episode.py` | ported; takes PatientConfig; bit-identical results (regression A/B/C) |
| `scripts/medphysai/backends/newton_progression.py` | `src/fluorovla/physics/progression.py` | ported; AST identical |
| `scripts/medphysai/backends/newton_stage2d_engine.py` | `src/fluorovla/physics/engine.py` | ported; AST identical |
| `scripts/medphysai/backends/protocol.py` | `src/fluorovla/physics/backend_protocol.py` | ported; AST identical |
| `scripts/medphysai/groot/catheter_modality.py` | `src/fluorovla/policy/catheter_modality.py` | ported; not run |
| `scripts/medphysai/groot/episode_schema.py` | `src/fluorovla/data/episode_schema.py` | ported; firewall test added |
| `scripts/medphysai/groot/launch_catheter_finetune.py` | `src/fluorovla/policy/finetune_launcher.py` | ported; env knobs renamed FLUOROVLA_*; not run (training deferred) |
| `scripts/medphysai/groot/catheter_modality_config/modality.json` | `src/fluorovla/policy/catheter_modality_config/modality.json` | ported |
| `scripts/protocol/build_patient_episode_config.py` | `src/fluorovla/anatomy/episode_config.py` | ported as pure function; configs identical for 45 episodes |
| `scripts/protocol/cloud_batch.py` | `tools/run_physics_mapping.py` | ported |
| `scripts/protocol/cloud_finalize.py` | `src/fluorovla/evaluation/physics_mapping_summary.py` | ported |
| `scripts/protocol/compare_episode_physics.py` | `src/fluorovla/evaluation/episode_comparison.py + tools/check_reproduction.py` | ported |
| `scripts/protocol/compare_reproduction.py` | `src/fluorovla/evaluation/episode_comparison.py + tools/check_reproduction.py` | ported |
| `scripts/protocol/episode_batch.py` | `tools/run_physics_mapping.py` | superseded (explicit PatientConfig makes module purging unnecessary) |
| `scripts/protocol/episode_with_probe.py` | `src/fluorovla/physics/initial_state_probe.py + tools/run_physics_mapping.py` | ported |
| `scripts/protocol/junction_audit.py` | `artifacts/provenance + docs/provenance (audit result); tool not ported` | one-off topology audit; result recorded in protocol records |
| `scripts/protocol/make_manifest.py` | `tools/file_manifest.py + src/fluorovla/common/provenance.py` | ported |
| `scripts/protocol/record.py` | `src/fluorovla/common/provenance.py` | ported |
| `scripts/protocol/stage1_compare.py` | `src/fluorovla/anatomy/candidate_pool.py (side assignment)` | side-assignment logic ported; design-vs-twin comparison is provenance (docs/provenance/cohort_reconciliation.md) |
| `scripts/protocol/stage2_select.py` | `tools/select_development_cohort.py + src/fluorovla/anatomy/cohort_selection.py` | ported |
| `scripts/protocol/stage2_select_attempt2.py` | `tools/select_development_cohort.py` | same logic; attempt 2 is the locked selection |
| `scripts/protocol/stage3_summarize.py` | `src/fluorovla/evaluation/physics_mapping_summary.py + tools/summarize_physics_mapping.py` | superseded by attempt-addressable summary |
| `scripts/protocol/twin_build_win.py` | `src/fluorovla/anatomy/twin_build.py + tools/build_patient_twin.py` | ported |
| `scripts/protocol/twin_geometry.py` | `src/fluorovla/anatomy/access_geometry.py` | ported (LOOP_COLLAPSE + per-entry guard included) |
| `scripts/protocol/wsl_physics.sh` | `tools/infra/wsl_run.sh` | ported (local WSL workaround only) |

## Documents

| old_path | new path | status |
|---|---|---|
| `docs/cvpr_adaptive_teacher_multipatient_pilot_v2.md` | `docs/provenance/adaptive_teacher_design.md` | provenance |
| `docs/cvpr_cohort_reconciliation_v1.md` | `docs/provenance/cohort_reconciliation.md` | provenance |
| `docs/cvpr_contrast_visibility_v1.md` | `docs/provenance/contrast_visibility.md` | provenance |
| `docs/cvpr_dsa_roadmap_protocol_v1.md` | `docs/provenance/dsa_roadmap_protocol.md` | canonical |
| `docs/cvpr_final_experimental_protocol_v1.md` | `experiments/protocols/experimental_protocol_v1.md` | canonical |
| `docs/cvpr_multipatient_dataset_design_v1.md` | `docs/provenance/multipatient_dataset_design.md` | provenance |
| `docs/cvpr_pilot_cohort_selection_v1.md` | `docs/provenance/cohort_selection_design.md` | provenance |
| `docs/cvpr_stage3_native_linux_migration.md` | `docs/cloud_workflow.md (superseded)` | provenance |
| `docs/cvpr_stage3_preliminary_init_audit.md` | `docs/provenance/initial_state_preliminary_audit.md` | provenance |
| `docs/cvpr_target_contract_v1.md` | `docs/provenance/target_contract_design.md` | provenance |
| `docs/cvpr_visible_bifurcation_target_v1.md` | `docs/provenance/visible_bifurcation_target.md` | provenance |

## Artifact trees

| tree | category | note | new location |
|---|---|---|---|
| `outputs/cvpr_tiny_overfit_v1` (385 files, 154113.7 MB) | KEEP_EVIDENCE | GR00T tiny-overfit run (checkpoints, evals, logs); 154 GB; intermediate checkpoints are candidates for external archive | legacy archive |
| `data/ct_cache` (493 files, 20932.7 MB) | REGENERABLE_CACHE | patient twins (mu/hu volumes, masks, centerlines); rebuilt deterministically by tools/build_patient_twin.py; MOVED (hash-verified) to FluoroVLA/data/cache/twin | FluoroVLA/data/cache/twin |
| `data/Totalsegmentator_dataset_small_v201` (12037 files, 4216.4 MB) | KEEP_CANONICAL | raw CT + segmentations (authoritative raw data); MOVED (hash-verified) to FluoroVLA/data/raw/totalsegmentator | FluoroVLA/data/raw/totalsegmentator |
| `data/Totalsegmentator_dataset_small_v201.zip` (1 files, 3244.6 MB) | DUPLICATE | zip archive of the raw dataset that now lives, extracted and verified, in FluoroVLA/data/raw/totalsegmentator; kept (raw-data protection) | legacy archive (not deleted) |
| `migration/cloud_stage3_v1` (2102 files, 490.6 MB) | KEEP_REPRODUCTION | immutable pre-refactor recovery bundle | unchanged |
| `third_party/i4h-workflows` (1568 files, 478.3 MB) | KEEP_EVIDENCE | external i4h repository checkout (renderer / twin sources) | legacy archive; document pinned commit |
| `environments/windows-cuda-shim` (104 files, 192.0 MB) | OBSOLETE | Windows CUDA shim from the abandoned Windows-native renderer attempt | legacy archive |
| `outputs/cvpr_visible_bifurcation_v1` (283 files, 66.5 MB) | KEEP_REPRODUCTION | contains the accepted s0011 landing zone / path (copied to artifacts/references/s0011) | artifacts/references/s0011 + legacy archive |
| `outputs/cvpr_dsa_roadmap_v1` (233 files, 61.3 MB) | KEEP_EVIDENCE | DSA roadmap protocol outputs (needed again at DSA calibration) | legacy archive |
| `outputs/cvpr_first_demo_v2_holdtail` (331 files, 51.0 MB) | KEEP_REPRODUCTION | contains the accepted s0011 episode (copied to artifacts/references/s0011) | artifacts/references/s0011 + legacy archive |
| `outputs/cvpr_first_demo_v1` (288 files, 44.0 MB) | OBSOLETE | superseded by first_demo_v2_holdtail | legacy archive |
| `data/4d_medphys` (363 files, 22.5 MB) | KEEP_EVIDENCE | earlier 4D trajectory frames of s0011 | legacy archive |
| `outputs/cvpr_physical_reachability_v1` (135 files, 13.8 MB) | KEEP_EVIDENCE | earlier stage output (validation JSON/figures) | legacy archive |
| `migration` (2 files, 13.1 MB) | KEEP_EVIDENCE |  |  |
| `outputs/cvpr_target_contract_v1` (175 files, 7.2 MB) | KEEP_EVIDENCE | earlier stage output (validation JSON/figures) | legacy archive |
| `outputs/cvpr_teacher_validation_v1` (33 files, 5.6 MB) | KEEP_EVIDENCE | earlier stage output (validation JSON/figures) | legacy archive |
| `outputs/cvpr_protocol_v1` (503 files, 4.0 MB) | KEEP_EVIDENCE | protocol execution records, geometry, selection, local physics-mapping results and infrastructure logs | artifacts/provenance/protocol_records + artifacts/summaries + legacy archive |
| `outputs/stage6_5_2a_demo_logger` (69 files, 3.8 MB) | KEEP_EVIDENCE | earlier stage output (validation JSON/figures) | legacy archive |
| `assets` (3 files, 3.3 MB) | KEEP_EVIDENCE | supplementary figure S5 gallery | legacy archive |
| `outputs/stage6_2_1_prefix_consistency` (4 files, 2.3 MB) | KEEP_EVIDENCE | earlier stage output (validation JSON/figures) | legacy archive |
| `outputs/cvpr_multipatient_design_v1` (31 files, 2.3 MB) | KEEP_EVIDENCE | earlier stage output (validation JSON/figures) | legacy archive |
| `outputs/newton_drr_observation_bridge` (47 files, 2.2 MB) | KEEP_EVIDENCE | earlier stage output (validation JSON/figures) | legacy archive |
| `outputs/icra_pre_ai_audit` (51 files, 2.1 MB) | KEEP_EVIDENCE | earlier stage output (validation JSON/figures) | legacy archive |
| `outputs/groot_catheter_schema_smoke` (56 files, 2.1 MB) | KEEP_EVIDENCE | earlier stage output (validation JSON/figures) | legacy archive |
| `07_MedPhysAI_materials.zip` (1 files, 1.9 MB) | KEEP_EVIDENCE | thesis materials archive (not inspected for redundancy) | legacy archive |
| `outputs.zip` (1 files, 1.5 MB) | KEEP_EVIDENCE | archive of early outputs (not inspected for redundancy) | legacy archive |
| `scripts` (148 files, 1.4 MB) | KEEP_EVIDENCE |  |  |
| `=0.40` (1 files, 0.0 MB) | DELETE | stray file created by an unquoted pip command ('pip install pkg>=0.40' redirect); no content of value | (deleted) |

## Deleted (all hash-recorded before deletion)

- 96 bytecode files (`__pycache__`) under `scripts/`
- 6 scratch debug scripts of the design stage (`scripts/design_v3/_dbg_*.py`, `_debug_*.py`, `_test_s0011.py`)
- 1 empty stray file `=0.40` (an unquoted `pip install pkg>=0.40` redirect)

Not deleted although candidates: see `artifacts/provenance/deletion_manifest.json` (`proposed_not_executed`).

## Consolidated duplicate implementations

- lumen-mesh construction / SDF query / contact generation: three modules (Stage 4A, 4D, shared backend) -> `physics/lumen_sdf.py`
- `quat_rotate`: four copies (engine, Stage 4D, observation bridge, shared backend) -> `physics/engine.quat_rotate` (formulas verified identical)
- polyline-from-body-pose: observation module copy -> `physics/polyline.py`
- access-path / landing-zone construction: audit scripts for one patient + prototype extractor + twin extractor -> `anatomy/access_geometry.py`, `anatomy/episode_config.py`
- cohort selection: nine design-stage step scripts -> `anatomy/cohort_selection.py`
- episode comparison: two scripts -> `evaluation/episode_comparison.py`
- episode drivers: canonical script + batch + probe wrappers + cloud batch -> `physics/episode_runner.py` + `tools/run_physics_mapping.py`
- manifests / records: `record.py`, `make_manifest.py` -> `common/provenance.py`
