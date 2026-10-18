# Architecture

## Data flow

```
raw CT + TotalSegmentator labels
   | tools/build_patient_twin.py   (anatomy/twin_build.py, unmodified i4h vasculature_digital_twin)
   v
patient twin  (mu volume, vessel mask, centerline graph)               data/cache/twin/<patient>
   | tools/verify_patient_geometry.py   (anatomy/access_geometry.py, candidate_pool.py)
   v
access paths per side (S_contact, seven descriptors), candidate pool    artifacts/references/candidate_pool_geometry
   | tools/select_development_cohort.py (anatomy/cohort_selection.py)   -> artifacts/provenance/cohort_lock.json
   | tools/build_physics_mapping_configs.py (anatomy/episode_config.py)
   v
episode configuration = PatientConfig + landing zone + extended path    (config.json, landing_zone.json, path_ext.npz)
   | tools/run_physics_mapping.py   (physics/episode_runner.py)
   v
Privileged-Teacher episode on Newton physics                            runs/physics_mapping/<patient>_<side>_F<k>/attempt_NNN/
   | tools/summarize_physics_mapping.py (evaluation/physics_mapping_summary.py)
   v
admission: side admitted iff SUCCESS at all three start conditions; patient admitted iff >= 1 side
```

Later stages (not yet run): rendered DSA observations (`observations/`), dataset writing (`data/episode_schema.py`), GR00T fine-tuning (`policy/`), closed-loop
evaluation (`evaluation/`).

## Module responsibilities

| module | responsibility | key types / functions |
|---|---|---|
| `task_contract` | task definition: velocities, decision period, success window, dwell, lumen tolerance, `PathProjector`, `evaluate` | constants, `PathProjector`, `TargetSpec` |
| `action` | backend-independent action API | `CatheterAction`, `HOLD`, `SLOW`, `CRUISE` |
| `teacher.privileged_teacher` | anatomy-aware rule: CRUISE until 2.4 mm before the zone, SLOW, HOLD | `TeacherState`, `teacher_action`, `POLICY_OBSERVATION_KEYS` |
| `physics.engine` | Newton 7-body rod, material feed, activation, contact materials | `build_scene`, feed/pose configuration |
| `physics.patient_config` | the only entry of patient geometry into the physics | `PatientConfig` |
| `physics.lumen_sdf` | lumen collision mesh, SDF query, contact generation | `LumenSDF`, `generate_lumen_contacts` |
| `physics.patient_episode` | decision-stepped patient episode | `PatientEpisode(cfg)` |
| `physics.episode_runner` | closed-loop Teacher episode with the success/failure taxonomy | `run_physics_episode` |
| `anatomy.access_geometry` | entry, first junction cluster, extended path, S_contact, descriptors | `extract_patient_geometry` |
| `anatomy.candidate_pool` | side assignment (TotalSegmentator labels), descriptor rows, pool definition | `assign_sides`, `candidate_pool` |
| `anatomy.cohort_selection` | deterministic representative-patient selection | `select_development_cohort` |
| `anatomy.episode_config` | portal, tangent, landing zone, timeout, start depths | `build_episode_config` |
| `evaluation.*` | regression / reproduction comparison, attempt-addressable summaries | `compare_regression`, `reproduction_verdict`, `summarize` |
| `common.provenance` | SHA-256 manifests, environment capture, run manifests | `write_run_manifest`, `verify_file_manifest` |

## Design rules

- Patient geometry enters only through `PatientConfig`; no module reads a patient constant at import time and no source file contains a patient identifier
  in logic (`tests/test_code_hygiene.py`).
- Fixed physical/task constants live in the modules that own them and are guarded by `configs/physics/fixed_constants.json` (`tests/test_fixed_constants`).
  Changing one needs a protocol erratum.
- The policy sees only `POLICY_OBSERVATION_KEYS`; everything `priv_*` / Teacher-only stays out of the dataset columns (`tests/test_privileged_firewall.py`).
- Source never lives under `runs/`, `artifacts/` or `data/`; generated outputs are never imported.
- Every run directory contains a `MANIFEST.json` (purpose, patients, config, code hash, environment, timestamp, status); evidence is never overwritten.

## Terminology

Privileged Teacher = anatomy-aware demonstration generator. Teacher reference = its closed-loop result under the same evaluation conditions. No upper-bound
claim is made unless operationally justified.
