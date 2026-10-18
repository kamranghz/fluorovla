# FluoroVLA

Patient-specific, physics-based catheter-insertion demonstrations rendered as **simulated** fluoroscopic / DSA observations, for training and evaluating
Vision-Language-Action (VLA) policies. **Simulation only. Not clinical. Not validated on real fluoroscopy.**

```
patient CT / anatomy
  -> Privileged Teacher (anatomy-aware demonstration generator)
  -> Newton physics (rod + lumen contact, one decision = 0.1 s)
  -> rendered DSA / roadmap observation (320 x 320)
  -> VLA policy  (image + inserted length + previous velocity + fixed language -> insertion velocity)
  -> Newton physics (closed loop)
  -> evaluation on unseen patient-specific anatomy
```

## Scope of the claim

Closed-loop **insertion-only** navigation to an aorto-iliac bifurcation landing zone, on held-out admitted patient-specific anatomy, using physics simulation
and rendered fluoroscopic/DSA observations. The action is one scalar, the insertion velocity (HOLD 0, SLOW 6, CRUISE 12 mm/s). The system does not steer,
select branches, or claim clinical performance or real-fluoroscopy validity.

Terminology: **Privileged Teacher** = anatomy-aware demonstration generator (sees privileged geometry; the policy never does). **Teacher reference** = its
closed-loop result under the same evaluation conditions.

## Repository layout

| path | content |
|---|---|
| `src/fluorovla/anatomy` | twin build, access-path geometry, candidate pool, cohort selection, episode configuration |
| `src/fluorovla/physics` | Newton rod engine, lumen SDF contact, patient episode, episode runner |
| `src/fluorovla/teacher` | Privileged Teacher |
| `src/fluorovla/observations` | DRR framing and the Newton-to-DRR renderer bridge (renderer stage) |
| `src/fluorovla/data`, `policy` | GR00T/LeRobot dataset writer, modality, fine-tuning launcher (training stage) |
| `src/fluorovla/evaluation` | episode comparison, reproduction verdict, physics-mapping summary |
| `tools/` | thin command-line entry points (`tools/infra` = local WSL2 workarounds only) |
| `configs/` | fixed constants snapshot, cohort-selection parameters |
| `artifacts/` | locked cohort, authoritative candidate-pool geometry, s0011 reference, reproduction references, provenance records |
| `experiments/protocols/` | pre-registered experimental protocol |
| `docs/` | architecture, data layout, reproducibility, cloud workflow, refactor map |
| `data/`, `runs/`, `migration/` | not versioned (raw CT, twins, run outputs, bundles) |

## Quick start

```
pip install -r requirements.txt && pip install -e .          # Python 3.12; Newton install line is in requirements.txt
export FLUOROVLA_DATA=/path/to/data                           # contains raw/totalsegmentator and cache/twin
pytest -m "not gpu"                                           # fast checks: cohort lock, firewall, constants, config handling
pytest -m gpu                                                 # accepted s0011 reference + completed SUCCESS / LUMEN_VIOLATION reproduction (CUDA + Warp + Newton)
```

Pipeline (each step is a tool; see `docs/architecture.md` and `docs/reproducibility.md`):

1. `tools/build_patient_twin.py`            patient twin from CT + segmentation
2. `tools/verify_patient_geometry.py`       access paths, descriptors, candidate pool (>= 150 mm, valid 3-arm bifurcation)
3. `tools/select_development_cohort.py`     deterministic development cohort (`--lock artifacts/provenance/cohort_lock.json` verifies the locked one)
4. `tools/build_physics_mapping_configs.py` three start conditions per access side
5. `tools/run_physics_mapping.py`            Privileged-Teacher episodes on Newton physics (attempt-addressable, never overwrites)
6. `tools/summarize_physics_mapping.py`     episode / side / patient admission
7. `tools/check_reproduction.py`            bit-identical regression or the native-Linux reproduction verdict

## Status

Anatomical candidate pool: 65 patients (twin-verified). Development cohort (locked): s0011 s0963 s1400 s1364 s0950 s0344 s1388 s1348 s1397 s0513 s1369 s0573; stress
sentinel s1082. Physics mapping on the development cohort is paused until it is reproduced on native Linux (see `docs/cloud_workflow.md`). Later stages
(Teacher calibration, DSA calibration, dataset generation, multi-patient GR00T training, K-fold evaluation) are specified in
`experiments/protocols/experimental_protocol_v1.md` and have not been run.

Earlier project name: MedPhysAI (history is preserved in `artifacts/provenance` and `docs/provenance`).
