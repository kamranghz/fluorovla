# Lumen-containment audit - native-Linux runbook

Purpose: measure, with independent references, whether the CURRENT simulator geometry / measurement semantics (collision-mesh signed distance, axis-sample violation test,
straight pre-deployment) explain the lumen violations. **Diagnostic only**: no physics constant, Teacher rule, task contract, admission rule, DSA code or GR00T code is changed.
The design, the thresholds and the mechanism labels are described in `docs/provenance/stage3b_containment_audit.md`; the pre-declared cases and the synthetic design values are in
`configs/evaluation/containment_audit.json` and must not be edited after a real-data run.

Do not run the 72-episode queue, SAFETY, DSA, LANDING, dataset generation or GR00T training. The pod's `/workspace` may be ephemeral: copy the results back (section 7) before stopping it.

## 0. What the later cloud bundle needs (the bundle is NOT built by this change)

`tools/build_cloud_bundle.py` is deliberately unchanged. A separate, reviewed operational step must produce a bundle containing:

| item | content |
|---|---|
| project tree | the committed repository including `src/fluorovla/anatomy/{containment_audit,synthetic_geometry}.py`, `src/fluorovla/physics/containment_probe.py`, `src/fluorovla/evaluation/containment_verdict.py`, `tools/audit_containment.py`, `configs/evaluation/containment_audit.json`, the `tests/test_containment_*.py` files |
| episode configurations | `workload/episode_configs/<name>/{config.json, landing_zone.json, path_ext.npz}` for **all** configurations that should enter the cheap straight-deployment analysis (the 72 physics-mapping episodes) and, necessarily, for every case listed in `trace_cases` |
| twin data | for every patient that has a configuration: `vessel_mask.npy`, `metadata.json`, `centerline_points_mm.npy`, `centerline_radii_mm.npy` and the collision-mesh cache named in the configuration |
| reference runs | `workload/reference_runs/<case>/attempt_*/run/episode_physics_ep.json` for the trace cases (bit-identity check of the instrumented re-run) |
| environment | the pins of the previous native-Linux runbooks (`requirements.txt`, `constraints.txt`, `env/newton-src`); **no new package**. `trimesh` was not in the previous cloud environment: the secondary mesh reference is then reported UNAVAILABLE, which is allowed |
| manifest | `MANIFEST_SHA256.txt` and `BUNDLE_INFO.json` as for the previous bundles |

## 1. Environment and integrity verification
```
cd /workspace && sha256sum -c <bundle>.tar.gz.sha256 && tar -xzf <bundle>.tar.gz
export BUNDLE=/workspace/<bundle>; cd $BUNDLE && sha256sum -c MANIFEST_SHA256.txt --quiet && echo MANIFEST_OK && cat BUNDLE_INFO.json
python3 -m venv /workspace/venv && . /workspace/venv/bin/activate && pip install --upgrade pip
cd $BUNDLE/project && pip install -r requirements.txt -c $BUNDLE/env/constraints.txt
pip install --no-deps -e $BUNDLE/env/newton-src && pip install --no-deps -e $BUNDLE/project
export FLUOROVLA_DATA=$BUNDLE/project/data && mkdir -p $BUNDLE/runs/logs
python tools/check_environment.py $BUNDLE/runs/logs/ENVIRONMENT_CLOUD.json; echo exit=$?
pip freeze > $BUNDLE/runs/logs/pip_freeze_cloud.txt
```

## 2. Tests
```
pytest -q -m "not gpu" 2>&1 | tail -3    # includes the synthetic containment tests; the one known exact-float baseline difference must not appear on native Linux
pytest -q -m gpu 2>&1 | tail -5          # includes tests/test_containment_audit_gpu.py: probe read-only / bit-identical, engine SDF path on a synthetic twin
```
All must pass. If a GPU test fails, stop and report; do not edit code on the pod.

## 3. Preflight
```
CFG=$BUNDLE/workload/episode_configs
python tools/audit_containment.py preflight --configs-root $CFG --out $BUNDLE/runs/containment/preflight | tee $BUNDLE/runs/logs/preflight.log
```
Checks: native Linux, canonical constants derived from the source equal the running physics modules, a CUDA device, the secondary mesh reference status, every configuration readable
with its inputs present and `sign_scale = 1`, every pre-declared trace case present, and the declared calibration radius bins cover the twin radii along the audited centerline stretch of every
configuration (anatomy only), reported per case and informational only: a traced sample outside every bin gets no tolerance and its event is INDETERMINATE (RADIUS_OUTSIDE_CALIBRATED_BINS); coverage never aborts a run. Any other FAIL stops the audit.

## 4. Synthetic numerical calibration (the locked artifact)
```
python tools/audit_containment.py calibrate --configs-root $CFG --out $BUNDLE/runs/containment/calibration | tee $BUNDLE/runs/logs/calibrate.log
sha256sum $BUNDLE/runs/containment/calibration/numerical_tolerances.json
```
Produces `numerical_tolerances.json` (+ `.sha256`): the reference error per twin-radius bin for the primary variant (`fullres` or the half-voxel `fullres_hc`, chosen by the declared rule), the engine-query failure count and the
discretization EFFECTS per bin (not tolerances); `locked: true` only on native Linux with the engine SDF path calibrated. A file produced on Windows is `ENGINEERING_ONLY_*` and is
refused by every later mode. Record the printed hash in the run log before continuing.

## 5. Read-only trace of the pre-declared cases
```
python tools/audit_containment.py trace --configs-root $CFG --reference-runs $BUNDLE/workload/reference_runs \
   --calibration $BUNDLE/runs/containment/calibration/numerical_tolerances.json --out $BUNDLE/runs/containment/trace --resume | tee $BUNDLE/runs/logs/trace.log
```
Each case re-runs the production configuration with the read-only probe and checks that the instrumented episode is **bit-identical** to the original; anything else makes the trace
unusable (reason code `TRACE_NOT_BIT_IDENTICAL`).

## 6. Analysis and findings (CPU)
```
python tools/audit_containment.py analyze --configs-root $CFG --trace-dir $BUNDLE/runs/containment/trace \
   --calibration $BUNDLE/runs/containment/calibration/numerical_tolerances.json --out $BUNDLE/runs/containment/analysis --resume | tee $BUNDLE/runs/logs/analyze.log
python tools/audit_containment.py verdict --analysis-dir $BUNDLE/runs/containment/analysis \
   --calibration $BUNDLE/runs/containment/calibration/numerical_tolerances.json --out $BUNDLE/runs/containment/findings | tee $BUNDLE/runs/logs/verdict.log
cat $BUNDLE/runs/containment/findings/FINDINGS.md
```
`analyze` computes the exact straight-deployment feasibility and the centerline clearance for ALL configurations (full-resolution and stride-subsampled references) and the trace
comparison for the pre-declared cases only. `verdict` writes narrow per-case labels with reason codes and the specificity table (positive-control roles come from the configuration); the success rate never enters. The straight-deployment metrics of all
configurations are reported QUANTITATIVE_ONLY.

## 7. Package and return verification
```
history > $BUNDLE/runs/logs/COMMANDS_EXECUTED.txt
cd $BUNDLE/project && python tools/package_cloud_return.py $BUNDLE $BUNDLE/runs $BUNDLE/return_containment --commands $BUNDLE/runs/logs/COMMANDS_EXECUTED.txt
cd $BUNDLE && tar -czf /workspace/containment_return.tar.gz return_containment && sha256sum /workspace/containment_return.tar.gz | tee /workspace/containment_return.tar.gz.sha256
cat $BUNDLE/return_containment/CHANGED_FILES.json | head      # ADDED / MODIFIED / REMOVED must be empty
```
Back on Windows: `scp` the archive and its hash into `runs\containment_audit\`, verify `Get-FileHash`, extract, then
`python tools\file_manifest.py runs\containment_audit\return_containment runs\containment_audit\return_containment\RETURN_SHA256.txt --verify` (mismatched/missing 0). Only then stop the pod.

## Pitfalls
- `calibrate` and `trace` need CUDA; `analyze` and `verdict` are CPU but also refuse non-Linux platforms. `selftest` runs anywhere and is engineering validation only.
- Output directories are never overwritten; use `--resume` for `trace` and `analyze`.
- Do not edit code or configuration on the pod. A Python error is reported with its output, not fixed.
