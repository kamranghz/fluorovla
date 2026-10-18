# Cloud workflow (native Linux) - physics mapping

The cloud is a **compute environment**, not the canonical store. The local Git repository is canonical; everything produced remotely returns with hashes.
Local WSL2 is unreliable for CUDA work (kernel VMA-lock hangs and 9p I/O errors observed repeatedly); hangs are infrastructure events, never physics results.

## 1. Environment contract

| item | requirement |
|---|---|
| OS | Ubuntu 22.04 or 24.04 x86_64, native (not WSL, not nested virtualization), stock distribution kernel >= 5.15 |
| GPU | NVIDIA, compute capability >= 8.0 (reference: RTX 4090, sm_89), >= 8 GB VRAM (episodes use < 2 GB) |
| driver / CUDA | driver supporting the CUDA 12.9 runtime (R570 or newer); reference driver 610.60 |
| Python | 3.12.x |
| packages | `requirements.txt` + `constraints.txt` (warp-lang 1.17.0, numpy 2.5.0, scipy 1.17.1, scikit-image 0.26.0); Newton from `env/newton-src` (commit `6d2ece7c314248960a249ebc46b0dafbb5a89232`), installed editable with `--no-deps` |
| environment variables | `FLUOROVLA_DATA=$BUNDLE/project/data` (no other variable is required; do not set allocator workarounds) |
| RAM / storage | >= 16 GB RAM; bundle ~0.6 GB, run outputs ~0.5 GB for 3 attempts of every episode; allow 20 GB |

Persistence: keep bundle and runs on a persistent volume. **Stop**: persistent disks survive, a running queue is interrupted (resume = new attempts for episodes
without a result). **Terminate**: instance-store disks are lost; persistent volumes survive only if delete-on-termination is off. Download results after every ~20
episodes. Upload/download: `tar` + `sha256sum`, `rsync -avP --partial` or `scp`; always re-verify with section 2 step 1.

## 2. Runbook

```
export BUNDLE=$HOME/cloud_stage3_v2           # the uploaded and extracted bundle
export FLUOROVLA_DATA=$BUNDLE/project/data
cd $BUNDLE/project
mkdir -p $BUNDLE/runs/{repro,physics_mapping,logs}
```

**1. Verify the upload (must print OK; any failure blocks everything)**
```
cd $BUNDLE && sha256sum -c MANIFEST_SHA256.txt --quiet && echo OK; cat BUNDLE_INFO.json
```

**2. Reconstruct and verify the environment**
```
python3.12 -m venv $HOME/venv-fluorovla && . $HOME/venv-fluorovla/bin/activate
pip install -r $BUNDLE/project/requirements.txt -c $BUNDLE/env/constraints.txt
pip install --no-deps -e $BUNDLE/env/newton-src          # Newton commit in env/NEWTON_COMMIT.txt
pip install --no-deps -e $BUNDLE/project
cd $BUNDLE/project && python tools/check_environment.py $BUNDLE/runs/logs/ENVIRONMENT_CLOUD.json     # exit 1 lists every difference from the reference stack
pytest -m "not gpu" -q                                                                                # fast checks (cohort lock, firewall, constants, hygiene)
```
A difference in warp/numpy/scipy/scikit-image or Newton is recorded; it makes bit identity unlikely and moves the decision to the tolerance rule.

**3. Reproduce the accepted reference episode (A)**
```
python tools/run_physics_mapping.py $BUNDLE/runs/repro artifacts/references/accepted_reference --post-hold 15 2>&1 | tee $BUNDLE/runs/logs/repro_A.log
```

**4. Reproduce one completed SUCCESS (B) and 5. one completed LUMEN_VIOLATION (C)**
```
python tools/run_physics_mapping.py $BUNDLE/runs/repro artifacts/references/reproduction/completed_success artifacts/references/reproduction/completed_lumen_violation 2>&1 | tee $BUNDLE/runs/logs/repro_BC.log
```

**6. Compare and 7. decide PASS / BLOCK (pre-declared rule)**
```
python tools/check_cloud_reproduction.py $BUNDLE/runs/repro $BUNDLE/runs/logs/REPRODUCTION_VERDICT.json ; echo exit=$?
python tools/check_reproduction.py artifacts/references/accepted_reference/accepted_episode.json $BUNDLE/runs/repro/accepted_reference/attempt_001/run/episode_physics_ep.json --regression    # strict row-level check
```
Rule: preferred bit-identical rows. If hardware/runtime differences prevent bit identity: identical status, identical action sequence and decision count, and
<= 0.05 mm in `priv_tip_extremity_end_mm` and in `axis_violation_mm`. Any status or action-sequence difference **blocks**: stop, do not run the queue, report.
If A, B and C pass but the violation rate stays high, **stop before any admission conclusion** and run the initial-state audit
(`docs/provenance/initial_state_preliminary_audit.md`); never retune physics.

**8. Resume the Stage-3 physics-mapping queue (only after PASS)**
```
python tools/list_workload.py $BUNDLE/workload/workload.json pending > $BUNDLE/runs/logs/pending.txt
nohup python tools/run_physics_mapping.py $BUNDLE/runs/physics_mapping $(cat $BUNDLE/runs/logs/pending.txt) > $BUNDLE/runs/logs/physics_mapping.log 2>&1 &
# then re-run the episodes that have a local result on the stable stack (cross-check and single-environment table):
python tools/list_workload.py $BUNDLE/workload/workload.json completed_local > $BUNDLE/runs/logs/completed_local.txt
python tools/run_physics_mapping.py $BUNDLE/runs/physics_mapping $(cat $BUNDLE/runs/logs/completed_local.txt) > $BUNDLE/runs/logs/physics_mapping_recheck.log 2>&1
```
Re-running an episode creates `attempt_002`; `attempt_001` is never touched. After an unexpected termination restart, repeat step 1, then run the episodes whose
newest attempt has no `DONE.json` (they are infrastructure interruptions, not physics results). One line per rerun with its reason goes to `$BUNDLE/runs/logs/RERUN_REASONS.md`.
The scientific timeout `T_MAX` is stored in each `landing_zone.json`; no wall-clock watchdog may alter or shorten it.

**9. Package the final results**
```
python tools/summarize_physics_mapping.py $BUNDLE/runs/physics_mapping $BUNDLE/workload/workload.json $BUNDLE/runs/logs/STAGE3_FINAL_SUMMARY.json
history > $BUNDLE/runs/logs/COMMANDS_EXECUTED.txt
python tools/package_cloud_return.py $BUNDLE $BUNDLE/runs $BUNDLE/return --commands $BUNDLE/runs/logs/COMMANDS_EXECUTED.txt
tar -C $BUNDLE -czf $BUNDLE/stage3_cloud_return.tar.gz return && sha256sum $BUNDLE/stage3_cloud_return.tar.gz > $BUNDLE/stage3_cloud_return.tar.gz.sha256
```
`package_cloud_return.py` writes ENVIRONMENT_CLOUD.json, CHANGED_FILES.json (ADDED / MODIFIED / REMOVED / UNCHANGED vs the uploaded manifest, with before/after
SHA-256; every changed file is copied back), STAGE3_FINAL_SUMMARY.json, RETURN_MANIFEST.json and RETURN_SHA256.txt. Scientific code stays unchanged unless a
demonstrable implementation bug is found; any fix is an erratum in the protocol and travels back in `changed_files/`.

**10. Return to the local canonical project**
```
# on the local machine, after downloading stage3_cloud_return.tar.gz and its .sha256
sha256sum -c stage3_cloud_return.tar.gz.sha256
mkdir -p runs/physics_mapping_cloud && tar -C runs/physics_mapping_cloud -xzf stage3_cloud_return.tar.gz
python tools/file_manifest.py runs/physics_mapping_cloud/return runs/physics_mapping_cloud/return/RETURN_SHA256.txt --verify     # must print "mismatched/missing 0"
```
Accept the artifacts only after both checks pass; then write the stage record and copy any `changed_files/` into the repository through a normal commit. Do not
terminate the instance before the local verification and a local backup.

## 3. What each attempt directory means

`runs/physics_mapping/<patient>_<side>_F<k>/attempt_NNN/`: `STARTED.json`; `run/episode_physics_ep.json` and `probe.json` when the episode ran; `DONE.json`
(`COMPLETED` or `PYTHON_ERROR`); `error.txt` for Python errors. A missing `DONE.json` is an infrastructure interruption. `ACCEPTED_ATTEMPT.json` (written by the
summary step) names the accepted attempt (the last one with a result) and lists the states of all attempts.

## 4. Scope limit

This workflow runs physics mapping only. SAFETY testing, DSA calibration, LANDING recalibration, dataset generation and GR00T training are not started until the
native-Linux mapping is reproduced and reviewed.
