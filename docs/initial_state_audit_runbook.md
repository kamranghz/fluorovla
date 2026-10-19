# Initial-state / registration audit - native-Linux runbook

Purpose: diagnose whether the very-early LUMEN_VIOLATION pattern of the physics mapping comes from a systematic initialization / registration / coordinate-frame
problem or from the physical construction. **Diagnostic only**: no physics constant, Teacher rule or task definition is changed. Four cases: `s0011_right_F0` (SUCCESS,
positive control), `s0963_left_F1` (SUCCESS, positive control), `s0963_left_F2` (late LUMEN_VIOLATION, same anatomy as F1) and `s0344_left_F1` (immediate violation).
Do not run the 72-episode queue, SAFETY, DSA, LANDING, dataset generation or GR00T training.

The pod's `/workspace` may be ephemeral: **copy the results back (section 5) before stopping or terminating the pod.**

## 1. Upload and verify
```
scp -P <PORT> -i <KEYFILE> initial_state_audit_bundle.tar.gz initial_state_audit_bundle.tar.gz.sha256 root@<IP>:/workspace/       # from Windows (migration\)
cd /workspace && sha256sum -c initial_state_audit_bundle.tar.gz.sha256 && tar -xzf initial_state_audit_bundle.tar.gz
export BUNDLE=/workspace/initial_state_audit_bundle
cd $BUNDLE && sha256sum -c MANIFEST_SHA256.txt --quiet && echo MANIFEST_OK && cat BUNDLE_INFO.json
```

## 2. Environment (same pins as the physics-mapping runbook)
```
python3 -m venv /workspace/venv && . /workspace/venv/bin/activate && pip install --upgrade pip
cd $BUNDLE/project && pip install -r requirements.txt -c $BUNDLE/env/constraints.txt
pip install --no-deps -e $BUNDLE/env/newton-src && pip install --no-deps -e $BUNDLE/project
export FLUOROVLA_DATA=$BUNDLE/project/data && mkdir -p $BUNDLE/runs/{audit,logs}
python tools/check_environment.py $BUNDLE/runs/logs/ENVIRONMENT_CLOUD.json; echo exit=$?      # report any difference before running episodes
pip freeze > $BUNDLE/runs/logs/pip_freeze_cloud.txt
```

## 3. Tests (CPU and GPU) before the audit
```
pytest -q -m "not gpu" 2>&1 | tail -3        # CPU: includes the audit calculations and the verdict logic
pytest -q -m gpu 2>&1 | tail -5              # GPU: reference reproduction, observer read-only check, violation tracking (about 5 min)
```
All must pass. If the GPU tests fail, stop and report (the observer must not change the physics).

## 4. The audit (four cases, with the pre-declared initialization-only counterfactuals)
```
python tools/audit_initial_state.py $BUNDLE/runs/audit --resume \
   --configs-root $BUNDLE/workload/episode_configs --reference-runs $BUNDLE/workload/reference_runs \
   --episodes s0011_right_F0 s0963_left_F1 s0963_left_F2 s0344_left_F1 --variant-cap 20 2>&1 | tee $BUNDLE/runs/logs/audit.log
cat $BUNDLE/runs/audit/COMPARISON.md; cat $BUNDLE/runs/audit/STAGE3A_VERDICT.json
```
Each case first re-runs the production configuration with the read-only observer and checks that the instrumented episode is **bit-identical** to the original
(`instrumented_matches_original_bit_identically`); anything else means the observer is not read-only and the case is void. `--resume` reuses finished cases after an
interruption. Do not edit code; if a Python error occurs, send me `error` output instead of fixing it on the pod.

## 5. Package and copy back
```
history > $BUNDLE/runs/logs/COMMANDS_EXECUTED.txt
cd $BUNDLE/project && python tools/package_cloud_return.py $BUNDLE $BUNDLE/runs $BUNDLE/return_audit --commands $BUNDLE/runs/logs/COMMANDS_EXECUTED.txt
cd $BUNDLE && tar -czf /workspace/audit_return.tar.gz return_audit && sha256sum /workspace/audit_return.tar.gz | tee /workspace/audit_return.tar.gz.sha256
cat $BUNDLE/return_audit/CHANGED_FILES.json | head      # ADDED / MODIFIED / REMOVED must be empty
```
Back on Windows:
```
New-Item -ItemType Directory -Force D:\projects\FluoroVLA\runs\stage3a_audit | Out-Null
scp -P <PORT> -i <KEYFILE> root@<IP>:/workspace/audit_return.tar.gz root@<IP>:/workspace/audit_return.tar.gz.sha256 D:\projects\FluoroVLA\runs\stage3a_audit\
cd D:\projects\FluoroVLA\runs\stage3a_audit; Get-FileHash .\audit_return.tar.gz -Algorithm SHA256; tar -xzf audit_return.tar.gz
cd D:\projects\FluoroVLA; python tools\file_manifest.py runs\stage3a_audit\return_audit runs\stage3a_audit\return_audit\RETURN_SHA256.txt --verify      # mismatched/missing 0
```
Only then stop the pod. The verdict rules are in `src/fluorovla/evaluation/initial_state_verdict.py` (`PASS_INITIALIZATION_VALID`, `BUG_CONFIRMED`, `INCONCLUSIVE`);
the verdict file is produced by the tool and must be interpreted together with `docs/provenance/stage3a_initial_state_audit.md`.
