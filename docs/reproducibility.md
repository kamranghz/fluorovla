# Reproducibility

## Reference stack (all results to date)

| item | value |
|---|---|
| Python | 3.12.13 |
| Newton | commit `6d2ece7c314248960a249ebc46b0dafbb5a89232` (clean checkout; install editable with `--no-deps`) |
| Warp | warp-lang 1.17.0 (CUDA toolkit 12.9) |
| NumPy / SciPy / scikit-image | 2.5.0 / 1.17.1 / 0.26.0 |
| GPU / driver | RTX 4090 (sm_89), driver 610.60 |
| OS (local) | Windows 11 + WSL2 Ubuntu 22.04.5, kernel 6.18.33.2-microsoft-standard-WSL2 |
| twin builder | i4h `vasculature_digital_twin` package, unmodified (copy it from the i4h image to a directory named by `VDT_PATH`); CPU, bit-identical to the container build |
| full package pins | `constraints.txt`, `requirements.txt` |

CUDA expectation: an NVIDIA GPU with compute capability >= 8.0 and a driver supporting the CUDA 12.9 runtime. Rendering (later stage) additionally needs the i4h
DiffDRR Slang renderer (`I4H_SIMULATION_PATH` or the i4h container).

## Accepted reference episode

`artifacts/references/accepted_reference/` holds the accepted episode (`accepted_episode.json`: SUCCESS, 105 rows incl. 15 post-success HOLD decisions), its landing zone, path
and configuration. Reproduce:

```
python tools/run_physics_mapping.py runs/regression/reference artifacts/references/accepted_reference --post-hold 15
python tools/check_reproduction.py artifacts/references/accepted_reference/accepted_episode.json runs/regression/reference/accepted_reference/attempt_001/run/episode_physics_ep.json --regression
```

Expected: `bit_identical_rows: true` (1680 numeric values compared). The same holds for the refactored code (verified; see `docs/project_cleanup_report.md`).

## Reproduction rule (pre-declared)

Preferred: bit-identical rows. If hardware/runtime differences prevent it: identical status, identical action sequence and decision count, and
<= 0.05 mm difference in `priv_tip_extremity_end_mm` and in `axis_violation_mm`. Anything else blocks (`tools/check_reproduction.py`, exit code 1).

## Errata policy

Scientific constants and algorithms change only through a documented erratum in `experiments/protocols/experimental_protocol_v1.md` (section "errata")
plus an update of `configs/physics/fixed_constants.json`; results from before and after are never mixed silently. Failed or interrupted evidence is kept.

## Determinism notes

- Warp kernels are deterministic on the reference stack; episodes within one process do not influence each other (regression: reference, then a LUMEN_VIOLATION
  episode, a SUCCESS episode and the same violation episode again are all bit-identical to their single-process results).
- The tangent of a patient configuration is normalized twice (once on read, once on use) to reproduce existing results bit-exactly (`physics/patient_config.py`).
- The lumen mesh is rebuilt by marching cubes when the cache is missing; the rebuilt mesh reproduces the old results bit-exactly.
