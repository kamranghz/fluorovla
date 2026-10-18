# Project cleanup and refactor report

Date: 2026-10-18. Project renamed **MedPhysAI -> FluoroVLA** (repository slug `fluorovla`, package `fluorovla`). Scientific behaviour is unchanged (see regression).

## Name

FluoroVLA: *fluoro*(scopic) observations for a *VLA* policy. Honest limits stay in the README (simulation only, rendered DSA, insertion-only, no steering, no clinical claim).
Alternatives considered: SimFluoroVLA (most explicit about simulation, clumsy), DSA-VLA (narrower than the roadmap/fluoro observations), InsertVLA (accurate for the action,
loses the imaging contribution). Naming risks: "Fluoro" can be read as real fluoroscopy; "VLA" as a generalist robot policy. Mitigation: the README states the scope
claim first and the repository description should read "simulated fluoroscopic/DSA, insertion-only".

## Before / after

| | files | size |
|---|---|---|
| BEFORE: single tree `D:\projects\MedPhysAI` (SHA-256 snapshot `PRE_REFACTOR_SHA256.txt`) | 19,820 | 184.0 GB |
| AFTER: Git repository `D:\projects\FluoroVLA` (tracked) | see `git ls-files` (129 at the refactor commits) | 1.9 MB |
| AFTER: `D:\projects\FluoroVLA\data` (not versioned: raw CT 12,0xx files, 69 twins) | 12,535 | 25.1 GB |
| AFTER: `D:\projects\FluoroVLA\runs` (regression-check outputs, not versioned) | 249 | 1.9 MB |
| AFTER: legacy archive `D:\projects\MedPhysAI` (unchanged evidence, immutable v1 bundle, GR00T tiny-overfit outputs) | 7,184 | 158.8 GB |

The file count of the whole project did not fall: 12.5k of the 17-20k files are the authoritative raw CT and its derived twins, which are neither deleted nor versioned.
What shrank is the code/document/config surface: 284 scripts + 91 tool files + 33 documents scattered over stage-named directories became 129 tracked files
(30 source modules, 23 tools, 7 test modules, 18 documents).

## Removed (all hashed before removal; 103 files, 1.5 MB)

96 bytecode files, 6 scratch debug scripts (`scripts/design_v3/_dbg_*`, `_debug_*`, `_test_s0011.py`), 1 empty stray file (`=0.40`). Nothing else was deleted.
Moved without loss (12,530 files, every SHA-256 re-verified): raw CT dataset -> `data/raw/totalsegmentator`, twin caches -> `data/cache/twin`.
Candidates NOT deleted because they need your decision (`artifacts/provenance/deletion_manifest.json`): intermediate GR00T checkpoint copies of the finished tiny-overfit
run (37.7 GB) and the training-run checkpoints duplicated in them (53.1 GB); regenerable caches (22 GB of mu/hu volumes); the zip of the raw dataset (3.2 GB).
Byte-identical duplicates found: 568 groups, 235 MB (listed in the audit records; not removed because they sit in evidence trees).
`migration/cloud_stage3_v1` is untouched (manifest and tarball re-verified after every step).

## Renames and consolidation

See `docs/refactor_map.md` (46 canonical code files mapped, every document and artifact tree classified, 430 entries in `artifacts/provenance/refactor_map.json`).
Consolidated implementations: lumen mesh/SDF/contact (3 modules -> 1), `quat_rotate` (4 -> 1), polyline conversion, access-path/landing-zone construction (3 -> 2 modules),
cohort selection (9 step scripts -> 1 module), episode comparison (2 -> 1), episode drivers (canonical + batch + probe + cloud wrappers -> runner + 1 tool),
manifests/records (2 -> 1). Import-time global patient configuration (`CFG`) was replaced by an explicit `PatientConfig`; the module-purge workaround is no longer needed.

## Scientific regression (all must be bit-identical; they are)

| check | result |
|---|---|
| A accepted reference episode (105 rows, 1,680 values) | bit-identical |
| B completed SUCCESS (`s0963_left_F1`, 64 rows) | bit-identical |
| C completed LUMEN_VIOLATION (`s0963_left_F2`, 37 rows), run again after B in the same process | bit-identical (both) |
| four more completed episodes of other patients (s1400, s1397, s0950, s1348) | bit-identical |
| mesh caches regenerated under the new names (marching cubes) | results unchanged |
| seven ported modules | AST identical to the legacy ones (docstrings/comments ignored) |
| twin geometry extraction for all 69 patients | descriptor rows and path arrays identical; pool 65, 125 candidate paths |
| locked development-cohort selection from regenerated descriptors | reproduces the lock exactly |
| 45 episode configurations regenerated | identical to the legacy ones |
| unit tests (`pytest -m "not gpu"`) | 26 passed |
Details with SHA-256: `artifacts/summaries/refactor_regression_physics.json`, `refactor_regression_other.json`.

## Behaviour differences

None in numerical results. Intentional interface changes: patient geometry via `PatientConfig`; episode directories and run names are semantic; env knobs of the
GR00T launcher are `FLUOROVLA_*` (was `MEDPHYSAI_*`); the s0011 constants live in `artifacts/references/accepted_reference/config.json`.

## Not regression-tested after the port (flagged)

`observations/fluoro_renderer.py` (needs the i4h DiffDRR renderer), `policy/*`, `data/episode_schema.py` (firewall checked statically), `tools/groot/*` and
`tools/infra/supervise_groot_training.ps1` (GR00T container stage). They were ported mechanically (names, imports, config injection) and must be regression-tested when
the DSA calibration and training stages start. The renderer/DSA exploration scripts were deliberately not ported (kept in the legacy archive and mapped as KEEP_EVIDENCE).

## Remaining technical debt

- renderer and training stages as above; DSA roadmap protocol still lives as exploration scripts in the archive.
- `physics/engine.py` is 800 lines and still contains a free-space mode that the patient pipeline does not use (kept: validated, AST-identical).
- the local WSL2 workaround scripts (`tools/infra`) should be deleted once all GPU work runs on native Linux.
- initial-state audit of the multi-patient episodes (41 of 47 local violations occur within 5 decisions) is deferred until the native-Linux reproduction passes.
- the legacy folder is still named `MedPhysAI` and the Newton venv `MedPhysAI.venv-newton` (renaming them needs a fresh session).
