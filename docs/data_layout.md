# Data, run and artifact layout

Separation of concerns:

| class | location | versioned | notes |
|---|---|---|---|
| SOURCE | `src/`, `tools/` | yes | no generated data under source |
| CONFIG | `configs/` | yes | fixed-constants snapshot, cohort-selection parameters |
| RAW DATA | `data/raw/totalsegmentator/<patient>/{ct.nii.gz, segmentations/}` | no | authoritative; never modified or deleted |
| DERIVED / CACHE | `data/cache/twin/<patient>/` | no | regenerable: `mu_volume.npy`, `hu_volume.npy`, `vessel_mask.npy`, `centerline_*`, `metadata.json`, `lumen_collision_mesh_ct_frame.npz` |
| RUN OUTPUTS | `runs/<kind>/<run>/` | no | `geometry_verification`, `cohort_selection`, `physics_mapping`, `admission`, `demonstrations`, `closed_loop_eval` |
| AUTHORITATIVE SUMMARIES | `artifacts/summaries/`, `artifacts/references/` | yes | locked cohort geometry, s0011 reference, reproduction references |
| PROVENANCE | `artifacts/provenance/` | yes | cohort lock, protocol records, refactor map, deletion manifest |

`FLUOROVLA_DATA` (default `<repo>/data`) is the data root; episode configurations refer to data-root-relative paths (`cache/twin/<patient>`).

## Run convention

`runs/physics_mapping/<patient>_<side>_F<k>/attempt_NNN/` with `STARTED.json`, `run/episode_physics_ep.json` (+ `.npz`, `probe.json`), `DONE.json`
(`COMPLETED` or `PYTHON_ERROR`; absent = infrastructure interruption, never a physics result) and, once accepted, `ACCEPTED_ATTEMPT.json` one level up.
Other run kinds contain a `MANIFEST.json` written by `common.provenance.write_run_manifest`.

## Episode configuration (per start condition)

`config.json` (patient, side, twin directory, mesh cache and frame, portal CT position, tangent, SDF sign scale), `landing_zone.json` (zone in tip-extremity
coordinates, timeout `T_MAX`), `path_ext.npz` (extended centerline path and radii). See `physics/patient_config.py` for the numerics of the tangent.

## Episode result row (physics)

Policy-visible quantities: `inserted_length_m`, `previous_insert_velocity_mps`, `action` / `insert_velocity_mps`. Privileged quantities carry the `priv_` prefix
(tip extremity, polyline arclength, distance to the centerline, local radius) and are never written to dataset columns.
