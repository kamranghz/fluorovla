# Stage 3B: lumen-containment audit

Status: **tooling implemented and tested on synthetic geometry; the native-Linux calibration and the real-data audit have NOT been run.** No finding exists yet. The Stage 3A documents
(`docs/scientific_protocol.md`, `docs/provenance/stage3a_initial_state_audit.md`) are known to be stale and are intentionally untouched here; they are read as evidence only.

## 1. Question and scope

Does the current simulator geometry / measurement semantics explain the early lumen violations? The audit examines a **bounded** mechanism set and gives per-case findings only (revision of 2026-10-28; the
earlier draft of the mechanism set was never run on real data). Numerical error of a reference is separated from the effect under test: the reference error enters the tolerances, the collision-surface
discretization is the effect and is never a tolerance.

| mechanism (traced cases only) | meaning of CONFIRMED |
|---|---|
| `COLLISION_DISCRETIZATION_DECISION_FLIP` | at the terminating event (the trigger event if the engine's violation exceeds `LUMEN_DELTA_MM` there, else the peak event; chosen from the trace and the protocol limit, not from the outcome label) the engine violation exceeds `LUMEN_DELTA_MM` while the primary-reference violation upper bound, `max(0, max(phi_i + tol_i))` with the calibrated error `tol_i` of the twin-radius bin of each sample, does not |
| `PRE_ACTION_INVALIDITY` | at the post-settle `ready` state the primary-reference violation lower bound, `max(0, max(phi_i - tol_i))`, already exceeds `LUMEN_DELTA_MM`; the straight-rod geometry is reported as evidence, not as the criterion |

**Specificity rule.** If any case whose pre-declared role (configuration, `stage3a_role`) is `positive_control` satisfies a mechanism criterion, that mechanism is `NON_DISCRIMINATING` for the whole audit and cannot be
CONFIRMED in any case (each case then carries the raw status as evidence). Roles never come from outcome statistics. The output lists, per mechanism, the raw status of the positive controls and whether
specificity was established (at least one control decided `NOT_SUPPORTED` and none fired).

`QUANTITATIVE_ONLY` (measured and reported, never a mechanism claim): the analytic straight-deployment metrics of ALL configurations (the protocol test never applies before the settle phases), collision-mesh offset,
centerline clearance against the catheter radius, axis-versus-body gap, engine query integrity (engine/reference sign disagreements, engine versus its own CPU mesh when available), and - descriptively - the median signed
(engine - reference) difference at contact-active samples against the calibrated reference error band. Case labels: `<MECHANISM>_CONFIRMED`, `MULTIPLE_MECHANISMS_IDENTIFIED`,
`NO_CONTAINMENT_AUDIT_MECHANISM_IDENTIFIED` (both mechanisms decisively not supported), `INCONCLUSIVE` (reason codes; includes NON_DISCRIMINATING). There is no global verdict. No outcome statistic is an input.

## 2. Semantics verified in the canonical source (Phase 0)

| item | verified fact (file) |
|---|---|
| SDF sign | `phi = sign_scale * mesh_query_sign * distance`, phi < 0 inside the lumen; `sign_scale` is a configuration value (all audited configurations: 1) (`lumen_sdf.py`) |
| catheter radius | `RADIUS = 0.003` m (`engine.py`); `CATHETER_RADIUS_M = float(RADIUS)`; read by the audit through `containment_audit.canonical_constants` (AST of the source, guard against drift) |
| contact activation | a rod axis sample generates a contact when `phi > TARGET_PHI_M`, `TARGET_PHI_M = -CATHETER_RADIUS_M` (`lumen_sdf.generate_lumen_contacts`) |
| violation metric | `measure_outside`: `max(0, max phi)` over the 5 axis samples of every non-root active body; compared with `LUMEN_DELTA_MM = 1.5` (`task_contract`, `episode_runner`: strictly greater terminates). It is an axis measure and ignores the radius |
| axis samples | per non-root body, fractions `SAMPLE_FRAC = (0, .25, .5, .75, 1)` at `com + (f - 0.5) * L_SEG * axis`; `L_SEG = 0.03` m; 7 bodies, root excluded; 3 bodies active after the settle phases (`PREDEPLOY_ACTIVE`) |
| collision mesh | marching cubes level 0.5 on `vessel_mask[::2, ::2, ::2]` (`MASK_DOWNSAMPLE = 2`), spacing x 2; a query farther than `MAX_DIST_M = 0.05` m has no hit and returns +0.05 m |
| real spacing | `metadata.json` (`spacing_zyx_mm`, `origin_xyz_mm`), per configuration; read by `calibrate` for the synthetic voxel grids |
| initial rod | straight along the configured tangent, root body centred on the portal, body k centred at `k * L_SEG` (checked against the recorded start state by the probe: `START_NOT_ANALYTIC_STRAIGHT`) |
| observer | `PatientEpisode(cfg, observer=...)` calls `observer(stage, episode, info)` at the checkpoints and every decision substep and gives access to bodies, state, SDF and contacts; no existing physics file is edited |
| synthetic mesh without touching caches | `PatientConfig` is a plain dataclass; a synthetic twin with its own `mesh_cache` path builds its mesh there (verified locally) |

**Derived penetration quantity.** A radius-R sphere centred on an axis sample overlaps the exterior iff its distance to the exterior, `-phi`, is smaller than R, i.e. iff
`phi - TARGET_PHI_M > 0` (`TARGET_PHI_M = -R`). Hence the body (capsule-surface) penetration of a sample is `max(0, phi - TARGET_PHI_M)` and the contact criterion is exactly its positivity.
This coincides with `max(0, phi + R)` only because the canonical source defines `TARGET_PHI_M = -R`; the audit reads `TARGET_PHI_M` and fails with `CanonicalConstantsDrift` otherwise.
A contact is generated only at the 5 axis samples, so the continuous body surface between samples is not tested by the engine; the audit reports the dense (continuous) axis and body penetration separately.

## 3. References (independent of the engine)

- **Primary** `fullres`: spacing-aware signed EDT of the full-resolution mask (`registration_audit.signed_distance_mm`, voxel convention `world = origin + index * spacing`), trilinear interpolation. `registration_audit` is unchanged.
- **Variant** `fullres_hc` (half-voxel, face-centred): the same EDT with each side shortened by half the finest spacing, so the wall lies on the voxel face and the interpolated field has unit slope across it (the centre-to-centre field has
  slope 2 there). It replaces `fullres` as the primary reference only if, on the pooled synthetic calibration, it has the lower |median signed error| **and** the lower p99 |error| (`variant_selection_rule`, declared in the configuration before the run);
  both variants are reported.
- `ds`: the same construction on the mask subsampled with the collision-mesh stride (separates mask resolution from the marching-cubes surface). It is an EFFECT measure.
- **Secondary** `mesh`: trimesh signed distance to a CPU rebuild of the collision mesh; imported lazily and optional (`trimesh` was NOT in the previous cloud `pip freeze`, no package is installed for it, so it is likely UNAVAILABLE on the cloud);
  validated on an analytic sphere, otherwise UNAVAILABLE. No custom winding-number / ray-parity code exists. Without it the engine query is checked by wrong-sign detection against the analytic truth (see below).
- Every reference is validated on analytic tubes (`synthetic_geometry`) before use, and its numerical error is measured in the calibration.

## 4. Threshold provenance (every conclusion-driving value)

| value | class | source |
|---|---|---|
| `LUMEN_DELTA_MM` (1.5 mm) | PROTOCOL | `task_contract` |
| contact threshold `-R` | IMPLEMENTATION | `lumen_sdf.TARGET_PHI_M` |
| observer consistency 1e-6 mm; start-state match 1e-3 mm | IMPLEMENTATION | float32 query / pose representation (1e-3 mm is the tolerance used when the Stage 3A observer was validated) |
| engine query vs a CPU query on the same triangles, 1e-3 mm | IMPLEMENTATION | float32 Warp query; larger differences are counted as rare query failures (rate, examples), not as discretization |
| `fullres_reference_vs_analytic` per twin-radius bin (primary variant) | SYNTHETIC_VALIDATION | `calibrate`, native Linux, locked by SHA-256 before any patient analysis; each real sample uses the tolerance of the bin of its local twin radius |
| collision-mesh vertex deviation bound (sign guard of the engine query) | SYNTHETIC_VALIDATION | `calibrate` |
| (none) | HEURISTIC | no heuristic threshold is defined; the code refuses a CONFIRMED status backed by a HEURISTIC class |

Removed from the earlier draft: `engine_vs_analytic`, `engine_vs_fullres = engine_vs_analytic + fullres_reference_vs_analytic` and `ds_reference_vs_analytic` as tolerances. They measured the effect under test. The calibration still reports, per
radius bin and labelled as EFFECTS, `ds`, CPU-mesh and engine against the analytic truth (engine after removal of counted query failures).
Voxel spacings of the synthetic grids come from the configurations; the radius bins, tube radii, bend radii, orientations and sub-voxel offsets are design values declared in `configs/evaluation/containment_audit.json` (2026-10-28) before any real-data
analysis. `preflight` and `calibrate` compare the bins with the twin radii along the audited centerline stretch of every configuration (anatomy only) and report the coverage per case; this never aborts a run. A traced sample whose local twin radius is outside every bin has no tolerance (UNAVAILABLE) and its event is INDETERMINATE with reason RADIUS_OUTSIDE_CALIBRATED_BINS. The declared bins are 1.5-2.0, 2-3, 3-4, 4-6, 6-9 and 9-12 mm.

A **secondary, descriptive-only** analysis `SECONDARY_P99_DESCRIPTIVE` re-evaluates both criteria with the per-bin p99 |reference error| instead of the maximum. Its threshold is HEURISTIC, it is reported as QUANTITATIVE_ONLY, it can never be CONFIRMED (the code refuses it), and no status, specificity decision or label reads it.

## 5. Pre-declared trace cases

Primary (Stage 3A roles): `s0011_right_F0` (control), `s0963_left_F1` (control), `s0963_left_F2` (late failure), `s0344_left_F1` (early failure).
Supplementary: `s0011_left_F1`, named in the committed Stage 3A document (section 7, item 3, commit `1cb5008`) as a straight start that still fails at decision 1; its selection predates this
audit and uses none of its statistics. It can be disabled by removing it from `trace_cases` before the run. No case is added because of an outcome or curvature statistic; broader evidence comes
from the cheap analytic straight-deployment analysis over all configurations.

## 6. ENGINEERING VALIDATION ONLY (local Windows runs, not the scientific calibration)

Development machine: Windows 11, RTX 4090, **Warp 1.15.0 (the cloud pin is 1.17.0)**; synthetic geometry only. None of these numbers is a tolerance for any patient analysis.

- Run 1 (engine path on the GPU, previous design, 2.0-5.0 mm tubes, 1.5 mm voxels): engine against its own CPU mesh identical to numerical precision for 215,999 of 216,000 queries; one wrong-sign query (engine +4.33 mm, truth about -4.4 mm); a separate
  400,000-query run reproduced one more (engine +3.73 mm, truth -4.34 mm, CPU mesh correct). About 5e-6 per near-surface query with the local Warp. To be re-checked on native Linux with Warp 1.17; this is why query failures are counted separately and
  engine/reference sign disagreements are recorded per event and block CONFIRMED when they occur at the decision event.
- Run 2 (CPU only, revised design 2026-10-28, 1.5 mm voxels, no engine): primary variant selected by the declared rule = `fullres_hc` (pooled |median| 0.13 mm vs 0.49 mm, p99 0.72 mm vs 1.38 mm; max 1.51 mm vs 1.63 mm).
  Binned reference error (max |error|): 1.39 / 1.49 / 1.51 / 1.35 mm for the radius bins 2-3 / 3-4 / 4-6 / 6-9 mm. Collision-resolution effect of `ds` (max, same bins): 5.02 / 3.32 / 3.13 / 3.18 mm; the thin-tube bin dominates.

Consequence: the worst-case reference error (about 1.4-1.5 mm) is about the size of `LUMEN_DELTA_MM` (1.5 mm), so the reference interval `[phi - tol, phi + tol]` is wide. `COLLISION_DISCRETIZATION_DECISION_FLIP` can only be CONFIRMED where the reference
violation is about zero while the engine exceeds the limit; `PRE_ACTION_INVALIDITY` only where the reference violation exceeds about 2.9 mm. Intermediate cases are INDETERMINATE by construction, not by accident.

## 7. Existing issues recorded, intentionally not fixed here

1. `task_contract.py` and `episode_runner` encode different task / success logic (legacy and current).
2. `LANDING` is already used as another internal concept; future goal naming must avoid the collision; `FINAL_LANDING` / `STAGING` are not implemented.
3. The canonical renderer is plain DRR plus catheter; the DSA / roadmap pipeline is incomplete.
4. Framing / field of view of the observation remains unresolved.
5. Closed-loop GR00T inference is not implemented; the trainable / fixed components of GR00T are UNKNOWN until the external configuration is inspected; a Windows-specific GR00T path is hard-coded.
6. `inserted_length_m` documentation is stale in several places; the privileged `action_override` cannot be the final policy interface.
7. `docs/scientific_protocol.md` and `docs/provenance/stage3a_initial_state_audit.md` are stale and need their own documentation checkpoint.

## 8. Limitations of the design

- The tolerances are maxima over a declared synthetic design and are close to `LUMEN_DELTA_MM`; the audit has little power for intermediate violations (see section 6). A tighter design needs a new dated design file before the cloud run.
- Radius coverage is not guaranteed: uncovered samples make their event INDETERMINATE (see section 4).
- The reference variants share the same mask, so a systematic error of the mask against the real lumen (segmentation) is outside the audit.
- Mechanisms outside the two examined ones (contact dynamics, friction, bend / twist stiffness) are not examined; the engine query is examined only through the recorded samples and the synthetic calibration.
