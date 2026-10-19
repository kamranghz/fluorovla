# Stage 3A: initial-state / registration audit of the physics mapping

Status: **tooling complete and tested; native-Linux audit run pending.** Current verdict: **INCONCLUSIVE** (provisional; it can only change after the native-Linux
run described in `docs/initial_state_audit_runbook.md`). No physics constant, Teacher rule or task definition was changed. Nothing here is an admission conclusion.

## 1. What the audit must decide

The native-Linux physics mapping (72 episodes, 13 patients) gave 68 LUMEN_VIOLATION and 4 SUCCESS; 47 of the 68 violations end at decision 0 or 1. Question: is that a
systematic initialization / registration / coordinate-frame problem, or a consequence of the physical construction? Evidence first, interpretation second.

## 2. Exact initialization path (traced in the code, not assumed)

1. `anatomy/episode_config.build_episode_config`: `s_p = z_tip - 73.5 - F` with `z_tip = S_contact - 7.72`; the portal is the point of the RAW centerline polyline whose projection onto
   the smoothed path has arclength `s_p` (clipped at the path start); the tangent is the 5-mm chord of the smoothed path at that arclength; the config stores `portal_ct_mm` and `tangent`.
2. `physics/patient_config.PatientConfig.from_json`: `ct_offset_m = portal_ct_mm / 1000` (the solver frame is the CT frame shifted to the portal; portal_m = 0); the tangent is normalized on read
   and again on use.
3. `physics/lumen_sdf.load_lumen_mesh`: marching cubes on `vessel_mask[::2, ::2, ::2]` (3 mm voxel), vertices in CT metres minus `ct_offset` (float32); SDF `phi = sign_scale * sign * distance`, phi < 0 inside.
4. `physics/patient_episode.PatientEpisode.__init__`: `configure_world_pose(portal=0, axis=tangent, root_at_portal=True)` -> `build_scene` (a STRAIGHT rod along the tangent, root body fixed at the portal,
   2 active bodies) -> settle phase 1 (30 frames, portal fixed, lumen contacts) -> third body enabled along its parent -> settle phase 2 (30 frames) -> soft-tip compliance -> **ready** (start of decision 0).
5. Contacts (`generate_lumen_contacts`): for every non-root active body, 5 axis samples; a sample with `phi > -R` (R = 3 mm) generates a contact against the wall point `p + n * (-phi)`.
6. Violation test (`measure_outside`, every substep of every decision): `excess = max(0, max phi over the axis samples of the non-root bodies)`; LUMEN_VIOLATION when the decision's maximum exceeds 1.5 mm.

Consequences that matter for the audit: (a) the rod is deployed **straight** over about 75 mm regardless of the vessel; (b) the violation test uses the **axis** (centre line), so a capsule may overlap the wall
while the test passes; (c) the root body (at the portal) is excluded from contacts and from the test.

Files read: `experiments/protocols/experimental_protocol_v1.md`, `docs/scientific_protocol.md`, `docs/cloud_workflow.md`, `docs/provenance/initial_state_preliminary_audit.md`, `docs/architecture.md`,
`anatomy/episode_config.py`, `physics/{patient_config,patient_episode,engine,lumen_sdf,episode_runner}.py`.

Existing instrumentation reused: `probe.json` and the per-decision rows of the native-Linux run; the audit adds a read-only observer (`PatientEpisode(cfg, observer=...)`, called at the checkpoints
build_scene / after_settle1 / after_body_enable / after_settle2 / ready / every decision substep). Checkpoint mapping to the requested A-D: A pre-sim = `initial_state.analytic_initialization` + the config; B before settle =
`build_scene`; C after settle = `after_settle2`/`ready` (and the settle traces); D first decision = `violation_track` (per substep, worst body and axis sample).

## 3. Evidence that already exists (native Linux, the 72 original episodes; CPU analysis `tools/analyze_start_geometry.py`)

- 47 of 68 violations end within the first decision. At decision 0 the tip is a median 7.9 mm from the centerline in violating episodes versus 3.2 mm in successes; in 32 episodes it is more than 8 mm away (up to 32 mm).
  The state is therefore already invalid **before any meaningful insertion**.
- Geometry of the start condition versus outcome (Spearman rank correlation with the tip distance at decision 0, 72 episodes): arc length / chord of the path over the 73.5-mm rod **0.90**, total turning of the path over the rod
  **0.87**, tangent-to-chord angle at 73.5 mm 0.59, start depth F -0.21. All four SUCCESS episodes have arc/chord <= 1.018 and turning <= 25 deg; failures reach arc/chord 2.25 and 149 deg of turning.
- The correlation is not a sufficient explanation: 10 of the 14 episodes with an almost straight start (arc/chord <= 1.025) also fail, mostly after a few decisions (1, 3, 4, 4, 5, 12, 14, 14, 16, 37), which points to a second mechanism (see 5).
- Even the SUCCESS episodes are marginal: `s0011_right_F0` reaches a maximum axis violation of 1.15 mm against the 1.5 mm limit.

## 4. Local development run (WSL2; development evidence only, the audit must be repeated on native Linux)

The instrumented re-runs of all four cases are **bit-identical to the original native-Linux episodes** (the observer is read-only; the result also reproduces across OS/driver). Findings:

| case | original | tangent vs chord angle 5/15/30/73.5 deg | straight-rod end deviation | post-settle max body-to-path | post-settle min capsule clearance | post-settle max axis violation | terminal decision | max settle speed |
|---|---|---|---|---|---|---|---|---|
| s0011_right_F0 | SUCCESS | 0.0 / 0.4 / 2.2 / 1.5 | 1.9 mm | 0.6 mm | -0.67 mm | 0.0 mm | 53 | 0.020 m/s |
| s0963_left_F1 | SUCCESS | 0.0 / 2.0 / 6.5 / 11.6 | 14.5 mm | 1.6 mm | -3.02 mm | 0.28 mm | 63 | 0.061 m/s |
| s0963_left_F2 | LUMEN_VIOLATION | 0.0 / 2.4 / 4.4 / 9.8 | 12.3 mm | 1.5 mm | -1.90 mm | 0.0 mm | 36 | 0.015 m/s |
| s0344_left_F1 | LUMEN_VIOLATION | 0.0 / 13.0 / 35.7 / 69.7 | **53.9 mm** | **18.2 mm** | **-15.3 mm** | **27.0 mm** | 0 | 0.086 m/s |

- **Coordinate frames are consistent** in the four cases: CT -> solver -> CT and CT -> voxel -> CT round trips are exact (<= 3e-14 mm); the portal lies on the raw polyline (1e-14 mm) and inside the mask; 100 % of the
  centerline nodes lie inside the mask and the EDT of the mask reproduces the twin radius to 1e-7 mm with the index convention used (a half-voxel shift would give 0.72 mm); the mesh vertices sit on the 0.5 iso-surface of the
  down-sampled mask (1e-6); the engine's initial rod equals the analytic straight rod (lateral offset 5e-7 mm).
- **SDF sign agrees with the mask** (measured for `s0011_right_F0` and `s0344_left_F1`): 0.08-0.13 % of the points deeper than 2 mm inside/outside the mask have the wrong sign. Exception found: at the **exact** portal point (the solver origin) the SDF can return the wrong
  sign (`s0344_left_F1`: +3.0 mm at the origin while +/-1 mm neighbours are -2..-3 mm), a degenerate query on the voxel-aligned mesh (several faces at exactly equal distance). It does not affect the contacts (the root body is excluded
  and moving bodies sit at generic positions); the audit therefore tests the neighbourhood median and records the anomaly separately.
- **The collision lumen is about 1 mm narrower than the twin mask** (median `phi + depth` +0.94 .. +1.04 mm), a consequence of the 3-mm marching-cubes mesh. For `s0963_left_F2` 97.5 % of the first 120 mm of the centerline
  are less than the catheter radius (3 mm) from the mesh wall, so the catheter cannot sit inside the contact criterion there; the violation occurs later (decision 36) as the tip advances. This is a second, anatomy/mesh-resolution
  mechanism.
- **s0344_left_F1** is invalid immediately after settle: the path turns 70 deg within the 73.5-mm rod length (arc/chord 1.24); the straight rod's end is 54 mm from the lumen; body 1 settles to 2.5 mm from the path but body 2
  remains 18 mm away with an axis violation of 27 mm, and the first decision terminates the episode. None of the pre-declared initialization-only variants (portal at the configured / smoothed / path-start position x 5, 15, 30, 73.5-mm chord
  tangents) brings the post-settle violation below 1.5 mm (best 6.4-8.3 mm); several tangent variants turn the positive controls into LUMEN_VIOLATION within 2-9 decisions, i.e. outcomes are sensitive to the convention.

## 5. What is demonstrated and what is hypothesis

Demonstrated (pending native-Linux confirmation of the instrumented values): no coordinate-frame or portal-registration inconsistency in the audited cases; for the immediate failures the rod is already outside the lumen after the settle phase;
the start-path curvature over the rod length strongly tracks that failure (rho 0.90 over 72 episodes).

Hypotheses (not proven): the straight pre-deployment of about 75 mm is the cause of the immediate failures on curved starts (a construction limitation, not a coordinate bug); the 3-mm collision mesh (about 1 mm narrower lumen) together with
vessel radii of 2-3.5 mm explains the later failures. Neither is an implementation error in the sense of the audit protocol: no concrete defect was located, no initialization-only counterfactual resolves the artifact.

## 6. Verdict logic and current verdict

Rules are in `src/fluorovla/evaluation/initial_state_verdict.py` (tested): `PASS_INITIALIZATION_VALID` needs consistent frames and no early failure already invalid after settle; `BUG_CONFIRMED` needs a reproducible mechanism and a pre-declared
initialization-only variant that makes every early failure valid after settle with all positive controls intact; otherwise `INCONCLUSIVE`. A success rate never grants a verdict.

With the local development evidence the rules give **INCONCLUSIVE** (`s0344_left_F1` is invalid after settle and no variant resolves it). The native-Linux audit will either confirm this or change it.

## 7. Next subsystem to audit if the native run confirms this

1. Start-condition construction on curved segments (a curvature-aware or shorter deployment would be a protocol erratum, because it changes the start state; it is not authorized by this audit).
2. Collision-mesh discretization (3-mm stride-2 decimation; catheter radius versus lumen at that resolution) and the axis-versus-capsule semantics of the violation test.
3. Cases with a straight start that still fail at decision 1 (for example `s0011_left_F1`): add them to a second audit round.

Phase E (bug confirmed) is not triggered: no patch is proposed and the 72-episode run is not repeated.
