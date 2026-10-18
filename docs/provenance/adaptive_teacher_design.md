# Adaptive (anatomy-aware) teacher and multi-patient pilot — design v2

Date: 2026-10-18. Scope: design and pilot plan ONLY. No generation, no training, no change to canonical code. Supersedes parts of `cvpr_multipatient_dataset_design_v1.md` (see section 1).
Tags: **[measured]** = computed from repository files/data in this session; **[prototype]** = computed with the design-stage extractor `scripts/design_v3/anatomy_profile.py`
(skimage skeleton, NOT the i4h twin; agrees with the twin on s0011 within about 4%, section 3); **[assumed]** = not measured, must be settled by the pilot.

## 0. Summary

- The policy stays vision-grounded: DSA/roadmap image + causal state (`inserted_length_m`, `previous_insert_velocity_mps`) + fixed language. Only the privileged teacher reads anatomy.
- SLOW has exactly two permitted causes, recorded per decision in privileged metadata: **LANDING** (a patient-specific braking envelope before the landing zone) and **SAFETY** (a physics-calibrated wall-budget hazard).
  Everywhere else the teacher commands CRUISE. HOLD only inside the landing zone (plus a short recorded hold after success).
- Nothing is tuned to balance classes. Each braking distance is the **minimum sufficient** distance given a measured perceptual need, and the SAFETY term is **test-gated**: it is dropped if the pilot shows that slowing does not reduce wall penetration.
- Expected action mix of the landing envelope alone on the real access paths of 110 patient-sides: approach phase CRUISE 77% / SLOW 21% of decisions (CRUISE covers 89% of the path length) at the central setting; sensitivity band 87% to 69% CRUISE. Details in section 6.
- Honest limits found while preparing this: (i) the geometric safety proxy I first considered is **not** usable (section 4.4); (ii) s0011 is the most benign anatomy in the pool, so the physics admission yield for other patients is unknown and may be low (section 3);
  (iii) the anatomy-driven part of the landing distance adds only about +-2.5 mm around the perceptual floor, so its value must be demonstrated by an ablation in the pilot (section 7).

## 1. Corrections to the previous design (v1)

1. **Navigation path.** The s0011 pipeline does not run down the aorta. The portal `PORTAL_M` / `CT_OFFSET_M` = (61.96, 201.96, -8.5) mm is centerline endpoint 0, the lowest end of an **iliac artery** [measured: `newton_fluoro_observation.py:44`, `centerline_points_mm.npy`].
   The catheter advances about 186 mm up that iliac to the **aortic bifurcation** (the iliac labels end at z = 162.5 mm, the first degree>=3 cluster is at z = 158.5 mm) [measured: `cvpr_visible_bifurcation_target_v1.md`].
   The v1 gate "aorta z-extent >= 200 mm" tested the wrong segment and is withdrawn.
2. **Feasibility gate.** What limits a patient is the access-path length `S_contact` of an iliac artery (portal endpoint to bifurcation). Landing zone near edge = `S_contact - 4.72 mm`; start depth F = feed to the zone;
   `F_max = S_contact - 81.2 mm` (tip starts 73.5 mm from the portal with 3 active bodies + 3 mm capsule + 4.72 mm; reproduces s0011: 183.6 - 81.2 = 102.4 mm, real feed 102.0 mm) [measured geometry].
3. **Fixed braking distances (7.2 / 9.6 / 12 mm) are withdrawn** in favour of the adaptive rule below.
4. **Physics-admission yield 85-90% was an unsupported assumption.** See section 3.
5. The provisional patient split of v1 was built on the old gate (61 subjects) and must be regenerated from the corrected pool after the pilot.

## 2. Contract and firewall

| | allowed at inference (policy) | teacher only (privileged), stored only in `meta/privileged_eval.json` |
|---|---|---|
| input | roadmap image 320x320 (R=G=B, fixed display constants), `inserted_length_m`, `previous_insert_velocity_mps`, fixed language string | centerline, radius profile, curvature/sagitta, bifurcation descriptors, distance to landing zone, SDF/wall clearance, target position, hazard index, braking distances, SLOW cause tags, patient id, access side, view, noise, contrast, start depth F, reaction lag |
| output | `insert_velocity_mps` in {0.000, 0.006, 0.012} | - |

Rules kept from the existing writer: parquet contains only `action`, `observation.state`, timestamp and index columns (`privileged_in_parquet == []` is a hard check); no privileged value is encoded in the image or video;
normalisation statistics are computed on the training patients only. The teacher's anatomy knowledge reaches the policy only through demonstrated behaviour.

## 3. Evidence base

- **Pool [prototype, all 73 subjects with aorta + both iliac labels]**: 62 extracted by the prototype (11 failed: no bifurcation cluster found, 7 of them were in the v1 gate); access paths with `S_contact >= 150 mm` (F_max >= 69 mm):
  **59 patients, 112 patient-sides, 53 patients with both sides**. F_max over feasible sides: min 69, quartiles 99 / 117 / 132, max 184 mm. The access side (left/right iliac) is an extra anatomical variation axis.
- **Prototype vs the s0011 twin [measured]**: skeleton length 209 vs 215 mm (twin raw), smoothed contact 175.9 vs 183.6 mm (-4%), daughter angle 138.6 vs 131.5 deg, inscribed radius along the real episode correlates 0.84 with the twin radius (ratio 0.97). Path lengths are therefore +-5% until the i4h twin replaces the prototype (pilot step P1).
- **s0011 is the benign end of the pool [prototype]**: 30-mm chord sagitta (max over the path) 1.7 mm vs pool 5/50/95% = 2.6 / 4.7 / 6.9 mm; tortuosity 1.18 vs pool median 1.34; median inscribed radius 3.9 mm vs 4.3 mm.
  The catheter capsule radius is 3.0 mm, so lumen clearance is only 0.35-2.9 mm on s0011 and smaller where the path bends. Most other patients lie outside the regime in which the physics was validated.
- **The physics is fragile [measured]**: of 44 documented tuning attempts on s0011, 29 ended in LUMEN_VIOLATION, 14 in SUCCESS, 1 in PHYSICS_FAILURE (`outputs/cvpr_physical_reachability_v1/attempt_log.json`); success needs the soft tip and 16 solver iterations.
- **Where the wall budget was actually used on s0011 at CRUISE [measured, 90 approach decisions]**: axis violation >= 0.45 mm (30% of the 1.5 mm budget) on 9% of decisions, >= 0.75 mm (50%) on 4%, >= 0.9 mm (60%) on 2%; maximum 1.04 mm at s = 154 mm.
- **Geometry did not predict it [measured]**: corr(violation, 1/clearance) = -0.05, corr(violation, sagitta) = 0.21 on the real episode (weak).
- **Speed dependence is untested**: no existing run varies speed along a path. Whether SLOW reduces wall penetration is an open physical question (test T-SPEED).

## 4. The adaptive teacher

Notation: `s` tip position along the access path (mm); `R_c = 3 mm` capsule radius; `L_seg = 30 mm`; `V_c = 12`, `V_s = 6 mm/s`; `dt = 0.1 s`; landing zone `[z_n, z_f]`, `z_f = S_contact - 2.32`, `z_n = z_f - 2.4` (extremity coordinates, same as the existing `landing_zone.json`).

### 4.1 Structure (replaces `teacher_action`; same causal, privileged, no-future-information contract)

```
cause set:  LANDING envelope  Z_L = [z_n - d_T , z_n)          (always present)
            SAFETY envelopes  Z_S = union_k [a_k - d_pre , b_k + h)  (zero or more, only if the hazard model is accepted by tests)
action(t):  HOLD    if s_lag >= z_n
            SLOW    if s_lag in Z_L or in Z_S
            CRUISE  otherwise                                 s_lag = tip position `lag` decisions ago (reaction lag in {0,1,2})
```
Every SLOW decision stores its cause (`LANDING`, `SAFETY` or both). Temporal regularisation: zones separated by less than 6 mm are merged, no zone is shorter than 3 decisions, one-decision hysteresis on exit.
Success, overshoot, lumen and timeout rules are the existing ones; the timeout generalises to `T_MAX = ceil((F - d_T)/V_c/dt + d_T/V_s/dt) + margin`.

### 4.2 LANDING envelope: patient-specific braking distance

```
d_floor = max( B_kin , N_px * mm_per_px )            B_kin = 2 V_c dt = 2.4 mm (two decisions of lag, existing)   mm_per_px = 0.36 [measured: 2.4 mm window = 6.66 px]
d_T     = d_floor + Delta_max * sigma_T               Delta_max = V_c * tau_ext, tau_ext <= 0.5 s  (extra look-ahead for the most severe landing)
sigma_T = ( n_c + n_f + n_a ) / 3 ,   each term clipped to [0,1] and scaled by FROZEN pool 5-95% ranges
  n_c = (c_hi - c_z)/(c_hi - c_lo)     c_z = r(z_n) - R_c   lumen clearance at the landing   (tight clearance -> more severe)       c: [1.55, 4.59] mm
  n_f = (f - f_lo)/(f_hi - f_lo)       f = r(S_c)/r(S_c - 20 mm)   lumen flare before the Y  (rapid widening -> tip can swing)     f: [0.79, 1.29]
  n_a = (a - a_lo)/(a_hi - a_lo)       a = angle between the two daughter vessels            (wider divergence -> larger error)    a: [123, 163] deg
```
- `N_px` is the number of image pixels of distance-to-go a vision policy needs to resolve the approach. It is **measured, not chosen** (test T-RES): the smallest `N_px` for which the tip-to-zone offset is detectable (matched-filter z >= 4, the project's predeclared criterion) under the noisiest declared imaging condition.
  Principle of minimum sufficient braking distance: take the smallest value that passes, never a larger one.
- `Delta_max` is fixed by the ablation T-LAND: keep `sigma_T` only if it improves a physical outcome (landing error, overshoot rate or violation) over `d_floor` alone; otherwise `Delta_max = 0` and the envelope is patient-specific only through where it sits.
- The pool ranges are frozen in the dataset manifest before generation.

### 4.3 How the braking distance varies with anatomy

| anatomy feature | effect on d_T | physical reading |
|---|---|---|
| tight lumen at the landing (small clearance) | longer | little lateral room to absorb the stop |
| strong lumen flare just before the Y | longer | wall guidance is lost, the tip swings |
| wide daughter angle | longer | a late stop is a bigger lateral error |
| access-path length / start depth | none | only changes how much CRUISE precedes the envelope |

At the central setting (`N_px = 20`, `Delta_max = 6 mm`) `d_T` is 7.3-12.6 mm over all 110 patient-sides (median 10.6 mm). Anatomy therefore moves the distance by about +-2.5 mm (+-4 SLOW decisions); the floor (7.2 mm) dominates.

### 4.4 SAFETY envelope: only where physics justifies cautious advance

A geometric trigger does not work [measured/prototype]: with the natural fit ratio `phi = e30 / (2 (r - R_c))` (rod chord sagitta over available clearance), `phi >= 1` marks a **median 29% of the last 100 mm of path** (IQR 17-47%; `phi >= 0.5`: median 63%) —
SLOW would dominate and CRUISE would not — while on s0011 the real CRUISE run stayed within the wall budget everywhere. The hazard must be a calibrated physical prediction:

```
H(s)  = predicted fraction of the 1.5 mm lumen budget consumed when the tip passes s at CRUISE     (monotone model of c(s), e30(s), local bend; fitted on CRUISE-only pilot runs)
SAFETY zones: connected sets {s : H(s) >= H_slow}, H_slow = 0.5 (absolute), extended ahead by d_pre = d_floor (the same perceptual distance) and 1 decision behind
```
Acceptance gates: **T-HAZ** — leave-one-patient-out Spearman rho >= 0.5 between predicted and measured violation; **T-SPEED** — slowing the same stretch must reduce measured violation by a pre-declared margin (>= 20%).
If either fails, SAFETY is removed and SLOW comes from LANDING only. The threshold is absolute, not a quantile, so a benign patient gets no SAFETY SLOW and a hard one gets more; a guard (section 8) flags any episode in which SAFETY covers more than 10% of the approach path.

### 4.5 Why this does not over-produce SLOW

1. There is no class-balance feedback anywhere. Class mix is an output, reported, never a control.
2. LANDING uses the minimum sufficient distance from a measured requirement; SAFETY needs an absolute physical trigger that must also pass a speed-effect test.
3. Hard budgets (validation, not control): in the approach phase CRUISE >= 70% of decisions and >= 80% of path length. The central setting gives 77% / 89%; the aggressive end of the band (`N_px = 28`, `Delta_max = 9 mm`) gives 69% / 83% and would be rejected.
4. The recorded hold after success is limited to 20 decisions (8 HOLD first-action samples per episode).

## 5. Target / landing zone for any patient

- Access path: skeleton endpoint of the chosen iliac (side L or R) to the **first true bifurcation cluster**, defined as a node whose removal of a 9 mm geodesic ball leaves >= 3 branches of >= 12 mm (handles double junctions such as s0011's two junctions 12-15 mm apart); the path stops before the cluster (`select_path` convention), smoothed with 3 mm Gaussian.
- `S_contact` = smoothed path end - 2.83 mm (measured offset on s0011); zone `[S_contact - 4.72, S_contact - 2.32]`, width 2.4 mm = existing `2 eps`; clearance 2.32 mm is a per-decision advance + stop glide (re-measured on the first admitted patients).
- Portal placement per episode: `s_portal = (z_n - R_c) - 73.5 - F`, which must be >= 0, so `F <= F_max`. Portal position and tangent are taken from the smoothed path at `s_portal` (this replaces the constants `CT_OFFSET_M`, `TANGENT`).
- The target is never a policy input; the policy sees the bifurcation only as anatomy in the roadmap.

## 6. Expected action distribution [prototype + kinematic planning model]

The planning model reproduces the real tiny episode exactly (105 decisions, 83 / 4 / 18). Start depth F ~ U[60, min(100, F_max)] mm, reaction lag in {0,1,2}, 20 recorded hold decisions. LANDING only (SAFETY adds zero to a few percent, section 3 and 4.4).

| setting | d_T median (range) | approach CRUISE / SLOW (decisions) | CRUISE share of path | sampled first actions HOLD / SLOW / CRUISE | all decisions HOLD / SLOW / CRUISE |
|---|---|---|---|---|---|
| lenient: N_px 12, Delta_max 3 mm | 6.0 (4.4-7.0) | 87% / 12% | 95% | 10% / 11% / 79% | 25% / 9% / 66% |
| **central: N_px 20, Delta_max 6 mm** | **10.6 (7.3-12.6)** | **77% / 21%** | **89%** | **10% / 19% / 71%** | **24% / 16% / 60%** |
| aggressive: N_px 28, Delta_max 9 mm | 15.2 (10.2-18.1) | 69% / 30% | 83% | 9% / 27% / 63% | 23% / 23% / 53% |
| reference: real tiny episode | 2.4 | 95% / 5% | - | 3% / 4% / 92% | 17% / 4% / 79% |

Across patients at the central setting the first-action SLOW share ranges 13-24% and CRUISE 66-77%; the pilot patients (both sides) give d_T 9.8-11.6 mm.
SAFETY contribution: on s0011 the real physics suggests at most 2-4% of approach decisions near half the wall budget; for tortuous patients it is unknown and is measured in P2/P3.

## 7. Multi-patient pilot

**Patients (train split only, both access sides feasible, every descriptor inside the pool 5-95% range, farthest-point selection from s0011) [prototype]:**

| patient | role | r_med mm | sagitta max mm | tortuosity | flare | daughter angle | S_contact mm | F_max L / R | study / pathology / sex / scanner |
|---|---|---|---|---|---|---|---|---|---|
| s0011 | benign reference, twin exists | 3.9 | 1.7 | 1.18 | 1.12 | 138 | 173 | 95 / 90 | neck-thorax-abdomen-pelvis / tumor / m / Emotion 16 |
| s1391 | tortuous, wide, long | 5.3 | 6.0 | 1.51 | 1.06 | 153 | 215 | 132 / 135 | CT angiography / bleeding / m / Somatom Force |
| s0727 | shortest path | 4.2 | 4.0 | 1.28 | 0.88 | 149 | 162 | 76 / 86 | polytrauma / trauma / f / Somatom Definition AS+ |
| s0058 | typical mid-range | 4.1 | 4.2 | 1.26 | 1.10 | 152 | 193 | 112 / 111 | thorax-abdomen-pelvis / tumor / f / Somatom Definition Flash |

(Pool medians: r_med 4.3, sagitta 4.8, tortuosity 1.34, flare 1.02, angle 147, S_contact 197.) The first farthest-point selection picked extraction outliers (flare 0.28) and three CT-angiography/vascular patients; the quality filter was added for that reason.

**Steps (each ends with a go/no-go):**
- **P0 parameterise the pipeline per patient** (code, section 9) — no data generated.
- **P1 twin cross-check**: build the i4h twin (`preprocess_ct`, `segment_vessels`) for the 4 patients; require prototype vs twin `S_contact` within 5 mm, same bifurcation cluster, radius correlation >= 0.8; otherwise replace the prototype by the twin everywhere. About 20 s per patient.
- **P2 CRUISE-only mapping and physics admission**: 4 patients x 2 sides x 2 start depths = 16 runs (44 s each). Output: admission (SUCCESS, violation <= 1.5 mm, finite, no overshoot) and the (feature, violation) pairs that train H(s). First measurement of the real admission yield.
- **P3 speed-sensitivity T-SPEED**: SLOW-only and CRUISE-with-SLOW-pocket runs on the same paths (8 runs): does slowing reduce wall penetration?
- **P4 resolvability T-RES**: render frames of the approach at noise levels {3000, 5000, 8000} photons/pixel and find the minimum `N_px`.
- **P5 landing ablation T-LAND (physics only)**: 4 patients x 2 sides x 3 start depths x 3 teachers (2.4 mm legacy, `d_floor` only, `d_floor + Delta_max sigma_T`) = 72 runs.
- **P6 teacher episodes with images**: 4 patients x 12 = **48 episodes**: 2 sides x 3 start depths (F at 0 / 50 / 100% of [60, F_max]) x 2 reaction lags {0, 2}; per patient the three view families (AP, LAO 45, RAO 45, +-10 deg) and two noise levels {3000, 8000}, contrast 800-1200 HU-equivalent, in a balanced (Latin-square) assignment.
- **P7 validation suite and decision report** (section 8).

**Expected pilot dataset [planning model]**: 48 episodes, about 4 670 decisions, 3 946 first-action samples; at the central setting HOLD / SLOW / CRUISE = 384 / 794 / 2 768 (10% / 20% / 70%); the SLOW count is 410 at the lenient and 1 162 at the aggressive end of the band. About 860 transition-window samples, about 1 440 chunk samples containing a class change, about 0.16 GB.
Compute: about 144 physics runs (about 2-3 h) and 48 renders-and-builds (about 1-1.5 h); too unreliable for the local WSL2 stack, so run the pilot on the native-Linux GPU machine.

## 8. Validation checks that must pass before any multi-patient generation

| gate | check | pass criterion |
|---|---|---|
| G1 extraction | prototype vs i4h twin on the 4 pilot patients | `S_contact` within 5 mm, same cluster, radius corr >= 0.8 |
| G2 physics | admission rule P2 | SUCCESS, violation <= 1.5 mm; the measured admission yield decides the scale (see below) |
| G3 hazard model | T-HAZ | leave-one-patient-out Spearman >= 0.5, else no SAFETY term |
| G4 speed premise | T-SPEED | slowing lowers measured violation by >= 20%, else no SAFETY term |
| G5 resolvability | T-RES | minimal `N_px` found; with it approach CRUISE >= 70% of decisions and >= 80% of path |
| G6 landing term | T-LAND | `sigma_T` improves a physical outcome, else `Delta_max = 0` |
| G7 distribution | realised vs expected | CRUISE the largest class in every patient; realised class shares within +-10% of section 6 |
| G8 cause tags | every SLOW decision tagged; no SLOW with no cause | 100% |
| G9 firewall | parquet columns, image cleanliness (R=G=B, no text/drawing), language constant | no privileged value outside `meta/privileged_eval.json` |
| G10 causality | previous action, monotone feed, image built from the state accepted before the action | exact |
| G11 no state shortcut | state-only classifier on `[inserted_length_m, previous_velocity]` | SLOW F1 not above always-CRUISE by more than 0.05 |
| G12 image | DSA protocol criteria per (patient, view), tip and zone in view >= 95% of decisions | pass; failing views dropped for that patient |
| G13 reproducibility | re-run of 3 episodes | bit-identical labels |

**Scale decision after the pilot**: the number of admitted patients = measured yield x 59. If fewer than about 36 admitted (24 train / 6 val / 6 test), generate the minimum tier or extend to the full TotalSegmentator release; do not lower the gates.

## 9. What must be parameterised beyond s0011 [measured from the code]

| item | where it is s0011-specific today | needed |
|---|---|---|
| vessel directory / mesh cache | `newton_stage4a_vessel_contact.py:55-56` `CT_DIR`, `MESH_CACHE` | per-patient `ct_cache/<id>` and mesh; the mesh is built from `vessel_mask.npy` |
| SDF sign and patient assertion | `:70` `SIGN_SCALE` "s0011 Stage 4.8 auto-selected", `:477` `assert patient_id == "s0011"`, `:569` | re-run the sign audit per patient; drop the assertion |
| portal, axis, CT offset | `newton_fluoro_observation.py:44-48` `CT_OFFSET_M`, `PORTAL_M`, `TANGENT` | a per-episode `PatientFrame(portal_ct_mm, tangent)` from the access path at `s_portal` |
| episode class | `newton_patient_episode.py` calls `build_or_load_vessel_mesh()` with no argument | take patient id and frame |
| path, landing zone, timeout | `_canonical_episode_physics.py:38-45` reads `cvpr_visible_bifurcation_v1/{landing_zone.json,path_ext.npz}` | an `AccessPath` object (nodes, radii, profile) and a computed zone and timeout |
| teacher | `teacher.py` single threshold rule, `CRUISE_MARGIN` constant | the two-envelope rule with cause tags (section 4) |
| rendering | `_render_demo_images.py:29` `CT_DIR`, accepted iso from `params_hu1000.json` | per-patient CT directory, iso from the zone, view/noise/contrast per episode, CT-angiography lumen normalisation |
| writer | `_build_demo_episode.py:53` `patient_id="s0011"`, one episode | patient id and multi-episode index, privileged cause tags |

Unchanged: the 7-body rod, canonical engine settings (16 solver iterations, soft tip 15), action API, timing (60 Hz physics, 6 frames per decision), success and lumen thresholds.

## 10. Decisions needed

1. Approve the two-envelope teacher (LANDING always, SAFETY only if G3 and G4 pass) and the minimum-sufficient-distance principle.
2. Approve building P0 (code changes in the repository) and running the pilot on the native-Linux machine.
3. Confirm that the physics admission yield, not an assumption, sets the final patient count.

Risks: unknown admission yield for tortuous patients; the speed premise may be false (then SLOW is landing-only); the anatomy term of the landing distance may not earn its place; CT-angiography and vascular pathology are confounded (13 of the 14 vascular patients are CT-angiography); the prototype extractor is not the production twin.

## 11. Reproduction

`scripts/design_v3/`: `anatomy_profile.py` (extractor), `_scan_access_paths.py`, `_pool_stats.py`, `_rule_diagnostics.py`, `_select_pilot.py`, `_expected_distribution.py`, `_pilot_counts.py`, `_s0011_physics_vs_geometry.py`.
Outputs in `outputs/cvpr_multipatient_design_v1/`: `access_path_scan.json`, `access_path_rows*.json`, `access_path_profiles.npz`, `landing_descriptor_ranges.json`, `pilot_patients.json`, `expected_distribution_band.json`, `planning_simulation.json`.
