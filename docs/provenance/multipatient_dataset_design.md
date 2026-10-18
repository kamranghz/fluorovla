# Multi-patient / multi-episode dataset design (v1) — design only, nothing generated, no training started

> **SUPERSEDED IN PART (2026-10-18) by `cvpr_adaptive_teacher_multipatient_pilot_v2.md`.** Corrections: (1) the catheter navigates an **iliac access path to the aortic bifurcation**, not the aorta; the patient gate and the "aorta >= 200 mm" criterion in section 3 are wrong and the pool is 59 patients / 112 access paths, not 61; (2) the fixed braking distances B in {7.2, 9.6, 12} mm and teacher v2 in section 6 are withdrawn in favour of the adaptive teacher; (3) the 85-90% physics-admission yield in section 3 was an unsupported assumption (s0011 is the most benign anatomy in the pool); (4) the split in section 4 must be regenerated from the corrected pool. Sections 2 (firewall), 8 (size arithmetic per episode), 9 (generation stages) and 10 (validation checks) remain valid in structure.


Date: 2026-10-18. Status: PROPOSAL awaiting approval of the three decisions in section 12.
All numbers marked **[measured]** come from files in this repository; **[planning-sim]** from `scripts/_plan_dataset_counts.py` (a kinematic model that
reproduces the real tiny-overfit episode exactly: 105 decisions, 83 CRUISE / 4 SLOW / 18 HOLD, first-action 83 / 4 / 3);
**[assumed]** means not yet measured and must be confirmed by the pilot (section 9).

## 1. Why more optimisation steps, or more anatomies alone, cannot fix SLOW

- Checkpoint-600 [measured]: HOLD recall 33% -> 100%, CRUISE 100%, transition accuracy 58% -> 82%, but SLOW recall 67% -> 33% (7 of 8 errors are SLOW predicted HOLD).
- The cause is structural in the data. The teacher brakes over `CRUISE_MARGIN = 2 * V_MAX * dt = 2.4 mm` [measured, `teacher.py`] and a SLOW decision advances 0.006 m/s * 0.1 s = 0.6 mm,
  so **every episode contains exactly B/0.6 = 4 SLOW decisions**, independent of patient, view, noise or start depth. Only 4 of 90 supervised first actions are SLOW (4.4%).
- Planning-sim scenario B (current teacher, feed varied 60-100 mm, 24 hold decisions): SLOW stays at 4.3% of first actions. More patients with the current teacher give 4 SLOW per episode forever.
- The cue is also nearly sub-resolution: the 2.4 mm window is 6.7 px at the roadmap scale [measured, `landing_zone.json`: 6.66 px]. A 9.6 mm braking distance is about 27 px.
  (Hypothesis, to be tested by the validation in section 10: a SLOW plateau that spans a visible distance is learnable; one that spans 7 px is not.)
- Therefore the lever is the **teacher's braking distance and reaction timing** (documented teacher parameters), plus anatomy/initial-condition variation that makes the image necessary (below).
  The action API (HOLD 0.000 / SLOW 0.006 / CRUISE 0.012 m/s) and the policy contract do not change.

## 2. Contract and privileged-information firewall (preserved)

- Policy input: fluoroscopic roadmap image (320x320, R=G=B) + state `[inserted_length_m, previous_insert_velocity_mps]` + fixed language string -> action `insert_velocity_mps`.
- Image formation is the predeclared DSA/roadmap protocol (`docs/cvpr_dsa_roadmap_protocol_v1.md`): D = R + S_k, fixed display constants D0 = 0.05, D_ref = 4.8, no per-image normalisation, nothing drawn.
- Everything new (patient id, view, contrast level, noise level, feed F, braking distance B, reaction lag, split, landing zone) is **episode metadata written only to `meta/privileged_eval.json`**,
  never to the parquet, video, state or language. The existing check `privileged_in_parquet == []` is enforced for every episode.
- Patient-level normalisation statistics (state q01/q99) are computed from the **train split only**.
- Why vision becomes necessary: for s0011 alone `inserted_length_m` solves the task (stop at ~103.6 mm of feed) [measured, `cvpr_visible_bifurcation_target_v1.md`].
  Randomising the start depth gives a required feed F in [60, 100] mm, so the encoder value at the stop is no longer constant; across patients the portal-to-bifurcation length also differs.
  Validation check V5 (section 10) measures this directly.

## 3. Patient pool

| step | patients [measured, `_scan_patient_pool.py` on all 102 TotalSegmentator-small subjects] |
|---|---|
| subjects | 102 |
| aorta label present | 99 |
| aorta + both iliac arteries present | 73 |
| **anatomical gate**: both iliacs >= 2000 voxels and >= 120 mm z-extent, aorta >= 200 mm z-extent | **61** (includes s0011) |

Composition of the 61: 16 CT-angiography (native lumen already contrast-enhanced), 14 with vascular pathology (13 of them overlap with CT-angiography), 41 male / 19 female / 1 unknown,
8 scanner models, age 30-91 (median 69). All volumes are 1.5 mm isotropic.
The pool cannot exceed 61 within this dataset. If more patients are needed the next step is the full TotalSegmentator release (not a design change).

Admission gates (run per patient before it receives episodes; failures are dropped inside their split, never moved):
- **G1 anatomy**: the aortic-bifurcation Y (the only degree>=3 cluster) is found; portal-to-bifurcation path length >= F_max + 20 mm; a single connected centerline.
- **G2 physics**: the canonical `PatientEpisode` runs one teacher-v2 pilot episode to SUCCESS with max axis violation <= 1.5 mm, finite state, no overshoot, root displacement <= 1e-6 m (existing thresholds).
  The landing-zone clearance constant (2.32 mm, measured on s0011) is re-measured on the first 3 admitted patients.
- **G3 visibility**: the predeclared criteria of the DSA protocol (TARGET_VISIBLE z >= 4, CATHETER_VISIBLE) are evaluated per (patient, view family). Views that fail are removed for that patient.
- **CT-angiography handling**: the MASK volume replaces the native lumen by 40 HU blood so that the synthetic contrast (+1000 HU-equivalent) is identical for every patient; the additivity check of the protocol must pass.
- Expected yield [assumed]: 85-90% of 61 -> 52-55 patients. This is the largest uncertainty (the physics, mesh, portal and landing-zone code are s0011-specific today, section 9).

## 4. Patient-level split (provisional, fixed before admission so failures cannot be steered by the test set)

`scripts/_make_split_plan.py` (salt `medphysai-multipatient-v1-2026-10-18`, stratified by CT-angiography x vascular pathology, ordered by sha256; s0011 forced into train) [measured]:

| split | patients (pool) | sex m/f/unk | CT-angiography | vascular | scanners | age median (range) |
|---|---|---|---|---|---|---|
| train | 43 | 29 / 13 / 1 | 12 | 10 | 8 | 70 (43-91) |
| val | 9 | 6 / 3 / 0 | 2 | 2 | 5 | 67 (30-75) |
| test | 9 | 6 / 3 / 0 | 2 | 2 | 5 | 70 (57-78) |

Files: `outputs/cvpr_multipatient_design_v1/split_plan_provisional.{json,csv}`.
Rules: no patient in two splits; test patients are not used for any choice of teacher parameters, views, noise levels, normalisation or checkpoint; the split salt is frozen; the TotalSegmentator "train/val" column is ignored.
Known limitation: the dataset gives no patient identity beyond age/sex/institute, so two scans of one person cannot be excluded; the check is centerline-length/age/sex coincidence screening.

## 5. Variation axes (per episode; one view per episode — a C-arm is not moved during navigation)

| axis | train range | justification |
|---|---|---|
| anatomy | the admitted patient | natural variation of path length, radius, bifurcation angle, tortuosity |
| initial condition | required feed F ~ stratified U[60, 100] mm (portal depth along the centerline) | decorrelates the feed encoder from the stop; 60-100 mm keeps both tip start and landing zone in the field of view and inside the validated feed regime (<= 103.6 mm) |
| DSA view | rotation about y in {-45, 0, +45} deg (project AP / LAO / RAO families) + U(-10, +10) deg; isocenter = midpoint(tip start, landing centre) + U(-10, +10) mm | clinical view variation; framing keeps tip and zone in view (fixing the earlier "tip leaves the image" failure) |
| contrast | lumen enhancement U[800, 1200] HU-equivalent (nominal 1000, the predeclared level) | injection variability; 2000 HU stays sensitivity-only |
| image noise | N0 in {3000, 5000, 8000} photons/pixel; MASK and CONTRAST averaged over 8 frames, LIVE single frame (protocol's supplementary noise model) | acquisition dose variation |
| transition timing | braking distance B in {7.2, 9.6, 12.0} mm | operators begin slowing at different distances |
| teacher action timing | reaction lag in {0, 1, 2} decisions (teacher sees the privileged state that many decisions late) | reaction-time variability; max extra advance 2*0.6 + 0.55 glide = 1.75 mm < 2.4 mm window, so no step-over |
| held-out shift probes (test only, never trained) | B = 14.4 mm + lag 2; view jitter +-20 deg; N0 = 2000 and contrast 600 | measures sensitivity to teacher style, view and image shift |

Cells are filled with a balanced (Latin-hypercube) design per patient, not i.i.d. sampling, so every patient covers all view families, all three braking distances and all four feed quartiles.
Physics itself is deterministic; diversity comes from anatomy, F, B/lag and imaging, not from random physics.

## 6. Teacher v2 (only change to the teacher; privileged, causal, no future information)

```
remaining = zone_near - tip_extremity(t - lag)        # privileged, lagged by `lag` decisions
CRUISE if remaining > B ; SLOW if remaining > 0 ; HOLD otherwise
success = existing landing-window + dwell rule; then 24 further HOLD decisions are recorded in the same simulation
```
Current teacher = this rule with B = 2.4 mm, lag 0, 15 hold decisions. B in {7.2, 9.6, 12.0} gives 12 / 16 / 20 SLOW decisions per episode (B / 0.6 mm).
Why 24 hold decisions: the loader samples first actions only for t <= n-16, so HOLD appears as a supervised first action only through post-success decisions; 24 gives 12 HOLD samples per episode (current: 3).
Optional phase 2 (NOT in this dataset): clearance-aware slowing at tight bends/narrowings. It would add mid-path SLOW but changes the task semantics and is unvalidated.

## 7. Action distribution and transition coverage [planning-sim, 20 000 simulated episodes; realised values are checked in the pilot, tolerance +-10%]

Per episode (first-action samples, t <= n-16):

| | decisions n | samples | HOLD | SLOW | CRUISE | transition-window samples (+-4 of a class change) | chunk-16 samples with a class change |
|---|---|---|---|---|---|---|---|
| real tiny episode | 105 | 90 | 3 | 4 | 83 | 11 | 19 |
| current teacher, varied F, 24 holds | 95.8 | 80.8 | 12.0 | 3.5 | 65.3 | 12.5 | 18.5 |
| **teacher v2 (train mix)** | **101.7 (83-120)** | **86.7** | **12.0** | **14.5** | **60.2** | **18.0** | **28.3** |
| teacher v2 shift probe (B 14.4, lag 2) | 105.4 | 90.4 | 12.0 | 21.5 | 56.9 | 18.0 | 30.0 |

Fractions of first-action samples, train mix: HOLD 13.8%, **SLOW 16.7%**, CRUISE 69.4% (tiny: 3.3 / 4.4 / 92.2). Each episode has exactly two class changes (CRUISE->SLOW, SLOW->HOLD) with onset positions varying by B and lag.
Genuine, not duplicated: each SLOW example comes from a different patient, feed, view, contrast, noise realisation and braking/timing combination.

## 8. Dataset size and disk

Per episode [measured]: 3.6 MB for the training format (105 decisions: mp4 + parquet + meta, about 34 KB per decision); the real pipeline also writes 43 MB of raw float32 renders (109 files x 0.41 MB) and 3.4 MB of PNGs.

| tier | patients (train/val/test) | episodes | decisions | train first-action samples | train HOLD / SLOW / CRUISE | disk (training format) |
|---|---|---|---|---|---|---|
| pilot | 4 | 24 | 2 440 | 2 080 | 288 / 348 / 1 445 | 0.08 GB |
| **minimum** | 36 (24 / 6 / 6) | 504 (384 train, 48 val, 48 test-ID, 24 shift probes) | 51 278 | 33 286 | 4 608 / 5 564 / 23 113 | 1.8 GB |
| **recommended** | 54 (36 / 9 / 9) | 1 134 (864 train, 108 val, 108 test-ID, 54 shift probes) | 115 375 | 74 893 | 10 368 / 12 519 / 52 004 | 4.0 GB |

Recommended train set: 15 552 transition-window samples and 24 428 chunk samples containing a class change, from 1 728 class-change events; 3 129x more SLOW samples than the 4 in the tiny run.
Working storage: CT cache about 0.36 GB per patient (19-22 GB for 54-61 patients, regenerable). Raw renders 43 MB/episode (49 GB for 1 134 episodes): **do not keep**; keep per-patient MASK/CONTRAST arrays only and rebuild live frames from stored physics polylines if needed.

## 9. Generation plan

Stages (each ends with a go/no-go before the next):
1. **Parameterise the pipeline per patient** [engineering, not done]. `PatientEpisode` currently hard-wires one patient: `build_or_load_vessel_mesh`, `PORTAL_M`, `TANGENT`, `CT_OFFSET_M`, `landing_zone.json`, `path_ext.npz`, patient id in `_build_demo_episode.py`.
   Needed: per-patient mesh/SDF, portal placement at depth F along the centerline, landing zone from the bifurcation cluster, DRR framing from the zone, multi-episode writer with `episode_index`, teacher v2.
2. **Pilot**: s0011 + 3 train-split patients spanning path length and CT-angiography, 6 episodes each. Pass if: all G2/G3 gates, realised class counts within +-10% of section 7, V1-V9 below pass, state-only leak test (V5) passes.
3. **Admission sweep** over the 61-patient pool (one pilot episode per patient + view visibility), freeze the admitted list and the final split.
4. **Bulk generation** by patient (embarrassingly parallel).
5. **Validation suite** (section 10) on the full set, then freeze (hash manifest).

Compute: physics 44 s for the 105 decisions of the real episode [measured; this timer starts AFTER the scene is built, so scene construction is not included]; scene construction (2 x 30 settle frames; Warp compile about 70 s on a cold cache, about 1.4 s cached) about 15 s per episode [assumed]; about 105 DiffDRR renders at about 0.1 s [from the 48-render, 4.4 s run recorded in the project notes]; composition + encode about 15 s [assumed] -> about 85 s per episode, about 27 h serial for 1 134 episodes (12 h for the minimum tier), plus 3-4 min per patient for set-up. All of these are estimates until the pilot.
With 8 parallel workers this is about 3.5 h. The local WSL2/Docker stack hung repeatedly during this project (kernel VMA-lock hangs, Docker restarts), including DiffDRR container renders; run bulk generation on the same native-Linux infrastructure as training.

## 10. Validation checks (all automatic, all predeclared; any failure blocks the freeze)

- V1 **firewall**: parquet columns exactly `action, observation.state, timestamp, frame_index, episode_index, index, task_index`; no privileged key in video/state/language; every policy PNG R=G=B, 320x320, no text chunks; language string constant.
- V2 **causality**: `previous_insert_velocity` equals the previous action; `inserted_length_m` non-decreasing; image_t built from the state accepted before action_t; video round-trip <= 1 gray level.
- V3 **physics**: status SUCCESS, max axis violation <= 1.5 mm, finite, no overshoot, final extremity inside the landing window, teacher repeats bit-identically.
- V4 **labels**: only {0.000, 0.006, 0.012}; exactly two class changes per episode; HOLD from the third decision after zone entry; realised class counts match section 7.
- V5 **vision necessity / no state shortcut**: within-patient SD of the feed at the stop >= 10 mm; a state-only classifier on `[inserted_length_m, previous_velocity]` trained on train and scored on val must not beat the always-CRUISE baseline by more than 0.05 in SLOW F1 (bound set now, before any model is trained).
- V6 **image**: per (patient, view) TARGET_VISIBLE/CATHETER_VISIBLE and protocol additivity pass; tip and landing zone in view in >= 95% of decisions; noise/contrast within the declared ranges; no consecutive identical frames outside the HOLD tail.
- V7 **splits**: patient sets disjoint; test patients absent from every statistic; normalisation stats from train only; hash manifest of every file.
- V8 **coverage**: every admitted patient has all view families, all three B values and all four feed quartiles; per-split minimum counts (SLOW >= 600 and chunk-with-change >= 1 000 per held-out set).
- V9 **loader smoke**: the real GR00T loader (`_groot_real_data_integration_v2.py` logic) reads the full dataset, horizon 16, and the sampled first-action histogram equals section 7.

## 11. Recommended first multi-patient training scale (to run on native Linux, multi-GPU DDP/NCCL; not started)

- Data: the **recommended tier** — 54 patients (36 / 9 / 9), 1 134 episodes, about 75k train samples, 4.0 GB. Go/no-go for scaling beyond it only after the minimum tier's validation results.
- Minimum viable first run: 36 patients (24 / 6 / 6), 504 episodes, about 33k train samples (5 564 SLOW, 6 912 transition-window samples), 1.8 GB. Below about 24 train patients a held-out anatomy is unlikely to fall inside the training range of path length, radius and bifurcation angle [heuristic, not proven].
- Budget guidance: 3-5 epochs over the recommended train set (about 225-375k sample-views), i.e. roughly 0.9-1.5k optimizer steps at a global batch of 256. Do not carry over learning rate 1e-4 / batch 8 unchanged to a global batch of 256; decide the scaling in a short LR check at training time. No scientific setting changes before then.
- Evaluation: per-patient, cluster-bootstrapped by episode on val and test-ID; report SLOW/HOLD recall and transition timing error in decisions, plus the shift probes.

## 12. Risks and decisions needed from you

Decisions:
1. **Approve teacher v2** (braking distance 7.2-12 mm, lag 0-2, 24 post-success holds). This is the only lever that raises SLOW; it changes the demonstrations, not the policy contract or API.
2. **Approve F in [60, 100] mm** (limited by the field of view and the validated feed regime), or authorise validating a longer regime.
3. **Approve the scale** (minimum 36 patients vs recommended 54) and generation on native Linux rather than WSL2.

Risks:
- Physics generalisation to other anatomies is unvalidated (the catheter model was tuned on s0011; narrower iliacs or tortuous aortas may violate the lumen). Mitigation: gate G2; the pool is capped at 61.
- DSA target visibility per patient is not guaranteed (for s0011 the catheter-free DRR failed the visibility criterion, best z = 2.5, and the roadmap protocol was adopted instead); mitigation: gate G3 per view.
- Reaction-lag jitter makes the SLOW onset inherently ambiguous by +-2 decisions (realistic, but it caps achievable timing accuracy).
- Vascular pathology and CT-angiography are confounded (13 of 14 vascular patients are CT-angiography).
- The planning model ignores physical glide and lumen interaction; realised counts are verified in the pilot.

Reproduction: `scripts/_scan_patient_pool.py`, `scripts/_make_split_plan.py`, `scripts/_plan_dataset_counts.py`; outputs in `outputs/cvpr_multipatient_design_v1/`.
