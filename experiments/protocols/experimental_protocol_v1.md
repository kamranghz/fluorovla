# Final experimental protocol v1 (pre-registration)

Registered: 2026-10-18, BEFORE any new multi-patient physics result exists. The SHA-256 of this file is recorded in `outputs/cvpr_protocol_v1/stage0/RECORD.json`.
Any later change is an **amendment**, recorded in section 22 with date, reason and the evidence that motivated it; this text is never silently edited.
Scope of execution: Stage 0 to Stage 8 below; execution STOPS before multi-patient GR00T fine-tuning and before bulk generation of the final training set.

## 0. Terminology

- **Privileged Teacher**: the anatomy-aware demonstration generator. It reads anatomy that the deployed policy never sees.
- **Teacher reference**: the Privileged Teacher's closed-loop result under the same evaluation conditions and seeds as a policy. It is a reference, not an "upper bound"; that phrase is not used unless operationally justified.
- **Development patients**: the finalized pilot cohort (about 12 representative patients + 1 stress sentinel). **Non-development patients**: everything else in the usable pool.
- **Same-source one-shot held-out cohort**: any later subjects from the same TotalSegmentator release. "External validation" is reserved for a different dataset/source.

## 1. Primary endpoint

Patient-level closed-loop navigation success: for each held-out patient, the fraction of its pre-declared closed-loop rollouts (section 7) that succeed (section 2); patients are weighted equally; uncertainty by a patient-level bootstrap (section 12).

## 2. Closed-loop success criteria (existing rules of `target_contract.evaluate`, unchanged)

A rollout is a **success** iff, within the timeout `T_MAX`, all of the following hold:
1. the tip extremity reaches the landing window: |extremity - window centre| <= EPS = 1.2 mm,
2. with the required dwell: step |ds| < 0.3 mm for DWELL = 3 consecutive decisions inside the window,
3. no overshoot: extremity never exceeds window centre + EPS,
4. lumen violation (max axis SDF excess over all substeps) <= LUMEN_DELTA = 1.5 mm at every decision,
5. finite, stable physics: all values finite, body speed <= 2.0 m/s, root displacement <= 1e-6 m.

Terminal priority (existing): PHYSICS_FAILURE > LUMEN_VIOLATION > OVERSHOOT > SUCCESS > (at `T_MAX`) STOPPED_SHORT / TIMEOUT.
Timeout: `T_MAX = ceil( 1.25 * ((F_c - 6 mm)/12 mm/s + 6 mm/(6 mm/s)) / 0.1 s ) + 3`, with `F_c` the distance from the start tip extremity to the landing-window centre.
(The original s0011 contract used the same expression without the 1.25 factor and gave 95 decisions; the s0011 regression keeps the original `T_MAX = 95`.)

## 3. Failure decomposition (pre-declared categories, mutually exclusive)

`OVERSHOOT`, `STOPPED_SHORT` (at T_MAX: stationary for DWELL decisions short of the window by more than EPS), `TIMEOUT` (at T_MAX, otherwise not in the window), `LUMEN_VIOLATION`, `PHYSICS_FAILURE`. Anything else is a protocol error and is reported, never silently dropped.

## 4. Admission criteria

- A **start condition** is a (patient, access side, start depth F). Start depths are F at 0%, 50% and 100% of the valid range `[60 mm, F_hi]`, `F_hi = min(100 mm, S_contact(twin) - 81.2 mm - 1 mm)`; `F` is the feed from the start tip extremity to the landing-window near edge; the portal sits `73.5 mm + F` upstream of the landing tip on the access path.
- A start condition **passes** iff the locked Privileged Teacher (reaction lag 0, no timing jitter) achieves SUCCESS (section 2) in the integrated physics episode.
- An access path (side) is **admitted** iff all three of its start conditions pass. A patient is **admitted** iff at least one side is admitted. Only admitted sides are used for policy evaluation.
- Physics admission is independent of view and image noise. Observation validity (DSA visibility per patient and view) is a separate gate and is reported separately.
- Reported at four levels: patient, side, start condition, and the cohort funnel (102 subjects -> anatomical availability -> corrected 59-patient pool -> selectable -> twin verified -> physics admitted), with failure reasons by section 3 and the anatomy descriptors of rejected patients.
- For the development-cohort mapping (Stage 3) the same definitions are applied with the **mapping Teacher** (the existing rule: CRUISE until 2.4 mm before the window, then SLOW, then HOLD), because the final Teacher does not exist yet; this is labelled "mapping admission" and is not the admission of Stage 7.

## 5. K-selection rule (N = admitted non-development patients)

N >= 25 -> 5 folds; 20 <= N <= 24 -> 4 folds; 15 <= N <= 19 -> 3 folds; N < 15 -> the main K-fold generalization claim is not made (per-patient results are reported descriptively).

## 6. Fold construction (seeded, constrained, patient-level)

- Patients that are suspected duplicates (same sex, age within 1 year, same scanner model and institute, and twin-measured access-path lengths within 3 mm on both sides) form one group and stay in one fold.
- Descriptors: the seven twin-measured descriptors (radius, path length, tortuosity, bend severity, landing clearance, flare, angle), robust-z (median/MAD over all admitted non-development patients), winsorised at +-3.5. Extremeness score = Euclidean norm of that vector; top quartile = "difficult".
- Candidates: 50 000 random assignments of groups to folds with fold sizes differing by at most one, from a generator seeded with the first 8 bytes of SHA-256("medphysai-folds-v1") read as an unsigned integer.
- Objective (minimise): the sum over folds of squared standardised deviations from the global value of (mean PC1, mean PC2, mean PC3 score of the descriptor vectors, number of difficult patients, number of CTA patients, number of one-sided patients, mean Teacher-demonstration difficulty = maximum lumen violation over the patient's admission episodes). Ties: lowest candidate index.
- The chosen assignment, its seed, its objective value and the balance table are stored before any training and never rerolled.

## 7. Closed-loop evaluation design (per held-out patient; defined now, run later)

- Factors: depth (the three start depths of section 4), view (AP, LAO45, RAO45), noise (low = 8000, high = 3000 photons/pixel), side. Contrast fixed at the nominal 1000 HU-equivalent.
- Two-sided patient, 12 runs: for each side s in {L, R} (shift 0 for L, shift 1 for R), for each depth index j in {0,1,2}, view index (j + shift) mod 3, for each noise level: 3 x 2 x 2 = 12. One-sided patient: the same construction with the side factor replaced by shifts {0, 1} on the single side. Each depth and view appears 4 times, each noise level and side 6 times, with balanced pairwise coverage; the view order is AP, LAO45, RAO45.
- Seeds: SHA-256(patient id + "|" + run index), first 4 bytes, identical across all compared methods.
- Methods run on identical conditions: GR00T policy, Teacher reference, state-only policy, always-CRUISE baseline.
- Out-of-range views and noise levels are exploratory only.

## 8. Training schedule scaling (rule registered now, value set from development data only)

Optimizer steps per fold = `ceil(c * N_train_samples / global_batch)`; `c` is determined once from development experiments (never from held-out folds) and recorded before the first fold is trained. Same recipe in every fold; every fold starts from the original pretrained GR00T N1.7 checkpoint; no checkpoint is carried between folds.

## 9. Checkpoint selection

A fixed pre-declared schedule; one pre-declared final checkpoint per fold is evaluated. Earlier checkpoints may be reported as a sensitivity analysis but are never used for selection. The held-out fold never influences step count, checkpoint, thresholds, normalization, augmentation, hyperparameters or Teacher parameters.

## 10. Crash / NaN rule

A failed or crashed evaluation rollout counts as a failure. Only a demonstrable infrastructure failure (section 11) can invalidate it.

## 11. Errata policy

A rerun is allowed only for a demonstrable implementation or infrastructure error identified independently of model performance. Every rerun is recorded (what, why, evidence, outcome) in the stage record. A method change after an admission sweep means the patients whose results motivated it are no longer untouched evaluation patients.

## 12. Reporting hierarchy

- **Primary**: patient-level closed-loop success with explicit denominator, a patient-level cluster bootstrap 95% CI (10 000 resamples, fixed seed), per-patient descriptive results, per-fold results, fold mean +- SD (descriptive). A positive control (success on training/development anatomy) is reported but is not a generalization claim.
- **Secondary (pre-declared)**: paired GR00T vs Teacher reference; paired GR00T vs state-only; failure decomposition; offline action metrics; the Teacher-adjusted anatomical-distance analysis; a random-intercept logistic sensitivity analysis.
- **Anatomical distance**: seven twin descriptors, winsorised robust-z with median/MAD from the current fold's training patients only; distance of a held-out patient = mean Euclidean distance to its three nearest training patients (averaged over its admitted sides); primary distance analysis = (policy success - Teacher reference success) vs distance; Mahalanobis distance only as sensitivity.
- **Exploratory**: subgroups with fewer than about 8 patients, out-of-range views and noise, unadjusted success-vs-distance, development-in vs development-out.
- "Vision-grounded" is claimed only if the state-only baseline materially underperforms the full policy under the pre-declared paired comparison.

## 13. Lockbox rule

If at least 40 admitted non-development patients remain: reserve 8 with a seeded descriptor-stratified rule (strata: extremeness tertile x CTA) before any training and seal the list by its hash. Otherwise no lockbox; all admitted non-development patients enter the K-fold evaluation. The decision depends only on the cohort size.

## 14. Stage gates (a failed gate stops its dependent branch; nothing is forced to pass)

- **Stage 0B regression**: the parameterized pipeline must reproduce the accepted s0011 episode (physics rows `priv_tip_extremity_end_mm`, `inserted_length_m`, `axis_violation_mm`, action sequence) bit-identically or within 1e-9 relative. No physics retuning to pass it; only genuine parameterization bugs may be fixed.
- **Stage 1 twin verification**: reported, not gated; it feeds Stage 2.
- **Stage 3 physics mapping**: no assumed yield; results reported at patient, side and start-condition level.
- **SAFETY (Stage 4)** remains in the Teacher only if BOTH hold; otherwise it is removed and the Teacher is LANDING-only.
  - Hazard sufficiency (precondition): among development patients excluding the sentinel, at least 30 CRUISE decisions with axis violation >= 0.45 mm (30% of the budget) from at least 3 patients; otherwise SAFETY is "not testable" and removed.
  - T-HAZ: a simple monotone hazard model (at most two features: lumen clearance and 30-mm sagitta) trained leave-one-patient-out inside the development cohort; the median per-held-out-patient Spearman correlation between predicted and measured violation >= 0.5 and at least two thirds of patients with correlation > 0.3.
  - T-SPEED: paired runs on identical start conditions where only the speed over a pre-declared 20-mm stretch (centred on the measured CRUISE violation peak) changes from CRUISE to SLOW; at least 8 paths with CRUISE stretch-maximum >= 0.45 mm; median paired relative reduction of the stretch-maximum violation >= 20% and a reduction on at least 70% of the paths.
  - Thresholds are not redesigned to rescue SAFETY. The sentinel is a physics stress case only and is excluded from fitting, thresholds and gate statistics.
- **N_px resolvability (Stage 5)**: a fixed, policy-independent image probe (small convolutional regressor, fixed architecture and epochs, trained leave-one-patient-out on development-cohort frames) predicts distance-to-landing from the rendered frame; sigma_d = RMS error for true distance <= 20 mm, worst over the three views and both noise levels; `N_px = ceil(2 * sigma_d / 0.36 mm)`; the landing floor is `d_floor = max(2.4 mm, N_px * 0.36 mm)`.
- **LANDING (Stage 5)**: three arms evaluated on the development conditions with physics only: legacy (2.4 mm), floor only, floor + anatomy term (severity from clearance, flare, angle, descriptor ranges frozen from development patients only, extra distance at most 6 mm). Metric: composite failure rate (overshoot + stopped-short + lumen violation) over all development start conditions. The simplest arm whose composite failure rate is within 2 percentage points of the best arm is chosen. The anatomy term is kept only if it reduces the composite failure rate by at least 5 percentage points or 20% relative, with no increase in timeouts.
- **No class-balancing feedback**: CRUISE / SLOW / HOLD proportions are reported outcomes. Guard (validation only): in the approach phase CRUISE >= 70% of decisions.
- **Integrated pilot (Stage 6)** must verify: causal alignment; no privileged leakage (parquet/video/state/language checks); no state-only shortcut (a state-only classifier must not exceed always-CRUISE by more than 0.05 SLOW F1); DSA visibility; action distributions; every SLOW decision tagged `LANDING` or `SAFETY` (validated) and no other cause; Teacher consistency (bit-identical re-run of 3 episodes); physics stability.

## 15. Development cohort (Stage 2 rules)

Re-run the deterministic selection (`scripts/design_v3/_sel_*`) in twin-measured descriptor space on all selectable patients. Provisional cohort: s0011, s1400, s1388, s1124, s1206, s1350, s0344, s0513, s0629, s0912, s1336, s1387; sentinel s1082; reserves s1391, s0727, s0058. Names are not preserved for continuity if the twin-space selection differs.
Conditional rules: s1206 is kept only if the twin gives at least one valid access path >= 150 mm with a valid landing zone, otherwise s0727; s1082 is kept as stress sentinel only if the twin confirms the severe tortuosity, and it must not drive Teacher calibration or threshold tuning. All development patients are permanently excluded from unseen validation/test; they may enter training if they pass final admission.

## 16. Policy-visible inputs and firewall

Policy sees only: the DSA image, `inserted_length_m`, `previous_insert_velocity_mps`, and the fixed language instruction. Never exposed: target XYZ, exact target distance, centerline coordinates, vessel radius, curvature/tortuosity values, SDF, wall clearance, landing-zone coordinates, full physics state, Teacher-only labels/features. Teacher constants are frozen from development patients only (no per-fold Teacher).

## 17. Claim scope

"Closed-loop success on held-out patient-specific aorto-iliac anatomies from CT, in physics simulation with rendered fluoroscopic images, estimated by patient-level K-fold cross-validation under a locked simulator-admission criterion." Not clinical validation, not real fluoroscopy, not robotic deployment, not steering or branch selection, not anatomies rejected by admission, not other catheter designs, not external validation.

## 18. One-shot admission sweep (Stage 7)

Run exactly once on all non-development patients with the locked method (physics, Teacher, thresholds, LANDING, SAFETY, N_px, rendering unchanged afterwards). A genuine implementation bug found there is handled under section 11.

## 19. Evidence

Every stage writes `outputs/cvpr_protocol_v1/<stage>/RECORD.json` with: patients, configuration, SHA-256 of relevant code and data files (there is no git repository), criteria, measured results, pass/fail decision, deviations, and reruns with reasons. Failed evidence is never overwritten; reruns go to a new numbered attempt directory.

## 20. Stop condition

Proceed through Stage 8 if gates permit, then stop and report before multi-patient GR00T training.

## 21. Operating-environment note (affects reliability, not method)

The local WSL2/Docker stack exhibited repeated kernel-level hangs and engine restarts. Execution uses: a keep-alive container, a 5-minute VM-age rule, a CUDA warm-up before GPU work, per-job timeouts and hang detection, and recording every infrastructure rerun under section 11. Infrastructure reruns never change results that were obtained.

## 22. Amendments

(none yet)
