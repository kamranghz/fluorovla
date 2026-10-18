# Representative pilot-cohort selection (v1) — analysis only

Date: 2026-10-18. **No physics, no generation, no training was run.** Selection uses only anatomical/path descriptors and metadata; downstream physics success was not used and is not known for any candidate.
Nothing here is final until the patient list is approved. All numbers come from `scripts/design_v3/` and `outputs/cvpr_multipatient_design_v1/` (section 14).
Caveat that applies to everything below: descriptors come from the design-stage extractor (skimage skeleton), not the i4h twin; on s0011 its path length is about 4% shorter than the twin's. The pilot's first step must re-measure the chosen patients with the twin.

## 0. Result in brief

- **Proposed pilot: 12 patients** — 1 benign reference (s0011), 4 valid edge cases, 2 compound-hard cases, 5 core cases (3 of them typical/median). Recommended size: 12 (section 12).
- Coverage of the 91 verified access paths, against the previous 4-patient set: mean distance to the nearest selected patient **1.21 vs 1.90**, worst-case distance **2.60 vs 4.23**, extreme tails covered **14/14 vs 6/14**, hard-combination pairs **13/15 vs 8/15**, descriptor-range coverage **1.00 vs 0.75**. A random 12-patient cohort (with s0011) averages 1.36, 12.9 tails, 10.2 pairs.
- 5 of the 12 are chosen by a fixed rule (reference + edge cover); the rest by an objective (compound-hard, then greedy facility-location fill). Stability: six patients are chosen in every one of 6 method variants, three more in 5 of 6 (section 13).
- Two items need your decision (section 13): whether to add one QA-confirmed extreme-tortuosity patient (s1082) as a 13th sentinel, and whether to use s0727 instead of the boundary-length patient s1206.

## 1. Method (systematic, no manual choice of patients)

**Unit of analysis**: the access path (patient x iliac side), 112 paths from the 59 usable patients (the corrected gate: access-path length >= 150 mm). **Unit of selection**: the patient; a selected patient contributes all its usable paths.

**Descriptors** (the seven requested), computed over the *analysis window* an episode can occupy — the last 173.5 mm before the landing tip (73.5 mm rod body + 100 mm maximum tip travel) — with robust statistics:
radius (window median of the inscribed lumen radius), access-path length `S_contact`, tortuosity (window arc length / chord of the smoothed path), bend severity (95th percentile of the 30-mm chord sagitta, the rod-segment length), landing-zone clearance (lumen radius at the landing tip minus the 3 mm capsule), lumen flare before the bifurcation (r at the path end / r 20 mm earlier), daughter-vessel angle. Also recorded: narrowest stretch (p5 radius) and maximum sagitta.

**Normalisation**: robust z = (x - median) / (1.4826 MAD) over the 110 paths with all descriptors, winsorised at +-3.5, equal weight on the seven descriptors (Spearman correlations between them are 0.80 at most — tortuosity vs bend — and 5 principal components are needed for 90% of the variance, so the space is genuinely multi-dimensional).

**Quality status per path** (automatic, before any selection):
- FLAGGED (not usable): no daughter angle; the two sides disagree on where the bifurcation is (> 10 mm); skeleton jaggedness outlier; flare outside [0.6, 1.6]; narrowest radius at the segmentation resolution limit (< 2.25 mm).
- EXTREME: no automatic flag but some descriptor beyond robust z +-3.5 — kept out of the *selection* (cannot be verified automatically), listed so the gap stays visible.
- BOUNDARY: path length within 8 mm of the 150 mm gate (the extractor has +-5% uncertainty). OK: none of the above.
Patients are **eligible** if at least one path is OK or BOUNDARY: 50 of 59.

**Selection algorithm** (deterministic, `_sel_step3_final.py`):
1. Reference: s0011 is fixed.
2. Edge cover: nine edge categories (narrow, wide, short, long, high tortuosity, large bend, large angle, strong flare, low landing clearance), each defined as the most extreme 5% of verified paths in its direction (the tail widens to 7.5%/10% only if no verified candidate exists; it never had to); a greedy set cover picks the fewest patients whose paths lie in all nine tails.
3. Compound-hard: a case is compound-hard if at least 3 of the 6 challenge dimensions (narrow, tortuous, strong bend, low clearance, strong flare, large angle) are in its hardest 20%; two are chosen greedily to maximise the hard-combination pairs covered, then hardness.
4. Core: greedy facility location (k-medoids objective) adds the patient that most reduces the mean distance from all verified paths to the nearest selected patient, until the target size.
Coverage is measured on the 91 *verified* paths (OK/BOUNDARY); a selected patient's flagged or extreme side does not count as coverage.

## 2. Quality status of the 112 paths

OK 88, BOUNDARY 3, EXTREME 8, FLAGGED 13. Nine patients have no OK/BOUNDARY path and cannot be selected: **s1029** (no daughter angle; sides disagree on the bifurcation), **s1361** and **s1373** (sides disagree), **s1383**, **s1405**, **s1379** (flare outside the plausible range; s1379 and s1403 also have very large flare and clearance), **s0950** (only the right side is a feasible path and it has extreme landing clearance), **s1082** and **s1372** (extreme tortuosity/clearance on both sides).
Visual QA of the extraction (`qa_*.png`) shows that "extreme" is not the same as "wrong": s1082 has genuine, continuous iliac looping on both sides; s1403 and s1379 have a wide, bulging terminal aorta (aneurysm-like), so their large flare is real, not an artifact.
Flare is the least reliable descriptor: for the same bifurcation the left and right paths disagree by a median 0.12 (p90 0.34, max 0.75), which equals the between-path spread (MAD 0.12). Results that depend on flare are therefore flagged and tested without it (section 13).

## 3. A. The 59 usable patients

Status per side: OK / BND (boundary length) / EXT (extreme, unverified) / FLG (flagged, not used); `-` = no feasible path on that side. z = robust z, mean of the usable paths (raw, not winsorised); `-` where no path is usable. Also in `outputs/cvpr_multipatient_design_v1/selection_table_A_59_patients.csv`.

| patient | L / R status | CTA | pathology | sex | z radius | z length | z tort. | z bend | z clear. | z flare | z angle | cohort role |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0011 | OK / OK |  | tumor | m | -0.8 | -1.0 | -1.5 | -2.0 | -0.7 | +1.0 | -0.8 | reference |
| s0058 | OK / OK |  | tumor | f | -0.5 | -0.2 | -0.7 | -0.3 | -0.2 | +0.8 | +0.6 |  |
| s0250 | OK / FLG |  | no_pathology | m | +0.7 | +0.2 | -0.2 | -0.3 | +0.1 | +0.3 | +1.0 |  |
| s0344 | OK / OK |  | inflammation | m | +0.0 | +1.1 | +2.3 | +1.2 | -1.0 | -0.2 | -1.2 | compound |
| s0358 | OK / OK |  | trauma | f | -0.1 | -0.7 | -0.2 | +0.0 | -0.3 | +0.5 | +0.0 |  |
| s0454 | OK / OK |  | trauma | f | -0.1 | -1.2 | -1.1 | +0.1 | +0.9 | +0.0 | -0.8 |  |
| s0461 | OK / OK |  | no_pathology | m | +0.4 | +0.5 | +0.0 | +0.4 | +0.4 | -0.6 | +1.1 |  |
| s0477 | OK / OK |  | tumor | f | -1.7 | -0.7 | +0.2 | -0.4 | -0.9 | -0.2 | -0.1 |  |
| s0513 | OK / OK |  | unclear | f | +0.0 | -0.1 | -0.4 | -0.1 | -0.4 | +0.2 | -0.1 | core |
| s0519 | OK / OK |  | tumor | m | -1.4 | -0.5 | -0.9 | -0.8 | -0.7 | +0.5 | +0.2 |  |
| s0573 | OK / OK |  | trauma | f | +0.4 | -0.5 | -0.6 | -0.8 | -0.5 | +1.0 | -0.1 |  |
| s0591 | - / BND |  | trauma | m | +0.2 | -1.8 | -0.4 | -0.4 | -0.9 | +0.8 | -0.1 |  |
| s0629 | OK / OK |  | tumor | m | -0.3 | -0.1 | +0.0 | +0.5 | +0.1 | -0.8 | -0.0 | core |
| s0649 | OK / FLG |  | tumor | f | -0.5 | -0.2 | -0.1 | -0.1 | +0.6 | -0.2 | +0.9 |  |
| s0703 | OK / OK |  | tumor | m | -0.1 | +0.7 | +0.9 | +0.0 | -0.5 | +0.3 | +0.1 |  |
| s0720 | OK / OK |  | tumor | m | +0.1 | +0.3 | -1.4 | -1.5 | -0.2 | +0.5 | +0.6 |  |
| s0726 | OK / OK |  | tumor | f | -0.0 | -0.4 | -0.3 | +0.1 | -0.7 | +0.2 | +0.9 |  |
| s0727 | BND / OK |  | trauma | f | -0.3 | -1.4 | -0.3 | -0.4 | -0.2 | -1.0 | +0.3 |  |
| s0763 | OK / OK |  | trauma | m | -0.2 | +0.4 | +0.1 | +0.4 | -0.7 | +0.2 | -1.3 |  |
| s0812 | OK / OK |  | trauma | m | -0.1 | -0.9 | -0.0 | -0.5 | -0.9 | -0.4 | +0.3 |  |
| s0815 | OK / OK |  | inflammation | m | +1.4 | -0.3 | -0.5 | -0.9 | +1.4 | -0.6 | +0.4 |  |
| s0859 | OK / OK |  | inflammation | f | -0.2 | -0.3 | -0.4 | -0.5 | +1.2 | -0.2 | +0.0 |  |
| s0878 | OK / OK |  | tumor | m | -1.5 | -0.9 | -1.3 | -1.3 | -1.3 | +0.9 | -0.0 |  |
| s0885 | - / OK |  | inflammation | m | +1.1 | +1.3 | +0.1 | -0.8 | +1.1 | -0.1 | +0.2 |  |
| s0912 | OK / OK |  | inflammation | m | +0.6 | +0.2 | -0.4 | -0.2 | +0.2 | +0.3 | +0.0 | core |
| s0915 | OK / OK |  | no_pathology | m | -0.3 | +0.4 | +1.0 | +0.8 | +1.1 | +0.2 | +0.5 |  |
| s0928 | OK / - |  | no_pathology | f | -0.5 | -0.3 | +2.0 | +0.6 | -0.3 | -1.8 | +0.2 |  |
| s0950 | - / EXT |  | tumor | f | -0.4 | +1.1 | +3.0 | +1.0 | +4.5 | -1.6 | -2.3 |  |
| s1029 | FLG / FLG |  | trauma | m | - | - | - | - | - | - | - |  |
| s1082 | EXT / EXT |  | trauma | m | +1.1 | -1.4 | +5.8 | +1.4 | -0.4 | +0.5 | -0.1 |  |
| s1086 | OK / OK |  | tumor | m | +0.6 | +0.7 | +1.5 | +0.7 | -0.0 | +0.2 | -0.6 |  |
| s1124 | OK / OK |  | trauma | m | +0.7 | +0.3 | -0.2 | -0.3 | +0.2 | +1.6 | -0.6 | edge |
| s1127 | OK / FLG |  | tumor | f | -0.4 | -0.3 | +0.4 | +0.9 | -0.2 | -0.7 | -0.7 |  |
| s1206 | BND / - |  | inflammation | m | -1.3 | -1.9 | -0.9 | -1.2 | -0.7 | -0.5 | -0.4 | edge |
| s1267 | OK / OK |  | no_pathology | m | +1.9 | +1.3 | +0.1 | +0.2 | +0.8 | -0.2 | -0.5 |  |
| s1287 | OK / OK |  | tumor | m | +1.6 | +1.1 | +1.7 | +1.1 | +0.6 | +0.3 | -2.3 |  |
| s1336 | OK / OK |  | no_pathology | m | +1.3 | +0.6 | +1.7 | +0.7 | +0.8 | -0.0 | -0.3 | core |
| s1348 | OK / OK |  | other | m | +0.2 | -0.9 | -0.0 | -0.2 | +0.2 | +0.2 | -0.2 |  |
| s1350 | OK / OK |  | trauma | f | -1.6 | -0.5 | +0.1 | +0.3 | -1.2 | +0.7 | +0.6 | compound |
| s1361 | FLG / FLG | CTA | vascular | m | - | - | - | - | - | - | - |  |
| s1364 | OK / OK | CTA | vascular | m | -0.6 | -0.5 | +0.6 | +0.4 | -0.5 | -0.0 | +0.4 |  |
| s1365 | OK / OK | CTA | vascular | f | -1.5 | -0.5 | -0.2 | +0.0 | -0.2 | -0.0 | +0.2 |  |
| s1366 | OK / OK |  | unclear | unk | -1.5 | -0.1 | -0.9 | -0.9 | -1.1 | +0.6 | +0.7 |  |
| s1369 | OK / OK |  | no_pathology | f | +0.0 | +0.6 | +1.3 | -0.1 | -0.6 | +0.5 | +0.3 |  |
| s1371 | OK / EXT |  | no_pathology | m | +2.2 | +0.9 | +2.8 | +0.2 | +1.2 | +0.3 | -0.7 |  |
| s1372 | EXT / EXT | CTA | vascular | f | +1.5 | +0.1 | +3.5 | +0.4 | +3.0 | -1.3 | -0.7 |  |
| s1373 | FLG / FLG | CTA | tumor | m | - | - | - | - | - | - | - |  |
| s1379 | FLG / EXT | CTA | vascular | m | +1.1 | +1.0 | +5.4 | +2.1 | +4.0 | +3.5 | -2.8 |  |
| s1383 | FLG / - | CTA | vascular | m | - | - | - | - | - | - | - |  |
| s1387 | OK / OK | CTA | vascular | m | -0.1 | -0.8 | +1.0 | +0.8 | +0.6 | +0.1 | -1.5 | core |
| s1388 | OK / OK |  | other | f | -1.9 | -1.1 | -0.9 | -0.9 | -1.5 | +0.5 | +0.3 | edge |
| s1390 | OK / OK | CTA | vascular | m | +0.9 | +1.2 | +1.3 | +0.6 | +0.1 | -1.6 | -0.2 |  |
| s1391 | OK / OK | CTA | bleeding | m | +1.5 | +0.6 | +1.3 | +0.8 | +0.1 | +0.5 | +0.7 |  |
| s1397 | OK / OK |  | tumor | m | +1.2 | +0.4 | +1.6 | +0.5 | +0.9 | -1.0 | -0.3 |  |
| s1400 | OK / OK | CTA | vascular | m | +2.1 | +2.3 | +2.6 | +1.1 | +1.1 | -2.1 | +1.5 | edge |
| s1401 | OK / OK | CTA | no_pathology | f | +0.1 | -0.3 | -1.4 | -1.7 | -0.5 | -0.7 | +0.4 |  |
| s1403 | EXT / OK | CTA | vascular | m | -1.2 | -0.3 | -0.8 | -0.1 | +0.5 | +1.3 | +0.7 |  |
| s1404 | OK / OK | CTA | no_pathology | f | -0.1 | -0.1 | -0.0 | -0.7 | -0.7 | +0.3 | -0.3 |  |
| s1405 | FLG / FLG | CTA | vascular | m | - | - | - | - | - | - | - |  |

## 4. B. Descriptor ranges over the 99 non-flagged access paths (including the 8 extreme)

| descriptor | min | 5th pct | median | 95th pct | max |
|---|---|---|---|---|---|
| vessel radius, window median (mm) | 3.23 | 3.38 | 4.29 | 5.21 | 5.88 |
| access-path length (mm) | 150.01 | 163.95 | 197.70 | 231.91 | 265.09 |
| tortuosity, window (arc/chord) | 1.03 | 1.05 | 1.20 | 1.59 | 2.03 |
| bend severity, p95 of 30-mm sagitta (mm) | 1.34 | 2.21 | 4.20 | 6.07 | 7.14 |
| landing-zone clearance (mm) | 0.62 | 1.55 | 2.44 | 4.50 | 6.87 |
| lumen flare before bifurcation (ratio) | 0.62 | 0.80 | 1.00 | 1.22 | 1.53 |
| daughter-vessel angle (deg) | 111.44 | 122.74 | 145.92 | 162.11 | 168.71 |
| narrowest stretch, p5 radius (mm) | 2.79 | 2.93 | 3.54 | 4.46 | 4.76 |
| maximum 30-mm sagitta (mm) | 1.40 | 2.46 | 4.59 | 6.80 | 8.23 |

## 5. C. Proposed pilot cohort (12 patients; patient means over usable paths)

**Core representative cases (5)**: s0513, s0629, s0912 (typical), s1336, s1387 (moderately larger / bendier core). **Valid edge cases (4)**: s1400, s1388, s1124, s1206. **Compound-hard (2)**: s1350, s0344. **Reference (1)**: s0011.

| patient | role | usable sides | CTA, pathology, sex, age | scanner | radius mm | length mm | tort. | bend mm | clearance mm | flare | angle | verified paths it represents |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0011 | reference | L+R | non-CTA, tumor, m, 45 | emotion 16 | 3.84 | 173 | 1.03 | 1.3 | 2.00 | 1.12 | 138 | 4 |
| s1400 | edge | L+R | CTA, vascular, m, 69 | somatom definition flash | 5.47 | 256 | 1.51 | 5.8 | 3.62 | 0.75 | 161 | 2 |
| s1388 | edge | L+R | non-CTA, other, f, 70 | somatom definition edge | 3.26 | 171 | 1.10 | 2.9 | 1.23 | 1.06 | 149 | 4 |
| s1124 | edge | L+R | non-CTA, trauma, m, 74 | somatom force | 4.71 | 207 | 1.18 | 3.8 | 2.85 | 1.19 | 140 | 2 |
| s1206 | edge | L | non-CTA, inflammation, m, 62 | somatom force | 3.58 | 150 | 1.10 | 2.5 | 1.98 | 0.94 | 142 | 3 |
| s1350 | compound | L+R | non-CTA, trauma, f, 65 | somatom definition as+ | 3.43 | 186 | 1.22 | 4.6 | 1.46 | 1.08 | 152 | 7 |
| s0344 | compound | L+R | non-CTA, inflammation, m, 57 | somatom definition as+ | 4.31 | 226 | 1.48 | 6.0 | 1.65 | 0.98 | 134 | 2 |
| s0513 | core | L+R | non-CTA, unclear, f, 67 | somatom definition flash | 4.32 | 195 | 1.16 | 4.0 | 2.29 | 1.03 | 146 | 19 |
| s1336 | core | L+R | non-CTA, no_pathology, m, 76 | somatom definition edge | 4.98 | 213 | 1.41 | 5.3 | 3.39 | 0.99 | 143 | 13 |
| s0629 | core | L+R | non-CTA, tumor, m, 64 | somatom definition flash | 4.15 | 197 | 1.21 | 4.9 | 2.76 | 0.91 | 146 | 16 |
| s0912 | core | L+R | non-CTA, inflammation, m, 69 | somatom force | 4.65 | 204 | 1.16 | 3.9 | 2.77 | 1.04 | 146 | 14 |
| s1387 | core | L+R | CTA, vascular, m, 72 | somatom definition edge | 4.25 | 179 | 1.33 | 5.4 | 3.18 | 1.02 | 131 | 5 |

Composition: CTA 2 / non-CTA 10 (both CTA patients also have vascular pathology — the pool confounds the two: 13 of 14 vascular patients are CTA); male 9 / female 3; 5 of 8 scanner models; 7 pathology groups; 11 patients with two usable sides and 1 (s1206) with one.

## 6. D. What each selected patient represents

- **s0011 (reference)** — the benign corner: 173 mm path, almost straight (tortuosity 1.03, robust z -1.5; bend 1.3 mm, z -2.0, in the lowest 10%), narrow-to-normal radius 3.84 mm, clearance 2.0 mm. 3 of the 14 extreme tails (short, low tortuosity, low bend). The only anatomy with a twin and real physics. It represents only 4 verified paths: nobody else is that straight.
- **s1400 (edge: large, long, tortuous)** — widest vessel (5.47 mm; the most extreme verified path is 5.76), longest path (256 mm; maximum 265), tortuosity 1.51, bend 5.8 mm, large clearance 3.6 mm, wide angle 161 deg, low flare 0.75. 7 of 14 tails. A CTA/vascular case with a distorted aorta and very winding iliacs (QA image). Tests the wide-lumen regime (loose catheter), the long-path/deep-start regime and strong bends.
- **s1388 (edge: narrow, tight)** — narrowest vessel (3.26 mm; most extreme verified 3.23) and the lowest landing clearance (1.23 mm; 0.62 on one side, the most extreme verified), short-ish path 171 mm. 4 tails. The regime where the 3 mm capsule has the least margin — the highest expected wall contact.
- **s1124 (edge: bifurcation geometry)** — the strongest verified flare (1.42 on the left) and a large daughter angle (163.6 deg on one side, small on the other: an asymmetric bifurcation), moderate size and length. 3 tails. Tests the landing-envelope descriptors (flare, angle). Flare is the least reliable descriptor, see section 9.
- **s1206 (edge: shortest path)** — 150.0 mm (left) and 149.3 mm (right): the shortest access path in the pool, so the start depth can only span about 60-69 mm (`F_max = S_contact - 81.2`). Straight (tortuosity 1.10, bend 2.5 mm). Tests the minimum-length handling, the narrow start-depth range and one-sided feasibility. 2 tails.
- **s1350 (compound: narrow + tight + wide angle)** — narrow lumen (3.43 mm, z -1.6), reduced clearance (1.46 mm, z -1.2), large angle (152 deg; 166 deg on one side) with moderate bend (4.6 mm). Three of the six challenge dimensions in the hardest 20% at once. The largest edge cell: it represents 7 verified paths.
- **s0344 (compound: long, tortuous, bendy, tight)** — 226 mm path, tortuosity 1.48, bend 6.0 mm (6.3 on one side, the most extreme verified), clearance 1.65 mm. 4 tails. The combination of strong bends with reduced clearance.
- **s0513, s0629, s0912 (core: typical)** — all seven descriptors within about +-1 robust z; their distance from the pool median is 0.6, 1.0 and 0.9 (a typical path is 2.4 away). 0 extreme tails each. They are the nearest selected patient for 19, 16 and 14 verified paths (49 of 91). s0513: female, 67, unclear pathology, Somatom Definition Flash; s0629: male, 64, tumor, Flash; s0912: male, 69, inflammation, Somatom Force.
- **s1336 (core: wide, moderately tortuous)** — radius 4.98 mm (z +1.3), tortuosity 1.41 (z +1.7), bend 5.3 mm, large clearance 3.39 mm; the "larger, moderately winding" mass, 13 verified paths, no pathology (QA shows a wide aortic bulb).
- **s1387 (core: CTA vascular, bendy)** — moderate radius 4.25 mm, strong bend 5.4 mm, tortuosity 1.33, large clearance 3.18 mm, smaller angle 131 deg, aneurysm-like sac above the bifurcation (QA). 5 tails, 5 verified paths. The only core patient with the CTA lumen-normalisation requirement.

## 7. E. What the cohort covers and what it does not

**The nine edge categories** (all at the most extreme 5% of verified paths):

| category | cohort patient (value) | rank among verified |
|---|---|---|
| narrow vessel | s1388 (3.23 mm, one side) | most extreme |
| wide vessel | s1400 (5.76 mm) | most extreme |
| short path | s1206 (150.0 mm) | most extreme |
| long path | s1400 (265 mm) | most extreme |
| high tortuosity | s1400 (1.58) | most extreme verified |
| large bend severity | s0344 (6.30 mm) | most extreme |
| large bifurcation angle | s1350 (166.0 deg) | 3rd (max 168.7, s1364) |
| strong flare | s1124 (1.42, left only) | most extreme verified |
| reduced landing clearance | s1388 (0.62 mm, one side) | most extreme |

**Tails**: all 14 one-sided tails (lowest and highest 10%, and also lowest/highest 5%, of the 7 descriptors) contain at least one selected patient. Range coverage 1.00 for every descriptor.

**Not covered**
1. Two hard-combination pairs (both dimensions in the hardest quartile): **narrow vessel + high tortuosity** (patients s0928, s1364) and **high tortuosity + strong flare** (s1369, s1391). Swapping one core patient in covers one of them (14/15) at a small cost; none covers both.
2. The **extreme tier** (eight EXTREME paths): tortuosity beyond z +3.5 (s1082 both sides, s1371 right, s1372 right, s1379 right), very large landing clearance (s0950, s1372 left), and aneurysm-type flare (s1403, s1379). These are probably the physically hardest anatomies; the nearest selected patient is 2.6-5.4 distance units away (3.1-3.4 for s1082).
3. The nine non-selectable patients (section 2) — in particular bifurcations the extractor cannot delineate consistently (s1029, s1361, s1373).
4. Strong flare is only covered up to 1.42; true aneurysmal flare is in the unverified tier.
5. Single-sided feasibility is represented by one patient (s1206); 9 of the 50 eligible patients have one usable side.

## 8. F. Comparison with the previous 4-patient set (s0011, s1391, s0727, s0058)

| metric (91 verified paths) | previous 4 | proposed 12 | random 12 (with s0011, mean and 5-95%) |
|---|---|---|---|
| mean distance to nearest selected patient | 1.90 | **1.21** | 1.36 (1.27-1.46) |
| 90th percentile distance | 2.89 | 2.11 | - |
| worst-case distance | 4.23 | 2.60 | - |
| extreme tails covered | 6/14 | **14/14** | 12.9 |
| hard-combination pairs covered | 8/15 | 13/15 | 10.2 |
| descriptor-range coverage | 0.75 | 1.00 | - |

The previous set has no narrow vessel, no long path, no high tortuosity, no strong bend, no low or high clearance and no angle extreme (tails missed: radius low, length high, tortuosity high, bend high, clearance low and high, angle low and high). Only s0011 is kept; s1391 (wide, tortuous, CTA) is superseded by s1400/s1336, s0727 (short path) by s1206, s0058 (typical) by s0513/s0629/s0912. All three are valid reserves: swapping s0912 for s0058 costs 1% (mean 1.224); s1206 for s0727 gives mean 1.201 with both sides usable (section 13).

## 9. G. Selected cases near an extraction-quality boundary

| patient | what is near a boundary | consequence |
|---|---|---|
| s1206 | path length 150.0 / 149.3 mm at the 150 mm gate; only the left side counts | the twin may put it slightly above or below the gate (prototype lengths are about 4% short on s0011); F range only 60-69 mm |
| s1400 | right flare 0.70 (z -2.5, bound 0.6); right tortuosity z +3.2; left radius z +2.7 and length z +2.6; unusual aorta | valid in QA but several descriptors near the outlier bound; its flare is the weakest |
| s1124 | left/right flare disagree by 0.46 (p90 0.34); left flare z +3.5 at the outlier bound; right angle z -3.0 | its "strong flare" status is uncertain; its angle role relies on the left side |
| s1387 | right angle z -3.4 (near bound); right skeleton jaggedness z +2.6 | descriptors of the right side less certain; left side is clean |
| s1350, s0344, s1388 | none flagged | - |
| s0011, s0513, s0629, s0912, s1336 | none | - |
No selected path is FLAGGED or EXTREME. Every selected path was inspected in the extraction QA images.

## 10. H. Structure of the proposed cohort

- **Core representative (5)**: s0513, s0629, s0912, s1336, s1387 — together nearest to 67 of 91 verified paths.
- **Valid edge (4)**: s1400, s1388, s1124, s1206 (plus the reference s0011, which sits at the benign corner).
- **Compound-hard (2)**: s1350 (narrow + tight + wide angle), s0344 (long + tortuous + bendy + tight).

## 11. I. Quantitative coverage

Distances are in robust-z units over the seven descriptors. For the proposed 12: mean 1.21, 90th percentile 2.11, maximum 2.60; 14/14 tails; 13/15 hard pairs; range coverage 1.00 for all seven descriptors; every verified path is within 2.6 of a selected patient (within 2.1 for 90%). Unverified extreme paths are not counted.
Cell sizes (verified paths whose nearest selected patient is this one): s0513 19, s0629 16, s0912 14, s1336 13, s1350 7, s1387 5, s0011 4, s1388 4, s1206 3, s1124 2, s1400 2, s0344 2 (sum 91).

## 12. J. Coverage curve and recommended pilot size

Nested cohorts: the forced set (reference + edges + compound = 7) and then core patients in order of benefit.

| cohort size | patient added | mean distance | change | 90th pct distance | max distance | tails covered | hard pairs covered |
|---|---|---|---|---|---|---|---|
| 7 | forced set | 2.090 |  | 3.22 | 3.70 | 14/14 | 13/15 |
| 8 | s0513 | 1.641 | -21.5% | 2.71 | 3.52 | 14/14 | 13/15 |
| 9 | s1336 | 1.466 | -10.7% | 2.36 | 3.21 | 14/14 | 13/15 |
| 10 | s0629 | 1.355 | -7.6% | 2.29 | 3.21 | 14/14 | 13/15 |
| 11 | s0912 | 1.278 | -5.7% | 2.23 | 3.21 | 14/14 | 13/15 |
| 12 | s1387 | 1.209 | -5.4% | 2.11 | 2.60 | 14/14 | 13/15 |
| 13 | s1086 | 1.150 | -4.9% | 2.06 | 2.48 | 14/14 | 13/15 |
| 14 | s1401 | 1.091 | -5.1% | 1.99 | 2.48 | 14/14 | 13/15 |
| 15 | s0859 | 1.033 | -5.3% | 1.91 | 2.48 | 14/14 | 13/15 |
| 16 | s1348 | 0.982 | -5.0% | 1.89 | 2.48 | 14/14 | 13/15 |

Reading the curve: all 14 tails are covered from size 7 on, so more patients add no tail coverage; the hard-pair coverage stays at 13/15. The mean distance falls steeply to size 10 (-7.6% at the third core patient) and then by about 5% per patient with no knee, but the **worst-case distance drops from 3.21 to 2.60 only at size 12** (s1387 covers the bendy CTA/aneurysm region) and to 2.48 at 13. A random 12-patient cohort is worse (1.36) and misses tails.
**Recommendation: 12 patients.** Size 10 is the smallest acceptable budget cohort (it keeps all tails and three typical patients but leaves a worst-case distance of 3.21 and drops the bendy CTA/aneurysm region); sizes above 12 add about 5% mean-distance gain per patient and no new tail, pair or edge coverage.

## 13. Stability, reserves and decisions needed

**Stability across method variants** (identical algorithm; each cohort's coverage re-measured in the primary space):

| variant | common with proposed | added by the variant | dropped | primary mean distance | primary worst-case |
|---|---|---|---|---|---|
| A percentile-rank normalisation | 11/12 | s0573 | s0912 | 1.208 | 2.60 |
| B flare removed from the distance | 9/12 | s1390, s1397, s1401 | s0912, s1336, s1387 | 1.253 | 2.98 |
| C PCA-whitened (5 components) | 11/12 | s1086 | s1336 | 1.217 | 2.53 |
| D edge tail 7.5% | 7/12 | s0058, s0461, s1267, s1364, s1369 | s0629, s0912, s1124, s1206, s1387 | 1.194 | 3.08 |
| E winsorise at 2.5 | 11/12 | s1086 | s1387 | 1.214 | 2.77 |

Chosen in all 6 runs: s0011, s1400, s1388, s1350, s0344, s0513. In 5 of 6: s0629, s1206, s1124. In 3-4 of 6: s1336, s1387, s0912 (interchangeable with s0573, s1086, s0058). The cohort's coverage is nearly the same in every variant (mean 1.19-1.25, 14/14 tails); the last slots are interchangeable, the edge patients are not.

**Reserves — the four most extreme verified patients per edge category (* = in the cohort)**

| category | most extreme verified patients (value of the patient's most extreme usable path) |
|---|---|
| narrow vessel | s1388 (3.23)*, s0477 (3.33), s0878 (3.36), s1350 (3.39)* |
| wide vessel | s1400 (5.76)*, s1267 (5.41), s1391 (5.3), s1287 (5.21) |
| short path | s1206 (150.01)*, s0591 (152.59), s0727 (157.54), s0454 (161.14) |
| long path | s1400 (265.09)*, s1267 (248.44), s1287 (235.81), s0885 (231.93) |
| high tortuosity | s1400 (1.58)*, s0344 (1.5)*, s1287 (1.48), s1086 (1.44) |
| large bend severity | s0344 (6.3)*, s1387 (6.19)*, s1390 (6.19), s1287 (6.0) |
| large bifurcation angle | s1364 (168.71), s1366 (167.36), s1350 (165.99)*, s1348 (164.66) |
| strong flare | s1124 (1.42)*, s0915 (1.24), s0573 (1.24), s1369 (1.22) |
| reduced landing clearance | s1388 (0.62)*, s0878 (1.21), s1350 (1.3)*, s1366 (1.44) |

**Decisions for you**
1. **s1206 (shortest path, at the gate) or s0727 (157.5 mm left / 162 mm right, both sides usable)?** Swapping gives mean distance 1.201 (slightly better) with no loss of tails, but s1206 is the more extreme short-path edge and the only single-sided-feasibility test.
2. **Add s1082 as a 13th "extreme-tier sentinel"?** The only patient whose extreme tortuosity (window tortuosity 1.74 / 2.03, robust z +4.6 / +7.0) is corroborated on both sides; paths follow the vessels in the QA image; bend 6.1-6.4 mm; F_max about 75 / 88 mm. It is outside the +-3.5 bound, so I did not select it automatically. It would expose the most severe tortuosity failure mode before scale-up. Adding it costs one patient and is not part of the 12.
3. **Continuity**: s0912 can be replaced by s0058 (the previous typical patient) for a 1% coverage cost if you prefer to keep a previously used patient.
4. **Split**: pilot patients are used for calibration (hazard model, resolvability, ablations) and must not be held-out test patients. The patient split must be regenerated for the corrected pool with the approved pilot patients forced into train/validation (the v1 split is invalid).

## 14. Reproduction

`scripts/design_v3/`: `_extract_for_selection.py`, `_sel_common.py`, `_sel_step1_tables.py`, `_sel_step2_select.py`, `_sel_step3_final.py`, `_sel_step4_report.py`, `_sel_step5_extras.py`, `_sel_step6_sensitivity.py`, `_sel_step7_last.py`, `_sel_qa_images.py`, `anatomy_profile.py`.
Outputs in `outputs/cvpr_multipatient_design_v1/`: `selection_descriptors.json`, `selection_table_A_59_patients.csv`, `selection_table_paths.csv`, `selection_descriptor_ranges.json`, `selection_step3.json`, `selection_final.json`, `selection_cohort_detail.json`, `selection_sensitivity.json`, `selection_extreme_tier.json`, `selection_edge_reserves.json`, `selection_coverage.png`, `qa_cohort_a.png`, `qa_cohort_b.png`, `qa_unverified_extremes.png`.
