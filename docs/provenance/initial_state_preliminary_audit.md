# Stage 3 preliminary initial-state audit (static, from existing local artifacts; nothing re-run)

Status: **PRELIMINARY. Local WSL2 results; not interpretable as anatomy infeasibility.** The full audit requested for the case that native-Linux reproduction (s0011 reference, one SUCCESS, one LUMEN_VIOLATION) passes and the violation rate persists is **deferred**. This file records what the existing artifacts already show, and the checklist for that audit.

## What the 51 completed local episodes show (47 LUMEN_VIOLATION, 4 SUCCESS)
- 41 of 47 violations occur within the first 5 decisions; the median violating episode lasts 1 decision (most patients violate at decision 0, i.e. in the first 0.1 s after the settle phase). Only one violation occurs after decision 15 (s0963_left_F2, decision 36).
- At decision 0 the tip is a median 7.2 mm from the centerline in violating episodes versus 2.1-3.8 mm in the 4 SUCCESS episodes (s0011_right_F0, s0963_left_F0, s0963_left_F1, s0963_right_F1).
- Straight-rod probe: in successes and failures alike a large part of the initial 73.5 mm straight rod is outside the lumen before settling (as in the legacy s0011 start); the settle phase pulls it in. Frame-0 penetration is therefore expected; what matters is whether the settled rod is inside.
- The pattern (instant failure at decision 0-1, dependence on patient rather than on start depth F, failure of every start condition for patients such as s1400, s1397, s1391, s1364, s1348) is what an initial-state / registration problem (hypothesis B) would look like, rather than a progressive physical limit (hypothesis A). It is **not proof**.
- s0011 in config mode: portal placed on the smoothed path start (1.17 mm inside the raw polyline) gave LUMEN_VIOLATION; portal on the raw polyline (as in the accepted episode) gave SUCCESS. A 1-mm placement change flipped the outcome, so the result is sensitive to placement convention.

## Differences between the s0011 initialization and the multi-patient generalization
1. s0011: portal = raw centerline node 0, frozen i4h tangent. Multi-patient: raw polyline point at s_p, tangent = 5-mm chord of the smoothed path. The straight rod then extends 73.5 mm along that tangent while the vessel curves.
2. Fixed 3 active bodies and 3 mm capsule radius; near the iliac cut end the s0011 lumen radius is 3.0-3.7 mm (almost no clearance).
3. Mesh: marching cubes with downsample 2 (3 mm voxels) for every patient, so the wall can be ~1.5 mm off, which equals the violation threshold.
4. SDF sign_scale fixed to 1.0 for every patient; checked only indirectly (phi < 0 at path points in most cases; 2 of 47 violating episodes have path points with phi > 0).

## Checklist for the deferred audit (no physics retuning)
- Frame 0 and after each settle phase: SDF phi at body centers and 5 axis samples per body, minimum wall clearance, number of bodies with phi > -R_c.
- Registration: distance of each active body to the twin path, portal-to-path distance, tangent versus local path direction at 5, 15, 30 and 73.5 mm, CT-offset round trip (solver-local -> CT -> volume index), vessel_mask value at the portal.
- Frame consistency: twin centerline points vs vessel_mask voxel centers vs mesh vertices (CT mm) for 3 patients.
- Strain, bend and twist at frame 0 and after settle: joint angles, maximal body speed during settle.
- Counterfactuals using only initialization choices (never physics constants): portal on raw vs smoothed path; tangent from 5/15/30-mm chords. This decides A vs B. Any change is an errata item.
