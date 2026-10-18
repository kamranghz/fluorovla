# DSA / roadmap observation: prospective protocol (declared BEFORE any render of this experiment)

Scope: target observability only. Target, landing zone, tolerance, overshoot, timeout and safety definitions are unchanged.
No physics, catheter, reachability, containment, teacher, dataset or training work.
The 2000-HU raw-contrast result of the previous task is `SENSITIVITY_RESULT_ONLY` and is not used here.

## 1. Image formation (authoritative i4h DiffDRR container, raw Beer-Lambert transmission, i0 = 1)

Notation: T = rendered transmission, A = -ln T (line integral of mu along the ray).

| Name | Definition |
|---|---|
| MASK | T of the native patient volume, no contrast, no catheter |
| CONTRAST | T of the volume in which arterial-lumen voxels carry contrast: mu = mu_w (1 + 1000/1000) = 0.010 /mm, mu_w = 0.005 /mm; no catheter |
| DSA | R = A_contrast - A_mask (>= 0) |
| LIVE_k | T of the native volume plus the catheter at physics state k, no contrast |
| LIVE subtraction | S_k = A_live_k - A_mask (the catheter alone, anatomy cancels) |
| ROADMAP (candidate policy image) | D_k = R + S_k |
| CATHETER+CONTRAST (check only) | T of the contrast volume plus the catheter at state k; additivity check: A_cc_k - A_mask must equal D_k |

Segmentation is used only to place contrast in the volume. No marker, outline, centerline or annotation is ever drawn.
Meaning of the roadmap: R is a stored contrast run (acquired once, before navigation, at the same C-arm geometry), S_k is the live subtracted fluoroscopy frame; their sum is what a digital roadmap with live catheter shows. The same vessel and catheter polarity (both bright = more attenuation) is kept; nothing is drawn.

Display (fixed constants, identical for every image; no per-image normalization):

    gray = clip( 255 * asinh(D / D0) / asinh(D_ref / D0), 0, 255 ),  D0 = 0.05,  D_ref = 4.8

- D0 = 0.05 = (0.010 - 0.005) /mm x 10 mm: the contrast line integral of a nominal 10 mm lumen (physics only, not fitted to data).
- D_ref = 4.8 = 2 x 3 mm (catheter diameter) x 0.8 /mm (catheter mu): the largest catheter line integral.
- asinh is the standard log-like compression for subtraction images; it keeps the vessel (D ~ 0.05-0.15, gray about 17-27 %) visible next to the catheter (up to 255).

Pure DSA (R only, no catheter) is reported with the same display for reference.

## 2. Contrast parameter

1000 HU-equivalent, chosen before this test (the previously declared primary). No other level is evaluated here.

## 3. Views, states, catheter

- Families: `roty` (the project's AP / LAO45 / RAO45 = rotations 0, +45, -45 deg about y: the views named in the task) and `clin` (ray perpendicular to the vessel axis, declared in the previous task): all views of both families are reported.
- Iso points as before: carina, landing_centre, tip_initial (catheter-free) and path_mid (tip of the final CRUISE state and the carina both in view; catheter-present).
- Catheter states: obs 0, 10, 20, 29 of the fresh CRUISE physics run `cvpr_target_contract_v1/drr_bridge_audit/cruise/render_jobs.npz` (accepted body_q polylines; radius 3 mm, mu 0.8 /mm). The physics is not modified.

## 4. Predeclared criteria

TARGET_VISIBLE = YES iff, on the **catheter-present ROADMAP image of state 29**, some view has silhouette matched-filter z >= 4 AND carina-window z >= 4 (same functions, 400-shift null, min/max shift 8/70 px, 128 px window; per-image seed = crc32 of the image name). It must also hold for the catheter-free ROADMAP (= pure DSA). Multiplicity disclosure: 6 catheter-present views (2 families x 3 projections) are tested; all are reported.

CATHETER_VISIBLE = YES iff, in the best view, for each of the states 0, 10, 20, 29 whose true tip is inside the image:
(i) gray contrast of the catheter footprint over the background (pixels outside the vessel silhouette and the footprint) >= 100 gray levels; and
(ii) a mask-free tip estimate (pixels with gray >= 128, connected component touching the image border = proximal end, tip = its farthest pixel from the border contact) is within 10 px (about 3.6 mm) of the true projected tip; and
(iii) the estimated tip position moves monotonically with the state index.
Footprint, silhouette and true tip are used for scoring only.

## 5. Other reported quantities

Vessel signal (mean gray in the eroded silhouette minus ring), background texture sd (ring high-pass sd), dynamic range (p1-p99), saturation fraction (gray 255) and clipping at 0, vessel-silhouette IoU of a mask-free Otsu threshold of R (patient-specific morphology), additivity error, quantum-noise sensitivity (N0 = 5000 photons/pixel, MASK averaged over 8 frames, CONTRAST averaged over 8 frames, LIVE single frame; supplementary, not part of the criterion), cleanliness check of every policy PNG (320x320x3, R = G = B, no text chunks, no drawing calls in the writer), hashes of all images and image-formation sources.

## 6. Status mapping

PASS_DSA_TARGET_VISIBILITY_READY needs TARGET_VISIBLE = YES and CATHETER_VISIBLE = YES. Otherwise DSA_TARGET_VISIBILITY_BLOCKED (target criterion fails), DSA_CATHETER_VISIBILITY_BLOCKED (catheter criterion fails), DSA_IMAGE_FORMATION_INVALID (additivity or cleanliness check fails), RENDERING_IMPLEMENTATION_BLOCKED.
