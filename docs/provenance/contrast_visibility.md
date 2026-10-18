# Contrast-enhanced visibility of the s0011 aortic bifurcation

Status: **TARGET_VISIBILITY_STILL_BLOCKED**. No physics, catheter, target, landing zone or GR00T schema change.
Evidence: `outputs/cvpr_visible_bifurcation_v1/` (`contrast/`, `contrast_visibility_metrics*.json`,
`contrast_visibility_summary*.json`, `contrast_fixedwindow_catheter*.json`, `FINAL_CONTRAST_VISIBILITY_VALIDATION.json`).

## Image formation (all in the authoritative i4h DiffDRR container)

- Volume: patient `mu_volume` (mu_water = 0.005 /mm, recovered from the preprocessing: mu = mu_w (1 + HU/1000)). For contrast variants the voxels of the arterial mask get mu = mu_w (1 + HU_c/1000), HU_c = 400 / 1000 / 2000. The mask only says where intravascular contrast exists; it never appears in an image.
- Renderer: Beer-Lambert, i0 = 1, 1 mm steps, raw transmission (normalize = False, invert = False). The pipeline's own display convention (min-max per image, then 1 − x) is applied on the host and reproduces the previous native images with 0.0 maximum difference.
- Detector 320 x 320, 0.6 mm pitch, SID 510 mm, SDD 1020 mm. Catheter: accepted-`body_q` polyline of the 09-12 CRUISE run (final observation), radius 3 mm, mu 0.8 /mm.
- DSA = ln T_native − ln T_contrast (the line integral of the added contrast), displayed min-max, bright = contrast.
- Primary level: 1000 HU-equivalent (declared before analysis). It is an assumption: monoenergetic, uniform filling, no flow/dilution, no energy dependence of iodine, no scatter.

## View conventions (correction)

The renderer's rotations in the project are about y only, and the vessels and catheter run along z. So the project "AP" (0°) looks down the vessel axis and "LAO45/RAO45" are 45° from it; they are not clinical views. This audit therefore tests two families:
1. `roty`: the previous nine views (0°, ±45° about y), the predeclared set.
2. `clin` (declared before rendering): ray perpendicular to the vessel axis. `AP_y` = (rx, ry, rz) = (90°, 0, 0); `OBL_p45/m45` = (90°, 0, ±45°). Left/right obliquity is not mapped to patient side.

## Results (catheter-free, noise-free; median / max over views)

| Condition | roty sil z | roty window z | clin sil z | clin window z |
|---|---|---|---|---|
| native | 0.85 / 2.72 | 0.29 / 2.26 | 1.18 / 1.79 | 1.01 / 2.68 |
| contrast 400 HU | 2.29 / 3.34 | 1.57 / 2.67 | 2.70 / 3.68 | 1.74 / 3.13 |
| contrast 1000 HU (primary) | 3.49 / 4.37 | 2.85 / 3.82 | 4.26 / 5.96 | 2.93 / 3.96 |
| contrast 2000 HU | 4.29 / 5.41 | 3.29 / 4.43 | 4.88 / 7.53 | 3.71 / 4.46 |
| DSA 1000 HU | 4.14 / 5.03 | 2.86 / 4.07 | 4.99 / 8.14 | 4.29 / 5.32 |

Predeclared criterion (a view with silhouette z >= 4 AND carina-window z >= 4, same test, null, seed): native not met; contrast 400 not met; **contrast 1000 not met** (best joint view 4.16 / 3.82 roty, 4.17 / 3.96 clin); contrast 2000 met; DSA met at all three levels in `clin`, and at 1000 HU only marginally in `roty` (4.10 / 4.07).

## Catheter present (the realistic policy image)

Best catheter-present silhouette z at 1000 HU: 4.31 (roty), 5.36 (clin); carina-window z only 1.8–3.0. At 2000 HU the window z is 3.2–3.9. The catheter's attenuation sets the per-image min-max window and compresses the anatomy. A fixed display window from the catheter-free frame does not fix this (window z 1.8–2.9 at 1000 HU, 3.1–3.9 at 2000 HU). DSA by static subtraction cancels a stationary catheter; live subtraction was not evaluated.

## Other checks

- Mask-free Y detector (sato + Otsu + skeleton; supporting only): DSA gives exactly one junction cluster in most carina-centred views, 14–29 px (5–10 mm) from the true carina; contrast fluoroscopy gives 7–17 spurious clusters; native 13–23.
- Quantum noise (N0 = 5000 photons/pixel, DSA mask averaged over 8 frames): contrast fluoroscopy and DSA silhouette z 1.5–5.8, carina-window z 0.1–4.0. The window criterion fails under noise.
- The test has limited dynamic range: it correlates a whole-tree silhouette with shifted copies of itself, so even noise-free DSA gives z of only 4–8 although the Y is obvious to the eye (`contact_sheet_policy_images_clin.png`). Not changed; reported as a limitation.
- Cleanliness: 84 policy PNGs are 320 x 320 x 3 gray, R = G = B, no text chunks; the policy-image writer has no drawing calls. Overlays exist only in the earlier analysis sheets.

## Scientific check

- Contrast makes the lumen physically present in the image: in `clin`, native silhouette z <= 1.8 versus 4–8 with contrast or DSA.
- With a contrast-filled lumen in the catheter-free image the Y is visible to a human and detectable by simple processing in DSA. In the catheter-present contrast fluoroscopy at the primary level, bifurcation localization does not meet the predeclared window criterion.
- Plausibility: contrast in a real workflow is a bolus with washout and dose limits, so continuously opacified live fluoroscopy is an idealization. DSA or a roadmap is the realistic route, and the roadmap policy image (live frame plus a stored vessel map) is not yet defined or evaluated.
