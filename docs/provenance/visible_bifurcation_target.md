# Visible-bifurcation target: validation (s0011)

Decision: **TARGET_OBSERVABILITY_BLOCKED** (also reachability-blocked and containment-unvalidated).
The GR00T schema, language string and `target_contract.py` were NOT changed.
Evidence: `outputs/cvpr_visible_bifurcation_v1/` (`anatomy_capacity.json`, `landing_zone.json`,
`visibility_metrics.json`, `visibility/*_sheet.png`, `FINAL_VISIBLE_BIFURCATION_VALIDATION.json`).

## Anatomy (derived from the centerline graph and verified against TotalSegmentator labels)

- Entry: centerline node nearest the portal (an endpoint, 0.0 mm). First branch point: the only degree>=3 cluster in the graph, a Y with one inbound arm and two daughters.
- It is the aortic bifurcation: the aorta label starts at z = 161 mm, the iliac labels end at z = 162.5 mm, the cluster is at z = 158.5 mm. The arterial mask is aorta + iliac arteries only (aorta 56,939 + iliacs 7,798 of 65,072 voxels).
- CT-frame coordinate of the bifurcation (cluster centroid): (-6.04, 174.96, 158.5) mm. Cluster entry on the portal arm: 183.6 mm of path length; centroid 186.4 mm.
- Portal-to-bifurcation path length: 186.4 mm smoothed (214.7 mm raw skeleton, 15 % too long); Euclidean 182.3 mm; tortuosity 1.02.
- Local geometry: lumen radius flares from 5.4 mm to 7.6 mm over the last 20 mm before the cluster; portal arm radius 5.4 mm, aorta 6.8 mm, other iliac 4.9 mm; daughters 131.5° apart; the aorta continues at 42.3° deflection, the other iliac at 94.3°.
- Relative to the current catheter tip (s = 73.5 mm, CT (40.0, 186.2, 60.4)): bifurcation is 109.0 mm away (Euclid), landing-zone centre 99.7 mm, 103.6 mm of feed.

## Landing zone (extremity coordinate = projected arclength of polyline end + 3 mm capsule radius)

| Quantity | Value | Derivation |
|---|---|---|
| split reference `s_split` | 183.6 mm | portal-arm / junction-cluster contact (a slab scan of the mask never separated two lumens inside 2 local radii, so the cross-section split is not detectable; skeleton-defined, uncertainty about one voxel) |
| clearance `c` | 2.32 mm | max measured per-decision advance 1.77 mm + max measured stop glide 0.55 mm |
| far edge (overshoot boundary) | 181.27 mm | `s_split - c` |
| window | [178.87, 181.27] mm | width 2·ε, ε = 1.2 mm = V_MAX·dt (6.7 px at 0.36 mm/px) |
| polyline-tip target centre | 177.07 mm | window centre minus capsule radius |
| overshoot | extremity > 181.27 mm | physical, from `body_q` |
| timeout | 95 decisions (9.43 s) | `(feed - 2 r_c)/V_MAX + 2 r_c/V_SLOW + K·dt`; all-CRUISE minimum 8.63 s |
| LUMEN_VIOLATION | axis SDF excess > 1.5 mm | half the 3 mm collision-mesh voxel |
| PHYSICS_FAILURE | non-finite, speed > 2 m/s, root moved > 1e-6 m | unchanged |

ε = 1.2 mm is the smallest tolerance a velocity policy can hit (no step-over); a larger tolerance needs a clinical or scientific justification that is not derivable from the system.

## Reachability

Model: 6 bodies of 30 mm, catheter radius 3 mm, 3 active at the start, so 90 mm of further feed activates every body. Available tip arclength 163.5 mm (measured: last body activates at 164.3 mm).
Shortfall: 14.8 mm to the far edge, 20.1 mm to the split, 22.9 mm to the centroid. Minimum mechanical change: one more 30 mm segment (7 bodies, 193.5 mm).
Beyond the validated 36 mm of feed (tip s = 109.5 mm) the model leaves the lumen: first outside sample at s = 130.9 mm (frame 292, 4.9 s), axis violation > 1.5 mm at s = 145.3 mm (6.2 s), max violation 5.3 mm, tip up to 10.3 mm from the centerline. The model also has no cap on feed once every body is active. A longer catheter alone does not make the target reachable.

## Visibility (catheter-free policy view, Docker DiffDRR, no overlay)

9 views (3 iso points x AP / LAO45 / RAO45) with a counterfactual in which the lumen voxels are replaced by the local tissue median.

| View | lumen delta (gray levels) | local texture sd | CNR | matched-filter z (silhouette) |
|---|---|---|---|---|
| carina AP | 6.5 | 3.9 | 1.7 | 2.1 (window 1.9) |
| carina LAO45 (current projection) | 3.7 | 1.2 | 3.0 | -0.9 |
| carina RAO45 | 2.9 | 1.0 | 3.0 | 1.0 |
| landing AP | 5.7 | 3.2 | 1.8 | 2.1 (window 2.3) |

A-priori criterion (set before looking at the results): visible if the best view has silhouette z >= 4 and the carina window z >= 4. Best observed z = 2.5. The lumen adds 3–7 of 255 gray levels; GR00T's processing (crop, resize to 256, brightness 0.3 / contrast 0.4 jitter) is larger than that signal.
**TARGET_VISIBLE_IN_FLUOROSCOPY = NO.**

## Remaining items

- Uniqueness: one branch point in the twin's arterial graph and no side lobes in the mask before it. Not established against the image: the CT also contains vessels absent from the twin (internal iliac and others), so a visually plausible earlier bifurcation may exist.
- Insertion-only: topologically yes (no branch point before the zone); physically not established (containment fails at s >= 131 mm in the current model).
- Vision necessity: for s0011 alone, `inserted_length_m` solves the task (stop at about 103.6 mm of feed). The window is 2.4 mm, 2.3 % of the feed, so a fixed depth only transfers between patients whose path lengths agree within about 1.2 mm. Requires multi-patient confirmation.
- Proposed language (not applied): "Advance the catheter tip to the first distal vessel bifurcation while remaining inside the lumen."
