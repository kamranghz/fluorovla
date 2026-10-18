# CVPR 2027 Target Contract v1 (insertion-only catheter task)

Status: **defined and validated for s0011; NOT ready for learning** (see §8).
Implementation: `scripts/medphysai/target_contract.py` (constants, path selection, `evaluate`).
Evidence: `outputs/cvpr_target_contract_v1/` (`target_contract_validation.json`, `FINAL_CVPR_TARGET_VALIDATION.json`).

Terminology: **task-construction privilege** = anatomy and physical state used to build the task and to
score episodes (never a policy input). **Policy input** = fluoro frame, `state[0..1]`, language.

## 1. Contract

| Field | Value |
|---|---|
| TARGET_ID | `pre_junction_path_waypoint_1seg` |
| TARGET_SELECTION_RULE | (1) entry = centerline node nearest the portal; (2) path = unique graph path from entry to the nearest degree>=3 node, junction excluded (topology only: no hardcoded tangent, node index or CT coordinate); (3) Gaussian-smooth the path (sigma = 3 mm = collision-mesh voxel) and re-parameterise by arc length s; (4) `s_tip0` = projection of the physical tip (distal end of the last active capsule, accepted `body_q`) onto the path at episode start; (5) `s_T = s_tip0 + D_T`, `D_T = L_SEG = 30 mm` |
| TARGET_COORDINATE_FRAME | construction in CT-absolute mm; scoring coordinate = arc length `s` (mm) on the smoothed path; solver-local = CT − `CT_OFFSET_M` |
| TARGET_GEOMETRIC_OBJECT | a centerline point with a success slab `|s − s_T| <= eps` along the path |
| TARGET_REACHABILITY_RULE | eligible only if `s_T + eps <= s_tip0 + validated_feed` (36 mm) and no degree>=3 node lies before `s_T + eps` (insertion-only; no steering needed) |
| TARGET_SUCCESS_METRIC | physical tip arc length: `|s_tip − s_T| <= eps` AND tip stationary (`|ds| < 0.3 mm` per decision) for `K = 3` consecutive decision steps |
| TARGET_SUCCESS_TOLERANCE | `eps = 1.2 mm` |
| TARGET_OVERSHOOT_DEFINITION | `s_tip > s_T + eps` at any decision step (terminal; the action range `[0, 0.012]` has no retraction, so it is unrecoverable) |
| TARGET_TIMEOUT | 30 decision steps = 3.0 s (180 physics frames) |
| TARGET_SAFETY_FAILURE | LUMEN_VIOLATION: any catheter-axis SDF sample outside the lumen by more than 1.5 mm; PHYSICS_FAILURE: non-finite state, `|v| > 2 m/s`, or root moved > 1e-6 m |
| TARGET_TERMINATION_RULE | first of: PHYSICS_FAILURE > LUMEN_VIOLATION > OVERSHOOT > SUCCESS; at timeout: STOPPED_SHORT if `s_tip < s_T − eps` and stationary, otherwise TIMEOUT |

Success is computed only from physical state (tip position from `body_q`, SDF samples, solver finiteness).
It never uses commanded insertion, the action history or a target label.

## 2. Where each constant comes from

| Constant | Value | Derivation |
|---|---|---|
| `eps` | 1.2 mm | `V_MAX * decision_dt = 0.012 m/s * 0.1 s`. Largest nominal one-decision advance; measured max per-step advance 1.77 mm (CRUISE) and 1.33 mm (SLOW) is < the window width 2·eps = 2.4 mm, so a velocity policy cannot step over the window |
| stop residual | <= 0.55 mm | measured Euclidean tip drift after CRUISE→HOLD (0.02–0.55 mm over 4 stop tests) < eps |
| image scale | 0.360 mm/px | measured along the CRUISE path; eps = 3.3 px at 320 px (about 2.8 px after the GR00T crop+resize to 256) |
| `dwell step` | 0.3 mm | half the smallest non-zero commanded advance (SLOW: 0.6 mm per step). HOLD advance is <= 0.008 mm per step |
| `K` | 3 | smallest integer with `K*dt` > window crossing time at V_MAX (`2*eps/V_MAX = 0.2 s`) |
| timeout | 3.0 s | the validated CRUISE episode (36 mm of feed). Minimum time to target at CRUISE is 2.5 s, so slack is 0.5 s |
| `D_T` | 30 mm | `L_SEG`: forces the material-activation event (validated at frame 149) and leaves 6 mm of overshoot room inside the validated envelope |
| lumen delta | 1.5 mm | half the 3 mm vessel collision-mesh voxel (surface location uncertainty) |
| smoothing sigma | 3 mm | skeleton arclength over-estimates: 211.5 mm raw vs 182.5 mm smoothed; smoothed Δs agrees with the tip's Euclidean travel to 0.3–1.6 % |

## 3. s0011 instance

Entry node 0 (an endpoint, 0.0 mm from the portal). The first junction is the degree-3 cluster {222, 223, 224}.
Smoothed entry→junction length 182.5 mm. `s_tip0 = 73.5 mm`, `s_T = 103.5 mm`
(CT mm [28.8, 173.6, 85.0], local lumen radius 4.07 mm). Inside the validated envelope (`s <= 109.5 mm`) with 6.0 mm overshoot room.

## 4. The draft rule is not valid

The draft rule ("last node before the first branch") puts the target at 182.5 mm of arc length, which is
108.9 mm of feed (3.03× the validated 36 mm, 9.1 s of CRUISE, 4 activations vs 1 validated).
The catheter has 6 bodies; with all deployed the tip reaches 163.5 mm, 19 mm short of that node.
The draft fails its own reachability test (`6·L_SEG = 180 mm` against the skeleton arclength 211.5 mm).

## 5. Validation (fresh physics, common start, 3 s, fresh reset per run)

| Action | Δs smoothed (mm) | Δ tip Euclid (mm) | commanded feed (mm) |
|---|---|---|---|
| HOLD | −0.06 | 0.21 | 0 |
| SLOW | 18.24 | 17.95 | 18.0 |
| CRUISE | 36.10 | 36.00 | 36.0 |

HOLD < SLOW < CRUISE from physical state; CRUISE repeat identical (0.0 difference). Outside fraction 0, lumen violation 0 (every substep), root delta 0, finite.

Stop-timing sweep (CRUISE for n decisions then HOLD): n = 24 → SUCCESS (−0.50 mm), n = 25 → SUCCESS (+0.89 mm); n ≤ 23 → STOPPED_SHORT; n ≥ 26 → OVERSHOOT. The target is reachable, with a success set of two stop timings.

## 6. Cross-patient rule

The rule uses graph topology, the physical tip and catheter constants only. A patient is eligible iff the
entry node is a graph endpoint within one voxel of the portal (otherwise the portal axis is needed as a
tie-break and the case is reviewed), the first junction lies beyond `s_T + eps`, and `s_T + eps` is inside the validated feed envelope.
Only s0011 has been instantiated. 102 TotalSegmentator cases exist but have not been processed.

## 7. State semantics used by the contract

`inserted_length_m` is the integral of the commanded velocity (`L_total += v·dt`) since episode start, parsed
from `accepted_state_id`. It is not a physical measurement. Physical tip travel differs from it by 0.0–0.9 mm.

## 8. Why the task is not ready for learning

1. The +30 mm waypoint is fully determined by `state[0]`: stopping when `inserted_length_m` is about 0.029–0.030
   succeeds with no vision. A vision ablation would show no effect.
2. No anatomical landmark exists in the reachable range. The segment from 73.5 to 109.5 mm is a smooth
   tube (radius 3.0–4.2 mm, turns <= 9° per 10 mm). The first landmark is out of reach for the 6-body catheter.
3. The bridge's C-arm framing (fixed at the first observation's catheter centroid) leaves the tip in the image for 12 of 30
   CRUISE observations and the target is never in the image (target at about (−42, 87) px).
4. Lumen contrast in the DRR is weak (mean µ 0.0055 inside vs 0.0045 in the surrounding shell, 1.24×; catheter µ = 0.8).

Visibility class of the current target: **D** (hidden privileged coordinate; recoverable from proprioceptive state only).
