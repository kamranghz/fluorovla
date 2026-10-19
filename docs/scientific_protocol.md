# Scientific protocol (summary)

The binding, pre-registered document is `experiments/protocols/experimental_protocol_v1.md`. This page summarizes what the code implements today.

## Task and success

Insertion-only catheter navigation: action = insertion velocity in {0, 6, 12} mm/s, decision period 0.1 s (6 physics frames at 60 Hz). A rollout succeeds iff,
within the timeout `T_MAX`, the tip extremity dwells in the landing window (width 2 x EPS = 2.4 mm, centre S_contact - 3.52 mm) for 3 decisions with no overshoot, lumen
violation <= 1.5 mm, finite physics and root drift <= 1e-6 m. Terminal priority: PHYSICS_FAILURE > LUMEN_VIOLATION > OVERSHOOT > SUCCESS > STOPPED_SHORT / TIMEOUT.

## Anatomy and admission

- Patient twin from CT; access path per iliac side from the centerline graph (entry endpoint -> first 3-arm junction cluster); candidate path iff S_contact >= 150 mm.
- Anatomical candidate pool: **65 patients** (twin-verified; `artifacts/provenance/cohort_lock.json`).
- Development cohort (locked, 12) and stress sentinel: see the lock file. Development patients are permanently excluded from unseen validation/test.
- Physics admission (Teacher reference, legacy rule): start conditions F at 0 / 50 / 100 % of [60 mm, min(100 mm, S_contact - 82.2 mm)]; a side is admitted iff the
  Teacher succeeds at all three; a patient is admitted iff at least one side is admitted.

## Stages (specified in the protocol, status)

| stage | status |
|---|---|
| anatomy: twin verification, cohort selection and lock | done |
| physics mapping on the development cohort | completed on native Linux (72 episodes: 68 LUMEN_VIOLATION, 4 SUCCESS, 0 sides admitted); **not interpretable as anatomy infeasibility** until the initial-state audit is done (`docs/provenance/stage3a_initial_state_audit.md`) |
| initial-state / registration audit | tooling ready, native-Linux run pending (`docs/initial_state_audit_runbook.md`) |
| SAFETY test, DSA resolvability, LANDING ablation | not run |
| integrated pilot, admission sweep, K-fold population | not run |
| GR00T multi-patient training, closed-loop evaluation | not run |

## Claim scope

Closed-loop insertion-only navigation to an aorto-iliac bifurcation landing zone, on held-out admitted patient-specific anatomy, using physics simulation and
rendered fluoroscopic/DSA observations. No clinical, real-fluoroscopy, steering or external-validation claim.

## Known limitation recorded in the protocol

The anatomy admission excludes patients whose twin skeleton has no valid 3-arm bifurcation under the current patient-independent contract (s1395: double junction plus
a 12.8 mm carina branch). This is a selection effect toward simpler carinas and must be reported.
