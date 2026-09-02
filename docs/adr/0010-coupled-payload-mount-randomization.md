# Couple payload mass to the CoM shift and inertia it physically causes

**Status:** accepted (2026-09-02). Supersedes the payload-DR half of WP2's plumbing
(`apply_payload_dr`, 2026-08-31), which sampled torso mass and torso CoM offset as two
**independent** events. It does not affect `base_com`, which keeps its own separate meaning.

The deployment rig is a weighted backpack mounted in front of the torso, so payload mass never
arrives without a CoM shift and an inertia change. The simulator was sampling them
independently — able to draw 12 kg with zero CoM shift, and unable to draw the configuration
the robot will actually be in. We replace the two independent events with one event that
samples a payload mass and a mount lever arm, then derives mass, CoM and rotational inertia
from them.

## Context

WP2 (2026-08-31) added torso payload DR by extending the existing `base_mass` event, alongside
the pre-existing `base_com` CoM randomization. Both are `mode="startup"` events on `torso_link`,
sampled independently. WP5's Bar B then measured `corr(commanded stride period, payload)` over
that DR and read **+0.089 / +0.268** across two seeds, failing its |r| >= 0.5 gate.

The owner supplied the missing physical context on 2026-09-02: the payload is a single
backpack at a fixed mount, and the payload-only null is independently corroborated on hardware
(removing the head changed nothing; the heavier-leg mismatch is negligible, consistent with
ADR-0009's inert leg mass). Mass and CoM shift are one physical fact, not two.

## Evidence

**The DR could not represent the deployment condition.** Torso mass 17.789 kg, and a pack of
mass `m` at lever arm `d` from the torso CoM shifts it by `m*d/(M+m)`. Measured on the robot,
`d = (+0.1625, 0, -0.2348) m`. At 12 kg that is `com_dx` **+0.066**, `com_dz` **-0.095** —
against a trained `base_com` support of **±0.05 m**. Only **4.8 kg** stays inside the support,
and the **vertical** axis binds first, not the forward one.

**The regression over an independent DR answers the wrong question.** Independent sampling
yields *partial* derivatives — the response to mass with CoM held fixed — which is a
counterfactual the hardware cannot realize. Measured directly along the coupled ray
(`--cf-e-col 0,1`, both seeds, physics pinned): `dT/dm` = **-0.00171 / +0.00378** at d=0.10 m
and **-0.00375 / +0.00370** at d=0.20 m, versus **+0.00033 / +0.00374** payload-only. Bar B
fails on every ray and the two seeds disagree in **sign**.

⚠ An additive recombination of the observational partials predicted `dT/dm` = -0.0086/-0.0055
and Bar B **-0.723/-0.567 (PASS)**. Direct measurement said FAIL. Two errors: observational
coefficients mix the read path with the proprioceptive path (seed 123 has `com_dx` t = -26.8 but
reads it only 17%), and the read-path response is **asymmetric 2.56x** (-1.010 s/m backward vs
-0.395 forward) so one linearization is invalid. **This is why the coupling must be in the DR
rather than recovered analytically afterwards.**

**mjlab was already warning us.** `dr.body_mass` emits a `UserWarning` on every payload run
(fired at Phase-1 `train_s42.log:163`): *"only appropriate when modelling a point mass added at
the COM"* — precisely the configuration the hardware never produces. The inertia it leaves
unchanged is the largest of the three effects: at 12 kg the pack adds **+0.81x roll, +1.43x
pitch, +1.48x yaw** of the torso's own inertia.

## Decision

One `mode="startup"` event on `torso_link` samples a **mounted payload** and derives everything:

    m   ~ U(0, 12) kg
    d_x ~ U(0.13, 0.19) m      d_y ~ U(-0.03, 0.03) m      d_z ~ U(-0.28, -0.18) m

- **CoM and inertia are derived, never sampled**, via the exact rank-1 pseudo-inertia update
  `J' = J + m [d;1][d;1]^T`, decomposed back to `body_mass` / `body_ipos` / `body_inertia` /
  `body_iquat`. Implemented locally, not via mjlab's underscore-private helpers.
- **`d` is randomized, not fixed.** Fixing it would collapse the payload block of `e` to one
  degree of freedom and make `com` a deterministic function of `m`, weakening the very question
  WP5 asks. The range covers the cable-cover ambiguity (`d_x` 0.1425-0.1865 across extreme
  readings of where the cover sits) and where weights sit inside the pack.
- **`base_com` is kept unchanged** — it means uncertainty in the *robot's own* CoM, a distinct
  claim that holds with no payload present. The payload event runs **last** and composes on the
  *current* `body_ipos`.
- **`apply_payload_dr` is replaced in place**; task ids are unchanged.
- **`e` stays ℝ⁵** — realized `(payload, com_dx, com_dy, com_dz, friction)`. `env_latent_e` is a
  pure readback and needs no change; the deploy contract (HL 94 -> 99, `φ` outputs 5) is untouched.
- **`--eval-payload-kg m` now means "the rig, loaded to m kg"** — it applies mass, CoM and
  inertia at the mean `d`, so eval physics is inside the training distribution and a Bar B sweep
  traverses the deployment ray by construction.

## Consequences

**`Operation.add` overwrites; it does not compose.** `dr/_types.py:94-99` sets
`uses_defaults=True`, so the engine writes `default + random`, and two `add` events on
`body_ipos` mean the second erases the first. `apply_payload_dr:352` records the stream order as
`base_mass` -> `foot_friction`, `encoder_bias`, `base_com`, `push_robot`, i.e. `base_com`
currently runs *after* the payload event. **The payload event must therefore be registered last
and write `body_ipos` itself.** Written the obvious way this fails silently: the CoM shift
vanishes, `env_latent_e` reports only the `base_com` residual, and every number still looks
plausible. Guarded by a startup assertion and a regression test with a reordering mutation.

**The payload's inertia is no longer inferable from `e`.** `d = (com_total - residual)*(M+m)/m`,
and the unknown ±0.05 `base_com` residual gives `d` an ambiguity of 0.124 m at 12 kg —
comparable to `d` itself. Accepted: `e` was already a chosen *subset* of the randomized
extrinsics (`encoder_bias` is randomized and omitted), so this is consistent with the existing
design rather than a new violation. It is recorded because the omitted quantity is now much
larger than before.

**Existing payload checkpoints are no longer reproducible against their own task id.** Replacing
in place means the two WP5 Phase-1 runs (`2026-09-01_20-25-30_..._s42`,
`2026-09-01_23-18-56_..._s123`) have `params/env.yaml` as the only record of what they trained
on. Provenance markers are written into both run dirs. Their **mechanism** findings survive the
change — read%, the `com_dx` read-path asymmetry, and the seed-dependence of which channel is
read are properties of the trained policies, not of the DR.

**The re-run is deferred and batched.** Phase 1 is not re-run now; it is re-run once, after A1a
itself is settled, so the cost is paid a single time. Until then WP5's Bar B verdict stands as a
measurement on a DR that cannot represent the rig — which is itself the finding.

**This commits the policy to one payload *family*.** A hip-mounted or rear-mounted pack is out
of distribution; covering it means widening `d`'s range, not restructuring. Worth noting that the
policy responds **2.56x more strongly** to backward CoM shifts than forward, so a rear mount
would elicit considerably more adaptation than the front mount now specified.

**Risk to watch in Phase 2.** Concentrating samples on the deployment ray makes the history
encoder's job easier, so H-adapt may score better than broader DR would justify. Check `φ`
against a held-out `d` outside the trained range before believing a Phase-2 pass.
