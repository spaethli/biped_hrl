# Record Bar B as FAILED with its mechanism, rather than re-stating it against a component that passes

**Status:** accepted (2026-09-03, owner's decision). Adjudicates the WP5 gate defined in the
H-adapt brief. Does not change any code; it fixes what the thesis reports and constrains how
Phase 2's gate must be written.

Bar B pre-registered `corr(commanded stride period, payload) >= 0.5`. Phase 1 measured
**+0.089 / +0.268** across two seeds, and it fails on the deployment-coupled ray too, with the
seeds disagreeing in sign. The same experiment showed the mechanism the gate was meant to detect
is *present*: a counterfactual that falsifies one latent column while the physics stays as
sampled moves the commanded period by 15+ SE. We record the **FAIL**, together with the reason
the gate could not have detected that mechanism, instead of restating it against `com_dx` or
`friction`, which would pass.

## Context

The gate was written before the coupling between payload and CoM was understood, and before any
Phase-1 policy existed. Two independent reasons emerged for why it reads near zero:

1. **Physics.** Stride period for an inverted-pendulum walker is set by pendulum *geometry* --
   CoM height and leg length -- not total mass: 12 kg added at the torso scales gravitational and
   inertial terms together and largely cancels out of the natural frequency. Shifting the CoM
   *position* changes that geometry. A policy that commands cadence off `com_dx`/`friction` and
   ignores `payload_kg` is the physically literate one. Bar B named the one component of `e` with
   almost no cadence response to give.
2. **Statistics.** The gate is **under-powered by construction**: its denominator carries the
   other four latent components' variation as noise. On seed 123 the HL demonstrably *reads*
   payload (counterfactual 9.7 SE, dT +0.041 s) and the gate still reads +0.268.

Re-stating the gate against `com_dx` or `friction` would make it pass -- `com_dx` scores -0.851
and `friction` +0.684 on a Bar-B-shaped statistic scaled to each component's own DR spread.

## Considered options

**A. Re-state Bar B against `com_dx`/`friction`.** Rejected. The component would have been chosen
*after* seeing which one passes. That is the shape of a result nobody outside the project can
trust, and it discards the most transferable finding: that a correlation gate on one component
of a multi-component latent can fail while the mechanism is present.

**B. Widen it to "any component of `e`".** Rejected for the same reason, with an added defect: an
"any of five" gate has an inflated false-positive rate that was never budgeted for, so its 0.5
threshold would no longer mean what it was set to mean.

**C. Record the FAIL with its mechanism.** Chosen.

## Decision

Bar B is reported as **FAILED**, with (a) the measured numbers on both seeds and both rays,
(b) the physical reason payload is the wrong component, (c) the statistical reason the gate is
under-powered, and (d) the counterfactual evidence that the underlying claim -- the high level
uses the privileged latent -- nonetheless holds.

The counterfactual result is reported as what it is: **evidence for the mechanism, not a
substitute gate.** It was not pre-registered, and this ADR does not retroactively promote it.

**Constraint on Phase 2.** Its gate is written fresh and pre-registered *before* the estimator is
fit, and it must **partial the other latent components out when it is written**, not afterwards.
The Phase-1 partial and counterfactual-slope estimators (+0.187/+0.485 and +0.093/+0.595) stay
labelled post-hoc diagnostics and do not become Phase 2's gate by default.

## Consequences

**A negative is the headline WP5 result, and it is publishable.** The brief anticipated this
("a FAIL is a publishable result here -- report it with the mechanism, per the M8 precedent").
Two transferable findings come out of it: a correlation gate on one component of a
multi-component latent must partial the others out, and the *partial* derivative an independent
DR yields is not the *directional* derivative the hardware realizes.

**The thesis carries a failed pre-registered gate alongside a positive mechanism result.** That
is more work to write than a clean pass, and it is the honest account. This is the third
occurrence of the pre-registration pattern already recorded in the project, so the framing is
established rather than novel.

**Bar B is not re-run under ADR-0010's coupled DR to try to rescue it.** The deferred Phase-1
re-run will re-measure it, but the verdict recorded here stands on the evidence as taken; a
better number under a better DR would be a new result, reported as such, not a correction to
this one.
