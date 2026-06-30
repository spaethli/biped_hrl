# A1 LL intrinsic reward may carry upper-body regularization (deviation from pure HIRO)

**Status:** accepted

## Context

A1's low level trains on the goal-distance intrinsic reward only
(`ll_task_reward_coef = 0`); the env's `variable_posture` and `action_rate_l2` terms,
which keep A0's arms calm, never reach the LL through the goal-only routing. As a result
the A1 LL drives the arms behind the back and twists the wrists continuously — not
deployable on the real robot. The thesis's primary metric is whether the hierarchy
closes the sim-to-real gap, so an undeployable LL blocks the headline result.

The supervisor explicitly authorised deviating from the canonical HIRO L2 intrinsic
(reshaping it, or adding terms that are not directly observable).

## Decision

The A1 LL intrinsic reward is allowed to include privileged upper-body regularization.
We stage it:

1. **Now (minimal blast radius):** add a *negative* upper-body deviation penalty plus an
   action-rate penalty to `r_lo` in `HierarchicalRunner`, keeping the intrinsic
   all-negative so the `fell_over = time_out` truncation and existing tuning stay valid.
   Behavioural target: arms like A0's (moderate, near-default, low-jitter).
2. **Follow-up (separately motivated):** switch the whole intrinsic to an all-positive
   exp form `Σ_c w_c · exp(-‖V*_c - s_c‖²/σ_c²)`, which removes the suicide attractor and
   lets `fell_over` revert to a true terminal (matching A0) — run as its own experiment,
   not bundled with the arm fix, so each change stays independently attributable.

## Consequences

- The regularizer is added uniformly enough that it *converges* A1's locomotion-shaping
  floor toward A0's, so it does **not** worsen the RQ2 A0-vs-A1 comparison; the only
  intended remaining difference is the hierarchy (goal-tracking vs command-tracking).
- The documented "pure HIRO, `ll_task_reward_coef = 0`" framing in the memory/docs is no
  longer the whole story — the A1 LL intrinsic is now a shaped, privileged signal.
- The exp follow-up, if adopted, is the point at which the A1 termination mode can be
  re-unified with A0 (one fewer confound).
