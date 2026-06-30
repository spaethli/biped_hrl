# A1a: give the HL a job beyond velocity passthrough (cost-of-transport objective via a cadence reference)

**Status:** proposed

## Context

The thesis research question is **does the hierarchy improve sim-to-real transfer?** The
supervisors (2026-06-18) called A1 a *velocity relay*: the high level forwards the twist
command on a coarser timescale and adds nothing, so even a clean A0-vs-A1 transfer study
would likely return "no difference" because no hierarchical structure is being exploited.
A relay HL makes the RQ unanswerable. The HL needs a job the flat command structurally
cannot do (`doc/hrl/hierarchy_benefit_roadmap.md`, Track A, Axis 1).

Transfer will be measured with a **sim-to-sim / OOD proxy** during development (train
Narrow DR, evaluate Wide DR / pushes / terrain; A1-vs-A0 fall-rate and tracking under the
held-out gap) and confirmed on the **real H1-2 as a capstone**.

A new goal dimension and a new HL reward are not two options: they are inseparable. Under
the current `hl_reward_mode=tracking` reward the HL has zero gradient to set any new gait
dimension (every cadence tracks velocity equally well), so a richer goal space is empty
without a paired richer objective, and a richer objective has nothing to act through
without a new lever.

## Decision

Add an **energy / cost-of-transport objective to the HL only**, acting through a **cadence
gait reference** the flat command cannot express.

- **Objective.** HL reward = tracking + (negative) **cost of transport**, where
  `CoT = window mechanical energy / window distance`, mechanical power
  `P = Σ_j |τ_j · q̇_j|`. CoT is gated to commanded motion via the existing
  `command_threshold` idiom (the same gate `feet_gait` / `feet_swing_height` already use),
  which removes the near-zero-command blow-up of `energy/distance`. Standing when commanded
  to stand is handled by the tracking term, not CoT.
- **Decoupling (the hierarchy claim).** CoT lives in the **HL reward only**; the LL stays a
  pure tracker. A flat `A0+energy` policy must trade tracking against efficiency at one
  timescale and regresses tracking; A1a keeps tracking as the LL's sole job and trims
  efficiency at the HL above it. The claim is: **A1a matches A0's tracking at lower CoT,
  where `A0+energy` cannot.**
- **Lever = cadence, routed via R2 (not the goal space).** The HL emits a step **period**
  as one extra action dimension. The LL **observes** the open-loop phase clock
  (`sin φ, cos φ` from the commanded period) and is **trained** with the existing
  `feet_gait` reward keyed to that period (added to the LL intrinsic, like the posture /
  action-rate terms in ADR-0002). Cadence is a *gait reference* the LL entrains to, not a
  goal-space L2 component. It is deploy-clean: the phase is a function of the commanded
  period and time, and `feet_gait` is a sim-only training reward, so no footfall estimator
  is required on hardware.
- **Two-channel HL->LL interface.** The HL directive is now (1) the goal `V*` (state
  targets reached by L2) and (2) the gait reference (cadence, entrained). See `CONTEXT.md`.

## Considered options

- **Denominator `P / v_commanded` with explicit 0-vs-nonzero handling** — rejected in
  favour of `energy / distance` gated by `command_threshold`, which reuses an existing
  codebase idiom and needs no bespoke standstill case.
- **Energy term in the LL reward** — rejected: it would destroy the decoupling claim and
  pull the LL away from "A1 LL ≈ A0 tracker".
- **Cadence as a goal-space L2 component (R1)** — rejected: needs a "current cadence"
  extractor (footfall timing → Axis-2 hardware gate), and cadence is a frequency-over-time,
  not a clean instantaneous state slice.
- **Cadence + step length as two goal dimensions** — rejected: at a fixed speed
  `v = cadence × step_length`, so they are one degree of freedom reparameterized;
  commanding both over-determines the gait.
- **Stance width as a second, genuinely independent lever** — deferred: it is orthogonal to
  `v = f·L`, but no measured deficiency yet justifies the larger HL action space.
- **Raw energy (not normalized)** — rejected: a stand-still attractor (zero velocity ≈ zero
  power), the same penalty-domination failure that forced `hl_reward_mode=task` off.

## Consequences

- **Shape changes.** HL action space `+1` (period); LL obs `+` phase clock; HL reward
  `+= -w·CoT`; LL intrinsic `+= feet_gait(commanded period)`.
- **Phase continuity.** `feet_gait` derives phase from absolute elapsed time
  (`global_phase = episode_length·dt / period`), so a mid-episode period change jumps the
  phase. Phase must be integrated incrementally (`phase += dt / period`) once the period is
  HL-commanded.
- **Control set for the claim.** plain A0, `A0+energy` (flat, fixed cadence), and A1a
  (CoT-in-HL + cadence). The hierarchy benefit is the gap between A1a and `A0+energy` at
  matched tracking, plus any OOD-proxy survival advantage.
- **RQ2.** The env reward is untouched (CoT is HL-internal; cadence entrainment is in the
  LL *intrinsic*, not the env reward), so the A0-vs-A1 env comparison is not confounded in
  the sense of `.claude/docs/experiment-design.md`. The deliberate deviation is that A1's LL
  now also entrains to a commanded cadence rather than behaving exactly like A0; this is the
  intended hierarchy mechanism, not a confound.
- **Roadmap fit.** The two-channel HL->LL interface is the stable seam across A1a → A2
  (A-RMA on the LL, interface unchanged) → A3 (a gait library produces the COM trajectory ≈
  goal and the footfall pattern ≈ cadence reference, plugging into the same socket).
- **Next stages (CLAUDE.md workflow).** This ADR is the feature definition. A precise
  implementation spec and a test plan (smoke + success criteria on the OOD proxy and the
  A0/A0+energy/A1a scorecard) are required before coding.
