# A1a: give the HL a job beyond velocity passthrough (cost-of-transport objective via a cadence reference)

**Status:** accepted (premise validated 2026-07-02; S1c machinery implemented + tested
2026-07-02: `hl_cadence_source="hl"` gives the TD3 HL the period as +1 action dim, CoT enters
the HL window reward via `hl_cot_coef` — see `doc/hrl/A1a_plan.md`; S3 = first learned-HL run)

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
  `CoT = window mechanical energy / (actual walked distance + floor)`, walked distance
  `Σ ‖v_xy‖·dt` and mechanical power `P = Σ_j |qfrc_actuator_j · joint_vel_j|`. CoT is
  *engaged* only when the commanded linear speed exceeds `command_threshold` (0.1), so
  standstill / pure-spin commands don't trigger it (tracking handles those). The denominator
  uses *actual* (not commanded) distance, so spending energy while going nowhere is correctly
  expensive; a small floor keeps it finite when the robot is genuinely stuck. The earlier
  `P / v_commanded` and commanded-distance variants are rejected (a commanded denominator
  rewards standing still while under command).
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
- **Next stages (CLAUDE.md workflow).** This ADR is the feature definition; the validation
  plan is below and tracked operationally in `doc/hrl/A1a_plan.md`. Implementation follows on
  approval.

## Validation / test plan

Defined up front (run after implementing, report honestly). Smokes use `WANDB_MODE=disabled`
and delete local logs; comparisons hold `num_envs` fixed and use >=2 seeds.

**Prerequisite metrics** added to the `play.py` benchmark `[BENCH]` json: `mech_power_w`
(`Σ|qfrc_actuator·joint_vel|`), `cot` (Σenergy / Σwalked-distance, gated by commanded linear
speed > 0.1, training floor), `stride_period_s` (measured same-foot touchdown interval = the
*achieved* cadence).

- **S0 precondition (no training).** A0's natural cadence + CoT across commanded speeds, and
  `cot` sensitivity to a perturbed gait period within A0's walkable band. *Go/no-go:* if `cot`
  is flat in period the lever is inert, stop. Sets the initial `cadence_period_range`. (The
  definitive `CoT(period)` curve needs S2's cadence-capable LL; S0 is the cheap pre-check and
  the metric prototype.)
- **S1 smoke / backward-compat.** `hl_cadence=False, hl_cot_coef=0` is byte-identical to
  current A1; phase continuity on a mid-window period change; `mdp.phase` / `feet_gait` consume
  `hrl_phase`; CoT gate/floor numerics sane (standing-while-commanded high, gated-off exactly 0,
  no NaN); HL action dim = `goal_dim + 1` only when enabled.
- **S2 LL entrains.** Oracle/random HL sweeping the commanded period: achieved `stride_period_s`
  tracks the command across the band at no worse velocity tracking than current A1.
- **S3 learned HL finds the minimum (frozen LL).** `freeze_ll_path` + TD3 HL + `hl_cot_coef>0`:
  the HL drives `cot` below the fixed-0.6 baseline at A1-level tracking; the converged period
  matches the `CoT(period)` minimum and is speed-dependent.
- **S4 comparison (the claim).** A0 / A0+energy / A1 / A1a, >=2 seeds. *Success:* A1a `cot` < A0
  `cot` at A0-level tracking and fall_rate, **and** A0+energy regresses tracking to buy its CoT
  cut. *Honest disconfirmer:* if A0+energy matches A1a on both, the hierarchy did not help for
  in-sim efficiency, and we report that.
- **S5 weight guard.** Sweep `hl_cot_coef`; fall_rate must stay flat (rising fall_rate = CoT
  beating tracking = the suicide attractor; cap the weight below that point).
- **S6 OOD proxy (secondary).** Narrow -> Wide DR / push / terrain. Exploratory: a large
  survival edge is not expected until A2 supplies dynamics adaptation; S4 is the load-bearing
  A1a result.

Two data-set knobs: `cadence_period_range` (from S0), `hl_cot_coef` (bounded by S5).
