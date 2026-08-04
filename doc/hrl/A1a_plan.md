# A1a — HL enrichment plan (cost-of-transport via commanded cadence)

Design and current status for A1a, the "give the HL a real job" effort. Design
rationale and the locked feature spec live in
`docs/adr/0004-a1a-energy-cadence-hl-enrichment.md`; domain terms (Goal, Gait
reference, Cost of transport) in `CONTEXT.md`.

> **Exploratory history lives outside this repo (2026-07-28).** The chronological
> record behind this document — every run tried, what failed, root causes, dated
> session logs and result tables — was moved to the author's research knowledge
> base to keep this repo publishable. What remains here is the current design and
> status: the options in use and how the working solution is implemented.

## Current status (2026-07-24)

- **Co-training is the A1a line** (user's call 2026-07-13, reaffirmed since).
  The staged/frozen variant is parked.
- **Keeper lever: `arm4d` (`ll_energy_coef=0.05`), seed-verified.** The
  2026-07-23/24 combo batch replicated it on a second seed and tested six
  combinations against it; none beat it.
- **A1 tracking is not a deploy risk.** Every A1 run holds a sustained command
  at least as well as A0. The only surviving A1-vs-A0 gap is **energy**.
- **Open premise question:** a pinned constant-cadence clock (`fix0p8`) beats
  every HL-cadence policy on cost of transport at equal tracking (1 seed).
  Co-training stays the line regardless, since the hierarchy is the
  architecture's point rather than this one metric.
- **Known gap:** A1 is twitchier than A0. Root-caused to a small structural
  cost from the goal channel's 6.25 Hz step (worth ~4%) plus a large uniform
  floor concentrated in the arms, pointing at A0's `pose`/`variable_posture`
  term as the missing mirror. **Mirroring A0's other smoothness term
  (`joint_acc_l2`, `ll_joint_acc_coef`) was tested at 2 doses 2026-07-28 and is
  CLOSED as negative**: it does not move the arms share, makes `ub_arm_vel`
  worse, is non-monotonic on action rate, and badly breaks standing at zero
  command (touchdowns 142 -> 346/5325). The knob stays in the code, default 0.0.
- **Deploy status lives in `A1a_deploy_plan.md`**; this doc is training/sim only.

## Thesis hook
The HL earns its keep by choosing a **speed-dependent gait cadence** to minimize **cost of
transport**, an authority the flat `(vx,vy,wz)` command structurally cannot express. Claim:
A1a holds A0 tracking at lower CoT, where a flat `A0+energy` policy regresses tracking. Feeds
A2 (A-RMA supplies adaptivity) and A3 (gait library plugs into the same cadence channel).

## Design (locked; rationale in ADR-0004)
- **HL reward** = tracking + (−`hl_cot_coef`·CoT); `CoT = window energy / (actual walked
  distance + small floor)`, engaged when commanded linear speed > 0.1. HL-only; the LL stays a
  pure tracker (the decoupling claim). Power `P = Σ_j |qfrc_actuator_j · joint_vel_j|`.
- **Lever** = cadence: HL commands `period` (s) in `cadence_period_range`; the LL entrains via
  the phase-clock obs (`mdp.phase`) + the `feet_gait` term in the LL intrinsic
  (`ll_cadence_coef=0.5`). R2 routing: cadence is a *gait reference*, not a goal-space component.
- `period` = stride period (same-foot touchdown interval); 0.6 s ≈ 200 steps/min ≈ 2x human.

## Config additions (`config/h1_2_a1/rl_cfg.py`, all default to current A1)
`hl_cadence: bool=False`, `cadence_period_range: tuple=(0.5, 1.4)`, `ll_cadence_coef: float=0.5`,
`hl_cot_coef: float=0.0`. With the defaults the A1 code path is byte-identical to today.

## Prerequisite metrics (`play.py` benchmark `[BENCH]`)
`mech_power_w`, `cot`, `stride_period_s`. Added once; reused by every stage.
