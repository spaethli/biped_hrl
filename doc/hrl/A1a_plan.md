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
  (`joint_acc_l2`, `ll_joint_acc_coef`) is CLOSED as an *action-rate* lever**
  (2026-07-28, 2 doses): it does not move the arms share, makes `ub_arm_vel`
  worse, is non-monotonic on action rate, and badly breaks standing at zero
  command (touchdowns 142 -> 346/5325). The knob stays in the code, default 0.0.
  **Amended 2026-08-09:** that verdict was scored against the wrong quantity —
  no joint-acceleration metric existed. With `jacc` now benched, the two
  jacc-trained arms are the *lowest*-`jacc` of 13 (35.3/35.8 vs 41.9+) and hold
  the best `jacc_legs_p95`. It works on what it targets; it is still not an
  action-rate fix.
- **Smoothness is 3 metrics, not 1 (2026-08-09).** `action_rate` (commanded,
  whole-body), `act_legs` (commanded, legs), `jacc*` (realized motion). They
  disagree and reorder the deploy shortlist: combo1 is best on `action_rate`
  and worst of 13 on `jacc_legs_p95`. Definitions + A0 refs -> `hrl-infra.md`;
  full table -> the A1a experiment journal (research KB).
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

## WP1 payload pilot (2026-08-27) — Bar A gate, sim-only

**Goal:** before building any payload-adaptation machinery, check whether an injected
torso payload actually shifts the CoT- or stability-optimal stride period at all, on
the frozen `arm4d`-lineage keeper (`2026-08-26_10-02-56_a1a_fullmirror_jacc1e-7_standing15_s42/model_10000.pt`).
This is a **zero-shot generalization probe**: the keeper was never trained with payload
DR, so it asks whether the *physics* wants a different gait under load, not whether the
policy has learned to supply one. No retrain, no reward-weight change; only `scripts/play.py`
was extended (three additions, see `hrl-infra.md`'s eval-tooling list):
- `--eval-payload-kg <kg>` — pins an exact torso payload for the whole eval rollout,
  reusing `dr.body_mass` with a degenerate `ranges=(kg,kg)` (the same idiom as
  `--eval-cadence-period`), added only when the flag is set (byte-identical otherwise).
- `cot_copper` — copper-loss-corrected CoT variant, **metric-only, never in any reward**:
  `Σ_j(|τ_j q̇_j| + k·τ_j²)`, k=0.3 from Yang et al. 2022 (CoRL), *Fast and Efficient
  Locomotion via Learned Gait Transitions* (arXiv:2104.04644) eq. 3 — a literature
  default (MIT-Cheetah/Di Carlo convention), **not fit to the H1-2's actual actuators**
  (no public winding-resistance/torque-constant spec for the M107/GO2HV motors exists).
- `omega_xy` (mean body roll/pitch-rate magnitude) and `ss_vx_var` (cross-env variance
  of achieved vx under a pinned `--eval-cmd-vx`, over the last 2/3 of the window — the
  sim analogue of the thesis's F5 repeatability metric: the 64 parallel envs under an
  identical pinned command **are** the "N repeats").

A fourth candidate stability proxy — fall rate under a standardised push — was
**deliberately scoped out of this pilot** (the user's call, 2026-08-27): it probes
disturbance recovery, which is closer to deploy-readiness testing than to "does payload
shift the energy/predictability-optimal gait," and there is no existing eval-side push
mechanism to reuse cheaply. `omega_xy`/`ss_vx_var` alone turned out sufficient to resolve
Bar A (below); revisit push-based stability if a future WP needs it.

**Sweep:** `T ∈ {0.35,...,0.9}` (9 pts, dense below 0.6) × payload `∈ {0,4,8,12} kg` ×
`vx ∈ {0.5, 1.0}`, 64 envs × 600 steps × 2 seeds/cell, 72 cells (144 rows), unthinned
(one cell timed at 69.4 s → 84 min extrapolated, under the 8 h budget). Raw CSV, per-seed
T* fits, and figures → `data/2026-08-27-wp1-payload-pilot/`.

**T\* extraction:** local quadratic fit over the 3-5 grid points bracketing the raw
grid-argmin (never a global 9-point fit — the grid spacing is non-uniform, 0.05 below
0.6s / 0.1 above). A grid-argmin sitting at T=0.35 or 0.9 is reported **edge-censored**,
not fit through.

**Result — vx=1.0 is edge-degenerate on every objective.** The grid-argmin sits at
T=0.35 (the tested floor) for every payload level, on CoT, `cot_copper`, `omega_xy`,
and `ss_vx_var` alike (`fig1_cot_surface.png` right panel). No T\* shift is observable
in `[0.35, 0.9]` at this commanded speed — the true optimum (if interior) likely sits
below the grid's floor, i.e. outside `cadence_period_range` as trained. **Inconclusive,
not a fail**: this vx just doesn't resolve the question with this grid; extending the
grid lower is future work, not a fix applied here (the grid floor is fixed by protocol).

**Result — vx=0.5, CoT: monotonic, magnitude-passing.**

| payload (kg) | T\* seed 42 | T\* seed 43 |
|---|---|---|
| 0  | 0.5171 | 0.5178 |
| 4  | 0.5064 | 0.5062 |
| 8  | 0.5043 | 0.5095 |
| 12 | 0.5038 | 0.5020 |

Heavier payload shifts the CoT-optimal stride shorter. Seed 42 is monotonic
end-to-end; seed 43 has one sub-noise wiggle at 8 kg (+0.0033, smaller than the
0.0051 max seed spread measured elsewhere) before continuing down. Effect
`|T*(12kg)-T*(0kg)|` = 0.0133 / 0.0158 (seed 42/43) vs. mean seed-to-seed spread
0.0019 (max 0.0051) — **effect exceeds 2x spread by ~3.4-4x**. `fig1_cot_surface.png`
shows a genuine shallow bowl at every payload level (not a boundary solution).

**Result — vx=0.5, stability (`ss_vx_var`): monotonic, magnitude-passing, but a
different curve shape.** `ss_vx_var` is monotonically **increasing** in T across the
*entire* grid for every payload (`fig2b_stability_ss_vx_var_surface.png`) — it has no
interior bowl. Its fitted "T\*" is really the point where a near-flat, near-zero-variance
plateau ends and variance starts climbing, not a comparable interior optimum to CoT's.
With that caveat:

| payload (kg) | T\* seed 42 | T\* seed 43 | status |
|---|---|---|---|
| 0  | 0.3500 | 0.3500 | edge-censored |
| 4  | 0.3936 | 0.3934 | fit |
| 8  | 0.3999 | 0.4003 | fit |
| 12 | 0.4156 | 0.4160 | fit |

Monotonic **up** for both seeds (0 kg's edge value included). Effect = 0.0656 / 0.0660
vs. mean seed spread 0.0002 (max 0.0004) — **effect exceeds 2x spread by >130x**, the
cleanest signal in the whole sweep, though on a metric whose "optimum" is structurally
a plateau-elbow, not a bowl.

`omega_xy` (the other stability proxy) does **not** corroborate: it is valley-shaped
and edge-pinned at both 0 kg and 12 kg (T\*=0.35, `fig2a_stability_omega_xy_surface.png`),
peaking at 4-8 kg (~0.522-0.524) — **non-monotonic for both seeds**, so it cannot
independently support or refute a direction on its own.

**Verdict: DIVERGENCE, not PASS/FAIL — per the task's explicit stop-rule.** CoT-optimal
T\* moves **down** with payload; the stability proxy that does pass its own
monotonicity+magnitude bar (`ss_vx_var`) moves **up** with payload — opposite
directions, both well outside seed noise. Per protocol this is reported and the reward
question is left to the planning chat, not decided here. Numbers above are the full
audit trail. A secondary, striking observation under `cot_copper` (not part of the Bar A
verdict, since it's edge-censored at 2 of 4 payload levels): the copper-loss-weighted CoT
optimum sits at the grid floor (T\*=0.35) for 0/4 kg but jumps to an interior ~0.64-0.66 s
at 8/12 kg (`fig1b_cot_copper_surface.png`) — a much larger, qualitative regime shift
(effect ≈0.31 vs. seed spread ≈0.0015-0.0036), worth a note for whoever picks this up,
but not usable as a clean 4-point T\*(payload) curve given the two edge points.

**CoT-optimal vs. stability-optimal T\* (reported, not adjudicated, per the task):**
at vx=0.5 they diverge — CoT wants ~0.50-0.52 s, softening slightly with payload;
`ss_vx_var` wants the fastest available stride, with payload only pushing its
low-variance plateau's edge from 0.35 s toward ~0.42 s. They do not coincide at any
payload level tested. Whether this triggers WP1's "add a stability term to the HL
reward" branch is the planning chat's call.
