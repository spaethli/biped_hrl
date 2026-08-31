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

### CORRECTION (WL-A, 2026-08-27) — the DIVERGENCE verdict above does not stand

Re-analysis of `wp1_sweep.csv` by the planning chat. The sweep itself, the tooling, and the
raw numbers above are sound and are kept as delivered; what does not survive is the
**T\* extraction** and the verdict built on it.

**1. The reported CoT `T*` shift is a fit artifact.** The raw grid argmin moves *up* with
payload; the delivered local-quadratic fit moves *down*:

| payload (kg) | grid argmin s42 | grid argmin s43 | reported fit s42 |
|---|---|---|---|
| 0  | 0.50 | 0.50 | 0.517 (**above** the grid min) |
| 4  | 0.50 | 0.50 | 0.506 |
| 8  | 0.50 | 0.55 | 0.504 |
| 12 | **0.55** | **0.55** | 0.504 (**below** the grid min) |

The fit sits above the data's minimum at 0 kg and below it at 12 kg — opposite to the grid
at both ends. That is the local quadratic being pulled by an asymmetric bracket on a
non-uniform grid, not a physical shift.

**2. Root cause: the mechanical-CoT bowl is flatter than the measurement noise.** Depth
across the 3 points bracketing the minimum, against the ±2.6% 64-env reproducibility floor:

| payload (kg) | seed 42 | seed 43 |
|---|---|---|
| 0  | 2.72% | 2.54% |
| 4  | **0.81%** | **0.34%** |
| 8  | **0.22%** | 3.93% |
| 12 | 5.22% | 6.44% |

Five of eight cells are at or under the floor, and the seeds already disagree on the grid
argmin at 8 kg. **An argmin cannot be located inside a bowl shallower than the measurement
error.** The delivered magnitude test (effect ≥ 2x seed spread) passed only because it
compared against the reproducibility of a *deterministic fit over noisy data*, which is not
the uncertainty that matters. Same class as the pre-registration lesson: the bar was
watching the wrong quantity.

**3. vx=1.0 is edge-censored, and that is the whole of the "payload has no influence"
reading.** `cot` rises monotonically across the entire grid (0.624 → 1.037); there is no
bowl, and T\*=0.35 is the window's left edge, not an optimum. A censored argmin cannot move
with payload. The physics is the `v = L/T` constraint with stride length bounded: at 1.0 m/s
and T=0.9 s the required ~0.9 m stride is out of hip range, so the robot fails outright
(`err_vx` → 0.27). Shorter period at higher speed is correct and expected.

**4. Payload IS a live latent — the signal is in the level, not the argmin.**

| | mech power @12 kg | copper-loss @12 kg |
|---|---|---|
| vx=0.5 | +16.5% | **+34.4%** |
| vx=1.0 | +10.6% | **+22.1%** |

Copper-loss responds ~2x harder at both speeds, exactly as plan §4 predicted: `Σ|τq̇|` is
blind to static holding torque, which is most of what a payload adds; `kτ²` is the term that
sees it. **The trained objective is the one metric structurally least able to see the chosen
latent.**

**5. New measurement, decisive for the reward question: tracking-optimal `T` is
payload-INVARIANT.** Grid argmin of `err_vx` is **0.50 s in 15 of 16 cells** (the lone
exception, vx=1.0/12 kg/seed 42 at 0.35, is contradicted by its partner seed), on a bowl with
real depth (0.0625 → 0.1615 across the grid, a 2.6x span far above noise). Payload moves the
tracking *level* but not its optimum.

Consequence: **a tracking-only HL objective has no gradient to change stride under load.**
Dropping `hl_cot_coef` to restore RQ2 reward parity with A0 would also remove the only driver
of the payload→stride mechanism that Bar B is defined to test. The two goals are in direct
conflict and the choice must be made explicitly, not by default.

**Corrected verdict.** Bar A is **UNRESOLVED on mechanical CoT**, **qualitatively PASSED on
copper-loss CoT** (argmin 0.35-edge at 0/4 kg → interior 0.60 at 8/12 kg), and payload is
confirmed a live latent in every level metric. This is **not a fail**: do not trigger the
plan's friction/slope pivot on it. Repeat scoped as **WP1b** — score bowl depth against the
noise floor, extend the grid below 0.35, raise the seed count, and settle the objective
question with a measurement rather than a preference.

### WP1b (2026-08-29) — Bar A RESOLVED and PASSES on CoT

Same frozen keeper, no retrain. Run in an isolated `git worktree` at commit `d077902`
(the tree the keeper trained on: model-v3 leg mass; the ADR-0009 revert landed after)
so it did not touch `main`. Grid: `T ∈ {0.40,0.45,0.50,0.55,0.60,0.65,0.70}` (uniform
0.05; 0.65/0.70 added so the copper-loss optimum for 8/12 kg is bracketed) × `payload
∈ {0,4,8,12} kg` × `vx ∈ {0.5,0.6,0.7}`, **R=8 independent play.py processes** per cell
(not `--eval-seeds` — see floor below), 1200 steps, 64 envs. **One continuous
uncontended run** (4.3 h) with the reference cell (T=0.50/0 kg/vx=0.5) re-measured
every 12 cells. `vx=1.0` excluded (edge-degenerate with this keeper, per WP1).

**Noise floor, measured not assumed.** Within-session, independent processes: `cot` CV
**0.21%** (`--eval-seeds` understates this ~8×: internal seeds share one process' RNG +
CUDA scheduling). Across the 4.3 h run (9 drift probes): `cot` **0.43%**, `cot_copper`
0.86%, `err_vx` 0.97%, `ss_vx_var` 11%; no drift, only a mild cold-start outlier on
probe 1. The assumed ±2.6% was ~6–10× pessimistic for pinned-command evals. **Separate,
larger floor — GPU contention:** a co-resident training run shifts every metric **6–15%**
(20–40× the process floor). Proven on the aborted first attempt: same cell, repeats
r0–5 (idle) `cot` 0.576 vs r6–7 (contended) `cot` 0.542, clean separation. Rule for
any WP1b-class sweep: uncontended GPU + one continuous session + interleaved drift probe.

**Step 4 — T\* with uncertainty (vx=0.5; local quadratic vertex on a uniform 5-pt
window, bootstrap 16–84% over the 8 repeats; reported only where local bracket depth >
2·SEM).**

| payload | mech CoT `Σ|τq̇|` — T\* | copper CoT k=0.3 — T\* |
|---|---|---|
| 0 kg  | 0.516 [0.515, 0.517] | 0.570 [0.567, 0.572] |
| 4 kg  | 0.490 [0.487, 0.494] | 0.575 [0.571, 0.578] |
| 8 kg  | 0.483 [0.481, 0.485] | 0.602 [0.598, 0.607] |
| 12 kg | 0.484 [0.482, 0.486] | 0.607 [0.603, 0.611] |
| trend | **DOWN −0.032 s, monotonic, \|span\|/CI 16×** | **UP +0.037 s, monotonic, \|span\|/CI 11×** |
| local depth vs 2·SEM | 1.5–4.1% vs 1.1–1.8% | 3.1–5.2% vs 0.8–1.1% |

vx=0.6: mech CoT resolvable only 1/4 payloads (bowls flatten as speed rises); copper CoT
resolvable 3/4, still UP (0.542→0.561 over 8 kg). vx=0.7: every objective edge-censored
at T=0.40 — the optimal stride drops below the grid floor at that speed, as WP1
predicted. **vx=0.5 is the speed with resolvable interior bowls, copper-loss the
deepest.**

**Step 5 — k-sensitivity.** WP1's "0/4 kg edge → 8/12 kg interior" copper regime shift
**does not reproduce** — it was a grid-ceiling artifact (WP1 grid stopped at T=0.60; the
copper optima are 0.54–0.61). With T to 0.70, all payloads have interior copper optima
at every k. The **T\*-rises-with-payload direction is robust to k**: k=0.1 →
0.543→0.589, k=0.3 → 0.570→0.607, k=1.0 → 0.573→0.614 (vx=0.5, 0→12 kg). Not an artifact
of the unfitted literature constant.

**Step 6 — objective comparison (report only, not adjudicated).**

| objective | interior optimum in T? (vx=0.5) | moves with payload beyond the floor? |
|---|---|---|
| mech CoT `Σ|τq̇|` | yes, 4/4 payloads | yes — **shorter** stride under load (−0.032 s / 12 kg) |
| copper CoT `Σ(|τq̇|+kτ²)` | yes, 4/4 (+3/4 at vx=0.6) | yes — **longer** stride under load (+0.037 s / 12 kg), robust to k |
| `err_vx` (tracking) | no — argmin pinned at T=0.40 edge, or flat ~0.45 | no — wants the shortest stride at every payload |
| `ss_vx_var` (F5 analogue) | no — monotone in T, argmin at the 0.40 edge in 12/12 cells | no interior optimum at all |

Level response reconfirmed (vx=0.5, 0→12 kg): mech power +5.6%, copper power +32.6%,
mech CoT +7.0%, copper CoT +34.7% — copper ~5× more payload-responsive.

**Verdict.** Bar A is **RESOLVED and PASSES on CoT**: an injected torso payload shifts
the CoT-optimal stride period, monotonically and above the measured noise floor, at
vx=0.5, on **both** CoT variants. WP1's "DIVERGENCE" reading (CoT vs `ss_vx_var`) is
superseded — `ss_vx_var` has no interior optimum, so it was never a competing optimum.
The real divergence is **within CoT**: mechanical `Σ|τq̇|` wants a *shorter* stride under
load, copper-loss `Σ(|τq̇|+kτ²)` wants a *longer* one, and it is not a k artifact.
`err_vx` confirms WP1 CORRECTION §5 — a tracking-only HL has no gradient to adapt stride
to load. The objective choice (which CoT, or a CoT+tracking blend) is the planning
chat's; the measurements are here. Raw data → `data/2026-08-28-wp1b-payload-repeat/`
(`wp1b_sweep.csv` 672 rows, `wp1b_drift.csv`, `wp1b_tstar_*.csv`, `wp1b_*_vs_T.png`).

## WP2 — payload DR + privileged-latent plumbing (2026-08-31)

Critical-path item: WP3 (baseline arms) and WP5 (H-adapt Phase 1) both gate on it.
Governing plan: `thesis_plan_8weeks.md` §3 (the latent + taxonomy), §4 (the 2x2, RMA
phases), WP2/WP5. Spec was grilled and approved before any code (per CLAUDE.md's
workflow) — two open questions below were resolved in conversation, not silently.

### Grilled open questions — resolved

**(a) `e` dimension: ℝ⁵, not the plan's ℝ⁶.** `grep` across `src/` finds no
`yaw_bias`/`gyro_bias`/`heading_bias` mechanism anywhere — nothing to sample yaw bias
from in sim. A same-day check (before deciding) pulled the raw hip-yaw encoder trace
from `logs/deploy_safety/2026-07-28_10-13-17.csv` (the WL-B0e lean-investigation hang
session, still on disk, logs `raw_q0`/`raw_q6` = left/right hip yaw + `meas_dq0/6` at
1 kHz). Segmenting by joint velocity finds two genuinely motionless windows (`max|dq|
< 0.011 rad/s`, ~76 s and ~226 s — almost certainly the WL-B0 item-2 "standing" and
"passive hang" poses): hip-yaw L−R = **−0.31° and +0.82-0.84°**. Compare: active A1a
walking, 2026-08-13, 7 hardware sessions, hip-yaw L−R = **+10.3° to +12.1°** (mean
0.19 rad, std 0.012) — **13x larger**. A static encoder-zero offset would read the same
magnitude whether the robot is standing still or walking; it doesn't, which points at a
control-induced effect (policy or load-dependent actuator behavior), not a fixed plant
parameter — consistent with the plan's own taxonomy placing yaw bias in the "contested,
slow execution-corrupting" class, not the "reference-changing" class this `e` channel is
for. Caveats: the 2026-07-28 session predates the 2026-08-13 measurement by 2+ weeks and
the checkpoint deployed that day isn't recorded in the session's meta.json, and "passive
hang" is inferred from motion signature, not an explicit label — suggestive, not a closed
verdict, worth a dedicated WL-B follow-up (hang + stand + walk, same session, same
policy, plus an A0 control). Either way, a control-induced effect isn't a physical
parameter a sim DR term can inject before a rollout, so it doesn't belong in `e` at all.
`e = (payload_kg, com_dx, com_dy, com_dz, friction)`; yaw bias stays a hardware-only
F2/F3 metric (WP0/WP7). **Flag for WP5**: compressing ℝ⁵→ℝ⁴ (`z=μ(e)`) is a much milder
reduction than the plan's ℝ⁶→ℝ⁴ — worth a conscious call at that point on whether ℝ⁴
still makes sense.

**(b) Runtime readability, confirmed per-component.** `dr.body_mass`/`dr.body_com_offset`
write `env.sim.model.body_mass`/`body_ipos` directly, per-env (`mjlab/envs/mdp/dr/body.py`);
`dr.geom_friction` writes `env.sim.model.geom_friction` (`operation="abs"`). All three are
plain per-env tensors, readable at any point after the `mode="startup"` event fires — no
private DR-engine state needed except the public `env.sim.get_default_field(field)`
(baseline before randomization, already how the DR engine itself samples).

**(c) `z ∈ ℝ⁴`** — not WP2's call; flagged above for WP5.

**(d) A0 critic exposure — approved, symmetric on every arm.** Same asymmetric
actor-critic pattern already shipped for `Unitree-H1_2-Rough`'s `height_scan`
(privileged value function, deployable actor untouched). Doesn't touch any actor's
inputs or the env reward (RQ2-safe by the existing rule), applied uniformly to
F-mem/H-mem/F-hist/H-adapt so no arm is asymmetrically privileged relative to another.

### Implementation

- **Payload DR** (`config/h1_2/env_cfgs.py`): `apply_payload_dr(cfg, ranges=(0,12))`
  extends the existing `base_mass` event (previously reachable only via the wide-DR
  bundle) — same event key, same `dr.body_mass`, same torso `asset_cfg` `base_com`
  already uses. **Strictly opt-in**: `dr.body_mass` samples RNG even at a degenerate
  `ranges=(0,0)`, so registering it unconditionally on the base env would shift the RNG
  stream every later-registered event (`foot_friction`, `encoder_bias`, `base_com`,
  `push_robot`) consumes, breaking byte-identical replay for every existing checkpoint —
  the base `Unitree-H1_2-Flat`/`-A1` tasks never call it. New task variants carry it:
  `Unitree-H1_2-Flat-Payload` (F-mem), `Unitree-H1_2-Flat-A1-Payload` (H-mem, via
  `unitree_h1_2_flat_a1_env_cfg(payload_dr=True)`). `--eval-payload-kg` (`play.py`)
  unchanged — it overwrites the same `base_mass` event key regardless.
- **`mdp.env_latent_e`** (`src/tasks/velocity/mdp/observations.py`): the single readback
  function both consumers below call — payload/CoM as deltas from
  `env.sim.get_default_field`, friction absolute.
- **Critic term**: unconditional entry in the existing `"critic"` `ObservationGroupCfg`
  (`velocity_env_cfg.py`), `torso_cfg`/`foot_cfg` set per-robot in `env_cfgs.py`
  alongside `base_com`/`foot_friction`'s own asset_cfgs.
- **HL-only channel**: `HrlRunnerCfg.hl_obs_e: bool = False` (`config/h1_2_a1/rl_cfg.py`)
  — exact mirror of `hl_obs_vel`/`obs["hl_vel"]`. The runner writes `obs["hl_e"]` at the
  HL fire step only (both `learn()` and `get_inference_policy()`; no post-fire refresh
  needed since `e` is privileged ground truth at train AND eval and constant within an
  episode — `mode="startup"` DR doesn't resample on reset). `HighLevelTd3` gained
  `obs_e_dim` (`_state_dim`/`_state_vec`), and `hl_obs_e` was added to `play.py`'s
  `structure_keys` (defaults `False`, no absence-shim needed — unlike `hl_obs_vel`/
  `hl_velocity_goals_only`, which flipped their defaults after shipping). Off →
  byte-identical. WP2 builds only this plumbing, not WP5's `μ(e)`/`z` encoder.
- **Logging**: `[BENCH]`/`[BENCH_SEED]` gained `payload_kg` — the ACTUAL sampled value
  (mean over envs, via `env_latent_e`), not the requested `--eval-payload-kg`, so WP3/WP6
  can segment a payload-DR training run by payload post-hoc without re-running.
  `cot_copper` (WP1 payload pilot) was already logged; unchanged.

### Testing

`tests/test_privileged_latent.py` (7 new tests): `env_latent_e` delta/absolute
arithmetic against a minimal fake sim (payload=0/CoM=0 exactly when DR is disabled,
regardless of friction's independent always-on DR); `apply_payload_dr` opt-in (base A0/
A1 tasks carry no `base_mass` event; the two new `-Payload` task variants do, at the
requested range); `HighLevelTd3` `obs_e_dim` wiring (state-dim math, byte-identical
`_state_vec` at `obs_e_dim=0`, correct concatenation at `obs_e_dim=5`). Full suite: 143
passed, 2 skipped (was 141 pre-WP2 per CLAUDE.md, which flagged that count as already
stale). `check_test_sensitivity.py` — 3 new mutations targeting the three load-bearing
seams (delta-vs-absolute readback, the opt-in guard, the state-dim math): **3/3 caught**
(dropping the default-subtraction, registering `apply_payload_dr` unconditionally on the
base A0 task, and omitting `obs_e_dim` from `HighLevelTd3`'s state-dim math all turn the
suite red on exactly the expected test).

### Inertness proof

**Config-level (exact, deterministic):** `test_apply_payload_dr_is_opt_in_not_on_the_base_tasks`
proves `Unitree-H1_2-Flat`/`-A1` never register the `base_mass` event — this is the actual
claim (no RNG-stream shift), checked by direct dict inspection, not statistics.

**Rollout-level (empirical, play.py): the naive "literal diff of zero" bar from the spec
does not hold on this platform, for a reason that predates WP2 — and the correct bar
does.** `git stash` of all 10 WP2-changed files (scoped by explicit pathspec, leaving
unrelated concurrent edits to `CONTEXT.md`/`worklines.md` untouched) gave a clean pre-WP2
tree. Three `Unitree-H1_2-Flat` runs, identical command
(`--checkpoint-file .../a0_v2_optB_fric0p1_kl01_s42/model_9900.pt --num-envs 64
--eval-steps 600 --eval-seeds 1`, seed 42 fixed internally):

| run | code | err_vx | jacc_legs | action_rate | cot | fall_rate |
|---|---|---|---|---|---|---|
| pre | stashed (original) | 0.079357 | 31.192 | 0.649445 | 0.563238 | 0.0 |
| post (1st launch) | WP2 | 0.080274 | 30.389 | 0.645050 | 0.561545 | 0.0 |
| post (2nd launch) | WP2 (same code, rerun) | 0.079098 | 33.143 | 0.650626 | 0.563266 | 0.0 |

pre-vs-post differs by 0.1-2.6% per field (18 fields checked). That is NOT zero — but
re-running the identical POST-WP2 code a second time (same command, same seed) shows a
**same-or-larger** spread on 12 of 18 fields (e.g. `jacc_legs` pre-vs-post = −2.6%, but
post-vs-post(rerun) = +9.1%; `jacc` −1.9% vs +6.4%). No field shows a systematic,
one-directional shift — the sign flips inconsistently across metrics between the two
same-code launches, which is the signature of noise, not a code effect. This matches a
pre-existing, documented property of this codebase, unrelated to WP2:
`hrl-infra.md`'s own gotchas list "Same-config runs diverge a lot (GPU non-determinism +
RL chaos)" — MuJoCo-Warp on GPU is not bit-reproducible across process launches even with
`torch.manual_seed` fixed, and jacc-family metrics (higher-order, from realized
acceleration) are the noisiest, exactly as seen here. **Literal bit-identical replay is
not achievable on this platform for ANY code, so it is not the correct bar; "indistinguishable
from the platform's own same-code noise floor" is** — the same standard the codebase
already applies everywhere else (`hrl-infra.md`: "score against a replicate band, never a
single control"). By that standard: PASS. Exact and non-statistical: `payload_kg` reads
**0.0** in both post-WP2 launches, confirming no payload was silently injected on the base
task. This deviates from the spec's stated "diff of zero, not a rounding question" bar —
flagged here rather than silently reinterpreted, with the evidence (the same-code control
run) that makes the deviation legitimate rather than a shortcut.

### WP1b T\* direction re-confirmation on the current (v2) plant

**Not needed — the 2026-08-29 run was already on the v2 plant.** The `git worktree`
it ran in (at `d077902`, pre-ADR-0009) only isolated `scripts/` and `doc/`. `src/`
and `src/assets/` — including `h1_2.xml` (leg mass) — resolve to the MAIN checkout
via the editable install's `__editable__.unitree_rl_mjlab-*.pth`
(`'src' -> .../unitree_rl_mjlab/src`), regardless of cwd or which worktree `play.py`
lives in. `git log c8043f3..HEAD -- src/assets/robots/unitree_h1_2/xmls/h1_2.xml` is
empty: the leg mass has been v2 (light) continuously since the ADR-0009 revert
(2026-08-28 14:30), which is *before* the 2026-08-29 01:25 run. That run's own drift
probes (`cot` CV 0.43%, no trend over 4.3 h) confirm `src/` was stable through it.

**Verified cell-by-cell (2026-08-31).** A fresh partial re-run on current HEAD (the
192-cell vx=0.5 block, R=8 each, one continuous run) agrees with the committed
2026-08-28 sweep to within the noise floor: mean |Δ| `cot` **0.56%**, `cot_copper`
0.49%, `err_vx` 1.17% (worst cells all at T ≥ 0.65 where per-cell CV is 3–6% anyway).
Two fully independent 8-repeat sweeps two days apart → the same numbers. The re-run
aborted partway on an unrelated `ImportError` from the parallel WP2 `env_latent_e`
edits to MAIN's live `src/` — which is also why a worktree cannot isolate a sweep
here; only a separate clone + venv could, and the agreement above shows that is
unnecessary. Verification data →
`data/2026-08-28-wp1b-payload-repeat/v2plant_verification_2026-08-31/`.

**Conclusion: the WP1b T\* directions stand on the v2 plant** — mech CoT shortens
under load, copper CoT lengthens, both above the floor. The real robot's heavier
legs remain a separate sim-to-real gap (WP0/WP7), not a WP1b confound.
