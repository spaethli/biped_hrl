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

## WP5 — H-adapt: spec, pre-flight, and Bar A re-confirmation (2026-08-31/09-01)

Thesis core. Governing plan: `thesis_plan_8weeks.md` §4 (RMA phase order, the 2x2) and
WP5. Spec grilled and approved before any code, per CLAUDE.md. **Phase 1 not yet run** —
this section records the spec, the pre-flight that changed it, and the Bar A
re-confirmation. Exploratory narrative belongs in the KB, not here.

### Agreed spec (decisions taken in the grilling session)

| Item | Decision |
|---|---|
| Frozen LL | `2026-08-28_08-27-45_a1a_fullmirror_standing15_rs8_s42/model_10000.pt` (post-ADR-0009). **Never retrained.** |
| Env | `Unitree-H1_2-Flat-A1-Payload`, `--agent.hl-obs-e True`, explicit `resampling_time_range (3.0, 8.0)` |
| Latent | **No `mu`/`z`.** Raw `e ∈ ℝ⁵` into the TD3 state vector (WP2 plumbing). HL obs = 94+5 = **99** |
| `hl_cot_coef` | **5** — see "coef selection" below. Binds **H-mem identically**, or the 2x2's interaction term is confounded |
| Phase 2 | Fresh offline rollout of the frozen Phase-1 policy; raw obs stored ONCE in a ring buffer, H=50 windows by index (~300 MB, not 9 GB). phi input **92 dims/step** (89 proprio+past actions + 3 command) |
| Held-out | **Split by ENV** (stratified over payload) **plus a separate fresh-run test**. `mode="startup"` DR fixes `e` per env for the whole run, so the env is the independent unit |
| Bar B readout | `corr(commanded T, payload)` primary + **counterfactual-`e` sensitivity** to separate "module unused" (fatal) from "wrong readout" (survivable) |
| Compute | Train on SLURM (`num_envs` 4096, 2 seeds, identical across all four arms); **score on one idle GPU** |

### Blocking fix — `_load_frozen_ll` could not load ANY existing checkpoint

WP2 added `env_latent_e` unconditionally to the `critic` obs group, taking h1_2's critic
obs **114 → 119**. `_load_frozen_ll` strict-loaded the critic, so WP5 Phase 1 crashed
before iteration 0 on every A1 checkpoint on disk. Fixed: the **actor** stays a strict
load (a wrong-shape actor is a real goal-space error and must still raise); the
**critic** is best-effort, because `learn()` guards `process_env_step`/`compute_returns`/
`update` on `not self.freeze_ll` — a frozen LL never reads it. A checkpoint saved by a
frozen-LL run therefore carries an UNTRAINED LL critic and must not be resumed unfrozen
(warned at load). Tests: `tests/test_freeze_ll_load.py` (3), pinning the fix and **both**
over-fixes — skipping the actor would silently freeze a randomly-initialised LL that
still trains and scores healthily.

⚠ **Generalisable:** WP2's inertness proof was thorough and could not have caught this,
because it verified through `play.py`, which never strict-loads the critic. **An
asymmetric actor-critic change is invisible on the eval path and fatal on the train path
— it needs a train-side smoke.**

### Pre-flight — the cadence is SATURATED at hl_cot_coef=0.2

Bar B asks for `|r| ≥ 0.5` between commanded stride period and payload. Measured on the
keeper (`--diagnose-goals 600`, 64 envs, 2 seeds, vx=0.5 pinned, 150 windows/env):

| payload | mean T | **sigma_e** (sd of per-env mean) | within-env sd |
|---|---|---|---|
| 0 kg | 0.35255 | 0.00084 | 0.0202 |
| 6 kg | 0.35275 | 0.00089 | 0.0218 |
| 12 kg | 0.35292 | 0.00087 | 0.0228 |

Commanded T is pinned at **0.3527** against a range floor of 0.35 (tanh ≈ **−0.992**),
and moves **0.0004 s** across the full 0→12 kg sweep. ⚠ **That tiny `sigma_e` is an
artifact of the clamp, not a property of the policy** — a clamped variable has no
variance. At vx=0.2 the cadence comes off the bound (0.370) and `sigma_e` jumps **12x**
to 0.0105. **0.0105 is the number to use for Bar B feasibility.**

Cause, reconstructed from WP1b's grid at zero GPU cost: the HL objective is
`R = sum_{k=1..8}[track_lin + track_ang] − hl_cot_coef·CoT`, and **tracking's span across
T is 36–71x the CoT penalty's**, so the objective's optimum in T sits at the cadence
floor for every payload. Giving the HL the latent cannot move T when the optimal T does
not depend on payload.

### Bar A re-confirmation on the frozen LL — direction NOT resolvable

`T*(payload)` is a property of plant **x LL**, and Phase 1 freezes a different LL than
WP1b measured. Re-run: same protocol (vx=0.5, T ∈ {0.40..0.70/0.05} x payload {0,4,8,12},
R=8 independent processes, 1200 steps, 64 envs, one continuous uncontended run, drift
probe every 12 cells). 224 rows, 83.6 min. Drift `cot` CV **0.34%** (inside WP1b's 0.43%
floor); ⚠ `err_vx` rose monotonically **+1.8%** across the run.

**The mech-CoT `T*` direction flips with the fit window:**

| window | 0 kg | 4 kg | 8 kg | 12 kg | dT*/12kg |
|---|---|---|---|---|---|
| 3-pt local | 0.5988 | 0.5973 | 0.5973 | 0.5908 | **−0.0080 DOWN** |
| 5-pt (WP1b's) | 0.5751 | 0.5808 | 0.5805 | 0.5820 | **+0.0070 UP** |
| 7-pt full | 0.5605 | 0.5714 | 0.5730 | 0.5778 | **+0.0173 UP** |

Window-to-window spread at payload 0 is 0.038 s against an effect of 0.008–0.017 s — the
**systematic is 2–5x the signal**. Cause: the CoT bowl rises steeply past T=0.65, and the
arm is steeper at 0 kg than at 12 kg, so a wide window drags the vertex left more at low
payload and manufactures a shift. **On this LL, `T*(payload)` is not resolvable, and the
Bar B sign cannot be taken from it.**

**WP1b itself was re-checked and STANDS** (`data/2026-08-31-wp5-barA-reconfirm/scripts/
wp1b_fitcheck` route): mech CoT DOWN in **3/3** windows (−0.0425 / −0.0321 / −0.0122) and
copper CoT UP in **3/3** (+0.0304 / +0.0371 / +0.0547). Direction robust; ⚠ **magnitude is
window-dependent (3.5x for mech)**, and WP1b's "|span|/CI 16x" is a bootstrap over
repeats that does NOT include fit-method systematic — quote the direction, caveat the
number. WP1b survives because its effect is ~4x larger and its bowls deeper (local depth
1.5–4.1% vs this LL's 0.86–1.96%).

### Coef selection — and the mechanism that replaces `T*`

`T*_HL(payload)` of the ACTUAL objective, both fit windows, on the rs8st15 grid:

| coef | 3-pt span / b | 5-pt span / b | verdict |
|---|---|---|---|
| 0.2–2 | 0.000 / +0.00000 | 0.000 / +0.00000 | pinned; Bar B impossible |
| **5** | **0.114 / +0.01060** | **0.110 / +0.00948** | **robust, monotonic** |
| 7 | 0.131 / +0.01178 | 0.134 / +0.01174 | robust, monotonic |
| 10 | 0.147 / +0.01155 | 0.149 / +0.01166 | robust, monotonic |
| 15 | 0.026 / −0.00219 | 0.013 / +0.00040 | collapsed, sign unstable |
| 20 | 0.020 / −0.00180 | 0.009 / +0.00057 | collapsed, sign unstable |

**Chosen: 5** (owner's call, from the training cot sweep — at 10 tracking already
regressed materially; a coef-7 arm is running). ⚠ **20 is past the peak**: span collapses
~7x and the sign is unstable between fit methods.

**The Bar B mechanism at coef 5–10 does NOT require the CoT bowl to move.** It is a
balance-point shift: the **tracking surface flattens under load** (span across T falls
0.917 → 0.486 from 0 to 12 kg), so the CoT term — whose optimum ~0.59 sits well above
tracking's edge at 0.40 — wins more ground as payload rises. That is a sturdier mechanism
than `T*(payload)`, and it is what survives the fit-window check.

### Pre-registered Bar B prediction (declared BEFORE Phase 1 is scored)

**SIGN: POSITIVE** — commanded T *increases* with payload. This **contradicts the WP5
brief's pre-registered negative**, which was taken from WP1b's keeper mech-CoT `T*`; the
mechanism above is a different (and more robust) one. Independently corroborated by the
parallel `hl_cot` training sweep (CLAUDE.md / `docs/adr/0004` Amendment 2026-09-01):
"larger coef = LONGER stride", threshold-like with 0.2→2 inert and 2→5 flipping — the
same sign, the same threshold location, from a training run rather than a grid.

⚠ **Scored on the REWARD's CoT, not the bench metric.** The HL reward's CoT denominator is
the SIGNED projection on the commanded direction (`hrl_runner.py:955-957`, `_d_par`),
while `play.py`'s `cot` metric keeps the UNDIRECTED norm (`_d_step`, `play.py:779`). The
correction is 1/cos(theta) = 1.007–1.056 here, small but enough to leave 8 kg floored at
coef 5. Predictions below use the signed form.

At coef 5 (signed): `T*` = 0.400 / 0.400 / 0.400 / 0.509 over 0/4/8/12 kg — **3/4 floored
at the cadence edge**, so the whole response is carried by the top third of the payload
range. Monte-Carlo on that real piecewise shape, payload ~ U(0,12), 64 envs,
`sigma_e` = 0.0105: **predicted r = +0.737** (5th pct +0.647), `P(|r| ≥ 0.5) = 1.000`.
(Undirected CoT would have said +0.939 — a 0.2 overstatement, hence the caveat.) Coef 7
signed gives +0.939 with 2/4 floored, i.e. Bar B margin is the price paid for coef 5's
better tracking.

⚠ **What this does NOT establish.** The grid is a frozen-LL **pinned-T** sweep: it says
where the optimum sits, not that a trained TD3 HL finds it, nor what tracking a trained
policy achieves. `sigma_e` for the Phase-1 policy at its own unsaturated operating point
remains **unmeasured** until Phase 1 runs. Also, at vx=0.5 the longer stride *improves*
`err_vx` at high payload (−4.0% at 8 kg, −5.1% at 12 kg), so this sweep is structurally
blind to the tracking regression the training sweep sees — WP1b showed the optimal stride
falls below the grid floor by vx=0.7, so the coef's tracking cost is **speed-dependent**.

### Instrumentation added

- `scripts/period_payload_stats.py` — the Bar B statistic (per-ENV mean commanded T vs
  per-env latent, Pearson r + slope, per-component reads). Its own module because
  `play.py` imports the mjlab env stack at module scope and would break the test suite's
  CPU-only contract. `play.py` prints `[PERIODDIAG] {json}` from it.
- `tests/test_period_payload_stats.py` (4) — pins the **env-not-window** aggregation unit
  (pooling by window inflates n ~150x and shrinks every interval while leaving the point
  estimate roughly right), the keep mask, and NaN-not-0.0 on a constant regressor.

Raw data + analysis → `data/2026-08-31-wp5-barA-reconfirm/`.

## WP5 Phase 1 — RESULTS and the Bar B verdict (2026-09-01/02)

Two seeds, 10001 iters, 4096 envs, `Unitree-H1_2-Flat-A1-Payload`, LL frozen at
`2026-08-28_08-27-45_a1a_fullmirror_standing15_rs8_s42/model_10000.pt` (never retrained),
`hl_obs_e=True`, `hl_cot_coef=5.0`, `rel_standing_envs 0.15`, `resampling_time_range
(3.0,8.0)`. Runs: `2026-09-01_20-25-30_..._cadhl_..._s42` and `2026-09-01_23-18-56_..._s123`.

### ⚠ A first Phase-1 run was discarded — `hl_cadence` silently defaulted False

`rl_cfg.py:572` still defaults `hl_cadence=False` and `:614` `hl_cadence_source="random"`,
while `train_h1_2_a1a_LL_rewards.sh` defaults them True/`hl`. The launch was hand-built
(the script exposes no `HL_OBS_E`/`FREEZE_LL` knob), so both silently reverted. Result: the
HL actor was **99→3**, velocity goals only, **no commanded period column**, and
`env.hrl_period`/`hrl_phase` were never created (`hrl_runner.py:340`) so `mdp.phase` ran its
own fixed `period: 0.6` clock. `corr(commanded T, payload)` was then undefined *by
construction*, not by policy failure. Cost ~3 h.
**The tell was a bench scalar**: `stride_period_s` read 0.588, i.e. the fixed clock.
Marked in-place as `BROKEN_no_cadence_channel.txt`; bench archived for the record.
⚠ That run also incidentally reproduced the **cadence-pin** result (`act_legs` 0.6016 vs
A0's 0.5960, 1.01x) — a fixed clock IS a cadence pin, so do not read it as a latent effect.
**Rule: verify the HL action dim (4, not 3) on the smoke checkpoint before committing GPU.**

### Policy bar — PASSES on both seeds

| | s42 | s123 | bar |
|---|---|---|---|
| `err_vx` | **0.1142** | **0.1067** | < 0.1219 ✓ |
| `act_legs` | 0.7027 | 0.6939 | < 0.8363 ✓ (A0 0.5960) |
| `fall_rate` | 0.0 | 0.0 | ✓ |
| `stride_period_s` | 0.4024 | 0.4067 | off the 0.35 floor ✓ |
| `gait_match` | 0.8768→0.9275 | 0.9320 | |

`hl_cot_coef=5` did what the pre-flight predicted: the commanded period sits at **0.402 s**,
off the cadence floor, against the pre-flight's saturated 0.3527 at coef 0.2. The grid
predicted T* ≈ 0.400 — an independent confirmation of the objective reconstruction.

### BAR B — **FAIL on both seeds**

7 pinned payloads x 64 envs x 2 eval seeds = 448 envs, command pinned vx=0.5
(period is near-binary in the command, so a random-command average is a standing/walking
mixture, not a policy property).

| seed | slope `b` | **r** | verdict |
|---|---|---|---|
| 42 | +0.00086 s/kg | **+0.089** | FAIL |
| 123 | +0.00265 s/kg | **+0.268** | FAIL |

*`fig1_bar_b.png` — the gate and why it fails. The payload trend is real on both seeds (and
POSITIVE, as pre-registered), but the shaded per-env spread `sigma_eps` it must beat is an
order of magnitude wider. The two seeds also settled at different cadences (~0.40 s vs
~0.49 s); the verdict is FAIL on both regardless.*

Stable across seeds despite a 3.1x spread in the payload slope itself. Sign is POSITIVE as
pre-registered (T rises with payload), so the pre-flight's sign call stands; the magnitude
does not. Predicted was +0.737 — **the prediction was wrong, and by a mechanism worth
recording**: it came from a frozen-LL pinned-T grid, which cannot say where a trained TD3 HL
actually settles, and the pre-flight flagged exactly this ("`sigma_e` for the Phase-1 policy
at its own unsaturated operating point is still UNMEASURED"). Measured `sigma_eps` is
**0.033**, 3.2x the 0.0105 the prediction used.

### But the HL DOES use the latent — counterfactually proven, both seeds

New `play.py` flags `--cf-e-col` / `--cf-e-val` falsify one column of the `e` the HL READS
while every body/geom property stays as sampled. All envs get the same forced value, so the
true per-env spread is identical noise in each arm and the between-arm contrast is clean.
This is the one manipulation the pinned sweep structurally cannot make — pinning moves
physics and observation together, so it can never separate "reads `e`" from "feels `e`".

Physics pinned at 6 kg in every arm. `read%` = counterfactual / observational:

| component | obs ΔT | cf ΔT | read% | | obs ΔT | cf ΔT | read% |
|---|---|---|---|---|---|---|---|
| | **s42** | | | | **s123** | | |
| com_dx | −0.0839 | **−0.0702** | 84% (15.4 SE) | | −0.0716 | −0.0120 | **17%** (2.9 SE) |
| friction | +0.0485 | **+0.0363** | 75% (7.9 SE) | | +0.0703 | **+0.0684** | 97% (16.4 SE) |
| payload | +0.0098 | +0.0048 | 49% (1.1 SE) | | +0.0305 | **+0.0406** | 133% (9.7 SE) |

*`fig2_read_decomposition.png` — the headline. Grey = observational (physics and observation
move together), colour = counterfactual (observation ONLY). `read%` is suppressed as `n.s.`
where the counterfactual sits inside the ±2 SE band, because a ratio of two near-zero numbers
says nothing about the read path.*

*`fig3_cf_dose_response.png` — the causal evidence. Physics pinned at 6 kg in every arm and
every body/geom property identical, so all movement is caused by the observation alone.
Dotted lines are the unfalsified `truth` arm.*

**The core WP5 claim PASSES**: falsifying an observation alone moves the commanded stride
period by **0.070 s / 0.068 s** (15.4 / 16.4 SE) — 14-17% of baseline. The HL reads `e`.

**Which channel carries it is SEED-DEPENDENT, and only `friction` is read on both.**
`com_dx` is read on s42 (84%) and essentially not on s123 (17%) — s123's large *observational*
com_dx coefficient (t = −26.8) is therefore mostly **proprioceptive**, the HL feeling the
consequences rather than reading the value. Payload is the mirror image: inert on s42
(1.1 SE) but genuinely read on s123 (9.7 SE).

### Two seeds were load-bearing — do not report single-seed component claims

`com_dy` looked like a solid third channel on s42 (t = **−17.0**, ΔT −0.041 s) and is
**t = −1.16** on s123; `com_dz` **flips sign** (−2.27 / +5.05). Both would have been written
up as findings from s42 alone. Only `com_dx` (ratio 0.85) and `friction` (1.45) reproduce
observationally with t > 20 on both.

*`fig4_seed_reproducibility.png` — which components survive a second seed. Dashed lines
mark |t| = 2.*

### ⚠ Bar B is under-powered BY CONSTRUCTION — a methodological finding

Its denominator carries the other four latent components' variation as "noise". On s123 the
HL **demonstrably reads payload** (counterfactual 9.7 SE, ΔT +0.041 s) and the pre-registered
r is still only +0.268. Three estimators, same data:

| estimator | s42 | s123 |
|---|---|---|
| pre-registered (simple, payload) | +0.089 | +0.268 |
| post-hoc: partial, other four removed | +0.187 | +0.485 |
| post-hoc: counterfactual slope | +0.093 | +0.595 |

The two post-hoc rows are **diagnostics, not a rescue** — they were chosen after seeing the
data and cannot re-adjudicate the gate. The verdict is FAIL. But the design lesson stands:
**a correlation gate on ONE component of a multi-component latent must partial the others
out, or it can fail while the mechanism it tests is present.**

### Why payload is the wrong component (mechanism)

Stride period for an inverted-pendulum walker is set by pendulum **geometry** — CoM height
and leg length — not total mass: adding 12 kg at the torso scales gravitational and inertial
terms together and largely cancels out of the natural frequency. Shifting CoM *position*
changes that geometry and the pitch moment the swing leg must catch. A policy commanding
cadence off `com_dx`/`friction` and ignoring `payload_kg` is the physically literate one.
Bar B pre-registered the component with almost no cadence response to give.
⚠ Hypothesis, n=2, NOT a finding: s42 sits 0.05 s above the cadence floor and s123 0.14 s,
and the seed with more headroom shows the larger payload response.

### Instrumentation added / fixed

- **`play.py --cf-e-col / --cf-e-val`** — the counterfactual-`e` disambiguator (above).
  Guarded: requires `--diagnose-goals`, a col in 0..4, and a value; stamped into
  `[PERIODDIAG]` so a falsified run's JSON is distinguishable from a true one.
- **BUG FIXED — `--diagnose-goals` was unusable on ANY `hl_obs_e` checkpoint.** It calls
  `runner.hl.act_inference()` directly, bypassing both places the runner writes
  `obs["hl_e"]` (`hrl_runner.py:565`/`:719`), so `td3._state_vec` raised
  `KeyError('hl_e')`. ⚠ **Same bug class hit the training launch.** `obs["hl_e"]` is written
  by *callers*, so all three fire paths must independently remember it; the bench only
  passed because it routes through `get_inference_policy`. A more robust shape would have
  `_state_vec` fetch the latent, or one `fire_obs()` helper — not done, it is a design
  change WP5 does not own.
- `period_payload_stats.py` now emits per-env `com_dx/dy/dz` (+ `r_period_comdy/comdz`);
  without them the partial regression above is impossible. `tests/` 166 pass,
  `check_test_sensitivity.py` **72/72** (2 new: mis-sliced CoM column, unmasked CoM column).
- ⚠ **No CPU-only regression test guards the `play.py` fixes** — it imports the mjlab env
  stack at module scope. Verified by execution only.

Raw data (33 JSON), `analyze_barB.py` (re-derives every number above) and `plot_barB.py` /
`plot_mount_geometry.py` (regenerate all six figures) → `data/2026-09-02-wp5-phase1-barB/`.
The analysis scripts read only `raw/*.json`: no GPU, no mjlab import.
⚠ `data/` is **gitignored** (`.gitignore:5`), so the figures named above are LOCAL — cite them
by filename as here, never as an embedded image link, or the published repo carries a dead link.

### ⚠ AMENDMENT (2026-09-02, owner input): payload and CoM are COUPLED on the robot

The deployment condition is **a weighted backpack mounted in FRONT of the torso, at torso
height** — so payload never arrives without a CoM shift. Owner also reports independent
hardware corroboration for the payload-only null: removing the head did not change policy
behaviour, and the heavier-leg model mismatch is negligible (consistent with `docs/adr/0009`,
where the leg mass was found inert).

**The DR cannot represent that.** `base_mass` and `base_com` are *independent* startup events
on the same `torso_link` (`env_cfgs.py:336-365`): the sim adds up to 12 kg without moving the
CoM and moves the CoM without adding mass. A regression on that DR therefore recovers
**partial** derivatives — the response to mass with CoM held fixed — which is a counterfactual
the hardware cannot realize. The deployment-relevant quantity is the **directional** derivative
along a 1-D ray fixed by mount geometry: `dx = m*d/(M_torso + m)`, `M_torso` = **17.789 kg**
(`h1_2.xml`), `d` = horizontal offset from the torso CoM to the loaded pack's CoM.

⚠ **RANGE INCONSISTENCY — ACTIONABLE BEFORE THE MOUNT IS FABRICATED.** The trained `base_com`
span (±0.05 m) does not cover what the payload DR's own 0-12 kg would physically produce:

| `d` | `dx` at 12 kg | inside ±0.05? | max in-range payload |
|---|---|---|---|
| 0.10 m | 0.040 | yes | 17.8 kg |
| 0.15 m | 0.060 | **no** | 8.9 kg |
| 0.20 m | 0.081 | **no** | 5.9 kg |
| 0.25 m | 0.101 | **no** | 4.4 kg |

Past `d ~ 0.10 m` a full 12 kg drives the policy outside its trained CoM support. Either widen
`base_com` to cover `m*d/(M+m)` at full load, or constrain the mount so 12 kg stays inside.

### Coupled-ray result — Bar B still FAILS, and the seeds disagree in SIGN

Measured directly (not inferred), via multi-column falsification (`--cf-e-col 0,1`), physics
pinned at 6 kg in every arm, both seeds. Ray A pins `com_dx=0`; rays B/C let it track payload.

| ray | s42 dT/dm | s123 dT/dm | Bar B s42 | Bar B s123 |
|---|---|---|---|---|
| A payload only (`com_dx`=0) | +0.00033 | +0.00374 | +0.040 | +0.424 |
| B coupled, d=0.10 m | **−0.00171** | +0.00378 | −0.203 | +0.427 |
| C coupled, d=0.20 m | **−0.00375** | +0.00370 | −0.415 | +0.420 |

**FAIL on every ray**, and on the coupled rays the two seeds have **opposite signs**. Seed 42
responds to the coupling (slope turns negative); seed 123's three rays are nearly identical
because **it does not read `com_dx`** (17% read, 2.9 SE) — internally consistent with the
counterfactual decomposition above, from a completely separate measurement.

*`fig5_coupled_ray.png` — seed 42's rays fan out under coupling; seed 123's overlap.*

### ⚠ RETRACTED: an additive recombination predicted a PASS. It was wrong.

Combining the fitted partials as `dT/dm = dT/dpayload + dT/dcom_dx * d/M` predicted **−0.0086 /
−0.0055 s/kg** at d=0.20 and a Bar B of **−0.723 / −0.567 (PASS both)**. Measured: **−0.00375 /
+0.00370, FAIL both.** Two compounding errors, both worth generalising:

1. **Observational partials cannot predict a counterfactual.** A falsification exercises only
   the READ path; the observational `beta` also contains the proprioceptive path. Seed 123's
   observational `com_dx` coefficient is large (t = −26.8) but it barely reads the value, so
   coupling changed nothing for it.
2. **The read-path response to `com_dx` is strongly ASYMMETRIC, so a single linearization is
   invalid.** Seed 42: **−1.010 s/m** backward vs **−0.395 s/m** forward, a **2.56x** ratio.
   The coupled ray only explores the forward side; the prediction used a slope fitted across
   both.

Correcting both — read-path slope, forward side only — predicts **−0.00182** vs measured
**−0.00171** for s42 (6% agreement) and +0.00239 vs +0.00361 for s123.
**Rule: to predict a counterfactual, use counterfactual coefficients, taken over the interval
the manipulation actually traverses.**

⚠ Design note (n=2, do NOT act on alone): a FRONT mount sits on the shallow side of seed 42's
`com_dx` response (2.56x shallower than backward). Seed 123's asymmetry runs the other way
(0.35) but its `com_dx` response is 2-16x smaller in absolute terms throughout.

### Mount geometry measured on the robot (2026-09-02) — `base_com` must roughly DOUBLE

Owner measurements: torso **19 cm** deep (x) and **46 cm** tall (z); payload CoM **7 cm** in
front of the front face and **4 cm** below the torso's lower end.

**Model check** (`h1_2.xml`, `torso_link`, `ipos` = [0.0005, 0.0028, 0.2048], mass 17.789 kg):

| quantity | measured | model | verdict |
|---|---|---|---|
| torso height z | 0.46 m | 0.44 m (`torso_collision` box) | ✓ 4.5% |
| torso depth x, at chest height | 0.19 m | **0.146 m** | ✗ **30% larger** |

⚠ The full visual mesh does span 0.198 m in x, but **only because of the head** (z 0.55-0.76,
x to +0.125). Sliced to the chest band (z 0.03-0.47) the torso is a symmetric ±0.073 m box, so
"0.19 ≈ 0.198" is a coincidence, not agreement. Either the measurement includes covers/cabling
absent from the mesh, or it was taken across the shoulders. **Worth re-measuring at chest
height** — it moves `d_x` by 2.2 cm and the required `base_com` x by ~0.9 cm.

**Lever arm** (torso CoM → payload CoM) and the induced shift `delta = m*d/(M+m)`:

| | `d_x` | `d_z` | \|d\| |
|---|---|---|---|
| from the measurements | +0.1645 | **−0.2348** | 0.287 m |
| from model geometry | +0.1425 | −0.2148 | 0.258 m |

| m (kg) | `dx` | `dz` | outside ±0.05? |
|---|---|---|---|
| 4 | +0.030 | −0.043 | no |
| 6 | +0.042 | −0.059 | **z** |
| 8 | +0.051 | −0.073 | **x + z** |
| 12 | +0.066 | **−0.095** | **x + z** |

**⚠ The VERTICAL axis binds first, not the forward one** — the pack hangs 0.235 m *below* the
torso CoM but only 0.165 m *ahead* of it. At the current ±0.05 m only **4.8 kg** stays inside
the trained support (x alone would allow 7.8 kg).

**Required `base_com` to cover the full 0-12 kg: x ±0.066 m, z ±0.095 m** — round to
**x ±0.07, z ±0.10**, i.e. roughly **double** the current ±0.05. `y` can stay ±0.05 (the pack
is laterally centred; keep some for mounting asymmetry).

*`fig6_mount_geometry.png` — regenerate with `plot_mount_geometry.py`.*

⚠ **Widening the two ranges independently is the CHEAP fix, not the right one.** It would let
the sampler draw physically impossible combinations (12 kg with zero CoM shift — exactly the
direction Bar B measured and the hardware cannot realize). The principled fix is a **coupled**
payload event: sample `m`, then set the mass AND the CoM offset from it via `m*d/(M+m)`, so the
DR traverses the 1-D deployment ray instead of a 5-D box containing it. That is a small change
to `apply_payload_dr` and it would also make Bar B measurable on the ray by construction.

⚠ Note `com_dz` is the component the HL responds to LEAST (r = −0.109 on s42, sign-flipping
across seeds). So the axis furthest out of distribution is also the least influential — mildly
reassuring, but the training distribution still never contained the deployment configuration,
and `docs/adr/0009` is the standing warning that a bench cannot see a reserve it never taxes.

## SPEC — coupled payload-mount randomization (2026-09-02, grilled + approved)

Decision record → `docs/adr/0010-coupled-payload-mount-randomization.md`. Glossary terms
(**Mounted payload**, **Mount lever arm** `d`, `ê` vs `z`) → `CONTEXT.md`.
**Status: IMPLEMENTED and verified 2026-09-02** — see "Implementation result" at the end.

### Scope

Replace the two independent torso events (payload mass, CoM offset) with **one** event that
samples a mounted payload and derives mass, CoM and rotational inertia from it.

**Explicitly NOT changed:** `env_latent_e` (a pure readback — it already reports realized
deltas, so it works unmodified); the `base_com` event and its meaning; task ids; `e`'s
dimension (ℝ⁵) and therefore the deploy contract (HL 94→99, `φ` outputs 5); the LL; the HL
reward weights including `hl_cot_coef`; `deploy/` in its entirety; PROVENANCE.json and
checkpoint staging; **every non-payload task, which must stay byte-identical** (the WP2
inertness gate — the event is still registered only by `apply_payload_dr`).

### The physics

Pseudo-inertia of a body, `h = m·c` (first moment), `Σ = ∫ r rᵀ dm` (second moment):

    J = [[Σ,  h ],        Σ  = ½ tr(I_o) I₃ − I_o        I_o = I_c + m(cᵀc I₃ − c cᵀ)
         [hᵀ, m ]]

Adding a point mass `m_p` at body-frame position `p` is an **exact rank-1 update**:

    J' = J + m_p [p;1][p;1]ᵀ

then read back `m' = J'₃₃`, `c' = J'₀:₃,₃ / m'`, `I_o' = tr(Σ') I₃ − Σ'`,
`I_c' = I_o' − m'(c'ᵀc' I₃ − c' c'ᵀ)`, and eigendecompose `I_c'` into MuJoCo's principal
moments (`body_inertia`) + principal frame (`body_iquat`).

⚠ **`p` is anchored to the DEFAULT torso CoM, not the current one**: `p = ipos_default + d`.
The pack is bolted to the shell, so its physical location cannot depend on `base_com`, which
represents *uncertainty about* the torso CoM rather than a real displacement. Anchoring to
`ipos_default` also makes `p` independent of event order; only the resulting combined CoM
depends on `base_com`, which is correct.

### Sampling

    m   ~ U(0, 12) kg           d_x ~ U(0.13, 0.19) m
    d_y ~ U(-0.03, 0.03) m      d_z ~ U(-0.28, -0.18) m

Per-env (`shared_random=False`), `mode="startup"` as before. `M_torso` is **read from the
model**, never hardcoded. Measured rig: `d = (0.1625, 0, -0.2348)`; the `d_x` range covers the
cable-cover ambiguity (0.1425–0.1865 across extreme readings of where the cover sits).

### Files

| file | change |
|---|---|
| `src/tasks/velocity/mdp/payload_inertia.py` | **NEW.** Pure-torch `add_point_mass(mass, ipos, inertia, iquat, m_p, p)` → `(mass', ipos', inertia', iquat')`, batched over envs. No mjlab imports **in the module itself**, so it stays trivially unit-testable. Plus the event `payload_mount(env, env_ids, mass_range, d_ranges, asset_cfg)`. |
| `src/tasks/velocity/config/h1_2/env_cfgs.py` | Rewrite `apply_payload_dr` to register `payload_mount` **last** (`cfg.events.pop("base_mass", None)` then insert), with a startup assertion that `base_com` precedes it. |
| `scripts/play.py` | `--eval-payload-kg m` applies mass + CoM + inertia at **mean `d` = (0.16, 0, −0.23)**, i.e. "the rig, loaded to m kg". |
| `tests/test_payload_mount.py` | **NEW**, four groups below. |
| `scripts/check_test_sensitivity.py` | One mutation per group. |

### ⚠ The ordering guard is load-bearing

`Operation.add` has `uses_defaults=True` (`dr/_types.py:94-99`), so the engine writes
`default + random` — two `add` events on `body_ipos` mean **the second erases the first**, and
`apply_payload_dr:352` records `base_com` as running *after* the payload event today. Written
the obvious way the CoM shift silently vanishes and `env_latent_e` keeps reporting a plausible
number. Hence: the event is registered **last**, writes `body_ipos` itself (composing on the
current value), and asserts at startup that `base_com` precedes it in the event dict.

### Test plan (defined up front, per the workflow)

1. **Composition + ordering.** After startup, `body_ipos` carries **both** the `base_com`
   residual and the payload shift. Mutation: swap the event order → must be CAUGHT.
2. **Physical consistency.** `(mass, ipos, inertia)` match closed form for a known `(m_p, p)`;
   inertia eigenvalues stay strictly positive and satisfy the triangle inequality across the
   full sampled `(m, d)` range. Mutation: apply the parallel-axis term about the body origin
   instead of the CoM.
3. **Eval-pin parity.** `--eval-payload-kg m` produces the same `(mass, ipos, inertia)` that
   training samples at that `m` with `d = d_mean`. Mutation: pin mass only.
4. **`e` readback fidelity.** `env_latent_e` still returns realized
   `(payload, com_dx/dy/dz, friction)` after the coupled event, payload column = added mass,
   com columns = **total** delta. Mutation: report the payload-induced shift only.

### Acceptance

`pytest` green including the 4 new tests; `check_test_sensitivity.py` catches all 4 new
mutations; a 2-iteration train smoke on `Unitree-H1_2-Flat-A1-Payload` showing a non-zero
`com_dz` at high `m` in `env_latent_e` (the composition defect's live signature); and
**non-payload tasks byte-identical** — the WP2 inertness gate re-run.

### Deferred, deliberately

Phase-1 seeds are **not** re-run now. They are re-run once, after A1a itself is settled, so the
cost is paid a single time (owner's call, 2026-09-02). Until then the Bar B verdict stands as a
measurement on a DR that cannot represent the rig — which is the finding, not a defect in it.
Provenance markers go into both Phase-1 run dirs, since replace-in-place leaves their
`params/env.yaml` as the only record of what they trained on.


### Implementation result (2026-09-02)

`pytest` **193 passed**, 2 skipped (185 + 8 new). `check_test_sensitivity.py`: **89/89, baseline
green, zero misses** — including all four new ADR-0010 mutations.

**Live probe, 256 envs on `Unitree-H1_2-Flat-A1-Payload`** — the coupling is real in the sim,
not just in the unit tests:

| `e` column | mean | min | max |
|---|---|---|---|
| payload | +5.714 | +0.006 | +11.969 |
| com_dx | +0.035 | −0.045 | **+0.097** |
| com_dz | −0.052 | **−0.132** | +0.044 |
| friction | +0.958 | +0.309 | +1.599 |

`corr(payload, com_dx)` = **+0.647**, `corr(payload, com_dz)` = **−0.753**. Not ±1 by design:
`base_com` still adds an independent ±0.05 residual and `d` is randomized, which is exactly
what keeps `com` from being a deterministic function of `m`.

**Inertia is written and tracks the payload**: principal moments span 0.129–0.282 /
0.420–1.362 / 0.494–1.391 (sorted ascending by `eigh`; compare the sorted nominal
`[0.1278, 0.4096, 0.4873]` — each starts at nominal near m=0 and grows),
`corr(payload, tr(I))` = **+0.867**, and `body_iquat` deviates from identity by up to 1.62,
i.e. the principal frame genuinely rotates. ⚠ The correlation is 0.867 rather than ~1 because
`tr(I) ∝ m·|p−c'|²` depends on the randomized `d` as well as on mass — that is the right
answer, not a shortfall.

**The `dr.body_mass` UserWarning is gone** from the training log (it fired on every previous
payload run): mass is no longer sampled through a function documented as valid only for a
point mass *at* the CoM. 2-iteration train smoke on the payload task exits 0 with
`payload_mount` listed in the event manager table.

⚠ **A third WP2 test had gone VACUOUS and neither `pytest` nor review could see it.**
`test_apply_payload_dr_is_opt_in_not_on_the_base_tasks` — the inertness gate protecting replay
of every existing A0/A1 checkpoint — asserted `"base_mass" not in cfg.events`. ADR-0010 renamed
that event, so the assertion started checking for a key that exists nowhere: green, and passing
even with payload DR applied unconditionally to the base A0 task. Caught only by
`check_test_sensitivity.py`, and only because it requires the **named** test to fail (the
mutation did cause a failure, in a different test). Now asserted on the event's **function
origin**, which cannot be renamed without changing behaviour. General rule → `hrl-infra.md`.

⚠ **Two other WP2 tests had to be rewritten**, not deleted: they asserted the old `base_mass`
contract, and `check_test_sensitivity.py` refused to run at all
(`BASELINE NOT GREEN`) until they were. They now assert the **new** contract *and* the
`base_com` → `payload_mount` ordering, which the old ones could not check because the
dependency did not exist. Without the harness's green-baseline gate this would have surfaced
as four "weak new tests" in a sensitivity score rather than as a design change needing a
decision.
