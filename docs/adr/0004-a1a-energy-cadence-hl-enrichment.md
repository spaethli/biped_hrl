# A1a: give the HL a job beyond velocity passthrough (cost-of-transport objective via a cadence reference)

**Status:** accepted (premise validated 2026-07-02; S1c machinery implemented + tested
2026-07-02: `hl_cadence_source="hl"` gives the TD3 HL the period as +1 action dim, CoT enters
the HL window reward via `hl_cot_coef` — see `doc/hrl/A1a_plan.md`; S3 = first learned-HL run).
⚠ **The S4/S5 gate verdicts below were scored on a CoT denominator that changed 2026-07-10 and
are retired — see the Amendments at the end. S4's A1a half PASSES at `hl_cot_coef=5`, and
`hl_cot_coef=5` + `rel_standing_envs=0.20` meets it in BOTH regimes (2026-09-05).**

## Context

The thesis research question is **does the hierarchy improve sim-to-real transfer?** The
supervisors (2026-06-18) called A1 a *velocity relay*: the high level forwards the twist
command on a coarser timescale and adds nothing, so even a clean A0-vs-A1 transfer study
would likely return "no difference" because no hierarchical structure is being exploited.
A relay HL makes the RQ unanswerable. The HL needs a job the flat command structurally
cannot do (the hierarchy-benefit roadmap (research KB), Track A, Axis 1).

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
  ⚠ **Amended 2026-09-01:** the 🔴 verdict is retired (old denominator). A1a half PASSES at
  `hl_cot_coef=5` — CoT −23% vs A0 at 0.61x its velocity error, 1 seed. A0/A0+energy controls
  still not built.
- **S5 weight guard.** Sweep `hl_cot_coef`; fall_rate must stay flat (rising fall_rate = CoT
  beating tracking = the suicide attractor; cap the weight below that point).
  ⚠ **Amended 2026-09-01:** the cap is between 5 and 10, not below 0.5, and `fall_rate` alone
  does NOT detect the failure (coef 10/20 walk smoothly at ~5x A0's `err_vx` without falling).
  Pair the guard with a tracking floor.
- **S6 OOD proxy (secondary).** Narrow -> Wide DR / push / terrain. Exploratory: a large
  survival edge is not expected until A2 supplies dynamics adaptation; S4 is the load-bearing
  A1a result.

Two data-set knobs: `cadence_period_range` (from S0), `hl_cot_coef` (bounded by S5).

## Amendment 2026-09-01 — the S4/S5 verdicts were scored on a superseded CoT denominator

**The S4 and S5 gates above were failed on a reward that no longer exists, and both verdicts
are retired.** The CoT denominator changed from the UNDIRECTED speed integral to the SIGNED
projection onto the commanded direction (`hrl_runner.py`, "HL-reward distance"). Both sweeps
that scored these gates predate that change:

| date | event | denominator |
|---|---|---|
| 2026-07-04 | S5 sweep `hl_cot_coef` {2,4,8} -> "raising the coef makes cadence *shorter*, collapses entrainment (gait_match 0.9->0.5), degrades tracking" | undirected |
| 2026-07-09 | co-train {0.2, 0.5} -> 0.2 kept, 0.5 "past the boundary" | undirected |
| **2026-07-10** | **denominator -> signed projection, explicitly "to promote longer strides"** | **changed** |
| 07-10 -> 09-01 | ~40 runs, all at `hl_cot_coef=0.2` | projected — never re-swept |

With an undirected denominator any motion earns distance, so a larger weight rewarded
displacement-per-joule in *any* direction — thrashing and sideways motion, hence a shorter
cadence. The projection closes that path; only forward progress earns distance.

**Re-sweep on the current reward (2026-09-01, `rse 0.10`, range 0.35-1.0, `src=hl`, 1 seed):**

| `hl_cot_coef` | 0.2 | 0.3 | 0.5 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| HL commanded period (s) | ~0.35 | 0.359 | 0.369 | 0.358 | **0.775** | 0.918 | 0.998 |
| `err_vx` | 0.198 | 0.073 | 0.064 | 0.076 | **0.054** | 0.480 | 0.437 |
| `gait_match` | 0.934 | 0.934 | 0.940 | 0.958 | 0.956 | 0.951 | 0.959 |

1. **The sign flipped: a larger coefficient now LENGTHENS the stride**, with no entrainment
   collapse at any weight (`gait_match` 0.93-0.96 throughout).
2. **The response is threshold-like, not graded** — 0.2 to 2 is a tenfold change that moves the
   period not at all; between 2 and 5 it flips. The old "0.5 is past the boundary" reading was
   measuring a different reward's boundary; on the current one 0.5 is still inert.
3. **S4's primary criterion is MET at `hl_cot_coef=5`** ("A1a `cot` < A0 `cot` at A0-level
   tracking and fall_rate"): CoT **0.391 vs A0's 0.507 (-23%)** at `err_vx` 0.0542 vs 0.0893
   (0.61x) and `fall_rate` 0, plus `act_legs` 0.96x / `ajit` 0.91x / mech power 0.91x A0.
   Run `2026-09-01_03-38-52_a1a_fullmirror_per0p35-1p0_cot5_standing10_s42`.
   ⚠ **S4's other half — the A0+energy control that must regress tracking to buy its CoT cut —
   is still not built, so this is a pass on the A1a half only, at 1 seed.**
4. **S5's cap is real but sits between 5 and 10**, not below 0.5: `cot` 10/20 hold `gait_match`
   and `fall_rate` yet drive `err_vx` to ~5x A0. **`fall_rate` alone does not detect this** —
   the arm walks smoothly and slowly rather than falling, so S5's stated guard would have
   passed it. Pair the weight guard with a tracking floor (`err_vx` / `ss_err_vx`).
5. Standing at `cot 5` (`rse 0.10`) is NOT scoreable: td 220 sits inside that cell's same-seed
   replicate band (130-2620). Re-run at `rse 0.20`, the only standing setting that reaches the
   touchdown floor on both seeds.

**Rule this generalizes:** a coefficient is calibrated against a *reward*, not a task. When a
reward term is reformulated, every weight tuned against it — and every sweep that ruled out
neighbouring weights — reverts to unvalidated.

Instrument: `python scripts/play.py <TaskID> --checkpoint-file <pt> --num-envs 64
--diagnose-cadence 600 --eval-cmd-vx 0.5` reports the HL's commanded period, its within-episode
std across windows, and windows-per-stride (`[CADDIAG] {json}`).

## Amendment 2026-09-05 — S4 met in BOTH regimes, and the CoT ratio's second exploit

**`hl_cot_coef=5` + `rel_standing_envs=0.20` is the first A1a arm to match or beat A0 in both
regimes at the benchmark speed** (`2026-09-02_17-34-56_a1a_fullmirror_cot5_standing20_s42`,
1 seed, all numbers re-measured in one session against a same-session A0):

| | A0 | c5_r20 | | | A0 | c5_r20 |
|---|---|---|---|---|---|---|
| WALK `act_legs` | 0.6161 | **0.94x** | | STAND `act_legs` | 0.0164 | **0.67x** |
| WALK `ajit_p95` | 62988 | **0.50x** | | STAND `ajit` mean | 377.0 | **0.93x** |
| WALK `jacc_p95` | 83.95 | **0.56x** | | STAND `jacc` mean | 0.726 | **0.95x** |
| WALK `ss_err_vx` | 0.0538 | **0.92x** | | STAND **touchdowns** | 128 | **128 (floor)** |
| WALK **CoT** | 0.5137 | **0.82x** | | STAND `double_support` | 0.997 | 0.998 |
| WALK `stride` | 0.578 | **1.00x** | | STAND `ajit_p95` | 63.9 | **11.1x** |

0 falls in both. Residual = the standing p95 (transient) channel, 3.3-11x, unclosed in every
arm ever measured. **`hl_cot_coef=7` never reaches the standing floor at any standing
fraction** (best 201 td), so the S5 cap is tighter from the standing side than the walking
side that placed it on 2026-09-01.

**A second exploit of the CoT ratio, and the fix.** The 2026-07-10 change stopped the HL
earning distance SIDEWAYS. It still earns distance by going FASTER than commanded, and the
incentive is steepest where the denominator is smallest:

| | signed `end_err_vx` | `hl_err_vx` | `ll_err_vx` |
|---|---|---|---|
| c5_r20 @ cmd 0.25 | **+0.1524** (runs ~0.40 for a 0.25 command) | **0.1975** | 0.1167 |
| c5_r20 @ cmd 0.50 | +0.0691 | 0.1224 | 0.1241 |
| `hl_cot_coef=0.2` @ cmd 0.25 | +0.0394 | 0.0777 | 0.0721 |

`hl_err > ll_err` at cmd 0.25 confirms the HL originates it. Consequence: **`ss_err_vx` is
4.5x A0 at cmd 0.25 and 2.2x at 1.0 while being 0.92x at 0.5.** The smoothness and energy
wins ARE speed-robust (`act_legs` 0.94-0.98x, `ajit_p95` 0.44-0.50x, CoT 0.68-0.82x over
0.25/0.5/1.0); only tracking is not, and `hl_cot_coef=0.2` at the same `rse` tracks A0-level
at all three, so the cause is the weight, not the standing fraction.

**`hl_cot_cap_commanded` (default False, byte-identical when off)** caps the credited
distance at the COMMANDED distance: `d = min(d_walked, d_commanded)` before the `d_floor`
clamp. Overshooting then costs energy and earns no further credit; under-shooting is
untouched, so `d_floor`'s anti-stall role and the 07-10 projection are unchanged. Formula
moved to `hrl_runner.window_cot()` for a test seam; `tests/test_window_cot.py` pins the
INCENTIVE (including `test_cap_off_lets_overshoot_pay`, which pins the defect itself).

**Generalizes:** a normalized reward whose denominator the agent influences is an incentive
to inflate that denominator. Check every direction it can be inflated — 07-10 closed
"sideways", this closes "faster". Ask the question once per axis of the denominator.

## Amendment 2026-09-07 — seed 123: half the 09-05 claim survives

Three seed-123 replicates, benched paired against their s42 twins with a same-session A0
anchor and the full 0.25/0.5/1.0 command sweep.

**Replicates on both seeds (`rse 0.20`):** **CoT 0.81x** (identical to 2 d.p.), `ajit_p95`
0.49x/0.48x, `jacc_p95` 0.56x/0.63x, standing `act_legs` 0.60x/0.70x (quieter than A0),
touchdowns 128/155 against A0's floor of 128, 0 falls.

**Does NOT replicate:** walk `act_legs` 0.93x/1.03x, `ss_err_vx` 0.88x/1.03x, `err_vy`
0.58x/1.15x — all STRADDLE parity. **The 09-05 wording "matches or beats A0 in BOTH regimes"
is withdrawn.** The defensible claim is: beats A0 on energy, commanded smoothness and standing
quietness; MATCHES it on walking magnitude and tracking within the seed spread. Note the
pattern — every MAGNITUDE survived, every ERROR-type metric moved, the same split as the
cross-session drift.

**S4's criterion is unaffected**: it is written on `cot` at A0-level tracking and fall_rate,
and `cot` is the metric that reproduced exactly (0.81x both seeds, and 0.77x twice on the
earlier `rse 0.10` arm). Scoring S4 on an energy magnitude rather than a velocity error was,
in hindsight, the right choice of gate quantity.

**The low-speed overshoot is CONFIRMED at n=6** — all six `cot 5` arms (3 standing fractions
x 2 seeds) sit at **3.13-5.37x A0 `ss_err_vx` at cmd 0.25** (mean 4.48x) while 0.64-1.03x at
0.5. Robust property of the weight, exactly as the denominator mechanism predicts. The
`hl_cot_cap_commanded` arm is now the highest-value pending run; pre-registration: cmd-0.25
`ss_err_vx` should fall from ~4.5x toward the ~1.1x that `hl_cot_coef=0.2` achieves, with CoT
staying <= ~0.85x and `ajit_p95` <= ~0.55x. If CoT regresses above A0, the cap removed the
pressure instead of redirecting it.

⚠ **`rel_standing_envs=0.15` is bimodal**: s42 breaks the WALKER (`ss_err_vx` 7.79x A0,
commanded period 0.884 s), s123 breaks the STANDER (td 1702). Same config, opposite failures.
2 of 6 `cot 5` runs failed, both in that cell — either 0.15 is a bifurcation point or the
family has a ~1-in-3 failure rate, and **n=2 per cell cannot separate these**. `rse 0.12` is
indistinguishable from 0.20 at n=2 (td 129/141 vs 128/155), so the operating point is
**0.12-0.20**, not 0.20 uniquely.

⚠ **Training stability:** `cot5_standing20_s123` died at iteration ~9124/10001 with an LL PPO
action-std blow-up (`normal expects all elements of std >= 0.0`) while reward and episode
length were healthy. Resumed from `model_9100` and cleared the same iteration immediately, so
it is a rare sampling event, not a poisoned state — but a keeper that diverges at ~9k
iterations is worth knowing before a deploy campaign.
