# Hierarchy-benefit roadmap — ideas backlog (parallel tracks)

> **Purpose.** A parking lot for forward-looking A1 extensions, organized around the
> one question the supervisors raised (2026-06-18): **if the HL just forwards the velocity
> command on a coarser timescale, what does the hierarchy buy?** Each idea below is a
> hypothesis that answers either (a) *give the HL a richer job than velocity passthrough*,
> (b) *make the walking smoother / safer / non-falling* (the actual thesis deliverable), or
> (c) *make the goal/feedback real-robot-deployable*. Not a commitment — a backlog to return
> to. Shared machinery lives in `.claude/docs/hrl-infra.md`; A1 status in `doc/hrl/A1_HIRO.md`.

## The framing (read first)

The thesis deliverable is **a real H1-2 walking smoothly without falling on flat ground —
mainly straight, omnidirectional capable.** A1's current honest standing (`A1_HIRO.md` §"Why
two levels"): A1 ≈ A0 in-sim is *expected*, and today the hierarchy is justified only as a
**control + A2/A3 substrate + interface**, not as a locomotion improver. The supervisors find
that thin. So the strategic axis of this backlog is:

- **Make HL's job richer than passthrough.** If HL only emits `V*≈command`, it is a relay.
  It earns its keep when it commands something the flat command *cannot express* (a richer
  goal space) or optimizes a *different objective* on a *longer horizon* (reward split).
- **Or: prove a hierarchy benefit empirically** (OOD/robustness, the test already open in
  `A1_HIRO.md`) — a benefit without changing the design.
- Keep RQ2 clean: any change to A1's **env reward / dynamics** that doesn't also apply to A0
  confounds the A0-vs-A1 comparison (`.claude/docs/experiment-design.md`). Changing an
  **HL-internal objective or goal space** does *not* confound (the env reward is untouched) —
  this is the safe lever, same as the solved `hl_reward_mode=tracking` fix.

## Track map (what can run in parallel)

| Track | Ideas | Kind | Parallel? | Touches RQ2? |
|---|---|---|---|---|
| **A. Richer HL job / goal space** | #5, #6, #2b | training experiments | after spec | no (HL-internal) |
| **B. Smooth/safe reward terms** | #1, #2a, #3 | analysis → training | think now, run later | **yes if env reward** |
| **C. Real-robot feasibility** | #8, #6-probe | measurement (no training) | **now, fully parallel** | no |
| **D. LL reward restructure** | #4 | training + cleanup | self-contained | reduces a confound |
| **E. Upper-body decoupling** | #7 | new task — parked | future work | n/a (new task) |
| **F. Lean-reward A0/A1 pair** | #9 | training (2 runs) | after spec | **pair — applies to both** |
| **G. Warm-start alternatives** | #10 | training experiment | A3-family | breaks A1's "LL=A0" control |

**Suggested parallelism (your plan):** run **Track C** (IMU/measurement, no GPU) alongside
**Track B** (reward-term analysis, paper/no GPU) immediately — neither blocks the other. **D**
is the cleanest standalone training change. **A** is the scientifically highest-value but
needs C's verdict to choose the goal-space basis. **E** is parked.

---

## Track A — Give the HL a richer job (the core answer to the critique)

**Two axes for any candidate goal dimension (apply before adding one).**
- *Axis 1 — not flat-representable?* Can A0's `(vx,vy,wz)` command already express it? Only a
  "no" gives the HL a new job (else just command the LL directly) — this is the hierarchy
  justification.
- *Axis 2 — hardware-measurable?* Can the real robot estimate it to close the goal loop?
  Deployability gate (Track C / M5).
Add a dimension only if Axis 1 = no; it is deployable only if Axis 2 = yes. **#6's accel/IMU
basis is Axis-2-only** (a flat command expresses the same authority) → a deployability
*substitution* for the velocity basis, **not** a richer HL job. The richer-job prize is the
gait-spillover set below (cadence, step length, stance width).

### #6 — Goal space in real-robot-measurable quantities (accelerations / IMU-derivable)
- **Idea.** Replace (or augment) the velocity goal with quantities the *real* robot can
  observe/track — accelerations, IMU-integrated velocity, base-frame quantities — so the
  HL→LL interface is physically grounded on hardware, not a sim-only abstraction.
- **Reclassified (2026-06-24): Axis-2-only.** Accel/IMU-derivable velocity is the *same*
  authority a flat command already expresses → it does **not** make HL a richer job; it is a
  deployability substitution for the velocity basis (→ M5 / Track C), not a Track-A win.
- **Status / risk.** Changing the goal basis **re-opens the HL reward + target-map solution**
  (the `absolute` + `tracking` fix assumes velocity goals; `GoalSpace.center/to_target/to_g`
  would need a new map). Medium effort. Likely a **new architecture slot (A2/A3), not an A1
  tweak** — decide deliberately (see HRL_plan dashboard).
- **Depends on:** Track C verdict (is the measurable quantity accurate enough to close the loop?).
- **Single source of truth to edit:** `goal_components` in `config/h1_2_a1/rl_cfg.py`;
  `GoalSpace` in `rl/hrl/goal_space.py`.

### #5 — Extend the goal space with more LL-conditioning inputs
- **Idea.** Add goal dimensions (beyond the current 7-dim `velocity/orientation/height`) so the
  LL can be steered on more axes, freeing the HL to optimize "full reward" rather than tracking.
- **Why / caveat.** The original premise ("HL can't track on full reward unless the goal is
  richer") is now **partly stale**: the 2026-06-16 `tracking` fix already reached A0-level vx/vy
  *without* widening the goal. So only widen if a *new* deficiency (smoothness, posture, OOD)
  demands an axis the LL currently can't be told about. Note the documented clean negative:
  velocity-only 3-dim is worse, orient+height help as regularizers (`A1_HIRO.md` §Goal space).
- **Status / risk.** Low-medium. Each added component enlarges the HL action space → harder HL
  learning; justify each axis.

### #2b — Reward *split* that gives HL a genuinely higher-level objective
- **Idea.** Put long-horizon / task-level terms (tracking, progress, safety margin) in the **HL**
  objective and motor-level terms (action rate, joint limits, contact) in the **LL** — so the two
  levels optimize *different* things on *different* timescales. That division of labor is itself
  the hierarchy's justification.
- **Why it answers the critique.** A hierarchy with a true objective decomposition is not a
  relay; HL reasons about the task, LL about execution. This is the strongest *design* answer.
- **Status / risk.** HL-internal reward changes are RQ2-safe (env reward untouched). Needs a
  principled split — see Track B for the candidate term inventory. Connects to the already-planned
  `action_rate` penalty (`A1_HIRO.md`: must be a direct LL reward term, **not** the goal/intrinsic
  channel — smoothness isn't a goal-space state).
- **HL-reward fork (who chooses the gait).** Under the `tracking` HL reward the HL has *no
  gradient* to choose a richer goal (cadence/step length don't improve velocity tracking). Each
  such dimension needs either (a) **exogenous command** (curriculum/operator sets it → pure LL
  tracking, no new HL reward, HL isn't doing the new job) or (b) **a downstream reward the gait
  serves** (energy/robustness) = the genuine richer-HL-job and the source of any new HL reward term.

---

## Track B — Reward terms for smooth, non-falling walking (the thesis deliverable)

> Spec-first (CLAUDE.md): add terms only against a **measured** deficiency, not speculatively.
> Known gaps today (`A1_HIRO.md` Current results): act_rate 1.68 vs A0 0.66 (jerky), orient_dev
> 0.13 / height_dev 0.11 vs A0 ~0.03 (posture). These are the evidence that justifies Track B.

### #1 — Re-inject reward terms stripped during the tracking-reward simplification
- **Idea.** The HL `tracking` reward deliberately dropped LL-execution penalties. Selectively
  reintroduce stripped terms (smoothness, posture, energy) **into the LL** (their natural home).
- **Why.** Directly targets the measured smoothness/posture gap = transfer liability.
- **RQ2:** if a term goes into the **env reward**, it must also apply to A0 or it confounds —
  flag for the user before applying (see `feedback-a0-comparison-cleanliness` memory).

### #2a — Mix A0's reward terms across HL vs LL (the inventory side of #2b)
- **Idea.** Take A0's full reward term list and decide, per term, whether it belongs at HL or LL.
- **Status.** Pairs with #2b; the design lives in Track A, the term-by-term inventory lives here.

### #3 — Reward-term catalog from sibling repos — **DONE (survey 2026-06-18)**
Surveyed the `~/ramlab_ws/src/` locomotion repos (IsaacLab + Spot, unitree_rl_gym,
unitree_rl_lab, h12_locomotion_rma, h12_loco_manipulation, mjlab, our own
unitree_rl_mjlab). ~66 raw terms, deduped below.

**Baseline-aware caveat (important).** A0 is **already richly shaped — 16 terms**, and most
"creative" terms the survey surfaced are *already in it*: `variable_posture` (speed-dependent
posture, the `pose` term), `feet_gait` (gait-phase matching, +0.5), `soft_landing`,
`foot_slip`, `foot_clearance`, `angular_momentum_penalty`, `body_angular_velocity_penalty`,
`stand_still`, `self_collision_cost`, plus `track_lin/ang_vel`, `body_orientation_l2`,
`joint_acc_l2`, `joint_pos_limits`, `action_rate_l2`, `is_terminated`
(`src/tasks/velocity/velocity_env_cfg.py:262-347`, h1_2 override adds self-collision).
**→ Don't double-add these.** The real levers are (i) the few genuinely-absent terms below,
(ii) **routing existing terms HL vs LL** (#2), and (iii) terms that imply **new HL goal
dimensions** (→ Track A, the hierarchy-justifying prize).

**Candidate terms NOT already in A0** (Use = value for *our* flat smooth/safe sim2real walk;
Cre = creativity/novelty; Port = implementation cost given we already reuse mjlab):

| Term | Gist | Use | Cre | Home | Source / port |
|---|---|---|---|---|---|
| **2nd-order action smoothness** | penalize `a_t−2a_{t-1}+a_{t-2}` (jerk, not just rate) | **H** | M | LL | h12_loco_manip; trivial |
| **energy / joint_power** | `Σ|τ·q̇|` (opt. ÷‖cmd‖) — mechanical power | **H** | M | LL | rma/loco-manip; trivial |
| **per-group joint_deviation** | separate posture penalty for arms / waist / hips | **H** | M | LL + **Track E** | rl_lab/loco-manip; small |
| **joint_mirror** | `Σ(q_L−q_R)²` enforce L/R gait symmetry | M-H | **H** | LL | rl_lab/rma; small |
| **no_fly / single-stance** | reward exactly one foot airborne | M-H | M | LL | loco-manip; small |
| **feet_distance_lateral / too_near** | keep stance width in `[min,max]` (anti cross-step) | M | M | LL | loco-manip/rma; small |
| **air_time_variance** | `var(air/contact time)` step symmetry | M | M | LL | rl_lab/rma; small |
| **action_vanish** | penalize actions saturating at bounds (anti bang-bang) | M | M-H | LL | loco-manip; trivial |
| **feet_swing_height** | peak swing height vs target, scored at landing | M | M | LL | **in mjlab, reuse**; not in our A0 |
| **feet_stumble** | `‖F_xy‖ > k·|F_z|` → toe-stub/scrape detect | L-M | M | LL | rma/gym; small (flat ⇒ low value) |
| **base_lin_vel ramping** | velocity-dependent tracking gain (harder at speed) | M | M-H | HL shaping | IsaacLab Spot; small |
| **motion_*_body_*_error_exp** | per-limb pose/vel tracking (Gaussian) | **H** | **H** | **Track E** | **mjlab `tracking` task — 0-cost reuse** |
| **staged reaching/bringing** | EE→object→goal manipulation reward | M | H | **Track E** | mjlab `manipulation` — reuse |

**Top picks for the A1 smoothness gap** (act_rate 1.68 vs A0 0.66): **2nd-order action
smoothness** + **energy/power** are the most on-target and near-free (both target jerk/effort
directly, both are LL terms, both ~trivial). **joint_mirror** is the highest-creativity add
(symmetric gait — sample-efficiency + natural-looking + sim2real friendly).

**Goal-dimension spillover (→ Track A).** Several terms describe a *reference the HL could set*
that A0's `(vx,vy,wz)` command cannot express — this is the strongest answer to the supervisors:
- **gait frequency / phase** (from `feet_gait` period) → HL commands cadence. **Implementable
  now, no footfall detector:** promote the `foot_gait` phase-clock `period` (rewards.py:190,199)
  to a goal dimension; the existing reward does the LL-side tracking (open-loop reference clock).
  **Step length** instead needs footfall — free in sim (`compute_first_contact` rewards.py:246 +
  foot pos; `current_contact_time/air_time` :144), but on hardware the open gate (force sensor /
  torque-GRF / learned contact estimator), Axis-2/M5. Commanding cadence/step-length keeps this
  **A1** (RL-learned gait); it's **A3** only once a model-based gait library is a permanent runtime
  component with LL residual.
- **body height / crouch** (base-height-as-command) → HL commands posture height (already a goal
  component; make it *actively modulated*, not fixed).
- **stance width** (from `feet_distance_lateral`) → HL commands lateral foot spacing.
- **posture-stiffness regime** (`variable_posture` std regime) → HL commands how tightly posture
  is held. If the HL sets any of these, it is no longer a velocity relay.

**Track E enrichment.** The mjlab `tracking` task already implements per-limb
`motion_relative_body_position/orientation_error_exp` and
`motion_global_body_linear/angular_velocity_error_exp`
(`mjlab/src/mjlab/tasks/tracking/mdp/rewards.py:44-113`) + staged manipulation rewards
(`.../manipulation/mdp/rewards.py`). These are the ready-made machinery for upper-body /
arm / end-effector tracking — Track E is a *reuse*, not a from-scratch build.

---

## Track C — Real-robot deployment feasibility (parallel, no training)

### #8 — IMU speed feedback: is it accurate enough?
- **Idea.** Measure how well IMU-derived (integrated-accel / estimator) base velocity matches
  ground-truth in sim, and assess drift — i.e. can the real robot supply the velocity signal the
  policy/goal needs, or does integration drift break it?
- **Why.** Gates whether the velocity goal space (and #6's accel variant) is deployable at all.
- **Kind.** A *measurement*, not a training change — **start now, fully parallel** with Tracks B.
  Likely realizable with existing sim sensors + `play.py` logging; no new architecture.
- **Feeds:** the #6 decision (velocity vs acceleration basis) and the deployment milestone.

### #8b — Train on *noisy* sensor velocity, not ground-truth (estimator-noise DR) — **VALIDATED (2026-06-22)**
- **Idea.** Feed the LL goal a **deploy-realistic base-velocity estimate** (DR: bias + drift +
  lag) instead of privileged sim ground-truth, so it is robust to the real onboard estimator
  (`rt/sportmodestate`) rather than overfitting a clean signal. Motivated by the sim-deploy
  finding (LL twitchy under injected estimator noise — `HRL_plan.md` dashboard).
- **As-built.** `GoalStateNoise` (`rl/hrl/state_noise.py`) corrupts only the estimator-supplied
  goal columns — base lin-vel `vx,vy` (yaw-rate is a clean gyro read; height optional, clean by
  default). **Asymmetric/privileged:** reward + HL stay on ground-truth; only the LL's observed
  goal `V*−s` sees noise (one filter step/env-step; same per-step offset carried to the post-step
  refresh). Off by default → byte-identical to the clean pipeline (RQ2-safe). Config =
  `GoalStateNoiseCfg` in `config/h1_2_a1/rl_cfg.py` (`--agent.goal-state-noise.*`); mechanism in
  `.claude/docs/hrl-infra.md`.
- **Both target maps trained (2026-06-23 update).** Both `absolute` and `delta` were run under noise
  (earlier "absolute-only" plan revised — the faithful-DR `_target_obs` fix made delta meaningful).
  In `delta` a constant bias cancels in `V*−s` by construction (`V*=v_est(t0)+g`), so bias-DR is a
  near no-op there (delta_bias≈delta_full); in `absolute` the bias persists. **Both maps are
  deployment-faithful** — noise enters via the LL's observed `V*−state_n`, exactly as on the real
  robot; the bias-cancellation in delta is a construction property, not a faithfulness edge.
- **Run matrix** (launcher `train_h1_2_noise.sh`, TD3+HIRO+tracking, warm-start polished-A0,
  4096 envs / 10001 it): `{abs,delta} × {bias ±0.10 m/s, full = bias + OU drift 0.01/0.99 + lag 3}`,
  seeds 42 & 123. **Verdict: all four reliable** (2/2 clean each) at clean-baseline quality (0 falls,
  no saturation). Absolute err_vx ~0.080 / yaw ~0.185; directional err_vx ~0.120 / yaw ~0.163 / smoother
  (act 2.8 vs 3.9) — the **same HL-side vx↔yaw trade-off as clean Track F**: absolute nails vx, directional
  nails yaw; LL reach near-perfect in both (HL is the wall). Full tables + verdict → `A1_findings.md`
  (Estimator-noise DR #8b). One early `abs_bias` collapse was a transient corruption, not the seed.

---

## Track D — LL reward restructure so `fell_over` need not be a timeout (#4)

- **Idea.** Replace the LL's negative goal-*distance* reward with a strictly-positive kernel,
  e.g. `exp(-‖V*−s‖²)` (Gaussian/RBF). A positive reward removes the **suicide attractor** that
  forced `fell_over=time_out` for A1 (the negative reward made early termination *preferable* →
  needed the bootstrap hack, `config/h1_2_a1/env_cfgs.py`).
- **Why it's attractive.** Lets `fell_over` go back to a **true terminal** (as in A0) →
  **reduces** an A0-vs-A1 divergence rather than adding one. Good RQ2 hygiene, self-contained,
  fast to test.
- **Status / risk.** Low. Re-tune the kernel width; re-validate with the oracle-HL control that
  the LL still tracks (it proved 0.05 m/s under the current reward). Touch points: the intrinsic
  reward computation in `rl/hrl/hrl_runner.py`, the `fell_over` term flag in
  `config/h1_2_a1/env_cfgs.py`.
- **Interaction:** changes the LL reward → coordinate with Track B (#1 re-injected LL terms ride
  on top of this kernel).

---

## Track E — Upper-body decoupling for manipulation-while-walking (#7) — PARKED

- **Idea.** Decouple upper-body control to prepare for manipulation tasks during walking.
- **Verdict.** **Out of current scope.** This is a *different task* (new observations, new
  reward, arguably a new thesis chapter), not an A1 tweak — high effort, muddies the locomotion
  A0–A4 comparison. Park as explicit future work unless it becomes a stated deliverable. The A1
  goal interface (`A1_HIRO.md` §"Why two levels", point 3) is the natural attach point if revived.

---

## Track F — Lean-reward A0/A1 pair (reward-shaping ablation) (#9) — RESULTS IN

**Purpose.** A0's 16-term reward is so well-shaped it already walks beautifully → no headroom
for a hierarchy to show value. Strip the shaping and compare lean-A0 vs lean-A1 to test whether
**reward shaping was masking the hierarchy benefit**. A controlled ablation *alongside* the
shaped pair (still deployment/primary), not a replacement.
**RQ2 constraints:** one lean reward set for **both** (leaning only A0 reintroduces the confound);
lean-A1's LL warm-starts from **lean**-A0 not shaped-A0 (else shaped gait leaks in via init);
compare at **matched iteration budget**.

### Implemented (2026-06-18) — tasks + launcher
Tasks **`Unitree-H1_2-Flat-Lean`** (A0) / **`Unitree-H1_2-Flat-A1-Lean`** (A1); single source
`apply_lean_reward()` in `config/h1_2/env_cfgs.py` (shaped A0 untouched). Launch
`./train_h1_2_lean.sh submit` → 4 jobs @ 4096 envs ×10001 iters, `a1_from_lean` gated on
`a0_scratch` (SLURM `afterok`). `a0_polished` resumes the polished run (`--agent.resume`, iters
*additional*); A1 uses `--agent.warm-start-path` (gap-aware partial load). Override via `POLISHED_A0=…`.

| variant | init | role |
|---|---|---|
| `a0_scratch`   | scratch                       | flat lean baseline |
| `a0_polished`  | resume polished shaped-A0     | does a good gait survive shaping removal? |
| `a1_from_lean` | warm-start `a0_scratch`       | clean axis (reward+init-matched) vs `a0_scratch` |
| `a1_polished`  | warm-start polished shaped-A0 | hierarchy from a good gait vs `a0_polished` |

**Two reward variants (keep distinct).** **semi-lean** (2026-06-19) zeros 7 shaping terms
(`foot_gait`, `foot_clearance`, `foot_slip`, `soft_landing`, `angular_momentum`, `body_ang_vel`,
`stand_still`) but **keeps** the `action_rate_l2`/`joint_acc_l2` smoothness floor — asymmetric,
since A1's LL never sees env reward (`ll_task_reward_coef=0`; all env terms diagnostic-only for
A1 → `a1_reward_routing` memory + hrl-infra reward decomp). **true-lean** (2026-06-22) also zeros
those two → neither side has smoothness guidance (fair read). Kept both: tracking +
`body_orientation_l2` + `pose`(height anchor) + safety floor (`is_terminated`, `joint_pos_limits`,
`self_collisions`).

### Results (benchmark: 64 envs × 600 steps × 2 seeds; equal 10k lean budget)

**semi-lean** (2026-06-19; `action_rate_l2`/`joint_acc_l2` still active):

| variant | err_vx | err_vy | err_yaw | fall_rate | act_rate | orient_dev | height_dev |
|---|---|---|---|---|---|---|---|
| `a0_scratch`   | 0.085 | 0.075 | 0.087 | 0.00 | 0.62 | 0.031 | 0.021 |
| `a0_polished`  | 0.085 | 0.098 | 0.090 | 0.00 | 0.48 | 0.036 | 0.023 |
| `a1_from_lean` | 0.065 | 0.045 | 0.205 | 0.00 | 2.90 | 0.061 | 0.133 |
| `a1_polished`  | 0.070 | 0.050 | 0.160 | 0.00 | 2.62 | 0.067 | 0.138 |

**true-lean** (2026-06-22; `action_rate_l2`/`joint_acc_l2` zeroed — fair smoothness
read). `a1_polished_delta` = A1 from polished with **directional** (delta) HL goals
`V*=state+scale·g` vs the default **absolute** `V*=center+scale·g`:

| variant | goals | err_vx | err_vy | err_yaw | fall_rate | act_rate | orient_dev | height_dev |
|---|---|---|---|---|---|---|---|---|
| `a0_scratch`        | —        | 0.133 | 0.097 | 0.264 | 0.00 | 35.20 | 0.023 | 0.030 |
| `a0_polished`       | —        | 0.084 | 0.098 | 0.283 | 0.00 | 20.22 | 0.040 | 0.029 |
| `a1_from_lean`      | absolute | 0.118 | 0.056 | 0.226 | 0.00 | 22.15 | 0.209 | 0.074 |
| `a1_polished`       | absolute | 0.088 | 0.057 | 0.186 | 0.00 |  2.47 | 0.112 | 0.164 |
| `a1_polished_delta` (06-22) | **directional** | 0.091 | 0.081 | **0.128** | 0.00 | **2.11** | 0.115 | 0.104 |
| `a1_polished_delta` (06-23) | **directional** | 0.143 | 0.083 | 0.124 | 0.00 | 2.46 | 0.083 | 0.128 |
| `a1_from_lean_delta` (06-23) | **directional** | 0.184 | 0.175 | 0.297 | 0.00 | 22.59 | 0.154 | 0.083 |

Goal-achievability probe (`--diagnose-goals 600 --eval-seeds 2`, A1 only; c=8, no
saturation `sat=0.000` in any):

| variant | reward set | goals | HL err vx | LL err vx | HL err yaw | LL err yaw | vx follow ratio |
|---|---|---|---|---|---|---|---|
| `a1_from_lean` | semi-lean | absolute | 0.027 | 0.035 | 0.171 | 0.130 | −0.28 |
| `a1_polished`  | semi-lean | absolute | 0.037 | 0.038 | 0.140 | 0.081 | +0.59 |
| `a1_from_lean`      | true-lean | absolute | 0.102 | 0.091 | 0.173 | 0.168 | — |
| `a1_polished`       | true-lean | absolute | 0.041 | 0.075 | 0.161 | 0.099 | — |
| `a1_polished_delta` (06-22) | true-lean | **directional** | 0.067 | 0.048 | 0.082 | 0.051 | — |
| `a1_polished_delta` (06-23) | true-lean | **directional** | 0.124 | **0.026** | 0.120 | 0.069 | — |
| `a1_from_lean_delta` (06-23) | true-lean | **directional** | 0.180 | 0.228 | 0.230 | 0.232 | — |

**Read.** *Semi-lean:* A1 keeps a translational edge (vx 0.065–0.070 / vy 0.045–0.050 vs A0
~0.085 / ~0.075) but *appears* ~5× jerkier — confounded (A0 penalized for jerk, A1's LL not).
*True-lean (the fair read, 2026-06-23) — smoothness story flips:* with no smoothness penalty
either side, flat **A0 act_rate blows to 20–35** (was 0.5–0.6 in semi-lean — it relied on
`action_rate_l2`) while **A1-from-polished stays ~2.1–2.5 (~10× smoother)** → the hierarchy's
intrinsic-goal LL is *inherently* smoother than flat PPO. Caveats: (1) **init dominates** —
`a1_from_lean` inherits the wild `a0_scratch` (act 22, orient 0.21), so the clean axis is
contaminated under true-lean; the smooth A1 is the *polished*-init one (carries shaped gait in).
(2) **directional goals → near-perfect LL, HL becomes the wall** — both `a1_polished_delta`
runs give the lowest LL reach errs of any A1 (vx 0.026–0.048, yaw 0.051–0.069); end error ≈ HL
error, so the directional LL nails its goals and the HL is the limiter. Best yaw of the set
(0.124–0.128). Run-to-run spread is large (two same-config polished_delta runs: err_vx 0.091 vs
0.143 — the documented same-config divergence). Directional does **not** rescue the scratch
init: `a1_from_lean_delta` fails like its absolute twin (vx 0.184, act 22.6, LL reach blown to
0.23) → **init dominates regardless of goal mode**. ("directional" = HIRO's term for
`V*=state+scale·g`, `goal_space.py:10-11`; `delta` = the `hl_target_mode` id.)

**Directional vs absolute — settled axis trade-off (clean + noise, 2026-06-23).** The same split
holds in clean true-lean *and* under estimator-noise DR (Track C #8b matrix): **absolute** goals give
better straight-line vx/vy (HL maps vx well, yaw poorly); **directional** goals give better yaw/turning
and smoother action (HL maps yaw well, vx loosely). The LL reaches either goal near-perfectly, so the
HL is always the limiter and the goal mode just selects which command axis the HL is good at. Neither
dominates — pick absolute for the "mainly straight" headline metric, directional for turning/smoothness.
Both train reliably with noisy velocity estimates. Decomposition tables → `A1_findings.md` (#8b verdict).

## Track G — Warm-start alternatives: model-based gait warm-start (#10)

- **Idea.** Replace the A0-checkpoint warm-start with a **model-based gait warm-start** — imitate
  a reference gait (trajectory-opt / analytic walker / IK gait) to initialize the policy, then RL
  takes over.
- **Distinct from A3 (the answer to "is this too close to A3?").** Axis = *does the gait model
  stay in the loop at runtime?* **Warm-start:** gait only **initializes** weights, then is
  **discarded** — deployed policy is standalone. **A3 (NaviGait):** gait library is a **permanent
  runtime component**, RL outputs a residual every step. So a gait warm-start is **not** A3.
- **But a bad fit for A1.** Swapping A0-warmstart → gait-warmstart **breaks A1's load-bearing
  "LL = A0 PPO" control** (`A1_HIRO.md`), confounding the A0-vs-A1 comparison.
- **Better home: an A3 ablation.** Frame as *"is the runtime residual necessary, or does
  gait-imitation pretraining + pure RL suffice?"* → strengthens A3 rather than competing. Park
  here; revisit when A3 starts.

## Open decisions before any track starts

1. **Architecture accounting:** does #6 (and possibly #5) become **A2/A3**, or stay an A1
   variant? This sets whether RQ2-cleanliness applies and which doc/task ID it gets.
2. **Goal basis for #6:** velocity vs acceleration — **blocked on Track C (#8) verdict.**
3. **Reward home (#1/#2):** which terms are HL-internal (RQ2-safe) vs env-level (needs A0 parity)?
4. **Is the OOD/robustness benefit test** (already open in `A1_HIRO.md`) worth running *before*
   adding complexity — it might answer the supervisors with the *current* A1 and no new code.
</content>
</invoke>
