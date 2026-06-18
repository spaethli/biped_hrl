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

### #6 — Goal space in real-robot-measurable quantities (accelerations / IMU-derivable)
- **Idea.** Replace (or augment) the velocity goal with quantities the *real* robot can
  observe/track — accelerations, IMU-integrated velocity, base-frame quantities — so the
  HL→LL interface is physically grounded on hardware, not a sim-only abstraction.
- **Why it answers the critique.** A goal space the flat A0 command *cannot* express (A0 only
  takes `(vx,vy,wz)` command) means HL is no longer a relay — it shapes a richer reference.
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
- **gait frequency / phase** (from `feet_gait` period) → HL commands cadence.
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

### #8b — Train on *noisy* sensor velocity, not ground-truth (estimator-noise DR)
- **Idea.** Feed the policy/goal the **noisy/estimated** base velocity (DR-injected estimator
  noise + bias + latency) instead of privileged sim ground-truth, so it learns to be robust to
  the real robot's velocity-estimate error rather than overfitting a clean signal.
- **Why now.** Directly motivated by the sim-deploy finding (`HRL_plan.md` dashboard / A1 sim
  deploy memory): **the LL goes twitchy under injected estimator noise** → state-noise DR is the
  named sim2real next step. This is that DR.
- **Scope / care.** The *reward* may keep ground-truth velocity (privileged is fine for the
  critic/reward); it's the **observation/goal channel** that should see noise. Pairs naturally
  with Track F runs — add an `obs-noise` variant to any lean run to test robustness vs accuracy.
- **Touch points.** The velocity-bearing obs terms in `velocity_env_cfg.py` (`enable_corruption`
  / a noise model on the base-velocity / goal obs); the A1 goal-obs write in `hrl_runner`.

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

## Track F — Lean-reward A0/A1 pair (reward-shaping ablation) (#9)

- **Idea.** Retrain a **pair** — lean-A0 and lean-A1 — on a stripped reward (≈ `track_lin`,
  `track_ang`, `body_orientation_l2`, base-height; **floor:** keep `is_terminated` +
  `action_rate`/`joint_acc` so it can still find a gait). Compare lean-A0 vs lean-A1.
- **Why it answers the critique (the real rationale).** A0's 16-term reward is so well-shaped
  it already walks beautifully → no headroom for a hierarchy to show value. **Hypothesis: reward
  shaping masks the hierarchy benefit.** If lean-A0 degrades (jitter, falls) while lean-A1 holds,
  that is a concrete, defensible "the hierarchy helps" result for the supervisors. This is a
  different *operating point*, not a "more correct" reward.
- **Hard constraints.**
  1. **It is a pair (RQ2).** A1's env reward == A0's by construction; leaning out only A0
     reintroduces the confound A1 exists to avoid. Define one lean reward set, use it for **both**.
  2. **Warm-start cascade.** lean-A1's LL must warm-start from **lean**-A0 (not shaped-A0), else
     shaped-objective gait quality leaks in via the init → contaminated ablation. So lean-A0
     trains **from scratch**, then seeds lean-A1.
  3. **Budget-match the baseline.** A1 = warmstart(A0)+co-train, so the fair flat baseline is A0
     at the **matched total iteration budget**, not fewer. State as a comparison rule.
- **Risk.** A humanoid on tracking+height+orientation alone may never find a smooth gait (shaping
  exists because bipeds are hard) → both fail, no signal. Mitigate with the floor terms above.
- **Scope.** Keep the **shaped** A0/A1 as the deployment + primary pair; lean is a *controlled
  ablation alongside*, not a replacement.

### Implemented (2026-06-18) — 4-variant run matrix + one-command launcher
Tasks: **`Unitree-H1_2-Flat-Lean`** (lean A0), **`Unitree-H1_2-Flat-A1-Lean`** (lean A1).
Lean reward = A0's 16 terms with **7 shaping terms zeroed** (`foot_gait`, `foot_clearance`,
`foot_slip`, `soft_landing`, `angular_momentum`, `body_ang_vel`, `stand_still`); **kept** =
tracking + `body_orientation_l2` + `pose`(variable_posture, the height anchor) + floor
(`is_terminated`, `joint_pos_limits`, `action_rate_l2`, `joint_acc_l2`, `self_collisions`).
Single source: `apply_lean_reward()` in `config/h1_2/env_cfgs.py`; shaped A0 verified untouched.

One launcher fans out 4 jobs (all **4096 envs, 10001 iters**), `a1_from_lean` gated on
`a0_scratch` via SLURM `afterok`, the other 3 concurrent:

```bash
./train_h1_2_lean.sh submit   # on the cluster login node (uses 4 of your 16 GPUs)
```

| variant | task | init | answers |
|---|---|---|---|
| `a0_scratch`   | lean A0 | none (scratch)            | does lean reward alone produce a gait? (flat baseline) |
| `a0_polished`  | lean A0 | resume polished shaped-A0 | does a good gait *survive* when shaping is stripped? |
| `a1_from_lean` | lean A1 | warm-start `a0_scratch`   | clean lean hierarchy (LL from lean-A0) vs `a0_scratch` |
| `a1_polished`  | lean A1 | warm-start polished shaped-A0 | hierarchy from a good gait vs `a0_polished` |

- **Clean axis** = `a0_scratch` vs `a1_from_lean` (reward- and init-matched → isolates the
  hierarchy; the lean version of the original Track F pair). The `*_polished` pair shares a
  fixed polished-A0 init to ask "what does each architecture do with the same good gait."
- **Mechanics.** Lean-A0 reuses `experiment_name=h1_2_velocity`, so `a0_polished` warm-starts
  via `--agent.resume` from the polished run (continues the iter counter → its 10001 are
  *additional*). A1 variants use `--agent.warm-start-path` (fresh iter 0, gap-aware partial load).
  Override the polished checkpoint with `POLISHED_A0=…`.
- **Verdict metric.** Deterministic benchmark (`scripts/play.py <task> --checkpoint-file <pt>
  --num-envs 64 --eval-steps 600 --eval-seeds 2`): the hierarchy "earns its keep" if a lean-A1
  variant holds (lower `fall_rate` / smoother `act_rate`) where its matched lean-A0 degrades.
- **Optional.** Add the #8b estimator-noise DR to any variant to also probe robustness.

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
