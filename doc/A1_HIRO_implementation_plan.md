# A1 — HIRO (Hybrid PPO + TD3) Implementation Plan

> **Status:** Milestones 1 & 2 DONE (scaffolding + oracle-HL LL validated & stable).
> **Milestone 3 — `hl=ppo` IMPLEMENTED & TESTED: naive on-policy ppo FAILS to track
> (4 runs; credit-assignment + co-training instability).** See §0 "M3 experiment results".
> **Milestone 4 — `hl=td3 relabel=none` IMPLEMENTED (2026-06-11): `HighLevelTd3` +
> `HLReplayBuffer`; smoke-tested; first 5k run `a1_td3_5k` launched.** See §0 "M4".
> **Milestone 5 — `hl=td3 relabel=hiro` IMPLEMENTED & RUN (2026-06-16): relabeling helps
> translational tracking (best vx/vy of any TD3 variant) but a c-sweep {4,8,12} shows
> HL cadence/step-count is NOT the limiter — the vx/vy wall (~4–5× A0) is STRUCTURAL.**
> See §0 "M5 + c-sweep". Next: goal-achievability probe (`doc/A1_goal_achievability_probe.md`).
> **Scope:** Architecture A1 of the thesis (structural hierarchy). Hybrid design:
> on-policy PPO low level + off-policy TD3 high level with HIRO goal-relabeling
> correction. Relabeling is a runtime toggle (enables a clean ablation).
> **Framework:** stays in `unitree_rl_mjlab` + `mjlab` + `rsl_rl`. No Isaac Lab.

---

## 0. CURRENT STATE & WHERE TO RESUME (updated 2026-06-10)

**Milestones 1 & 2 are DONE. The oracle-HL low level is validated and stable. Next
is Milestone 3 (`hl=ppo`).** The implementation diverged from the original plan
(§1–§13 below) in several important ways. **This section is authoritative where it
conflicts with the older sections.**

### What's built and working
- Task `Unitree-H1_2-Flat-A1` registered (`config/h1_2_a1/{__init__,env_cfgs,rl_cfg}.py`).
- `HierarchicalRunner` (`rl/hrl/hrl_runner.py`) **inherits `VelocityOnPolicyRunner`**
  (so it gets the ONNX export). Drives the co-train loop, A0 warm-start, save/load.
- `HighLevel` ABC + `OracleHighLevel` (`rl/hrl/high_level.py`). Interface:
  `act(env,obs,state)→V*`, `begin_window`, `accumulate(task_rew)`, `end_window`,
  `update`, `state_dict/load_state_dict`, `train_mode/eval_mode`.
- Declarative goal space `rl/hrl/goal_space.py` (**replaces** the planned `goal_env.py`),
  now with `GoalSpace.scale(env)` (per-dim delta scale; velocity tracks the twist
  curriculum) for the learned-HL HIRO map `V* = state + scale*g`.
- `HighLevelPpo` (`rl/hrl/high_level.py`, M3) — a 2nd `rsl_rl.PPO` at the HL timescale,
  slotted into the existing `HighLevel` interface (no co-train-loop changes beyond
  threading `extras` into `end_window` for truncation bootstrap). Smoke-tested: trains,
  warm-starts, saves `hl` state, resumes, exports ONNX.

### Key deviations from the original plan (read these)
1. **Warm-start is the LOAD-BEARING fix** (plan §12.3 said train from scratch — WRONG
   in practice). On-policy PPO can't discover walking from scratch in feasible time.
   The LL is warm-started from a converged **A0 actor** via a *gap-aware* partial
   state-dict copy: the `command` term sits **mid-vector (cols 6:9), not last**, so
   the copy drops A0's command cols and shifts the rest (A1 `[0:6]←A0[0:6]`,
   `[6:89]←A0[9:92]`, goal cols fresh). Critic aligns cleanly (keeps command). This
   is THE enabler: iter-0 ep_len jumps to hundreds; A1 even **out-tracks** A0.
   `--agent.warm-start-path <A0 model.pt>`. Diagnostic it worked: high iter-0 ep_len.
2. **Goal space is declarative & configurable; evolved 3→7→3 dims.** `goal_components`
   list is the single source of truth (set in `rl_cfg.py`; the env derives its goal
   obs dim from it via `__init__.py`). `goal_dim` is always derived, never hardcoded.
   History: started velocity-only (3), expanded to **velocity+orientation+height (7)**
   to inject survival signal, then ran a **velocity-only (3)** ablation.
   **Velocity-only ablation result (CONFIRMED, full 10k, 1 seed each; default 7-dim
   run `2026-06-09_16-22-40` vs velocity-only `2026-06-10_08-13-49`+resume, same
   warm-start/envs/entropy — only goal_components differs):**
   - Survival IDENTICAL: both ep_len ~1000, falls ~0 → orient/height NOT needed for
     survival (warm-start+bootstrap carry it).
   - **Default (7-dim) tracks BETTER**, esp. yaw: @10k err_yaw 0.47 vs 0.62 (~33%),
     err_xy 0.325 vs 0.365 (~12%); and the default policy is calmer (action std 0.87
     vs 1.30). Gaps consistent over iters 5k–10k (yaw & std gaps robust; the err_xy
     gap is within plausible 1-seed noise).
   - Interpretation: orientation+height act as **stabilizing regularizers** — they
     anchor the torso upright/at height (even though proj-gravity is yaw-invariant),
     which lowers std and makes velocity (esp. the hard yaw axis) easier to track.
   - **DECISION: keep the default 7-dim goal space.** Velocity-only is a clean
     negative result for the writeup ("richer base-pose goal isn't needed for
     survival but materially improves tracking quality, yaw most").
3. **Encoding = ABSOLUTE target + directional observation** (oracle), NOT the planned
   `V*=v_t+g` delta. Oracle emits absolute `V*` = command velocity (+ nominal upright/
   height for non-velocity comps). The LL **observes** the remaining delta `V*−s_i`
   and is rewarded `−Σ_c w_c‖V*−s_{i+1}‖`. For the **learned** HL the chosen encoding
   is **HIRO delta** `V*=s_t+scale·g`, `scale` read from the twist curriculum
   (confirmed decision). For the oracle, absolute and delta coincide; the difference
   only bites for a learned HL. (See §`Canonical HIRO` note at end of §0.)
4. **Suicide fix: `fell_over=time_out` (NOT in original plan).** The LL's always-
   negative goal-distance reward made early termination an attractor (die fast → stop
   accumulating negative reward; HIRO's domains never terminate so they never hit
   this). Fix: mark `fell_over` as a **truncation** (`time_out=True`) so PPO bootstraps
   it (`γ·V`) instead of cutting value to 0. Set in the **A1 env cfg ONLY**
   (`config/h1_2_a1/env_cfgs.py`); A0 keeps a true terminal (A0 diverges with
   time_out). One-line toggle for the no-bootstrap ablation.
5. **Hyperparams that matter:** `entropy_coef=0.005` (0.01 lets action std blow up to
   ~2 and collapse), `desired_kl=0.005`, velocity goal weight 3, `num_steps_per_env=24`,
   `c=8`, `gamma_hi=0.99**8`. LR is adaptive and tends to sit near its floor (1e-5).

### Results so far (oracle HL)
- **Survival SOLVED:** ep_len ~1000, falls ~0, stable to **10k iters** (two long runs,
  no collapse) with the newest small-interval A0 warm-start.
- **A1 LL OUT-TRACKS its A0 warm-start:** err_xy 0.56→0.33, err_yaw 0.81→0.49 — a real
  positive signal for the hierarchy (goal-conditioned LL beat the flat baseline).
- One collapse (run `11-36-15`, a *different* warm-start) at ~iter 2400 — the
  "deferred-suicide" idea (bootstrap only defers) is an **UNPROVEN hypothesis**; two
  long runs held, so it's likely warm-start-specific or stochastic. Don't assume the
  bootstrap is inadequate. Candidate fix only if it recurs: positive reward / alive
  bonus (not adopted; decision: stay close to canonical HIRO).
- std creeps (0.33→0.87 over 10k) but benign at entropy 0.005.
- Inherited A0 weakness: **poor yaw tracking → circling** (yaw is unconstrained by the
  orientation/height goals, which are yaw-invariant; yaw command range is widest).
  Controlled-for in the A0-vs-A1 comparison.
- Reproducibility: same-config runs diverge a lot (GPU non-determinism + RL chaos);
  `num_envs` is effectively a hyperparameter — hold it fixed within a comparison set,
  use ≥2 seeds.

### Milestone 3 — `HighLevelPpo` (IMPLEMENTED; naive ppo FAILS to track — see results)
Slots into the **existing** `HighLevel` interface + `hrl_runner` co-train loop.
`HierarchicalRunner._make_high_level()` returns it for `hl_algorithm=="ppo"`. Config:
`HlPpoCfg` (`config/h1_2_a1/rl_cfg.py`, field `hl_ppo`).
- A 2nd `rsl_rl.PPO`. HL actor obs = `("policy","command")` — the HL **sees** the
  command (it's the task input); action head dim = `goal_dim`. Own `RolloutStorage`
  sized `num_steps_per_env//c` (= 3) per env.
- `act(env,obs,state)`: sample goal `g`, store HL transition (obs/action/value/logprob),
  return `V* = state + scale·g` (HIRO delta). `begin_window` zeroes the window-reward;
  `accumulate(task_rew)` sums the A0 task reward; `end_window` finalizes (reward=Σ, done,
  + `γ_hi·V` truncation bootstrap when the step extras carry `time_outs`) and pushes to
  the HL `RolloutStorage`; `update()` = `compute_returns(γ_hi)` + PPO `update`, returning
  a loss dict logged under `hl/` (incl. `hl/goal_abs_mean`).
- `GoalSpace.scale(env)` added (per-dim; velocity = half of the live twist command
  ranges, orientation = 1.0, height = 0.2 m).

**Deviations from this section's original sketch (now the truth):**
1. **Linear scaling, NOT tanh.** §5a said "GaussianDistribution squashed/scaled". A
   tanh squash needs a log-prob Jacobian correction that rsl_rl's `GaussianDistribution`
   does **not** apply → it would be a silent bug. `g` is the raw Gaussian sample and
   `scale` linearly maps it. `scale` (≈ command half-range) is the goal's effective
   reach; tune it if `g` magnitudes (logged as `hl/goal_abs_mean`) look off.
2. **HL critic obs = `("critic",)`, not `("critic","command")`.** The `critic` group
   **already contains** the command term (A0 critic, unchanged), so appending `command`
   would duplicate it.
3. **One small loop change:** `end_window` now takes `extras` (for the `time_outs`
   bootstrap). The ABC default ignores it (oracle unaffected).

**Known approximation (M3):** windows are always exactly `c` steps; a mid-window episode
reset is not closed early — the window's done/reward use the window-end step. Acceptable
for the naive baseline (matches the existing oracle loop); revisit if HL credit
assignment looks weak.

### M3 experiment results — naive `hl=ppo` does NOT track (4 runs, 2026-06-10/11)
All: 4096 envs, A0 warm-start `2026-06-09_08-16-27/model_10000.pt`. The learned HL
**never learns the command→goal mapping**; it lands in one of two failure basins (or
oscillates between them). Diagnosis runs (W&B project `biped_hrl`):

| run | knobs | outcome |
|---|---|---|
| `a1_ppo_hl_5k` | ent 0.005, **velocity-only (3-dim)** | survive (ep_len ~1000) but **don't track** (err_xy ~1.9, yaw ~2.1). HL entropy collapsed +4.3→−2.1 (std ~0.075), goals shrank → "emit ~no change". |
| `a1_ppo_7dim_ent02_5k` | ent 0.02, 7-dim | std **blew up** (entropy 9.9→19, goal_abs 8.7) → impossible targets → **robot died** (ep_len ~20). Aborted ~it1500. |
| `a1_ppo_7dim_ent01_stdcap_5k` | ent 0.01 + **std cap** `(1e-3,1.0)` | **stable & survives** (ep_len ~980), goals bounded (~2) — but **still no track** (err_xy ~1.3, yaw ~2.9, worsening). Aborted ~it2000. → exploration/survival are NOT the bottleneck; the HL *mean* won't learn to track. |
| `a1_ppo_track4x_nocap_2k` | ent 0.01, cap OFF, **+4× track_lin/track_ang weight** (A1 env) | **oscillated.** Best tracking-while-alive any ppo got (it200–800: err ~0.2 @ ep_len ~100–155) → goals grew 1.5→3.6 → **constant falling** (fell_over ~88, ep_len ~46) → survival recovered but **stopped tracking** (final err_xy ~1.05, yaw ~1.74). Never held survive+track. |

**Conclusion:** naive on-policy `hl=ppo` hits a **credit-assignment + co-training-
instability wall.** The HL gets only **3 transitions/env/iter** (`num_steps_per_env//c`)
and the dense task reward's small tracking term is swamped by penalties (`joint_pos_limits`
~−3.6, `action_rate` ~−3.4) → the HL mean drifts to "stay put". Upweighting tracking (4×)
creates the incentive (run D's early phase proves it) but on-policy co-training then
oscillates between *track-and-fall* and *survive-and-don't-track*. This is the expected
naive-hierarchy failure — **it motivates the off-policy TD3 HL (M4)**: a replay buffer =
orders-of-magnitude more HL updates, and a deterministic actor + bounded exploration
noise instead of entropy-driven std (no collapse/blowup failure modes).

**Config state after the session (clean baseline restored):** 4× tracking reward
**reverted** (A1 env back to A0-matched — RQ2 confound removed by decision); HL std cap
**off**; HL `entropy_coef=0.01`; goal space 7-dim default; `hl_algorithm` default still
`oracle` (pass `--agent.hl-algorithm ppo` to run ppo).

**Untried ppo levers if M3-ppo is revisited (low priority — TD3 is the designed fix):**
(a) **HL horizon**: `num_steps_per_env` 24→48+ → 6+ HL transitions/iter (directly targets
the credit-assignment thinness; also lengthens the LL rollout); (b) 4× tracking reward
**+ std cap** together (run D had the incentive, run C had the stability — never combined);
(c) the std cap + lower entropy. None expected to beat what off-policy TD3 gives for free.

- Optionally export `policy_hl.onnx` too (deferred to the deployment milestone).
- After ppo: Milestones 4 (`td3 relabel=none`) and 5 (`td3 relabel=hiro`) per §5b/§6.

### Milestone 4 — `HighLevelTd3` (IMPLEMENTED 2026-06-11; first run launched)
`rl/hrl/td3.py` (`HighLevelTd3`) + `rl/hrl/storage.py` (`HLReplayBuffer`/`HLBatch`),
config `HlTd3Cfg` (`rl_cfg.py`, field `hl_td3`), wired into `_make_high_level` for
`hl_algorithm=="td3"` (`relabeling` must be `"none"` until M5). Slots into the existing
`HighLevel` interface — zero co-train-loop changes.

**Design (deviations from §5b's sketch are noted):**
- **Actor** = raw `rsl_rl.modules.MLP` `[92 → 256,256 → goal_dim]` + `tanh` → `g ∈
  [-1,1]^goal_dim`; window target is the same HIRO delta map as ppo: `V* = state +
  GoalSpace.scale(env) · g` (so the tanh bound ≈ the command half-range, curriculum-
  tracking). NOT an `MLPModel` (no distribution; TD3 is deterministic).
- **Twin critics** `Q1,Q2`: MLP `[(92+goal_dim) → 256,256 → 1]` over `[norm(s), g]`.
- **One shared `EmpiricalNormalization(92)`** over the HL state (`policy ++ command`),
  updated online in `act`, applied *outside* the nets to online and target forwards
  alike — target deepcopies therefore never hold stale normalizer stats.
- **TD3 tricks:** soft targets (`tau=0.005`), target-policy smoothing (clipped Gaussian
  `0.2/0.5` on the next goal), clipped double-Q, delayed actor+target update
  (`policy_freq=2`), Gaussian exploration noise `0.2` on the bounded `g` at act time.
- **Replay buffer:** flat GPU ring (`buffer_capacity=500k` ≈ 40 iters of history at
  4096 envs × 3 windows = 12 288/iter). `n_grad_steps=8` × `batch_size=512` per iter,
  `learning_starts=10k` (≈ 1 iter), `lr=1e-3`.
- **Truncation handling mirrors the ppo HL:** the buffer `done` excludes `time_outs`
  (incl. A1's `fell_over`) so the TD target bootstraps falls (`γ_hi·Q`) — same
  suicide-attractor fix, expressed as a bootstrap mask instead of a reward patch.
- **Logging:** `hl/q_loss`, `hl/actor_loss`, `hl/q_value`, `hl/act_abs_mean` (raw |g|,
  pre-scale — compare against the ppo runs' `hl/goal_abs_mean` *after* multiplying by
  scale), `hl/buffer_size`.
- Checkpoint `hl` key holds actor/critics/targets/normalizer/optimizers (resume-safe).
  Replay buffer NOT saved (a resume refills it in ~40 iters).

**Launch:** `--agent.hl-algorithm td3` on the standard A1 command. TD3 knobs override
as `--agent.hl-td3.<field>` (e.g. `--agent.hl-td3.expl-noise-std 0.3`).

### M4 run 1 (`a1_td3_5k`, 2026-06-11) — FAILED: actor saturation + coverage collapse
5k iters, 4096 envs, A0 warm-start, default `HlTd3Cfg`. W&B `atouumzj`, log dir
`2026-06-11_11-58-13_a1_td3_5k`.

**Trajectory:** track-and-fall to ~it500 → best point ~it1300 (ep_len 1000,
trk_lin 0.31, err_xy 1.4) → monotonic decay to it5000 (reward −15→−330, err_xy 3.1,
LL std 0.33→1.02, joint_pos_limits −0.06→−5.7, action_rate −1.6→−6.7, HL Q −1.5→−27).
Falls ~0 throughout after it750: the failure basin is *survive-but-don't-track*,
penalty-driven decay — the mirror image of M3-ppo's "stay put" failure.

**Root cause (confirmed by per-dim g inspection at it4999):**
1. The HL actor **saturated immediately** (|g|≈0.88 by it250, lr 1e-3 vs a fresh
   critic; deterministic policy gradient races to the tanh bounds).
2. Saturation is **self-locking via coverage collapse**: exploration noise σ=0.2
   around a ±0.88 mean (clamped) puts every buffered goal in ≈[0.68,1.0] — the
   critic has NO interior-goal data, so no gradient ever points back inward.
   Standard TD3 prevents this with uniform-random warmup actions
   (`start_timesteps` in Fujimoto's reference impl) — which was missing.
3. The decay phase = the LL escalating against unreachable saturated targets
   (intrinsic reward flat ~−4 all run): action aggression grows, penalties explode,
   HL task reward sinks, Q tracks it down uniformly (saturated-only data → the
   argmax never moves).
   At it4999 the deterministic goal is a near-constant vector with velocity dims
   (−0.67, −0.84, −0.19) → the *fast-backwards-circling* seen in replays.

**Fixes implemented (F1/F2, 2026-06-12):** F1 = uniform random goals `g~U(-1,1)`
while `len(buffer) < warmup_transitions` (new `HlTd3Cfg` field, default 100k ≈ 8
iters) — TD3's `start_timesteps`; F2 = `learning_rate` split into
`actor_learning_rate=3e-4` / `critic_learning_rate=1e-3`. Smoke verified the
random→actor boundary (|g| ≈ 0.50 uniform → 0.22–0.30 after handover).

### M4 run 2 (`a1_td3_warmup_5k`, 2026-06-12) — F1+F2 NOT sufficient; worse collapse
5001 iters, 4096 envs, A0 warm-start, defaults otherwise. Log dir
`2026-06-12_12-34-54_a1_td3_warmup_5k`, W&B `lj623x4l`, final ckpt `model_5000.pt`.

- **Saturation returned ~it250** (|g|≈0.86 with expl noise; deterministic mean less
  saturated: |g| 0.60 at it5000 vs run 1's 0.72). The 100k random-coverage
  transitions were **evicted from the 500k ring by ~it50 of post-warmup data** —
  F1's coverage is transient; F2 delayed nothing measurable.
- Same survive-but-don't-track basin (ep_len ~1000 from it1000, err_xy flat 2.0–2.3)
  BUT the **penalty decay spiral was WORSE than run 1**: reward 0.3→**−552** (run 1:
  −330), action_rate −15.3, joint_pos_limits −7.5, **LL std 0.33→1.73** (run 1:
  1.02), intrinsic flat-to-worse (−4→−6.8). err_xy *looked* flat while reward
  collapsed — the decay was pure action-violence, not tracking change.
- Deterministic closed-loop eval @5000 (64 envs/600 steps): err (0.52, 0.56, 0.54),
  0 falls, g signed means moderate/varied — better than run 1 (0.56, 0.86, 1.13),
  still ~2× worse than M3-ppo's mean (0.25, 0.20, 0.46) and ~10× worse than oracle.
- **Revised root cause (two runs, same signature):** not just buffer coverage — a
  **co-training spiral the HL cannot gradient out of**: extreme/noisy goals degrade
  the LL (std explosion, penalty growth); the HL reward (= summed task reward) is
  then dominated by LL-violence penalties that are nearly **goal-independent**, so
  Q(s,g) carries almost no tracking gradient; the actor drifts/saturates on noise
  while everything sinks. Echoes the M3 finding (tracking term swamped by penalties)
  at the HL timescale.
- F3 candidate levers (decision pending): freeze-LL phase (train HL against the
  frozen warm-started LL first, then co-train); persistent random-goal fraction
  (ε-style, coverage never evicted); goal-scale cap (reachability); exploration-noise
  decay; HL-reward composition (tracking-weighted — architecture-internal, logged
  A0-comparable reward unchanged; design decision).

### M4 F3 — Freeze-LL TD3 (IMPLEMENTED 2026-06-15; run launched)
Decision: freeze a **converged velocity-only oracle LL** and train the TD3 HL against
it. Stationary LL ⇒ the co-training spiral is structurally impossible (the LL cannot
degrade), isolating the open question: *can the TD3 HL learn command→goal at all?*
**Velocity-only chosen** because the frozen LL is memoryless (conditioned on the
instantaneous delta `V*−s`): velocity deltas a HIRO-delta HL emits are in-distribution
(⊆ what the oracle LL saw; g=0 ⇒ maintain), but the oracle only ever trained
orientation/height toward *nominal*, so a learned HL emitting non-zero
orientation/height goals would be off-distribution (this is what bit run 2). Restricting
to velocity removes those dims entirely.

**Implementation:** `HrlRunnerCfg.freeze_ll_path` (mutually exclusive with
`warm_start_path`); `_load_frozen_ll` does a strict full load of actor+critic from an A1
checkpoint (shape mismatch = wrong goal space, errors loudly). In `learn()` the LL acts
via its **deterministic mean** (`get_policy()(obs)`; its trained action std ~1.3 is far
too noisy to sample), no rollout storage, no `process_env_step`/`compute_returns`/
`update` — only the HL learns. F1/F2 stay active. One throwaway stochastic forward at
learn() start populates the (constant) action distribution so `action_std` logging
works.

**Goal-space coupling fix (Option A, TEMPORARY):** the env's `goal` obs dim is baked at
task-registration from the *default* `goal_components`; `--agent.goal-components` only
changes the runner, not the env (the dict param isn't a tyro flag — same desync class as
the play.py bug). So the registered default in `unitree_h1_2_hrl_runner_cfg()` was
flipped to `("velocity",)` (3-dim) for this experiment. **REVERT to 7-dim when done**
(the line is marked in `rl_cfg.py`). The 7-dim default is the documented decision.

**Smoke-validated (256 envs):** goal group 3-dim; frozen LL loads; **LL actor
byte-identical after 5 train iters** (freeze works); HL buffer fills, losses finite;
deterministic-mean LL competent under oracle goals via F0 eval (err (0.056,0.044,0.085),
0 falls — the low in-loop ep_len early on is just the `init_at_random_ep_len` transient).

**Run `a1_td3_frozenLL_5k` (W&B `gnsp3xjv`) — clean NEGATIVE, and it pinpointed the real
bug.** Aborted ~it2100. With the LL frozen (ll_std constant 1.30, spiral impossible) the
HL **still failed to track**: err_xy plateaued ~2.5 (worse than the co-train runs!),
|g| settled ~0.43, reward flat ~−61. So the spiral was never the core problem.

**Hypothesis at the time (LATER FALSIFIED by F4 — see below):** the HL is blind to its
own translational velocity. The training-metric err_yaw (~1.0) tracked ~2.4× better than
err_xy (~2.5), and the HL actor obs (`policy ++ command` = the A0 actor) omits
`base_lin_vel` (vx/vy critic-only) — so the HL sees yaw-rate but not vx/vy. Seemed to
explain the asymmetry. **But the F4 A/B det-eval showed velocity obs changes nothing** —
this hypothesis was wrong; the training-metric asymmetry was misleading (det eval shows
the HL tracks yaw best and vx worst *regardless* of velocity observability).

### M4 F4 — velocity observability — TRIED then REVERTED (2026-06-15): a NO-OP
**Result first:** the A/B deterministic eval (same frozen velocity-only LL) showed blind
HL (0.85, 0.39, 0.28) ≈ velocity-obs HL (0.85, 0.38, 0.25) — **velocity observability
changed nothing.** The velocity-blindness hypothesis is FALSIFIED; F4 was reverted (it
added a privileged input — `base_lin_vel`, needing an on-robot estimate at deploy — for
zero tracking benefit). **Process lesson:** I built F4 on the training-metric yaw/xy
asymmetry without first running the deterministic A/B, which would have killed the
hypothesis in a minute. Always det-eval before committing to a fix.

**Bigger correction:** det evals (which strip the goal exploration noise that inflates
training err to ~2.4) show every learned HL achieves *partial* tracking, never oracle:
oracle (0.06,0.04,0.09); **M3 PPO track4x (0.25,0.20,0.46) — best vx/vy**; TD3 run2
(0.52,0.56,0.54); TD3 frozen (0.85,0.39,0.28) — best yaw, but **freezing made vx WORSE**.
No lever (warmup/LR/freeze/velocity-obs/5×) closed the gap to oracle. Judge HLs by det
eval, not training err.

**Config state (2026-06-15):** goal_components default = velocity-only (Option A,
TEMPORARY — revert to 7-dim); A1 env carries **5× track_lin/track_ang** (reward-comp
lever, RQ2 confound vs A0); HL PPO **std cap (1e-3,1.0)** re-added (fixes std blowup); F4
reverted. The 5× frozen-LL run was also unsuccessful (cluster). Next: **M5 relabeling**.

<details><summary>Original F4 implementation notes (now reverted)</summary>

Add `base_lin_vel` to the **HL actor** obs (keep the relative HIRO goal). New env obs
group `base_lin_vel` (A1 only, `_restructure_obs_groups`), corruption matching the actor
(it's a noisy IMU/estimate term); HL actor obs = `policy ++ command ++ base_lin_vel`
(95-dim), critic Q-in 98. `HighLevelTd3._state_vec`/`_state_dim` and `HighLevelPpo`
actor obs_groups updated; LL and A0 unchanged. Foot terms (also critic-only) deliberately
NOT added — irrelevant to forming a velocity goal. **Deploy flag:** `base_lin_vel` is an
`imu_lin_vel` sensor (noise ±0.5), so the deployed HL needs an on-robot base-velocity
estimate (yaw-rate is a true gyro reading). Smoke-validated: group present (3-dim), HL
actor in-dim 95, frozen LL still loads/frozen, buffer fills.
**Run: `a1_td3_frozenLL_velobs_5k`** (5001 iters, 4096 envs, frozen velocity-only LL),
W&B pending. Sharp prediction: **err_xy drops toward the err_yaw level** (the observed
axis already works). If it does, velocity-blindness is confirmed and the fix transfers to
the co-training TD3/PPO HLs (add base_lin_vel there too). If err_xy stays high even with
velocity observed, the HL-learning problem is deeper than observability.
</details>

### Play/replay was broken for A1 until 2026-06-11 (F0 fix — IMPLEMENTED)
`play.py` uses `get_inference_policy()`, which returned the bare LL actor; nothing
fired the HL or wrote `env.hrl_goal` in play, so **every A1 replay (all milestones)
showed the LL with the goal frozen at zero** — the command physically can't reach
the LL in that path. All pre-F0 qualitative replay impressions are void; W&B
training metrics (HL in the loop) were always the ground truth.
Fix: `HighLevel.act_inference` (deterministic, side-effect-free; PPO/TD3 override)
+ `HierarchicalRunner.get_inference_policy` override that mirrors the training
loop's goal wiring (fire HL every c-th call, write remaining delta).
**Validated headless (64 envs, 600 steps):** oracle ckpt `model_9999` tracks
(|err| ≈ 0.05/axis, 0 falls) through the inference path; td3 `model_4999` shows the
true broken behavior (err (0.56,0.86,1.13), saturated constant g). Eval script
pattern: see `get_inference_policy` docstring (window clock not reset on mid-window
env resets — same approximation as training).

**play.py structure restore (2026-06-12).** play.py rebuilt the runner from the
*current task defaults*, so (a) checkpoints with a non-default goal space crashed the
load (dim mismatch — hit by the velocity-only run), and (b) ppo/td3 checkpoints
silently replayed with an **oracle** HL (default `hl_algorithm`; the trained `hl`
state was ignored by the oracle's no-op `load_state_dict`) — i.e. those replays
showed *trained-LL + command-as-goal*, not the closed loop. Fix: play.py now restores
the structure keys (`c, goal_components, goal_weights, hl_algorithm, hl_ppo, hl_td3,
relabeling, gamma_hi`) from the run's own `params/agent.yaml` (yaml.full_load — dumps
carry python/tuple tags), deep-merging dict cfgs over defaults because the dump is
written *after* runner construction and MLPModel pops `distribution_cfg["class_name"]`
in place (so saved yamls are missing it); the env's goal obs dim is re-derived from
the restored components. Prints `[INFO]: Restored run structure ...` so a replay
always says which HL it runs. Verified on 4 checkpoints (velocity-only, td3, oracle,
ppo-track4x).

**Deterministic closed-loop evals (post-fix, 64 envs, 600 steps, play env) — REVISES
the M3 verdict:** `a1_ppo_track4x` `model_1999` with its real ppo HL tracks
|err| (0.25, 0.20, 0.46), 0 falls — vs oracle (0.05, 0.04, 0.06) and td3 run-1
(0.56, 0.86, 1.13). The ppo HL **mean** did learn a partial command→goal map; the
bleak training-time numbers (err_xy ~1.05+) were measured *with sampling noise* and
during co-training oscillation. "Naive ppo fails" still holds for the training
process (unstable, never converged to survive+track simultaneously), but the M3
artifact is better than the training metrics implied — worth re-evaling other M3
ckpts deterministically before final writeup comparisons.

**Launch:** add `--agent.hl-algorithm ppo` to the standard A1 launch (warm-start path
included). e.g. `... Unitree-H1_2-Flat-A1 --env.scene.num-envs 4096
--agent.hl-algorithm ppo --agent.warm-start-path <A0 model.pt> --agent.run-name <name>`.

**Run launch convention:** activate the env and run python directly (NOT `conda run`,
which buffers and breaks live wandb): `conda activate unitree_mjlab_h1_2_rl && python
scripts/train.py Unitree-H1_2-Flat-A1 --env.scene.num-envs 4096 --agent.max-iterations N
--agent.warm-start-path <A0 model.pt> --agent.run-name <name>`.

### M5 — HIRO relabeling + c-sweep (IMPLEMENTED & RUN 2026-06-16)

**Relabeling (`relabel=hiro`) IMPLEMENTED** inline in `HighLevelTd3` (no separate
`relabeling.py`): `HLReplayBuffer(relabel=True)` stores the per-window LL trace
(policy/goal-state/action seqs + scale + s_{t+c}); `_relabel(batch)` builds k=10 candidate
goals (stored g, empirical `(s_{t+c}-s_t)/scale`, +8 Gaussian), scores each by the current
LL's log-likelihood of the stored action trace (one flat `ll_actor` forward), and
substitutes `batch.actions` before the critic update. Logs `hl/relabel_frac`.

**Results (deterministic benchmark, 64 envs × 600 steps × 2 eval seeds; see "Benchmark"
below). Run `a1_td3_relabel_3k` (c=8) + c-sweep `a1_td3_relabel_c{4,12}_3k`:**

| c | HL steps/ep | err_vx | err_vy | err_yaw | action_rate | height_dev | falls |
|---|---|---|---|---|---|---|---|
| 4 | 250 | 0.47 | 0.30 | 0.41 | 1.52 | 0.072 | 0 |
| 8 | 125 | 0.42 | 0.35 | 0.64 | 1.41 | 0.064 | 0 |
| 12| 83  | 0.39 | 0.26 | 0.58 | 1.23 | 0.048 | 0 |
| A0 ref | — | 0.09 | 0.11 | 0.10 | 0.66 | 0.027 | 0 |

- **Relabeling helps translational tracking:** the c=8 relabel run's vx/vy (0.42/0.35) is
  the best of any TD3 variant (vs run2 0.52/0.56, frozen 0.85/0.39; nears M3-PPO 0.25/0.20).
- **"Too few HL steps" hypothesis NOT supported.** Finer cadence (c=4 → 2× the HL
  transitions of c=8) did not improve tracking; vx/vy stay ~0.26–0.47 (still ~4–5× A0)
  with no monotonic gain. yaw is non-monotonic (c8=0.64 an outlier-high) → the early
  "c=4 fixes yaw" read was training-seed noise. The only CLEAN monotonic effect: coarser
  c → smoother (action_rate 1.52→1.23) + better height (0.072→0.048), at no tracking cost —
  the opposite of "needs more HL steps."
- **Conclusion: the translational-tracking wall is STRUCTURAL** (goal-space scale/
  reachability, LL goal-achievability, or the command→goal map) — both data quantity AND
  temporal resolution are now ruled out, as is the co-training spiral (frozen-LL) and
  velocity observability (F4). **CAVEAT: 1 training seed per c**; per the reproducibility
  rule, c-to-c tracking gaps are likely within training-seed noise (eval std only
  ~0.01–0.05) — the smoothness/height trends are the credible signal.
- **Next:** decompose the tracking error into HL-goal-error vs LL-reach-error — see
  `doc/A1_goal_achievability_probe.md`.

**`gamma_hi` footgun FIXED (2026-06-16):** was hardcoded `0.99**8`, decoupled from `c` —
any `c != 8` run silently got the wrong HL discount. Now derived `0.99**c` **unconditionally**
in `HrlRunnerCfg.__post_init__` (a `if None` guard fails because tyro carries the
factory-derived c=8 value forward when only `c` is overridden, so it must re-derive every
construct). Verified via the real tyro path: `--agent.c 4` → `gamma_hi 0.96`.

**Benchmark tool (`scripts/play.py --eval-steps N --eval-seeds K`):** the canonical in-sim
policy comparator — deterministic, multi-metric scorecard (err_vx/vy/yaw, fall_rate,
action_rate, orient_dev, height_dev) + `[BENCH] {json}` line, multi-seed mean±std. Reuses
play.py's checkpoint/structure-restore. Outcome metrics (architecture-agnostic, strip
exploration noise) → replaces ad-hoc det-eval; A0 baseline = 0.09/0.11/0.10. NOTE: `ep_len`
is uninformative in play mode (`episode_length_s=1e9`, no resets) — use `fall_rate` for
survival. Sim-to-sim (ONNX → deploy MuJoCo) is a SEPARATE, later transfer check, not a
substitute for this in-sim benchmark.

---

## 1. Design goals & constraints

| Goal | How it's met |
|---|---|
| Isolate the *hierarchy* effect for RQ2 (A0 vs A1) | Low level uses the **same PPO** as A0; same reward/obs/DR env. Only the structure changes. |
| Genuine HIRO goal relabeling | High level is **off-policy TD3 with a replay buffer**; relabeling correction operates on replayed HL transitions where it is mathematically valid. |
| Don't fork mjlab/rsl_rl | All new code lives under `src/tasks/velocity/`. We reuse `rsl_rl.PPO` for the LL and write a small self-contained TD3 for the HL (≈ acts on a 3-dim goal only → tiny nets). |
| Deployment-faithful | Both HL and LL consume **only real-robot-available observations** (the `actor` obs group; no privileged `base_lin_vel`). Intrinsic reward uses privileged sim velocity, but reward is never needed at deployment. |
| Hierarchy + relabeling as ablation knobs | `hl_algorithm {ppo,td3}` and (for td3) `relabeling {none,hiro}` → see the run matrix below. |

### The switch is the HL learner, not just a relabel flag
Relabeling = off-policy correction = **needs a replay buffer**. PPO is on-policy:
it collects a rollout, runs a few epochs, then discards the data — there is
nothing stored to relabel and replay. So the choice is structural:

- **`hl_algorithm="ppo"`** → high level is on-policy PPO → **2-level PPO**
  (no relabeling possible). Reuses `rsl_rl.PPO` for *both* levels → minimal new code.
  This is the "naive hierarchy" baseline.
- **`hl_algorithm="td3"`** → high level is **off-policy TD3 + replay buffer**, with
  a `relabeling {none,hiro}` sub-toggle. This is the new code (TD3 + buffer +
  relabeling). HIRO's correction lives here, at the sparse HL (one transition per
  `c` steps → a small buffer is cheap).

Why relabeling can't be on-policy (the math): PPO's surrogate uses the ratio
`π_new(a|s,g)/π_old(a|s,g)` with `π_old` the behaviour policy at collection time.
Relabeling `g → g̃` changes the policy **input**, so the stored action `a` was
never sampled from `π_old(·|s,g̃)`; the relabeled tuple is off-policy w.r.t. the
new goal and needs importance weights or a replay buffer. **The low level stays
on-policy PPO in all configs and is never relabeled** (relabeling is HL-only).

### Run matrix (all from one codebase)
| Run | HL learner | Relabel | Answers |
|---|---|---|---|
| **A0** (`Unitree-H1_2-Flat`) | — | — | baseline |
| **A1 `hl=ppo`** | PPO | n/a | does *structural hierarchy* help? (clean: both PPO) |
| **A1 `hl=td3 relabel=none`** | TD3 | ✗ | isolates *off-policy HL* (and is the clean control for relabeling) |
| **A1 `hl=td3 relabel=hiro`** | TD3 | ✓ | full HIRO; vs the row above isolates the *relabeling correction* |

Caveat to keep in mind for the writeup: `hl=ppo` → `hl=td3 relabel=hiro` changes
*two* things (PPO→TD3 **and** relabeling). The `relabel=none` TD3 row is the clean
control that attributes any further gain to relabeling alone.

---

## 2. Roles, timescales, and the goal space

```
                    ┌─────────────────────────────────────────────┐
external twist  c → │  HIGH LEVEL  μ_hi(s, c) → g    (TD3, off-pol)│  fires every c steps
(vx*,vy*,ψ̇*)        │     g = (Δvx, Δvy, Δψ̇) ∈ ℝ³  (velocity delta)│
                    └───────────────────────┬─────────────────────┘
                                            │ goal g (held constant over the window)
                    ┌───────────────────────▼─────────────────────┐
state s         →   │  LOW LEVEL   μ_lo(s\c, g) → a  (PPO, on-pol) │  fires every step
(proprio, no cmd)   │     a = 27 joint position targets            │
                    └───────────────────────┬─────────────────────┘
                                            │ a → PD → MuJoCo
                                            ▼
                       intrinsic reward r_lo = −‖(v_t + g) − v_{i+1}‖   (sim velocity)
```

- **Control rate:** decimation 4 × dt 0.005 s → 50 Hz control step.
- **`c` (HL period):** **8** (= 0.16 s) [confirmed]. Must divide `num_steps_per_env`.
- **Goal `g`:** velocity **delta** per the thesis spec, `(Δvx, Δvy, Δψ̇)`. The goal
  bound is **derived from the active command-curriculum ranges** [confirmed] — the
  HL output is `tanh × goal_scale` where `goal_scale` is read from the twist
  command ranges at construction (and re-read if the curriculum stage changes), so
  editing the curriculum automatically rescales the goal. The window target is
  `V* = v_t + g` computed at HL fire-time from the privileged sim velocity `v_t`.

### Critical design decision — the command is removed from the LL observation
The LL obs group is the A0 `actor` group **minus the `command` term, plus the
`goal` term**. If the LL could see the external command, it would bypass the
hierarchy and track the command directly, collapsing A1 into A0. Removing the
command forces the goal `g` to be the *only* channel through which task intent
reaches the LL — which is exactly what makes the A0↔A1 comparison meaningful.

| Network | Obs groups | Dim | Notes |
|---|---|---|---|
| HL actor `μ_hi` | `actor` (incl. command) | 92 | deployable obs only |
| HL critic (twin Q) | `actor` ⊕ `goal` | 92 + 3 | TD3 Q(s,g) |
| LL actor `μ_lo` | `actor∖command` ⊕ `goal` | 89 + 3 = 92 | deployable obs only |
| LL critic | `critic∖command` ⊕ `goal` | (critic−3) + 3 | privileged ok (not deployed) |

(92 = ang_vel 3 + proj_grav 3 + command 3 + phase 2 + joint_pos 27 + joint_vel 27 + last_action 27.)

---

## 3. Reward decomposition

| Level | Reward | Source |
|---|---|---|
| **HL** | the **A0 task reward**, summed over the `c`-step window: `R = Σ_{i=t}^{t+c-1} r_task,i` | existing reward manager, unchanged |
| **LL** | intrinsic `r_lo,i = −‖V* − v_{i+1}‖₂` with `V* = v_t + g` (velocity subspace = (vx, vy, ψ̇)) | computed in the LL env wrapper from sim base velocity |

- HL discount `γ_hi = γ^c` (≈ 0.99⁸) so **HL and LL horizons align** [confirmed].
  Applies to both the `hl=ppo` (GAE) and `hl=td3` (TD target) paths.
- LL keeps `γ=0.99, λ=0.95` (A0 values).
- The LL receives **no** task reward (pure HIRO). Optional `ll_task_reward_coef`
  blend is left as a config knob (default 0) but off for the clean ablation.

---

## 4. Co-training loop (one learning iteration)

```
for it in range(max_iterations):
    # ---- Rollout: num_steps_per_env LL steps (must be a multiple of c) ----
    for k in range(num_steps_per_env):
        if k % c == 0:                         # HL fires
            v_t   = env.base_velocity()        # privileged sim velocity (vx,vy,ψ̇)
            g     = hl.act(actor_obs)          # TD3 actor (+ exploration noise)
            V_tar = v_t + g                    # window target, frozen for c steps
            hl_open = HLTransition(s_t=actor_obs, g=g, R=0, seq=[])  # start accumulating
        ll_obs = build_ll_obs(obs, g)          # drop command, append goal
        a      = ppo.act(ll_obs)               # rsl_rl PPO act() (stores LL transition)
        obs, _, dones, extras = env.step(a)
        v_next = env.base_velocity()
        r_lo   = -(V_tar - v_next).norm(dim=-1)
        ppo.process_env_step(ll_obs_next, r_lo, dones, extras)   # LL transition complete
        hl_open.R   += task_reward             # accumulate task reward into HL return
        hl_open.seq.append((ll_obs, a))        # store (s_i,a_i) for relabeling
        if (k+1) % c == 0 or dones.any():      # HL window closes
            hl_open.s_next = actor_obs_next; hl_open.done = dones
            hl_buffer.add(hl_open)             # push to TD3 replay buffer

    # ---- Updates ----
    ppo.compute_returns(last_ll_obs); ll_losses = ppo.update()       # on-policy LL update
    if hl_algorithm == "ppo":                                        # on-policy HL update
        hl.compute_returns(last_actor_obs)                           # GAE over the few HL steps
        hl_losses = hl.update()                                      # standard rsl_rl PPO
    else:                                                            # off-policy HL update
        hl_losses = hl.update(hl_buffer, ll_policy=ppo.actor,
                              relabel=relabel_strategy, n_grad_steps=…)
    log(it, ll_losses, hl_losses, …)
    if it % save_interval == 0: save(...)
```

The HL window-close step routes the transition to the right sink:
`hl=ppo` → an HL `RolloutStorage` (on-policy, consumed & cleared each iter);
`hl=td3` → the replay buffer (off-policy, persists across iters).

Notes:
- `num_steps_per_env=24`, `c=8` → 3 closed HL windows / env / iter × 4096 envs ≈
  **12 k HL transitions/iter**. For `td3` this fills the buffer; for `ppo` it's the
  on-policy batch (short GAE horizon of 3 HL steps — dense windowed reward makes
  this workable; bump `num_steps_per_env` if HL credit assignment looks weak).
- HL TD3 does `n_grad_steps` (default 8) sampled minibatch updates per iteration.
- Episode resets mid-window close the HL window early and mask the bootstrap.

---

## 5a. High-level PPO path (`hl_algorithm="ppo"`)

`src/tasks/velocity/rl/hrl/hl_ppo.py` (thin wiring around `rsl_rl.PPO`)

- A second standard `rsl_rl.PPO` whose **actor** is an `MLPModel` `[92 → 256 → 256 → 3]`
  with a `GaussianDistribution` (scalar std) squashed/scaled to `goal_scale`, and a
  **critic** `MLPModel` `[92 → 256 → 256 → 1]`. Same machinery as the A0 agent, just
  a 3-dim action head.
- Fed HL transitions at the HL timescale via its own `RolloutStorage`
  (`num_transitions = num_steps_per_env / c` per env). `act → process_env_step →
  compute_returns → update` mirror the LL exactly.
- **No relabeling** (on-policy). This is the naive-hierarchy baseline.

## 5b. High-level TD3 path (`hl_algorithm="td3"`, self-contained)

`src/tasks/velocity/rl/hrl/td3.py`

- **Actor** `μ_hi`: MLP `[92 → 256 → 256 → 3]`, `tanh` × goal_scale. EmpiricalNormalization on input (reuse `rsl_rl.modules.EmpiricalNormalization` for export parity).
- **Twin critics** `Q1,Q2`: MLP `[(92+3) → 256 → 256 → 1]`.
- **TD3 tricks:** target nets (soft τ=0.005), target-policy smoothing (clipped
  Gaussian noise on next-goal), clipped double-Q target, **delayed actor update**
  (policy_freq=2), exploration noise (Gaussian, decaying).
- **Replay buffer:** ring buffer of `HLTransition` storing
  `(s_t, g_t, R, s_{t+c}, done)` **plus** the per-window sequence `[(s_i, a_i)]`
  (needed only for relabeling; can be dropped from buffer when `relabeling=none`
  to save memory).

### TD3 target (with relabeling applied at sample time)
```
g̃        = relabel_strategy.relabel(batch, ll_policy)     # HL-only correction
a'       = clip(μ_hi_target(s') + clip(ε,−c,c), g_low, g_high)
y        = R + γ_hi · (1−done) · min(Q1_target(s',a'), Q2_target(s',a'))
L_Q      = MSE(Q1(s,g̃), y) + MSE(Q2(s,g̃), y)
L_actor  = −Q1(s, μ_hi(s)).mean()           # every policy_freq steps
```

---

## 6. Pluggable relabeling

`src/tasks/velocity/rl/hrl/relabeling.py`

```python
class RelabelStrategy(ABC):
    @abstractmethod
    def relabel(self, batch: HLBatch, ll_policy: MLPModel) -> torch.Tensor:
        """Return goals to use for this HL minibatch (shape [B,3])."""

class NoRelabel(RelabelStrategy):
    def relabel(self, batch, ll_policy): return batch.goals

class HiroOffPolicyCorrection(RelabelStrategy):
    """HIRO §3.2: g̃ = argmax_g Σ_i log π_lo(a_i | s_i, g)."""
    n_candidates = 10          # original + achieved-delta + 8 Gaussian
    sigma = 0.5                # std of candidate sampling (× goal range)
    def relabel(self, batch, ll_policy):
        # candidates: [B, K, 3] = {g_orig, (v_{t+c}-v_t), 8×N(achieved, σ)}
        # for each candidate, set LL goal, eval stored a_i under LL Gaussian,
        # sum log-probs over the window seq, pick argmax → [B,3]
        ...
```

- Candidate set per HIRO: original `g`, the **achieved** velocity delta
  `v_{t+c}−v_t`, and 8 Gaussian samples centred on the achieved delta, clipped to
  goal bounds.
- Log-prob uses the **current** LL Gaussian (`ll_policy.distribution`), evaluated
  on the stored `(s_i, a_i)` with the candidate substituted into the goal slot.
- Log a **relabel-acceptance rate** (fraction where `g̃ ≠ g_orig`) — a useful
  diagnostic and an ablation talking point.

---

## 7. File-by-file plan

**ACTUAL current structure (✅ exists / ⬜ to build):**
```
src/tasks/velocity/
├── velocity_env_cfg.py      # ✅ base env; fell_over true-terminal (A1 overrides to time_out)
├── config/h1_2_a1/
│   ├── __init__.py          # ✅ register Unitree-H1_2-Flat-A1; runner cfg = source of
│   │                        #    truth for goal_components, env derives goal obs dim
│   ├── env_cfgs.py          # ✅ reuse flat env; _restructure_obs_groups (split actor →
│   │                        #    policy/command/goal, drop actor); set fell_over time_out
│   └── rl_cfg.py            # ✅ HrlRunnerCfg: LL=PPO + c/goal_components/goal_weights/
│                            #    hl_algorithm/relabeling/gamma_hi/warm_start_path/...
└── rl/
    ├── runner.py            # ✅ VelocityOnPolicyRunner (+ _export_policy_onnx, reused by A1)
    └── hrl/
        ├── __init__.py      # ✅ exports HierarchicalRunner, HighLevel, OracleHighLevel
        ├── hrl_runner.py    # ✅ HierarchicalRunner(VelocityOnPolicyRunner): co-train loop,
        │                    #    warm-start (_partial_load gap-aware), save/load+hl, onnx
        ├── goal_space.py    # ✅ GoalComponent/GoalSpace + registry (REPLACES goal_env.py);
        │                    #    extract/reward/oracle_target/scale; goal_dim derived
        ├── high_level.py    # ✅ HighLevel ABC + OracleHighLevel + HighLevelPpo (M3)
        ├── td3.py           # ✅ HighLevelTd3 (M4)
        ├── storage.py       # ✅ HLReplayBuffer + HLBatch (M4)
        └── relabeling.py    # ⬜ RelabelStrategy + NoRelabel + HiroOffPolicyCorrection (M5)
```
NOTE: there is **no `GoalConditionedWrapper`** — the goal is a plain obs term
(`mdp.hrl_goal` reads `env.hrl_goal`, written by the runner each step); the goal space
logic lives in `goal_space.py`. The LL obs surgery is done by `_restructure_obs_groups`
in `env_cfgs.py`, not a wrapper.

### Goal injection mechanism (keeps the env config reusable)
The A1 `env_cfgs.py` adds a 3-dim `goal` observation **group** whose term reads
`env.unwrapped.hrl_goal` (a `[num_envs,3]` buffer). The `HierarchicalRunner`
writes that buffer when the HL fires. Construction-time obs sample includes a
zero goal so `MLPModel._get_obs_dim` resolves correctly. This way the LL/HL
models are **standard `MLPModel`s** (clean ONNX export) and the env stays a plain
mjlab env (the goal is "just another observation").

### Config sketch (`rl_cfg.py`)
```python
@dataclass
class HrlRunnerCfg(RslRlBaseRunnerCfg):
    class_name = "HierarchicalRunner"
    c: int = 8                              # HL period (divides num_steps_per_env)
    ll: RslRlOnPolicyRunnerCfg              # reuse A0 PPO cfg (actor/critic/algorithm)
    hl_algorithm: Literal["ppo","td3"] = "td3"
    hl_ppo: HlPpoCfg                        # used when hl_algorithm=="ppo"
    hl_td3: HlTd3Cfg                        # used when hl_algorithm=="td3" (τ, policy_freq, noise)
    relabeling: Literal["none","hiro"] = "hiro"   # td3 only; ignored for ppo
    gamma_hi: float = 0.99 ** 8             # = γ^c, horizon-matched [confirmed]
    goal_scale: tuple | None = None         # None → derive from twist command ranges [confirmed]
    ll_task_reward_coef: float = 0.0        # 0 = pure HIRO (LL intrinsic only)
```
`goal_scale=None` means the runner reads the active `commands["twist"].ranges`
(`lin_vel_x/y`, `ang_vel_z`) at build time to bound the goal, so editing the
command curriculum rescales the goal automatically.

---

## 8. Checkpointing, ONNX, deployment

- **Checkpoint** (`model_<it>.pt`): `{ll_actor, ll_critic, ll_normalizer, hl_actor,
  hl_critics, hl_targets, hl_normalizer, hl_optimizer, iter}`. Replay buffer
  optionally saved for resume.
- **ONNX export:** two files — `policy_hl.onnx` (92→3) and `policy_ll.onnx`
  (92→27). Reuse the proven `MLPModel.as_onnx()` path for both (normalizer copied
  via deepcopy — same rule as A0).
- **Deployment (later milestone):** the C++ `State_RLBase` runs `policy_ll.onnx`
  every step and `policy_hl.onnx` every `c` steps, feeding the HL output into the
  LL input slot. Build the LL input as `proprio∖command ⊕ goal`. This is the only
  C++ change; flagged for after sim works. (Two-model chaining; no base-linear-
  velocity estimate required because the LL consumes the raw goal, not `V*`.)

---

## 9. Logging (W&B project `biped_hrl`, experiment `h1_2_velocity_a1`)

LL (existing PPO metrics) + `intrinsic_reward/mean`, `task_reward/mean`,
`hl/q_loss`, `hl/actor_loss`, `hl/goal_mean_{vx,vy,wz}`, `hl/relabel_accept_rate`,
`hl/buffer_size`, `episode_length`, plus the A0 reward-term breakdown.

---

## 10. Milestones (recommended order)

1. ✅ **DONE — Scaffolding + plumbing.** A1 task package, goal obs group (declarative
   `goal_space.py`, not a wrapper), `HierarchicalRunner`. Env builds; shapes verified.
2. ✅ **DONE — Oracle-HL sanity test** (plus the fixes it forced: warm-start,
   `fell_over=time_out`, entropy 0.005). LL tracks goal=command and reaches/exceeds A0
   (out-tracks it). Stable to 10k iters. See §0 for results & deviations.
3. ⚠️ **IMPLEMENTED & TESTED — `hl=ppo` (2-level PPO) does NOT track.** `HighLevelPpo`
   wired in and co-trains (smoke + 4 diagnosis runs). The learned HL never learns
   command→goal (credit-assignment + co-training instability). **Decision pending:** one
   more ppo lever (HL horizon, or 4× track + std cap) vs proceed to M4. Full results in
   §0 "M3 experiment results" (authoritative over §5a).
4. ✅ **IMPLEMENTED — `hl=td3 relabel=none`.** `HighLevelTd3` + `HLReplayBuffer` built,
   smoke-tested, first 5k run launched (`a1_td3_5k`). Compare to `hl=ppo`. See §0 "M4".
5. **`hl=td3 relabel=hiro`.** Add the correction; log accept-rate; ablate vs row 4.
6. **Tuning + full omnidirectional run on cluster** (A100). Sweep `c ∈ {5,8,10,16}`,
   goal ranges, HL LR.
7. **Deployment** (two-model ONNX + C++ chaining) — after sim is stable.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Co-training instability (HL & LL are moving targets) | TD3 delayed/soft updates, small HL LR, relabeling; staged milestones (oracle → ppo → td3) catch instability early |
| Hierarchy collapse (LL ignores `g`) | command removed from LL obs; monitor intrinsic reward & goal-sensitivity; oracle-HL milestone catches plumbing bugs early |
| Too few HL transitions/iter | off-policy buffer accumulates across iters; `num_steps_per_env % c == 0` |
| Deployment needs base lin-vel | avoided: LL consumes raw `g`, not `V*`; sim-only privileged velocity used only for the (train-time) reward |
| `c` sensitivity | explicit ablation sweep |

---

## 12. Confirmed decisions  *(⚠ items 2–3 superseded by §0 — see notes)*

1. **`c` = 8** (0.16 s). ✅ still true.
2. **Goal range tied to the command curriculum** — still the intent; for the *learned*
   HL the scale is read from `commands["twist"].ranges`. ⚠ Not yet implemented:
   `GoalSpace.scale(env)` must be added for Milestone 3.
3. ⚠ **SUPERSEDED:** "train from scratch" was tried and FAILED. A1 is now **warm-started
   from a converged A0** (gap-aware partial copy) — this is load-bearing (§0 dev. 1).
4. **`γ_hi = γ^c`** — HL and LL horizons aligned. ✅
5. **HL switch is the learner:** `hl_algorithm ∈ {oracle, ppo, td3}`; `relabeling ∈
   {none, hiro}` applies only to `td3`. ✅ (`oracle` added for M2.)
6. **NEW — `fell_over=time_out` in the A1 env only** (bootstrap the fall to kill the
   suicide attractor from the negative goal-distance reward). A0 keeps a true terminal.
7. **NEW — entropy_coef=0.005, desired_kl=0.005, velocity goal weight 3**; goal space
   currently **velocity-only (3-dim)** (ablation), declarative via `goal_components`.

---

## 13. Potential A1 variant — HAC (Hindsight Actor-Critic) [not committed]

A possible *second* HRL algorithm for the A1 slot, comparing HIRO vs HAC under the
same env/reward/DR. **Not part of the current plan** — noted for scope discussion.

**What HAC is** (Levy et al., ICLR 2019; ref: `Hierarchical-Actor-Critic-HAC-PyTorch/HAC.py`,
`h-baselines/.../goal_conditioned`): off-policy, ≥2-level, with three mechanisms —
(a) **subgoal testing** (penalize HL by −H when the LL can't reach a tested subgoal),
(b) **hindsight action transition** (relabel the HL action with the *achieved* state →
HL trains as if the LL were optimal), (c) **hindsight goal transition** (HER on the
lowest level). Sparse goal-reached reward; DDPG/TD3 learners.

**Why it does NOT swap into the current hybrid:** HAC's strength needs hindsight at
**both** levels — its low level is fundamentally HER-based (off-policy, replay buffer,
sparse reward). Our hybrid deliberately keeps **LL = on-policy PPO with a dense
distance reward** (so the A0↔A1 comparison isolates the hierarchy). Doing HAC properly
means **off-policy at both levels** → migrate the LL off PPO (skrl/Isaac, since rsl_rl
is on-policy only) → which *breaks the "LL = A0 PPO" control* that A1 relies on.

**If pursued, treat as A1-HAC, a parallel variant, not a toggle:**
- Off-policy both levels (TD3/DDPG). Reference: port from `HAC-PyTorch` (PyTorch,
  single-env → batch/GPU) or `h-baselines` (TF, reference only).
- Engine options: a custom batched TD3 at both levels, **or** adopt `skrl` (GPU
  off-policy, but it owns the rollout loop → larger refactor).
- New comparison it buys: **A1-HIRO vs A1-HAC** = "off-policy correction" vs
  "hindsight + subgoal testing" for non-stationarity, on the same robot/task.
- Cost: a second off-policy LL path (not shared with A0/A2), more tuning (subgoal
  test ratio, tolerance/threshold, level horizon H), and it no longer reuses the
  proven PPO LL.

**Recommendation:** keep HIRO as the committed A1; revisit A1-HAC only if time remains
after A1-HIRO and A2, as an extra RQ-relevant data point.
```
