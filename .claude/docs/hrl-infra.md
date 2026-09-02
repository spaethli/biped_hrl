# HRL shared infrastructure

Stable reference for the machinery **every** HRL architecture (A1–A4) reuses. Per-arch
plans (`doc/hrl/A1_HIRO.md`, …) link here instead of re-deriving it. See also
`codebase-map.md` (file paths), `experiment-design.md` (A0–A4 matrix, RQ2 rule).

## Control rates & timescales
- decimation 4 × dt 0.005 s → **50 Hz** control step.
- `c` = HL period (HL fires every `c` LL steps); must divide `num_steps_per_env`.
  A1 default `c=8` (0.16 s). Treat `c` as a swept hyperparameter.
- `num_steps_per_env=24` → `24/c` closed HL windows / env / iter.
- **`gamma_hi` is DERIVED `0.99**c`** in `HrlRunnerCfg.__post_init__`, *unconditionally*
  (a `if None` guard fails — tyro carries the factory-derived `c=8` forward when only `c`
  is overridden, so it must re-derive every construct). Horizon-matched to the LL; never
  hardcode it. Verified: `--agent.c 4` → `gamma_hi 0.96`.

## HierarchicalRunner (co-train loop)
`rl/hrl/hrl_runner.py`. **Inherits `VelocityOnPolicyRunner`** → free ONNX export. Drives
the co-train loop, A0 warm-start, save/load (+`hl` state). One iteration:

```
rollout num_steps_per_env LL steps:
  every c-th step  → HL fires: g = hl.act(state); V* = GoalSpace.to_target(state,g)
  every step       → LL obs = proprio∖command ⊕ goal(=V*−s_i); a = ppo.act(); env.step
                     r_lo = −Σ_c w_c‖V*−s_{i+1}‖ ; hl.accumulate(task_rew)
  window close     → hl.end_window(extras)  → on-policy storage (ppo) | replay buffer (td3)
ppo.update()                       # on-policy LL
hl.update()                        # PPO GAE | TD3 n_grad_steps off-policy
```
Mid-window env resets: window clock is NOT reset (windows are always exactly `c` steps;
the done/reward use the window-end step). Same approximation in training and in
`get_inference_policy` (play). Acceptable; revisit only if HL credit assignment looks weak.

### HighLevel interface (ABC, `rl/hrl/high_level.py`)
`act(env,obs,state)→V*`, `act_inference` (deterministic, side-effect-free; play path),
`begin_window`, `accumulate(task_rew)`, `end_window(extras)`, `update`,
`state_dict/load_state_dict`, `train_mode/eval_mode`. `end_window` takes `extras` for the
`time_outs` truncation bootstrap (the ABC default ignores it). Implementations:
`OracleHighLevel`, `HighLevelPpo`, `HighLevelTd3`.

## Goal space (declarative — `rl/hrl/goal_space.py`)
`GoalComponent`/`GoalSpace` + registry. **`goal_components` (a list, set in the runner cfg)
is the single source of truth**; the env derives its `goal` obs dim from it via the task
`__init__.py`. **`goal_dim` is always derived, never hardcoded.**
- Components: `velocity` (vx,vy,ψ̇, 3-dim), `orientation` (3), `height` (1).
- **A1 default = 7-dim `(velocity,orientation,height)`** — a confirmed decision (orientation
  +height act as stabilizing regularizers: better tracking, esp. yaw, lower action std;
  not needed for survival). Velocity-only (3) is the documented clean negative result.
- The HIRO map lives in ONE place — `GoalSpace.center / to_target / to_g`:
  - **delta mode**  `V* = s_t + scale·g`  (`center` short-circuited / not evaluated)
  - **absolute mode** `V* = center + scale·g`, `center` = command-range midpoint
    (velocity) / nominal (orient, height); `scale` = half-range → `g∈[-1,1]` spans exactly
    the command range (static map). Selected by `hl_target_mode` (`delta`|`absolute`).
- `scale` is per-dim and **tracks the live twist command curriculum** (velocity = half the
  command range; orientation 1.0; height 0.2 m), re-read so editing the curriculum
  rescales the goal automatically.
- The goal is a **plain obs term** (`mdp.hrl_goal` reads `env.hrl_goal`, written by the
  runner each step) — NOT a wrapper. LL/HL nets stay standard `MLPModel`s (clean ONNX). LL
  obs surgery (`actor` → `policy`/`command`/`goal`, drop `command`) is done by
  `_restructure_obs_groups` in the A1 `env_cfgs.py`.

### Goal-channel estimator noise (`GoalStateNoise`, `rl/hrl/state_noise.py` — #8b sim2real DR)
Optional DR that trains the LL on a deploy-realistic base-velocity estimate (the real robot
reads `vx,vy` from `rt/sportmodestate`, not ground-truth). Corrupts only the estimator-supplied
goal columns: `velocity`→`vx,vy` (yaw-rate stays clean — gyro), `height` optional. Components:
**bias** (per-episode constant), **drift** (OU random walk), **lag** (1st-order low-pass).
**Privileged/asymmetric:** the runner extracts a **clean** `state` (reward + HL target unchanged)
and a **noisy** `state_n` used *only* for the LL's observed delta `V*−s_n`; the same per-step
offset is carried to the post-step goal refresh so the filter steps once per env-step. **Off by
default** → `state_n==state`, path byte-identical (RQ2-safe). Config `GoalStateNoiseCfg`
(`config/h1_2_a1/rl_cfg.py`), flags `--agent.goal-state-noise.*`. **Meaningful in `absolute` only**
— a constant bias cancels in the `delta` difference map. Launcher: `train_h1_2_noise.sh`.

### HL velocity channel: simulated leg odometry (`rl/hrl/leg_odom.py`, WL-F 2026-08-09)
What feeds `obs["hl_vel"]` (the TD3 HL's `hl_obs_vel` input), selected by **`hl_vel_source`**:
* **`state`** (default) — ground truth, plus the optional exogenous **`HlVelJitter`**. Keeps every
  pre-2026-08-09 comparator reproducible.
* **`leg_odom`** — a batched-torch transcription of the DEPLOYED estimator
  `hrl::leg_odom_velocity` (`deploy/.../hrl/base_state.h`), `v = -d(p_stance)/dt - w_P × p_stance`,
  c-averaged per HL window exactly as `State_RLHRL.cpp:444` does. **Parity verified to 6.3e-7 m/s**
  against the unmodified shipped header (`tests/test_leg_odom_parity.py` pins C++-printed golden
  vectors, so the test cannot drift into restating the code).

**Why it exists:** leg odometry is computed *from the legs*, so its error is a function of the
policy's own leg motion. On hardware that loop closes and escalates. `HlVelJitter` is i.i.d. and
policy-independent, so in sim thrashing the feet is free. Measured, same base policy, 31 iters:
`corr(leg action rate, |v_est−v_true|)` = **+0.213 mean / 29-of-31 iters positive** under
`leg_odom` vs **+0.0006 / 0-of-31** under jitter — at the **same** error magnitude (0.083 vs
0.085). A structural difference, invisible to any magnitude-only metric.

**Load-bearing details for A2/A3:**
* **Difference at the PHYSICS rate, never the control rate** (the deploy bench measured the same
  reconstruction 2.4x worse at 50 Hz). Wired as a `per_substep` metrics term — the only hook mjlab
  gives inside the decimation loop. Sim gets 200 Hz vs the robot's ~991 Hz.
* Inside that loop **only qpos/qvel are current**; `xquat`/`cvel` are one substep stale. So it reads
  the free joint directly. MuJoCo stores a free joint's angular velocity in the **body** frame and
  the root body is the pelvis ⇒ `qvel[3:6]` *is* `w_P`, no rotation.
* Feeds `obs["hl_vel"]` **only** — never `state_n` or the LL goal delta
  (`tests/test_hl_vel_jitter_isolation.py` guards both the leak and the aliasing paths).
* Latched at the fire and **held** for the window, matching `hl_vel_lo_`; a mid-window recompute
  would decorrelate the TD3 buffer's `next_s` from what the actor conditioned on.
* **`hl_vel_residual`** (task 4) layers the missing magnitude on top (encoder noise, contact
  geometry, the 200-vs-991 Hz gap), sized in quadrature against hardware: std 0.3065/0.1563, bias
  −0.0550/+0.0485. Distinct from `hl_vel_jitter` and mutually exclusive with it — jitter *replaces*
  the error model, residual *adds* to a real one; the runner raises on either misuse.
* **Eval deliberately stays on ground truth** (same precedent as `HlVelJitter`), which is what keeps
  the `g_legs`/`err_vx` anchors comparable across arms.

### Privileged environment latent `e` (WP2, 2026-08-31; WP5 will consume this)

`e = (payload_kg, com_dx, com_dy, com_dz, friction)` ∈ ℝ⁵ — **not** the plan's ℝ⁶
(payload/CoM/friction/yaw-bias): yaw bias has no sim DR mechanism anywhere in `src/`
(grep-confirmed), and a same-day check of hardware hip-yaw hang data
(`logs/deploy_safety/2026-07-28_10-13-17.csv`) found the L/R asymmetry is <1° when the
robot is genuinely motionless but ~11° during active A1a walking — a 13x gap, pointing
at a control-induced effect rather than a static encoder offset, i.e. not a physical
parameter a sim DR term could sample before a rollout. Yaw bias stays a hardware-only
F2/F3 metric (WP0/WP7); `e` covers only the four components that already have (or now
have) a DR mechanism to sample from.

**Single source of truth: `mdp.env_latent_e`** (`src/tasks/velocity/mdp/observations.py`)
reads the randomized sim state back directly — `env.sim.model.body_mass`/`body_ipos`
minus `env.sim.get_default_field(...)` for payload/CoM (both are `operation="add"` DR),
`env.sim.model.geom_friction` absolute (that DR is `operation="abs"`). No separate
tracking mechanism; both consumers below call it.

* **Payload DR**: `apply_payload_dr` (`config/h1_2/env_cfgs.py`) extends the existing
  `base_mass` event (previously reachable only via the wide-DR bundle) to 0-12 kg at the
  torso COM — same event key, same `dr.body_mass`, same torso `asset_cfg` as `base_com`.
  **Opt-in, never unconditional**: `dr.body_mass` samples RNG even at a degenerate
  `ranges=(0,0)`, so registering it on the base task would shift the RNG stream every
  later event consumes and break byte-identical replay of every existing checkpoint.
  New task variants carry it: `Unitree-H1_2-Flat-Payload` (F-mem), `Unitree-H1_2-Flat-A1-Payload`
  (H-mem, via `unitree_h1_2_flat_a1_env_cfg(payload_dr=True)`). `--eval-payload-kg`
  (`play.py`) is unaffected — it overwrites the same `base_mass` event key regardless of
  which task defines one.
* **Critic term (A0 and A1 LL)**: `env_latent_e` is a term in the existing `"critic"`
  observation group (`velocity_env_cfg.py`, `enable_corruption=False`) — same asymmetric
  actor-critic pattern already used for `Unitree-H1_2-Rough`'s `height_scan` (privileged
  value function, deployable actor untouched). Applied unconditionally to every arm's
  critic (symmetric, RQ2-safe: actor and env reward untouched) — flagged per the
  A0-comparison-cleanliness rule rather than decided silently.
* **HL-only channel (A1, TD3 only)**: `HrlRunnerCfg.hl_obs_e: bool = False` — exact
  mirror of `hl_obs_vel`/`obs["hl_vel"]`. On: the runner writes `obs["hl_e"]` at the HL
  fire step (both in `learn()` and `get_inference_policy()` — `e` is privileged ground
  truth at train AND eval, no noise model, and constant within an episode since
  payload/CoM/friction are `mode="startup"` DR, so no post-fire refresh is needed the
  way `obs["hl_vel"]`'s window-averaging needs one), and `HighLevelTd3` appends it via
  `obs_e_dim` (`_state_dim`/`_state_vec`, same shape as `obs_vel_dim`). Off →
  byte-identical, RQ2-safe. Added to `play.py`'s `structure_keys` (defaults `False`, so
  no absence-shim needed, unlike `hl_obs_vel`/`hl_velocity_goals_only` which flipped
  their defaults). **WP2 builds only this plumbing** — WP5 Phase 1 is what appends the
  *learned* `z=μ(e)` to `o^hi`; no encoder exists yet.
* **Logging**: `play.py`'s `[BENCH]`/`[BENCH_SEED]` gained `payload_kg` — the actual
  sampled value (mean over envs, read via `env_latent_e`), not the requested
  `--eval-payload-kg`, so a payload-DR training run can be segmented by payload post-hoc
  without re-running. `cot_copper` (WP1 payload pilot) was already logged; unchanged.

### Base-velocity estimator comparison (WL-G, 2026-08-11; A2 will reuse this)

`scripts/replay_base_estimators.py` replays six estimator arms offline on ONE flight-recorder
session, so robot/policy/session/encoder-offset are all removed as confounds: A deployed leg
odometry, B complementary filter, C position-only EKF (Bloesch core), D C + Rotella flat-foot
orientation, E Jacobian leg odometry, F C + vendor attitude. `--selftest` checks the measurement
Jacobians against finite differences (1e-6 / 4e-11) and the filter against synthetic truth.
Design + all frames: `doc/hrl/h1_2_ekf_design.md`; kinematics, IMU convention and the contact
frame `(0.04, 0, -0.045)`: `doc/hrl/h1_2_kinematics.md`.

Three seams that bite any consumer of the flight recorder, not just this tool:

* **`EstSample` is the only DDS-rate block** (`safety_logger.h`): `est_acc/quat/gyro/q[13]/dq[13]`
  at ~500 Hz and full float precision. Everything else on the row is the 50 Hz articulation cache.
  Enabled for A1 **and A0** (2026-08-12) — A0 is the only policy that stands genuinely still, so
  it is the only source of exact ground truth (v = 0).
* **Sessions hold several FSM entries.** Re-entry appends to the same CSV with only the `entry`
  column separating runs; replaying a file whole differences across the seams. Split on `entry`.
* **Never score across mixed regimes, and `cmd == 0` is not proof the robot is still.** The tool
  segments by command *and* verifies stillness from the encoders (moving max of `max|dq_legs|`,
  0.3 rad/s), then reports what it dropped. Both guards were bought the hard way: a pooled
  standing+walking metric produced a completely spurious winner (error cancellation), and a 5 s
  set-down transient inside a `cmd == 0` stretch carried a whole session's apparent noise
  (incumbent std vx 0.141 pooled vs 0.013 on the still part).

Verdict (2026-08-12, 3 sessions): **ship B, the complementary filter.** C fails the pre-registered
bar; the EKF arms cut noise ~2x but add bias, and leg odometry is already good on a still robot.
Numbers and the reasoning: `doc/hrl/h1_2_ekf_design.md` status block.

### Obs groups / dims (A1; 92 = ang_vel 3 + proj_grav 3 + command 3 + phase 2 + joint_pos 27 + joint_vel 27 + last_action 27)
| Network | Obs | Dim |
|---|---|---|
| HL actor | `policy ++ command [++ hl_vel] [++ hl_e]` (the A0 actor, deployable, +2 if `hl_obs_vel`, +5 if `hl_obs_e` — WP2) | 92 (+2/+5) |
| HL critic (twin Q) | `state ⊕ goal` | 92 + goal_dim |
| LL actor | `actor∖command ⊕ goal` | 89 + goal_dim |
| LL critic | `critic∖command ⊕ goal` (now includes `env_latent_e`, +5 — WP2) | privileged ok (not deployed) |
**The command is removed from the LL obs on purpose** — if the LL saw it, it would bypass
the hierarchy and A1 collapses into A0. The goal is then the only task-intent channel.

## A0 warm-start (LOAD-BEARING for the l2 kernel)
A1's LL is warm-started from a converged **A0 actor** — under the default l2 intrinsic
it can't discover walking from scratch. `--agent.warm-start-path <A0 model.pt>`.
With `ll_goal_kernel=exp` + true-terminal `fell_over`, from-scratch training works
(better-than-A0 tracking, 2026-07-06) — the warm-start is then an optimization, not a
requirement; see the A1 findings ledger (research KB) (from-scratch kernel row) for the chain.
- **Gap-aware partial state-dict copy** (`HierarchicalRunner._partial_load`): the `command`
  obs term sits **mid-vector (actor cols 6:9), not last**, so the copy drops A0's command
  cols and shifts the rest (`[0:6]←A0[0:6]`, `[6:89]←A0[9:92]`, goal cols fresh). Critic
  aligns cleanly (keeps command).
- Diagnostic it worked: **high iter-0 ep_len** (jumps to hundreds; A1 even out-tracks A0).
- Removing warm-start under **l2** is catastrophic (confirmed: `|g|` pins to 1.0,
  action_rate ~13.5 vs A0 0.66, violent).
- **Freeze-LL variant** (`HrlRunnerCfg.freeze_ll_path`, mutually exclusive with
  `warm_start_path`): strict full load of a converged A1 LL, LL acts via deterministic mean,
  no LL update — only the HL learns (isolates "can the HL learn the map at all"). A0
  checkpoints: `logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt`.
  **Gotcha — match `hl_target_mode` to the frozen LL's training HL:** a frozen LL is a fixed
  obs→action map, so the learning HL must produce V\* in the distribution the LL was trained on.
  An LL trained under the **oracle** (absolute command-range V\*, e.g. A1a's v2) needs
  `hl_target_mode=absolute`; `delta` (`V*=state+scale·g`) feeds it OOD targets at speed. A
  *co-trained* LL adapts to whatever the HL emits, so it uses the `delta` default. The frozen LL
  is saved into the run's checkpoint via `alg.save()`, so play.py reconstructs it (no
  `freeze_ll_path` needed at eval). (A1a S3, 2026-07-03.)

## Reward decomposition
| Level | Reward |
|---|---|
| **LL** | intrinsic, kernel via `ll_goal_kernel`: `l2` (default) `r_lo,i = −Σ_c w_c‖V*−s_{i+1}‖`; `exp` = A0-parity positive-bounded kernels on `V*−s` (from-scratch training; `goal_weights` is l2-only — exp weights hardcoded in `GoalSpace.reward`). No task reward (pure HIRO); `ll_task_reward_coef` blend + `ll_alive_coef` constant knobs, default 0. ADR-0002 deploy-hygiene penalties add on: posture anchor (arms+waist+hip yaw/roll, per-joint `err²` clamp 9.0) × `ll_posture_coef`, with optional **`ll_posture_weights`** per-joint multipliers (pattern→weight dict, resolved at init, unmatched=1.0, weighted *mean not renormalized* so all-ones ≡ uniform; stage-D arm-calm lever 2026-07-14: shoulders 16 / elbow+wrist 4 gives ~pose_dev 10× down; set via temp rl_cfg edit — tyro dict CLI untrusted) + whole-body action-rate × `ll_action_rate_coef`. |
| **HL** | `hl_reward_mode=task` (default): the **A0 task reward** summed over the window. `hl_reward_mode=tracking`: velocity command-tracking only (`mdp.track_linear_velocity + track_angular_velocity`, A0 exp terms reused, std read live via `reward_manager.get_term_cfg`), LL-execution penalties removed. The **env reward is unchanged** either way (no RQ2 confound) — only what the HL optimizes internally changes. |
- **`fell_over=time_out` in the A1 env ONLY** (`config/h1_2_a1/env_cfgs.py`): the LL's
  always-negative goal-distance reward makes early termination an attractor (die fast → stop
  accumulating negative reward). Marking the fall a **truncation** makes PPO/TD3 bootstrap it
  (`γ·V`) instead of cutting value to 0. A0 keeps a true terminal (A0 diverges with
  time_out). The buffer `done` / GAE mask excludes `time_outs` to express the same fix.
  **Kernel pairing: l2 ↔ `time_out`; `exp` ↔ true terminal** (override
  `--env.terminations.fell-over.time-out False`) — a positive kernel with `time_out`
  re-frees falls (bootstrap fantasy value); a negative kernel with a true terminal
  resurrects the suicide attractor.

## TD3 HL (`rl/hrl/td3.py`, `rl/hrl/storage.py`) — see `doc/hrl/A1_HIRO.md` for tuning
Self-contained TD3 over a goal-dim action. Actor MLP `[92→256,256→goal_dim]`+`tanh`; twin
critics over `[norm(state), g]`; one shared `EmpiricalNormalization(92)` applied outside the
nets (targets never hold stale stats). Tricks: soft targets `τ=0.005`, target-policy
smoothing `0.2/0.5`, clipped double-Q, delayed actor `policy_freq=2`, exploration noise.
Flat GPU ring replay buffer; HIRO relabeling toggled by `relabeling` (`none`|`hiro`),
implemented inline (no separate `relabeling.py`). Checkpoint `hl` key holds
actor/critics/targets/normalizer/optimizers (resume-safe); replay buffer not saved
(refills in ~40 iters).

## Tools
- **Radar comparison chart** (2026-08-10): `scripts/plot_radar.py <baseline.log> <run2.log>
  ...` (each a `play.py` stdout capture, or bare JSON, with exactly one `[BENCH]` line) →
  one figure, every run normalized to the baseline's `[BENCH]` metrics (baseline traces the
  regular polygon at radius 1.0, outward = better on every scored axis). `stride_period_s`
  is plotted but deliberately unscored (descriptive only, no better/worse judgement).
- **Benchmark** (canonical comparator): `play.py --checkpoint-file <pt> --num-envs 64
  --eval-steps 600 --eval-seeds 2` → deterministic multi-metric scorecard (err_vx/vy/yaw,
  fall_rate, action_rate, orient_dev, height_dev) + `[BENCH] {json}`, multi-seed mean±std.
  **Use `fall_rate`, NOT `ep_len`, for survival** (`episode_length_s=1e9`, no resets in play).
  A0 baseline 0.09/0.11/0.10. Strips exploration noise → judge HLs by this, not training err.
- **`--num-envs` is part of the measurement, not a perf knob (2026-08-09).** Every eval/probe
  reports a distributional mean, so at 1 env a single command draw *is* the sample and the
  result goes bimodal: the same A0 checkpoint gave `jacc_legs` 24.75/25.14/25.81/26.23 **and
  42.27** across five draws (±26%), and `err_vx` 0.053–0.097, vs ±2.6% at 64 envs. Eval paths
  (`--eval-steps`/`--diagnose-*`/`--check-vel-increment`) therefore **default to 64** and warn
  below 8; interactive play still gets 1. Numbers taken at different env counts are **not
  comparable** — this silently corrupted results twice (2026-07-15 goal probe, 2026-08-09
  jacc benches). **Measured 64-env noise floor: ±2.6% on `jacc`**, so orderings inside ~3%
  (e.g. full_jacc 35.8 vs full_jacc_noEnergy 35.3) are NOT resolved at n=1 seed.
- **Smoothness metrics — read all three, they disagree (2026-08-09).** `action_rate` is
  whole-body and on *commanded actions*; `act_legs` restricts it to hip/knee/ankle (the
  joints that keep the robot up, and it uses ARDIAG's exact `g_legs` formula, so the two
  probes cross-check); `jacc`/`jacc_legs`/`jacc_arms` (+`_p95`) measure *realized* motion,
  and since `τ = M(q)q̈ + C + G` they are the physically deploy-relevant quantity. **Always
  report the p95 next to the mean**: A0's leg p95/mean is 2.39 (low sustained accel, hard
  contact spikes on its 0.59 s stride) vs A1's ~1.17 (high sustained, softer spikes on a
  0.35-0.43 s stride) — the mean alone hides that opposition. A0 ref: act 0.641, act_legs
  0.593, jacc 19.8/29.5/12.2, jl_p95 70.6. **The metrics reorder candidates** — combo1 is
  best on `action_rate` and worst of 13 on `jacc_legs_p95` (1.5× A0), so never rank a deploy
  shortlist on `action_rate` alone.
- **Lean / calibration-sensitivity probe** (ADR-0006, arch-agnostic): `play.py
  --checkpoint-file <pt> --num-envs 64 --eval-seeds 2 --eval-cmd-vx 0 --probe-lean 400`
  → steady-state base pitch + leg joint speed + action rate at a held zero command, per
  swept perturbation, with `[LEANPROBE] {json}`. Two modes: `--probe-lean-bias/-joints`
  sweeps a **deterministic `encoder_bias`** on a leg group (d(lean)/d(calibration error),
  superposition across joints holds to 2.3%); `--probe-gravity-noise` instead corrupts the
  **`projected_gravity` observation** with the encoder clean, which separates IMU-dependence
  from encoder-OOD. Both force `joint_pos` to `biased=True` for the eval — **without that
  flag mjlab shows the policy the TRUE joint angle, so the bias is trivially nulled and the
  probe measures nothing.** The gravity mode also re-enables `enable_corruption` (play mode
  strips it) and zeroes every other actor term's noise, so gravity is the only varying
  channel. **Judge policies on ABSOLUTE `leg|dq|`/`act_rate`, not the ratio to their own
  control** — a policy with a worse baseline flatters itself in ratio terms.
- **Goal probe** (A1 HL-vs-LL error decomposition): `play.py --checkpoint-file <pt>
  --diagnose-goals 600 --eval-seeds 2 --num-envs 64` → per-window `(command−achieved) =
  (command−V*) [HL goal err] + (V*−achieved) [LL reach err]`, raw `|g|` + saturation frac,
  fwd/bwd vx split, realized/requested ratio + `[GOALDIAG] {json}`. Decomposition closes
  exactly. **Pass `--num-envs` — the probe defaults to 1 env.** With `--eval-cmd-vx` it also
  prints `[HOLDDIAG]`: per-env bwd/fwd group split with signed goals, `|g|`, HL period, and
  per-window goal-vs-achieved traces (the probe that appeared to expose a 2026-07-15
  velocity-hold HL degeneracy — **which was itself an artifact of this very flag: FALSIFIED
  2026-07-16, WL-C.** `--eval-cmd-vx` collapses the twist ranges to a point and the HIRO goal
  scale used to be derived from them → scale hit its 1e-3 floor → `V* ≈ s`, inerting the goal
  channel. **Fixed by baking `goal_scale` into the checkpoint** (`GoalSpace.freeze_scale`;
  play.py prints `[SHIM] ...` for pre-2026-07-16 checkpoints). Any `--eval-cmd-vx` `|g|` /
  `ll_err` / `[HOLDDIAG]` read from BEFORE that fix is meaningless; see the A1 findings ledger (research KB)
  WL-C). **Caveat: pre-2026-07-15 probe numbers on cadence-HL checkpoints ran with a
  frozen `hrl_phase` clock** (the loop didn't advance it; de-entrained LL ⇒ flattering) —
  fixed to mirror `get_inference_policy`.
- **Action-rate decomposition** (smoothness, 2026-07-28): `play.py --checkpoint-file <pt>
  --diagnose-action-rate 600 --eval-seeds 2 --num-envs 64` → bins `||a_t − a_{t−1}||` by
  position in the HL window (`step % c`; bin 0 = the fire step, where the goal obs jumps) and
  by joint group (legs/arms/waist), + `[ARDIAG] {json}`. Answers "is A1's twitch the goal
  channel stepping at `1/(c·dt)` Hz, or a uniform floor?" Key fields: `fire_excess`
  (= bin0 ÷ mean of the rest; **A0 ≈ 0.98 is the flat control** — run it, any structure there
  is a binning artifact), `ar_mean` (reproduces the bench `action_rate`, a built-in
  cross-check), and the per-group shares. Works on A0 (no `c`; defaults to 8).
- **Training-time smoothness (W&B run filtering, 2026-08-09):** `Loss/metrics/act_rate`,
  `act_rate_legs`, `jacc`, `jacc_legs` are logged **every step, unconditionally, independent
  of every `ll_*_coef`**, and never enter `r_lo`. Filter runs on these, **not** on
  `ll/action_rate_pen` — that key is `coef × value` accumulated only when its coef is
  nonzero, so it is incomparable across coefs and reads a flat 0.0 at coef 0, which looks
  like perfect smoothness but is no measurement. Same L2-norm convention as the bench, so a
  converged run's curve is directly comparable to its `[BENCH]` number (early iterations sit
  ~8× higher purely from the untrained action std).
- **IMU velocity-increment bench** (deploy feasibility, 2026-08-01): `play.py
  --checkpoint-file <pt> --num-envs 64 --check-vel-increment 480 --eval-seeds 2` → per-axis
  RMS error of an IMU-only reconstruction of the within-window base-velocity increment, over
  a rung ladder (raw / gravity-removed / +Coriolis / vs IMU site / waist-corrected @50Hz /
  @200Hz) + `[VELINC] {json}`, with a verdict against the ≤0.05 / 0.05–0.15 / >0.15 m/s bands.
  **The mechanism it exists to exploit — load-bearing for every deploy of a `delta`-mode
  hierarchy:** the LL goal obs is `V*−s_i = scale·g − (s_i−s_t0)`, so **absolute base velocity
  cancels**. Deploy needs only the *increment* since window start, which resets every `c` steps
  (0.16 s at c=8) so drift cannot accumulate — which is why A1 is deployable despite E1 proving
  the onboard absolute estimator absent. Two corrections the bench forced: gravity removal must
  happen in the **torso** frame (the `imu` site is on `torso_link` behind the yaw `torso_joint`,
  NOT the pelvis whose velocity the goal space uses — `h1_2.xml:138-145`), and the residual is
  dominated by **sampling rate**, so the deploy integrator belongs in `State_RLHRL::run()` (1 kHz),
  never in `policy_step()` (50 Hz). Numbers + verdict → the A1a deploy journal (research KB), 2026-08-01.
- ⚠ **The `delta` cancellation covers the LL ONLY, and the gap is now MEASURED (2026-08-05).**
  Under `base_vel_from_imu` the increment resets at every window start, so the HL's `(vx,vy)`
  input is **exactly 0.000000 at every fire step** (`base_vel_increment(psi,w,0,lev0) = 0` by
  construction; fire steps are identifiable in telemetry as the exactly-zero `est_vx` rows,
  modal gap = `c`). An HL trained on absolute velocity therefore reads "stationary" every
  time it fires. Bridge A/B on the shipped plant, same binary/plant/sequence: **estimator
  3/3 falls, ground truth 0/3.** The estimator itself is NOT at fault — it tracks height to
  ~5-13 mm and velocity to ~0.02-0.06 m/s up to the fall. Consequence: a `delta` hierarchy
  with `hl_obs_vel=True` needs a real absolute-velocity source (leg odometry) before deploy;
  the IMU increment alone is enough for the LL and not for the HL. Numbers → the A1a deploy
  journal (research KB), 2026-08-05.
- **Leg-odometry velocity bench** (the HL's ABSOLUTE velocity, which `delta` does NOT cancel):
  `play.py --check-leg-odometry 480 --num-envs 64 --eval-seeds 2` → `[LEGODOM] {json}`.
  **RESOLVED ON THE DEPLOY SIDE 2026-08-06** — `hrl.hl_vel_from_leg_odom` (C++,
  `hrl/base_state.h::leg_odom_velocity`): differenced per tick in `run()` (~991 Hz),
  c-averaged, latched at each fire, wired to the **HL only** so the LL's goal delta keeps
  the IMU increment. Bridge A/B (n=3, one binary, true falls `gt_h<0.9`): gt 0/3,
  IMU increment 3/3, **leg odometry 1/3**; open loop passive 0.047/0.053, closed loop
  0.083/0.121. ⚠ Fisher p=0.40 at n=3, and `arm4d` predates `HlVelJitter` ⇒ "estimator
  scored, policy pending". The bench point is the foot **SITE** `(0.04,0,-0.04)` in
  ankle_roll, NOT the sole `lowest_foot_z` uses — 0.13 m apart in x, and `p_foot` enters
  `w x p`. New rung `fastfilt_*` = the same estimate scored against the WINDOW-MEAN truth,
  i.e. tracking error with the c-averaging lag removed; the pair (`fasthl_*` vs
  `fastfilt_*`) is what separates "the estimator is wrong" from "the average is stale".
  `v_pelvis_b = -d(p_foot_b)/dt - w_b x p_foot_b` from the stance foot (gravity-projected
  lower foot; `LowState` has no foot force sensor), scored vs truth with phase/speed splits,
  a physics-rate rung and contact-ORACLE reference rungs. Ruled out by it: `tau_est` load
  stance (net regression) and contact detection in general (a perfect contact signal buys
  ~nothing — 85% of steps are single-support, where there is no attribution ambiguity).
- ⚠ **Rate gotcha, cost 2-4x and flipped two verdicts on 2026-08-03: never finite-difference
  or integrate a fast signal at the 50 Hz control rate.** Both `[VELINC]` and `[LEGODOM]`
  first measured at `step_dt` and read 2-4x worse than at the physics rate. Benches must hook
  `metrics_manager.compute_substep()`, which the decimation loop calls per substep
  (`manager_based_rl_env.py:421-427`).
  ~~**But the deploy platform does NOT sustain 1 kHz**~~ **RETRACTED 2026-08-04. The
  deploy platform DOES sustain ~1 kHz; the "10 ms tail" was a printf artifact.** Two
  defects made the CSV unable to measure rate at all: (i) `t` was `tick_counter × nominal
  dt` (`safety_logger.h:133`), an iteration index, never a clock; (ii) `setprecision(4)` is
  4 SIGNIFICANT digits, so past t=10 s the column quantised to 0.01 s bins — which is the
  entire "p90=p99=max=10.00 ms". Below 10 s the same files show dt ∈ {1 ms, 2 ms} and
  nothing else, exactly what `log_every_=2` predicts. The three "mean rates"
  (942/525/700 Hz) were logged-row fractions, i.e. `trig_joint` in disguise (triggered
  ticks bypass decimation). **Measured rate: 990.4 Hz**, now read directly off the new
  `t_wall` column, and independently 989±2 Hz from comparing `bridge_session.py`'s scripted
  hold durations against `t`. Corroborating: the bridge publishes `lowstate` at 1 kHz
  (`unitree_sdk2_bridge.h:170-171`). **Lesson: never measure a rate with a clock you have
  not verified is a clock.**
- ⚠ **Verify which policy is actually deployed before any bridge run.** Use
  `scripts/deploy_provenance.py --check --checkpoint-file <pt>`: it md5s the deployed ONNX
  against a reverse index of `logs/rsl_rl/**/*.onnx` and **names the source run**, then
  fails if that is not the run you asked to validate. Never trust the ONNX `run_path`
  metadata (training-time exports write `local`). On 2026-08-03 three bridge runs and an
  estimator measurement were invalidated because the deploy dir held a parked WL-D arm-6
  export (`rolloverfix0p5`) that fell reproducibly. The deploy dir is a single mutable slot
  with hardcoded filenames (`State_RLHRL.cpp:206-207`), so `--stage` now owns filling it and
  writes `exported/PROVENANCE.json` recording run, checkpoint and md5s.
  **`onnx_parity.py --onnx-dir` scores the DEPLOYED file** — without it that tool only ever
  compared a checkpoint against its own fresh temp export, so a correct export function plus
  a stale deployed file passed clean, which is why W4 could never have caught this.
  Attribute any bridge failure by **policy-swap against A0 as control** before suspecting
  code: A0 exercises the shared layer (articulation, `joint_offset`, safety filter) without
  the HRL path, so A0-clean + HRL-fails isolates the fault to the HRL path or the policy.
- **`find_joints`/`find_sites` return MODEL order, not query order** (`preserve_order=False`
  by default, `mjlab/entity/entity.py:505,549`). Pass `preserve_order=True` and assert the
  names whenever index order is load-bearing (e.g. pairing per-leg joints with foot sites).
- **Play structure-restore:** play.py restores structure keys (`c, goal_components,
  goal_weights, hl_algorithm, hl_ppo, hl_td3, relabeling, gamma_hi, hl_target_mode, hl_obs_vel,
  hl_cadence, cadence_period_range, ll_cadence_coef, hl_cot_coef, hl_cadence_source,
  cadence_swing_time, cadence_duty_range`) from the run's own
  `params/agent.yaml` (deep-merge dict cfgs over defaults). **Every new `HrlRunnerCfg` structure
  field MUST be added here at creation** — a missing key silently rebuilds the runner with that
  feature OFF (2026-07-02: missing `hl_cadence` → fixed 0.6 clock, `--eval-cadence-period` inert
  → weeks of false "no-entrainment" verdicts, only caught by `gait_match` being identical to 3
  decimals across all commands). Re-derives goal obs dim. Pre-2026-06-11 A1 replays (no restore)
  ran oracle/goal-0 → void; W&B training metrics were always ground truth.
  **Absence-shim rule:** when a structure field's *default* later flips (e.g.
  `hl_velocity_goals_only` → True 2026-07-09, `hl_obs_vel` → True 2026-07-10), absence from an
  old yaml must restore the OLD default explicitly — "absent keeps defaults" silently rebuilds
  wrong-shaped nets (2026-07-15: missing `hl_obs_vel` shim broke every pre-velobs TD3 replay,
  92 vs 94 dims).
- **A1a fixed-command / stride-period eval** (`play.py`, eval-only): `--eval-cmd-vx/vy/wz` pin the
  twist command (standing/heading off) so a stride sweep isolates the commanded period from the
  natural v→period map; `--eval-cmd-heading <rad>` holds a world heading (P-controller corrects
  wz) for straight replays; `--eval-cadence-period <s>` pins the HL stride period (needs
  `hl_cadence`). Holds also report `ss_err_vx/vy` (last 2/3, separates the accel ramp from held
  tracking) + `t90_s` (time to 90% of commanded vx; NaN = never) since 2026-07-14. **Batch means
  hide bimodal hold failures — always check the `[HOLDDIAG]` per-env split** (a mean achieved
  +0.15 coexisted with 27/64 envs walking backwards, 2026-07-15). New `[BENCH]` metrics: `mech_power_w`, `cot` (dimensionless E/(m·g·d), m=75 kg,
  command-gated), `stride_period_s` (footfall interval), `gait_match` (feet_gait agreement vs the
  commanded clock: ~1 locked, ~0.5 drifting). (H1 phase-slaving probe found A0 is slaved to its
  clock obs, match 0.973→0.51 when scrambled; the `--eval-phase-obs` diagnostic that showed this
  was removed 2026-07-03 as spent.)
- **A1a S1c cadence-HL** (2026-07-02): `hl_cadence_source: "random"|"hl"` — `hl` makes the TD3 HL
  emit the stride period as +1 tanh action dim (action = goal_dim+1; relabel carries the period
  column through untouched), affine-mapped to `cadence_period_range`, written to `hrl_period` per
  fire (phase integrates incrementally → no clock jump). CoT enters the HL window reward:
  `-hl_cot_coef · E_win/(m g · max(d_win, 0.1·c·dt))`, per-step command-gated (all-standing window
  = exactly 0). Logs `hl/cot_pen`, `hl/period_mean`. Default `random` keeps pre-S1c checkpoints
  loading with goal_dim-sized HL nets. Eval: with a cadence-HL ckpt, `--eval-cadence-period`
  *overrides* the HL's own period at each fire; without the pin the HL's period runs.
- **d(T) duty schedule** (2026-07-03, slow-stride enabler): `cadence_swing_time` (s) > 0 makes
  the `feet_gait` stance threshold period-dependent, `d(T) = clamp(1 - swing_time/T,
  *cadence_duty_range)` — single support held ~= swing_time (human strategy; double stance
  absorbs long periods). 0.31 = continuity calibration (floor departs at T = 0.31/0.44 ≈ 0.70 s,
  fast band byte-identical; NOT the pendulum constant 0.30, see A1a_plan). `cadence_duty_range`
  (default (0.56, 0.70)) is the single source of truth for BOTH `feet_gait` and play.py's
  `gait_match` (both restored via structure_keys — they must never diverge, or eval fabricates
  gait drift). Floor 0.5 = walk-run boundary; running needs more than the floor (see the cfg
  docstring). Sim-only: the phase obs / deploy path are untouched.

- **Regression suite** (`tests/`, born 2026-07-29): `pytest` — 49 tests, ~3 s, CPU-only (no
  MuJoCo/GPU/checkpoints; a duck-typed `FakeEnv` in `conftest.py`). Four seams, each chosen
  because a defect there yields a confident WRONG RESULT rather than a crash:
  `test_goal_space.py` (scale pin/derive, delta-vs-absolute decode, `to_g` round-trip,
  `task_only` nominal pin, kernel sign invariants), `test_rewards.py` (push-off direction +
  contact gate, commanded-vs-episode gait clock, d(T) duty schedule), `test_warm_start.py`
  (the gap-aware column map below), `test_deploy_parity.py` (all 27 joint limits vs the
  compiled model, obs-vector layout, PD gains / default pose / action scale / cadence range
  vs the training config, baked `goal_scale` metadata). A2/A3 inherit all four.
- **Prove new tests can fail:** `python scripts/check_test_sensitivity.py [name-substring]`
  re-introduces each historical defect in the source and asserts the suite goes red (19/19).
  A test written against already-correct code is green on arrival and proves nothing
  otherwise. The harness edits files in place and refuses to restore if one changed
  underneath it (concurrent session) rather than clobbering the other edit.

## Checkpoint / ONNX / deploy
- `model_<it>.pt`: LL actor/critic/normalizer + `hl` (per-algorithm) + iter.
- **ONNX:** two files `high_level.onnx` (state→goal) and `low_level.onnx` (state→27) via
  `HierarchicalRunner.export_hierarchy_to_onnx` (per-HL `as_onnx`: ppo mean / td3 `tanh`, +
  normalizer baked in; oracle skips the HL net). Parity-checked vs `get_inference_policy` ~1e-6.
- **Deploy DONE (2026-06-18, sim):** C++ `State_RLHRL` runs `low_level.onnx` every step +
  `high_level.onnx` (or analytic oracle `V*`) every `c` steps; LL input = `proprio∖command ⊕
  goal(=V*−s)`. A2/A3 reuse this two-ONNX pattern + `include/hrl/goal_space.h`. Needs a runtime
  base-velocity+height estimate (sportmode HighState in sim) → full mechanism, `hl_algorithm`
  oracle/learned switch, `state_noise` test knob, and the sim2real noise finding in
  `deployment.md`.

## Frozen LL (`freeze_ll_path`) — WP5

`_load_frozen_ll` loads the **actor strict** and the **critic best-effort**. The asymmetry
is deliberate: a wrong-shape actor means a mismatched goal space and must raise loudly,
while the critic is never evaluated or updated when the LL is frozen (`learn()` guards
`process_env_step` / `compute_returns` / `update` on `not self.freeze_ll`). This is what
lets a pre-WP2 checkpoint (critic obs **114**) be frozen inside the post-WP2 env (critic
obs **119**, the unconditional privileged `env_latent_e` term). ⚠ A checkpoint saved by a
frozen-LL run therefore carries an **untrained LL critic** and must not be resumed as an
unfrozen run — warned at load.

⚠ **Generalisable, and it cost WP5 a blocked launch:** WP2's inertness proof was thorough
and still could not have caught this, because it verified through `play.py`, which never
strict-loads the critic. **An asymmetric actor-critic change is invisible on the eval path
and fatal on the train path — score it with a train-side smoke, not a play/bench run.**

## Bar B statistic (`scripts/period_payload_stats.py`, `[PERIODDIAG]`)

`period_payload_stats(per_w, keep_w, e)` → per-env mean commanded stride period vs the
per-env latent: Pearson `r` + slope against payload, plus per-component reads (friction,
CoM). Printed by `play.py --diagnose-goals` as `[PERIODDIAG] {json}`.

- **The independent unit is the ENV, not the window.** `mode="startup"` DR fires at
  construction and never on reset, so every window from env *i* carries the identical
  label. Pooling by window inflates *n* ~150x and shrinks every interval **while leaving
  the point estimate roughly right** — it fails silently. The same fact makes a
  random-window train/test split leak: hold out **envs**.
- Correlations return **NaN, never 0.0**, when a regressor has no variance (the
  fixed-payload probe case).
- Its own module because `play.py` imports the whole mjlab env stack at module scope; a
  test importing `play.py` would pull MuJoCo in and break the suite's CPU-only contract.
- ⚠ **A saturated cadence has no variance.** `sigma_e` measured while the commanded period
  sits on the `cadence_period_range` bound (e.g. 0.00087 s at `hl_cot_coef=0.2`) is a
  property of the clamp, not the policy — off the bound it is **~12x larger** (0.0105 s).
  Never quote a Bar B noise floor from a pinned operating point.
- ⚠ **Score the objective on the REWARD's CoT, not the bench metric.** The HL reward's CoT
  denominator is the SIGNED projection on the commanded direction (`hrl_runner.py:955-957`);
  `play.py`'s `cot` (`play.py:779`) is the UNDIRECTED norm. Ratio 1.007-1.056 at vx=0.5 —
  small, but it moved a predicted Bar B `r` from +0.939 to +0.737.
- Overlaps `--diagnose-cadence`'s `[CADDIAG]` on three aggregates (`period_mean`,
  `period_std_win`, `period_spread_env` = `sigma_e`); `[CADDIAG]` owns transmission /
  entrainment, `[PERIODDIAG]` owns the latent correlation. **Consolidation candidate:**
  have `--diagnose-cadence` call this module rather than recomputing.
- Emits per-env `payload / period / friction / com_dx / com_dy / com_dz`, all masked to the
  same envs so a caller's regression stays row-aligned. **You need the CoM columns**: a
  payload-only correlation buries the other four components in its residual (below).

## Counterfactual `e`: separating "reads the latent" from "feels it" (WP5, 2026-09-02)

`play.py --cf-e-col <0..4> --cf-e-val <x>` forces one column of the latent **the HL reads**
to a constant — `e` = (0 payload, 1 com_dx, 2 com_dy, 3 com_dz, 4 friction) — while every
body/geom property stays exactly as sampled. Requires `--diagnose-goals` (the only path that
writes `obs["hl_e"]` itself); the forced value is stamped into `[PERIODDIAG]` so a falsified
run's JSON is distinguishable from a true one. Keep the value inside the trained DR support
(payload 0-12, com_d\* ±0.05, friction 0.3-1.6) or you measure extrapolation, not sensitivity.

**Why it is necessary, and what a pinned sweep cannot do.** `--eval-payload-kg` moves physics
and observation *together*, so `corr(commanded T, true latent)` is reproduced exactly by a
policy that reads nothing and merely responds proprioceptively to how the robot is moving.
The falsification breaks that tie. All envs get the SAME forced value, so their true per-env
spread becomes identical noise in every arm and the between-arm contrast is clean.

**Read it as `read%` = counterfactual ΔT / observational ΔT**: ~100% means the response runs
through the observation, ~0% means the HL is feeling the physics. Measured on Phase 1
(2 seeds, physics pinned 6 kg): the observation alone moves commanded T by **0.070 / 0.068 s**
(15.4 / 16.4 SE), but **`read%` is seed-dependent** — `com_dx` 84% on s42 vs **17%** on s123
despite an observational t of −26.8 on *both*. ⚠ **A large observational coefficient is not
evidence the policy reads the value**; only `friction` was read on both seeds (75/97%).

## ⚠ A correlation gate on ONE component of a multi-component latent is under-powered

Bar B pre-registered `corr(commanded T, payload) >= 0.5`. Its denominator carries the other
four components' variation as "noise", so it can **fail while the mechanism it tests is
present**: on Phase-1 seed 123 the HL demonstrably reads payload (counterfactual 9.7 SE,
ΔT +0.041 s) and the gate still read **+0.268**. Partialling the other four out gives +0.485;
the counterfactual slope gives +0.595. Partial the other components out **when you write the
gate**, not after seeing the data — post-hoc estimators are diagnostics, never a rescue.

## ⚠ Observational partials cannot predict a counterfactual (WP5, 2026-09-02)

Two traps, both hit while predicting a coupled-ray response from a fitted regression:

1. A regression `beta` mixes the **read path** and the **proprioceptive path**; a falsification
   exercises only the read path. A component with a large observational `t` may be barely read
   (WP5 seed 123: `com_dx` t = −26.8, read% **17**), so recombining `beta`s over-predicts.
   **Predict a counterfactual with counterfactual coefficients.**
2. The read-path response can be strongly **asymmetric** about 0 — WP5 `com_dx`: −1.010 s/m
   backward vs −0.395 s/m forward, **2.56x** — so a slope fitted across the full DR range is
   wrong for a manipulation that traverses only one side. **Fit over the interval the
   manipulation actually covers.** Correcting both took a 2.3x over-prediction to 6%.

## ⚠ Independent DR events can make a deployment direction unrepresentable (WP5, 2026-09-02)

`base_mass` and `base_com` are separate `mode="startup"` events on the same `torso_link`, so the
sim can add 12 kg without moving the CoM. Hardware cannot: a pack of mass `m` at offset `d` from
the torso CoM shifts it by `dx = m*d/(M_torso + m)`, `M_torso` = **17.789 kg** (`h1_2.xml`). Two
consequences. **(a)** A regression over that DR yields *partial* derivatives — the deployment
quantity is the *directional* derivative along the realizable ray, and when the partials have
opposite signs the two point in different directions. **(b) Check the ranges are mutually
consistent**: `base_com`'s ±0.05 m does not cover what `base_mass`'s own 0-12 kg physically
implies past `d ~ 0.10 m` (at `d`=0.20 m only **5.9 kg** stays in support). Falsify several
columns together with `--cf-e-col 0,1 --cf-e-val <m>,<dx>` to measure the ray directly.

## T\* fits are fit-window-sensitive (WP5, 2026-09-01)

A local quadratic vertex on a CoT bowl with a **steep, payload-dependent far arm** is
biased: a wider window drags the vertex toward the shallower side, and if that asymmetry
varies with payload it *manufactures* a `T*(payload)` shift. Measured on the rs8st15 LL:
3-pt −0.0080 (DOWN) vs 5-pt +0.0070 (UP) vs 7-pt +0.0173 (UP) — the **window systematic
(0.038 s) is 2-5x the effect**. WP1b's keeper result survives the same check (mech DOWN
3/3 windows, copper UP 3/3) because its effect is ~4x larger and its bowls deeper, but its
**magnitude spans 3.5x across windows**. **Report `T*` direction with all three windows;
a bootstrap CI over repeats does NOT include this systematic and will read far too tight.**

## Gotchas (apply to all HRL arches)
- **The run-to-run floor is REGIME-DEPENDENT — 4x at standing, ~10% at walking.** Measured
  2026-08-27 on two config-identical, same-seed-42 replicate pairs: walking `act_legs`/`ajit`/
  `jacc` reproduce to 1.02-1.31x, while standing `ajit` spans 4.15x, standing touchdowns 4.22x,
  `|g|vy` 3.57x and held `ss_err_vx` 3.32x (one replicate never reached the commanded speed,
  `t90 = nan`). This is TRAINING noise, not bench noise — re-measuring the same checkpoint
  reproduces to 1.6-4.2%. Standing is a near-marginal equilibrium (quiet double-support vs a
  corrective limit cycle is a bifurcation); walking is a limit cycle either way. **Score any
  standing arm against a replicate band, and attach a measured floor to any pre-registered
  stop rule on a standing metric** — a 30% change is unresolvable. Also: at `cmd 0` the MEAN
  sits above the p95 for quiet policies (the set-down transient carries it), so mean and p95
  can disagree in direction; report both.
- **Score smoothness per REGIME — the aggregate bench cannot see a standing defect.**
  `rel_standing_envs=0.05`, so the aggregate is ~95% walking; a 31-44x zero-command defect
  reported as "+24%" (2026-08-26). Use `--eval-cmd-vx 0.5` for walking and `--eval-cmd-vx 0.0`
  for standing, always separately. Vocabulary and the metric set → `CONTEXT.md`.
- **`--diagnose-symmetry` is SILENTLY SKIPPED when `--eval-steps > 0`** — the eval block
  ends `env.close(); return` before reaching it, so no `[SYMDIAG]` is emitted and the touchdown
  counts are simply absent. Run it as its own invocation.
- **`cadence_period_range` IS settable from the CLI**, with Python syntax and quoted:
  `--agent.cadence-period-range "(0.625,0.625)"`. Only the space-separated form raises
  "Unrecognized options" (mjlab sets `tyro.conf.UsePythonSyntaxForLiteralCollections`). The
  2026-07-03 note read that as "not overridable" and routed range changes through code edits
  for two months — retracted 2026-08-26.
- **Commanded-side smoothness is reported in radians, never raw action units** — `kappa =
  0.25*tau_max/Kp` spans 6.7x across the body, so whole-body raw norms over-weight the arms
  3-6.7x. Legs-only is unaffected in ranking (kappa is uniform 0.25 across all 12 leg joints).
- **The exp kernel's "strictly positive" guarantee rests on `orientation` being in the goal
  space** (found 2026-07-29 while writing `tests/`). `exp(-d²/σ²)` underflows to *exactly*
  0.0 in float32 past `|Δv| > 4.7 m/s` / `|Δh| > 0.94 m`; the sum stays positive only
  because projected gravity is a unit vector, capping `‖Δorient‖` at `2√3` so that term can
  never underflow. A velocity-only goal space would let the per-step reward reach 0, break
  the exp ↔ true-terminal pairing rule, and could resurrect the suicide attractor. Asserted
  by `test_exp_positivity_depends_on_the_bounded_orientation_term`.
- **Deploy command ranges are an operator safety clamp and deliberately do NOT match
  training** (`ang_vel_z` ±0.5 vs ±1.0 trained). Never "fix" them to match; the goal scale
  travels with the policy via ONNX metadata precisely so the clamp is free to differ.
- A1 LL `entropy_coef=0.005` (0.01 lets action std blow up to ~2 and collapse);
  `desired_kl=0.005`; LR adaptive, sits near floor 1e-5.
- Same-config runs diverge a lot (GPU non-determinism + RL chaos). Treat `num_envs` as a
  hyperparameter: hold it fixed within a comparison set; use ≥2 seeds.
- **Always det-eval (benchmark) before committing to a fix** — training-metric asymmetries
  mislead. Corollary: a det-eval verdict is **regime-specific** — F4 velocity-obs benched as
  a no-op under `absolute`, but is a large linear-tracking win under `delta`; re-test a
  "no-op" when the regime changes.
- Launch via the activated env + `python scripts/train.py` directly — **NOT `conda run`**
  (buffers output, breaks live wandb).
