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

### Obs groups / dims (A1; 92 = ang_vel 3 + proj_grav 3 + command 3 + phase 2 + joint_pos 27 + joint_vel 27 + last_action 27)
| Network | Obs | Dim |
|---|---|---|
| HL actor | `policy ++ command` (the A0 actor, deployable) | 92 |
| HL critic (twin Q) | `state ⊕ goal` | 92 + goal_dim |
| LL actor | `actor∖command ⊕ goal` | 89 + goal_dim |
| LL critic | `critic∖command ⊕ goal` | privileged ok (not deployed) |
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
- **Benchmark** (canonical comparator): `play.py --checkpoint-file <pt> --num-envs 64
  --eval-steps 600 --eval-seeds 2` → deterministic multi-metric scorecard (err_vx/vy/yaw,
  fall_rate, action_rate, orient_dev, height_dev) + `[BENCH] {json}`, multi-seed mean±std.
  **Use `fall_rate`, NOT `ep_len`, for survival** (`episode_length_s=1e9`, no resets in play).
  A0 baseline 0.09/0.11/0.10. Strips exploration noise → judge HLs by this, not training err.
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
- **Leg-odometry velocity bench** (the HL's ABSOLUTE velocity, which `delta` does NOT cancel):
  `play.py --check-leg-odometry 480 --num-envs 64 --eval-seeds 2` → `[LEGODOM] {json}`.
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

## Gotchas (apply to all HRL arches)
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
