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
requirement; see `doc/hrl/A1_findings.md` (from-scratch kernel row) for the chain.
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
- **Goal probe** (A1 HL-vs-LL error decomposition): `play.py --checkpoint-file <pt>
  --diagnose-goals 600 --eval-seeds 2 --num-envs 64` → per-window `(command−achieved) =
  (command−V*) [HL goal err] + (V*−achieved) [LL reach err]`, raw `|g|` + saturation frac,
  fwd/bwd vx split, realized/requested ratio + `[GOALDIAG] {json}`. Decomposition closes
  exactly. **Pass `--num-envs` — the probe defaults to 1 env.** With `--eval-cmd-vx` it also
  prints `[HOLDDIAG]`: per-env bwd/fwd group split with signed goals, `|g|`, HL period, and
  per-window goal-vs-achieved traces (the probe that exposed the 2026-07-15 velocity-hold HL
  degeneracy). **Caveat: pre-2026-07-15 probe numbers on cadence-HL checkpoints ran with a
  frozen `hrl_phase` clock** (the loop didn't advance it; de-entrained LL ⇒ flattering) —
  fixed to mirror `get_inference_policy`.
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
