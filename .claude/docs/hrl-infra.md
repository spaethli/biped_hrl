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

## A0 warm-start (LOAD-BEARING)
A1's LL is warm-started from a converged **A0 actor** — on-policy PPO can't discover
walking from scratch in feasible time. `--agent.warm-start-path <A0 model.pt>`.
- **Gap-aware partial state-dict copy** (`HierarchicalRunner._partial_load`): the `command`
  obs term sits **mid-vector (actor cols 6:9), not last**, so the copy drops A0's command
  cols and shifts the rest (`[0:6]←A0[0:6]`, `[6:89]←A0[9:92]`, goal cols fresh). Critic
  aligns cleanly (keeps command).
- Diagnostic it worked: **high iter-0 ep_len** (jumps to hundreds; A1 even out-tracks A0).
- Removing warm-start is catastrophic (confirmed: `|g|` pins to 1.0, action_rate ~13.5 vs
  A0 0.66, violent). Not optional.
- **Freeze-LL variant** (`HrlRunnerCfg.freeze_ll_path`, mutually exclusive with
  `warm_start_path`): strict full load of a converged A1 LL, LL acts via deterministic mean,
  no LL update — only the HL learns (isolates "can the HL learn the map at all"). A0
  checkpoints: `logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt`.

## Reward decomposition
| Level | Reward |
|---|---|
| **LL** | intrinsic `r_lo,i = −Σ_c w_c‖V*−s_{i+1}‖` (sim velocity subspace). No task reward (pure HIRO); `ll_task_reward_coef` blend knob, default 0. |
| **HL** | `hl_reward_mode=task` (default): the **A0 task reward** summed over the window. `hl_reward_mode=tracking`: velocity command-tracking only (`mdp.track_linear_velocity + track_angular_velocity`, A0 exp terms reused, std read live via `reward_manager.get_term_cfg`), LL-execution penalties removed. The **env reward is unchanged** either way (no RQ2 confound) — only what the HL optimizes internally changes. |
- **`fell_over=time_out` in the A1 env ONLY** (`config/h1_2_a1/env_cfgs.py`): the LL's
  always-negative goal-distance reward makes early termination an attractor (die fast → stop
  accumulating negative reward). Marking the fall a **truncation** makes PPO/TD3 bootstrap it
  (`γ·V`) instead of cutting value to 0. A0 keeps a true terminal (A0 diverges with
  time_out). The buffer `done` / GAE mask excludes `time_outs` to express the same fix.

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
  --diagnose-goals 600 --eval-seeds 2` → per-window `(command−achieved) = (command−V*) [HL
  goal err] + (V*−achieved) [LL reach err]`, raw `|g|` + saturation frac, fwd/bwd vx split,
  realized/requested ratio + `[GOALDIAG] {json}`. Decomposition closes exactly.
- **Play structure-restore:** play.py restores structure keys (`c, goal_components,
  goal_weights, hl_algorithm, hl_ppo, hl_td3, relabeling, gamma_hi, hl_target_mode`) from the
  run's own `params/agent.yaml` (yaml.full_load; deep-merge dict cfgs over defaults since the
  dump is written after construction and pops `distribution_cfg["class_name"]`). Re-derives
  the goal obs dim from restored components. Prints `[INFO]: Restored run structure …`.
  Without this, A1 replays silently ran an oracle HL / goal frozen at 0 (all pre-2026-06-11
  qualitative replays are void; W&B training metrics were always ground truth).

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
