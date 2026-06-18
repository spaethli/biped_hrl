# A1 — HIRO (hybrid PPO + TD3) — design & status

> **Status (2026-06-16): tracking wall SOLVED.** `absolute`-target HL + `tracking`-weighted
> HL reward reaches deterministic benchmark **err_vx 0.098 / err_vy 0.091 / err_yaw 0.167,
> 0 falls — A0-level** on vx/vy (A0 0.09/0.11/0.10), 4.3× better than the prior best A1.
> Remaining = polish (yaw, smoothness) + writeup ablations. See "Current results" + the
> chronological journey in `doc/A1_findings.md`.

Shared machinery (co-train loop, goal space, warm-start, reward decomp, TD3 internals,
benchmark/probe tools, checkpoint/ONNX, gotchas) is in **`.claude/docs/hrl-infra.md`** —
this doc only covers what is A1-specific.

## What A1 is
Architecture A1 of the thesis: the **structural hierarchy**. A hybrid HIRO design —
**on-policy PPO low level** (= the A0 learner, so RQ2 isolates the hierarchy) + **off-policy
TD3 high level** with HIRO goal-relabeling. The HL fires every `c` steps and hands the LL a
velocity(+pose) goal; the LL never sees the command. Relabeling is a runtime toggle (clean
ablation). Stays in `unitree_rl_mjlab` + `mjlab` + `rsl_rl`; no Isaac Lab.

## The two levers that solved the wall
Each fixes one probe-isolated failure (see `doc/A1_findings.md` for the diagnosis chain):

1. **`hl_target_mode=absolute`** (`HrlRunnerCfg`, default `delta`; CLI
   `--agent.hl-target-mode absolute`). The HL emits a state-independent
   `V* = center + scale·g` (static command→goal map) instead of the relative
   `V* = s_t + scale·g` (which needs a state-dependent law the HL never learned →
   `|g|→1` saturation collapse). Fixes **LL reachability/saturation**: `|g|` 0.84→0.16,
   LL reach err 0.61→0.11. The map lives in `GoalSpace.center/to_target/to_g` (one place);
   relabel uses `ref = center` (absolute) | `s_t` (delta). LL unchanged (still sees delta
   `V*−s_i`, same reward).
2. **`hl_reward_mode=tracking`** (`HrlRunnerCfg`, default `task`; CLI
   `--agent.hl-reward-mode tracking`). The HL accumulates **velocity command-tracking only**
   (`mdp.track_linear_velocity + track_angular_velocity`) with LL-execution penalties
   removed (the LL's concern). Under the default `task` reward the HL objective is
   penalty-dominated → value-max policy is `g≈0` ("ask for neutral velocity", refuses
   forward). Fixes the **HL command→goal collapse**: HL goal err vx 0.71→0.11,
   forward-avoidance gone. The **env reward is unchanged** (no RQ2 confound); only the
   HL-internal objective changes. Safe with `absolute` (V* bounded to command range).

Both are needed (clean A/B: delta-task → absolute-task → absolute-tracking).

## HL learners (`hl_algorithm`)
The switch is the **HL learner**, not just a flag — relabeling needs a replay buffer, so it
is structural. `--agent.hl-algorithm {oracle,ppo,td3}`.

- **`oracle`** (M2) — emits the absolute command as `V*` (no learning). Validation harness:
  proved the warm-started LL is a competent goal-conditioned tracker (0.05 m/s, no
  directional bias). Use as the LL upper-bound control.
- **`ppo`** (M3) — a 2nd `rsl_rl.PPO` at the HL timescale (`HighLevelPpo`). **Naive on-policy
  HL fails to track** (credit-assignment thinness: ~3 HL transitions/env/iter; tracking term
  swamped by penalties). HL actor obs `policy ++ command`; critic obs `critic` (already
  contains command — do NOT append). Linear `scale` map, NOT tanh (rsl_rl's
  `GaussianDistribution` applies no Jacobian correction → a tanh squash would be a silent
  bug). Std cap `(1e-3,1.0)` available (fixes blowup). Baseline only; kept for the writeup.
- **`td3`** (M4/M5, **the designed HL**) — self-contained off-policy TD3 (`HighLevelTd3` +
  `HLReplayBuffer`). Internals/tuning in `hrl-infra.md`. `relabeling {none,hiro}` sub-toggle
  (HIRO off-policy correction, inline). `relabel=none` is the clean control that attributes
  any further gain to relabeling alone.

### A1 run matrix (writeup)
| Run | HL learner | Relabel | Answers |
|---|---|---|---|
| A0 (`Unitree-H1_2-Flat`) | — | — | baseline |
| A1 `hl=ppo` | PPO | n/a | does structural hierarchy help? (clean: both PPO) — naive HL fails |
| A1 `hl=td3 relabel=none` | TD3 | ✗ | isolates off-policy HL (clean relabel control) |
| A1 `hl=td3 relabel=hiro` | TD3 | ✓ | full HIRO; vs row above isolates the relabeling correction |
`hl=ppo`→`hl=td3 relabel=hiro` changes two things (PPO→TD3 **and** relabeling); the
`relabel=none` row is the control.

## Goal space (A1)
Default **7-dim `(velocity, orientation, height)`** — confirmed decision (orient+height are
stabilizing regularizers: better tracking esp. yaw, lower std; velocity-only 3-dim is the
documented clean negative result). `goal_components` in `config/h1_2_a1/rl_cfg.py` is the
single source of truth; `goal_dim` derived. Goal weights: velocity weight 3.
See `hrl-infra.md` for the `GoalSpace` map and `scale`/`center` mechanics.

## Config & launch
- Defaults (`config/h1_2_a1/rl_cfg.py`, `HrlRunnerCfg`): `hl_algorithm=oracle`(*),
  `c=8`, `gamma_hi` derived `0.99**c`, goal 7-dim, `entropy_coef=0.005`,
  `hl_target_mode=delta`(*), `hl_reward_mode=task`(*).
  (*) the SOLVED config overrides these — see below.
- **Solved-config launch** (the A0-level result):
  ```bash
  conda activate unitree_mjlab_h1_2_rl
  python scripts/train.py Unitree-H1_2-Flat-A1 --env.scene.num-envs 4096 \
    --agent.max-iterations 5001 --agent.hl-algorithm td3 --agent.relabeling hiro \
    --agent.hl-target-mode absolute --agent.hl-reward-mode tracking \
    --agent.warm-start-path logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt \
    --agent.run-name <name>
  ```
- TD3 knobs override as `--agent.hl-td3.<field>`. Launch via the env + `python` directly,
  not `conda run`. `max-iterations` is N+1 (final ckpt named after the last index).

## Current results (deterministic benchmark, 64 envs × 600 steps × 2 seeds; c=8, warm-start)
| run | err_vx | err_vy | err_yaw | act_rate | \|g\|vx (sat) | HL err vx | LL reach vx | fwd/bwd end vx |
|---|---|---|---|---|---|---|---|---|
| delta baseline (relabel) | 0.42 | 0.35 | 0.64 | 1.41 | 0.84 (58%) | 0.51 | 0.61 | — |
| absolute, task reward | 0.65 | 0.45 | 0.28 | 2.40 | 0.16 (2%) | 0.71 | 0.11 | 0.92 / 0.19 |
| **absolute + tracking** | **0.098** | **0.091** | **0.167** | 1.68 | 0.16 (0.6%) | **0.11** | 0.10 | **0.096 / 0.083** |
| A0 reference | 0.09 | 0.11 | 0.10 | 0.66 | — | — | — | — |

Run `a1_td3_absolute_hltrack_5k` (td3+relabel, c=8, warm-start, absolute+tracking, range
(-0.5,1.0), `model_5000`). The absolute→tracking A/B (only `hl_reward_mode` differs) is the
decisive step: HL goal err vx 0.71→0.11, forward-avoidance gone, `|g|` stays de-saturated,
LL reach ~0.10. **realized/requested ratio is now low only because V*≈command and the robot
is already near it** — that metric is informative only when failing.

## Remaining gaps & open work
- **Polish (not the wall):** yaw 0.167 (largest residual = HL yaw goal err 0.171; still 4×
  better than prior A1); smoothness act_rate 1.68 vs A0 0.66; posture (orient_dev 0.13,
  height_dev 0.11 vs A0 ~0.03) — LL-execution quality, not tracking.
- **Open ablation (running `a1_td3_delta_hltrack_5k`):** is `absolute` *necessary*, or does
  the `tracking` HL reward alone (delta mode) suffice? A/B on `hl_target_mode`, tracking
  reward held fixed.
- **Reserve variant (not implemented):** LL observes the **absolute** target `V*` instead of
  the delta `V*−s_i` (matches A0's absolute-command training format). Low priority — the
  oracle control already proves the delta LL representation is fully learnable (0.05 m/s).
  Touch points: the goal-obs write in `hrl_runner` learn()/`get_inference_policy`, the probe,
  a matching `ll_goal_obs` flag.
- Then: full omnidirectional cluster run, optional `c`/goal-range/HL-LR sweep, ONNX
  `policy_hl.onnx` + C++ two-model chaining (deployment milestone).

## A1 file map (`src/tasks/velocity/`)
```
config/h1_2_a1/
  __init__.py   register Unitree-H1_2-Flat-A1; runner cfg = goal_components source of truth
  env_cfgs.py   reuse flat env; _restructure_obs_groups (actor→policy/command/goal, drop
                actor); fell_over=time_out (A1 only)
  rl_cfg.py     HrlRunnerCfg: LL=PPO + c/goal_components/goal_weights/hl_algorithm/relabeling/
                gamma_hi/warm_start_path/freeze_ll_path/hl_target_mode/hl_reward_mode/hl_ppo/hl_td3
rl/hrl/
  hrl_runner.py  HierarchicalRunner(VelocityOnPolicyRunner): co-train loop, warm-start
                 (_partial_load gap-aware), freeze-LL, save/load+hl, onnx, get_inference_policy
  goal_space.py  GoalComponent/GoalSpace + registry; center/to_target/to_g/scale; goal_dim derived
  high_level.py  HighLevel ABC + OracleHighLevel + HighLevelPpo
  td3.py         HighLevelTd3 (+ inline HIRO relabeling)
  storage.py     HLReplayBuffer + HLBatch
```
No `GoalConditionedWrapper`, no `relabeling.py` — the goal is a plain obs term and relabeling
is inline in `td3.py`.

## A1-HAC — possible parallel variant (NOT committed)
A second HRL algorithm for the A1 slot (HIRO vs HAC, same env/reward/DR). **Does not swap
into the current hybrid:** HAC needs hindsight at *both* levels (its LL is HER-based,
off-policy, sparse reward) → would migrate the LL off PPO → breaks the "LL = A0 PPO" control
A1 relies on. If pursued, treat as **A1-HAC, a parallel variant** (off-policy both levels,
TD3/DDPG; port from `Hierarchical-Actor-Critic-HAC-PyTorch`), buying an A1-HIRO vs A1-HAC
comparison ("off-policy correction" vs "hindsight + subgoal testing"). Cost: a second
off-policy LL path, more tuning, no shared PPO LL. **Recommendation:** keep HIRO committed;
revisit A1-HAC only if time remains after A1-HIRO and A2.
