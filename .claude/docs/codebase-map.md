# Codebase map

## Key files

- `scripts/train.py` — training entry point (tyro CLI, task registry).
- `scripts/play.py` — visualize / export checkpoints. Restores the run's structure
  (`goal_components`, `hl_algorithm`, ...) from its `params/agent.yaml` before
  building the runner. `--export-onnx` (bare flag) exports ONNX without the viewer.
- `src/tasks/velocity/velocity_env_cfg.py` — base velocity env factory.
- `src/tasks/velocity/config/h1_2/` — A0 config: `env_cfgs.py`, `rl_cfg.py`, `__init__.py`.
- `src/tasks/velocity/config/h1_2_a1/` — A1 config (same trio; see below).
- `src/tasks/velocity/rl/runner.py` — `VelocityOnPolicyRunner` (ONNX + W&B export on save).

## Task registration

Each config package's `__init__.py` calls
`register_mjlab_task(task_id, env_cfg, play_env_cfg, rl_cfg, runner_cls)`.
All tasks are pulled in via `import src.tasks` in train.py/play.py;
`src/tasks/__init__.py` must import each config package.

## Model / runner hierarchy (rsl_rl)

- `MLPModel` — MLP actor/critic; `obs_normalization`, `GaussianDistribution`.
  ⚠ Pops `distribution_cfg["class_name"]` in place at construction — cfg dicts
  dumped *after* runner construction (params/agent.yaml) are missing that key.
- `RNNModel(MLPModel)` — adds LSTM/GRU before the MLP head.
- `PPO` — actor + critic + `RolloutStorage`.
- `OnPolicyRunner` → `MjlabOnPolicyRunner` → `VelocityOnPolicyRunner` → `HierarchicalRunner` (A1).

## Actor observation (flat terrain, H1-2)

92 dims, ORDER: base_ang_vel(3) + projected_gravity(3) + **command(3 @ cols 6:9)** +
phase(2) + joint_pos(27) + joint_vel(27) + last_actions(27). 27 actions (joint pos
offsets). Critic = actor terms + base_lin_vel + foot terms (107).
**command is the 3rd term, NOT last** — this drives the gap-aware A1 warm-start copy.

## A1 / HRL (`src/tasks/velocity/rl/hrl/`)

- `hrl_runner.py` — `HierarchicalRunner(VelocityOnPolicyRunner)`: co-train loop
  (LL=PPO every step, HL fires every `c=8` steps), gap-aware A0 warm-start
  (`_warm_start_low_level` / `_partial_load`), save/load (+`hl` key), ONNX export,
  hierarchy-aware `get_inference_policy` (fires `hl.act_inference` every c-th call —
  without it, play would freeze the goal at zero).
- `high_level.py` — `HighLevel` ABC (`act / act_inference / begin_window / accumulate /
  end_window / update / state_dict / ...`) + `OracleHighLevel` (M2) + `HighLevelPpo` (M3).
- `td3.py` — `HighLevelTd3` (M4): deterministic tanh actor, twin critics, shared
  input normalizer, TD3 target tricks.
- `storage.py` — `HLReplayBuffer` / `HLBatch`: flat GPU ring buffer of HL transitions.
- `goal_space.py` — declarative `GoalComponent`/`GoalSpace` + registry;
  `goal_components` list → `goal_dim` always derived; `extract / reward /
  oracle_target / scale` (scale tracks the live twist curriculum).
- Obs restructure: `config/h1_2_a1/env_cfgs.py:_restructure_obs_groups` splits the A0
  `actor` group → `policy` (no command) / `command` / `goal`, deletes `actor`.
  Routing: LL actor=(policy, goal), LL critic=(critic, goal); HL sees (policy, command).
- `mdp.hrl_goal` obs term reads the `env.hrl_goal` buffer the runner writes
  (no wrapper class).
- A1 env sets `fell_over.time_out = True` (bootstrap); A0/base keeps a true terminal.

## ONNX export rule

Always go through `runner.export_policy_to_onnx()` (copies the obs normalizer via
deepcopy). Never reconstruct a model standalone from state-dict keys — it silently
loses normalizer stats.

## mjlab

Editable install from source at `../mjlab` (workspace sibling; PyPI mjlab 1.3.0 is
incompatible — broken warp API). `MjlabOnPolicyRunner` at `mjlab/rl/runner.py`;
`RslRlOnPolicyRunnerCfg` / `RslRlModelCfg` at `mjlab/rl/config.py`.
