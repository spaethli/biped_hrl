# CLAUDE.md

Thesis project: 5 HRL locomotion architectures (A0–A4) for the Unitree H1-2, compared
for sim-to-real transfer. Built on `mjlab` + `rsl_rl` + MuJoCo-Warp (NOT Isaac Lab).

## Environment & running

- Conda env: **`unitree_mjlab_h1_2_rl`**
- **Launch trainings via the activated env + `python` directly — NOT `conda run`**
  (`conda run` buffers output and breaks live wandb streaming):
  ```bash
  conda activate unitree_mjlab_h1_2_rl
  python scripts/train.py <TaskID> --env.scene.num-envs 4096 --agent.max-iterations N \
      --agent.run-name <name> [--agent.warm-start-path <A0 model.pt>]
  ```
- Resume: `--agent.resume True --agent.load-run <dir> --agent.load-checkpoint model_<it>.pt`
  (max-iterations counts ADDITIONAL iters: it runs `start_it + max_iterations`).
- **max-iterations: always N+1** (e.g. `5001`, not `5000`) — the final checkpoint is
  named after the last iteration *index*, so `5000` ends at `model_4999.pt` while
  `5001` gives the round `model_5000.pt`.
- Play / export ONNX: `python scripts/play.py <TaskID> --checkpoint-file <pt> [--export-onnx | --num-envs 1]`
- Benchmark (deterministic in-sim policy comparison): `python scripts/play.py <TaskID>
  --checkpoint-file <pt> --num-envs 64 --eval-steps 600 --eval-seeds 2` — prints a
  multi-metric scorecard (err_vx/vy/yaw, fall_rate, action_rate, orient/height dev) +
  `[BENCH] {json}`. The canonical comparator (use `fall_rate`, not `ep_len`, for survival).

## Task IDs

- `Unitree-H1_2-Flat` — A0 (flat PPO baseline)
- `Unitree-H1_2-Flat-A1` — A1 (HIRO hierarchy; runner = `HierarchicalRunner`)

## Project skills (`.claude/skills/`)

- **`/rl-formulas`** — generate report/thesis-ready LaTeX formulas (+ `file:line`
  locations) for the RL components (goals, rewards, observations, PPO/GAE, the A1
  hierarchy). Reads the live code first so the math matches the current implementation.
- **`/wandb-rl-interpreter`** — interpret/diagnose RL training results from W&B (reward
  curves, episode length, losses, locomotion metrics, training health).
- **`/sync-docs`** — route session results/decisions/conventions into their canonical
  docs (plan doc §0, `.claude/docs/`, CLAUDE.md) + auto-memory, with a
  personal-identifier redaction gate (public repo).

## Code conventions

- If a fix is 1–2 lines and won't be reused, write it **inline** — no premature helper
  function/method. Factor only on real reuse (called from 2+ places).
- **Don't just write more and more code: reuse or extend existing functions** before
  adding new ones. Reuse keeps the codebase understandable and modular.
- Match the surrounding code's terseness/idiom. 2-space indent.

## Development workflow (no vibe coding)

For any non-trivial change (new feature, algorithm/hyperparameter change, fix beyond
a few lines), work in explicit stages and get explicit user go-ahead between them:

1. **Analysis** — what happened, with evidence (metrics, traces, file:line), before
   proposing anything.
2. **Feature definition** — the precise change(s), what is explicitly NOT changed,
   and the expected effect.
3. **Implementation** — step by step, matching the agreed spec.
4. **Testing** — a test plan defined up front (unit/smoke + success criteria), run
   after implementing; report results honestly.

Don't start editing code mid-diagnosis because a fix "seems obvious" — present the
analysis and the proposed change first.

## Load-bearing gotchas

- A1 LL is **warm-started from a converged A0** — this is load-bearing, not optional.
  The `command` obs term sits **mid-vector (actor cols 6:9), not last**, so the
  warm-start copy is gap-aware (`HierarchicalRunner._partial_load`). Diagnostic it
  worked: **high iter-0 ep_len**.
- `fell_over` is a **true terminal for A0** but a **truncation (`time_out=True`) for A1
  only** (set in `config/h1_2_a1/env_cfgs.py`). A0 diverges if given `time_out`; A1's
  negative goal-distance reward needs the bootstrap to avoid a suicide attractor.
- A1 `entropy_coef=0.005` (0.01 lets action std blow up to ~2 and collapse).
- `goal_components` (in `config/h1_2_a1/rl_cfg.py`) is the **single source of truth**
  for the goal space; the env derives its goal obs dim from it. `goal_dim` is always
  derived, never hardcoded.
- `gamma_hi` is **derived from `c`** (`0.99**c`, in `HrlRunnerCfg.__post_init__`,
  unconditional) — horizon-matched, NOT independently settable. Don't re-hardcode it.
- Same-config runs diverge a lot (GPU non-determinism + RL chaos). Treat `num_envs` as
  a hyperparameter: hold it fixed within a comparison set; use ≥2 seeds.

## Where the deep context lives

- **A1 architecture + current status + next steps → `doc/A1_HIRO_implementation_plan.md` §0**
  (read §0 first; it's authoritative over the older sections).
- Codebase layout, obs dims, A1 class map → `.claude/docs/codebase-map.md`
- Architecture matrix A0–A4, RQ2 comparison-cleanliness rule, reproducibility →
  `.claude/docs/experiment-design.md`
- HPC cluster (SLURM, NVRTC fix, versions) → `.claude/docs/cluster.md`
- C++ deploy, ONNX export, replay → `.claude/docs/deployment.md`
- A0 warm-start checkpoints: `logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt`.
- W&B: project `biped_hrl` (A1 experiment `h1_2_velocity_a1`, A0 `h1_2_velocity`).
