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
  Also run `--eval-cmd-vx` holds (0.5, 1.0 — the treadmill case; report the `ss_err`
  steady-state + `t90` numbers, and check the `[HOLDDIAG]` per-env split — batch means
  hide bimodal backwards modes): the random-command aggregate hides sustained-command
  failures (2026-07-13/15, `A1a_plan.md` tables e/f).
- A1 goal probe (HL-vs-LL error decomposition): `python scripts/play.py <TaskID>
  --checkpoint-file <pt> --diagnose-goals 600 --eval-seeds 2 --num-envs 64` (defaults to
  1 env without the flag; pre-2026-07-15 probes on cadence ckpts ran frozen-phase — see
  `hrl-infra.md`) — per-window HL goal error
  vs LL reach error + `|g|` saturation + fwd/bwd vx split + `[GOALDIAG] {json}`. Verdict
  chain: LL competent; the original `|g|→1` saturation was a **pre-tracking-reward**
  failure, **cured by `hl_reward_mode=tracking`** (the primary lever; `absolute` only
  refines, NOT required). **Correction 2026-06-24: do NOT cite saturation against `delta`** —
  true-lean delta+tracking runs are unsaturated. Authoritative: `doc/hrl/A1_findings.md`
  (Track F + "fix chain").

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
  docs (the `doc/` HRL plan + `.claude/docs/`, CLAUDE.md) + auto-memory, with a
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

- **A1 default (2026-07-09): `ll_goal_kernel=exp` + true-terminal `fell_over`**, trains
  from scratch (no warm-start needed). Warm-start from a converged A0 is still load-bearing
  for the old `l2` kernel (which also needs `fell_over.time_out=True`). The `command` obs term
  sits **mid-vector (actor cols 6:9), not last**, so the warm-start copy is gap-aware
  (`HierarchicalRunner._partial_load`). Diagnostic it worked: **high iter-0 ep_len**.
- **`Unitree-H1_2-Flat-A1` config = the FULL final A1 structure (2026-07-10):** the
  `rl_cfg.py` field defaults now carry `hl_algorithm=td3`, `relabeling=hiro`,
  `hl_reward_mode=tracking`, `hl_obs_vel=True` (were oracle/none/task/False; the shell
  script used to force them). So the **bare task trains a learned TD3 HL, not oracle** —
  set `--agent.hl-algorithm oracle` for the clean LL-isolation baseline. Old checkpoints
  restore their own saved structure via play.py, so replays are unaffected.
- **Kernel pairing rule (never mix): exp ↔ true terminal (A1 default now), l2 ↔
  `time_out=True` truncation.** Set in `config/h1_2_a1/env_cfgs.py`. exp is all-positive so
  a true terminal is safe; l2's always-negative reward needs the truncation bootstrap to
  avoid a suicide attractor. A0 always uses a true terminal (diverges under `time_out`).
- A1 `entropy_coef=0.005` (0.01 lets action std blow up to ~2 and collapse).
- `goal_components` (in `config/h1_2_a1/rl_cfg.py`) is the **single source of truth**
  for the goal space; the env derives its goal obs dim from it. `goal_dim` is always
  derived, never hardcoded.
- **`hl_velocity_goals_only=True` is the A1a default (2026-07-09):** the TD3 HL emits
  only the velocity goal columns (+period); orientation/height targets are pinned to
  nominal (oracle path). Fixes the posture sag (tracking-rewarded HL had no reason to
  command upright). HL action = `task_dim(+1)`, not `goal_dim(+1)`; pre-change checkpoints
  restore as `False` (play.py absence-shim). See `A1_findings.md`.
- `gamma_hi` is **derived from `c`** (`0.99**c`, in `HrlRunnerCfg.__post_init__`,
  unconditional) — horizon-matched, NOT independently settable. Don't re-hardcode it.
- Same-config runs diverge a lot (GPU non-determinism + RL chaos). Treat `num_envs` as
  a hyperparameter: hold it fixed within a comparison set; use ≥2 seeds. Gait lift-off
  iteration scales with num_envs — never judge stuck-vs-slow before ~2x the expected
  lift-off (see `docs/adr/0005` amendment).
- **Model v2 = option B** (2026-07-09, `docs/adr/0005`): torso 300/3 + arm hold gains,
  derived scales, frictionloss 0, **desired_kl=0.01** (required). v1 checkpoints invalid;
  v2 logs to `*_v2`. Replays need the constants the checkpoint trained with (env-side scales).

## Where the deep context lives

- **HRL master index + A0–A4 status dashboard → `doc/hrl/HRL_plan.md`** (start here).
- **Shared HRL machinery** (co-train loop, goal space, warm-start, reward decomp,
  benchmark/probe tools, checkpoint/ONNX, gotchas) → `.claude/docs/hrl-infra.md`
  (A2/A3 reuse this — read before starting a new architecture).
- **A1 design (as-built) + current status + open ablations → `doc/hrl/A1_HIRO.md`**;
  A1 findings ledger (what was tried/ruled out, M1→M5 + probes) → `doc/hrl/A1_findings.md`;
  goal-achievability probe → `doc/hrl/A1_goal_achievability_probe.md`.
- Codebase layout, obs dims, A1 class map → `.claude/docs/codebase-map.md`
- Architecture matrix A0–A4, RQ2 comparison-cleanliness rule, reproducibility →
  `.claude/docs/experiment-design.md`
- HPC cluster (SLURM, NVRTC fix, versions) → `.claude/docs/cluster.md`
- C++ deploy, ONNX export, replay → `.claude/docs/deployment.md`
- A0 warm-start checkpoints (v1 env only): `logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt`; v2 replacement = `a0_v2_baseline` `model_10000.pt` once the run finishes.
- W&B: project `biped_hrl` (A1 experiment `h1_2_velocity_a1`, A0 `h1_2_velocity`).

## Agent skills

### Issue tracker

Issues tracked as GitHub issues in `liamlate/biped_hrl` via the `gh` CLI; external PRs are NOT a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary (`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.
