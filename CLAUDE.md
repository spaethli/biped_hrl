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
- IMU velocity-increment bench (deploy feasibility): `python scripts/play.py <TaskID>
  --checkpoint-file <pt> --num-envs 64 --check-vel-increment 480 --eval-seeds 2` — prints
  `[VELINC] {json}`. `delta`-mode goal obs cancels absolute velocity, so deploy needs only the
  within-window increment → `.claude/docs/hrl-infra.md`.
- Commanded joint-limit check (deploy parity): `python scripts/play.py <TaskID>
  --checkpoint-file <pt> --num-envs 64 --eval-steps 600 --check-joint-limits True` — scores the
  policy's COMMANDED targets against `h1_2_limits.h`, in the same shape as `safety_analyzer.py`'s
  `raw_policy_violations`, so sim and hardware compare directly. `[LIMITS] {json}`.
- A1 goal probe (HL-vs-LL error decomposition): `python scripts/play.py <TaskID>
  --checkpoint-file <pt> --diagnose-goals 600 --eval-seeds 2 --num-envs 64` (defaults to
  1 env without the flag; pre-2026-07-15 probes on cadence ckpts ran frozen-phase — see
  `hrl-infra.md`) — per-window HL goal error
  vs LL reach error + `|g|` saturation + fwd/bwd vx split + `[GOALDIAG] {json}`. Verdict
  chain: LL competent; the original `|g|→1` saturation was a **pre-tracking-reward**
  failure, **cured by `hl_reward_mode=tracking`** (the primary lever; `absolute` only
  refines, NOT required). **Correction 2026-06-24: do NOT cite saturation against `delta`** —
  true-lean delta+tracking runs are unsaturated. Authoritative: the A1 findings ledger (research KB)
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
- **`/sync-docs`** — route session results/decisions/conventions to their canonical
  homes. **Three destinations since 2026-07-28:** current design/status → this repo
  (`doc/`, `.claude/docs/`, `docs/adr/`, CLAUDE.md), kept lean because it gets
  published; exploratory narrative (what was tried, what failed, why, dated session
  logs, ideas backlogs) → an **append-only journal** at
  `~/biped_hrl_wiki/raw/engineering-journal/`, verbatim and never condensed; plus
  auto-memory. It writes a *source*, not wiki pages — a separate Ingest in the vault
  compiles the journal into queryable pages. Redaction gate applies to this repo only.
- **`/delegate`** — turn a task into a hand-off prompt for a fresh chat/subagent
  (model recommendation included). Worklines registry + who-owns-what:
  `doc/hrl/worklines.md`. Planning chats delegate implementation; don't grind it inline.

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

**Regression suite (2026-07-29):** `pytest` (CPU-only, no MuJoCo/GPU, ~3 s) guards the
goal decode, reward-term direction/gating, the warm-start column map, deploy/training
config parity and (2026-08-04) the readiness-pipeline seams. Run it before and after any
change to those. New tests must be proven able to fail:
`python scripts/check_test_sensitivity.py` re-introduces each historical defect and checks
it is caught (96 tests, 37/37 mutations). A test that restates the logic it guards cannot
fail when that logic breaks — the harness catches that too. Details → `hrl-infra.md`.

**Deploy readiness (2026-08-04): one command, one verdict.**
```bash
python scripts/deploy_readiness.py <TaskID> --checkpoint-file <pt> --tag <name>
```
`provenance → sim → bridge (candidate + A0 control) → analyze`, ~15-25 min, exit
**0 GO / 3 CAVEAT / 1 NO-GO / 2 INFRA**. Pass `--scene` (it SETS `/opt`'s plant and gates on
the read-back) and note `--bridge-cfg` defaults to **`deploy_est.yaml`**, the deployable
estimator — what the robot runs; `deploy.yaml` is the privileged ground-truth arm, kept for
attribution. Fail-closed: never GO from a stage it could not
score. Use it instead of hand-driven bridge sessions. **Always check what is deployed
first** — `scripts/deploy_provenance.py --check --checkpoint-file <pt>` md5-identifies the
deployed ONNX against every run export and names the source run (`run_path` metadata is
always `local`). **Step-by-step operating manual, each step mapped to its manual W/G gate
and the command to run it by hand → `doc/hrl/A1a_deploy_plan.md` "HOW TO RUN THE GATES".**
⚠ `bridge_replica.py` is **retired as a gate** (no C++, no DDS, doesn't model the safety
filter or `hold_joint_ids`); opt-in via `--stages ...,replica` for plant/latency A/B only.

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
- **The A1 goal SCALE is baked into the checkpoint (2026-07-16, F2) — never re-derive it
  at inference.** It defines what the HL's `g` means (`V* = ref + scale*g`), so it is a
  policy property, not an env property. Training derives it live (it tracks the twist
  curriculum); `save()` bakes `goal_scale`, `load()` pins it, `learn()` un-pins.
  Pre-2026-07-16 checkpoints have no baked value → `play.py`'s absence shim pins the
  trained scale and prints `[SHIM] ...`. **Why this is load-bearing:** the old live
  derivation read the live twist ranges, and `--eval-cmd-vx` collapses those to a point →
  scale hit its `1e-3` floor → `V* ≈ s` → the goal channel went inert and every A1 hold
  eval reported a phantom "HL hold degeneracy" (A0 was immune: no goal space). Deploy C++
  reads the baked `goal_scale` from the HL ONNX metadata (`State_RLHRL.cpp`) and only
  derives from `deploy.yaml` ranges when that metadata is absent (pre-2026-07-16 exports).
  **The deploy command ranges are an operator safety clamp and deliberately DON'T match
  training** (`ang_vel_z` ±0.5 deployed vs ±1.0 trained: train wide for faster turning,
  deploy narrow for a tame joystick) — which is precisely why the scale must travel with
  the policy. `tests/test_deploy_parity.py` asserts the metadata is present so the legacy
  derive-from-ranges path stays unreachable. See the A1 findings ledger (research KB) WL-C.
- **`hl_velocity_goals_only=True` is the A1a default (2026-07-09):** the TD3 HL emits
  only the velocity goal columns (+period); orientation/height targets are pinned to
  nominal (oracle path). Fixes the posture sag (tracking-rewarded HL had no reason to
  command upright). HL action = `task_dim(+1)`, not `goal_dim(+1)`; pre-change checkpoints
  restore as `False` (play.py absence-shim). See the A1 findings ledger (research KB).
- **The HL's velocity error must be ENDOGENOUS (WL-F, 2026-08-09).** `hl_vel_source=leg_odom`
  feeds `obs["hl_vel"]` from a simulated leg-odometry estimator (`rl/hrl/leg_odom.py`, a
  transcription of the deployed `hrl::leg_odom_velocity`, parity 6.3e-7 m/s) instead of
  ground-truth + `HlVelJitter`. Leg odometry is computed *from the legs*, so its error is a
  function of the policy's own motion — measured `corr(leg action rate, |v_est−v_true|)`
  **+0.213 (29/31 iters)** vs **+0.0006 (0/31)** for jitter, at the *same* error magnitude.
  Jitter was the right size and the wrong shape. Two rules: **difference at the physics rate,
  never the control rate** (50 Hz is a measured 2.4x worse — hence the `per_substep` metrics
  term), and the estimator feeds `obs["hl_vel"]` ONLY, never `state_n`. Default is `state`, so
  historical comparators stay reproducible; `hl_vel_jitter` (exogenous arm) and
  `hl_vel_residual` (magnitude top-up) are mutually exclusive and the runner enforces it.
  Details → `hrl-infra.md`.
  ⛔ **SHELVED 2026-08-10 — the mechanism is proven, the training benefit is NOT. Do not adopt
  `hl_vel_source=leg_odom` as a default.** Both retrained arms failed the policy bar (`g_legs`
  0.9206 residual / 0.8779 bare vs the 0.8363 jitter keeper; `err_vx` 0.1295 / 0.1466 vs 0.1219)
  and the bare arm is **NO-GO on `deploy_readiness.py`** (bridge fall at cmd 0, A0 control arm
  clean ⇒ the policy, not the rig). **Root cause, and the rule that generalizes: the HL consumes
  a `c=8`-averaged estimate, and that averaging cancels the policy coupling.** Measured on the
  keeper across cmd 0→1.0: raw per-step error spans **7.62x** (0.0703→0.5359) while the
  HL-consumed error is nearly **flat** (0.0506→0.0782, 1.55x). So standing is an *equal*-error
  state, not a low-error one, there is no estimate-quality gradient to learn from, and a
  correctly-shaped, correctly-sized error model still exerts almost none. **Before investing in
  a better HL velocity estimator (e.g. a Kalman filter), check what the consumer does to it —
  score the `c=8`-averaged physics-rate rung, never the per-step RMS; the two disagree in
  direction.** Verdicts → the research KB, `data/2026-08-09-wl-f-endogenous-estimator/`
  (`task5b_bare_verdict.md`, `task6_nearzero_probe.md`).
- `gamma_hi` is **derived from `c`** (`0.99**c`, in `HrlRunnerCfg.__post_init__`,
  unconditional) — horizon-matched, NOT independently settable. Don't re-hardcode it.
- Same-config runs diverge a lot (GPU non-determinism + RL chaos), and **how much depends
  on the REGIME**: measured on two config-identical same-seed pairs (2026-08-27), walking
  metrics reproduce to ~10% but standing ajit/touchdowns span **4.2x** and held tracking
  **3.3x** — one replicate never reached the commanded speed at all. Score a standing arm
  against a replicate BAND, never a single control; a <2x standing effect from one run is
  not evidence. Treat `num_envs` as a hyperparameter: hold it fixed within a comparison
  set; use ≥2 seeds. Gait lift-off iteration scales with num_envs — never judge
  stuck-vs-slow before ~2x the expected lift-off (see `docs/adr/0005` amendment).
- **`Unitree-H1_2-Rough` (2026-08-26): the actor is BLIND, and sim sizing is the binding
  constraint on a 12 GB card.** The terrain `height_scan` is **critic-only** (asymmetric
  actor-critic). An actor that reads it exports at **279** dims and is undeployable: the
  H1-2 has no such sensor, the deploy vector is 92, and the C++ runner does not validate
  the length, so the robot goes limp via the safety hold rather than erroring
  (`.claude/docs/deployment.md`, "Obs-dim contract"). Blind actor = flat's 92 dims = drop-in
  export. **Sizing:** EPA collision scratch is `num_envs × nconmax × (376 + 132·ccd_iterations)`
  bytes, so stock rough (`ccd_iterations=500`, `nconmax=48`) wants **13 GB** at 4096 envs.
  Train with `--env.sim.mujoco.ccd-iterations 200` (open-loop probe: divergence vs ccd 500
  equals the GPU non-determinism floor) or drop to 2048 envs. Play sets `nconmax=512`
  because play draws random tiles and the **initial-pose** contact count reaches ~195;
  steady-state demand is only 38, so training's 48 is correct and was never dropping contacts.
- **Model v2 = option B** (2026-07-09, `docs/adr/0005`): torso 300/3 + arm hold gains,
  derived scales, frictionloss 0, **desired_kl=0.01** (required). v1 checkpoints invalid.
  Replays need the constants the checkpoint trained with (env-side scales).
- **Model v3 (2026-08-26, `docs/adr/0008`, supersedes ADR-0006) = action clip + leg mass; logs
  to `*_v3`.** Commanded joint targets are clipped to the hard limits in training AND in all 8
  deploy yamls (the `State_RLBase` safety clamp is NOT the same operation: it acts after the
  hold blend), plus an L1 penalty on the clipped excess (`action_clip` -3.5 / A1
  `ll_action_clip_coef` **+3.5** — the mirror is a positive MAGNITUDE, since the runner computes
  `r_lo - coef*excess`). Leg mass corrected x1.20787 as fidelity only, NOT as a lean fix (V1's
  falsification stands). ⚠ `joint_pos_limits` (-10.0) is now known **inert** (< 5e-7/step on a
  trained policy): it scores MEASURED position, which MuJoCo's `qpos` clamp already prevents.
  v2 checkpoints are not comparable across the boundary.

## Where the deep context lives

- **Personal research + narrative KB → `~/biped_hrl_wiki`** (separate Obsidian
  vault, OKF v0.2 bundle, own `CLAUDE.md` — read it before writing there).
  Literature review (HRL/RMA/gait papers) AND, since 2026-07-28, the home of all
  exploratory narrative. That narrative lives as an **append-only source** in
  `raw/engineering-journal/` (A1 findings ledger, A1a experiment and deploy
  journals, hierarchy-benefit backlog, closed-workline verdicts, A0 model delta)
  and is **compiled into queryable `wiki/` pages by Ingest** — read the compiled
  pages, drill into the journal to verify a number. **Migration COMPLETE (2026-07-28).** `doc/hrl/`
  now holds current design and status only — the options in use and how the
  working solution is implemented — because this repo gets published and the KB
  doesn't. **Don't re-accumulate narrative here**; route it to the KB via
  `/sync-docs`. Files removed in the migration (`A1_findings.md`,
  `A1_goal_achievability_probe.md`, `hierarchy_benefit_roadmap.md`) live there
  now; don't re-create them.
- **HRL master index + A0–A4 status dashboard → `doc/hrl/HRL_plan.md`** (start here).
- **Shared HRL machinery** (co-train loop, goal space, warm-start, reward decomp,
  benchmark/probe tools, checkpoint/ONNX, gotchas) → `.claude/docs/hrl-infra.md`
  (A2/A3 reuse this — read before starting a new architecture).
- **H1-2 frames, IMU convention, contact frame, FK/Jacobians → `doc/hrl/h1_2_kinematics.md`**;
  base-velocity EKF design + the 6-arm offline comparison (WL-G) → `doc/hrl/h1_2_ekf_design.md`,
  tool `scripts/replay_base_estimators.py --selftest`. **The IMU sits on `torso_link`, not the
  pelvis** — leg FK is pelvis-referenced, so `p^I = Rz(-psi)·p^P − r_imu`.
- **A1 design (as-built) + current status + open ablations → `doc/hrl/A1_HIRO.md`**;
  A1 findings ledger (what was tried/ruled out, M1→M5 + probes) → the A1 findings ledger (research KB);
  goal-achievability probe → the goal-achievability probe writeup (research KB).
- Codebase layout, obs dims, A1 class map → `.claude/docs/codebase-map.md`
- Architecture matrix A0–A4, RQ2 comparison-cleanliness rule, reproducibility →
  `.claude/docs/experiment-design.md`
- HPC cluster (SLURM, NVRTC fix, versions) → `.claude/docs/cluster.md`
- C++ deploy, ONNX export, replay → `.claude/docs/deployment.md`
- A0 warm-start checkpoints (v1 env only): `logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt`; v2 replacement = `a0_v2_baseline` `model_10000.pt` once the run finishes.
- W&B: project `biped_hrl` (A1 experiment `h1_2_velocity_a1`, A0 `h1_2_velocity`).

## Agent skills

### Issue tracker

Issues tracked as GitHub issues (repo inferred from `git remote -v`) via the `gh` CLI; external PRs are NOT a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary (`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at repo root. See `docs/agents/domain.md`.
