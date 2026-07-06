# HRL implementation plan — master index

Thesis: 5 HRL locomotion architectures (A0–A4) for the Unitree H1-2, compared for
sim-to-real transfer. Built on `mjlab` + `rsl_rl` + MuJoCo-Warp.

This is the **navigation hub + status dashboard**. The architecture matrix, RQ2
comparison-cleanliness rule, and reproducibility rules live in
`.claude/docs/experiment-design.md` (not duplicated here).

## Status dashboard

| Arch | Task ID | Status | Headline |
|---|---|---|---|
| **A0** | `Unitree-H1_2-Flat` | ✅ done | Flat PPO baseline. Benchmark ref err_vx/vy/yaw 0.09/0.11/0.10. |
| **A1** | `Unitree-H1_2-Flat-A1` | ✅ **tracking wall SOLVED** (2026-06-16) | HIRO: PPO LL + TD3 HL + relabel. `absolute` target + `tracking` HL reward → **A0-level** err 0.098/0.091/0.167, 0 falls. Sim deploy done (two-ONNX C++ `State_RLHRL`); state-noise DR **validated** (#8b `GoalStateNoise`; abs_bias & abs_full both reliable, 2/2 clean, clean-baseline quality). **A1a HL-enrichment** (commanded stride period → cost-of-transport): lever **validated 2026-07-02**; **S1c machinery ✅ 2026-07-02** (HL period action dim + CoT window reward, tested 18/18); d(T) slow-band enabler **parked** (v5/v6/v7 negative, 2026-07-03; v8 repro control ✅ code exonerated); keeper LL = v2 `model_5000`, envelope (0.35, 1.0). **S3 ✅-qualified** (2026-07-03, `a1a_s3_cadence_hl`): frozen-LL learned cadence HL drives CoT below the fixed-0.6 clock at every speed (−8…−22%, 0 falls, gait_match 0.94, tracking held) → **mechanism GO**, but at `hl_cot_coef=1.0` it settled on a near-natural flat ~0.65 stride (not the map optimum); coef = S5's lever (deferred to the full arch). **S4 next** (co-trained vs A0/A0+energy/A1) → `doc/hrl/A1a_plan.md`. |
| **A2** | `Unitree-H1_2-Flat-A2` | ⬜ next — **spec ready** (`doc/hrl/A2_ARMA.md`) | HIRO + A-RMA (privileged latent z + 1D-CNN adaptation + Phase-3 fine-tune). |
| **A3** | `Unitree-H1_2-Flat-A3` | ⬜ planned | NaviGait: offline gait library + RL residual. |
| **A4** | — | ⏸ likely skipped | Trajectory + MPC. Lowest priority. |

## Where things live

- **Shared HRL machinery** (co-train loop, goal space, warm-start, reward decomp,
  benchmark/probe tools, checkpoint/ONNX, gotchas) → **`.claude/docs/hrl-infra.md`**.
  A2/A3 reuse this; read it before starting a new architecture.
- **A2 design spec** (HIRO + A-RMA: e_t, 3 phases, S0–S5, force-perturbation extension) → `doc/hrl/A2_ARMA.md`.
- **A1 design (as-built) + current status + open ablations** → `doc/hrl/A1_HIRO.md`.
- **A1 findings ledger** (what was tried / ruled out / lessons, M1→M5 + probes) →
  `doc/hrl/A1_findings.md`.
- **A1 goal-achievability probe** (the structural-limiter diagnosis) →
  `doc/hrl/A1_goal_achievability_probe.md`.
- **Hierarchy-benefit roadmap** (ideas backlog answering the supervisors' "what does the
  hierarchy buy?" critique — richer HL job, smooth/safe reward terms, real-robot feasibility,
  LL reward restructure; parallel tracks) → `doc/hrl/hierarchy_benefit_roadmap.md`.
- **A1a HL-enrichment plan** (the chosen Track-A answer: HL minimizes cost-of-transport via a
  commanded gait cadence the flat command can't express; staged S0–S6) →
  `doc/hrl/A1a_plan.md` (design rationale: `docs/adr/0004-a1a-energy-cadence-hl-enrichment.md`).
- Architecture matrix, RQ2 rule, reproducibility → `.claude/docs/experiment-design.md`.
- File paths / class map / obs dims → `.claude/docs/codebase-map.md`.
- Cluster (SLURM) → `.claude/docs/cluster.md`. C++ deploy/ONNX → `.claude/docs/deployment.md`.

## Building a new architecture (A2/A3)

Each registers as a new task ID `Unitree-H1_2-Flat-A<N>` with its own
`config/h1_2_a<N>/{__init__,env_cfgs,rl_cfg}.py` under `src/tasks/velocity/`. Reuse
`hrl-infra.md` machinery; deviate from A0's env only via explicit, called-out structural
changes (RQ2 rule). Each architecture gets a `doc/hrl/A<N>_*.md` design+status doc that links
back to `hrl-infra.md` rather than re-deriving the shared parts.

## Process (no vibe coding)
Per CLAUDE.md: analysis → feature spec → approval → implement → test, with explicit
go-ahead between stages. Judge policies by the deterministic benchmark, not training err.
