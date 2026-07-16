# Experiment design (thesis)

Goal: 5 HRL locomotion architectures for the Unitree H1-2, compared for
sim-to-real transfer.

## Architectures

- **A0** — Flat PPO: single MLP, obs → joint targets. DONE (`Unitree-H1_2-Flat`).
- **A1** — HIRO hierarchy: PPO low level + off-policy TD3 high level, HL fires every
  `c=8` steps, HIRO delta goals, optional off-policy goal relabeling.
  (`Unitree-H1_2-Flat-A1`; design `doc/hrl/A1_HIRO.md`, findings `doc/hrl/A1_findings.md`,
  shared machinery `.claude/docs/hrl-infra.md`.)
- **A2** — HIRO + A-RMA: HL unchanged; LL gets (1) privileged base policy with
  extrinsics latent z, (2) supervised adaptation encoder (1D-CNN over history),
  (3) A-RMA fine-tune with imperfect ẑ.
- **A3** — NaviGait-style: offline trajectory-optimization gait library + RL
  residual (`q̂ = q_lib(v̂) + Δq`), minimal reward.
- **A4** — Trajectory + MPC — lowest priority, likely skipped.

New architectures register as new task IDs (`Unitree-H1_2-Flat-A<N>`) with their own
`__init__.py`, `env_cfgs.py`, `rl_cfg.py` under `src/tasks/velocity/config/h1_2_a<N>/`.

## Evaluation

- Primary: straight-line treadmill. M1 = velocity-tracking error, M2 = success rate,
  M3 = sim2real performance drop.
- Training: omnidirectional (full velocity command distribution).

## THE comparison-cleanliness rule (RQ2 confound)

Architectures must share **the same env / reward / DR / obs content** so that
performance differences are attributable to *structure*. Any change that diverges an
Ax env from A0 — reward reweighting, termination changes, obs changes beyond the
structural split — is a confound and must be an explicit, conscious decision (called
out, not buried in tuning). Expected structural A1 deviations (fine): command removed
from the LL obs, intrinsic LL reward, goal obs group, A0 warm-start,
`fell_over=time_out`.

## Reproducibility rules

- Same-config runs diverge a lot (GPU non-determinism + RL chaos).
- `num_envs` is effectively a hyperparameter: hold it fixed within a comparison set.
  Gait lift-off iteration scales with it (8192 envs: ~330 it; 4096: ~550-1400) — never
  judge stuck-vs-slow before ~2x the expected lift-off (`docs/adr/0005` amendment).
- **A0 lift-off is ~2x later since `desired_kl` 0.01 -> 0.005** (`rl_cfg.py:39`, changed
  post-2026-05-26 "to prevent collapse"). rsl_rl's adaptive LR holds each update near
  `desired_kl`, so a smaller target = smaller steps = slower early learning (05-26 vs a
  today control at identical 4096 envs + identical env config: tracking @400 = 0.31 vs
  0.04). Verified as the *only* config delta between the eras (rewards/commands/curriculum
  byte-identical). **Adopted 0.01 as the A0 default (2026-07-09)**: verified stable to 10k + ~3x faster,
  and required by v2 option B (torso+arm hold stalls at 0.005). A1 keeps its own value
  until separately verified.
- Use ≥2 seeds for any claim.
- **Model v2 boundary (2026-07-08)**: v1-vs-v2 runs are not comparable (`docs/adr/0005`);
  v2 experiments log to `h1_2_velocity_v2` / `h1_2_velocity_a1_v2`. Replaying a
  checkpoint needs the constants it trained with (action scales are env-side).
- **rs20 boundary (2026-07-14)**: training `resampling_time_range` moved (3,8)→(3,20)
  (A0+A1 alike, held commands in-distribution); the play/bench command process stays
  pinned (3,8) so ALL aggregate-bench tables remain cross-generation comparable (keeper
  invariance verified 0.0797 vs 0.080). Training-side comparisons are within-generation
  only; new-generation A0 reference = `a0_v2_optB_rs20_baseline` `model_10000`.
- Compare checkpoints with the deterministic closed-loop eval (hierarchy-aware
  inference policy), not training-time stochastic metrics.

## W&B

Project `biped_hrl`: A0 experiment `h1_2_velocity`, A1 `h1_2_velocity_a1`.
