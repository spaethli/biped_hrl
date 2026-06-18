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
- Use ≥2 seeds for any claim.
- Compare checkpoints with the deterministic closed-loop eval (hierarchy-aware
  inference policy), not training-time stochastic metrics.

## W&B

Project `biped_hrl`: A0 experiment `h1_2_velocity`, A1 `h1_2_velocity_a1`.
