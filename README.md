# biped_hrl — Hierarchical RL for Unitree H1-2 Locomotion

Master's thesis project: five hierarchical reinforcement-learning architectures
(**A0–A4**) for velocity-tracking bipedal locomotion on the Unitree H1-2, compared for
training stability, tracking accuracy, energy efficiency, and sim-to-real transfer.

This repo is forked from
[unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab),
which provides the [mjlab](https://github.com/mujocolab/mjlab.git) training scaffold
(Isaac-Lab-style API on MuJoCo/MuJoCo-Warp physics), PPO via
[rsl_rl](https://github.com/leggedrobotics/rsl_rl.git), and a C++ sim2real deploy
pipeline for several Unitree robots. Everything below the flat PPO baseline — the
hierarchical architectures, the H1-2 actuator/gain retuning that got a policy onto real
hardware, the goal-conditioned HRL runner, and the extended deploy/telemetry stack — is
this project's own work, built on top of that scaffold.

<div align="center">

| MuJoCo | Physical H1-2 |
|---|---|
| <img src="doc/gif/h1_2-velocity.gif" width="320"/> | <img src="doc/gif/h1_2-velocity-real.gif" width="320"/> |

</div>

## What this fork adds

- **HRL architectures A0–A4** (`Unitree-H1_2-Flat*` task family): a flat PPO baseline
  (A0) plus a HIRO-style two-timescale hierarchy (A1: PPO low-level + TD3 high-level +
  hindsight relabeling) that a goal-conditioned low-level policy is trained against.
  A2 (RMA-style extrinsics adaptation on top of A1) is specced; A3/A4 are planned. See
  the status dashboard in [`doc/hrl/HRL_plan.md`](doc/hrl/HRL_plan.md).
- **A retuned, hardware-validated H1-2 actuator model.** The stock H1-2 config in
  upstream never made it onto real hardware from this project's testing; this fork's PD
  gain restructuring, derived action scaling, and added passive joint damping did.
  Every deviation from upstream is catalogued and ranked by causal weight in
  [`doc/hrl/A0_model_delta_vs_upstream.md`](doc/hrl/A0_model_delta_vs_upstream.md).
- **A goal-conditioned HRL training stack** (`src/tasks/velocity/rl/hrl/`): a TD3
  high-level policy (`high_level.py`, `td3.py`) issuing velocity/gait goals to a PPO
  low-level policy, off-policy replay + HIRO goal relabeling (`storage.py`), a
  configurable goal space (`goal_space.py`) with an inference-time-frozen goal scale
  baked into the checkpoint, and estimator-noise domain randomization
  (`state_noise.py`) for the sim-to-real gap.
- **An extended C++ deploy pipeline for the H1-2** (`deploy/robots/h1_2/`): a second
  runtime state, `State_RLHRL`, alongside upstream's flat-policy `State_RLBase`, running
  the two-ONNX (high-level + low-level) hierarchy on hardware at its own cadence; a
  runtime joint-limit safety filter; and a real-hardware DDS bridge/session toolkit
  (`scripts/bridge_session.py`, `scripts/bridge_replica.py`) for scripted keyboard-driven
  hardware trials with flight-recorder telemetry.
- **A benchmark/probe toolset** (`scripts/play.py --checkpoint-file ... --eval-seeds ...`)
  for judging policies deterministically (tracking error, fall rate, cost of transport,
  action smoothness) instead of relying on training-time reward curves, plus an HRL
  goal-achievability probe (`--diagnose-goals`) that decomposes tracking error into
  high-level goal-setting error vs low-level goal-reaching error.
- A running experiment ledger under [`doc/hrl/`](doc/hrl) and
  [`docs/adr/`](docs/adr) — design docs, findings, and architecture decisions, kept
  current as the thesis work progresses.

Everything else — the multi-robot task registry (Go2, G1, A2, R1, H1_2, H2, As2), the
generic mjlab training/play/motion-imitation workflow, and deploy for the other robots —
is inherited from upstream and works as documented there; this README focuses on the
H1-2 HRL work.

## Architecture status

| Arch | Task ID | Idea | Status |
|---|---|---|---|
| **A0** | `Unitree-H1_2-Flat` | Flat PPO baseline | ✅ trained, deployed on real hardware |
| **A1** | `Unitree-H1_2-Flat-A1` | HIRO: PPO low-level + TD3 high-level + goal relabeling | ✅ tracking at A0 parity, energy gap remains; deployed in sim, hardware deploy in progress |
| **A2** | — | A1 + RMA-style privileged-extrinsics adaptation | ⬜ spec ready ([`doc/hrl/A2_ARMA.md`](doc/hrl/A2_ARMA.md)) |
| **A3** | — | Offline gait library + RL residual | ⬜ planned |
| **A4** | — | Trajectory + MPC | ⏸ likely out of scope |

Full status, numbers, and open questions: [`doc/hrl/HRL_plan.md`](doc/hrl/HRL_plan.md)
(start here), architecture-specific docs linked from it.

## Installation

See [`doc/setup_en.md`](doc/setup_en.md) (conda environment, mjlab/MuJoCo-Warp,
dependencies).

Conda env used throughout this project: `unitree_mjlab_h1_2_rl`.

## Training

Flat baseline (A0):

```bash
conda activate unitree_mjlab_h1_2_rl
python scripts/train.py Unitree-H1_2-Flat --env.scene.num-envs 4096 \
    --agent.max-iterations 5001 --agent.run-name a0_baseline
```

Hierarchical (A1), optionally warm-started from a converged A0 checkpoint:

```bash
python scripts/train.py Unitree-H1_2-Flat-A1 --env.scene.num-envs 4096 \
    --agent.max-iterations 5001 --agent.run-name a1_td3_pose0p5_ar0p02 \
    --agent.warm-start-path logs/rsl_rl/h1_2_velocity/<A0 run>/model_<it>.pt
```

> [!NOTE]
> Launch with the activated conda env + `python` directly, not `conda run` — the
> latter buffers stdout and breaks live W&B streaming.
>
> `--agent.max-iterations` counts *additional* iterations on top of any resume, and the
> checkpoint is named after the last iteration index, so ask for `N+1` (e.g. `5001`) to
> land on the round number `model_5000.pt`.

Training logs to `logs/rsl_rl/h1_2_velocity(_a1)/<date_time>/`; W&B project `biped_hrl`.

## Evaluation

Replay a policy in MuJoCo:

```bash
python scripts/play.py Unitree-H1_2-Flat-A1 --checkpoint-file <path to model_N.pt> --num-envs 1
```

Deterministic benchmark (the canonical way to compare policies — use `fall_rate`, not
episode length, for survival):

```bash
python scripts/play.py <TaskID> --checkpoint-file <ckpt> --num-envs 64 \
    --eval-steps 600 --eval-seeds 2
```

Sustained-command hold diagnostic (batch-averaged metrics can hide bimodal failure
modes at a fixed commanded speed):

```bash
python scripts/play.py <TaskID> --checkpoint-file <ckpt> --eval-cmd-vx 0.5
```

A1-specific high-level-vs-low-level error decomposition:

```bash
python scripts/play.py Unitree-H1_2-Flat-A1 --checkpoint-file <ckpt> \
    --diagnose-goals 600 --eval-seeds 2 --num-envs 64
```

Export to ONNX for deployment with `--export-onnx` (also done automatically during
training).

## Sim-to-real deployment (H1-2)

Requires [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git) and
[unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git).

```bash
cd deploy/robots/h1_2
mkdir build && cd build
cmake .. && make
```

Place the exported flat-policy ONNX under
`deploy/robots/h1_2/config/policy/velocity/v0/exported/` (`State_RLBase`, A0), or the
paired high-level/low-level HRL ONNX pair under
`deploy/robots/h1_2/config/policy/velocity_hrl/v0/exported/` (`State_RLHRL`, A1).

```bash
./h1_2_ctrl --network=lo        # simulation deployment (unitree_mujoco), or
./h1_2_ctrl --network=<iface>   # real robot, e.g. enp5s0
```

Sim-first deployment via [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
is strongly recommended before touching real hardware. For scripted, repeatable
hardware trials (keyboard-command sequences with flight-recorder telemetry logging),
see `scripts/bridge_session.py`.

Deployment details, ONNX metadata (baked HRL goal scale), and the runtime safety filter
are documented in `.claude/docs/deployment.md`.

## Key findings so far

- A1's HIRO hierarchy matches A0's tracking accuracy (err_vx/vy/yaw ≈ 0.10) at zero
  additional fall rate; the remaining gap is energy efficiency (cost of transport), not
  stability or tracking.
- The H1-2 actuator retuning (PD gains, derived action scale, passive damping) — not
  domain randomization or reward shaping — is what separated a policy that only worked
  in simulation from one that walked on real hardware.
- Full experiment history, ablations, and dead ends: the A1 findings ledger (research knowledge base, not published),
  [`doc/hrl/worklines.md`](doc/hrl/worklines.md).

## License

Apache License 2.0 — see [`LICENCE`](LICENCE), inherited unchanged from upstream.

## Acknowledgements

Built on top of:

- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab): the base
  RL/deploy framework this project is forked from
- [mjlab](https://github.com/mujocolab/mjlab.git): training and execution framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): PPO implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated simulation
- [mujoco](https://github.com/google-deepmind/mujoco.git): rigid-body physics engine
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): motion tracking reference
