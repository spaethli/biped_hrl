# Deployment pipeline (H1-2)

## C++ controller

Binary: `deploy/robots/h1_2/build/h1_2_ctrl`

Setup scripts (**source them, don't execute**):

- `source setup_all_mujoco.sh` → `H1_2_DOMAIN_ID=1`, `NETWORK=lo`,
  `CFG=deploy.yaml` → run `h1_2_sim`
- `source setup_all_robot.sh` → `H1_2_DOMAIN_ID=0`, `NETWORK=enp11s0`,
  `CFG=deploy_real.yaml` → run `h1_2_real`

FSM states + keyboard: `i`=FixStand, `o`=Velocity/walk, `p`=Passive.
Velocity keys: `w/s`=fwd/bwd, `a/d`=strafe, `q/e`=turn.
Config: `deploy/robots/h1_2/config/config.yaml` (`keyboard_transitions`).
Observation assembly: `deploy/robots/h1_2/src/State_RLBase.cpp`
(`keyboard_velocity_commands`).

## Bridge-plant mismatch (known, accepted 2026-07-08)

The `simulate/` MuJoCo bridge loads the **raw** `scene_h1_2.xml`, whose joint defaults
(`damping="1" armature="0.1" frictionloss="0.2"`) are much harsher than the training
nominal (per-motor armature 0.025/0.04/0.005/0.002, frictionloss 0, no passive damping —
mjlab injects these via `h1_2_constants.py`, which the bridge never reads). Consequence:
policies look dirtier in the bridge than in play.py and can fall at the max
vx=1.0-step-from-stand command even when mjlab shows fall_rate 0.0 for the identical
condition (measured 2026-07-08, `a0_v2_baseline`). Pre-existing (v1 identical), NOT a
policy or export bug — the ONNX/obs/gain path itself validated clean. Accepted as-is for
now; **revisit before real deploy**: either align the scene XML joint defaults with the
training nominal (faithful bridge) or with the vendor reference (0.01/0.1/0.001, mild
sim2real proxy) — decide which job the bridge is doing.

## Deploy configs

- Sim: `config/policy/velocity/v0/params/deploy.yaml` (keyboard_velocity_commands)
- Real: `config/policy/velocity/v0/params/deploy_real.yaml` (velocity_commands/joystick)
- **Split deploy** (ADR-0005 step 3b, 2026-07-07): `hold_joint_ids: [12..26]` in the
  `deploy_real` params makes torso+arms track `default_joint_pos` instead of the policy
  action (`State_RLBase.cpp`/`State_RLHRL.cpp`, robot-local; obs still see all 27
  joints). Absent key = full forward (sim yamls). Gains/scales in all 4 param yamls
  MUST stay in lockstep with `h1_2_constants.py` (v2-final: torso 200/2.5, shoulders
  120/2, elbow+wrists 80/1, derived scales).

## ONNX export (the correct path)

```bash
python scripts/play.py Unitree-H1_2-Flat \
    --checkpoint-file logs/rsl_rl/h1_2_velocity/<run>/<model>.pt \
    --export-onnx
cp logs/.../policy.onnx deploy/robots/h1_2/config/policy/velocity/v0/exported/policy.onnx
```

`--export-onnx` is a bare flag; it calls `runner.export_policy_to_onnx()` and exits
without the viewer. Training saves also auto-export `policy.onnx` (with metadata)
next to each checkpoint via `VelocityOnPolicyRunner._export_policy_onnx`.

**A1/HRL deploy (as-built, 2026-06-18 — sim).** Two-ONNX hierarchy, all robot-local
(no `deploy/include/isaaclab/` edits):
- Export: `HierarchicalRunner.export_hierarchy_to_onnx` (via `play.py --export-onnx`) →
  `high_level.onnx` (obs `policy++command`→g; ppo Gaussian-mean / td3 `tanh` + normalizer
  baked in) + `low_level.onnx` (obs `policy++goal`→action). Oracle skips `high_level.onnx`.
- Run: `State_RLHRL` (`src/State_RLHRL.cpp`, FSM `type: RLHRL`, reach via key **`h`** /
  gamepad `RT+X`); loads both ONNX lazily on first `enter()`, fires HL every `c` (holds
  `V*`), feeds LL `V*−s`. Config dir `config/policy/velocity_hrl/v0/`; goal math in
  `include/hrl/goal_space.h`. A0 (`State_RLBase`, key `o`) untouched.
- `deploy.yaml` `hrl:` block switches checkpoints with no rebuild: `hl_algorithm`
  (`oracle`=analytic `V*`, no HL net | `learned`), `c`, `hl_target_mode`, `goal_components`,
  `nominal_root_height` (=1.3076, imu-site standing z; offset cancels in height delta).
- **Gotcha:** `main.cpp` must `#include "FSM/State_RLHRL.h"` or the `REGISTER_FSM` registrar
  is dropped from the static lib → runtime `Unknown FSM type RLHRL`.

## A1 needs a runtime base-velocity estimate (A0 does not)

The goal the LL reads is `V* − s`, and the velocity part of the goal-space state `s` is the
current base linear velocity (`goal_space._vel_extract`). So forming the goal at runtime needs
vx/vy (+ height); yaw-rate and orientation come from the gyro/IMU and are fine. **No network
takes `base_lin_vel`** (actor obs omit it, like A0) — it's used only to build `s`.
`State_RLHRL.cpp` reads it (+height) from the sportmode HighState (`highstate_->velocity()` is
WORLD-frame, rotated to body via the IMU quat; `position().z` is height): in sim that's the
MuJoCo bridge's **ground-truth** `rt/sportmodestate` (privileged); on the real robot it must come
from the onboard sport-mode estimator (noisy/drifting) or a custom one. A0 has no such dependency.
**Test knob:** `deploy.yaml` `hrl.state_noise: {velocity, orientation, height}` injects per-step
Gaussian noise into `s` in sim, to emulate that estimator noise (default 0). Finding (2026-06-18,
learned `absolute` TD3): under realistic velocity/height noise the clean-trained LL still stands
but is **much twitchier, worst when standing still** (cmd=0 → goal = −noise → phantom corrections;
HL input isn't noised, so this is the LL reacting to noisy goal feedback) → see
`doc/hrl/A1_findings.md`. The **training-side** counterpart of this deploy knob is now built —
`GoalStateNoise` injects bias/drift/lag into the LL goal channel during training (#8b, see
`.claude/docs/hrl-infra.md`), so the policy learns robustness rather than only being tested for it.
To remove the dependency entirely instead: switch the LL to observe the **absolute `V*`** instead
of the delta (`doc/hrl/A1_HIRO.md` reserve variant) → no runtime velocity estimate needed.

## Safety filter + flight recorder (deploy-side, 2026-06-03)

Real-time safety filter in `State_RLBase::run()` (A0) and `State_RLHRL::run()` (A1).
Compile switch `#define SAFETY_FILTER` in each (`State_RLBase.cpp`=1 on; `State_RLHRL.h`=0
off by default — flip to 1 to enable on A1). Robot-local only: `robots/h1_2/include/{h1_2_limits.h,
safety_logger.h}`; the shared `deploy/include/FSM/State_RLBase.h` guards its logger include with
`__has_include` so g1/go2/a2 still build. Thresholds centralized in `h1_2_limits.h` (`H1_2_*`).

- **Triggers (OR → ramped hold):** joint pos limits (`h1_2_joint_limits`); IMU tilt
  >`0.44` rad (~25°); downward-accel fall (`a_world_z < −7 m/s²` sustained 60 ticks).
  Each engages a 50-tick (50 ms @ 1 kHz) ramp `α:0→1` blending the command
  `q=(1−α)·policy + α·q_meas` (position hold; kp/kd unchanged). Complements — fires earlier/softer
  than — the existing FSM `bad_orientation`→Passive check.
- **IMU accel convention (verified in sim):** physical IMU = *specific force* → standing reads
  +9.81, so `a_world_z=(R·a_imu).z−9.81` ≈0 standing/hanging, ≈−9.81 free-fall. Sim free-fall
  measured ~−10.7 (leg flailing accelerates pelvis past g) → triggers correctly; no false positive
  at rest. The `−9.81` is only correct if the real IMU reports specific force — re-verify on hardware.
- **Flight recorder** (`safety_logger.h`, header-only): active only when `SAFETY_FILTER=1` **and**
  `H1_2_SAFETY_LOG` set. Launch scripts (`h1_2_sim`/`h1_2_real`) auto-set it to
  `logs/deploy_safety/<ts>` (timestamped, no manual preamble). Writes `<base>.csv` (per tick: raw
  policy `action`=**pre-filter** intent, measured q/dq, IMU quat+accel, α, trigger flags) +
  `<base>_meta.json` (limits/names/thresholds/dt). Hot-path safe: RAM buffer, flush on exit + ~2.5 s.
  Limitation: one flat folder, not the per-arch run dir (A0/A1 chosen at runtime by key; deployed
  ONNX carries no source-run record). For per-arch routing, add a config-driven `safety_log_dir` key.
- **Offline analyzer:** `python scripts/safety_analyzer.py <base>` → `<base>_report.json` +
  `[SAFETY] {json}` one-liner. Reports raw-vs-measured limit-violation rates, filter engagement
  (per trigger), per-joint stats. **Raw violations = target-space aggressiveness** (policy commands
  `processed_actions` past mechanical stops, esp. ankles), NOT unsafe excursions; measured excursions
  are the physical truth. Example 64 s run: 55% raw vs 11% measured, measured max-over ≤0.04 rad.

## Visualize a checkpoint

```bash
python scripts/play.py Unitree-H1_2-Flat \
    --checkpoint-file logs/rsl_rl/h1_2_velocity/<run>/<model>.pt \
    --num-envs 1
```

For A1 runs play.py restores the run's goal space / HL algorithm from its
`params/agent.yaml` automatically (check the `Restored run structure` line).
