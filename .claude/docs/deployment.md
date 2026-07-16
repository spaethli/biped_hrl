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

## Bridge plant (RESOLVED 2026-07-15 — faithful + stress variant)

**The live bridge is `/opt/unitree_mujoco`** (`simulate/build/unitree_mujoco`, config
`/opt/unitree_mujoco/simulate/config.yaml`, scene `unitree_robots/h1_2/scene.xml` →
`h1_2_handless.xml`) — NOT the in-repo `simulate/` copy (never built/used; its
`scene_h1_2.xml` is body-identical but is not what the bridge loads). The bridge config
also provides `enable_elastic_band: 1`, a virtual lifting harness (sim gantry).

**Plant = vendor reference** (`armature 0.01 / frictionloss 0.1 / damping 0.001`, clean
free base) since 2026-07-15. NOT the mjlab training nominal, and that is deliberate:

- The old defaults (`damping=1 armature=0.1 frictionloss=0.2`, applied to the FREE base
  joint too — a hidden 6-DOF base damper) were much harsher than training; policies
  looked dirtier in the bridge and fell at vx=1.0-from-stand where mjlab shows fall_rate
  0.0 (2026-07-08).
- A first fix set the plant to the exact training nominal (per-motor armature,
  frictionloss 0). **It failed immediately** (2026-07-15): FixStand leaned forward with
  heels unloading and A0 became violently twitchy/unstable. Root cause (headless
  bridge-replica + delay sweep): the bridge computes **explicit torque PD at 500 Hz with
  ~2-4 ms real feedback latency** (DDS + threads); on a dissipation-free plant, 2 ms of
  delay turns the PD into an energy-injecting oscillator (qvel_rms x30, falls), while the
  old harsh plant absorbs 6 ms without a trace. mjlab has neither latency nor explicit PD
  (implicit position servos), so "faithful plant" ≠ faithful dynamics in this bridge.
- The vendor reference absorbs the full 0-6 ms latency range with graceful degradation,
  passes FixStand→takeover→stand→0→1.0-step→stop→walk at 0/2/4 ms in the replica, and is
  also the honest hardware proxy (real joints have friction; firmware PD runs multi-kHz).

The original harsh values live on as `h1_2_handless_stress.xml` + `scene_stress.xml` =
the robustness stress gate (G2.7). Switch via `robot_scene:` in the /opt config. NOTE:
these edits live OUTSIDE the thesis repo (the /opt clone has its own git); re-verify
after any unitree_mujoco update. Bring-up note: pure-PD FixStand is NOT self-stable on
any plant — the elastic band does the stabilizing until the policy takes over.

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
**The velocity MUST be the BASE (pelvis) velocity** (2026-07-15): the bridge's `frame_vel` sensor
originally sat on the torso-mounted imu SITE, whose ω×r sway component feeds back through the LL
goal delta and destabilizes A1 (keeper slammed ankle limits in 0.2 s; fine with pelvis velocity).
Both bridge scenes now publish pelvis `frame_vel`; a real estimator must equally output base-frame
velocity, not imu-frame (E1 gate criterion).
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

## Stage-D bridge validation battery (designed 2026-07-14, grilled; run before any H1-2 session)

Both scenes (faithful + stress variant, see Bridge plant above), per candidate checkpoint:
1. **ONNX↔torch parity** + 27/27 gain/scale lockstep vs both YAMLs (V2-gate procedure).
2. **Held-command walk** (the user-required deploy gate): stand → hold vx 0.5 ≥30 s → stop;
   repeat at 1.0. Direction held (no crab/backwards), ramp-then-track, arms calm in-bridge.
   Known blocker: the A1a velocity-hold HL degeneracy (`doc/hrl/A1a_plan.md` table f) — this
   gate fails until that is fixed; A0 passes (rs20 baseline ss 0.055/0.077, t90 <1 s in mjlab).
3. **Command steps**: stand→walk→stand, yaw both directions, short vy hold (vx-1.0 step from
   stand = the known hard case).
4. **Safety-envelope audit**: bridge logs vs `h1_2_joint_limits` over the whole battery; arm
   joint velocities get their own line (stage-D subject).
5. **Safety-filter audit**: run with `SAFETY_FILTER=1` + flight recorder; extract trigger
   counts by type, ticks with α>0, max α, tripping joint. Report margins even when clean
   (worst tilt in the vx-1.0 ramp vs 0.44 rad; worst heel-strike `a_world_z` + consecutive-tick
   count vs −7/60; per-joint min distance to limits — ankle pitch + hip roll tightest).
   Pass = zero triggers with comfortable margins; triggers during visually-correct walking →
   adjust that threshold (data-driven, user call); A/B `SAFETY_FILTER=0` only if triggers fire.
Parked until after the held-command blocker: goal-channel state-noise DR arm (#8b machinery
validated, keeper trains without it).

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
