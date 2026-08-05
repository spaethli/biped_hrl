# Deployment pipeline (H1-2)

## C++ controller

Binary: `deploy/robots/h1_2/build/h1_2_ctrl`

Setup scripts (**source them, don't execute**):

- `source setup_all_mujoco.sh` → `H1_2_DOMAIN_ID=1`, `NETWORK=lo`,
  `CFG=deploy.yaml` → run `h1_2_sim`
- `source setup_all_robot.sh` → `H1_2_DOMAIN_ID=0`, `NETWORK=enp11s0`,
  `CFG=deploy_real.yaml` → run `h1_2_real`

**GOTCHA — deploy-config fixtures must exist in BOTH policy dirs** (2026-07-22): `CtrlFSM`
constructs EVERY registered state at startup, and `State_RLBase` (`velocity/v0`) and
`State_RLHRL` (`velocity_hrl/v0`) both read the SAME `H1_2_DEPLOY_CFG` name and both call
`YAML::LoadFile(policy_dir/params/<name>)`. So a fixture present in only one dir makes
`h1_2_ctrl` die at startup with a yaml-cpp **`BadFile`** exception ("bad file"), before any
state runs — even if you only intend to use the other one. This bit `deploy.yaml.g2_4_split_test`,
which existed only under `velocity/v0`; the A1 twin was added 2026-07-22. When adding any new
fixture, create it in both dirs (`g26_pinned` and `w1_legacy_test` already follow this).

FSM states + keyboard: `i`=FixStand, `o`=Velocity/walk, `p`=Passive.
Velocity keys: `w/s`=fwd/bwd, `a/d`=strafe, `q/e`=turn (clamped to training ranges),
`1`-`9`=held vx 0.1-0.9, `0`=stop (2026-07-16, for held-command gates; `w`=vx **1.0**,
the range edge — use numbers first).
Config: `deploy/robots/h1_2/config/config.yaml` (`keyboard_transitions`).
Observation assembly: `deploy/robots/h1_2/include/h1_2_observations.h`
(`keyboard_velocity_commands`; moved there from `State_RLBase.cpp` on 2026-07-21 with the
`gait_phase_cmd` fix, see A1a_deploy_plan.md "Defect 0 FIXED").

## Bridge plant

> **NAMING CORRECTED + PLANT SWITCHED, 2026-08-04.** Everything below called
> `h1_2_handless.xml` the "vendor reference" and treated it as the hardware-realistic
> plant. **That label was wrong.** `h1_2_handless_stress.xml` / `scene_stress.xml` is the
> **official Unitree-shipped scene**; `h1_2_handless.xml` is a **hand-made
> training-proximate variant**. The 2026-07-15 reasoning about latency (below) is still
> correct and still explains why the exact training nominal fails — only the provenance of
> the values, and hence the realism argument, was mislabelled.
>
> **The shipped scene is now the single gating plant** (`robot_scene: scene_stress.xml`).
> Its harsh joint values are the better hardware proxy: real harmonic-drive joints carry
> substantial friction and damping. Its floating base is now explicitly zeroed — inheriting
> the file default put damping 1 / armature 0.1 / frictionloss 0.2 on all six free-base
> DOF, i.e. drag and fictitious inertia on motion through space, which nothing physical
> produces. Measured A/B: removing it changed behaviour negligibly.
>
> **Two costs on record.** (1) The shipped plant is *calmer*, so it is the LESS sensitive
> twitch detector — and twitch is what separated every failed A1 candidate from arm4d
> (act_rate A0 0.57–0.75 vs A1 0.92–1.28). Keep `h1_2_handless.xml` for A/B work.
> (2) **Every reference number predating 2026-08-04 was measured on the training-proximate
> scene and does NOT transfer** — leg action rates 0.5908 / 0.7552 / 0.8125, arm4d's live
> pass, the G2.x parity bars. Re-baseline before comparing. Measured example of the gap:
> over an identical cmd-0 stand window the same keeper's estimator residual reads ~5x
> higher on the training-proximate plant than on the shipped one.
>
> Falsified along the way: that the fictional free-base damper was what made the shipped
> scene look calm. It is the **joint** damping (1 vs 0.001, a 1000x difference) — zeroing
> the base changed almost nothing.

### Original 2026-07-15 record (values right, "vendor" label wrong)

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

**E1 result on real H1-2 (2026-07-20): FAIL, structural, not a code bug.**
`unitree_go/msg/SportModeState` (topic `odommodestate`; `read_all_joints.cpp` subscribes to
the wrong name `sportmodestate`, which doesn't exist, but even the correct topic reads
all-zero) only populates while Unitree's OWN built-in motion-control service owns the
robot — it goes silent the moment a custom low-level policy takes command, which is
required to run ours at all. Confirmed via `ros2 topic info -v` (live publisher, correct
type, zero payload) plus Unitree's own docs. **Does not block A0** (no velocity
dependency in its obs). Blocks the eventual A1 hardware attempt; three fallback options
are on the table (not yet chosen), see `doc/hrl/A1a_deploy_plan.md` "E1/E2 result": (1) a
custom leg-odometry estimator from joint encoders + IMU (both fine under `lowstate`
independent of the vendor's motion service), (2) the absolute-`V*` LL retrain below
(removes the dependency entirely), (3) swap velocity/height for acceleration in the goal
space — the existing roadmap idea the hierarchy-benefit roadmap (research KB) `#6` / thesis M4,
not a new idea; this E1 result is the trigger that makes it live rather than deferred.
E2 (IMU specific-force convention) passed cleanly on the same session.

**Offline hardware sensor validation tool:** `scripts/robot_estimator_check.py` — generic
analyzer for `read_all_joints` CSV logs (specific-force check, velocity bias/noise,
a rotation-vs-yaw-rate frame check for base-vs-other-frame estimator bugs, a
walk-distance integration check). Not gate-specific; point it at any `all_joints_*.csv`
and mark time windows.
### A1 goal scale vs the command-range safety limit (RESOLVED 2026-07-16, WL-C/WL-B)

`deploy.yaml commands.base_velocity.ranges` drives **two unrelated things**:

- `observations.h:118-120` (`velocity_commands`, joystick = the REAL robot) **clamps the
  commanded twist** → the operator's safety limit; authoritative (the user: deploy.yaml is the
  last call for deployed policies).
- `hrl::GoalSpace::scale()` **decodes the A1 HL's `g` into `V*`** → trained policy semantics.

Before the fix these were the same knob, so narrowing the ranges for safety (e.g.
`lin_vel_x: [-0.25, 0.5]` on a policy trained at `(-0.5, 1.0)`) silently halved the A1 goal
scale (0.75 → 0.375) and gutted the HL's goal authority — the deploy twin of the sim bug
(`--eval-cmd-vx` collapsed the same scale to its 1e-3 floor; the A1 findings ledger (research KB) WL-C).
A0 was always immune (no goal space).

**Fixed:** `HierarchicalRunner` exports `goal_scale` in both ONNX files' metadata, and
`State_RLHRL::ensure_models_loaded` pins it into `GoalSpace::freeze_scale` at load — before
any range is read. **`commands.base_velocity.ranges` is now a pure operator clamp in `delta`
mode: narrow it for safety freely.** Look for `[HRL] goal scale pinned from ONNX metadata
[...]` at FSM entry; the deployed keeper pair carries `0.750,0.500,1.000,1.000,1.000,1.000,
0.200`. Verified by a standalone test of the robot-local path (8/8, incl. "scale() ignores
yaml ranges" and a wrong-dim throw); `onnx_parity.py` re-passes (LL 4.2e-05, HL 8.5e-06).

**Two limits to know:**
- **Legacy ONNX** (exported before 2026-07-16) carry no `goal_scale` → the C++ warns loudly
  and falls back to the old range-derived path. **Then the ranges must equal the trained
  ones and narrowing is unsafe.** Fix by re-exporting (`play.py --export-onnx`).
- **`hl_target_mode: absolute`** also decodes against `GoalSpace::center()`, which stays
  range-derived (it is NOT pinnable: the ONNX `goal_center` holds absolute TRAINING-frame
  values — its height column is pelvis z 1.02, while this deploy measures at the imu site,
  `nominal_root_height` 1.3076; adopting it would inject that 0.29 m offset). So **do not
  narrow ranges under `absolute`** — `State_RLHRL` warns. `delta` is the A1 default and what
  is deployed, so this does not bite today.

Bridge footnote: `keyboard_velocity_commands` (`include/h1_2_observations.h`) reads the ranges
into a `cfg` var and **never uses it** (key map hardcoded, `w`=1.0) — the sim bridge has no clamp at
all, so don't validate a command limit there. Only the joystick path clamps.

**Test knob:** `deploy.yaml` `hrl.state_noise: {velocity, orientation, height}` injects per-step
Gaussian noise into `s` in sim, to emulate that estimator noise (default 0). Finding (2026-06-18,
learned `absolute` TD3): under realistic velocity/height noise the clean-trained LL still stands
but is **much twitchier, worst when standing still** (cmd=0 → goal = −noise → phantom corrections;
HL input isn't noised, so this is the LL reacting to noisy goal feedback) → see
the A1 findings ledger (research KB). The **training-side** counterpart of this deploy knob is now built —
`GoalStateNoise` injects bias/drift/lag into the LL goal channel during training (#8b, see
`.claude/docs/hrl-infra.md`), so the policy learns robustness rather than only being tested for it.
To remove the dependency entirely instead: switch the LL to observe the **absolute `V*`** instead
of the delta (`doc/hrl/A1_HIRO.md` reserve variant) → no runtime velocity estimate needed.

## Stage-D bridge validation battery (designed 2026-07-14, grilled; run before any H1-2 session)

> **RUN THIS VIA `scripts/deploy_readiness.py` (2026-08-04).** One command chains
> provenance → sim → bridge (candidate + A0 control) → analyze into a single verdict
> (exit 0 GO / 3 CAVEAT / 1 NO-GO / 2 INFRA). Step-by-step operating manual, with the
> manual command for each step and its corresponding W/G gate, is in
> `doc/hrl/A1a_deploy_plan.md` ("HOW TO RUN THE GATES"). Check what is deployed FIRST:
> `python scripts/deploy_provenance.py --check --checkpoint-file <pt>`.

Tooling (2026-07-15/16): `scripts/onnx_parity.py` (step 1, `[PARITY]` json, CPU-vs-CPU;
pass `--onnx-dir` to score the **deployed** file rather than a fresh temp export);
`scripts/bridge_replica.py` (⚠ **RETIRED as a gate 2026-08-05** — no C++, **no DDS at all**,
`--delay-ms` emulates a constant latency via a deque; models neither the safety filter nor
`hold_joint_ids` nor `joint_offset` nor the base-state estimator. It passed every phase on
the shipped scene where the real bridge fell on the 0.5→0 decel. Out of the default chain;
keep for headless plant/latency A/B only — it screens, it cannot clear);
`scripts/deploy_gate_analyzer.py` (per-segment metrics from the `<base>_hrl.csv` telemetry
that State_RLHRL writes when `H1_2_SAFETY_LOG` is set). Full gate plan:
`doc/hrl/A1a_deploy_plan.md`. **Two bugs fixed in `bridge_replica.py` (2026-07-17):** a
variable name collision (`c` = HL decision period, shadowed every tick by the
swing-clearance instrument's per-contact loop variable also named `c`) crashed any
`--policy hrl` run — renamed to `con`. Its `gscale` also used to derive from `deploy.yaml`
ranges (the legacy path) instead of the HL ONNX's `goal_scale` metadata, so it wasn't
exercising the same scale the real C++ path (`GoalSpace::freeze_scale`) does — fixed to
read the metadata, falling back to the legacy derivation with a warning if absent.

Both scenes (vendor + stress variant, see Bridge plant above), per candidate checkpoint:
1. **ONNX↔torch parity** + 27/27 gain/scale lockstep vs both YAMLs (V2-gate procedure).
2. **Held-command walk** (the user-required deploy gate): stand → hold vx 0.5 ≥30 s → stop;
   repeat at 1.0. Direction held (no crab/backwards), ramp-then-track, arms calm in-bridge.
   ~~Known blocker: the A1a velocity-hold HL degeneracy — this gate fails until that is
   fixed~~ **[VOID 2026-07-16, WL-C: that "degeneracy" was a sim-eval artifact. Re-measured,
   every A1 run holds ≥ A0 in mjlab (ss@0.5 0.030–0.048 vs A0 0.057; ss@1.0 0.039–0.097 vs
   0.076, 0 falls) — `A1a_plan.md` table (f). There is no known blocker on this gate; run it.]**
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

- **Triggers:** IMU tilt >`0.44` rad (~25°) and downward-accel fall (`a_world_z < −7 m/s²`
  sustained 60 ticks) engage a 50-tick ramp `α:0→1` blending `q=(1−α)·policy + α·q_meas`.
  ⚠ that "hold" re-reads q_meas per tick → at α=1 it is DAMPING-ONLY (near-passive), not a
  rigid freeze; G3.0 must decide latched-hold vs damp-mode as the terminal behavior
  (`A1a_deploy_plan.md` post-mortem issue 3). **Joint limits: split by state (2026-07-16)** —
  A0 (`State_RLBase`) keeps any-joint-out → hold; A1 (`State_RLHRL`) CLAMPS the offending
  command to `h1_2_joint_limits` instead (the A1a gait rides its stops every stride; the hold
  response completed the 2026-07-16 bridge fall at tilt 0.196). `trig_joint` in the A1 flight
  log now means "clamp active". Audit note: `h1_2_limits.h` disagrees with the sim model at
  the knee ([-0.26, 2.05] vs XML [-0.12, 2.19]) — re-audit vs URDF before hardware.
  Complements — fires earlier/softer than — the existing FSM `bad_orientation`→Passive check.
- **IMU accel convention (verified in sim):** physical IMU = *specific force* → standing reads
  +9.81, so `a_world_z=(R·a_imu).z−9.81` ≈0 standing/hanging, ≈−9.81 free-fall. Sim free-fall
  measured ~−10.7 (leg flailing accelerates pelvis past g) → triggers correctly; no false positive
  at rest. The `−9.81` is only correct if the real IMU reports specific force — re-verify on hardware.
- **Flight recorder** (`safety_logger.h`, header-only): active only when `SAFETY_FILTER=1` **and**
  `H1_2_SAFETY_LOG` set. Launch scripts (`h1_2_sim`/`h1_2_real`) auto-set it to
  `logs/deploy_safety/<ts>` (timestamped, no manual preamble). Writes `<base>.csv` (per tick: raw
  policy `action`=**pre-filter** intent, measured q/dq, IMU quat+accel, α, trigger flags, `entry`
  id) + `<base>_meta.json` (limits/names/thresholds/dt). Hot-path safe: RAM buffer, flush on exit
  + ~2.5 s. Limitation: one flat folder, not the per-arch run dir (A0/A1 chosen at runtime by key;
  deployed ONNX carries no source-run record). For per-arch routing, add a config-driven
  `safety_log_dir` key.
  **Multi-attempt sessions (fixed 2026-07-17, two rounds):** `init()` used to truncate the CSV
  and reset the tick counter on EVERY FSM entry (not once per process), silently destroying an
  earlier attempt's telemetry (e.g. a fall) the moment a later attempt in the same session got
  logged. Round 2: A0 and A1 hold SEPARATE `SafetyLogger` instances (`State_RLBase.h` vs.
  robot-local `State_RLHRL.h`), so a per-instance "have I run before" check still let the FIRST
  A0->A1 switch in a session truncate A0's data — the new instance had never run, so from ITS
  view it was still a first entry. Fixed properly: truncate-vs-append is decided by whether the
  CSV already exists ON DISK (robust across different C++ objects, not instance memory), and
  `entry` is a process-wide counter (`g_safety_logger_entry`, a C++17 inline global), unique
  across both same-type retries and cross-type switches. The A1-only `hrl::Telemetry`
  (`hrl_telemetry.h`, `<base>_hrl.csv`) got the same fix, plus its flush cadence dropped from
  ~10s to ~2.5s (matching the safety logger) — an abrupt session end (not a clean FSM exit) used
  to lose the telemetry tail, which is exactly when it matters most; caught once by cross-checking
  `trig_fall` in the base CSV against an apparently-clean `_hrl.csv`. `deploy_gate_analyzer.py`
  now also segments on `entry` changes (not just command changes) and never drops a fallen
  segment under its 3s minimum-duration floor (a fast fall must never be filtered out).
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
