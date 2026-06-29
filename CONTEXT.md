# HRL Locomotion (H1-2) — Domain Language

Glossary for the thesis project: HRL locomotion architectures (A0–A4) for the Unitree
H1-2, compared for sim-to-real transfer. Definitions only — design/implementation lives
in `doc/hrl/` and `.claude/docs/`; decisions in `docs/adr/`.

## Language

**Extrinsics** (`e_t`):
The privileged, per-episode environment-factor vector (friction, base mass / CoM,
per-joint motor-strength and damping, …) available only in simulation. The *inputs* to
the RMA encoder, not its output.
_Avoid_: latent, z (those are the encoding), "privileged obs" (broader).

**Extrinsics latent** (`z`):
The compact 8-dim encoding `z = μ(e_t)` the encoder produces and feeds to the policy.
At deployment it is replaced by the adaptation estimate `ẑ`.
_Avoid_: extrinsics, e_t (those are the raw inputs).

**Adaptation module** (`φ`):
The deployable network that estimates `ẑ` from a window of recent proprioceptive
history (no privileged access). A 1D-CNN over the history; trained to regress to `z`.
_Avoid_: encoder (that is `μ`, the privileged one).

**Narrow DR**:
The domain-randomization set A1 currently trains under (friction, base CoM, push, …).
The narrow column of the A2 2×2 evaluation.
_Avoid_: "A1 DR", "base DR" (use Narrow DR consistently).

**Wide DR**:
Narrow DR plus per-joint motor-strength and damping randomization. The randomization
A2's encoder is designed to adapt to; the wide column of the 2×2.
_Avoid_: "A2 DR", "heavy DR", "full DR".

**Motor strength** (DR):
Randomization of the actuator **PD gains** (Kp/Kd) — the RMA "motor model" term. An
extrinsic in the Wide DR set.
_Avoid_: torque limit / effort (that is actuator *capacity*, a different knob we are
not using); "Kd damping" (the controller Kd ≠ passive joint damping below).

**Joint damping** (DR):
Randomization of the **passive** joint damping (`dof_damping`) — a physical joint
property, sampled per-joint. An extrinsic in the Wide DR set.
_Avoid_: controller Kd (that lives in Motor strength / `pd_gains`).

**Actuation delay** (DR):
Per-episode command latency (physics timesteps) between the policy output and the
motor, modeled natively by mjlab's actuator `delay_min_lag`/`delay_max_lag`. A
Wide-DR extrinsic (quasi-static per episode).
_Avoid_: observation delay (sensor-pipeline latency — a separate thing).

**Goal** (`V*`):
The high level's target in a velocity/orientation/height subspace that the low level is
rewarded for reaching. The hierarchy's only task-intent channel to the LL (the velocity
command is removed from the LL obs).
_Avoid_: command, setpoint (the *command* is the user/eval twist, a different thing).
