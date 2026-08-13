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
The high level's *state target* in a velocity/orientation/height subspace that the low
level is rewarded for *reaching* (HIRO L2 distance). One of two HL->LL intent channels;
the other is the _Gait reference_. The velocity command is removed from the LL obs, so
these two channels carry the LL's only task intent.
_Avoid_: command, setpoint (the *command* is the user/eval twist, a different thing);
"the only HL->LL channel" (the gait reference is a second, non-L2 channel since A1a).

**Gait reference** (cadence):
The high level's commanded step *period*, the second HL->LL channel (A1a onward). Unlike
the _Goal_, it is not a state target reached by L2 distance: the low level *entrains* to
it via the open-loop phase clock and the `feet_gait` reward. Deploy-clean, since the phase
is a function of the commanded period and time, and the entrainment reward is sim-only.
This is the channel the A3 gait library's footfall pattern plugs into.
_Avoid_: goal, V* (those are the L2-reached state targets); step length (the reciprocal of
cadence at a fixed speed, v = f*L, not an independent channel).

**Cost of transport** (`CoT`):
The high level's efficiency objective (A1a onward): window mechanical energy / *actual
walked* distance (`Σ ‖v_xy‖·dt`), engaged only when the commanded linear speed exceeds A0's
`command_threshold` (0.1). Actual (not commanded) distance so that spending energy while
going nowhere is correctly expensive; a small denominator floor guards the stuck-robot
blow-up (it is not the engagement gate, which is the command threshold). Added (negative) to
the HL reward *only*, so the low level stays a pure tracker. That decoupling (efficiency at
the HL, tracking at the LL) is what justifies the hierarchy against a flat A0+energy baseline.
The HL trims CoT through the _Gait reference_ at the commanded velocity.
_Avoid_: energy, power (CoT is energy normalized by distance; raw energy has a stand-still
attractor, the failure that killed `hl_reward_mode=task`); "LL energy term" (CoT never
enters the LL reward).

**Intrinsic reward**:
The A1 low level's *training* signal: the goal-distance reward `-Σ_c w_c ‖V*_c - s_c‖`.
The only thing the LL learns on (`ll_task_reward_coef = 0`). May also carry privileged
upper-body regularization (see _Posture penalty_) and, from A1a, a gait-entrainment term
(`feet_gait` keyed to the _Gait reference_); it is not restricted to canonical HIRO L2.
_Avoid_: task reward (that is the A0-comparable env reward, not what the LL trains on).

**Task reward**:
The shared A0 env reward (velocity tracking + posture + penalties). For A1 it is logged
and accumulated for the high level, but it does **not** train the LL.
_Avoid_: intrinsic reward (the LL's actual signal).

**Posture penalty**:
A negative upper-body regularizer added to the A1 LL _intrinsic reward_ so the LL keeps
the arms near default and low-jitter (the env's `variable_posture`/`action_rate_l2`
never reach the LL through the goal-only routing). The deploy-hygiene fix for A1's
uncontrolled arm swing.
_Avoid_: pose reward (`variable_posture` is a positive exp term; the LL uses a negative
deviation penalty).

**Nominal model**:
The unrandomized simulation plant (actuator PD gains, joint friction, armature, effort
limits) that every architecture trains on; DR samples around it, it is not itself
randomized.
_Avoid_: "the sim", "the env" (broader: those include rewards/observations); "model"
alone (collides with the policy networks).

**Model v2**:
The corrected, versioned nominal model (hold-gain arms + waist at 300/3, legs unchanged,
joint friction 0; trained with desired_kl=0.01) that all A0-A4 runs use from its landing on.
Checkpoints and benchmark scores are not comparable across the v1/v2 boundary.
_Avoid_: "new sim", "fixed env".

**Split deploy**:
The hardware deployment mode where the RL policy commands only the legs while a separate
onboard controller holds the upper body at a fixed pose. The full 27-dof policy is still
trained in sim; only its leg commands reach the robot.
_Avoid_: "12-dof deploy" (the trained policy stays 27-dof); "lower-body policy" (there is
one policy, partially forwarded).

**Hold gains**:
The stiff arm PD set used to pin the arms at a fixed pose during split deploy, and
(from Model v2) also the sim's arm actuator gains, so trained and held arm dynamics
match. The waist is held at 300/3 too (option B); the torso+arm hold only trains under
desired_kl=0.01 (stalls at 0.005).
_Avoid_: "arm gains" (ambiguous with the old soft RL-arm gains); "safety gains".

**Base-velocity increment**:
The *change* in pelvis linear velocity since the start of the current HL window, `s_i - s_t0`.
Not a velocity: it is re-zeroed every `c` steps and is exactly 0 at each window start. It is
what a `delta`-mode _Goal_ makes sufficient for the low level, because `V* - s_i = scale*g -
(s_i - s_t0)` cancels the absolute term. Deployable from the IMU alone.
_Avoid_: "the velocity estimate", "base velocity" (the increment is not one, and reading it
as one is exactly the 2026-08-05 defect: the high level, which needs the *absolute* value,
read 0 at every fire).

**Leg odometry**:
The *instantaneous absolute* pelvis velocity computed from stance-foot kinematics and the
gyro, `v = -d(p_foot)/dt - w x p_foot`. Drift-free because nothing is integrated. The high
level's velocity source, since it is the one quantity a `delta` _Goal_ does not cancel.
_Avoid_: "odometry" unqualified (the word normally implies an integrated *position*, which
this deliberately is not); "the estimator" (ambiguous with the IMU _Base-velocity increment_
and the leg-FK height, which are different signals feeding different consumers).

**Fused base velocity**:
A pelvis-velocity estimate that combines the IMU with leg kinematics through a filter, as
opposed to _Leg odometry_, which is instantaneous and unfiltered. Covers the complementary
filter and every EKF variant alike. The distinction is load-bearing because the two classes
fail in opposite directions: _Leg odometry_ is noisy but unbiased, a fused estimate trades
that noise for bias (see _Contact-transition loss_).
_Avoid_: "the estimator" (already flagged as ambiguous under _Leg odometry_); "EKF" (a
complementary filter is fused and is not an EKF); "filtered odometry" (the fusion adds the
IMU, it does not merely smooth the leg signal).

**Contact-transition loss**:
The roughly fixed displacement a _Fused base velocity_ loses per step, because the
stationary-foot measurement keeps being applied across touchdown and lift-off, when the foot
is not yet or no longer stationary. Measured 9-17 mm per step and independent of walking
speed, so it scales with steps per metre rather than with distance. A *filter* artifact:
_Leg odometry_, which applies no such measurement, does not show it.
_Avoid_: "foot rollover" (that names the shelved WL-D arm-6 reward, a policy behaviour);
"stance-point error" (which point is chosen is settled and irrelevant — see _Stance point_;
this is about the transition, not the point); "slip" (a robot/ground phenomenon, whereas this
is generated inside the estimator).

**Stance point**:
The single point _Leg odometry_ assumes is fixed in the world while a foot is stance
(`kFootSiteA`, x = +0.04 in the ankle_roll frame). **Which point is chosen does not matter**,
and that is the settled answer, not an open question: a planted foot is rigid, so every point
attached to it is equally stationary. Measured end-to-end 2026-08-11 — moving the reference
to the ankle joint, the heel or the toe changes the `c`-averaged estimate by 0.04-0.21 m/s in
no consistent direction, and the heel (the geometrically "correct" pivot for this robot's
toe-up stance) is *worse* on both sessions. The spread tracks |p_foot| through the `w x p`
term, i.e. it rescales gyro error rather than fixing a modelling defect.
_Avoid_: "contact point" (unqualified, ambiguous with *which foot* is in contact — a separate
and already-settled question); "foot rollover" (that names the shelved WL-D arm-6 reward, a
policy *behaviour*, and is a different subject entirely).

**Suicide attractor**:
The failure where an always-negative reward makes early termination optimal (stop
accruing negative reward). The reason A1 uses `fell_over = time_out` (a truncation that
bootstraps) instead of A0's true terminal. An all-positive (exp) intrinsic would remove
it — the motivation for the planned exp follow-up.
_Avoid_: collapse (broader — also covers std blowup / coverage collapse).
