# HRL Locomotion (H1-2) — Domain Language

Glossary for the thesis project: HRL locomotion architectures (A0–A4) for the Unitree
H1-2, compared for sim-to-real transfer. Definitions only — design/implementation lives
in `doc/hrl/` and `.claude/docs/`; decisions in `docs/adr/`.

## Language

**Extrinsics** (`e_t`):
The privileged, per-episode environment-factor vector (friction, mounted payload and
the CoM shift it causes, per-joint motor-strength and damping, …) available only in
simulation. The *inputs* to the RMA encoder, not its output.
_Avoid_: latent, z (those are the encoding), "privileged obs" (broader).

**Extrinsics latent** (`z`) — **A2/RMA only**:
The compact 8-dim encoding `z = μ(e_t)` the encoder produces and feeds to the policy.
At deployment it is replaced by the adaptation estimate `ẑ`. A learned *compression*:
it exists only where an encoder `μ` exists.
_Avoid_: extrinsics, e_t (those are the raw inputs); using it for H-adapt, which has no
`μ` — that is `ê` below.

**Extrinsics estimate** (`ê`) — **H-adapt only**:
What H-adapt's `φ` produces: a direct estimate of the extrinsics themselves, same space
and dimension as `e` (ℝ⁵ on the H1-2), not a compression of them. H-adapt has no encoder
`μ`, so there is nothing to compress — the high level consumes raw `e` in simulation and
`ê` on the robot.
_Avoid_: `z` (that is A2's compressed code, a different object); "latent" unqualified.

**Mounted payload**:
The weighted backpack carried in front of the torso. A *single* physical placement, so
its mass and the CoM shift it causes are one extrinsic, not two: mass never arrives
without a shift. Contrast the robot's own CoM uncertainty, which is a separate extrinsic
that exists with no payload present.
_Avoid_: "payload" alone when the CoM shift is meant; treating payload mass and CoM
offset as independently sampled quantities.

**Mount lever arm** (`d`):
The vector from the torso's own centre of mass to the mounted payload's centre of mass.
Together with the payload mass it determines the whole inertial effect — CoM shift and
added rotational inertia alike. A property of the *rig*, not of the load: changing how
much is in the pack moves the mass, not `d`.
_Avoid_: "offset" (ambiguous with the resulting CoM shift, which is a different, smaller
vector).

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
⚠ On the robot the channel can be **switched off without any error**: a non-zero `pin_period`
in the deploy yaml holds the clock at a constant, and the HL's period output is then never
decoded at all, so the commanded cadence is absent from behaviour *and* from the telemetry.
A pinned session yields no evidence about cadence, which matters because this is the
hierarchy's only actuated adaptation channel and the one every payload claim runs through.
_Avoid_: goal, V* (those are the L2-reached state targets); step length (the reciprocal of
cadence at a fixed speed, v = f*L, not an independent channel); reading a pinned session's
logged period as the HL's choice.

**Cost of transport (objective)** (`CoT`):
The high level's efficiency objective (A1a onward): window mechanical energy / distance
walked **along the command** (the *signed projection* `Σ (v_xy·ĉ_xy)·dt`, since 2026-07-10;
the undirected form let the HL earn cheap metres sideways). Engaged only when the commanded
linear speed exceeds A0's `command_threshold` (0.1). Achieved (not commanded) distance so
that spending energy while going nowhere is correctly expensive; a small denominator floor
guards the stuck-robot blow-up (it is not the engagement gate, which is the command
threshold). Added (negative) to the HL reward *only*, so the low level stays a pure tracker.
That decoupling (efficiency at the HL, tracking at the LL) is what justifies the hierarchy
against a flat A0+energy baseline. The HL trims CoT through the _Gait reference_ at the
commanded velocity.
This is the quantity `hl_cot_coef` is calibrated against and the *only* one
`hl_cot_cap_commanded` acts on (it caps the credited distance at the commanded distance).
_Avoid_: energy, power (CoT is energy normalized by distance; raw energy has a stand-still
attractor, the failure that killed `hl_reward_mode=task`); "LL energy term" (CoT never
enters the LL reward); using it interchangeably with the _diagnostic_ below.

**Cost of transport (diagnostic)**:
The *reported* efficiency number: the same energy over **undirected** distance
(`Σ ‖v_xy‖·dt`). This is what `metrics/cot` logs in training, what the sim benchmark's `cot`
key holds, and what a hardware run yields once a ground-truth distance is available. It is
the only form defined identically at training, sim and hardware **for both architectures**,
because the projection needs a high level and A0 has none, so it is the form the
training→sim→real comparison uses.
⚠ It is **not** the trained objective and must never be cited as evidence about
`hl_cot_cap_commanded`, which leaves it untouched by construction. A figure showing it must
say which of the two it is.
_Avoid_: calling it "CoT" unqualified in any claim about the HL's incentives.

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
A tracking error scored against one of these, rather than against ground truth, is a
different quantity from the sim metric of the same name and is named with an `_est` suffix,
carrying the reference it was scored against alongside it. The suffix names the *class*
because the reference is expected to change (WL-G ships B for A1 and carries C to A2), so
the specific arm belongs in the payload, never in the key.
_Avoid_: "the estimator" *in prose* (already flagged as ambiguous under _Leg odometry_ --
the `_est` suffix above is a key-naming convention, and is unambiguous only because the
reference travels with it); "EKF" (a complementary filter is fused and is not an EKF);
"filtered odometry" (the fusion adds the IMU, it does not merely smooth the leg signal).

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

**Smoothness**:
Not a scalar, and never reported as one: a family of quantities indexed by three
coordinates — *where* in the chain (commanded → torque → realized joint → body), *which*
statistic (mean = sustained agitation, p95 = impact severity), and *which* _Regime_. The
coordinates are load-bearing because they rank policies oppositely: one arm was best of 13
on commanded action rate and worst of 13 on realized `jacc_legs_p95`.
_Avoid_: "smoothness" bare in any claim (always name the coordinate); "action rate" as a
synonym (that is one cell of the table); "jitter" unqualified (say _Action jitter_ or _DoF
position jitter_ — they are different rungs of the chain).

**Action jitter**:
The third time-derivative of the *commanded* joint targets, `d³q_des/dt³` in rad/s³,
legs-only. The primary smoothness quantity (LCP, Chen 2025), chosen because it separates
smoothing methods ~13x where first/second-derivative metrics separate them 1.2-1.7x.
Published Unitree H1 references: **0.44 rad/s³** in MuJoCo, 1.11-1.20 on real ground.
_Avoid_: action rate (the *first* derivative — a different, much less discriminating
number); `act_legs` (the same first derivative in raw action units).

**DoF position jitter**:
The third time-derivative of the *realized* joint position, rad/s³, legs-only — the
realized-motion counterpart of _Action jitter_, and the rung where the plant's own
filtering shows up. Published Unitree H1 reference: 0.10 rad/s³ in MuJoCo.
_Avoid_: joint acceleration / `jacc` (the second derivative, kept only as the p95
impact-severity statistic).

**Regime**:
Which of the two disjoint command conditions a smoothness number was measured under:
**walking** (‖cmd‖ > 0.1, A0's `command_threshold`) or **standing** (‖cmd‖ ≤ 0.1). Always
reported separately, never pooled, because a lever can improve one while destroying the
other (measured: 5325 vs 142 zero-command touchdowns) and because the aggregate bench is
~95/5 walking/standing (`rel_standing_envs = 0.05`), too thin to surface a standing defect.
_Avoid_: "the benchmark number" (the aggregate pools regimes and is not a walking number);
"stand still" (that names the `ll_stand_still_coef` reward term, not a measurement
condition).

**Seed band**:
The same training recipe run with two or more seeds, scored side by side. A claimed effect of a
lever (payload DR, adaptation) must exceed the between-seed spread of the SAME recipe, and a
difference whose sign flips with the seed is not an effect of the lever. Standing behaviour
replicates worst (4.2x at identical config and seed in training), so hardware and sim standing
claims are made per (variant, seed), never from one pooled arm.
_Avoid_: "the DR arm" / "the keeper" as if one checkpoint stood for a recipe.

**Physical units rule**:
Every commanded-side smoothness number is reported in radians, never raw action units,
because the action scale `κ_j = 0.25·τ_max_j/Kp_j` spans 6.7x across the body (legs 0.25,
shoulder_yaw 0.0375). Legs-only numbers are unaffected in *ranking* (κ is uniform 0.25
across all 12 leg joints, so the conversion is a constant ×0.25), but whole-body raw norms
physically over-weight the arms by 3-6.7x and are invalid across the Model v2 PD-gain
restructure.
_Avoid_: whole-body `action_rate` in any cross-policy claim.

**Flight recorder**:
The deploy-side per-tick CSV written by the C++ `SafetyLogger` to `logs/deploy_safety/`,
carrying the *commanded* side of a hardware run: raw policy intent `raw_q`, the operator
command, measured `q`/`dq`, IMU, the passive base-estimator bank and the safety triggers.
Decimated to ~500 Hz, so its rows are not policy steps. The only log that records what the
policy *asked for*.
_Avoid_: "the hardware log" (ambiguous with the _Joint telemetry log_, which carries
different columns from a different process); "the safety log" (the triggers are one block
of many); "the deploy CSV".

**Joint telemetry log**:
The ROS-side CSV written by `read_all_joints` to `~/ramlab_ws/trajectories/`, sampled on a
clean undecimated 50 Hz grid. Uniquely carries **`tau_est`**, the measured joint torque,
which exists in no other log and is what makes a hardware _Cost of transport_ a measurement
rather than a PD-law estimate. Its `vel_*`, `pos_*`, `body_height`, `foot_force_*` and
`foot_raise_height` columns are the dead `sportmodestate` block and are identically zero.
_Avoid_: "all_joints" bare, "the trajectory log" (nothing is a trajectory here);
"the estimator log" (its estimator columns are the dead ones).

**Run**:
One continuous occupancy of an RL state on the robot: the operator enters the policy,
it walks, it leaves. **This, not a file, is the unit every hardware metric is scored over.**
It is identified by the _Flight recorder_'s `entry` column, which is process-wide and
increments on every re-entry, so one _Flight recorder_ file holds as many Runs as the
operator started without restarting the controller (three, in `15-00-37.csv` on 2026-09-14,
under two different experimental conditions). Scoring a file whole pools distinct Runs and
is always wrong.
A Run's wall-clock extent comes from `t_wall` only. The `t` column is a tick counter that
advances one control step per loop iteration and does not run while the robot sits in
Passive, so the gaps *between* Runs are invisible in it.
_Avoid_: "session" or "recording" for a single Run (a session is a day's work and a
recording is a file, and both hold several); "a log" (one Run is a slice of two logs).

**Session pair**:
The alignment of one _Run_ with the _Joint telemetry log_ that was recording during it, so
the commanded side and the torque can be read on the same clock. The relation is **not
one-to-one in either direction**: several Runs share one _Flight recorder_ file, and one
_Joint telemetry log_ can cover several consecutive Runs.
Matching is decided by **cross-correlating a shared measured joint against every candidate**,
fitting offset and rate together. Overlapping recording intervals are a *sanity flag only*,
never the proposer: on 2026-09-14 the intervals excluded the correct partner for 4 of 12
Runs, because the two loggers are started and stopped independently. A pair whose wall-clock
offset sits far outside the rest of the day's is reported for a human decision and refused by
default, since a quasi-periodic walking knee plus a free rate parameter can correlate highly
against the *wrong* walking bout.
Neither filename is a first-sample time: the _Flight recorder_'s is process launch, and
logging begins whenever the RL state is entered, so the offset between the two logs is
operator timing (-3.3 s and -10.3 s on two days; -6 s to +113 s across 2026-09-14). The two
time bases also differ in RATE (measured -2462 to -6613 ppm; mostly the recorder's `t`, a tick
counter that runs 0.27-1% slow of `t_wall`), so the alignment is affine in time,
never a constant offset: over a 170 s Run a constant offset drifts by most of a stride
period, and the pair then fails to correlate at all even when it is the right one.
_Avoid_: "sync", "the offset" (both imply a single constant, which is the defect); "the
session start time" (there are two, and neither filename records one).

**Motion-capture ground truth** (`gt_*`):
The pelvis twist recovered from the marker cluster: time-aligned on `|ω|` against the
_Flight recorder_'s `t_wall` (invariant to the unknown marker-to-body rotation), frame solved
by orthogonal Procrustes, then lever-corrected `v_pelvis = v_marker − ω × r`. Only **clean**
capture is used: a row seen on fewer than 3 markers, or within 0.2 s of a dropout, is excluded
from both fits and written as NaN, never interpolated. Written per _Run_ as
`mocap_aligned.csv`, one row per flight-recorder row, keyed by its `t`. It is **ground truth**,
so it takes the sim key names (`err_vx`, not `err_vx_est`) and the _Fused base velocity_
becomes the quantity under test, reported separately as `est_rms_v*`. This is the only
hardware reference that is not a function of the policy's own motion, which is what the `_est`
suffix exists to warn about. Its quality travels with it in `run.json` (`mocap_align`: window
agreement, gravity check, `flags`); a flagged Run is written but low quality. The torso frame is
fitted from the gyro by default; a **geometric frame** from the measured marker layout is the
IMU-free alternative and cross-check. Method and formulas: `doc/hrl/mocap_alignment.md`.
Available for 4 Runs of 2026-09-14 and all 21 of 2026-09-16; the bench uses it only above 50%
coverage. A _Run_ without it scores against the estimator and says so in `err_v_reference`, so
the two are never silently mixed in one column.
_Avoid_: "the Vicon data" (the raw capture is not this; the alignment and lever correction
are what make it a pelvis twist); "validated" for a Run that has no capture; aligning on `t`
(a tick counter: slow and bent against real time).
