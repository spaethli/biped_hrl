# H1-2 kinematics, frames and IMU conventions — state-estimation reference

Written for a Rotella-style flat-foot EKF (Rotella, Bloesch, Righetti, Schaal, IROS 2014,
[arXiv:1402.5450](https://arxiv.org/abs/1402.5450)). No EKF is implemented here; this fixes the
conventions it has to be built on.

**Every claim is tagged.** ✅ = verified from the URDF/MJCF/code/SDK header on this machine, with
the file named. ⚠️ = assumption or inference, with what would settle it. Numbers marked *measured*
came from MuJoCo FK or mesh vertices, re-derivable with the commands in §9.

# Nice *.html visualization
https://claude.ai/code/artifact/2a68c47b-e45e-4fa9-b84e-ab41e054dc6a4

---

## A. Frame / kinematic tree

✅ Root is `pelvis`, carrying the freejoint. The IMU is on `torso_link`, one revolute joint away
from the root. Contact-relevant frames in **bold**.

```
world
└── pelvis ........................... floating base (freejoint), model root
    ├── left_hip_yaw_link ............ Rz(q0)
    │   └── left_hip_pitch_link ...... Ry(q1)
    │       └── left_hip_roll_link ... Rx(q2)          [zero offset: axes 1,2 intersect]
    │           └── left_knee_link ... Ry(q3)
    │               └── left_ankle_pitch_link ..... Ry(q4)
    │                   └── **left_ankle_roll_link** ... Rx(q5)   foot body
    │                       ├── site left_foot ......... (0.04, 0, -0.04)
    │                       ├── **CONTACT frame (proposed)** (0.04, 0, -0.045)
    │                       └── 7 foot capsules / 1 mesh geom, sole plane z = -0.045
    ├── right_hip_yaw_link ........... mirrored (y sign), same chain
    └── torso_link ................... Rz(ψ), torso_joint, ZERO body offset
        ├── **imu site / imu_link** ... (-0.04452, -0.01891, 0.27756), rpy 0
        ├── left_shoulder_pitch_link .. arm chain, 7 DoF
        └── right_shoulder_pitch_link . arm chain, 7 DoF
```

✅ The load-bearing structural fact: `torso_link` has body offset **(0, 0, 0)** from the pelvis and
hangs off `torso_joint`, a z hinge anchored at the pelvis origin (`h1_2.xml:138-140`;
`h1_2_handless.urdf` `torso_joint` origin `xyz="0 0 0"`). **The pelvis and torso frame origins
coincide**; they differ only by the waist yaw ψ. The IMU is therefore a pure Rz(ψ) plus a fixed
lever arm away from the base.

✅ Model source: training loads **MJCF, not URDF** —
`src/assets/robots/unitree_h1_2/xmls/h1_2.xml` via `get_spec()` (`h1_2_constants.py:28`).
29 bodies, 27 actuated hinges + 1 freejoint, nq/nv = 34/33, mass 66.984 kg.

---

## B. Frames and transforms

Left leg; the right mirrors the y sign of the two hip offsets. Angles in rad, lengths in m.

| # | frame (link) | parent | joint / type | transform parent→child at q = 0 | axis | range | what it physically is |
|---|---|---|---|---|---|---|---|
| — | `pelvis` | world | `floating_base_joint`, free (URDF: root) | — | — | — | base link; **not** where the IMU is |
| — | `torso_link` | `pelvis` | `torso_joint`, revolute | **(0, 0, 0)**, rpy 0 | z | −2.35 … 2.35 | waist yaw; origin coincides with pelvis |
| — | `imu_link` / site `imu` | `torso_link` | fixed | (−0.04452, −0.01891, 0.27756), rpy 0 | — | — | **physical IMU location**, axes = torso axes |
| 0 | `left_hip_yaw_link` | `pelvis` | `left_hip_yaw_joint`, revolute | (0, +0.0875, −0.1632) | z | −0.43 … 0.43 | joint frame |
| 1 | `left_hip_pitch_link` | hip_yaw | `left_hip_pitch_joint`, revolute | (0, +0.0755, 0) | y | −3.14 … 2.5 | joint frame |
| 2 | `left_hip_roll_link` | hip_pitch | `left_hip_roll_joint`, revolute | **(0, 0, 0)** | x | −0.43 … 3.14 | joint frame; coincident with hip_pitch ⇒ pitch and roll axes intersect |
| 3 | `left_knee_link` | hip_roll | `left_knee_joint`, revolute | (0, 0, −0.4) | y | −0.12 … 2.19 | joint frame; thigh = 0.4 |
| 4 | `left_ankle_pitch_link` | knee | `left_ankle_pitch_joint`, revolute | (0, 0, −0.4) | y | −0.897334 … 0.523598 | joint frame; shank = 0.4 |
| 5 | `left_ankle_roll_link` | ankle_pitch | `left_ankle_roll_joint`, revolute | (0, 0, −0.02) | x | −0.261799 … 0.261799 | **foot body**; roll axis 20 mm *below* pitch axis |
| — | site `left_foot` | ankle_roll | fixed | (0.04, 0, −0.04) | — | — | virtual; what current leg odometry differences |
| — | **contact frame `F_L`** (proposed, §C) | ankle_roll | fixed | **(0.04, 0, −0.045)** | — | — | virtual; sole-plane point at the support-polygon centroid |

Right-side deltas: hip offsets become (0, −0.0875, −0.1632) and (0, −0.0755, 0);
`right_hip_roll_joint` range mirrors to −3.14 … 0.43. Everything else is identical. ✅

Fixed transforms worth having as constants:

```
r_imu   = (-0.04452, -0.01891, 0.27756)     IMU site in the torso frame
c_A     = ( 0.04,     0.0,     -0.045  )    contact frame in the ankle_roll frame  (§C)
site_A  = ( 0.04,     0.0,     -0.04   )    existing left_foot / right_foot site
sole_z  = -0.045                            sole plane in the ankle_roll frame
```

---

## C. Recommended contact frame

**Recommendation: a virtual frame rigidly attached to `*_ankle_roll_link`, origin at
(0.04, 0, −0.045) in that link's frame, axes aligned with the link.** Not the ankle, not a sole
corner, not the CoP.

### The geometry the choice rests on (all *measured*)

| quantity | value | consequence |
|---|---|---|
| sole plane, ankle_roll frame | z = −0.045 | ankle_roll origin sits **45 mm above** the ground when flat |
| sole hull, x extent | −0.086 … +0.173 | foot is 259 mm long, strongly forward-biased about the ankle |
| sole hull, y extent | ±0.042 | 84 mm wide |
| sole hull area | 0.02115 m² | support polygon when flat |
| sole hull **area centroid** | **x = +0.03999, y = +0.00001** | the ankle is **40 mm behind** the centroid |
| existing site `left_foot` | (0.04, 0, −0.04) | its x/y is the sole centroid **to 0.01 mm**; it is 5 mm above the sole plane |
| ankle pitch vs roll axis | separated by 20 mm in z | **there is no single "ankle point"** — the two ankle axes do not intersect |

### Why this frame

1. **A flat-foot contact constrains 6 DoF, so the estimated object must be a full pose rigidly
   attached to the foot body.** Rotella's extension over the point-foot filter is exactly the foot
   *orientation* state `z_i` (eq. 13, 15). A frame attached to `ankle_roll_link` supplies both the
   position and the orientation with no extra modelling.
2. **Origin on the sole plane makes "the foot does not move during contact" literally true.** With
   the foot flat and not slipping, a point on the sole is world-fixed. The ankle origin is not: it is
   45 mm above the contact plane, so any foot pitch/roll rotation translates it. That violates the
   process model ṗ_i = C^T w_p, which is the assumption the whole filter leans on.
3. **The support-polygon centroid minimises the error from unmodelled foot micro-rotation.** Roll or
   pitch slip of ε about some edge of the contact patch displaces a point by ε times its distance
   from the rotation axis. The centroid minimises the RMS of that distance over the patch, so it is
   the origin choice that keeps the residual smallest without knowing where the CoP actually is.
4. **It is 5 mm from the existing site**, so the calibration history of the current leg-odometry
   estimator (`rl/hrl/leg_odom.py`, `hrl/base_state.h`) remains comparable; nothing else about the
   chain changes.

### Why not the alternatives

* **The ankle joint (last joint frame).** 45 mm above the sole and 40 mm behind the centroid, and
  there is no single ankle point at all — the pitch and roll axes are 20 mm apart. Rejected on the
  process-model grounds in (2).
* **The foot-link frame (`ankle_roll_link` origin).** Identical to the ankle point; same objection.
* **A sole corner or a specific capsule endpoint.** Arbitrary, and maximally sensitive to foot
  rotation (largest lever from the centroid). The deploy height estimator legitimately evaluates
  *corner* points (`lowest_foot_z` uses x ∈ {−0.08, 0.17}), but that is a min-over-the-foot query,
  a different job from a contact frame.
* **The centre of pressure.** ⚠️ The CoP migrates inside the support polygon during a step and is
  not measurable on this robot anyway (no foot F/T — see §H). Rotella's formulation does not want
  the instantaneous CoP: it wants a body-fixed frame whose pose is constant during contact, with the
  CoP-induced micro-motion absorbed by the process noise `w_p` / `w_z`.

⚠️ **Assumption:** that the real sole is flat and rigid at z = −0.045. The models say −0.045
(five variants checked, §H item 7) but the physical foot has a compliant pad. Sole compression under
load is a systematic offset on p_z, and if it is a few mm it directly biases the estimated height,
not the velocity.

---

## D. IMU frame convention

### Where it is

✅ **Torso frame.** Site `imu` inside body `torso_link` at (−0.04452, −0.01891, 0.27756) with
no `quat` attribute, so its axes are `torso_link`'s (`h1_2.xml:145`). The URDF agrees exactly:
`imu_link`, fixed joint, parent `torso_link`, `origin rpy="0 0 0"` (`h1_2_handless.urdf:827-831`).

### What each channel is expressed in

| channel | field / sensor | frame | ✅/⚠️ |
|---|---|---|---|
| angular velocity ω̃ | `imu_state.gyroscope[3]`, sim sensor `imu_ang_vel` | **torso** | ✅ code + MJCF |
| specific force f̃ | `imu_state.accelerometer[3]`, sim sensor `imu_lin_acc` | **torso** | ✅ code + MJCF |
| attitude | `imu_state.quaternion[4]`, sim `framequat imu_quat` | **torso in world** | ✅ code + MJCF |
| Euler | `imu_state.rpy[3]` | torso in world | ✅ SDK header; unused by this project's code |
| linear velocity | sim sensor `imu_lin_vel` (velocimeter) | torso, at the site | ✅ sim only, no hardware equivalent |
| leg FK | `leg_fk` in `hrl/base_state.h` | **pelvis** | ✅ |

✅ **Gravity is included** in the accelerometer, as in any strapdown IMU: the deploy code removes it
as `a + proj_grav·|g|` with |g| = 9.81 (`State_RLHRL.cpp:601-607`, `base_state.h:46`), and the unit
test asserts an upright stationary reading cancels to zero (`base_state_estimator_test.cpp:115-123`).
Upright, the z channel reads ≈ +9.81.

✅ **Quaternion element order is (w, x, y, z)**: the code constructs
`Eigen::Quaternionf(quaternion()[0], [1], [2], [3])`, and Eigen's 4-arg constructor is
(w, x, y, z) (`State_RLHRL.cpp:626-629`, `unitree_articulation.h:36-41`). ⚠️ This is verified as
*what our code assumes*, not from Unitree documentation.

⚠️ **Units are inferred, not documented.** `IMUState_.hpp` carries no unit comments. Everything in
the code is consistent with rad, rad/s, m/s²: gravity removal uses 9.81 (m/s²), joint ranges are
radian-valued, and the trained policy consumed rad/s gyro without a scale factor. Settle it with a
static bench reading (‖accelerometer‖ ≈ 9.81 confirms m/s²) and a known 90° rotation.

✅ **Axis convention**: x forward, y left, z up, right-handed — inherited from the URDF/MJCF, where
the pelvis and torso frames are so oriented and the IMU site adds no rotation.

### Relationship to the base frame

With ψ = waist yaw = `motor_state[12].q` and r = r_imu:

```
R_P←T = Rz(ψ)                              orientation:  torso → pelvis
p_I^P = Rz(ψ)·r                            IMU position in the pelvis frame
ω_P   = Rz(ψ)·ω_T − ψ̇·ẑ                    pelvis angular velocity from the torso gyro   ✅ State_RLHRL.cpp:652-654
v_P^P = Rz(ψ)·(v_I^T − ω_T × r)            pelvis linear velocity from the IMU-site velocity ✅ base_state.h:18-22
p^I   = Rz(−ψ)·p^P − r                     any pelvis-frame point, expressed in the IMU frame
```

*Measured:* the last identity reproduces MuJoCo to **6.7e-16 m** over 300 random poses with
|ψ| ≤ 1.5 rad, and the orientation form `R^I = Rz(−ψ)·R^P` to 3.3e-16.

⚠️ **Known frame inconsistency inside our own stack.** Training reads `base_ang_vel` from the
IMU-site gyro (**torso**, `velocity_env_cfg.py:59-62`) but `projected_gravity` from the root-link
quaternion (**pelvis**). Deploy takes both from the IMU quaternion (**torso**,
`unitree_articulation.h:36-42`). The mismatch is a pure Rz(ψ): identically zero for the gyro, and
second-order for projected gravity since a yaw rotation barely moves a near-vertical unit vector.
Harmless for the policy, but do not assume "body frame" means one thing across the obs vector.

---

## E. Forward kinematics and foot velocity

### E.1 T_BF(q), with the base B ≡ the IMU/torso frame

Per leg, with q = (q₀ … q₅) = (hip yaw, hip pitch, hip roll, knee, ankle pitch, ankle roll) and
s_y = +1 left, −1 right:

```
R₀ = Rz(q₀)                       p₀ = (0, s_y·0.0875, -0.1632) + R₀·(0, s_y·0.0755, 0)
R₁ = R₀·Ry(q₁)·Rx(q₂)             p₁ = p₀ + R₁·(0, 0, -0.4)        thigh
R₂ = R₁·Ry(q₃)                    p₂ = p₁ + R₂·(0, 0, -0.4)        shank
R₃ = R₂·Ry(q₄)                    p₃ = p₂ + R₃·(0, 0, -0.02)
R₄ = R₃·Rx(q₅)                    R_PA = R₄                        ankle_roll orientation in P
                                  p_F^P = p₃ + R₄·c_A              contact point in P,  c_A = (0.04, 0, -0.045)
```

Then into the IMU/torso frame (this is the piece Rotella does not have, because his IMU is at the
base):

```
T_IF(q, ψ) = [ R_IF | p_F^I ]      R_IF  = Rz(-ψ)·R_PA(q)
                                   p_F^I = Rz(-ψ)·p_F^P(q) - r_imu
```

*Measured:* the pelvis-frame chain reproduces MuJoCo site positions to **5.6e-16 m**, and the
IMU-frame composition to **6.7e-16 m** (300 random poses, |ψ| ≤ 1.5 rad). ✅ It is also the exact
chain already deployed (`hrl/base_state.h` `leg_fk`) and simulated
(`rl/hrl/leg_odom.py` `foot_site_b`, C++ parity 6.3e-7 m/s) — only `c_A` changes, from the site's
−0.04 to the sole plane's −0.045.

### E.2 Jacobians

Geometric Jacobian of the contact frame, pelvis frame, column i built from joint axis a_i and joint
origin o_i (both in P, both produced by the same forward pass):

```
J_v^P[:,i] = a_i × (p_F^P − o_i)          J_ω^P[:,i] = a_i           i = 0…5
```

*Measured:* matches finite differences to **4.0e-8** (eps 1e-7, i.e. truncation-limited).

Into the IMU frame, and adding the waist column (∂/∂ψ, using d/dψ Rz(−ψ) = −[ẑ]×·Rz(−ψ)):

```
J_v^I = Rz(-ψ)·J_v^P            J_ω^I = Rz(-ψ)·J_ω^P            (leg columns)
∂p_F^I/∂ψ = -ẑ × (Rz(-ψ)·p_F^P) = -ẑ × (p_F^I + r_imu)          (waist column, linear)
∂/∂ψ of R_IF ⇒ angular column = -ẑ                              (waist column, angular)
```

So the full contact-frame Jacobian in the IMU frame is 6×7 per leg: six leg joints plus the waist.

### E.3 Foot velocity from joint velocities

```
v_F^I = J_v^I·q̇_leg + (∂p_F^I/∂ψ)·ψ̇          ω_F^I = J_ω^I·q̇_leg − ẑ·ψ̇
```

---

## F. How a stationary foot enters the EKF

### F.1 Rotella's structure, in H1-2 symbols

Rotella's state, extended to two flat feet (his §IV, eq. 7-15). Here `C = C(q_WI)` is the rotation
matrix corresponding to the state quaternion, mapping **world → body**, and the body **is the IMU /
torso frame** (see §G item 1):

```
x  = [ r,  v,  q_WI,  p_L,  p_R,  z_L,  z_R,  b_f,  b_ω ]      nominal dim 30
δx = [ δr, δv, δφ,    δp_L, δp_R, δθ_L, δθ_R, δb_f, δb_ω ]   ∈ R^27   (15 + 6·N_feet)

ṙ   = v
v̇   = Cᵀ(f̃ − b_f − w_f) + g
q̇   = ½·[ω̃ − b_ω − w_ω ; 0] ⊗ q
ṗ_i = Cᵀ·w_p,i                      ← "the foot does not move while in contact"
ż_i = ½·[w_z,i ; 0] ⊗ z_i           ← "the foot does not rotate while in contact"
ḃ_f = w_bf     ḃ_ω = w_bω
```

**There is no explicit foot-velocity measurement.** Stationarity is encoded in the process model
(ṗ_i, ż_i driven by noise only); the correction comes from a *kinematic pose* measurement. Velocity
becomes observable because r and p_i are coupled through that measurement over time.

### F.2 The measurement, in H1-2 conventions

Measured (from encoders + §E FK), per foot in contact:

```
s_p,i = p_F,i^I(q_leg,i, ψ)                     = Rz(-ψ)·p_F,i^P(q) − r_imu          + n_p
s_z,i = quat( R_IF,i(q_leg,i, ψ) )              = quat( Rz(-ψ)·R_PA,i(q) )           ⊗ exp(n_q)
```

Predicted (from the state, Rotella eq. 23-24):

```
ŝ_p,i = C·(p̂_i − r̂)
ŝ_z,i = q̂ ⊗ ẑ_i⁻¹
```

Innovations, with `log()` the SO(3)→so(3) map (his eq. 5):

```
e_p,i = s_p,i − ŝ_p,i
e_z,i = log( s_z,i ⊗ ŝ_z,i⁻¹ )
```

Measurement Jacobian per foot, in the state ordering above (his §V-B, with our block layout):

```
        δr     δv    δφ                δp_i   δθ_i                δb_f  δb_ω
H_p =[ -C      0     (C(p_i − r))ˣ     C      0                   0     0   ]
H_z =[  0      0     I                 0      −C[q̂ ⊗ ẑ_i⁻¹]       0     0   ]
```

with zero blocks for the *other* foot. Contact switching is handled without extra models: when a
foot lifts, drop its two measurement rows and inflate `w_p,i` / `w_z,i`, which lets its pose
uncertainty grow and makes the pose snap to the new foothold when contact returns.

### F.3 The equivalent algebraic form (bridge to the estimator already deployed)

For intuition, and because it is exactly what the current code computes: a stationary contact point
means v_F^W = 0, so in the IMU frame

```
0 = v_I^I + ω^I × p_F^I + v_F,rel^I     ⇒     v_I^I = −(J_v^I·q̇ + ∂p_F^I/∂ψ·ψ̇) − ω^I × p_F^I
```

The deployed estimator is this with the Jacobian term replaced by a finite difference:
`v = −(p_stance − p_stance_prev)/dt − ω_P × p_stance` (`base_state.h` `leg_odom_velocity`), computed
in the **pelvis** frame because leg FK is pelvis-referenced. ✅ The two agree to first order. The EKF
version replaces this one-shot algebraic inversion — which has no memory, trusts every encoder
sample equally, and is why differencing rate mattered so much (50 Hz measured 2.4x worse than 1 kHz)
— with a recursive estimate that fuses the same information against the IMU.

---

## G. Rotella mapped onto the H1-2

### Transfers unchanged ✅

* The entire prediction model (eq. 7-12) and the flat-foot orientation extension (eq. 13).
* Both measurement equations (14, 15) and their linearisations, including H above.
* The error-state / quaternion machinery: exponential map, error rotation vector φ ∈ so(3),
  innovation via `log(s ⊗ z⁻¹)`, state update `q̂⁺ = exp(Δφ) ⊗ q̂⁻` (eq. 3-6).
* Contact switching by noise inflation rather than model switching (his §IV).
* The discretisation and covariance propagation of his §V.
* The observability conclusion: absolute position and yaw stay unobservable. Irrelevant here — the
  goal is base *velocity* during standing and stepping in place.

### Needs adaptation ⚠️

1. **"r is the position of the IMU (assumed to be located at the base)" is false on the H1-2.** The
   IMU is on `torso_link`, one revolute joint (the waist) plus a 0.278 m lever from the pelvis.
   **Recommended resolution: define the filter's body frame as the torso/IMU frame.** Then every
   Rotella equation holds verbatim with no transport terms, and the entire adaptation is confined to
   the FK that produces the measurement (§E.1: the `Rz(−ψ)` and `−r_imu` in `p_F^I`). Convert to the
   pelvis at the *output* with `v_P^P = Rz(ψ)·(v_I^T − ω_T × r)` if a pelvis velocity is what the
   consumer wants.
   The alternative — propagating the pelvis — requires
   `a_P = Rz(ψ)·[f̃ − (α_T × r + ω_T × (ω_T × r))]` plus waist terms, and α_T means differentiating
   the gyro. Not worth it.
2. **The waist joint sits inside the IMU→foot chain.** The measurement function and H therefore
   depend on ψ, which Rotella's leg-only chain has no analogue for. ψ is *commanded* to hold on
   hardware (`hold_joint_ids: [12..26]`, `deploy_real.yaml:32`) but a PD hold is not a lock: read the
   encoder, and fold its uncertainty into `n_p` (a 0.01 rad ψ error displaces a foot ~1 mm laterally
   at nominal stance).
3. **Contact detection has no force sensor.** ✅ Verified against the SDK header: `LowState_` contains
   `version[2], mode_pr, mode_machine, tick, imu_state, motor_state[35], wireless_remote[40],
   reserve[4], crc` — **no foot force/FT field** (`unitree_sdk2/include/unitree/idl/hg/LowState_.hpp`).
   ✅ `SportModeState_` *does* carry `foot_force[4]`, `foot_position_body[12]` and
   `foot_speed_body[12]`, but it is **not available on the real robot at all** (user-confirmed
   2026-08-11; consistent with `State_RLHRL.cpp:150-165`, where `rt/sportmodestate` reads identically
   zero once our controller has command). It is a sim-bridge-only signal and must not be counted as a
   contact source. Rotella's platform had ankle F/T. What is left, in order of expected quality:
   * ✅ `MotorState_` exposes `tau_est` (same header dir), so an ankle/knee torque-based normal
     force via Jᵀ is *available in principle* — ⚠️ untested, and its noise/latency are unknown.
   * The geometric heuristic already deployed (stance = foot further along projected gravity), which
     was measured to lose ~nothing against a perfect oracle: 86% of steps are single support, flight
     is 0.13%.
   * The policy's own gait phase clock, which is a command rather than an observation — usable for
     stepping in place, dishonest as a sensor.
4. **Attitude double-counting.** ⚠️ Unitree already publishes a fused `imu_state.quaternion`. Rotella
   estimates attitude from raw gyro + accel. Using both without care double-counts the same
   information. Decide explicitly: either the EKF owns attitude (gyro + accel only, their quaternion
   used for nothing but sanity checks) or their quaternion enters as an attitude measurement with
   inflated, honestly-tuned noise. Cannot be settled from the files — see §H item 3.
5. **Bias states are new to this stack.** Nothing in the deploy path estimates `b_f` / `b_ω` today;
   gravity removal leans on Unitree's attitude. The EKF introduces them, and they need initial
   covariances from a real static log (§H item 4).
6. **FK bias is a first-class error source here in a way it is not in the paper.** Encoder-zero
   offsets (`joint_offset`, ADR-0006 Spec B) are currently empty in `deploy_est.yaml` ⇒ all zeros,
   and there is an open, unexplained backward lean on the real robot whose leading hypothesis is a
   leg re-zero. Any such offset enters `s_p` directly as a bias, and the filter will happily
   attribute it to velocity.
7. **The target regime is the easy one, and that has a sting.** Standing and stepping in place means
   long, continuous double support — the best case for a kinematics-corrected filter. But note the
   prior finding that the *consumer* of this estimate (the HL, on a c = 8 average) sees a nearly
   flat error across speeds (1.55x from command 0 → 1.0 m/s) while the per-step error spans 7.6x.
   Score the filter on the averaged rung as well as per-step RMS; the two have disagreed in direction
   before.

---

## H. Assumptions and open questions to resolve before implementing

Ordered by how much damage getting it wrong does.

1. ✅ **RESOLVED — which URDF matches the physical robot's IMU.** Two mutually inconsistent H1-2 URDF
   families sit in this workspace, both with `imu_link` parented to `torso_link`:
   * `(-0.04452, -0.01891, 0.27756)` — `/opt/unitree_mujoco/.../h1_2_handless.urdf`, the ROS2
     `h1_description/urdf/h1_2.urdf`, all `*_handless*` variants, **and our training MJCF**.
   * `(-0.04233868314, 0.00166, 0.152067)` — `unitree_rl_gym/resources/robots/h1_2/h1_2.urdf`,
     `h1_2_12dof.urdf`, `h1_2_simplified.urdf`, and the corelllab / RMA / legged_gym copies.

   A **125.5 mm disagreement in IMU height**, not a compensating frame difference. A full diff of the
   two families says it is a *description revision*, not a different robot: of 25 shared joints only
   **3** differ — `imu_joint`, and the two `wrist_pitch` joints (2 mm, and re-parented from
   `wrist_roll` to an older `elbow_roll` link name) — while all 24 shared links have **identical**
   mass and COM. **Use 0.27756**, on two independent grounds:
   * Provenance: it is what Unitree's own simulator model carries *and* what the vendor ROS2
     `h1_description` for this robot carries. The 0.152067 value appears only in the `unitree_rl_gym`
     training-repo lineage with the older arm naming.
   * Measurement: the deploy bench improved v_y error from 0.227 to 0.025 m/s when the lever-arm
     correction was applied **with r_z = 0.27756** (`base_state.h:18-22`). Since the v_y correction is
     dominated by −ω_x·r_z, a true r_z of 0.152 would have made that an 83% over-correction; a ~9x
     error reduction is not consistent with it.

   ❌ **It cannot explain the backward lean, and that hypothesis is already closed.** Three
   independent reasons: (a) *physics* — on a rigid body the gyro reads the same ω everywhere and, at
   rest, the accelerometer reads only gravity rotated into the sensor frame, so attitude is
   **independent of where the IMU sits**; the transport terms ω̇ × r + ω × (ω × r) are identically
   zero when standing still; (b) the 2026-07-28 **inclinometer session physically measured** pelvis
   pitch at 6.80° against 6.30° reported, agreeing to 0.50° ⇒ the IMU is honest (ADR-0006 Amendment);
   (c) "IMU mounting offset" is already listed as FALSIFIED (`A1a_deploy_plan.md:62`). The lean's
   established causes are the leg-encoder zero error and the leg-mass/knee-droop gap — see item 8.
2. ⚠️ **IMU units and quaternion order from an authoritative source.** Not documented in
   `IMUState_.hpp`. **Needed:** the Unitree SDK/IMU documentation, or a 5-minute bench test —
   ‖accel‖ at rest (expect 9.81 ⇒ m/s²), a slow known 90° yaw (confirms w-first and the sign), and a
   known-rate turn (confirms rad/s).
3. ⚠️ **Is `imu_state.quaternion` a fused estimate, and does its yaw drift?** Determines whether it
   can be used as a measurement at all (§G item 4). **Needed:** Unitree documentation, or a static
   log long enough to watch yaw walk.
4. ⚠️ **IMU noise and bias-stability parameters.** Nothing on disk. **Needed:** a ≥10 min stationary
   log at full rate → Allan variance for `w_f, w_ω, w_bf, w_bω`. Without it the process covariances
   are guesses.
5. ⚠️ **True sample timing — and the DDS rate is ~500 Hz, not 1 kHz.** ✅ Measured on an undecimated
   capture (`2026-07-16_09-18-04`, 6 significant figures): the accelerometer column changes at
   **494 Hz**, so `LowState` itself arrives at about 500 Hz while `run()` loops at ~990 Hz and reads
   each message roughly twice (KB `2026-08-10-kf-estimator-gates`). Treat 500 Hz as the sensor rate
   and 1 kHz as the loop rate; they are not the same number. `LowState_.tick` is a `uint32_t` with no
   documented unit. **Needed:** confirm the tick unit, or timestamp on arrival and accept the jitter.
6. ⚠️ **Contact detection method** — undecided (§G item 3). **Needed:** a stepping-in-place log with
   `tau_est` on the ankle and knee joints to see whether a Jᵀ-based normal force is usable.
7. ⚠️ **The sole plane on the real robot.** *Measured:* all five models on this machine put the sole
   at −0.045 in the ankle_roll frame (training capsules, `/opt` `h1_2_handless.xml`,
   `h1_2_handless_stress.xml`, `scene.xml`, our `scene_h1_2.xml`) — which **contradicts
   `hrl/base_state.h:28-33`**, where the bridge's sole is claimed to sit at −0.058. That claim does
   not reproduce: an AABB-based measurement of the mesh geom gives −0.138, not −0.058. **Needed:**
   either a re-derivation of the −0.058 figure or its retraction; plus the real sole pad thickness
   and its compression under load.
8. ⚠️ **Encoder zero offsets — a per-session quantity, not a constant.** This is the single largest
   threat to the measurement model. ✅ ADR-0006: `joint_offset` (Spec B) is **CONFIRMED as the dominant
   lever** on the stand — it moves the robot from +5.5° back to −4.0° forward — but **the required
   value is not reproducible between sessions**, so it must ship as a per-session calibration and
   never as a committed constant. `joint_offset` is currently absent from `deploy_est.yaml` (all
   zeros). Consequences for the EKF: an encoder-zero error enters `s_p` as a *bias* that the filter
   will happily attribute to velocity, and it **changes between power cycles**. Either run a stand
   calibration before each session, or carry per-joint offset states and accept the observability
   cost. Related: ~0.9° of the lean is a sim-side foot-sole wedge, and leg-mass-only was falsified as
   a fix (V1, 2026-07-31).
9. ⚠️ **Model vs real mass**: 66.984 kg model against 73.7 kg measured, with the leg links the known
   suspects (ADR-0006 Model v3). Kinematics are unaffected, but it is a live signal that the model
   and the robot have diverged; if link *lengths* have too, FK bias follows.
10. ⚠️ **Where the EKF runs.** The deploy controller is C++/DDS direct, colcon-ignored; a separate
    full ROS2 stack (`/opt/mybotshop_*`, `h1_platform`, `h1_description`) also exists on this machine.
    ✅ Its URDF is *identical* to ours on all 31 shared joints, differing only by 26 hand joints and
    0.384 kg — so no frame conflict — but nothing in `h1_platform` was found publishing a
    `base_link`/`odom` tf or a `sensor_msgs/Imu`. **Needed:** the decision on which process owns the
    1 kHz loop; if it is ROS2, the tf frame naming and time source need pinning down too.

### Prior art in this project, and the one blocking data gap

✅ A decoupled linear KF **and** a Bloesch-style EKF with Rotella's foot-orientation constraint were
already prototyped offline (KB `raw/engineering-journal/data/2026-08-10-kf-estimator-gates/`:
`replay.py`, `ekf_sweep.py`, `ekf_decomp.py`). Recorded verdict was **"do not build yet"**: gate 1
capped the addressable share of the *HL-consumed* error at ~24%, gate 2 measured ~2%, and tightening
the foot-orientation constraint made drift worse (0.712 m with it off, 31.5 m at `sig_phifoot` 0.001).

⚠️ **Read that verdict with three caveats before it is treated as settled.** (i) It scores the
estimator on the HL-consumed `c = 8`-averaged rung, i.e. against the question "does this help the RL
policy" — which is *not* the question "is base velocity accurate during standing and stepping in
place". The averaging is known to compress exactly the differences being measured. (ii) Gate 1's
decomposition was computed in sim, where the README itself records the bias share as understated
(`fastbias_vy` 0.0002 sim vs +0.061 hardware). (iii) Gate 2 ran on a replay with 56.5 Hz cached joint
data and a **reconstructed** gyro, in which `|ω × p|` carries half the estimate's magnitude — so the
foot-orientation result in particular is a statement about degraded input, not about the constraint.

✅ **The blocker is concrete: no gyro has ever been logged.** The `EstSample` block (`est_acc[3]`,
`est_quat[4]`, `est_gyro[3]`, `est_q[13]`, `est_dpsi` at the DDS rate, full float precision) landed
in `safety_logger.h` on 2026-08-10, but **no hardware session on disk carries it** — the newest
(`logs/deploy_safety/2026-08-10_15-06-26`) still has 108 columns with `quat_*` and `acc_*` and no
`est_gyro`. One standing session recorded with `with_estimator` enabled is the prerequisite for any
honest offline evaluation of a Rotella filter.

### Verified with no open question ✅

* URDF ↔ training MJCF: **identical** — all 27 joint origins, rpy, axes and limits match to
  ≤ 1.03e-6 rad (quaternion rounding in the MJCF), all link masses and COMs match exactly, totals
  agree at 66.984 kg. The URDF adds only `imu_link`, `logo_link`, `camera_link`, `lidar_link`, which
  the MJCF folds away.
* ROS2 `h1_description` URDF ↔ `/opt` handless URDF: **0 differing joints** of 31 shared.
* FK chain and its Jacobian: verified against MuJoCo and finite differences (§E).
* IMU is torso-mounted with axes = torso axes, gravity included, in all three descriptions.

---

## 9. Re-deriving these numbers

```bash
conda activate unitree_mjlab_h1_2_rl
# frame tree, sites, masses:      mujoco.MjModel.from_xml_path("src/assets/robots/unitree_h1_2/xmls/h1_2.xml")
# sole geometry:                  mesh_vert of *_ankle_roll_link @ mesh_quat/mesh_pos, then scipy ConvexHull
# FK / Jacobian check:            src/tasks/velocity/rl/hrl/leg_odom.py foot_site_b vs d.site_xpos
# existing estimator + bench:     python scripts/play.py <TaskID> --checkpoint-file <pt> --check-leg-odometry
```

Deploy-side transcription of the same chain: `deploy/robots/h1_2/include/hrl/base_state.h`
(`leg_fk`, `foot_site_b`, `leg_odom_velocity`, `lowest_foot_z`).
