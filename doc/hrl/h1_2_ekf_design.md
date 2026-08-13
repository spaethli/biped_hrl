# H1-2 base-velocity EKF — mathematical model and architecture

Design only, no implementation. Objective: **base linear velocity during standing and stepping in
place**, on the real robot. Incumbent: the deployed encoder-only leg odometry
(`hrl::leg_odom_velocity`).

Frames, link geometry, contact-frame definition and the verified FK/Jacobians come from
[`h1_2_kinematics.md`](h1_2_kinematics.md) and are not re-derived here. Rotella's equations are
cited from [arXiv:1402.5450](https://arxiv.org/abs/1402.5450) by their numbering.

**Why this gets built regardless of the experiment (agreed 2026-08-11).** A2 needs online state
estimation whatever happens here, so §10 does **not** decide whether the EKF exists — it decides
whether it *replaces the incumbent in A1 deploy*, and sizes what fusion actually buys on this
platform. A negative result is a finding, not a failure, and the pre-registered bar exists so that
finding is honest either way.

**Headline verdict, argued in §4 and §9:** the *filter core* is well-motivated here for a reason that
has nothing to do with simulation scores — **the incumbent differentiates encoders at 500-1000 Hz and
multiplies the raw gyro by a ~1 m lever, and the EKF does neither.**

> ### STATUS 2026-08-12 — VERDICT, 3 of 3 sessions: **C fails the bar, ship B**
>
> P0 landed and verified on hardware: `est_dq[13]` plus an **A0** `EstSample` (A0 is the only
> policy that stands genuinely still, hence the only exact truth v = 0). Joints now log at
> **484-501 Hz**, not the 50 Hz cache. Measured on a 655 s hang: `sigma_q` 1.64e-5, `sigma_psi`
> 2.61e-5 rad, gyro/acc densities 0.0016 / 0.0145 per sqrt(Hz), and **|a| at rest = 9.9030**
> (0.95% above the 9.81 used for gravity removal).
>
> Three A0 stand+walk sessions, scored on the verified-still segment only (45.9 / 47.2 / 70.6 s,
> truth v = 0), **std vx | drift x in m/s | m**:
>
> | arm | `10-16-41` | `10-41-08` | `10-43-34` |
> |---|---|---|---|
> | A legodom (incumbent) | 0.0130 \| -0.067 | 0.0206 \| 0.041 | 0.0156 \| 0.020 |
> | **B complementary** | **0.0069 \| -0.083** | **0.0062 \| -0.039** | **0.0050 \| -0.056** |
> | C ekf (position-only) | 0.0263 \| **6.349** | 0.0111 \| 0.239 | 0.0089 \| 0.666 |
> | C+grav | 0.0329 \| 0.038 | 0.0069 \| 0.012 | 0.0059 \| 0.115 |
> | D ekf+rot | 0.0134 \| 0.238 | 0.0094 \| 0.092 | 0.0079 \| 0.106 |
> | E jacobian | 0.0116 \| -0.107 | 0.0170 \| 0.059 | 0.0121 \| 0.110 |
> | F ekf+att | 0.0080 \| 0.064 | 0.0083 \| 0.106 | 0.0064 \| 0.114 |
>
> **Bar (§12): C must beat both A and B on ≥2 of 3.** C beats B on **0 of 3**, and on drift — the
> tie-breaker — loses to *both* on **3 of 3**. C therefore fails, and the pre-registered
> consequence applies: **ship B for A1, carry C forward as A2 groundwork.** B wins std on 3/3 at
> a bias indistinguishable from the incumbent's, and it is a true complementary filter
> (high-passed IMU + low-passed leg odometry), so its walking bandwidth is preserved by
> construction rather than bought with lag. The EKF arms trade the other way: D and F cut noise
> ~2x versus A but carry **more** bias than A on 3/3.
>
> Second-order reads: the **orientation block is what makes the EKF safe** — bare C is worst on
> drift in all three and diverges outright in one; D, F and C+grav all cure it. **A is already
> good** on a genuinely still robot (std 0.013-0.021), so fusion buys noise, not accuracy.
>
> ⚠️ Corrections that generalize: **`cmd == 0` is not proof the robot is still** — a 5 s set-down
> transient at the head of `10-16-41` moved the feet 137 mm and alone carried that session's
> apparent noise (A std vx 0.141 pooled vs **0.013** on the still segment), which is why the
> earlier 1-session numbers here were far larger and are now superseded. `replay_base_estimators.py`
> verifies stillness from the encoders and reports what it dropped. Also: **never score across
> mixed regimes** (a pooled "E is the standout" result was error cancellation), and
> **PROVENANCE.json cannot establish what ran** when ONNX files are swapped by hand — `mv`
> preserves mtime, only the directory mtime records the swap. Full narrative: the deploy journal
> (research KB), 2026-08-11/12.

---

## 0. Frames used at each stage

| stage | quantity | frame | source |
|---|---|---|---|
| input | f̃ specific force, ω̃ angular rate | **torso** (= IMU axes) | `imu_state.accelerometer/gyroscope`, ~500 Hz |
| input | attitude (optional, §7.4) | torso in world | `imu_state.quaternion`, (w,x,y,z) |
| input | q[0..11] leg, ψ = q[12] waist | joint coords, **true** = reported + `joint_offset` | `motor_state[]` |
| state | r, v, p_L, p_R | **world** | filter |
| state | q_WI, z_L, z_R | world → torso / world → foot | filter |
| state | b_f, b_ω | **torso** (sensor-fixed) | filter |
| measurement | s_p,i, s_z,i | **torso/IMU frame** | FK, §3 |
| internal FK | p_F^P, R_PA | **pelvis** | `leg_fk` (`hrl/base_state.h`) |
| output | v_I^W → v_I^T | world → torso | `C·v` |
| output | v_P^P (what the goal space wants) | **pelvis** | `Rz(ψ)·(v_I^T − ω_T × r_imu)` |

The body frame of the filter is the **torso/IMU frame**, never the pelvis. That choice is what lets
every Rotella equation transfer unchanged (§9.1 of the kinematics doc); the pelvis conversion happens
once, at the output.

---

## 1. Complete EKF state

Following Rotella's §IV with N = 2 flat feet, i ∈ {L, R}.

```
nominal   x  = [ r,  v,  q_WI,  p_L,  p_R,  z_L,  z_R,  b_f,  b_ω ]        dim 30
error     δx = [ δr, δv, δφ,    δp_L, δp_R, δθ_L, δθ_R, δb_f, δb_ω ]  ∈ R^27 = 15 + 6N
```

| symbol | dim | frame | meaning |
|---|---|---|---|
| `r` | 3 | world | position of the **IMU site** (torso frame origin + r_imu), not the pelvis |
| `v` | 3 | world | its linear velocity — **the quantity of interest** |
| `q_WI` | 4 | — | attitude, world → torso. `C := C(q_WI)` maps **world → body** |
| `p_i` | 3 | world | position of contact frame `F_i` = (0.04, 0, −0.045) in `*_ankle_roll_link` |
| `z_i` | 4 | — | attitude, world → foot frame `F_i` |
| `b_f` | 3 | torso | accelerometer bias |
| `b_ω` | 3 | torso | gyroscope bias |

Attitude errors are 3-vectors in so(3) via the exponential map (Rotella eq. 3-6): `q = exp(δφ) ⊗ q̂`,
`z_i = exp(δθ_i) ⊗ ẑ_i`. The covariance is on the 27-dim error state throughout.

**Deliberately excluded, with reasons.** Foot *contact-point* offsets (absorbed by `p_i`); joint
offsets as states (⚠️ tempting given the per-session encoder-zero problem, but they are only weakly
separable from `p_i` — see §8.4); IMU scale factors and non-orthogonality (unidentifiable at this
excitation level); ground-plane parameters (not needed, the feet carry it).

---

## 2. IMU prediction model

Continuous (Rotella eq. 7-13), with the H1-2 measurement conventions verified in the kinematics doc —
f̃ is in the **torso frame and includes gravity**, ω̃ is in the torso frame, g = (0, 0, −9.81)ᵀ in world:

```
ṙ    = v
v̇    = Cᵀ (f̃ − b_f − w_f) + g
q̇    = ½ · [ ω̃ − b_ω − w_ω ; 0 ] ⊗ q
ṗ_i  = Cᵀ w_p,i                        ← "a contacting foot does not translate"
ż_i  = ½ · [ w_z,i ; 0 ] ⊗ z_i         ← "a contacting foot does not rotate"
ḃ_f  = w_bf          ḃ_ω = w_bω        ← random-walk biases
```

Discrete, first order with the second-order position term (eq. 16-22), at the **sensor** rate
`Δt ≈ 1/500 s` — not the loop rate; `run()` reads each DDS message about twice, so either decimate to
new samples or use the true arrival Δt. Writing `f̂ = f̃ − b̂_f`, `ω̂ = ω̃ − b̂_ω`:

```
r̂⁻   = r̂⁺ + Δt·v̂⁺ + (Δt²/2)·(Ĉᵀ f̂ + g)
v̂⁻   = v̂⁺ + Δt·(Ĉᵀ f̂ + g)
q̂⁻   = exp(Δt·ω̂) ⊗ q̂⁺
p̂_i⁻ = p̂_i⁺        ẑ_i⁻ = ẑ_i⁺        b̂⁻ = b̂⁺
```

Error dynamics `δẋ = F_c δx + L_c w`, in the ordering of §1:

```
δṙ    = δv
δv̇    = −Cᵀ f̂ˣ δφ − Cᵀ δb_f − Cᵀ w_f
δφ̇    = −ω̂ˣ δφ − δb_ω − w_ω
δṗ_i  = Cᵀ w_p,i
δθ̇_i  = w_z,i
δḃ_f  = w_bf        δḃ_ω = w_bω
```

`F_k ≈ I + F_c Δt`, `Q_k ≈ F_k L_c Q_c L_cᵀ F_kᵀ Δt` (eq. 32-33 truncation). `Q_c = diag{Q_f, Q_ω,
Q_p,L, Q_p,R, Q_z,L, Q_z,R, Q_bf, Q_bω}`.

⚠️ **`Q_f, Q_ω, Q_bf, Q_bω` cannot be guessed.** They are the IMU's own noise and bias-instability
parameters and nothing on disk documents them. Get them from a ≥10 min stationary log (Allan
variance) *before* tuning anything else, or every subsequent comparison is a tuning artifact.

`Q_p,i` and `Q_z,i` are **not** IMU properties — they encode "how much may a contacting foot move",
and §7 makes them a function of contact confidence rather than constants.

---

## 3. Stationary-foot measurement model

Per foot in contact. The measurement is produced by forward kinematics; the prediction comes from the
state. This is the H1-2-specific piece, because Rotella's IMU is at the base and ours is not.

**Measured** (encoders + the verified chain, kinematics doc §E.1):

```
s_p,i = p_{F,i}^I (q_i, ψ) = Rz(−ψ) · p_{F,i}^P(q_i) − r_imu                 + n_p,i
s_z,i = quat( R_{IF,i} )   = quat( Rz(−ψ) · R_{PA,i}(q_i) )                  ⊗ exp(n_q,i)
```

with `p_{F,i}^P` the pelvis-frame FK to the contact frame (contact offset `c_A = (0.04, 0, −0.045)`)
and `r_imu = (−0.04452, −0.01891, 0.27756)`. Verified against MuJoCo to 6.7e-16 m.

**Predicted** (Rotella eq. 23-24):

```
ŝ_p,i = Ĉ (p̂_i − r̂)
ŝ_z,i = q̂ ⊗ ẑ_i⁻¹
```

**Innovations** (eq. 5):

```
e_p,i = s_p,i − ŝ_p,i                    ∈ R³
e_z,i = log( s_z,i ⊗ ŝ_z,i⁻¹ )           ∈ R³
```

### 3.1 Where the velocity information actually comes from

There is **no explicit velocity measurement anywhere in this filter**, and that is the design's whole
point. Stationarity lives in the process model (`ṗ_i = Cᵀ w_p,i`); velocity becomes observable because
`r` and `p_i` are coupled through `e_p,i` over successive samples. The equivalent algebraic statement
— the thing the incumbent computes directly — is

```
v_I^I = −( J_v^I q̇ + ∂p_F^I/∂ψ · ψ̇ ) − ω^I × p_F^I
```

Note the two terms the EKF does **not** contain: a **derivative of the encoders**, and a **cross
product of the raw gyro with a ~1 m lever**. §9 is the consequence.

### 3.2 Measurement noise, built rather than guessed

`n_p,i` is dominated by encoder noise propagated through the kinematics, so build it from the FK
Jacobian (kinematics doc §E.2) rather than tuning a scalar:

```
R_p,i = J_{v,i}^I · Σ_q · (J_{v,i}^I)ᵀ  +  σ_slip²·I  +  σ_model²·I
R_z,i = J_{ω,i}^I · Σ_q · (J_{ω,i}^I)ᵀ  +  σ_rot²·I
```

`Σ_q` = per-joint encoder variance (7×7: six leg joints + waist). `σ_model` covers link-length and
sole-geometry error; `σ_slip` covers real foot motion and is the term §7 modulates.

⚠️ **Rotella assumes the two feet's measurements are uncorrelated; on the H1-2 they are not.** Both
Jacobians contain the **same waist column** (ψ is shared), so `cov(n_p,L, n_p,R) = J_{ψ,L} σ_ψ² J_{ψ,R}ᵀ
≠ 0`. Do not silently assume independence.

✅ **Implemented** (`scripts/replay_base_estimators.py`, `build_R`). Rather than assembling the blocks
by hand, the joint-space covariance is propagated through **one stacked kinematic Jacobian**
`G ∈ R^{m×13}` (12 leg joints + waist):

```
R = G Σ_q Gᵀ  +  diag(σ_slip², σ_model², σ_rot²)  +  contact inflation
```

which produces *both* couplings automatically — cross-foot through the shared waist column, and
position/orientation within a foot through the shared encoder noise. The stacked form is then used in
a **single both-feet update**; updating one foot at a time would discard the off-diagonal blocks
regardless of how carefully they were built. Self-tested for symmetry, positive-definiteness, and that
the cross-foot block is non-zero and **vanishes exactly when σ_ψ = 0**.

*Measured magnitude:* with the provisional σ_ψ = 1 mrad the cross-foot term is ≈ 3.8e-8 m² (≈ 0.2 mm),
about 15x below the σ_slip / σ_model diagonal. So modelling it is a correctness matter, not currently a
large numerical effect — which changes if the measured waist-encoder noise turns out larger.

⚠️ `σ_q`, `σ_psi` are **provisional** in the code and flagged as such. They are measurable: take the
per-joint std over the static Allan window (§10 P0), which the same session provides.

---

## 4. The flat-foot orientation constraint — is it appropriate here?

> **SUPERSEDED 2026-08-12 by hardware.** The recommendation below ("do not enable it in the
> baseline") was wrong, and the reasoning that produced it was only half wrong. §4.1(1) is still
> correct — the orientation block does not improve *velocity* observability. But on hardware the
> position-only core **diverges**: 72.8 deg tilt error, 0.117 m/s standing bias, 5.7 m drift over
> 49 s, and the wrong sign in commanded backward walking. In the strapdown formulation the
> accelerometer is an input, never a measurement, so attitude is corrected only through
> `(C(p-r))x dphi`, which is far too weak here — and attitude error feeds velocity through gravity
> projection. **The orientation block is load-bearing because it pins ATTITUDE, not velocity.**
> Arm D is stable exactly where C is not. The alternative fix is `update_gravity()` (accelerometer
> as a rank-2 tilt observation, gated on ||f|-g|<0.5, in neither Rotella nor Bloesch), which cures
> C to parity. **Enable one of the two; a bare position-only core is not viable on this robot.**

**Original recommendation (superseded, kept for the reasoning): implement it, keep it behind a
switch, and do not enable it in the baseline configuration for this objective.** The reasoning is
structural, not empirical.

### 4.1 What it can and cannot buy

The constraint (`ż_i = ½[w_z,i;0] ⊗ z_i` plus the `s_z,i` measurement) asserts that a contacting foot
holds its **orientation**. Three observations decide its value here:

1. **It does not improve velocity observability, because velocity is already fully observable without
   it.** A single stationary contact point plus the IMU makes `v` observable (§8); the unobservable
   subspace with ≥1 contact is absolute position and yaw, and *the orientation constraint does not
   shrink it* — global yaw stays unobservable because `z_i` is itself a free state referenced to the
   same unobservable world yaw. What the constraint adds is information about the **foot pose**, which
   is a nuisance state for this objective.
2. **Its documented benefit is for degenerate point-foot geometry, which standing does not have.**
   Rotella motivates the extension by singular cases of the point-foot filter. In double support —
   85% of samples in the logged keeper session — the H1-2 presents **two contact points 0.33 m apart**,
   which is precisely the non-degenerate case. The extension is aimed at a problem this phase does not
   exhibit.
3. **It encodes an assumption this gait violates exactly when it matters.** The foot is 259 mm long;
   touchdown and toe-off roll it about heel and toe edges. During those windows `ż_i ≈ 0` is false,
   and a *tight* `Q_z` turns a modelling error into a state bias rather than into extra noise. The
   failure is asymmetric: too-loose costs nothing (the measurement simply carries little weight), and
   too-tight is catastrophic.

### 4.2 The honest counter-argument

In **single support** the constraint is doing real work: one contact point leaves the foot's rotation
about that point unpinned, so an orientation measurement genuinely adds information that the position
measurement cannot. If stepping in place turns out to spend meaningful time in single support with a
*flat* foot, the orientation block should help there. That is a testable proposition, not a settled
one, which is why it belongs behind a switch rather than deleted.

⚠️ **What I have not verified:** Rotella's detailed observability derivation (their §V-D). I have read
the model and measurement sections directly; the claim "rotational constraints improve observability"
is theirs, and my argument above is that whatever it improves, it is not the velocity channel in
double support. If the exact unobservable subspace of their flat-foot filter matters to the decision,
that section needs reading in full.

### 4.3 Prior evidence, and how much weight it deserves

An offline sweep in this project (KB `2026-08-10-kf-estimator-gates/ekf_sweep.py`) found drift of
0.712 m with the orientation block off against 31.5 m at `σ_φfoot` = 0.001, with 0.5–10 behaving like
off. That is consistent with §4.1(3) — but it ran on a replay with 56.5 Hz cached joint data and a
**reconstructed** gyro, so treat it as corroboration of the shape of the failure, not as a measurement
of the constraint's value.

---

## 5. Measurement Jacobians

In the §1 ordering `[δr, δv, δφ, δp_L, δp_R, δθ_L, δθ_R, δb_f, δb_ω]`, per foot i, with `δ_{ij}` the
block selector:

```
            δr    δv    δφ                 δp_L        δp_R        δθ_L         δθ_R         δb_f  δb_ω
H_p,i = [  −C     0     (C(p_i − r))ˣ      C·δ_{iL}    C·δ_{iR}    0            0            0     0    ]
H_z,i = [   0     0     I                  0           0          −C[q̂⊗ẑ_L⁻¹]·δ_{iL}  −C[…]·δ_{iR}  0  0 ]
```

`C[m]` = rotation matrix of quaternion m; `(·)ˣ` = skew. Stacked for both feet in double support this
is 12×27 (or 6×27 position-only), 6×27 in single support (3×27 position-only), empty in flight.

Discrete measurement covariance `R_k ≈ R_c / Δt` (Rotella §V-C), with `R_c` from §3.2. The FK
Jacobians `J_v^I`, `J_ω^I` from the kinematics doc §E.2 appear **only** in building `R`, never in `H` —
a useful check that the implementation has not confused the two.

Standard error-state update: `K = P⁻Hᵀ(HP⁻Hᵀ+R)⁻¹`, `Δx = K e`, additive for all non-rotational
blocks, `q̂⁺ = exp(Δφ) ⊗ q̂⁻` and `ẑ_i⁺ = exp(Δθ_i) ⊗ ẑ_i⁻` for the rotational ones (eq. 6), then
`P⁺ = (I−KH)P⁻(I−KH)ᵀ + KRKᵀ`.

---

## 6. Contact phases

Rotella's mechanism is deliberately *not* a mode switch: the model is identical in all phases, and
only the noise and the active measurement rows change. Keep that.

| phase | active rows | `Q_p,i`, `Q_z,i` | notes |
|---|---|---|---|
| **double support** | both feet | nominal | dominant phase for both objectives; redundancy is what reveals foot slip (the two feet disagree) |
| **single support** | stance foot only | stance nominal; **swing inflated** | swing-foot pose uncertainty must grow, otherwise the stale pose fights the next touchdown |
| **flight** | none | both inflated | pure IMU integration; rare here (0.13% of steps measured) but must not special-case |
| **touchdown** | new foot re-enters | — | inflated covariance makes the first measurement snap the pose to its new foothold; **no explicit reset** |
| **liftoff** | foot drops out | inflate before dropping | inflate *first*, then stop updating, so the covariance never jumps discontinuously |

Two H1-2-specific cautions:

* **Double support is over-determined and will fight.** Two feet asserted stationary also assert a
  fixed inter-foot vector. Any real slip, sole compression, or FK bias shows up as an inconsistency the
  filter must absorb somewhere — and with tight `R_p` it absorbs it into `v`. This is the most likely
  way a naively-tuned EKF ends up *worse* than the incumbent during standing. `σ_slip` and `σ_model`
  in §3.2 exist for exactly this.
* **Never hard-switch.** The incumbent picks a single stance foot by a geometric test and differences
  only that foot, explicitly refusing to difference across a switch. The EKF has no such restriction —
  but only if confidence is continuous (§7). A binary switch reintroduces the same transient.

---

## 7. Contact confidence

### 7.1 Mechanism

Maintain a continuous `α_i ∈ [0,1]` per foot and drive the covariances with it:

```
R_p,i(α_i) = R_p,i(nominal) + (1/α_i − 1)² · σ_slip² · I          α_i → 0 ⇒ measurement is ignored
Q_p,i(α_i) = Q_p,i(min) + (1 − α_i) · Q_p,i(free)                 α_i → 0 ⇒ pose free to move
Q_z,i(α_i) likewise
```

This is one continuous family, so a foot crossing in or out of contact traces a smooth path rather
than a step. Both knobs must move together: a foot whose measurement is discounted must also be
allowed to move, or the filter keeps a confidently-wrong stale pose.

### 7.2 Cues available on the H1-2 — and their standing

Ranked by information content, all ⚠️ untested for this purpose:

1. **Ankle/knee `tau_est` through Jᵀ** → an estimated normal force. `MotorState.tau_est` exists (✅
   SDK header). This is the only cue that is a genuine *force* measurement, and it is the closest
   available substitute for the ankle F/T Rotella's platform had. Unknown noise, latency, and
   whether `tau_est` on the H1-2 is trustworthy at all — validate against a session where contact is
   unambiguous before trusting it.
2. **Kinematic height** — the deployed heuristic: stance = the foot further along projected gravity.
   Cheap, no new plumbing, and measured to lose ~nothing against a perfect oracle in the regime that
   matters (86% single support, 0.13% flight). Its weakness is exactly at the transitions, where it
   is least certain and most consequential.
3. **Innovation gating** — a Mahalanobis test on `e_p,i` against `H P Hᵀ + R`. Not a contact detector,
   but the correct *safety net*: it catches slip and mis-detected contact regardless of cause. Should
   be present whatever else is chosen.
4. ❌ **`SportModeState.foot_force`** — the message carries `foot_force[4]`, but the topic is not
   available on the real robot at all. Not an option; noted because the IDL invites the mistake.
5. **Gait phase from the policy** — a command, not an observation. Usable as a prior for stepping in
   place, dishonest as a sensor; it will hide exactly the touchdown-timing errors worth finding.

**Proposal:** `α_i` = product of a smooth height term and (if validated) a smooth force term, with
hysteresis on the rising/falling edges, and innovation gating on top. Start with height + gating,
which needs no new hardware trust, and add the force term as an ablation.

### 7.3 The self-reference trap

Any cue derived from the filter's own state (foot height from `p̂_i`, or the innovation itself) closes
a loop: bad state → wrong contact → worse state. Gating is safe because it only ever *removes*
information; height-from-state is not. Prefer height computed from **FK relative to the other foot**,
which depends on encoders alone.

### 7.4 Whether to use Unitree's attitude at all

⚠️ Open decision, flagged in the kinematics doc. `imu_state.quaternion` is a fused estimate from an
internal filter. Feeding it as a measurement while also estimating `q_WI` from the same raw gyro and
accelerometer double-counts. Two defensible configurations — (a) the EKF owns attitude, their
quaternion used only for sanity checks; (b) their quaternion enters as an attitude measurement with
deliberately inflated noise.

**Decided 2026-08-11: not by argument — measure it.** Arm F in §10 runs configuration (b) against C's
(a) on identical data. ⚠️ With the caveat recorded there: F's attitude-measurement noise describes an
undocumented vendor filter and so cannot be grounded in the Allan log the way every other covariance
can. It must be swept, which makes F tuning-dependent and therefore **exploratory, never a gate**. My
prior remains (a) — gravity plus the contact constraint already make roll and pitch observable, and (b)
makes accuracy a function of a filter we cannot inspect — but the data decides.

---

## 8. Observability

The universal legged-EKF result: with **at least one foot in contact**, the unobservable subspace is
**absolute position (3) and yaw (1)**. Neither matters for base velocity. The interesting question is
not what is observable in principle but what is *well-conditioned* in each phase.

### 8.1 Standing / double support (static)

| quantity | status |
|---|---|
| `v` | **strongly observable** — two stationary points pin it; and the truth is exactly 0, which §10 exploits |
| `b_ω` gyro bias | **observable** — a gyro bias tilts the attitude, which makes a stationary foot appear to translate in the body frame; the position measurement catches it. Convergence ∝ lever arm × time |
| roll / pitch | observable via gravity — **but see below** |
| `b_f` horizontal | ⚠️ **degenerate with tilt** |
| `p_i`, `z_i` | observable relative to `r`, `q`; absolute only up to the global position/yaw offset |

⚠️ **The important structural fact for this application: in perfectly static standing, the horizontal
components of `b_f` are not separable from a roll/pitch error.** A stationary accelerometer measures
`−Cg`; a small tilt `δφ` and a body-fixed bias `δb_f` produce the *same* constant horizontal signature.
They separate only when the attitude changes, modulating the gravity projection differently from a
body-fixed bias. So **standing is the worst possible regime for accelerometer-bias estimation** —
expect `b_f` to wander along the degenerate direction, and either clamp `Q_bf` near zero during long
quiet stands or accept that the split is arbitrary.

This is an independent, structural confirmation of the prior project finding that the accelerometer-
bias state was inert over 8.44 s of standing. That result should be believed — not because of the
measurement, which was degraded, but because the degeneracy is real.

### 8.2 Single support

Same unobservable subspace, worse conditioning: one contact carries the whole geometric constraint,
and there is no second foot to arbitrate slip. Compensating advantage — the body pivots over the
stance foot, so attitude *changes*, which is exactly the excitation §8.1 says is missing. `b_f`
becomes better observable in single support than in quiet standing.

### 8.3 Stepping in place

The best regime of the three, for three reasons: repeated contact switching **re-anchors** the foot
states so their random walk cannot accumulate; torso pitch/roll oscillation supplies the attitude
excitation that breaks the tilt/bias degeneracy; and the true net displacement over a session is
≈ 0 and directly measurable, which turns the whole exercise into something falsifiable (§10).

Against that: every touchdown re-injects the FK bias afresh, and the touchdown/toe-off windows are
where the stationary-foot assumption (and, more sharply, the flat-foot orientation assumption) is
violated.

### 8.4 Why joint offsets should not be states

Tempting, given that the encoder-zero correction is a **per-session** quantity (ADR-0006 Spec B: the
required `joint_offset` moves the stand from +5.5° to −4.0° and is not reproducible between sessions).
But a constant joint offset and a constant foot-position error produce nearly the same measurement
signature, so `δq_offset` and `δp_i` are only weakly separable — the filter would trade them without
converging. The correct handling is a **stand calibration before each session**, with the residual
absorbed by `p_i`, which the filter already estimates.

⚠️ Note this cuts *for* the EKF: see §9.

---

## 9. Against the incumbent: where an EKF can actually help

The current estimator is `v = −(p_stance − p_stance,prev)/dt − ω_P × p_stance`, computed in the pelvis
frame at the physics rate, with a geometric stance choice and no memory.

| error source | encoder-only leg odometry | EKF | net |
|---|---|---|---|
| encoder noise / quantisation | **differentiated** at 500-1000 Hz — amplified by 1/dt | enters `R_p` undifferentiated; smoothed by the filter's own dynamics | **EKF, structurally** |
| gyro noise | multiplied by a ~1 m lever every sample (`ω × p`, measured to carry half the estimate's magnitude) | ω is **integrated** into attitude, never multiplied by a lever in the measurement | **EKF, structurally** |
| gyro **bias** | passes straight through as `b_ω × p` ≈ 1 m × bias, unestimated | `b_ω` is an observable state in every contact phase (§8.1) | **EKF** |
| constant FK bias (per-session encoder zeros) | enters `ω × p` ⇒ velocity **bias** ∝ angular rate; **measured below** | absorbed into the foot-position state `p̂_i`; a *constant* offset produces no steady velocity error | **EKF, but on bias only** |
| accelerometer bias | not used at all — immune | ⚠️ **new** error source, and poorly observable while standing (§8.1) | **incumbent** |
| stance switching | must not difference across a switch; hard geometric choice | continuous confidence, both feet weighted (§7) | **EKF** |
| foot slip / rotation | silently becomes velocity | detectable via innovation gating and inter-foot disagreement | **EKF** |
| latency / lag | the raw estimate is too noisy to use, so a `c = 8` box average is applied downstream — 25% of the consumed error variance by the project's own decomposition | a filter with a correct dynamic model replaces the box average | **EKF** |
| complexity, tuning surface, failure modes | ~20 lines, no tuning | 27 states, ~8 covariance blocks, contact logic | **incumbent** |

### 9.1 The FK-bias row, quantified — and downgraded

An earlier draft of this document leaned on the FK-bias row as a general argument. Measuring it says
it is a **bias** argument only, and a second-order one. *Measured* foot displacement per 1° of encoder
error at the nominal stand:

| joint | |δp| per 1° |
|---|---|
| hip pitch | **14.6 mm** |
| hip roll | 14.3 mm |
| knee | 7.8 mm |
| hip yaw | 1.3 mm |
| ankle pitch / roll | 1.3 / 0.7 mm |

Real angular rates, from the logged quaternion: median |ω| = **1.23 rad/s** in the bare-arm session
`2026-08-10_15-06-26` (stepping at zero command) and 0.37 rad/s in `13:25`; p99 3.5 rad/s. So
ADR-0006's 1.8° encoder-zero error, taken at the hip pitch, gives ≈ 26 mm of foot displacement and
**≈ 32 mm/s of velocity error** through `ω × δp` at the median rate.

Compare against the incumbent as logged: `lo_vx` mean **+0.0315** m/s, std **0.197** m/s at zero
command. The FK bias is therefore the *same order as the observed bias* and roughly **6x below the
observed noise**. Real, worth removing, but not the reason to build a filter — and the user's report
that the lean is not visually apparent on hierarchical policies is fully consistent with a 1-2°
offset, since 14.6 mm/° means the foot-position consequence is invisible to the eye.

**The honest summary:** the EKF's advantage is not exotic, and it is not the FK bias. It comes almost
entirely from *not differentiating encoders* and *not levering the raw gyro*, plus estimating the gyro
bias. Those are structural, visible in the measurement equations, and independent of any simulation
result.

**But this is precisely why the experiment in §10 must include a cheap middle arm.** A complementary
filter — accelerometer high-passed, leg odometry low-passed — captures the "stop differentiating and
stop box-averaging" benefit in a few lines, without 27 states or contact logic. If the EKF's gain over
*that* is small, the correct conclusion is "you needed a filter, not an EKF", and it is a legitimate
outcome. The prior gate-2 study compared a KF against **raw** leg odometry, which conflates the value
of filtering with the value of the specific estimator; that comparison cannot answer the question.

---

## 10. Experiment protocol on the real H1-2

**Agreed 2026-08-11.** What this experiment decides: whether the EKF **replaces the incumbent in A1
deploy**. It does *not* decide whether the EKF gets built — that is settled independently by A2, which
needs online estimation regardless (§0). A negative result here is therefore a finding about this
platform at this fidelity, not a failed project.

### P0 — prerequisite, blocking

**One logging change, covering both needs:**

1. Enable the `EstSample` block (`init(..., with_estimator=true)`) — `est_acc[3]`, `est_quat[4]`,
   `est_gyro[3]`, `est_q[13]`, `est_dpsi` at the DDS rate, full float precision. ✅ The code landed
   2026-08-10; ⚠️ **no session on disk carries it** — the newest (`2026-08-10_15-06-26`) still has 108
   columns and no gyro.
2. **Extend it with `est_dq[13]`.** `MotorState.dq` is already inside the same locked snapshot that
   `est_dpsi` reads (`State_RLHRL.cpp:617-643`), so this costs no extra DDS read, no extra lock and no
   extra rows — and without it **arm E cannot run at sensor rate**, only at the 50 Hz cache.

Verify with one 10 s capture that the columns actually appear before spending a robot session.
Without a logged gyro there is no honest offline evaluation of anything here: the incumbent's `ω × p`
term and the filter's attitude propagation both need it, and reconstructing ω from the 50 Hz cached
quaternion reproduces exactly the handicap that made the 2026-08-10 gate-2 study uninformative.

**Also record once:** a ≥10 min stationary IMU log (robot powered, motionless, no policy) for Allan
variance. `Q_f, Q_ω, Q_bf, Q_bω` come from it and are **frozen before any scoring**.

Meanwhile, dry-run the replay pipeline on existing logs — FK parity, contact detection, tuning ranges —
but **score no arms** until P0 lands.

### Design principle: one session, six estimators

Restarting the controller changes the encoder offset, so **across-session comparison is unreliable**.
Record sessions, then replay **all arms offline on byte-identical data**. This removes the robot, the
policy, the session and the offset as confounds, and costs nothing per additional arm. Per-session
offsets are **not** calibrated out (agreed: an A2 problem if it persists for a working A1) — legitimate
precisely because all six arms share the same bias within a session.

| arm | what it isolates | status |
|---|---|---|
| **A** deployed leg odometry, finite difference | the incumbent, as shipped | baseline |
| **B** leg odometry + complementary filter | "any filtering at all" | **gates adoption** |
| **C** EKF, position-only (Bloesch core), H1-2 contact frame | the proposed system | **primary** |
| **D** C + flat-foot orientation, honest `Q_z` | whether Rotella's extension earns its keep (§4) | secondary |
| **E** Jacobian leg odometry, `−(J q̇ + ∂p/∂ψ ψ̇) − ω × p` | where the differentiation happens | secondary |
| **F** C consuming `imu_state.quaternion` as an attitude measurement | the double-counting question (§7.4) | **exploratory only** |

⚠️ **Arm E does not remove differentiation, it relocates it** — `dq` is differentiated on the motor
board. Read its result as "our loop vs the motor controller", not as "derivative-free".

⚠️ **Arm F is weaker evidence than the rest and must not gate anything.** Its attitude-measurement
noise describes an undocumented vendor filter, so unlike every other covariance it cannot be grounded
in the Allan log — it has to be swept, which makes F's outcome tuning-dependent in a way C-vs-B is not.

Contact detection for C, D, F: **height heuristic + Mahalanobis innovation gating**, with height
computed by FK relative to the *other foot* (encoder-only), never from the filter's own state (§7.3).

### Sessions

Support: **slack overhead harness bearing no load**, so tape-measured displacement is valid truth and
the legs feel true body weight. Spotter must not touch the robot inside a scored window; any catch
voids that session.

| regime | policy | file in `exported/` | count | duration |
|---|---|---|---|---|
| **E1** quiet stand | **A1a deploy keeper** `2026-07-17_21-46-14_a1a_cot0p2_cad0p5_energy0p05_s42` | `high_level.onnx` (the default in place) | 3 | 90 s |
| **E2** stepping in place | bare `2026-08-10_08-40-47_…legodomsim_bare_s42` | `high_level_bare.onnx` | 3 | 60 s |

⚠️ **"Keeper" is ambiguous in this project and an earlier draft of this table named the wrong
one.** Two different checkpoints answer to it:

| | run | filed in `exported/` as | what it is |
|---|---|---|---|
| **A1a deploy keeper** | `2026-07-17_21-46-14_a1a_cot0p2_cad0p5_energy0p05_s42` | `high_level.onnx` | the shipped, hardware-validated stander — **use this for E1** |
| WL-F comparator | `2026-08-06_08-14-16` | `high_level_hlVelEst.onnx` | the KF/WL-F gate-1 comparator, a sim-side baseline |

Only `high_level.onnx` / `low_level.onnx` are ever loaded (`State_RLHRL.cpp:308-309`); the
suffixed files are inert copies, selected by renaming. Renaming preserves mtime, so **file
mtimes cannot tell you what ran** — only the directory's mtime shows that a swap happened.
`PROVENANCE.json` goes stale under manual swapping and will report a false mismatch; that is
a bookkeeping artifact, not evidence about which policy ran. If a session's identity ever
has to be proven after the fact, the only sound method is to replay the logged observations
through each candidate ONNX and match the logged `tgt*`.

⚠️ **The bare arm is the NO-GO policy**, chosen because it is the only one that genuinely steps at zero
command (median |ω| 1.23 rad/s). The accepted assumption: it is the *harder* regime, so a win there
should transfer down to the keeper's gentler motion. This assumption is not tested by this experiment;
if C wins on E2, re-check it on a keeper session before shipping.

### Metrics

**E1** — true `v ≡ 0` exactly, the only free ground truth available. Per arm: `mean(v)` (bias),
`std(v)`, `|∫v dt|` over the last 60 s.

**E2** — feet marked at start, **net displacement tape-measured at the end**, robot returned to the
same posture. Per arm, primary `|∫v̂ dt − measured displacement|` (drift); secondary `std(v)` within
the step cycle; **drift breaks ties**.

Report every arm's error at its native rate **and** at the `c = 8` averaged rung, so the two are never
conflated again.

### Pre-registered decision rule

**Primary comparison, fixed before any scoring: C vs A and C vs B, on E2 drift.** Everything else —
D, E, F, E1, the std metrics — is secondary and reported as such. With six arms across two regimes and
three sessions there are enough numbers that an unregistered "best result" would mean nothing.

* Adopt **C** only if it beats **both A and B**, consistently (same sign of improvement on **≥2 of 3**
  E2 sessions), without worsening E1 standing bias.
* **If C ties B, ship B.** Adopting 27 states and contact logic for a gain a ~20-line complementary
  filter also delivers is the specific failure this arm exists to prevent.
* Adopt **D** over **C** only if D beats C on E2 drift. Otherwise drop the orientation block (§4).
* **F is reported, never decisive.**
* A negative verdict ⇒ ship B for A1, and carry C forward as A2 groundwork with this result recorded
  as the honest sizing of what fusion buys on this platform.

### What would invalidate the test

Visible foot slip or a shifted floor mark; a stand that is not quiet (E1's truth assumes it); a taut or
load-bearing harness (voids E2's truth and changes the leg loading); the spotter touching the robot;
a session recorded before P0's columns are verified present; or **any arm tuned on a session it is
scored on** — `Q` and `R` are fixed from the Allan log and §3.2 before scoring, and never per-session.

---

## 11. Implementation architecture

```
                    LowState @ ~500 Hz (DDS)
                            │
   State_RLHRL::run()  ─────┤ snapshot ALREADY taken at State_RLHRL.cpp:617-643:
   (~990 Hz loop)           │   est_acc_b, est_quat, est_gyro_T, est_q[13], est_dpsi
                            │
                ┌───────────┴────────────┐
                │  new-sample gate       │  ← use arrival Δt; the loop reads each msg ~2x
                └───────────┬────────────┘
                            │
     ┌──────────────────────┼──────────────────────┐
     │ PREDICT              │ MEASURE              │ CONTACT
     │ f̃,ω̃ torso frame      │ leg_fk (pelvis)      │ α_L, α_R  (§7)
     │ §2                   │ → Rz(−ψ)·p − r_imu   │ height + gating
     │                      │ → torso frame  §3    │ [+ tau_est Jᵀ]
     └──────────────────────┴──────────┬───────────┘
                                       │ UPDATE §5
                                       ▼
                          state: r, v, q, p_i, z_i, b_f, b_ω
                                       │
                                       ▼
              v_I^T = C·v      →      v_P^P = Rz(ψ)·(v_I^T − ω_T × r_imu)
                                       │
                                       ▼
                            consumer (HL goal state / logging)
```

Placement: the EKF replaces the call site of `hrl::leg_odom_velocity`, inside the same 1 kHz snapshot
block that already reads **everything it needs** — no new DDS read, no new lock, no new plumbing. Keep
the incumbent computed in parallel and log both; the arms in §10 then come for free on every session.

Suggested code shape, mirroring the existing separation: pure math in a header with no I/O
(`hrl/base_ekf.h`, testable on CPU like `base_state.h` is), a thin adapter in `State_RLHRL`, and a
numpy transcription for offline replay — the KB's `replay.py` already establishes that pattern and
demonstrated FK parity at 2.2e-16.

---

## 12. Summary of the critical position

1. Build the **position-only Bloesch core** with the H1-2 contact frame, torso body frame, and
   FK-derived `R`. It is well-motivated on measurement structure alone: it stops differentiating
   encoders, stops levering the raw gyro, and estimates the gyro bias.
2. **Do not enable Rotella's flat-foot orientation block by default.** It does not improve velocity
   observability, its documented benefit targets a degeneracy that double support does not have, and
   it fails asymmetrically when the foot rolls at touchdown. Keep it switchable, test it in single
   support.
3. **Expect the accelerometer-bias state to be inert during standing** — that degeneracy is structural,
   not a tuning failure.
4. **The complementary-filter arm gates adoption**, because otherwise the experiment cannot
   distinguish "an EKF helps" from "filtering helps". If C ties B, ship B.
5. **The FK-bias argument is a bias argument only** (§9.1): 14.6 mm per degree at the hip pitch, ≈ 32
   mm/s at the measured median |ω| — the same order as the incumbent's observed bias, ~6x below its
   noise. Not a reason to build a filter.
6. **Log the gyro and `est_dq` first, in one change.** Everything above is unfalsifiable until P0 is
   done, and no session on disk carries either.

### Decision record (agreed 2026-08-11)

| branch | decision |
|---|---|
| scope | offline verdict only; no closed-loop, so the per-restart offset never confounds |
| P0 | gyro **and** `est_dq[13]` in one change; dry-run on existing logs, score nothing until it lands |
| arms | A, B, C, D, E, F (§10); primary comparison **C vs A and C vs B on E2 drift** |
| sessions | 3 × 90 s keeper stand + 3 × 60 s bare stepping; bare first (already staged) |
| attitude | undecided by argument — arm F measures it, exploratory only |
| contact | height heuristic (FK, other-foot relative) + Mahalanobis gating |
| metrics | E1 truth `v ≡ 0`; E2 drift vs tape primary, std secondary, drift breaks ties |
| bar | C must beat **both** A and B on ≥2 of 3 sessions; tie with B ⇒ ship B |
| noise ID | one 10-min static log, Allan variance, `Q`/`R` frozen before scoring |
| offsets | ignored — an A2 problem if it persists for a working A1 |
| support | slack overhead harness, no load ⇒ tape measure is valid truth |
| if negative | ship B for A1, carry C forward as A2 groundwork |

## 13. On-robot deployment spec (agreed 2026-08-13)

Ports the offline arms to C++ so all seven run on hardware simultaneously. Decisions below
were settled in a grilling session; each row is a real fork with an alternative that was
rejected, not a default.

**Operating manual** (which arm feeds the policy, how to run a session, how to score it):
`.claude/docs/deployment.md` § "Estimator bench". This section is the *spec* — what was
decided and what is tested — not the runbook.

### 13.1 Scope

**In:** C++ ports of arms B-F faithful to the Python arms as scored (arm A already ships);
per-arm logging; a fail-closed config selector; a window-parity test; a loop-budget gate.

**Out, deliberately:** the ~9-17 mm/step contact-transition loss (§13.6), contact-detector
hysteresis, and **any change to the shipped leg odometry**. Shipping the arms exactly as
scored is what keeps hardware numbers comparable to the four tape-measured legs; fixing the
estimator first would deploy an arm no offline data covers. Re-evaluate the loss against the
hardware data and spec it separately.

### 13.2 Decisions

| branch | decision | rejected alternative |
|---|---|---|
| architecture | **all 7 arms compute every tick, all logged, one config-selected arm feeds `obs["hl_vel"]`** | one arm per session (9+ sessions, each arm on different data with its own encoder offset) |
| arms | **all 7.** C/C+grav/D/F are ONE filter class with three booleans (`use_ori`, `use_att`, `use_grav`), so "all 7" is 1 EKF + B + E, not 4 EKFs | B/D/F subset |
| prediction | **every `run()` tick, fixed `dt = 1 ms`**, exactly as arm A | real dt; gated |
| **update** | **only when `acc/gyro/q` change** (~500 Hz), using the same change-detect key the Python harness dedupes on | every tick — would count each measurement twice, shrink `P` ~2x too fast, over-trust the stationary-foot assumption and make the C++ arms *more* zero-biased than the scored ones |
| parity bar | **c=8 window average within 1e-4 m/s; session mean within 1e-5 m/s** | per-sample 1e-6 (tests a code path the robot never runs) |
| config | **required `base_estimator:` key; missing or unknown aborts at startup**, no default | silent default to `legodom` — the exact pattern behind the dead-signal splay |
| logging | **`vx, vy` per arm per logged tick** (14 floats, ~+7% file) | +`vz` as a divergence canary |
| divergence | **isolate only, log raw.** Each arm owns its state; NaN/divergence is preserved as evidence, never reset | reset-and-count |
| init | **mirror the Python warm start**: `R0` = vendor quaternion, `v0` = leg odometry, `P_vv = (0.25 m/s)^2`, foot anchors at current FK | cold start |
| budget | **measure, then gate**: `p99(run tick) < 800 us`. Over budget ⇒ decimate PASSIVE arms only, log `N` | fixed decimation; separate thread |

### 13.3 Data flow

```
run() @ ~1 kHz
  ├─ predict(acc, gyro, dt=1ms)          every tick, all arms
  ├─ if (acc|gyro|q changed):            ~500 Hz
  │     update_feet(), update_gravity()  each measurement counted ONCE
  ├─ EstSample.v_arm[7][2]  ──────────►  flight recorder (all arms)
  └─ selected arm ──► lo_sum_/lo_n_ ──►  policy_step() drains at HL fire ──► obs["hl_vel"]
```

**Invariant:** a passive arm reaches the LOG and nothing else. No shared state between arms,
no path from a passive arm to `lowcmd`, the safety filter, or the selected estimate. This is
what makes it safe to run the known-divergent arm C on the robot.

`obs["hl_vel"]` remains the ONLY consumer; the estimate never enters `state_n` (project rule).
A0 needs no key: its obs carries no base linear velocity, so on A0 every arm is passive.

### 13.4 Changes, by file

| file | change |
|---|---|
| `deploy/robots/h1_2/include/hrl/base_state.h` | add `BaseEkf` (fixed-size Eigen, 27-state error), `complementary_velocity()` (arm B), `jacobian_velocity()` + `leg_jacobian()` (arm E). Header-only, `hrl::` namespace, reusing existing `leg_fk`/`foot_site_b` |
| `deploy/robots/h1_2/include/safety_logger.h` | `EstSample += float v_arm[7][2]`; header + writer emission |
| `deploy/robots/h1_2/src/State_RLHRL.cpp` | instantiate 7 arms; predict/update per §13.3; fill `EstSample`; route the selected arm into `lo_sum_`/`lo_n_` |
| `deploy/robots/h1_2/src/State_RLBase.cpp` | same arms, passive, logging only |
| `deploy/robots/h1_2/config/policy/velocity_hrl/v0/params/*.yaml` | add `base_estimator: legodom` to every HRL deploy config |
| `deploy/robots/h1_2/test/base_state_estimator_test.cpp` | extend: EKF math against golden vectors |
| `tests/test_estimator_parity.py` | NEW: window parity, C++ vs Python, on a real session |

Robot-local only. **Never** edit `deploy/include/isaaclab/`.

### 13.5 Test plan (defined before implementation)

| # | test | success criterion |
|---|---|---|
| T1 | EKF Jacobians vs finite differences, in C++ | max abs diff < 1e-5 |
| T2 | **Window parity** on a real session, all 7 arms | c=8 mean within **1e-4 m/s**; session mean within **1e-5 m/s** |
| T3 | Arm A unchanged | byte-identical `hl_vel_lo_` to pre-change build on the same log |
| T4 | Config fail-closed | missing key ⇒ abort with a named error; unknown value ⇒ abort listing valid names |
| T5 | Loop budget | `p99(run tick) < 800 us` on hardware, estimator block reported separately |
| T6 | Isolation | with arm C forced to NaN, selected estimate, `lowcmd` and safety filter bit-unchanged |
| T7 | Mutation sensitivity | `scripts/check_test_sensitivity.py` catches a deliberate defect in each new test |

T2 is the gate that makes hardware results interpretable: without it a hardware disagreement
cannot be attributed between the port and the robot. T3 protects the shipped keeper.

### 13.5b Implementation read-out (2026-08-13)

Landed: `hrl/base_estimators.h` (all arms + `EstimatorBank`), logging, fail-closed config,
A0 + A1 wiring, `tests/test_estimator_parity.py`, the C++ gate in
`test/base_state_estimator_test.cpp`, golden fixture in `tests/fixtures/`.

**T2 PASSES for all seven arms at 4e-11 to 1e-7 m/s**, four orders under the 1e-4 bar.
Three things the implementation forced, none of them anticipated by the spec:

1. **The filter runs in DOUBLE, not float.** `P` spans 1e2 (deliberately inflated position)
   to 1e-2, so the recursion carries a condition number ~1e4; float32 epsilon predicts ~1e-3
   relative error and that is exactly what was measured — arms B and E (no covariance
   recursion) matched at 1e-7 while every EKF arm sat at 2-3e-4, failing the bar on precision
   alone. `leg_fk` in `base_state.h` is now scalar-templated for this; the float
   instantiation is the untouched shipped path.
2. **Offline and online cannot share an initial condition.** The harness warm-starts velocity
   at sample 0 from arm A's first finite reading — it can look ahead. Online, arm A needs a
   previous foot position, so the warm start lands one tick later. Contracting arms forget
   that (B/D/E/F agree to 1e-5 post-settle); **the position-only arms do not** (C, C+grav
   separate to 2-4e-4), because a divergent filter amplifies any initial difference. The gate
   therefore primes both sides explicitly from `estimator_parity_v0.txt`, which is what
   isolates a porting error from a convention difference. That C amplifies it at all is one
   more independent symptom of the arm that already failed §12's bar.
**T5 PASSES on a live bridge session** (`2026-08-13_10-36-02_t5_loopbudget`, 47 649 ticks over
48 s = **993 Hz**, so `run()` sustains its 1 kHz period with all seven arms live):

| quantity | p50 | p99 | max |
|---|---|---|---|
| `run()` work | ≤50 µs | **≤75 µs** (gate: <800 µs) | 1960 µs |
| estimator block | ≤50 µs | ≤50 µs | 851 µs |
| tick period | ≤1000 µs | ≤1025 µs | — |

Overruns >800 µs: **14 of 47 649 (0.029%)**. Those outliers are scheduling, not compute — the
estimator's work is fixed-size and branch-light, and its own p99 is ≤50 µs, so a single 851 µs
sample is preemption (the recorder's ~5 s flush is the likeliest source). **The decimation
fallback is not needed.** Instrumentation is bucketed counters reported once from `exit()`,
never a per-tick print: the "10 ms loop tail" retracted on 2026-08-04 was *created* by the
logging that measured it.

All seven arms ran live and logged 100% finite, and the selected arm was `legodom`, i.e. the
shipped behaviour is unchanged while the other six ride along passively. Their bridge std vx
ordering (F 0.034 < B 0.040 < C+grav 0.044 < D 0.051 < C 0.054 < E 0.064 < **A 0.198**)
reproduces the shape of the offline result — the incumbent much the noisiest, every filtered
arm well under it — but **do not read the bridge ranking as a hardware result**: the bridge
understates estimator error 3-6x, and this run had the elastic band attached.

3. **Cost, measured: ~20 µs/tick amortized for all seven arms, on the target CPU.** The
   controller runs on the dev machine and talks to the robot over DDS, so the desktop Ryzen
   IS the deployment processor — this is not an extrapolation. Breakdown: predict 9 µs every
   tick, update 20 µs on the ~50% of ticks carrying a fresh DDS sample, 1.5 µs for the
   kinematics plus arms B and E. Against the 1 ms tick that is ~2% of budget, so the
   passive-arm decimation fallback is very unlikely to be needed. T5 still wants one live
   session to confirm the whole `run()` tick under real DDS load.

Deviation from §13.4: the arms live in a **new** `hrl/base_estimators.h` rather than growing
`base_state.h`, so the shipped leg-odometry path is not even in the same file as the new code.

Not yet done: a real-robot session (the bridge covers the loop, not the robot's own DDS traffic).


### 13.6 Vocabulary (proposed for `CONTEXT.md`)

Two terms are needed and one earlier usage was wrong.

**Fused base velocity** — a pelvis-velocity estimate that combines the IMU with leg
kinematics through a filter (arms B-F), as opposed to _Leg odometry_, which is instantaneous
and unfiltered. _Avoid_: "the estimator" (ambiguous per the existing glossary entry); "EKF"
(only four of the arms are EKFs).

**Contact-transition loss** — the roughly fixed displacement a _Fused base velocity_ loses
per step (measured 9-17 mm/step, speed-independent, 4 tape legs), from the stationary-foot
measurement being applied across touchdown and lift-off, when the foot is not yet or no
longer stationary. It is a *filter* artifact: _Leg odometry_ does not show it.
_Avoid_: "rollover" (the glossary reserves that for the shelved WL-D arm-6 reward, a policy
behaviour); "stance-point error" (the glossary records the point choice as settled and
irrelevant, and this is about the transition, not the point); "slip" (a ground phenomenon).

### 13.7 Risks

* **Loop budget.** Four 27-state EKFs at 1 kHz is the main unknown; T5 gates it. Fixed-size
  Eigen throughout, no dynamic allocation in the loop.
* **Fail-closed breaks old configs by design.** Every HRL deploy config is updated in the same
  change; any config not updated will refuse to boot. That is the intent, and it is why this
  is worth an ADR.
* **NaN columns downstream.** "Log raw" means the offline harness must tolerate NaN from arm
  C; it already does (`~np.isnan(v).any(axis=1)` everywhere).
