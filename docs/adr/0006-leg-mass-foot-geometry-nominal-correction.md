# Leg-mass and foot-geometry nominal correction (Model v3)

**Status: NOT SUPERSEDED (2026-08-28).** ADR-0008 briefly superseded this and was itself
reverted by **ADR-0009**, which cut the leg-mass correction with the rest of Model v3 after
measuring it inert. So item 1 (leg mass) is **not applied**, and item 2 (foot sole geometry)
is live again as a staged, unscheduled proposal. The historical framing below is kept: Model v3 is now defined by ADR-0008 as
**action clip + leg mass**. The leg-mass spec in item 1 below is carried into that cut
**verbatim**, but justified as a *fidelity* correction only — V1's falsification (leg mass moved
the hardware lean by nothing) stands and the lean claim is not made. **Item 2, the foot sole
geometry, is NOT cut in v3**; it stays a live, unscheduled proposal, since ours is still the only
H1-2 sim using hand-authored flat foot capsules and it remains the CoP-transfer lever. Everything
below is retained as the investigation record — the V1 read-out, the Spec A hardware rejection,
the Spec B `joint_offset` confirmation, and probe P1 — none of which is superseded.

**Original status:** PROPOSED / staged, held for the next model version (the user's call 2026-07-24).
Not applied. The leg-mass edit was made, verified, and reverted. **Amended 2026-07-28 after the
inclinometer session** (see "Amendment" below): scope stays **leg mass + foot sole geometry**,
and Component A (the attitude error) is now split into ~0.9 deg of foot-sole wedge (a SIM defect,
fixed here in the XML) plus ~1.8 deg of leg-encoder zero error (a HARDWARE defect, fixed deploy-side
by Spec B). Three implementation specs are appended: **Spec A** (wire the observation half of the
existing encoder-bias DR, a v3-cut change), **Spec B** (the deploy `joint_offset`, independent of
the cut and testable now), and **Spec C** (set `mode_pr` explicitly, a robustness fix on the
parallel A/B ankle). **Validation experiment V1 is approved (2026-07-28): cut leg mass ALONE
first, retrain A0, re-measure the lean on the robot, and direct the rest from that result.**
**V1 RAN 2026-07-31 and was FALSIFIED — leg mass moved nothing on hardware (see "V1 read-out"
below).** Leg-mass-only does not proceed to a v3 cut on this evidence; item 2 (sole wedge) and
Spec A do not get an automatic green light from V1 either. Investigation redirects to Spec B
(encoder-zero) and identifying which joint(s) the encoder error concentrates at. **Probe P1
(2026-07-31) measured the lean transfer function in sim** (superposition across joints holds to
2.3%; ankle gain 1.684 vs hip 0.733). *P1's cross-policy hardware predictions were later retracted*
— see "Spec A hardware read-out": stand pitch is confounded by how still a policy stands
(r = −0.96), so P1's gains are valid only for within-policy interventions. **Outcome so far:
Spec A REJECTED on hardware (2026-08-03); Spec B's `joint_offset` CONFIRMED as the dominant lever
— it drives the stand from +5.5° back to −4.0° forward — but the required value is NOT reproducible
between sessions, so it must ship as a per-session calibration, never a committed constant.** The
V1 leg-mass edit was also found still applied to the training XMLs on 2026-07-31 and reverted (it
must not ride along into the WL-D gait batch). Supersedes
nothing in ADR-0005; it remains the next checkpoint-invalidating nominal correction on top of
Model v2 if and when the mass branch is revisited.

## Context

First real-hardware A0 runs (2026-07-23, `a0_v2_optB_rs20_baseline`, split arms) showed a
persistent 3-5 deg backward pitch lean at stand that sim never has (bridge ~0). The
investigation (WL-B0b, full record in `doc/hrl/A1a_deploy_plan.md` "THE BACKWARD LEAN,
DECOMPOSED") split it into two independent causes, both plant-modeling gaps:

1. **Excess standing load / knee droop.** Three independent pitch estimates (IMU quaternion;
   encoder feet-flat FK; ankle-torque statics) plus a 3-torque per-leg GRF solve agree that
   the real legs carry more than the model weighs. Reconciled with the robot's own `tau_est`
   (the naive `kp*(cmd-meas)` over-reads real knee torque ~1.3x) the clean standing load equals
   real body weight. The user then measured it directly: **real robot 73.70 kg (head off), one leg
   including all 3 hip motors 18.60 kg.** The model (`h1_2.xml`) is 66.98 kg with a 15.399 kg
   leg. **The entire 6.7 kg gap is in the legs** (+3.20 kg/leg; the non-leg remainder is
   +0.31 kg). It is neither uniform body scaling nor a torso-CoM shift.

2. **A 1.8 deg encoder-to-attitude map error** (constant across 25 deg of posture and 207 deg
   of heading; the IMU is honest, corroborated by the torque-statics estimate). The leading
   physical candidate is the foot-sole geometry: **ours is the only H1-2 sim using
   hand-authored flat foot capsules** (7 per foot, all at z=-0.035, a plane exactly parallel to
   the ankle frame, 0.00 deg by construction), while every reference (opt bridge, isaac-gym,
   both RMA stacks, h1v2-Isaac) uses the actual foot mesh for collision. A real sole inclined
   ~2-3 deg would produce exactly the observed bias; a flat capsule cannot. This is also the
   CoP-transfer lever for push-off (WL-D arm 6), since CoP rolls heel-to-toe along the sole.

Origin and universality of the leg-mass gap (verified in-repo, 2026-07-24):

- The 15.399 kg leg (per-link 2.829 / 2.920 / 4.962 / 3.839 / 0.102 / 0.747) is byte-identical
  across all five sims (ours, `/opt/unitree_mujoco` bridge, `unitree_rl_gym`, RMA-27dof,
  RMA-book) AND the official Unitree URDFs (`/opt/unitree_mujoco`, the mybotshop
  `h1_description` package, the RMA URDFs) AND both isaac repos (`h1v2-Isaac`). The HuggingFace
  `unitreerobotics/unitree_model` H1-2 will read the same. This is a **manufacturer CAD-vs-real
  discrepancy** (real cabling, connectors, as-built component mass the CAD omits), inherited by
  every downstream H1-2 sim, not a repo error.
- **Not an H1-vs-H1-2 swap** (hypothesis tested and rejected): H1-v1 legs are 10.9 kg with a
  different per-link split (2.24 / 4.15 / 2.23 / 1.72 / 0.55), lighter and differently
  distributed. The 15.4 kg is authentically H1-2's official value.

Friction and armature were ruled out as the static cause: at a static stand the hold torque
equals the gravity torque, and frictionloss/armature are velocity/acceleration terms that
vanish at rest. The measured knee droop (100-170 mrad) is reproduced by `droop = tau/kp` from
gravity alone, against a friction deadband of only 7-20 mrad. Head mass is negligible
(removing the head, at x=+0.05 m, moves whole-body CoM 1-5 mm, versus the ~55 mm a 3.5 deg
lean needs) and the falsification stands.

## Decision

Create **Model v3**, a versioned nominal-plant correction applied uniformly to all
architectures A0-A4, comprising two changes cut together:

1. **Leg mass -> real.** Scale the 6 leg links per side (hip_yaw, hip_pitch, hip_roll, knee,
   ankle_pitch, ankle_roll) by **1.20787 = 18.6 / 15.399** on `mass` and all three
   `diaginertia` values, leaving `ipos` and `quat` unchanged so no CoM shift is introduced.
   Per-link mass: hip_yaw 2.829->3.417, hip_pitch 2.920->3.527, hip_roll 4.962->5.993,
   knee 3.839->4.637, ankle_pitch 0.102->0.123, ankle_roll 0.747->0.902. New total 73.386 kg,
   one leg 18.600 kg (verified: torso/pelvis/arms untouched, compiles). Uniform scaling assumes
   the missing mass is distributed like the CAD leg; if the real excess is concentrated (e.g. a
   heavier hip actuator or pelvis-side cabling) the per-link split is wrong even though the
   total and CoM height are right. Revisit if a per-link teardown weight becomes available.

2. **Foot collision geometry -> real sole. CONFIRMED as a real contributor by the 2026-07-28
   inclinometer session; stays in the v3 cut.** Measured sole thickness shows **both** soles
   thicker at the front (right +3.34 mm, left +5.04 mm over ~260 mm), i.e. **both feet toe-up by
   0.74 and 1.11 deg, mean 0.92 deg**. A toe-up ankle frame tilts the body backward, so this
   accounts for **~0.92 deg = ~41% of the 2.26 deg constant attitude offset** (~29% of the
   3.14 deg residual at the deep-lean stand), in the correct direction. The model's flat capsule
   plane (constant z=-0.035, parallel to the ankle frame) cannot represent it; every reference
   model uses the foot mesh. Fix: tilt the capsule sole plane by the measured angle, or switch
   foot collision to the mesh (also the better option for **push-off CoP fidelity**, WL-D arm 6).
   Note this reverses an intermediate call in this ADR made on a transcription error (the right
   foot's front/back readings were initially swapped, which made the two wedges appear to cancel);
   the corrected measurement makes them reinforce. The wedge does **not** explain the L/R
   asymmetry (differential is only 0.375 deg, wrong sign), which stays open.

3. **NOT a model change: correct the leg encoder zeros on the deploy side (Spec B below).** The
   inclinometer confirmed the IMU is honest (6.80 deg measured vs 6.30 deg reported, agreeing to
   0.50 deg) while the encoder feet-flat FK is off by 3.14 deg, and that with the true attitude
   and the measured encoders the model puts the foot 2.74 deg toe-up while it is provably flat.
   After subtracting the 0.92 deg sole wedge (item 2), **~1.8 deg remains at that posture (~1.3 deg
   of it session-constant)**, and a leg-encoder zero error of that size reproduces it exactly (the
   servo drives the *reported* angle to target, leaving the *true* joints offset). Component A is
   therefore a **two-part** problem: ~0.9 deg foot geometry (XML, item 2, a SIM defect) + ~1.8 deg
   encoder zeros (a HARDWARE defect). **A vendor zero-point re-calibration was considered and is
   NOT the recommended route**: Unitree publishes no H1-2 procedure, and the parallel A/B ankle
   turns one motor-zero error into a coupled pitch+roll offset on the joint with the least travel,
   so a botched re-zero is worse than the error we have and cannot be verified before it is
   committed to the motors. **Spec B (a reversible, verifiable `joint_offset` in the deploy
   config) is the recommended fix**, with no retrain and no XML change. Expect it to take the
   residual from -0.048 rad to about -0.016 rad (the wedge floor), not to zero.

**Two plants, both set at the cut** (they serve different roles):

- **Training model** `src/assets/robots/unitree_h1_2/xmls/h1_2.xml` (loaded via `H1_2_XML`) and
  its unused viz twin `scene_h1_2.xml`: required for the policy to *learn* the corrected plant.
- **Bridge test plant** `/opt/unitree_mujoco/unitree_robots/h1_2/h1_2_handless.xml`: the
  sim-to-sim stand-in for hardware in the G2 gates. Set to the same real mass (and foot
  geometry) so the bridge does not test a lighter/idealized robot than both training and
  reality. It is an external clone, not version-controlled with this repo, so its edit is noted
  separately at the cut.

Deploy C++ and the 27-value gain YAMLs carry no link masses and need no change.

**Validation gate before rebasing A1/A1a on v3** (mirrors ADR-0005): (1) smoke run; (2) full
A0 retrain on v3, same protocol/seeds, benchmark within noise on err_vx/vy/yaw + fall_rate;
(3) cross-eval a v2 checkpoint in the v3 env to quantify the modeling gap (thesis-reportable:
this is a direct sim2real-fidelity delta); (4) v3-A0 ONNX through the C++ bridge on the
mass-corrected bridge plant, checking the stand pitch against the real robot's.

**Sequencing:** held until the inclinometer session closes the foot question, so both changes
land in one rebase. No hardware-bound training runs happen before the cut; only WL-D LL-reward
gait-shape refinement, which stays on Model v2 (15.4 kg) to remain comparable to the `arm4d`
batch. Because nothing transfers to hardware before v3, applying the correction to v3 carries
no sim-to-real transfer risk.

## Alternatives considered

- **Apply the mass fix now (mid-WL-D).** Rejected: rebases the training plant mid-batch and
  confounds every gait-refinement comparison against `arm4d`. Held to a clean version boundary
  instead.
- **Fold the mass gap into A2 Wide DR only** (randomize base/leg mass, never correct the
  nominal). Rejected as the primary fix for the same reason ADR-0005 rejected it for gains: the
  nominal stays knowably wrong and A0/A1 deploy results stay confounded. A mass-magnitude DR
  channel is still wanted *around* the corrected nominal for robustness (base_com DR is +-5 cm
  of position only and never covered mass magnitude), and a biased projected-gravity / foot-angle
  DR channel is the robustness complement to the foot-geometry fix. DR complements the
  correction; it does not replace it.
- **Zero-mean IMU/encoder noise in DR to fix the lean.** Rejected: both lean components are
  biases, not noise; a policy averages zero-mean noise out and still centers on the wrong mean.
  Only biased DR (a projected-gravity pitch offset, a mass-magnitude channel) addresses a bias.
- **Recalibrate/zero the IMU at stand.** Rejected: the IMU is honest (independent torque-statics
  estimate agrees with it; at passive hang IMU pitch equals accelerometer-implied to 0.0013 rad).
  The robot genuinely stands pitched back, so zeroing the IMU would feed the policy a false level
  and deepen the lean. If a mount offset were ever found, the only correct fix is a fixed
  quaternion pre-rotation in `unitree_articulation.h` before `projected_gravity_b`; none is in
  evidence.
- **Non-uniform leg-mass distribution** (put the 3.2 kg in specific links). Deferred: no per-link
  real weights available; uniform scaling preserves CoM and is the neutral default until a
  teardown weight exists.

## Consequences

- All Model v2 checkpoints, including the deployed A0 `a0_v2_optB_rs20_baseline` and the A1
  keeper `arm4d`, are invalidated as baselines once v3 is cut; A0 retrains first, then A1/A1a
  rebase on the new A0, exactly as in the v1->v2 transition. Replays of v2 checkpoints under the
  v3 XML carry a plant mismatch (heavier legs than trained), so cross-boundary comparisons are
  invalid and the ledgers keep v2 results as historical evidence only.
- The WL-D gait-refinement batch, in flight on v2, completes on v2 and is compared internally;
  its winner is re-validated on v3 after the rebase.
- The bridge plant edit (`/opt`) must land with the training edit to keep the G2 sim-to-sim gate
  a faithful hardware stand-in; leaving it at 15.4 kg would make the bridge lighter than both
  training and reality.
- This directly addresses the backward-lean Component B (knee droop) and, via the conditional
  foot-geometry change, Component A (attitude bias) and push-off CoP fidelity. Both components
  affect A0 and A1 identically (same `projected_gravity`, same offsets, same plant), so v3 is
  shared and is not an RQ2 confound; it lifts both architectures and should precede the A1
  hardware track.
- Thesis-reportable finding independent of the fix: every published H1-2 sim result (ours and
  the references) is on a leg 21% lighter than the real robot, a manufacturer CAD gap, which is
  worth stating as a sim2real caveat.

## Amendment (2026-07-28): inclinometer session splits Component A and re-scopes the cut

The closing hardware measurement (flat floor; concurrent log
`all_joints_2026-07-28_10-13-12.csv`, dead-still loaded stand at t 340-620 s) resolved the
conditional in decision item 2 and changed what this ADR should carry. Full record:
`doc/hrl/A1a_deploy_plan.md` "INCLINOMETER SESSION 2026-07-28".

1. **The IMU is honest** (pelvis 6.80 deg measured vs 6.30 deg reported, agreeing to 0.50 deg;
   torso and pelvis read identically, confirming the pelvis-vs-torso non-issue). The encoder
   feet-flat FK is the side that is wrong, by 3.14 deg. The "recalibrate the IMU" option is
   definitively closed.
2. **2.74 deg of error is in the encoder->geometry chain**, established without any
   surface-parallelism assumption: with the true attitude and the measured encoders, the model
   puts the foot 2.74 deg toe-up while it is provably flat on a flat floor.
3. **The sole wedge is a real contributor, ~41% of the constant offset** (CORRECTED: both soles
   are thicker at the FRONT, so both feet are toe-up by 0.74 deg right / 1.11 deg left, mean
   0.92 deg, reinforcing rather than cancelling; the first reading of this had the right foot's
   front/back swapped). Foot geometry therefore **stays in the v3 cut**, and also serves push-off
   CoP fidelity (WL-D arm 6). It does not explain the L/R asymmetry (differential only 0.375 deg,
   wrong sign), which stays open.
4. **Component A splits in two: ~0.9 deg foot geometry (item 3) + ~1.3 deg leg-encoder zero
   error.** The encoder part is well under 1 deg per joint across hip/knee/ankle (the thigh
   reading tentatively puts more of it at the hip, though link-face measurements carry +-2 deg
   since the shank's own front and back faces differ by 2.1 deg). **Unitree's zero-point
   calibration removes that part with no retrain and no XML change**, which is cheaper than
   anything in this ADR and should be tried first; it should leave the ~0.9 deg wedge floor,
   which the XML fix then clears.
5. Refined error model, over both sessions and a 0.35 rad posture range (n=1456):
   `p_IMU = 1.132 * p_kin - 0.0394` (r=0.973), i.e. a ~2.3 deg constant offset plus a ~13%
   scale error, so it is not pure load-proportional compliance.

**Revised decision:** Model v3 = **leg mass + foot sole geometry**, as originally scoped. What
changed is the split of Component A: the wedge is ~0.9 deg of it (XML, in the cut) and leg-encoder
zeros are the other ~1.3 deg (hardware, outside this ADR). Sequencing gains a cheap step ahead of
the cut: run the leg-joint re-zero, repeat the 60 s policy stand, re-measure. Expect the residual
to fall from -0.048 rad to about **-0.016 rad**, the wedge floor; hitting that floor confirms the
split and the XML fix clears the rest. If it does not move at all, the encoder-zero hypothesis is
wrong and the whole residual is geometry. The leg-mass evidence is independent of all of this and
is unaffected.

## Specs (2026-07-28), PROPOSED, awaiting approval

Two changes fall out of the inclinometer session. They fix **different halves of the same
problem and must not be confused**:

- The **sole wedge (~0.9 deg) is a SIM defect**: the real foot genuinely sits toe-up, and the
  model does not know it. Fix the **XML** (decision item 2). Nothing is wrong on the robot.
- The **encoder zeros (~1.8 deg) are a HARDWARE defect**: the robot misreports its own joint
  angles. Fix the **deploy** (Spec B). Nothing is wrong in the model.

Applying either fix on the wrong side would inject a new error of the same size.

### Spec A (training): wire the observation half of the existing encoder-bias DR

**What is already there.** `encoder_bias` DR is active and correctly ranged:
`src/tasks/velocity/velocity_env_cfg.py:248`, `mode="startup"`,
`bias_range=(-0.015, 0.015)` rad = **+-0.86 deg per joint**, sampled once per env and held for
the run (persistent, like a real calibration error). The **actuation** half is modelled
correctly in mjlab: `envs/mdp/actions/actions.py:222-223` sets
`target = processed_actions - encoder_bias`, so the true joint lands offset from the command
exactly as a real servo driving a miscalibrated encoder to target does.

**What is missing.** The **observation** half is off. `envs/mdp/observations.py:51` is
`joint_pos_rel(biased: bool = False)` and no config of ours passes `biased=True`. Consequence:

| | true joint | policy observes | encoders vs gravity |
|---|---|---|---|
| sim today (`biased=False`) | action - bias | action - bias | **agree** |
| sim with `biased=True` | action - bias | action | **disagree** |
| real robot | action + eps | action | **disagree** |

Today the policy is shown the truth, so gravity and joint angles stay consistent and the bias
is nulled by a trivial action shift: the DR is close to a no-op for a closed-loop policy, which
is why it trains cleanly and was never noticed. **The real failure mode (the policy fed a
contradiction between `projected_gravity` and joint angles) is therefore never presented during
training.** That contradiction is exactly what the hardware does, and why the deployed policy
cannot correct the lean.

**Change.** In `src/tasks/velocity/velocity_env_cfg.py:76-79`, the shared `joint_pos`
observation term gains `params={"biased": True}`. One term; A1 inherits the same base obs
config (no `joint_pos_rel` override exists in `config/h1_2_a1/env_cfgs.py`), so A0 and A1 both
pick it up from one edit, which is what RQ2 comparison-cleanliness requires.

**Explicitly NOT changed:** the bias range (+-0.86 deg is already right-sized: three independent
joints sum to std ~0.86 deg, max 2.6 deg, covering the measured ~1.3-2.2 deg residual at
1.5-2.5 sigma), the DR mode, the actuation path, the noise terms, any reward.

**Expected effect.** The policy must learn to stand upright when its joint angles and its
gravity vector disagree, i.e. to trust contact + gravity over encoder zeros. Expect slightly
slower convergence and possibly a small tracking cost; that is the price of the robustness and
should be measured, not assumed.

**Risk / scope.** This changes the training distribution for every architecture, so it is a
**v3-cut change**, not a hot fix, and it invalidates v2 checkpoints for comparison exactly like
the leg-mass correction. It belongs in the same rebase.

**Test plan.** (1) Smoke run (`WANDB_MODE=disabled`), confirm it trains and the obs dim is
unchanged. (2) Full A0 retrain on v3, benchmark err_vx/vy/yaw + fall_rate against the v3
no-bias control; a regression beyond noise means the band is too wide and gets revisited.
(3) The honest disconfirmer: evaluate a v3 policy in an env with a **deliberate fixed** encoder
offset of the measured size and check the stand pitch stays near 0. If it still leans, Spec A
does not deliver robustness for a memoryless policy and that is a reportable negative result
(and an argument that this belongs to A2's adaptation, not to A0/A1).

### Spec B (deploy): `joint_offset` correction for the real robot's encoder zeros

**Rationale.** Unitree does not publish an H1-2 zero-calibration procedure, and the ankle is a
parallel A/B mechanism (`h1_2_low_level_controller/src/invDyn_robot.cpp:352`: `enum PRorAB`,
with indices 4/5 aliased `ANKLE_PITCH/ANKLE_B` and `ANKLE_ROLL/ANKLE_A`), where a single motor
zero error appears as a **coupled pitch AND roll** offset, each half the motor error, on the
joint with the least travel (ankle roll is +-15 deg total). A botched vendor re-zero there is
worse than the error we have, and cannot be verified before it is committed to the motors.
A software offset is reversible, verifiable, per-joint, and needs no vendor tooling.

**Definition.** `eps_j = (true joint angle) - (reported angle)`, per joint. Measured lumped
value from the 2026-07-28 stand: total over the leg chain `+0.0478 rad`, minus the `0.0161 rad`
sole wedge (which is a *sim* defect, not a hardware one), gives **`eps_total ~ +0.0317 rad
(1.8 deg)`**. The feet-flat constraint pins only the **sum**, so the per-joint split is not
identified by this data; put the lumped value on `ankle_pitch` (the joint that sets foot vs
shank) or split it, and let the re-measurement decide whether the split matters.

**Change.**
1. `deploy/robots/h1_2/config/policy/*/params/deploy_real.yaml`: new optional
   `joint_offset:` vector (27 floats, default all-zero, absent = zero so old configs are
   unaffected).
2. `deploy/include/unitree_articulation.h:38`: on read,
   `data.joint_pos[i] = motor_state.q() + joint_offset[i]` so **everything downstream
   (observations, the safety clamp's `q_meas`, the hold) works in TRUE joint coordinates**,
   consistent with `projected_gravity`.
3. `deploy/robots/h1_2/src/State_RLBase.cpp` (and the same edit in `State_RLHRL.cpp`): the
   value written to `motor_cmd().q()` converts back to reported coordinates,
   `q_sent = q_cmd - joint_offset[i]`, **after** the per-joint limit clamp, so the clamp keeps
   protecting against firmware trips in the coordinates the firmware actually compares against.
4. Flight recorder: keep logging the **raw** reported `q` (unchanged) so old and new sessions
   stay comparable, and log the offset vector once into the existing `_meta.json`.

**The load-bearing detail:** the offset must be applied on **both** the read and the write. A
one-sided application creates a fresh sensor-vs-command inconsistency of exactly the kind this
whole investigation was about. `deploy_real.yaml` `default_joint_pos` / action `offset` stay
untouched: they are policy-space constants, not hardware calibration.

**Explicitly NOT changed:** gains, action scale, `hold_joint_ids`, the safety thresholds, the
ONNX, the training model. The sim needs no `joint_offset` (its encoders are perfect by
construction; Spec A is what teaches tolerance).

**Test plan.** (1) Bridge regression with `joint_offset` all-zero: byte-identical behaviour to
today, proving the plumbing is inert when unset. (2) Bridge with a deliberate non-zero offset:
confirm the commanded true posture shifts by exactly the offset and the clamp still fires at
the right reported angle. (3) Hardware, 60 s policy stand at cmd 0, spotted, no walking.

**Pass criterion, decided in advance:** the residual `p_IMU - p_kin` moves from **-0.048 rad to
about -0.016 rad**, the sole-wedge floor. **Not to zero** -- landing exactly on the wedge floor
is what confirms the two-part split. Overshooting past zero means the sign is wrong; no
movement at all falsifies the encoder-zero hypothesis and sends the whole residual back to
geometry (in which case Spec B is reverted and item 2's XML fix absorbs it).

### Ordering (superseded by V1 below)

Original recommendation: Spec B first, since it is independent of the v3 cut, testable on the
currently deployed policy, and the cheapest confirmation that the diagnosis is right; Spec A
rides the v3 rebase.

**Superseded 2026-07-28 by the user's call:** run **V1 (leg mass alone)** first, because it isolates
the best-measured error. Spec B remains free and orthogonal (it moves the residual, V1 moves
`p_kin`), so it can be run alongside without spoiling V1. Spec A rides the later v3 rebase.
Spec C folds into whichever hardware session comes next.

### Spec C (deploy): set `mode_pr` explicitly

**Rationale.** The H1-2 ankle is a **parallel A/B mechanism**, not a serial pitch/roll pair
(`h1_2_low_level_controller/src/invDyn_robot.cpp:352`: `enum PRorAB { PR = 0, AB = 1 };` with
joint indices 4/5 aliased `LEFT_ANKLE_PITCH`/`LEFT_ANKLE_B` and `LEFT_ANKLE_ROLL`/`LEFT_ANKLE_A`;
the reference controller sets `lowcmd_.mode_pr = PR` explicitly). The same two wire indices mean
*pitch/roll* in PR mode and *motor B/A* in AB mode.

**Our deploy never sets it.** `deploy/robots/h1_2/main.cpp:53` sets `mode_machine() = 6` and
nothing else. We run in PR mode only because the IDL in-class default happens to be
`uint8_t mode_pr_ = 0` and `PR == 0`, i.e. by zero-initialization, not by intent.

**Why it matters even though it currently works.** If an SDK update, a message reuse, or a
copy-paste from another robot ever changed that default, the ankle commands would be silently
reinterpreted as A/B motor commands. With the usual linkage convention (`theta_A = P + R`,
`theta_B = P - R`) a commanded pitch would come out roughly **halved with a spurious roll of the
same size**, on `ankle_roll`, the joint with the least travel in the whole robot (+-0.2618 rad =
+-15 deg) and the one that already dominates the safety-clamp events. That is a silent,
hard-to-diagnose failure on the most margin-critical joint.

**Change.** In `deploy/robots/h1_2/main.cpp`, immediately after the `mode_machine() = 6` line:

```cpp
// H1-2's ankle is a PARALLEL A/B mechanism: indices 4/5 (and 10/11) mean ankle
// pitch/roll in PR mode and motor B/A in AB mode. We command pitch/roll, so PR is
// required. Set explicitly -- we were previously relying on the IDL default
// (mode_pr_ = 0 == PR), which is not a contract.
FSMState::lowcmd->msg_.mode_pr() = 0;  // PR
```

**Explicitly NOT changed:** no behaviour change is intended or expected; this is a robustness /
intent-documenting fix, and it pins the value we already had. Scoped robot-local (`robots/h1_2/`),
so the shared `deploy/include/` tree and the other robots are untouched, per the standing rule.

**Test plan.** Bridge run, confirm behaviour is byte-identical to the current build (it should be:
same value, now written explicitly). No hardware session needed on its own; fold it into the next
one. Optionally log the value once at startup so a future mismatch is visible in the session log.

## Validation experiment V1 (approved 2026-07-28): leg mass ALONE, then re-measure the lean

the user's call, and the right one: **the leg mass is the most factual of the errors** (weighed
directly: 73.70 kg, 18.60 kg/leg), whereas the wedge and the encoder zeros are inferred from
angle measurements with a few degrees of interpretation. So change that one variable, retrain,
re-measure on the robot, and let the result direct the rest.

**Scope: leg mass ONLY.** Decision item 1 applied; **item 2 (sole wedge) and Specs A/B/C stay
out** of this cut. That is deliberate, not an oversight: the wedge and the encoder zeros both
live in Component A (the `p_IMU - p_kin` residual), while the leg mass lives in Component B
(`p_kin` itself), so leaving them out keeps V1 a genuine single-variable test and does not
confound it.

**Protocol.** Retrain A0 at the same protocol as the current deployed baseline
`a0_v2_optB_rs20_baseline` (same seed, same `num_envs`, same iteration budget, per the
reproducibility rules) so the only intended difference is the plant. Sim-benchmark first: if
err_vx/vy/yaw or fall_rate move outside noise, some of any hardware change is policy luck rather
than mass, and a second seed is needed before reading the hardware result. Set the bridge plant
(`/opt/unitree_mujoco/.../h1_2_handless.xml`) to the same real mass so the G2 pre-check proxies
reality; note that this makes G2 numbers non-comparable to earlier G2 runs.

**Measure BOTH metrics, not just the lean.** This is what makes V1 diagnostic rather than merely
encouraging. The two components move **orthogonal** quantities:

| metric | what it is | V1 (leg mass) should | Spec B (encoder offset) should |
|---|---|---|---|
| `p_kin` (encoder feet-flat FK) | Component B, the leg configuration | **move toward 0** | not move |
| `p_IMU - p_kin` (residual) | Component A, the map error | **stay put** | move toward the wedge floor |
| `p_IMU` (the visible lean) | the sum | shrink by whatever `p_kin` gives up | shrink by the residual change |

**Pre-registered predictions** (baseline = mean of the four 2026-07-23/24 alpha==0 quiet stands:
`p_IMU` -0.0668, `p_kin` -0.0345, residual -0.0323 rad):

- `p_kin`: **-0.0345 -> ~0** if the policy fully learns to anticipate the droop, or at least
  halves. This is the quantity V1 is actually testing.
- residual: **stays at ~-0.032 rad**, unchanged. This is the control. If the residual moves, the
  A/B decomposition is wrong somewhere and the whole analysis needs revisiting.
- `p_IMU`: **-0.0668 -> about -0.032 to -0.045 rad**, i.e. the visible lean roughly halves, from
  ~3.8 deg to ~2 deg. It should NOT go to zero, and that is not a failure.

**Read-outs.**
- Both metrics move as predicted -> decomposition confirmed; proceed to Spec B (free, 60 s) for
  the encoder half, then the wedge XML fix, then Spec A for robustness.
- `p_kin` improves but the residual also moves -> the two components are not independent;
  re-derive before spending anything further.
- Nothing moves -> leg mass was not the driver of Component B despite being correctly measured,
  which would be a genuine surprise and would send the knee droop looking for another cause
  (revisit compliance, which we only ever partly separated).

**Cost/benefit note, stated honestly:** V1 costs one full A0 retrain plus a hardware session, to
test the component we are *most* confident about. Spec B tests the component we are *least*
confident about for free and in 60 s. Running Spec B first (or in parallel, since it needs no
retrain) would buy information sooner, and the metrics are orthogonal so it does not spoil V1.
Sequencing is the user's call; V1 as specified is clean either way.

## V1 read-out (2026-07-31): FALSIFIED — leg mass is not the driver

**Protocol as run.** Applied decision item 1 only (6 leg links/side x1.20787 on mass +
diaginertia, `ipos`/`quat` unchanged) to `h1_2.xml`, `scene_h1_2.xml`, and the external
`/opt/unitree_mujoco` bridge plant (noted separately, not repo-versioned; G2 numbers from
here on are not comparable to pre-cut G2 runs). Verified: total 73.384 kg, one leg 18.599 kg,
torso/pelvis/arms untouched, both XMLs compile. Retrained A0 at the exact deployed-baseline
protocol (`a0_v2_optB_rs20_baseline`: seed 42, 4096 envs, 10001 iters) as `a0_legmass_rs20_s42`.
Sim-benchmark showed err_vx/err_yaw up ~14-16% and CoT up ~7% vs a freshly-measured baseline
benchmark — large enough to check against noise before trusting a hardware read, so a second
training run at the original rs8 command-resampling protocol (`a0_legmass_rs8_s42`, matching
`a0_v2_optB_baseline`'s protocol) was added. The two V1 runs agreed with each other to within
0.001 on every tracking metric (far tighter than either does with its own native-protocol
baseline), while both showed the same elevated CoT direction — read as a real, reproducible
mass effect on tracking/energy, not seed luck. `fall_rate` = 0 throughout.

**Hardware measurement (60 s policy stand, spotted, inclinometer method identical to the
2026-07-28 session): upper-body lean 83.1°, thigh 74.9°, shank front 74° / back 76° — all
identical to the pre-cut baseline reading.** Matches the pre-registered "nothing moves at
all" branch exactly, not the halving predicted for `p_kin`/`p_IMU`.

**Verdict: leg mass was NOT the driver of Component B, despite being correctly weighed
(73.70 kg / 18.60 kg-leg, measured directly, not inferred).** This is the genuine-negative-
result branch called out in the pre-registration, not an inconclusive one — it is a repeat of
an already-validated measurement method landing exactly on the predicted null. **Does the
decomposition survive?** Partially, with a correction: the residual-stays-put control was
never separately exercised this round (V1 only re-measured the two segments already covered
by the "nothing moved" reading), so the A/B decomposition itself is not re-litigated here. What
does NOT survive is this ADR's framing of leg mass as a live causal factor in the lean — see
the correction below. The investigation redirects toward the encoder-zero side (Spec B) rather
than toward the sole-wedge/Spec-A path that a confirmed V1 would have unlocked.

**Correction to this ADR's own framing (the user, 2026-07-31):** the "other H1-2 sims/checkpoints
run a different, lighter leg mass and are therefore not a valid comparator" reasoning (see
Consequences, above) does not hold **as an explanation of the backward lean specifically**.
That framing implicitly treated leg mass as a live causal factor in the lean; V1 shows it is
not one, over this range. The 21%-lighter-leg fact about other published H1-2 sims is still
true and still worth stating as a general sim2real modeling-fidelity caveat — it just is not
grounds to dismiss a comparison *because of the lean*.

**Follow-up check (the user's hypothesis: is it hip_pitch specifically?).** Compared steady-state
standing leg-joint angles across the pre-cut baseline (proper sim), V1 (proper sim), V1 (G2
bridge replica), and V1's real-hardware flight-recorder `meas_q` from the same session as the
readout above. Sim-to-real agreement is tight (~1-3°) on every leg joint including hip_pitch —
this comparison does not isolate hip_pitch as an outlier. It also isn't the right tool for the
hypothesis: `meas_q` is each side's own belief about its joint angle (perfect by construction
in sim, exactly what's suspected of being biased on hardware), so agreement there shows
posture-tracking fidelity, not encoder-zero accuracy. Testing the hip_pitch-specific hypothesis
needs the geometric (feet-flat FK / per-segment inclinometer) method, which per this ADR's own
text cannot identify the per-joint split from the lumped `eps_total` alone. Full numbers, the
joint table, and a corrected `raw_q`-is-not-a-joint-angle note → the deploy journal in the
research KB.

**Practical consequence:** item 2 (sole wedge XML) and Spec A do NOT get an automatic green
light from this result — V1 tested the leg-mass branch of the decomposition, and it came back
null. The next step is Spec B (encoder-zero, free, no retrain) and per-joint identification of
where that error concentrates, not the wedge/Spec-A path.

## Probe P1 (2026-07-31): the lean transfer function, measured in sim

V1 left the encoder-zero branch *plausible but unquantified* — the geometry said how big the
offset is, nothing said how much lean an offset of that size actually produces. P1 measures it
directly, in sim, with no retrain and no hardware.

**Method.** New `scripts/play.py --probe-lean N` (sibling of the `--diagnose-*` family). It
builds the env, pins the twist to zero (`--eval-cmd-vx 0`), writes a **deterministic**
`encoder_bias` onto a chosen leg joint group (overwriting the startup event's random sample),
runs N steps and reports the steady-state base pitch from `projected_gravity_b`. The load-bearing
detail is that it **flips the `joint_pos` obs term to `biased=True` for the eval only**: with the
training default (`biased=False`) mjlab shows the policy the TRUE joint angle, so gravity and
encoders never disagree and the bias is nulled by a constant action shift — the probe would
measure nothing. With the flag on, the reported angle equals the command and only the true joint
is displaced, which is what a real servo does. mjlab's `target = action - encoder_bias` means a
bias `b` reproduces a deploy-side Spec B `joint_offset` of `-b`.

Run on the **deployed** A0 checkpoint (`a0_v2_optB_rs20_baseline/model_10000.pt`), stock plant
(the V1 leg-mass edit was found still applied to the training XMLs on 2026-07-31 and reverted
first), 64 envs x 2 seeds x 400 steps, bias swept over +-0.04 rad. Cross-seed spread <= 2e-4 rad.

| bias location | d(pitch)/d(bias) | chain offset needed to explain the observed lean | vs the 2.74° measured geometrically |
|---|---|---|---|
| `ankle_pitch` | **+1.684** | +0.0328 rad (1.88°) | −31% |
| `knee` | +1.287 | +0.0430 rad (2.46°) | −10% |
| all three, equal split | +1.263 | +0.0448 rad (2.57°) | −6% |
| `hip_pitch` | +0.733 | +0.0754 rad (4.32°) | +58% |

Unbiased sim stand = **−0.0115 rad (−0.66°)**, not zero: this policy has its own small backward
lean, so only **−0.0553 rad (−3.17°)** of the real −0.0668 rad needs a plant explanation. All
"needed" figures are that delta, not the raw lean.

**Four things this establishes.**

1. **The mechanism is quantitatively sufficient — the first time that has been shown.** Feeding
   the *independently measured* chain-sum discrepancy (2.74°, from the inclinometer + feet-flat
   FK) through the measured equal-split gain predicts a stand pitch of **−0.0719 rad (−4.12°)
   against −0.0668 rad (−3.83°) observed: 8% error.** Geometry and dynamics were measured by
   completely different methods and agree. The sign also matches: a *positive* `joint_offset`
   (true = reported + offset) is what produces backward lean, and positive is the sign measured.
2. **Superposition holds**, so the probe is trustworthy and composable: the mean of the three
   single-joint gains is 1.235 vs 1.263 measured for the equal split, **2.3% apart**. Any
   distribution's effect can be predicted from the three single-joint gains as `Σ w_j·g_j`.
3. **The offset is NOT concentrated at the ankle.** This is a genuine narrowing, and it cuts
   against the standing A/B-ankle suspicion. Requiring both the measured 2.74° total *and* the
   observed lean pins the effective gain at **1.157**, which sits between the hip (0.733) and
   knee (1.287) gains. A purely ankle-located 2.74° would over-produce the lean by ~1.4x; a
   purely hip-located one cannot reach it at all with only 2.74° available. Knee-dominant or
   roughly-even distributions fit both numbers to within ~10%. **Falsifiable prediction for the
   per-joint hardware measurement.**
4. **The ankle channel has 2.3x the leverage of the hip** (1.684 vs 0.733), and the **foot-sole
   wedge enters through the ankle channel** — a toe-up sole is kinematically indistinguishable
   from an `ankle_pitch` offset for the standing attitude. So the wedge and any ankle encoder
   zero are **not independent contributions to be summed separately**, as the Amendment above
   implicitly treats them; they are one channel, and P1 caps that channel's total well below
   2.74°. This tightens item 2's expected benefit rather than expanding it.

**Policy-dependence, and a pre-registered prediction for the policy-swap test (the user's
suggestion).** Re-running P1 on `a0_v2_optB_baseline` (rs8, same plant, different command
resampling) gives a **measurably lower** sensitivity: chain gain **1.031** (vs 1.263, −18%),
ankle gain 1.298 (vs 1.684, −23%), unbiased stand −0.0069 rad (−0.40°). So the lean is *not*
purely a plant property — the same calibration error transfers ~20% less through the rs8 policy.
Carried through the same 2.74° chain offset:

| policy | unbiased sim stand | chain gain | predicted hardware stand pitch |
|---|---|---|---|
| rs20 (deployed) | −0.0115 rad | 1.263 | −0.0719 rad (−4.12°) — **observed −0.0668 (−3.83°)** |
| rs8 | −0.0069 rad | 1.031 | **−0.0562 rad (−3.22°)** |

**Prediction: swapping to the rs8 policy should reduce the stand lean by ~0.9°, to about −3.2°.**
That is at the edge of inclinometer resolution (±0.5°) but comfortable for the IMU. A null (rs8
leans the same as rs20) falsifies the gain model and would say the lean is set by something
outside this channel; a ~0.9° reduction confirms it and hands over a free partial mitigation.

**What P1 does NOT show.** It does not locate the offset per joint (only constrains the
distribution, point 3), does not prove the error is encoder-zero rather than any other source of
the same chain discrepancy, and does not validate `joint_offset` on hardware. It measures a
transfer function; the hardware session still decides.

## Spec A trained and measured (2026-07-31/08-01): the biased-observation policy

the user's call, against my earlier "defer it to a cut" position. The reasoning that overturned mine:
the policy has **never once seen encoders and gravity disagree**, because `joint_pos_rel` defaults
to `biased=False` and every one of our configs took the default. P1 then made the retrain
self-validating — the gain can be re-measured in sim before spending a hardware session — which
removed the "unquantified bet" objection that had justified deferring it.

**Run:** `a0_biasedobs_rs20_s42` (`logs/rsl_rl/h1_2_velocity_v2/2026-07-31_13-31-05_...`), seed 42,
4096 envs, 10001 iters, rs20 resampling (3.0, 20.0) — the deployed keeper's exact protocol, so
`a0_v2_optB_rs20_baseline` is the like-for-like comparator. `bias_range` left at ±0.015 rad so the
flag is the only variable. Config snapshot verified in `params/env.yaml`: `biased: true` at line
398 (actor) and 526 (critic).

**Actor AND critic, deliberately.** An intermediate version gave the critic its own unbiased term
on privileged-critic grounds; that was wrong and was reverted. The encoder bias is a per-env hidden
constant that changes *what the actor does*, so a critic given the true angle **but not the bias**
cannot distinguish two envs with identical true state and different biases — its target becomes
less well-defined, not more. Privileging the critic properly would mean giving it the true angle
*and* the bias vector, not the true angle alone. The `**actor_terms` spread shares term objects, so
inheriting the flag is both the correct behaviour and what Spec A always said; a comment now records
that this is intentional rather than an accident of the spread.

**Rewards are unaffected either way** (checked, not assumed): every reward term reads
`asset.data.joint_pos` from sim state directly, and `joint_pos_biased` is a separate property used
only by the observation term. The `projected_gravity` obs term is also separate and stays true.
So the flag cannot skew any reward.

**P1 re-run — the gain fell across the board, and crossed below unity:**

| bias at | baseline rs20 | biasedobs rs20 | change |
|---|---|---|---|
| all three, chain sum | **1.263** | **0.772** | **−39%** |
| `ankle_pitch` | 1.684 | 1.231 | −27% |
| `knee` | 1.287 | 0.835 | −35% |
| `hip_pitch` | 0.733 | 0.245 | −67% |

**The qualitative flip is the result:** the baseline *amplifies* a static calibration offset
(chain gain 1.263 > 1), the biased-observation policy *attenuates* it (0.772 < 1). The unbiased sim
stand is unchanged (−0.0113 vs −0.0115 rad), so this is not the policy simply standing differently
— its intrinsic posture is identical and only its rejection of offsets improved. Single variable,
clean read.

**It is free.** Sim benchmark vs the baseline, 64 envs x 600 steps x 2 seeds: `err_vx` −2.7%,
`err_vy` −0.2%, `err_yaw` +4.6%, `action_rate` **−4.8%**, `orient_dev` −1.6%, `CoT` −1.0%,
`gait_match` −0.1%, `fall_rate` 0.0 both. All inside the measured 2-seed spread (±5.8% CoT /
±1.2% act), with action-rate and forward tracking mildly better. Training cost was likewise small
and in the expected direction for a harder, more partially-observed task: mean reward 43.90 vs
45.09 (−2.6%), episode length 987.8 vs 1000.0 (−1.2%).

**Pre-registered hardware prediction**, carrying the unchanged 2.74° measured chain offset through
each policy's measured gain:

| policy | unbiased sim stand | chain gain | predicted hardware stand pitch |
|---|---|---|---|
| rs20 baseline | −0.0115 rad | 1.263 | −0.0719 rad (−4.12°) |
| rs8 | −0.0069 rad | 1.031 | −0.0562 rad (−3.22°) |
| **biasedobs rs20** | −0.0113 rad | **0.772** | **−0.0482 rad (−2.76°)** |

So **~1.4° less lean than the deployed keeper**, and ~0.5° less than the rs8 session. Note this is
mitigation, not a cure: the policy still transfers **61%** of the offset it did before, because the
underlying plant error is untouched. Spec B remains the thing that removes the input.

**This is now the first data-backed argument for widening `bias_range`.** At ±0.015 rad i.i.d. the
3-joint chain sum has std 0.015 against the real 0.0317, so the true defect sits at ~2.1σ (about
1.5% of envs) — the policy is being asked to generalise to a case it almost never sees. A widened
range (or a per-leg *correlated* component, which matches the real defect's structure better than
i.i.d. does) is the obvious next lever if the hardware read-out confirms the sim gain drop but
falls short. Do not widen speculatively before that read-out.

**Status:** Spec A stays PROPOSED as a *default*. `biased=True` is currently in the working tree on
the shared term and must be reverted before any WL-D training, or that batch silently diverges from
`arm4d`. The run's own `params/env.yaml` records what it trained with, so reverting loses nothing.

## Harness-support hypothesis: FALSIFIED (2026-08-01)

the user's alternative explanation for the rs8 session's reduced lean: rs8 leaned less because it hung
more on the safety harness, making the improvement an artifact rather than a policy property.

**Test.** Per-leg vertical GRF from the 3 sagittal torques (hip_pitch, knee, ankle_pitch), which
are three moment balances in three unknowns (`f_z`, `f_x`, `x_cop`) and therefore exactly
determined. `tau_est` (the robot's own estimate, not `kp*err`, which over-reads knee torque ~1.3x),
MuJoCo FK for joint positions from the measured encoders + IMU attitude, quiet-stand filter
`max|dq| < 0.1` over the 12 leg joints. **The leg links' own weight must be in the moment balance**
— omitting it (first pass) inflated the `Σf_x ≈ 0` self-check residual to ~25 N and put the totals
20-30% low; including it dropped the residual to ~11 N (1.5% of body weight) and brought both of
the 2026-07-31 sessions to within 3% of the directly weighed 73.70 kg. That agreement is what
licenses the comparison.

| session | policy | total leg load | vs real 73.7 kg | R/L | stand pitch |
|---|---|---|---|---|---|
| 2026-07-31 10:32 | rs20 keeper | 72.9 kg | 98.9% | 1.064 | −4.98° |
| 2026-07-31 12:13 | **rs8** | **75.9 kg** | **102.9%** | 1.081 | **−3.15°** |
| 2026-07-28 10:13 | rs20 baseline | 95.5 kg | 129.5% | 1.053 | −5.41° |

**Verdict: falsified.** Harness support can only *unload* the legs, so the hypothesis predicts rs8
carries materially less than body weight. It carried **103%** — slightly more — and **+3.0 kg more
than the keeper session while leaning 1.8° less**, the opposite sign. Robust to the leg-mass
question: at the corrected 1.20787 scale the numbers move only to 75.0 / 77.9 / 97.1 kg and the
ordering is unchanged. So the rs8 lean reduction stands as a policy property, consistent with the
P1 gain difference (chain gain 1.031 vs 1.263).

**Two side findings.**

1. **The 2026-07-28 session read 130% of body weight**, ~22 kg of excess downward load. That was
   the inclinometer session, i.e. the one with hands on the robot taking physical measurements —
   the most likely explanation, and a reason not to treat 07-28's absolute stand numbers as a
   clean unattended stand. (It does not affect that session's *attitude* conclusions, which came
   from the inclinometer and IMU, not from load.)
2. **The open L/R asymmetry is a plant property, not a learned one.** The right leg carries
   5.3-8.1% more in all three sessions across two independently trained policies. That rules out
   policy idiosyncrasy and narrows it to calibration or geometry — the same channel Spec B and the
   per-joint identification (step 2) address.

Tooling: the solve lives in the job scratch dir, not the repo; promote it to `scripts/` if the
per-leg load becomes a recurring measurement (per the generic-tooling-placement rule).

## Spec A hardware read-out (2026-08-03): FALSIFIED, and it exposed a confound in the method

`a0_biasedobs_rs20_s42` deployed to the real robot, same 60 s spotted stand protocol.

**The prediction failed.** Inclinometer read **83.4°** (= 6.6° lean) against the 83.1-83.2° baseline
(6.8-6.9°): a 0.2-0.3° change, inside the instrument's ±0.5° resolution. IMU, same dq-filtered
statistic as every earlier table: **−5.51° vs the keeper's −6.13°**, a 0.62° improvement against
**1.36° predicted**, and inside the 1.0° between-session spread already measured on two nominally
identical baselines. No effect worth claiming.

**And the policy got worse.** The operator observed it swinging into steady state and taking stabilising
steps. That is quantified on the zero-command rows (`|cmd|<0.05`, `alpha==0`):

| policy | still (`max|dq|<0.1`) | mean `max|dq|` | dq-filtered stand pitch |
|---|---|---|---|
| rs20 keeper | **91.6%** | 0.057 | −6.13° |
| rs8 | 62.8% | 0.279 | −3.97° |
| **biasedobs** | **76.0%** | **0.243** | −5.51° |

**4.3x the joint motion at zero command** vs the keeper. Sim showed none of this (`fall_rate` 0,
`action_rate` −4.8%, `orient_dev` −1.6%).

### The confound: stand pitch is not comparable across policies

Across the three policies, **stillness predicts the lean at r = −0.96, while the P1 chain gain
predicts it at r = −0.25 — and the gain model gets the rs8-vs-biasedobs ordering backwards**
(it ranks biasedobs as the least-leaning; it is actually the middle one). A policy that fidgets
settles into a less-drooped stance and reads less lean, regardless of any calibration offset.
This holds even on dq-filtered rows, so it is not just moving samples contaminating the mean.

**This retracts the 2026-07-31 rs8 result as a confirmation of the gain model.** rs8 leaned less,
but it was also the *least* still policy of the three (62.8%), and stillness accounts for it better
than its gain does. Reported at the time as consistent with P1; it is not evidence for P1.

**What survives and what does not.** P1's *within-policy* measurement is untouched — inject a bias
into one fixed policy and measure the lean change is a controlled experiment. What does not survive
is using those gains to **compare different policies on hardware**, because policies differ in
stance behaviour and that difference moves the lean more than the gain does. Any future cross-policy
lean comparison must report stillness alongside, and preferably be avoided in favour of
within-policy interventions.

### Why Spec A backfired: the policy has no memory

`history_length=1` on both observation groups (`velocity_env_cfg.py:128,134`). With no history the
policy **cannot infer the encoder bias** — nothing in a single frame identifies it. So the only
strategy available under biased observations is to *down-weight `joint_pos` and lean harder on
`projected_gravity`*. In sim that is free: gravity comes from the ground-truth quaternion with
±0.05 uniform noise, no lag, no linear-acceleration coupling. On hardware the IMU has all three.
**Spec A traded encoder-error sensitivity for IMU-error sensitivity, and on this robot the IMU is
the worse of the two** — which explains both the fidgeting (chasing a noisy, laggy gravity signal)
and why the sim benchmark showed no cost.

Corollary: encoder-bias DR is only exploitable by an architecture that can *estimate* the hidden
parameter from a history. That is precisely the RMA adaptation module, i.e. **A2's design**, not
something a memoryless A0/A1 actor can use. Spec A should not be retried on a `history_length=1`
policy, and widening `bias_range` (the previously "data-backed" next lever) would make this worse,
not better — it deepens the same trade.

### Decisions

- **Spec A is REJECTED as a default.** Not adopted; revert the deployed policy to the keeper.
- **Spec B is now the clearly correct next step** and its advantage is structural, not just
  cheapness: it is a **within-policy** intervention (same policy, same stance behaviour, only the
  calibration changes), so the stillness confound that invalidates cross-policy comparison does not
  apply to it. It still needs the per-joint identification (worklines step 2) first.
- **Do not trust the 176%-of-body-weight leg load** computed for this session. Mean leg torque was
  11.6 Nm vs the keeper's 11.4 Nm — essentially identical — so a 78% jump in solved `f_z` is the
  static-equilibrium assumption breaking down under 4.3x the joint motion, not real load. The GRF
  solve is only valid on genuinely quiet stands.

## Spec B `joint_offset` sweep (2026-08-03): the channel WORKS, but the parameter is not constant

First unambiguously positive result of this investigation, immediately followed by the finding
that limits it. Inclinometer, 90° = upright, lean = 90 − reading.

**Run 1** — the offset drives the lean straight through upright, monotonically:

| chain offset | reading | lean |
|---|---|---|
| 0.000 rad (0.00°) | 84.5° | **+5.50°** back |
| 0.025 rad (1.43°) | 88.7° | +1.30° |
| 0.050 rad (2.86°) | 91.7° | −1.70° fwd |
| 0.075 rad (4.30°) | 94.0° | **−4.00°** fwd |

Hardware gain 2.20 °/°, zero-crossing at **0.0397 rad** — matching the operator's own 0.036 estimate.
At that value the robot was upright by inclinometer and by eye. **So the encoder-offset channel is
confirmed as the dominant lever on stand pitch, in the predicted direction, with a positive
`joint_offset` as derived.** The response saturates as it approaches upright (local slope
168 → 120 → 92 °/rad), which is expected — near upright, posture regulation and CoP limits take
over from the offset.

**Run 2, same offsets, different session — it does not reproduce:**

| chain offset | run 1 lean | run 2 lean (initial / settled) | spread |
|---|---|---|---|
| 0.036 rad | (null point) | +1.40° / +0.90° | — |
| 0.050 rad | −1.70° | +1.00° / +0.40° | **2.7°** |
| 0.075 rad | −4.00° | +0.00° / −1.30° | **4.0°** |

Null point moved from 0.0397 to **0.054-0.076 rad**; fitted gain 0.63-1.01 °/° vs run 1's 2.20
(sim P1 said 1.263, which run 2 brackets). The zero-offset baseline also varies across sessions:
inclinometer 83.1 / 83.2 / 83.4 / 84.5 → lean 6.9 / 6.8 / 6.6 / 5.5°, a **1.4° spread at nominally
identical configuration**.

**Reading (leading, not exclusive): the encoder zeros shift between sessions**, by roughly 1-2°.
That single assumption also retro-explains the ~1° between-session spread that dogged every earlier
lean measurement, and why the documented −3.83° baseline never reproduced against my −4.84 to
−6.13° recomputes. Not excluded: setup/placement differences, harness tension, thermal or
mechanical settling. n=2 sessions — this needs a third before it is more than the leading reading.

**Consequences.**

- **A fixed `joint_offset` in the config is NOT a durable fix.** It is right for one session and
  1-2° wrong for the next. Ship it only as a per-session calibration, set from a measurement at
  startup, not as a committed constant. The all-zero default stays.
- **Within-session drift is the stillness effect again**: all three run-2 readings moved 0.5-1.3°
  toward upright after the robot took a few steps, the same direction and magnitude as the
  independently measured stillness-vs-lean correlation (r = −0.96). Report stillness with every
  stand-pitch number.
- **This is now the second leg of A2's motivation** (the first being Spec A's failure): the latent
  is not observable from one frame *and* not constant across sessions, so neither DR-for-robustness
  nor offline calibration suffices, leaving online estimation. Written up in
  `doc/hrl/A2_ARMA.md` → "Hardware motivation".

### Confidence note on the Spec A mechanism (raised 2026-08-03: "maybe the conclusion is wrong")

The IMU-fragility explanation is the **current reading, not a proven mechanism.** It is the only
one that accounts for all three observations together (no lean change, 4.3x zero-command motion,
and sim showing neither), and it follows deductively from `history_length=1` — a memoryless policy
genuinely has no way to identify the bias, so shifting weight onto gravity is the only strategy
available to it. But these are not excluded:

- **Out-of-distribution rather than IMU-quality.** The DR range is ±0.86°/joint; the real chain
  error is 2-4° and, per the sweep above, session-varying. The policy may be failing because the
  real bias is outside what it trained on, not because the IMU is noisy.
- **The instability may be upstream of the lean**, not a side effect: a policy that fidgets more
  reads less lean (r = −0.96), so part of the "no improvement" could be the two effects cancelling.
- **Single seed.** One biased-obs run, one hardware session. The sim benchmark matched the baseline
  within the 2-seed spread, but the hardware behaviour was not replicated.

**The discriminating test was RUN (2026-08-03) and supports the IMU-dependence reading.**
`--probe-gravity-noise` sweeps corruption of the `projected_gravity` observation with the encoder
held clean (so nothing is OOD on the encoder side), zero command, 64 envs x 2 seeds x 400 steps:

| grav noise | keeper `leg|dq|` | biasedobs | b/k | keeper `act_rate` | biasedobs | b/k |
|---|---|---|---|---|---|---|
| ±0.05 (**training default**) | 0.015 | **0.027** | **1.80** | 0.240 | **0.348** | **1.45** |
| ±0.10 | 0.032 | 0.052 | 1.62 | 0.475 | 0.681 | 1.43 |
| ±0.20 | 0.075 | 0.177 | 2.36 | 0.916 | 1.268 | 1.38 |
| ±0.40 | 0.437 | 1.004 | 2.30 | 1.693 | 2.226 | 1.31 |
| ±0.80 | 1.442 | 1.791 | 1.24 | 2.835 | 3.341 | 1.18 |

**The biased-obs policy is 1.2-2.4x more agitated than the keeper at every gravity-noise level,
including the training default — with the encoder perfectly clean.** The only varied quantity is
gravity noise, so this is a direct measurement of gravity-channel dependence, not encoder OOD.

The refinement worth keeping: it is **not that it degrades faster per unit of added noise** (the
relative slopes are similar, and at ±0.80 the ratios even favour it because its own control is
already elevated) — it is that it sits at a **higher baseline agitation whenever gravity noise is
present at all**. That is why the standard benchmark missed it entirely: play mode sets
`enable_corruption=False`, so the benchmark runs with *no* observation noise, the one condition
where the two policies are equal. A real IMU always has noise, so on hardware the biased-obs policy
is permanently in its elevated regime. At ±0.80 it also fails qualitatively differently, pitching
**+6.83° forward** (a lunge/breakdown) where the keeper merely sags to −3.99°.

Residual uncertainty: single seed-pair per policy, one biased-obs run, and the OOD explanation is
weakened rather than excluded (the ±0.05 row is in-distribution for both, and the gap is already
1.8x there, which is the strongest single point against OOD). **The A2 argument does not depend on
the mechanism either way** — a memoryless policy failed to convert hidden-parameter DR into
hardware robustness regardless of which channel it over-trusted.

### Operational note (2026-08-03): the offset is being kept, for gait quality not pitch

The user is keeping a non-zero `joint_offset` (0.012/joint, 0.036 chain) in the A0 deploy config
for now — **not** to chase the last degree of lean, but because it gives visibly better posture and
a **more stable walk**. That is a stronger result than the stand-pitch metric alone and it is
mechanically consistent: the offset corrects where the controller believes the feet are, so it
improves swing-foot placement and CoP location, not just torso attitude. The lean itself is no
longer the binding constraint ("1° more or less is not that big of a deal now").

**Standing caution, given the session-to-session finding above:** a value that nulls one session
can be 1-2° off the next, and on the wrong side it *adds* to the lean instead of subtracting.
Re-check the quiet stand at the start of each session before walking, and treat a suddenly worse
lean as a stale calibration rather than a policy regression.

**Consequence for the regression suite:** `test_joint_offset_absent_or_present_defaults_to_27_zeros`
fires on the committed non-zero value. That is the guard working as designed, but it now blocks
`check_test_sensitivity.py` (baseline not green). The guard's *intent* — no silently-enabled
offset, and never confused with a policy-space constant — is still right; only its "must be all
zeros" form conflicts with a deliberately-set operational value. See the test-suite journal for the
replacement assertions when they land.
