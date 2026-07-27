# Leg-mass and foot-geometry nominal correction (Model v3)

**Status:** PROPOSED / staged, held for the next model version (the user's call 2026-07-24).
Not applied. The leg-mass edit was made, verified, and reverted; it is bundled with the
foot-geometry change and applied together at the next model-version cut, after the
2026-07-23-initiated real-hardware backward-lean investigation is closed by an inclinometer
session. Supersedes nothing in ADR-0005; it is the next checkpoint-invalidating nominal
correction on top of Model v2.

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

2. **Foot collision geometry -> real sole** (conditional on the inclinometer read). If the
   hardware inclinometer (foot sole vs pelvis, taken under the policy at cmd 0) shows the sole
   inclined relative to the ankle frame, replace the flat capsule sole with either the foot
   mesh (matching every reference) or a capsule plane tilted by the measured angle. If the sole
   reads level, the 1.8 deg lives in encoder zeros or downstream compliance and this ADR
   records that instead; the leg-mass change proceeds regardless.

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
