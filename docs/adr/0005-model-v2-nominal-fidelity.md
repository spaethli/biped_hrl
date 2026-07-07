# Model v2: versioned nominal-model correction (hold-gain upper body + joint friction)

**Status:** accepted (grilled 2026-07-07; implementation pending, sequenced before A1a S3)

## Context

Multiple groups report that policies trained in mjlab-style MuJoCo setups fail on the
real H1-2 while IsaacGym-based pipelines transfer, blaming actuator and friction
modeling. A survey of every H1-2 repo in the workspace (`unitree_rl_gym` IsaacGym
baseline; the CorelLab stack: `h12_rma`, `h12_locomotion_rma`, `h12_adaptive_policy`,
`h12_ros2_controller`, `h12_safety_layer`; plus the IsaacLab repos as secondary
references) found two defensible gaps in our nominal model and several non-gaps:

1. **Upper-body PD gains.** Our elbow/wrist/shoulder-yaw group runs kp=7.9, kd=0.5
   (`src/assets/robots/unitree_h1_2/h1_2_constants.py`), and our deploy YAMLs send the
   same values to the hardware. Every reference uses 40-240. Crucially, kp=7.9 only
   works in a sim with zero joint friction; real stiction swallows it.
2. **Joint dry friction.** Our XML has `frictionloss` unset (= 0). Every reference XML
   (Unitree-derived) sets `<joint damping="0.001" armature="0.01" frictionloss="0.1"/>`
   on all joints.
3. **Non-gaps.** Leg gains (200/300/40 + kd 2.5/4/2), 50 Hz control rate, and effort
   limits (torso 200, shoulder 40, distal 18/19, verified against the official URDF)
   already match all sources. Armature values disagree *between* the references (0.01
   XML vs 1e-3 IsaacGym cfg), so ours (per-motor 0.025/0.04/0.005/0.002) are kept.

Deployment context that shaped the decision: the CorelLab stack confirms a **split
deploy** (verified in code, not just README): `h12_rma` trains 27-dof but its
`deploy_real` config has `num_actions: 12`, holding joints 12-26 at fixed targets with
stiff gains (waist 300/3, shoulder 120/2, elbow+wrist 80/1, identical to Unitree's own
`deploy_real` set), while `h12_safety_layer` (`split_mode`) merges an RL leg stream and
a ros2 upper-body stream into `rt/lowcmd`. We intend the same split on our robot.

## Decision

Create **Model v2**, a versioned correction of the nominal model, applied uniformly to
all architectures A0-A4:

- **Upper-body gains -> hold gains**: waist 300/3, shoulder(pitch/roll/yaw) 120/2,
  elbow + wrists 80/1. Pure PD (CorelLab's real upper-body controller also carries an
  integral term, which a sim PD actuator cannot represent; the h12_rma deploy hold set
  does not need it). Legs unchanged.
- **Joint friction**: `frictionloss = 0.1` on all joints (+ passive `damping = 0.001`
  for XML parity). The 0.1 is the vendor placeholder, adopted as nominal; A2 Wide DR
  randomizes around it.
- **Unchanged**: armature, effort limits, Narrow DR set (base-mass DR and actuation
  delay stay A2 Wide-DR extrinsics), rewards, observations. The v2-vs-v1 delta is
  purely the nominal plant, so benchmark differences attribute cleanly.
- **Upper body stays in the action space** (27-dof RL + posture penalty), trained at
  hold gains so split deploy sees the same arm dynamics that training saw.
- Derived consequence, accepted: `H1_2_ACTION_SCALE = 0.25 * effort / kp` shrinks arm
  authority (elbow 0.25 -> ~0.056), consistent with arms-mostly-hold; the actuator
  groups split (torso and shoulders leave their current groups).

**Validation gate before rebasing A1/A1a**: (1) smoke run; (2) full A0 retrain on v2,
same protocol/seeds, benchmark within noise of v1-A0 on err_vx/vy/yaw + fall_rate;
(3) cross-eval v1 checkpoint in v2 env and v2 in v1, the drop quantifies the modeling
gap (thesis-reportable); (4) v2-A0 ONNX through the C++ MuJoCo bridge with regenerated
gain YAMLs.

**Sequencing**: v2 lands now, before A1a S3, so no A1a result is ever v1-based and A2
builds on v2 directly.

## Alternatives considered

- **Fold everything into A2 Wide DR**: no retrains, but the nominal stays knowably
  wrong and A0/A1 deploy results stay confounded; DR cannot rescue a policy whose arms
  trained at 7.9 gain against a frictionless plant.
- **12-dof action space (arms clamped in sim)**: cleanest leg-transfer match, but
  changes obs/action dims, breaks every warm-start, and forecloses upper-body use in
  later architectures.
- **CorelLab real safety_split gains as sim gains**: hardware-proven, but they belong
  to a scripted PID (with ki) upper body, and the high kd (8-12) is risky under RL
  exploration noise.

## Consequences

- All v1 checkpoints, including the A0 warm-start
  (`logs/rsl_rl/h1_2_velocity/2026-06-09_08-16-27/model_10000.pt`), are invalidated as
  baselines; A0 retrains first, then A1/A1a rebase on the new A0.
- Every deploy YAML carrying the 27 gain values (velocity + velocity_hrl, sim + real)
  must be regenerated together with the sim change; sim/deploy gain lockstep is the
  invariant this ADR exists to protect.
- Cross-boundary comparisons (v1 run vs v2 run) are invalid; the findings ledgers keep
  v1 results as historical evidence only.
- Split mode (step 3b) is implemented alongside the YAML regeneration: `hold_joint_ids`
  in the `deploy_real` params (torso + arms, ids 12-26) makes held joints track
  `default_joint_pos` instead of the policy action (robot-local change in
  `State_RLBase.cpp` / `State_RLHRL.cpp`; observations still see all 27 joints). The
  bridge gate runs full-forward vs split-forward on the same v2-A0 checkpoint; the
  difference measures the arm-freeze mismatch accepted by the split-deploy decision.
