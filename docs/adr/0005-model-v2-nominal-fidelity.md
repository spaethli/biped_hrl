# Model v2: versioned nominal-model correction (hold-gain upper body + joint friction)

**Status:** accepted, amended 2026-07-08 (see Amendment: frictionloss and torso hold
gain removed from the training nominal after a 9-run bisect; final config below)

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

## Amendment (2026-07-08): bisect findings and final configuration

The original v2 stalled A0 gait discovery (stand-lean, tracking ~0.11 vs v1 ~0.70 at
iter 1500). A 9-run bisect on identical code/recipe (4096 envs, seed 42, local RTX
5070) isolated the causes. Tracking reward at the stated iteration, all runs:

| torso | arms | arm scale | frictionloss | result |
|---|---|---|---|---|
| v1 200/2.5 | v1 7.9/0.5 | derived | 0 | 0.32 @800 (control, walks) |
| v1 | v1, 7-group split | derived | 0 | 0.52 @1000 (walks; structure innocent) |
| v1 | hold 120/80 | derived (0.04-0.08) | 0 | 0.51 @2000 (walks, lift-off ~1400) |
| 300/3 | hold | derived | 0.1 | 0.12 @1846 (stuck) |
| 300/3 | hold | derived | 0 | 0.20 @3200 (stuck-slow) |
| 300/3 | hold | flat 0.25 | 0 | 0.14 @3500 (stuck) |
| v1 | hold | flat 0.25 | 0 | 0.10 @2500 (stuck, plateaued) |

Three corrections to the original decision:

1. **frictionloss stays 0 in the training nominal.** The fric-vs-nofric pair differed
   marginally (both stuck via the gain issue); the references' 0.1 lives only in their
   deploy-sim XMLs, their training URDFs carry no joint friction at all (likely
   pipeline artifact, not principle). Joint friction belongs to the deploy-sim
   robustness eval and the A2 Wide DR set, where its unknown true value is randomized
   over instead of guessed.
2. **Torso hold gain 300/3 rejected; torso stays v1 200/2.5.** It stalls gait
   discovery in both scale variants (waist counter-rotation is load-bearing for
   stepping). Deploy still holds joint 12 via `hold_joint_ids`; the deploy hold gains
   are set to 200/2.5 for lockstep.
3. **Arm action scale stays derived (0.25*effort/kp), NOT the references' flat
   0.25.** With kp 120/80 arms, flat 0.25 lets exploration noise destabilize the
   robot (falls + action-std collapse) and never converges; the small derived
   authority converges and additionally matches split deploy, where arm commands are
   ignored. The references' flat 0.25 works with their lower arm kp (40-80) and
   different reward stacks; it does not transplant.

Final v2 nominal: legs v1, torso v1, shoulders 120/2, elbow+wrists 80/1, derived
scales, frictionloss 0, viscous_damping 0.001, 7 actuator groups, conservative URDF
effort limits (datasheet ceilings ~2-6x higher; noted in A2_ARMA.md §3). Known cost:
gait lift-off ~1400 iters at 4096 envs (vs ~600 for v1); converged quality matches.
Also learned: lift-off iteration scales with num_envs (the 06-09 v1 baseline used
8192 envs and lifted off at ~330; all 4096-env runs cross 0.3 at 550-800) — hold
num_envs fixed within any comparison and never judge stuck-vs-slow before ~2x the
expected lift-off.
