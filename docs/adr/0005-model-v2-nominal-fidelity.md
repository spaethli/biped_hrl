# Model v2: versioned nominal-model correction (hold-gain upper body + joint friction)

**Status:** accepted, amended 2026-07-08, finalized 2026-07-10 (Model v2 = OPTION B,
full-body hold gains incl. torso 300/3 at desired_kl=0.01; confirmed as the default A0
baseline at a full 10k budget by `a0_v2_optB_baseline` — see Amendment and final config below)

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
| 300/3 | **v1** 7.9/0.5 | derived | 0 | 0.39 @800, 0.67 @2500 (walks, FAST) |
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
2. **Torso stays v1 200/2.5 (interaction, not torso alone).** Correction (2026-07-09,
   `a0_torsov2_armsv1`): torso 300/3 with *soft v1 arms* walks fine and FAST (0.39 @800,
   0.67 @2500) — faster than arm-hold/torso-v1. The stall is an **interaction**: torso-
   hold and arm-hold are each individually tolerable (arm-hold slows lift-off to ~1400;
   torso-hold alone barely matters), but *together* they stall (0.2 @3200). Earlier rows
   that stalled all had BOTH holds, so they never isolated the torso. v2-final keeps
   torso v1 to remove the interaction risk (deploy holds joint 12 via `hold_joint_ids`
   regardless; hold gains 200/2.5 for lockstep). RESOLVED (2026-07-09, `a0_fullv2_kl01`):
   the stall is a learning-rate artifact, not a barrier — full v2 (torso 300/3 + arm-hold)
   converges under kl 0.01 (0.72 @3000, 0.743 @5000, 0 falls), it just lifts off slower
   than a clean config. **DECIDED 2026-07-10: kl 0.01 is adopted and full v2 (option B,
   torso at its true deploy gain 300/3) is THE default nominal** — the cleaner deploy story
   with no measurable tracking cost. Confirmed at a full 10k budget by `a0_v2_optB_baseline`
   (benchmark err_vx 0.089 / vy 0.109 / yaw 0.100, act_rate 0.63, 0 falls — indistinguishable
   from the v1-A0 target), with `a0_fullv2_kl01` (5k, err_vx 0.096) as the same-config
   sibling (tight spread → stable config, not a lucky seed). torso-v1 200/2.5 is retired
   as the interim safe choice (it only existed as a kl-0.005 hedge).
3. **Arm action scale stays derived (0.25*effort/kp), NOT the references' flat
   0.25.** With kp 120/80 arms, flat 0.25 lets exploration noise destabilize the
   robot (falls + action-std collapse) and never converges; the small derived
   authority converges and additionally matches split deploy, where arm commands are
   ignored. The references' flat 0.25 works with their lower arm kp (40-80) and
   different reward stacks; it does not transplant.

Final v2 nominal (OPTION B, chosen 2026-07-09 for deploy fidelity): legs v1, torso
**300/3**, shoulders 120/2, elbow+wrists 80/1, trained with **desired_kl=0.01** (required:
torso+arm hold stalls at 0.005, converges at 0.01), derived
scales, frictionloss 0, viscous_damping 0.001, 7 actuator groups, conservative URDF
effort limits (datasheet ceilings ~2-6x higher; noted in A2_ARMA.md §3). Under kl 0.01
the full-v2 baseline reaches 0.72 @3000 / 0.74 @5000, 0 falls (`a0_fullv2_kl01`); the
canonical baseline is the full 10k run **`a0_v2_optB_baseline` (`model_10000`)**: benchmark
err_vx 0.089 / vy 0.109 / yaw 0.100, act_rate 0.63, orient_dev 0.030, height_dev 0.024,
cot 0.531, 0 falls — on the v1-A0 target. All A1/A1a/A2 work rebases on this run.
Also learned: lift-off iteration scales with num_envs (the 06-09 v1 baseline used
8192 envs and lifted off at ~330; all 4096-env runs cross 0.3 at 550-800) — hold
num_envs fixed within any comparison and never judge stuck-vs-slow before ~2x the
expected lift-off.

## Amendment 2 (2026-07-17, WL-E): correction 1's stall attribution is FALSIFIED at kl 0.01

The bisect's frictionloss row ("0.12 @1846, stuck") ran at desired_kl 0.005 AND with
both holds active, so it never isolated friction. The WL-E control run
(`a0_v2_optB_fric0p1_kl01_s42`: optB config + frictionloss 0.1 on all joints,
kl 0.01, 10001 iters, 4096 envs, seed 42) settles it. Comparator note: the run
trained under the post-2026-07-14 command resampling (3, 20) s, NOT the (3, 8) s the
2026-07-10 `a0_v2_optB_baseline` saw, so its training-side twin is
`a0_v2_optB_rs20_baseline` (fric 0, same (3, 20)); the benchmark protocol itself is
unconfounded (play mode pins (3, 8) for every checkpoint):

| torso | arms | frictionloss | kl | result |
|---|---|---|---|---|
| 300/3 | hold | 0.1 | 0.005 | 0.12 @1846 (stuck; the confounded row above) |
| 300/3 | hold | 0.1 | **0.01** | lift-off ~1108, **0.81 @10k, bench err_vx 0.083 / vy 0.108 / yaw 0.087, 0 falls** |

Friction at kl 0.01 trains from scratch to at-least-baseline quality on both
comparators (rs20 twin: err_vx 0.085, CoT 0.537; 2026-07-10 optB: 0.089/0.109/0.100,
CoT 0.531; fric run act_rate 0.635 vs 0.63, orient/height dev equal). Held commands:
ss_err_vx 0.050 @0.5 (t90 0.56 s), 0.089 @1.0 (t90 0.86 s), 0 falls (rs20 twin:
0.055 / 0.077, so 0.5 slightly better, 1.0 slightly worse, both same class).
Lift-off slows ~1.5-2x (1108 vs 550-800), no stall. So correction 1's technical
basis ("friction stalls gait discovery") is void; friction was only ever blocked by
the kl-0.005 double-hold confound.

Deploy-side evidence (bridge replica, vendor plant, clearance metric added to
`scripts/bridge_replica.py` 2026-07-16): swing apex clearance is plant-INDEPENDENT
(walk-0.5 apex 0.061-0.068 m on vendor / old-harsh / training-nominal alike, ~35%
under the 0.10 m trained target), so the "barely lifts feet" look is a training-side
character unmasked by the vendor plant, not caused by it; the old harsh plant merely
damped the jitter (stand qvel_rms flat ~0.06 at 0-4 ms delay vs vendor 0.11→0.31).
The fric-0.1 policy gains only marginally in the bridge: clearance +4-6 mm, stand
qvel_rms at 4 ms 0.254 vs 0.308, full chain passes at 0/2/4 ms on vendor + stress.

Reference audit (WL-E, verified in-repo): both reference stacks train friction-free
AND with less rotor inertia than us (unitree_rl_gym armature cfg 1e-3; CorelLab
h12_rma armature 0, no URDF <dynamics> in either), while our nominal carries
per-motor 0.002-0.04. So "adopt friction for reference parity" was never the right
frame; the honest frame is train->deploy/hardware parity (vendor sim and real joints
both have friction). Also found: CorelLab's deploy XML (`h12_locomotion_rma/
MujocoDeploy/h1_2/h1_2_handless.xml`) is `damping 1 / armature 0.1 / frictionloss
0.2`, i.e. exactly the old harsh bridge plant; that plant's damping masks jitter, so
"policies looked better on the old bridge" is not evidence of better transfer.

**Standing decision (Liam's ruling, 2026-07-17): frictionloss stays 0; fric 0.1 is a
sanctioned option, not adopted.** Adopting it costs nothing in-sim and buys a small
bridge robustness margin, but it invalidates every v2 checkpoint (A0 baseline, A1/A1a
keepers) like the v1->v2 rebase did; that trade stays open for a later rebase window.
The constants comment now cites this amendment instead of the falsified stall. The
audit's three falsifiable follow-ups (explicit-PD actuation, reference-strength
perturbation DR, correlated obs noise) are folded into the WL-D batch protocol as
arms 7-9 (`doc/hrl/worklines.md`).
