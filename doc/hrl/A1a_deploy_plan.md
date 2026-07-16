# A1a stage D: sim-to-real deployment validation plan (v1, APPROVED 2026-07-14)

**Execution status (2026-07-15):**
- **W1 ✅ code+build** — `State_RLHRL` keeper support (hl_vel input, velgoal 3+1 output
  mapping, HL-owned phase clock via the shared YAML params node, `pin_period`, ONNX
  input/output dim guards at load); keeper ONNX pair staged in `exported/` (dims verified:
  HL 94→4, LL 96→27). SAFETY_FILTER=1 on the HRL state. Pending: G2.0 bridge load smoke +
  the 7-dim-checkpoint regression run.
- **W2 ✅ (revised 2026-07-15)** — first attempt (training-nominal plant) FAILED on the
  live bridge (user session: FixStand lean + heels off, A0 twitchy/unstable). Root cause
  found via a headless bridge replica (same scenes, explicit PD @500 Hz, elastic band,
  A0 ONNX in the loop): **feedback latency**: training-nominal plant + 2 ms delay =
  qvel_rms x30 + falls; old harsh plant insensitive to 6 ms. Fix: default scene =
  **vendor reference** (0.01/0.1/0.001, clean base) — passes the full procedure chain
  (FixStand→takeover→stand→0→1.0 step→stop→walk) at 0/2/4 ms delay in the replica,
  including the historical from-stand killer. Stress scene unchanged. Process note: the
  original W2 was marked done on numeric verification only, skipping its own behavioral
  pass criterion — that skipped check is exactly what failed. Behavioral re-check in the
  REAL bridge still pending (next session).
- **W3 ✅ code** — `hrl_telemetry.h` (50 Hz cmd/s/V*/period/hip CSV, on when
  H1_2_SAFETY_LOG set) + `scripts/deploy_gate_analyzer.py` (`[DEPLOY-GATE]` json;
  validated on synthetic data, stride proxy recovers a planted 0.63 s). Also: number-key
  speed presets 1-9/0 in `keyboard_velocity_commands` (old a/d strafe ±1.0 was outside
  the trained ±0.5 — fixed).
- **W4 ✅ PASSED** — `scripts/onnx_parity.py`; keeper LL 4.2e-5 / HL 8.5e-6 max abs diff,
  metadata all OK; A0 optB passes too. Caveat: compare CPU-vs-CPU (deploy runs CPU).
- **W5 ✅** — both velocity_hrl YAMLs carry the keeper structure (delta + velgoal +
  cadence + hl_obs_vel; `deploy_real` de-staled, `pin_period: 0.625` for bring-up).
- **G1.2 ✅ for the keeper** (via W4). Next: G2.0 bridge shakedown (needs the sim GUI
  session), G1.1 re-run only for new M2 candidates.
- **G2.0 ✅ / G2.1 partial (2026-07-16 live):** keeper loads, runs, STANDS at cmd 0 in
  the real bridge after the pelvis-velocity fix (post-mortem issue 2). Stepping in
  place unsettled = the known TD3-HL hold degeneracy (training thread). vx=1.0 press
  fell via the safety filter's joint freeze (post-mortem issue 3, fixed; rebuilt).
  Re-run pending with the clamp-filter build; use number keys (5 = held 0.5) before w.
- **Replica permanent (2026-07-16):** `scripts/bridge_replica.py` = the headless bridge
  replica (A0 + HRL, both scenes, delay knob, band, full chain, reads deployed YAML +
  exported ONNX). Pre-session sanity + plant/latency A/B instrument. Validated: A0 and
  keeper both pass the chain at 2 ms on the vendor plant.

Operational gated checklist for `A1a_plan.md` Plan v2 stage D. Grilled 2026-07-14.
Execution is gate-by-gate; no gate starts before its blockers pass. The battery
(phases G1+G2) is checkpoint-agnostic: it runs once now on the current keeper
(pipeline shakedown) and re-runs cheaply on every M2 candidate. Hardware (G3) only
on the M2 winner.

## Decisions locked (grill session 2026-07-14)

1. **Bridge plant: vendor reference + stress variant** (REVISED 2026-07-15; the
   original "faithful to training nominal" choice was implemented and FAILED on the
   robot bridge — FixStand leaned/twitched, A0 unstable). Diagnosis: the bridge's
   explicit 500 Hz PD carries ~2-4 ms real feedback latency; on a dissipation-free
   plant 2 ms of delay is violently destabilizing (replica delay sweep, W2 row), while
   mjlab's implicit servos never see latency at all — so the training-nominal plant is
   NOT faithful to training dynamics in this bridge, and real hardware (which has joint
   friction + multi-kHz firmware PD) is better proxied by the vendor values. Default
   scene = vendor (`armature 0.01 / frictionloss 0.1 / damping 0.001`, clean free
   base); stress scene keeps the old harsh values (incl. its fictional base damper) as
   the robustness gate. Consequence for gates: G2.2 stays a PARITY gate but the
   reference tolerance must absorb a plant gap (vendor ≠ mjlab nominal); the honest
   comparison is bridge-vs-bridge across candidates + mjlab as directional reference.
   Resolves the ADR-0005 "until deploy" deferral; amend the ADR when the user confirms.
2. **Estimator-noise DR: both paths in M2.** M2 trains a DR-on variant
   (`goal_state_noise.enable=true`, the #8b-validated bias config) and a DR-off
   variant; the deploy candidate is picked by benchmark + the bridge noise gate
   (G2.5). Note: the current keeper trained with `goal_state_noise.enable: false`.
   Scope clarification (2026-07-14): this DR corrupts only the estimator-supplied
   goal-state columns (base vx, vy; optionally height) that the whole LL policy
   conditions on. It is NOT joint/plant randomization of any body part; whole-plant
   DR is A2's Wide DR, out of stage-D scope.
3. **Real velocity/height estimate: verify onboard first** (gate E1). Fallbacks in
   order: custom leg-odometry estimator, absolute-V* LL retrain. Fallback choice
   returns to the user if E1 fails.
4. **Cadence: full HL-period plumbing + pin knob.** The C++ bridge gets the complete
   HL-owned period path plus a `hrl.pin_period` deploy-YAML override. First hardware
   walks run pinned; the HL gets period authority as the stretch tier.
5. **No treadmill this campaign.** First hardware task = free walking around the room.
   Held-command gates are PARITY gates (bridge vs the same checkpoint's mjlab
   held-command numbers), not absolute tracking bars; tracking is secondary until A2.
6. **Safety: SAFETY_FILTER=1** on `State_RLHRL` for all hardware sessions, flight
   recorder on; gantry/harness for stand handoff and first walks.
7. **Arms: split first, live later.** First hardware walks run split deploy
   (`hold_joint_ids` 12-26); live arms (the M2-calmed policy) are an extension after
   stable walking. Bridge gates test both modes.
8. **Stage D done:** required tier = repeatable free walk around the room (>= 2 min
   continuous, multiple sessions), clean from-stand entry, no safety-filter
   engagements, arms calm. Stretch tier (reported if reached) = un-pinned HL cadence
   authority live on hardware. Stage D closes on the required tier.

## Recon facts this plan builds on (verified 2026-07-14)

- YAMLs + constants + desired_kl are COMMITTED and in optB lockstep (`6decd8b`);
  the "uncommitted working tree" note is stale.
- The keeper (`2026-07-09_10-04-50_*`, cot0.2 velgoal DIR kl01) restores as: td3,
  delta, tracking, `hl_velocity_goals_only=true`, `hl_cadence` source=hl, cot 0.2,
  kl 0.01, `goal_state_noise` OFF.
- `State_RLHRL.cpp` cannot run the keeper today. Three gaps: (i) HL input is
  `policy ++ command` but the keeper's TD3 trained on `policy ++ command ++ hl_vel(2)`
  (`td3.py:147-150`); (ii) the HL output dim check demands goal_dim(7) while the
  keeper emits 4 (3 velocity goals + period), and the velgoal mapping (orientation/
  height pinned to nominal) is absent; (iii) `gait_phase` is a fixed 0.6 s YAML clock,
  no HL-period integrator. The ONNX metadata already carries all needed fields
  (`hl_velocity_goals_only`, `hl_cadence_source`, `cadence_period_range`).
- `deploy_real.yaml` is stale for the keeper: `hl_target_mode: absolute` (keeper is
  delta), no velgoal/cadence/hl_obs_vel keys.

## Phase W: code groundwork (blocks everything downstream)

| Gate | What is tested / built | Pass criterion | Unblocks |
|---|---|---|---|
| **W1** | C++ keeper support in `State_RLHRL` (robot-local): append `hl_vel` (body-frame vx,vy from the sportmode estimate, same value used for `s`) to the HL input when configured; velgoal HL-output mapping (3 velocity dims -> velocity target cols, orientation/height targets = nominal, oracle path); 4th tanh dim -> affine map to `cadence_period_range` -> phase integrator overwriting the `gait_phase` slice of the LL obs; `hrl.pin_period` override. New YAML keys with absent-key back-compat (old 7-dim checkpoints run byte-identical). | Builds; regression: a pre-velgoal 7-dim checkpoint (e.g. the 2026-06-18 TD3 deploy pair) runs unchanged; keeper ONNX pair loads with no dim errors and steps in sim. | G2 |
| **W2** | Faithful plant in the LIVE bridge (**`/opt/unitree_mujoco`**, found 2026-07-15 — the in-repo `simulate/` copy is unused): `unitree_robots/h1_2/h1_2_handless.xml` joint defaults aligned to training nominal (per-joint armature; frictionloss 0; damping 0.001; clean free joint — the old defaults put damping 1/armature 0.1 on the BASE too). Harsh values preserved as `h1_2_handless_stress.xml` + `scene_stress.xml`, selectable via `robot_scene:` in the /opt config. Note: lives outside the thesis repo. | `a0_v2_optB_baseline` ONNX passes the vx=1.0-step-from-stand test in the faithful bridge (mjlab shows fall_rate 0.0 for this condition; the old bridge fell); stress scene loads and runs. | G2 |
| **W3** | Bridge telemetry: extend the flight recorder (robot-local) to also log the active command, sportmode velocity, HL target V*, and commanded period per tick; small analyzer (extend `safety_analyzer.py` or a sibling) computing achieved-vs-commanded err_vx/vy, falls, stride-period proxy, action-rate from the CSV. | A smoke bridge run produces the CSV + a `[DEPLOY-GATE]` style report with those metrics. | Quantified G2 |
| **W4** | ONNX parity script: run N random obs vectors through the torch actor(s) and the exported ONNX pair; compare; verify metadata fields against the run's `agent.yaml` structure. | max abs diff < 1e-4 on both nets; metadata matches (c, target mode, velgoal, cadence range, components). | G1.2 |
| **W5** | Deploy YAML refresh (`velocity_hrl/v0/params/`): keeper structure keys in both YAMLs (`hl_target_mode: delta`, `hl_obs_vel`, `hl_velocity_goals_only`, `hl_cadence` + range, `pin_period`), `deploy_real.yaml` de-staled. Gains/scales untouched (already optB lockstep). | C++ loads them; a deliberate mismatch (e.g. wrong goal_components) fails loudly against ONNX metadata. | G2 |

W1+W5 first (critical path); W2/W3/W4 parallelizable. Spec-first applies inside W1:
implementation follows the mapping above, matching the training-side semantics at
`hrl_runner.py` (hl_vel append order: policy, command, hl_vel) and `td3.py`
(tanh -> range affine map, same endpoints as the S1c test: g=-1 -> range lo, g=+1 -> hi).

## Phase G1: mjlab candidate battery (per candidate, ~30 min)

| Gate | What is tested | Pass criterion | Unblocks |
|---|---|---|---|
| **G1.1** | Deterministic bench (64x600x2) + held-command evals `--eval-cmd-vx 0.5` and `1.0` + goal probe. | fall_rate 0 in all evals; metrics recorded. The held-command ACHIEVED vx values become (a) the G2.2 parity reference and (b) the expected room-walk speed. | G1.2 |
| **G1.2** | ONNX export (`play.py --export-onnx`) + W4 parity script. | Parity < 1e-4; metadata consistent with `agent.yaml`. | G2 |

## Phase G2: C++ bridge gates (faithful scene unless stated; keyboard sim)

Every gate is logged via W3 telemetry; "no filter trigger" refers to the safety
filter compiled in (SAFETY_FILTER=1 in bridge too, same build as hardware).

| Gate | What is tested | Pass criterion | Unblocks |
|---|---|---|---|
| **G2.0** | Shakedown: FSM reaches RLHRL (`h`), both ONNX load. | No dim/FSM errors; LL acts. | G2.1 |
| **G2.1** | Stand: enter at cmd 0, 60 s. | No fall, no filter trigger; action-rate within ~1.5x of the same checkpoint's mjlab stand level (exact bound set at keeper shakedown and then frozen for M2 candidates). | G2.2 |
| **G2.2** | Held-command walk PARITY: held vx 0.3, 0.5, 1.0 (>= 30 s each), plus backward -0.3, strafe +-0.3, yaw +-0.5. | 0 falls; achieved vx within +-0.05 m/s of the same checkpoint's mjlab held-command achieved vx (G1.1); stride period within +-0.05 s. | G2.3 |
| **G2.3** | Command steps: 0 -> 0.5 -> 0; 0 -> 1.0 from stand (the historical bridge killer); walk -> turn -> stop. | 0 falls; clean return to stand. | G2.4 |
| **G2.4** | Split mode: repeat G2.1-G2.3 with `hold_joint_ids` 12-26 (the hardware config, run in sim via the deploy_real params). | Same bars as G2.1-G2.3. | G3 (config-wise) |
| **G2.5** | Estimator-noise robustness: G2.1 + G2.2@0.5 under `hrl.state_noise` velocity 0.05 and 0.1, height 0.02-0.05, orientation 0.01-0.02. Run for BOTH M2 variants (DR-on / DR-off). | 0 falls; bounded twitch (action-rate inflation vs clean run recorded; bound frozen after keeper shakedown). Output doubles as the M2 DR-arm pick. | M2 candidate selection, G3 |
| **G2.6** | Cadence modes: pinned (`pin_period` at the candidate's best map value) vs HL-owned, held vx 0.5. | Both walk 0-fall; HL-owned period stays in-range and plausible vs the candidate's stride map. | G3.5 (stretch) |
| **G2.7** | Stress scene (harsh plant): G2.2@0.5 held + G2.3 steps at 0.5. | 0 falls at 0.5 (BLOCKING); vx 1.0 from stand recorded but ADVISORY only. | G3 |

Exit rule: all blocking gates pass on the SAME candidate + the exact config
(YAML + ONNX pair + build) that goes to hardware.

## Phase E: hardware estimator gate (policy NOT in control; parallel to G2)

| Gate | What is tested | Pass criterion | Unblocks |
|---|---|---|---|
| **E1** | Onboard odometry availability + quality: in low-level mode on the real H1-2, log whatever publishes velocity/height (rt/sportmodestate or equivalent) during FixStand, manual perturbation, and (if available) a Unitree-controller walk; score noise std and bias drift against references (tape-measure walks, stopwatch). **Must be a BASE (pelvis) velocity estimate** (2026-07-15 finding: an IMU/torso-frame velocity destabilizes the LL via the goal channel; the bridge sensor was retargeted to pelvis for the same reason). | The topic exists in low-level mode AND supplies base-frame vx/vy with noise/bias within the DR-trained envelope (bias_range 0.1); height usable. FAIL -> decision returns to the user (custom estimator vs absolute-V* retrain). | G3 |
| **E2** | IMU convention: standing in FixStand, verify `a_world_z = (R a_imu).z - 9.81 ~ 0` (specific-force assumption behind the fall trigger). | Within tolerance; no false fall trigger over >= 60 s standing. | G3 |

## Phase G3: real robot (M2 winner only; gantry/harness)

Session protocol for every gate: SAFETY_FILTER=1 build verified, flight recorder
auto-on, roles assigned (operator with remote, spotter, harness minder), E-stop
chain rehearsed (remote damping / `p` Passive / hardware kill). Abort rule: any
safety-filter engagement or unexpected behavior ends the session; CSV analysis
(safety_analyzer + W3 metrics) before the next attempt.

| Gate | What is tested | Pass criterion | Unblocks |
|---|---|---|---|
| **G3.0** | Dry run without policy: FixStand under harness, E-stop chain exercised, E2 re-checked on the day. | All steps rehearsed; recorder produces a CSV. | G3.1 |
| **G3.1** | Harnessed stand: FixStand -> `h` at cmd 0, >= 30 s, 3 repetitions. Split mode, pinned period. | Calm stand, no filter engagement; twitch level comparable to the G2.5 bridge-with-noise run (not the clean run). | G3.2 |
| **G3.2** | Harnessed first steps: held vx 0.2-0.3 bouts with slack harness. | 3 consecutive clean bouts, no filter engagement, no harness catches. | G3.3 |
| **G3.3** | Free walk: room walking >= 2 min continuous, repeated across >= 2 sessions; gentle turns and stops; arms held. | No falls, no filter engagements, arms calm, from-stand entries clean. **STAGE D REQUIRED TIER.** | Stage D close; G3.4/G3.5 |
| **G3.4** | Stretch A, live arms: repeat G3.1 -> G3.3 with `hold_joint_ids` reduced/removed (the M2-calmed policy driving arms). | Same bars as G3.1-G3.3. | reporting |
| **G3.5** | Stretch B, cadence authority: un-pin the period (HL-owned), walk at 2-3 speeds. | Stable walking; measured period varies with speed consistent with the candidate's sim map. **STRETCH TIER.** | reporting |

## Bridge post-mortem (2026-07-15): plan, issues, solutions

**The plan was:** make the bridge plant faithful to the training nominal
(h1_2_constants.py values), so that any bridge failure attributes to plumbing rather
than plant, with the old harsh values kept as a stress scene (decision 1, 2026-07-14).

**Issue 1: the training-nominal plant destabilized the whole bridge** (live session:
FixStand leaned until the heels unloaded, A0 twitchy and falling). Factor breakdown,
measured on the FixStand+band loop (steady-state joint-velocity rms; training-nominal
plant 0.125 vs old plant 0.004):

| factor removed by the faithful edit | restoring it alone | share |
|---|---|---|
| frictionloss 0.2 -> 0 | 0.023 (5.5x recovery) | largest single factor |
| armature 0.1 -> per-motor 0.002-0.04 | 0.028 (4.5x) | rotor inertia damped the discrete PD, worst at light distal joints |
| joint damping 1.0 -> 0.001 | 0.033 (3.8x) | acted as ~+1 kd on every joint |
| base free-joint damping 1 -> 0 | 0.104 (minor) | was unphysical anyway |

No single factor suffices; only the combination restores calm. The amplifier that
turns "less damped" into "unstable": the bridge computes EXPLICIT torque PD at 500 Hz
with ~2-4 ms of real feedback latency (DDS + threads). On a dissipation-free plant,
2 ms of delay makes that loop energy-injecting (rms x30, falls); the old plant absorbs
6 ms without a trace. mjlab has neither the latency nor the explicit PD (implicit
position servos), so a training-nominal plant is NOT faithful to training dynamics in
this bridge. Side finding: pure-PD FixStand is not self-stable on any plant; the
elastic band always did the stabilizing until policy takeover.

**Solution 1: vendor reference plant** (armature 0.01, frictionloss 0.1, damping
0.001, clean free base). Latency-robust across the whole 0-6 ms range with graceful
degradation, and the honest hardware proxy: real joints have friction, and the real
robot's firmware PD runs at multi-kHz so it tolerates its own latency. Verified in the
replica (full chain incl. the historical 0->1.0-from-stand killer at 0/2/4 ms) and
live (A0: 87 s walking, zero safety triggers).

**Issue 2: the A1 keeper tripped the safety filter instantly.** Not the plant. The
goal-space state s took its velocity from the sportmode `frame_vel` sensor, which sat
on the torso-mounted IMU SITE; training extracts the PELVIS root velocity. Torso sway
adds an w x r component that the goal delta V*-s feeds straight back into the LL:
phantom corrections escalate (the #8b estimator-noise mechanism, but structural).
Flight recorder: the LL commanded ankle targets 3x past the mechanical stops within
0.2 s; the filter fired correctly on real violence. Replica discriminator: imu-site
velocity = falls at stand; pelvis velocity = stands and walks at 0.5 AND 1.0.

**Solution 2:** `frame_vel` retargeted to the pelvis body in both bridge scenes (the
bridge binds sensors by name; no C++ change). Real-robot consequence, folded into gate
E1: the onboard estimator must supply a BASE (pelvis) velocity estimate; an IMU/torso
velocity does not qualify. Residual: the keeper is still ~10x twitchier than A0 in the
bridge (its known aggressive character, ub_arm_vel 2.7-3.5x A0); that is M2's job, not
a bridge bug.

**Walk aesthetics note:** the "nicer" early walks were Model-v1 policies with soft
arms (kp 7.9); optB deliberately holds arms at deploy gains (ADR-0005 trade: deploy
fidelity over looks). M2 arm calming is the active fix for the look.

**Issue 3 (2026-07-16 session): the safety filter's joint trigger CAUSED the A1 fall.**
After the pelvis-velocity fix the keeper stood at cmd 0 (unsettled stepping = the known
TD3-HL velocity-hold degeneracy, a training issue tracked in the A1a thread, faithfully
reproduced by the bridge). On a vx=1.0 press it fell; flight recorder: filter at
alpha=1.0 (whole-body position hold) at t=17.5 with tilt only 0.196, fall AFTER. The
A1a gait rides its ankle/hip/knee stops by 0.01-0.2 rad every stride (trig_joint active
40% of the session, from t=0.8 standing), so the any-joint-out -> freeze-all-27 response
turns a normal stride into a fall. Also found: `h1_2_limits.h` disagrees with the sim
model (knee [-0.26, 2.05] vs scene XML [-0.12, 2.19]); audit against the official URDF
before hardware. **Fix (State_RLHRL only, A0 untouched): joint violations now clamp the
offending joint's COMMAND to its limit (firmware-like) instead of feeding the hold ramp;
tilt/fall keep the ramped whole-body hold; trig_joint in the flight log now means
"clamp active".** Replica cross-check: the keeper passes the full chain incl. the
1.0-step at 2 ms delay with no filter, so the freeze (not the gait) was the fall cause.
Review note: this changes safety behavior; revisit the response design before hardware
(G3.0) — a per-joint clamp is what the H1-2 firmware does anyway.

## Rollback rules

Any hardware anomaly: flight-recorder CSV first, then reproduce in the bridge in
this order: faithful scene, stress scene, faithful+state_noise. Only then propose a
code or training change (spec-first per CLAUDE.md). No same-day retry after a
safety-filter engagement without a written cause hypothesis.

## Doc corrections to sync on approval

- The "4 YAMLs + constants uncommitted in the working tree" note is stale
  (committed in `6decd8b`).
- `a1_estimator_noise_8b` memory: DR machinery validated, but the current keeper
  did NOT train with it (`goal_state_noise.enable: false`); requirement moved to M2
  (decision 2 above).
- `deploy_real.yaml` staleness (fixed by W5).
