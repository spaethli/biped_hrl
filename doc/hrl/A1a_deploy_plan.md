# A1a stage D: sim-to-real deployment validation plan (v1, APPROVED 2026-07-14)

**Execution status (last updated 2026-07-20):**
- **W1-W5, G1.2: ✅ done.** `State_RLHRL` keeper support built (hl_vel input, velgoal
  mapping, HL-owned phase clock, `pin_period`, dim guards); `bridge_replica.py` (headless
  A0+HRL replica, both scenes, delay knob) and `deploy_gate_analyzer.py`
  (`[DEPLOY-GATE]` json, segments by command AND by FSM-entry `id`) are the standing
  pre-check/analysis tools; `onnx_parity.py` passes for both A0 and the A1 candidate.
  **W1 back-compat regression PASSED live 2026-07-17**: a pre-velgoal 7-dim absolute-mode
  checkpoint (`2026-06-18_10-10-27_..._7k/model_7000.pt`) loads and steps with no dim
  errors via a `deploy.yaml.w1_legacy_test` fixture (loaded through `H1_2_DEPLOY_CFG`,
  live config never touched) — absent-key back-compat confirmed still working.
- **G2.0/G2.1: ✅ live 2026-07-16/17.** The A1 candidate loads, stands, and holds cmd 0
  cleanly in the real bridge. **Correction of stale docs**: the checkpoint actually staged
  in `exported/` is the arm-calm/rs20 run
  (`2026-07-14_16-03-57_a1_td3_pose0p5shw16-4_ar0p02_cadhl_cot0p2_kl0p01_rs20_s42`), not
  the older 2026-07-09 run some earlier notes call "the keeper" — that name is stale.
- **Goal-scale ONNX-metadata pin: shipped + validated live.** The g->V* decode is pinned
  from the HL ONNX's `goal_scale`/`goal_center` metadata (`hrl::GoalSpace::freeze_scale`)
  instead of `deploy.yaml`'s command ranges, so those ranges are now a pure operator
  safety clamp. Confirmed live via the `[HRL] goal scale pinned from ONNX metadata ...]`
  log line at FSM entry.
- **Open risk, unresolved: 4 ms-delay latency x goal-scale takeover fall.** Under the
  TRUE trained yaw scale (1.0, vs. a previously-used incorrect 0.5), the A1 candidate
  passes 0/2 ms cleanly but falls during takeover at the 4 ms delay bucket (bridge delay
  is quantized to the 500 Hz tick, so 0/2/4 ms are the only 3 distinct buckets in this
  range) — deterministic, reproduced. A yaw-scale isolation sweep is **non-monotonic**
  (0.5/0.6/0.9 pass, 0.7/0.8/1.0 fail at different phases), matching the W2
  latency/PD-energy-injection chaos, not a clean trend. A startup goal-authority ramp was
  tested as a mitigation and **rejected as unreliable** (2.0 s passed the full chain but
  2.5 s — more conservative — failed elsewhere; a hand-tuned ramp duration can't be
  trusted off one passing point). Untried options: (a) a slew-rate limit on |ΔV*| per HL
  tick instead of a time-based ramp; (b) characterize actual live DDS latency against the
  risky 3-4 ms band rather than patch blind; (c) training-side quiescent-|g| regularizer
  or an actual latency reduction as root fixes. Real bridge latency is documented as
  ~2-4 ms, i.e. this sits right at the edge of what the live bridge may exhibit — treat
  takeover (`h`, first ~3 s) as the highest-risk moment until resolved.
- **G2.6 cadence modes: BOTH fail at held 0.5 m/s (2026-07-17, live).** HL-owned
  (`2026-07-17_16-09-52`) fell repeatedly and clearly. Pinned 0.625s
  (`2026-07-17_16-12-02`) looked clean in `_hrl.csv` alone but the safety CSV's raw
  `trig_fall` column shows two real filter engagements including a likely collapse right
  at the session's end — caught only because the base safety CSV has a tighter flush
  cadence than the HRL telemetry did (see flight-recorder fix below). **Cadence source
  does not rescue held-0.5 stability** — reinforces the WL-D routing below, not a
  cadence-mechanism question.
- **Held-command instability, 0.5+ m/s: routed to WL-D (gait/clearance, not a WL-B fix).**
  Across two live sessions, 0.5 m/s failed the large majority of attempts (~1 clean pass
  in ~10+ tries total) and 1.0 m/s failed every attempt, **including the final step of a
  deliberate gradual ramp** (0.2->0.3->0.4 clean, then fell exactly at 0.5) — so this is
  not a step-transient/foot-clearance-at-demand story specifically, it looks like a
  chronic shortfall at that speed regardless of how it's reached. Matches the user's
  swing-lift hypothesis: the A1 candidate's step-from-stand swing clearance
  (0.050-0.057 m mean, headless replica) is roughly half of A0's (0.065-0.098 m) in the
  identical test. Flagged for WL-D (the swing-clearance metric + "raise clearance" lever
  already exist from WL-E, `docs/adr/0005` Amendment 2 item 1); not implemented here.
- **Flight-recorder bugs found + fixed (2026-07-17, two rounds).** Both `SafetyLogger`
  (`safety_logger.h`) and `hrl::Telemetry` (`hrl_telemetry.h`) used to truncate their CSV
  and reset their tick/entry counters on EVERY FSM entry, silently destroying earlier
  attempts' telemetry the moment a later attempt in the same session was logged (this is
  why an early live re-run showed a spuriously clean file despite an observed fall
  earlier that session). Round 2: A0 and A1 hold SEPARATE `SafetyLogger` instances, so
  the round-1 per-instance fix still let the FIRST A0->A1 switch in a session truncate
  A0's data. **Fixed properly**: truncate-vs-append is now decided by whether the CSV
  already exists on disk (robust across different C++ objects), `entry` is a
  process-wide counter, and `hrl::Telemetry`'s flush cadence dropped from ~10s to ~2.5s
  (matching the safety logger, so an abrupt session end no longer loses the tail).
  `deploy_gate_analyzer.py` segments on `entry` changes too and never drops a fallen
  segment under its 3s minimum-duration floor. All rebuilt, confirmed clean.
- **E1/E2 hardware gate: E2 ✅ PASS, E1 FAIL (structural, real robot, 2026-07-20).** See
  "A0-first hardware track" below for the full result and the 3 documented fallback
  options for A1 (E1 does not block A0 — A0 has no velocity dependency).

Operational gated checklist for `A1a_plan.md` Plan v2 stage D. Grilled 2026-07-14.
Execution is gate-by-gate; no gate starts before its blockers pass. The battery
(phases G1+G2) is checkpoint-agnostic: it runs once now on the current keeper
(pipeline shakedown) and re-runs cheaply on every M2 candidate. Hardware (G3) only
on the M2 winner.

**Scope note (2026-07-20): this doc now covers TWO deploy tracks.** The A1 keeper track
(everything below, as written) and the **A0-first hardware track** (the user's call: the
first real-robot deployment is the flat A0 policy, not the hierarchy). Both are owned by
WL-B and share one bridge, one replica, one flight recorder and one C++ tree, so they are
not separate worklines. The A0 track's ladder, its carry-overs and the four A0-specific
items are in "A0-first hardware track" near the end of this doc; read that section FIRST
if you are working the A0 path, since it drops roughly half the gates below as
hierarchy-only.

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

## Phase G3: real robot (A1 track: M2 winner only; gantry/harness)

**Scoping correction (2026-07-20):** "M2 winner only" is the **A1 track's** entry rule, not
a global one. It gated hardware on the arm-calmed HRL candidate back when A1 was the only
deploy target. The A0-first track has its own entry rule (see "A0-first hardware track");
A0 does not wait on any M2 candidate. The session protocol, abort rule and per-gate bars
below apply unchanged to both tracks.

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
After the pelvis-velocity fix the keeper stood at cmd 0 (unsettled stepping ~~= the known
TD3-HL velocity-hold degeneracy, a training issue tracked in the A1a thread, faithfully
reproduced by the bridge~~ — **corrected 2026-07-16 (WL-C): that cause is FALSIFIED. The
"velocity-hold degeneracy" was a sim-EVAL artifact (`--eval-cmd-vx` zeroed the goal scale),
so it never existed in training and cannot have reached the bridge, which does not use
play.py. The keeper tracks held commands ≥ A0 in sim (ss@0.5 0.043, ss@1.0 0.097). The
bridge's unsettled stepping is therefore UNEXPLAINED — treat as open.** One bridge-side
scale defect is real and could contribute: the C++ derives the goal scale from `deploy.yaml`
ranges, whose `ang_vel_z [-0.5,0.5]` gives **yaw scale 0.5 vs 1.0 trained** (vx/vy are
correct) — see `.claude/docs/deployment.md`). On a vx=1.0 press it fell; flight recorder: filter at
alpha=1.0 (whole-body position hold) at t=17.5 with tilt only 0.196, fall AFTER. The
A1a gait rides its ankle/hip/knee stops by 0.01-0.2 rad every stride (trig_joint active
40% of the session, from t=0.8 standing), so the any-joint-out -> hold-all-27 response
turns a normal stride into a fall. Mechanism correction (2026-07-16, user question):
the "hold" is NOT a rigid freeze. q_cmd = (1-a)*action + a*q_meas re-reads q_meas every
tick, so at a=1 the target chases the measured position and the PD degenerates to
damping-only torque = near-passive; a biped under gravity collapses. Plus a ratchet:
the collapse drives tilt past 0.44, the tilt trigger then sustains a=1, so the filter
cannot release mid-stumble. Honest causality: the stumble was the policy's (degenerate
HL + vx=1.0 edge command); the filter completed it at a recoverable tilt (0.196).
G3.0 design item: choose the terminal behavior deliberately - a TRUE hold needs the
target latched at trigger time (rigid statue, tips whole); the current chasing hold =
damping-only (Unitree damp-mode-like, collapses gently into the harness - arguably the
right terminal behavior on the gantry, wrong as a response to a recoverable graze). Also found: `h1_2_limits.h` disagrees with the sim
model (knee [-0.26, 2.05] vs scene XML [-0.12, 2.19]); audit against the official URDF
before hardware. **Fix (State_RLHRL only, A0 untouched): joint violations now clamp the
offending joint's COMMAND to its limit (firmware-like) instead of feeding the hold ramp;
tilt/fall keep the ramped whole-body hold; trig_joint in the flight log now means
"clamp active".** Replica cross-check: the keeper passes the full chain incl. the
1.0-step at 2 ms delay with no filter, so the freeze (not the gait) was the fall cause.
Review note: this changes safety behavior; revisit the response design before hardware
(G3.0) — a per-joint clamp is what the H1-2 firmware does anyway.

## A0-first hardware track (added 2026-07-20, the user's call; owned by WL-B)

**Decision: the first sim-to-real deployment is A0 (`State_RLBase`, key `o`), not the A1
keeper.** Kept inside WL-B rather than opened as a new workline: it reuses the same live
bridge, the same `bridge_replica.py`, the same flight recorder / `deploy_gate_analyzer`,
the same E gates and the same C++ tree. A separate workline would buy nothing and would
create a second editor of `deploy/`, which is exactly the collision WL-B's ownership rule
exists to prevent.

**Entry rule (replaces "M2 winner only" for this track):** A0 goes to hardware once its
own G1/G2 gates are recorded and the four A0-specific items below are closed. It does not
wait on WL-D, on M2, or on the A1 keeper.

### What A0 inherits unchanged (no rework)

- **W2 vendor plant.** Validated with A0 itself as the test policy (full chain at 0/2/4 ms
  in the replica; 87 s live walking, zero safety triggers, post-mortem "Solution 1").
- **W3 telemetry + `deploy_gate_analyzer.py`**, including the round-2 logger fix: A0 and A1
  hold SEPARATE `SafetyLogger` instances, and the truncate decision is now disk-based with
  a process-wide `entry` id, so an `o`-then-`h` session no longer destroys A0's rows.
- **W4 ONNX parity** (A0 optB passes), **E1/E2**, **G3.0 protocol**, **rollback rules**.

### What does NOT apply to A0 (drop from its ladder)

Roughly half the A1 ladder is hierarchy machinery with no A0 counterpart:

- All goal-scale work (F2 baking, ONNX metadata pinning, the C++ `freeze_scale` path).
  A0 has no goal space, so it cannot have a goal-scale bug.
- **The open 4 ms latency x goal-scale knife-edge finding** (G2.0 pre-check item 3) is
  goal-scale-specific and does NOT block A0: A0 passes 0/2/4 ms cleanly on the vendor plant.
- W1/W5 HRL structure keys, `hl_vel` input, velgoal mapping, `pin_period`.
- **G2.6 cadence modes.** A0's `gait_phase` is a fixed 0.6 s YAML clock that matches its
  training clock exactly; there is no HL period to pin or hand over.
- **Decision 2 (estimator-noise DR, M2 DR-on/DR-off variants).** A0's actor never consumes
  base velocity (`deployment.md:99`; the flat config also deletes `height_scan` from actor
  and critic at `config/h1_2/env_cfgs.py:184-185`, so the deploy.yaml 7-term obs list is
  complete and correct - verified 2026-07-20). The pelvis-vs-IMU failure that nearly killed
  the keeper (post-mortem issue 2) is structurally unreachable for A0: no goal channel, no
  V*-s feedback. E1 still runs, but a noisy real estimator degrades A0 far less.

### A0-specific items to close before hardware (the actual new work)

1. **Safety-filter terminal behavior is still the UN-fixed variant on A0 (highest-value
   item).** `State_RLBase.cpp:101-144`: any policy-controlled joint outside `h1_2_limits.h`
   sets `joint_hold`, which ramps `alpha` to 1 and drives
   `q_cmd = (1-alpha)*action + alpha*q_meas` on all 27 joints. That is exactly the chasing
   hold the 2026-07-16 post-mortem identified as degenerating to damping-only torque and
   *completing* a recoverable A1 stumble (tilt only 0.196) into a fall. The A1 fix
   (per-joint command clamp) was deliberately scoped "State_RLHRL only, A0 untouched", so
   A0 still carries the mechanism. **Action: measure before deciding.** Run a bridge walk
   with the flight recorder and check whether A0's gait actually grazes the limits the way
   A1a's did (`trig_joint` active 40% of session, stops grazed 0.01-0.2 rad every stride).
   If A0 does not graze, leaving the hold as-is is defensible and the G3.0 terminal-behavior
   choice stands as written. If it does graze, port the clamp to `State_RLBase` before
   hardware. This is a cheap measurement, not a speculative code change.
2. **`h1_2_limits.h` vs. the sim model disagreement is now BLOCKING for A0** (knee
   [-0.26, 2.05] vs. scene XML [-0.12, 2.19]). It was flagged as "audit against the
   official URDF before hardware" during the A1 post-mortem; on the A0-first path it gates
   the first hardware session, because A0's filter reads those same limits and item 1's
   whole-body hold is what fires on a false positive.
3. **Formal G2 recording pass for A0.** A0's current evidence is real but informal (the
   87 s live walk and the replica chain passes were produced while debugging W2's plant,
   not as recorded gates). Re-run G2.1/G2.2/G2.3/G2.4/G2.7 against A0's exact deploy build
   with the analyzer on, so the G2.2 achieved-vx and stride numbers exist as the parity
   reference G3.2/G3.3 score against.
4. **Arms: split-vs-live is an A0-specific call.** Decision 7 ("split first, live later")
   was written because the A1 keeper's arms were wild. A0-optB's arms are already calm in
   sim (bench `ub_arm_vel` 0.143 and `pose_dev` 0.0004, vs the A1a keeper's 0.50-0.64 and
   ~70x). Split (`hold_joint_ids` 12-26) is still the right first hardware session (fewer
   actuated joints, fewer failure modes, and it is the locked hardware config), but the
   live-arms step is materially less risky for A0 than decision 7 assumed and need not wait
   for a full stage-D close.

### A0 gate ladder

Bars are the A1 track's bars unless noted; parity references are A0's own mjlab numbers.

| Gate | A0 status / what to do | Reference numbers |
|---|---|---|
| **G1.1** | Bench + holds already exist for `a0_v2_optB_rs20_baseline`. Re-record only if the deployed ONNX is re-exported from a different run. | err_vx 0.085, CoT 0.537, 0 falls; ss@0.5 **0.055** (t90 0.6 s), ss@1.0 **0.077** (t90 0.9 s) |
| **G1.2** | ✅ A0 optB passes `onnx_parity.py` (W4). Single net, `velocity/v0/exported/policy.onnx`, no goal-scale metadata to check. | max abs diff < 1e-4 |
| **G2.0** | ✅ informally (loads and runs live). Fold into the item-3 recording pass. | no dim/FSM errors |
| **G2.1** | Record: stand at cmd 0, 60 s, analyzer on. | action-rate within ~1.5x A0's mjlab stand level |
| **G2.2** | Record: held vx 0.3/0.5/1.0 (>=30 s each) + backward -0.3, strafe +-0.3, yaw +-0.5. Deploy YAML clamps vx [-0.5, 1.0], vy/wz +-0.5 (pure operator clamp for A0: no goal scale derives from these). | achieved vx within +-0.05 m/s of G1.1; stride within +-0.05 s |
| **G2.3** | Record: 0 -> 0.5 -> 0, and 0 -> 1.0 from stand (the historical killer; A0 passes it in the replica at 0/2/4 ms). | 0 falls, clean return to stand |
| **G2.4** | Record in split mode (`hold_joint_ids` 12-26), per item 4. | same bars as G2.1-G2.3 |
| **G2.5** | **N/A for A0** (no estimator-fed goal channel). | - |
| **G2.6** | **N/A for A0** (fixed 0.6 s clock). | - |
| **G2.7** | Record: stress scene, held 0.5 + steps at 0.5. | 0 falls at 0.5 (BLOCKING); 1.0 advisory |
| **E1/E2** | **E2 ✅ PASS (2026-07-20, real robot).** **E1 FAIL — no odometry data at all** (see below). | per Phase E |
| **G3.0-G3.3** | As written, with item 1's terminal-behavior decision made first. | G3.3 = repeatable free walk |

### E1/E2 result (2026-07-20, real H1-2, offline session)

Tooling: `ros2 run h1_2_low_level_controller read_all_joints` (logs `lowstate` IMU/joints +
`sportmodestate` to one CSV at 50 Hz) + new `scripts/robot_estimator_check.py` (generic
offline analyzer for this log format — specific-force check, velocity bias/noise, a
rotation-vs-yaw-rate frame check, a walk-distance integration check; not gate-specific).
Session: `all_joints_2026-07-20_13-44-07.csv`, 622 s total, both a wireless/sport-mode
stand (~33-153 s) and a debug-mode stand with HL control deactivated (~473-553 s, the
condition the user judged the better match for our own low-level controller — stiffer, same
starting pose as mjlab's FixStand, would fall without a harness). Also two in-place yaw
rotations (~198-233 s left, ~233-250 s right) meant for E1's frame check. Free walking was
done but NOT tape-measured (deferred; the user's call that it's not necessary for A0 right now).

**E2: PASS.** `a_world_z = R*a_imu.z - 9.81` on the debug-mode stand: mean **0.084**,
std 0.048, min **-0.119** (nowhere near the -7 sustained fall-trigger threshold), over an
80 s window (comfortably above the 60 s bar). Sport-mode stand gives the same read
(mean 0.053). The real IMU follows the specific-force convention the sim-derived
fall-trigger assumed — no sign-convention fix needed, no false-trigger risk from statics.

**E1: FAIL — not a bias/noise problem, a complete absence of data.** `vel_x/y/z`,
`body_height`, `yaw_speed`, `pos_x/y/z`, `foot_raise_height` are **exactly 0.0 for all
31120 rows of the entire 622 s session** (every phase: sport-mode stand, debug stand, both
rotations). `read_all_joints.cpp` subscribes to a topic named `"sportmodestate"`
(`unitree_go::msg::SportModeState`), which is not in this robot's topic list at all (only
`/odommodestate` and `/lf/odommodestate` exist) — so `latest_sport_state_` never received
a single message and stayed at its zero-initialized default the whole time. The user
independently confirmed `/odommodestate` also reads all-zero directly.

**Resolved 2026-07-20 (structural, not a code bug):** `ros2 topic info /odommodestate -v`
shows a live publisher of the exact right type (`unitree_go/msg/SportModeState` — the same
type `read_all_joints.cpp` already expects), so it was never a topic-name or schema
mismatch. The publisher itself emits all-zero payloads. Per Unitree's own docs (see below),
`SportModeState` on real hardware is only populated while the vendor's BUILT-IN motion
control service owns the robot; it goes silent once that service is off, which is exactly
the state a custom low-level policy requires to command the robot at all. That is true
regardless of which stand condition was tested (sport-mode-looking stand vs. debug stand)
or which topic name is used — **this is a structural incompatibility between "Unitree's
onboard odometry" and "our own low-level controller in command," not a misconfiguration to
chase further.**

**Consequence: does NOT block A0.** A0's actor never consumes base velocity from any
source (no goal channel, no `V*-s` feedback) — this finding is moot for the A0-first
track. It only matters for the eventual A1 keeper hardware attempt, which is already
downstream of A0 per the existing sequencing. When that comes up, this is E1's documented
fallback for real, and this session's result is also the real-hardware verdict for the
thesis's own M5 "IMU-velocity feasibility probe" (`sec:a1a-deploy`, still marked
`\planned` there) — three options, undecided:

1. A custom leg-odometry estimator (kinematics from the joint encoders + IMU, which work
   fine under `lowstate` independent of the vendor's motion service).
2. The absolute-`V*` LL retrain (removes the runtime velocity dependency entirely).
3. **Swap velocity/height for acceleration in the goal space** — this is the existing
   roadmap idea `#6` / thesis M4 ("Richer, Real-Robot-Measurable Goal Space",
   `sec:a1a-goal`), not a new idea; today's E1 result is the concrete trigger that makes
   it a live decision instead of a deferred one. Canonical spec + open questions (the
   held-window target-semantics problem, the target-map rework) →
   `doc/hrl/hierarchy_benefit_roadmap.md` `#6`. Not re-described here.

Not resolved now; the user's call when A1 hardware is actually next.

IMU-side data (E2, gyro for the frame-check windows) came through the `lowstate`
subscription correctly throughout, so this was isolated to the sport/odom-state path, not
a general DDS/topic problem.

Sources: [Unitree G1 Humanoid - OpenMind](https://docs.openmind.org/robotics/unitree_g1_humanoid),
[Motion Switcher Service Interface](https://support.unitree.com/home/en/developer/Motion%20Switcher%20Service%20Interface)

Timestamp note: the session's own notes were wall-clock (epoch) times, not the CSV's
internal relative-time column; converted via the filename's timestamp (`13:44:07` local =
epoch 1784547847, cross-checked against the file's own mtime and independently verified
against the IMU gyro-z signal during the two rotation windows — clean baseline outside
them, clear elevated signal inside, confirming the conversion).

### DR ruling for the A0 deploy candidate (2026-07-20)

**Deploy the current A0 policy and test; do NOT pre-emptively widen DR.** A0 already
trains with push kicks (0.5 m/s linear, 0.52-0.78 rad/s angular, every 5-6 s), foot
friction (0.3, 1.6), base-CoM offset (+-5 cm/axis), encoder bias (+-0.015 rad) and obs
noise on ang_vel / gravity / joint_pos / joint_vel (`velocity_env_cfg.py:192-262`; the
`push_robot` pop at `config/h1_2/env_cfgs.py:148` is play-mode-only, so training keeps it).
What is deliberately absent (joint friction/damping/armature, actuation delay, motor
strength) is ADR-0005's explicit scoping to A2's Wide DR, not an oversight.

Rationale: (i) the validation ladder above is built to catch what extra DR would blindly
compensate for, and A0 has already cleared the one failure mode that actually bit this
project (latency instability, replica 0/2/4 ms); (ii) **WL-E is the precedent against
guessing** - it started from "A0 barely lifts its feet, it must need friction", measured
it, and falsified the plant attribution (the cause was training-side clearance shaping);
(iii) if hardware does fail, the flight recorder plus this ladder localizes the factor, and
WL-D arms 7/8/9 (explicit-PD actuation, the unitree_rl_gym perturbation package,
correlated obs-noise) are already specced as targeted reactive fixes. Margin for the first
attempt comes from the G3 protocol (harness, spotter, E-stop rehearsal, abort-on-filter-
engagement), not from pre-inflating training DR on a guess.

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
