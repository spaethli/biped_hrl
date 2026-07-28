# A1a stage D: sim-to-real deployment validation plan (v1, APPROVED 2026-07-14)

> **Exploratory history lives outside this repo (2026-07-28).** The chronological
> record behind this document — every run tried, what failed, root causes, dated
> session logs and result tables — was moved to the author's research knowledge
> base to keep this repo publishable. What remains here is the current design and
> status: the options in use and how the working solution is implemented.

## Current status dashboard (updated 2026-07-28 — read this first)

**A0 track (WL-B0):** first real-hardware session ran 2026-07-23 on the locked candidate
`a0_v2_optB_rs20_baseline`, split arms. Full bridge G2 battery (G2.1-G2.4, G2.7) passed
clean 2026-07-22 (parity within 0.01 m/s of the 0.05 m/s bar). On the real robot: **no
falls in the walking data**, but the user described it as "drunk stumbling" — this has **two
separate, now-disentangled causes**:
1. **ROOT CAUSE of the stumbling, FOUND 2026-07-23**: the whole-body joint-limit hold
   **does arm on real hardware** (unlike the clean 2026-07-22 bridge captures) — it fires
   on ankle/hip overshoots as small as **0.002 rad (0.1°)**, mostly at stand, because
   MuJoCo silently clamps at the limit while real hardware trips the hold. This is the
   **same defect fixed for A1 on 2026-07-16 and never ported to A0**; the earlier
   "empirically closed" verdict is withdrawn. **PORTED AND VALIDATED ON HARDWARE
   2026-07-23** (WL-B0a): 225.8 s of real free walking with the filter never engaging
   once (`alpha` max 0.000 over 26544 clamp rows), against 12463 full-hold rows in the
   pre-fix run. → the deploy journal in the research KB (`wiki/architectures/a1a-deploy-journal.md`).
2. **A persistent ~3-5° backward pitch lean, present even at stand (sim ~0), is a
   SEPARATE finding** (independent of the filter — one run shows the full lean with zero
   filter engagement). **FULLY DIAGNOSED 2026-07-23/28 (WL-B0b); no cause remains
   unidentified.** It is a sum of two plant-model errors, each now measured:
   - **(A) encoder→attitude map error, ~2.3° constant + ~13% scale.** The
     **inclinometer session (2026-07-28) vindicated the IMU by direct measurement**:
     pelvis read **−6.80°** against the IMU's **−6.30°** (agreeing to 0.50°) while the
     encoder feet-flat FK said only −3.16°. Torso and pelvis read identically, killing
     the pelvis-vs-torso frame question physically. Assumption-free core result: with
     the true attitude and the measured encoders the model puts the foot **2.74° toe-up
     while it is provably flat on a flat floor**. This splits again into **~0.9° of
     foot-sole wedge** (both soles measured thicker at the front, so both feet sit
     **toe-up** by 0.74°/1.11°, tilting the body backward; the flat capsule sole in
     `h1_2.xml` cannot represent it) and **~1.3° of leg-encoder zero error** (a hardware
     re-zero fixes that part, no retrain, no XML change). **Do NOT "recalibrate" the
     IMU** — it is the honest sensor.
   - **(B) excess knee droop from unmodelled leg mass — RESOLVED 2026-07-24**: the real
     robot weighs 73.70 kg with an 18.60 kg leg vs the model's 66.98 kg / 15.40 kg leg
     (the earlier "~22% load excess, origin uncertain" framing is withdrawn — it was a
     stacked measurement error, corrected once the real mass and the robot's own
     `tau_est` were used).

   Fix staged for the next model version (leg mass + foot-sole geometry,
   `docs/adr/0006-leg-mass-foot-geometry-nominal-correction.md`), held so it does not
   rebase the training plant mid-WL-D-batch. **Next action is free and needs no retrain:
   run Unitree's zero-point calibration on the 6 leg joints, repeat the 60 s policy
   stand, and re-measure — expect the residual to fall from −0.048 rad to about −0.016
   rad (the sole-wedge floor), not to zero.** FALSIFIED along the way: IMU mounting
   offset, attitude-estimator convention, lab floor slope, pelvis-vs-torso frame,
   symmetric CoM shift, harness down-force, and head mass. **Still open**: the L/R leg
   asymmetry (the sole wedge does not explain it — differential only 0.375°, wrong
   sign). → the deploy journal in the research KB for the full method (three independent
   pitch estimates), the hypothesis table, and the 2026-07-24/28 corrections.

E-stop chain was also corrected this session: `p`→Passive is the verified primary stop;
Ctrl+C is **not** a verified E-stop (no signal handler exists) — see "E-STOP chain"
below. → the deploy journal in the research KB (`wiki/architectures/a1a-deploy-journal.md`) for the dated session-by-session detail; hand-off prompts WL-B0a/WL-B0b in
`worklines.md`.

**A1 track (WL-B1):** deploy candidate LOCKED = `arm4d` (`ll_energy_coef=0.05`) —
the only A1 candidate that passes the live bridge (2026-07-22, all directions incl.
0→w step-from-stand). D2/old keeper/arm3/arm4c all fail live on **twitch, not
clearance** (action_rate splits cleanly: A0 0.57-0.75 vs A1 0.92-1.28, no overlap).
`arm5` (ankle-roll) fails catastrophically with a now-understood mechanism (bilateral
ankle-roll+hip-roll saturation decaying a lateral command to zero). Headless replica
sweep confirms arm4d 6/6 clean, matching live. **Open**: full live G2 battery on arm4d
has not been run yet (only the smoothness spot-check above). → the deploy journal in the research KB (`wiki/architectures/a1a-deploy-journal.md`).

**Both tracks share:** the `gait_phase_cmd` fix (2026-07-21, "Defect 0") — the
prior finding that the deploy gait clock was permanently dead under keyboard control,
which voids every pre-2026-07-21 live bridge result for both A0 and A1. Confirmed
working on real hardware (joystick path) in the 2026-07-23 A0 session too.

**Not yet done / still open, either track:** WL-B1's full live gate battery on arm4d;
the **leg-joint zero-point re-calibration** and its re-measured stand (free, no retrain,
the next action on the lean — expect residual −0.048 → ~−0.016 rad); the Model v3
leg-mass + foot-sole-geometry fix (staged, `docs/adr/0006`, held so it does not rebase
the plant mid-WL-D-batch); the **L/R leg asymmetry** (right leg carries more load; the
sole wedge was checked and does NOT explain it); the repeat/second session for G3.3
"repeatable" plus the user's qualitative sign-off that the stumbling is gone; the
keyboard-latch fix (Defect 1, downgraded, characterized exactly by `bridge_session.py`
but not fixed). Resolved since the last pass: **Component A of the backward lean is no
longer open — the 2026-07-28 inclinometer session vindicated the IMU and split the error
into ~0.9° foot-sole wedge + ~1.3° encoder zeros** (see item 2 above); the out-of-process
E-stop question (LAN-cable pull verified, see "E-STOP: verified behaviour" below) and
Defect 2 (the ONNX re-export — confirmed correct as of the 2026-07-22 G2 battery).

The evidence, numbers and mechanism detail behind this dashboard (dated sessions, bridge
post-mortems, defect-by-defect diagnosis, the backward-lean hypothesis table) live in
the deploy journal in the research KB (`wiki/architectures/a1a-deploy-journal.md`), per the migration note at the top. What follows here is the gate ladder and the
current design.

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
chain rehearsed (see the E-STOP section immediately below). Abort rule: any
safety-filter engagement or unexpected behavior ends the session; CSV analysis
(safety_analyzer + W3 metrics) before the next attempt.

### E-STOP chain — CORRECTED 2026-07-22 (read before any hardware session)

The original one-line chain above ("remote damping / `p` Passive / hardware kill") was
written at the 2026-07-14 grill and **the "hardware kill" leg was never verified — it was an
assumption, and per the user (2026-07-22) it does not exist; the team expected `Ctrl+C` to be it.**
Corrected, with code evidence:

**1. `p` -> Passive is the primary software E-stop. USE THIS.** Registered as a keyboard
transition for the Velocity state (`config/config.yaml`, `Velocity.keyboard_transitions`) and
live *simultaneously* with remote control, because joystick and keyboard transitions are
appended to the same `registered_checks` list (`deploy/include/FSM/FSMState.h:55-78`). Passive
commands motor mode 1 with kd 5/5/5/5/2/2 legs, 6 torso, 2 arms — an **actively commanded
damped collapse**. The remote's own `LT + B` maps to the same Passive state and is the
operator's equivalent.

> **VERIFIED ON REAL HARDWARE 2026-07-23 (the user) — this section's pessimism is CORRECTED.**
> All three paths were tested from Velocity mode (`o`) on the robot: **`p` -> Passive,
> `Ctrl+C` -> same damped result, and PULLING THE LAN CABLE -> same damped result.** The
> "stiff statue" risk described below did NOT materialise on this hardware. See the
> "E-STOP: verified behaviour" box after point 3 for the corrected judgement, which
> supersedes points 2 and 3 where they conflict.

**2. `Ctrl+C` IS NOT AN E-STOP. Do not plan around it.** There is **no `SIGINT`/`sigaction`/
`atexit` handler anywhere in `deploy/robots/h1_2/` or `deploy/include/`** (verified by grep,
2026-07-22). `Ctrl+C` therefore terminates the process with **zero cleanup**: no damping
command, no zeroed gains, nothing. Publishing simply stops and the robot is left holding the
last `LowCmd` — full `kp` on mid-stride position targets — until its own firmware watchdog
fires. That is the stiff-statue failure mode, not a gentle collapse. Contrast the neighbouring
lab stack, which does this correctly: `corelllab/h12_ros2_controller/.../channel_interface.py`
`shutdown()` calls `estop()` (zero mode/tau/kp/kd) *before* closing the publisher. **Ctrl+C is
a last resort after `p`, never the primary.**

**3. There is currently NO out-of-process kill, and that is a real gap.** Both `p` and the
remote's `LT + B` are handled by *our* FSM, so **if `h1_2_ctrl` hangs or deadlocks, neither
works**. Note also that the remote cannot fall back to Unitree's built-in damping while we are
in low-level control: per the E1 finding, the vendor motion service must be off for our
controller to command the robot at all, so with that service off the remote is only an input
device forwarded through `lowstate->joystick` — its vendor-side functions are not available.
**Before G3.0, establish and rehearse a true out-of-process stop.** Candidates, in order of
preference, to be verified against the physical robot and Unitree's manual (NOT assumed):
  (a) a physical E-stop / power cut on the robot itself, if this unit has one;
  (b) battery disconnect;
  (c) the gantry/harness taking the full load, with the harness minder briefed to take weight
      on any anomaly — this is the only guaranteed-available option today;
  (d) as a software improvement (spec-first, WL-B's file): add a `SIGINT` handler that sends a
      Passive/zero-gain command before exit, which would at least make `Ctrl+C` fail safe.
      **Concrete design (the user's reference, 2026-07-22: Tom Howard's ROS 2 course, Part 2
      Exercise 5 "Implementing a Shutdown Procedure",
      `https://tom-howard.github.io/ros2/course/part2/`).** Their ROS2/Python pattern is:
      disable the framework's own signal handler (`SignalHandlerOptions.NO`), catch
      `KeyboardInterrupt` around `spin()`, and in a `finally:` block call `on_shutdown()`
      which publishes a zero-velocity `TwistStamped`, then **busy-wait on a `self.shutdown`
      flag until the stop message has actually gone out** before destroying the node. The
      load-bearing idea is that last step: publishing is asynchronous, so exiting immediately
      after the publish call can drop the stop command.
      Mapped onto our C++/Unitree-DDS controller (`deploy/robots/h1_2/main.cpp`):
        - `static volatile sig_atomic_t g_shutdown = 0;` + a `sigaction` handler that ONLY
          sets the flag (a real signal handler must stay async-signal-safe — no DDS publish,
          no logging, inside the handler itself);
        - `main`'s current `while (true) { sleep(1); }` becomes `while (!g_shutdown)` with a
          much shorter sleep (~10 ms), so response is prompt rather than up to a second late;
        - on exit, **transition the FSM to Passive (id 1) rather than hand-rolling a zero
          command** — that reuses the existing, already-tuned damped-stop path (mode 1,
          kd 5/5/5/5/2/2 legs, 6 torso, 2 arms) instead of adding a second stop semantics,
          per the reuse-over-new-code convention;
        - then hold for several control cycles (the analogue of their busy-wait) so the
          Passive `LowCmd` is actually published before `exit`, and only then tear down.
      **Scope honesty: this makes `Ctrl+C` fail SAFE, it does not make it an E-stop.** It
      cannot help if the process hangs (the handler never runs), so it does NOT close gap 3 —
      `p` stays primary and the out-of-process kill (a)/(b) is still required.
**Do not run G3.0 until (a)/(b) is answered and whichever applies is rehearsed.** Option (c) is
not sufficient on its own for anything beyond a harnessed stand.

### E-STOP: verified behaviour (2026-07-23) — the gap is CLOSED for harnessed sessions

The user tested all three from Velocity mode on the real robot; all three ended in the damped
Passive-like state. What each one proves:

| trigger | result | what it demonstrates |
|---|---|---|
| `p` | Passive | the intended in-process software stop works on hardware |
| `Ctrl+C` | same | process death does NOT leave the robot stiff — my earlier warning was wrong here |
| **LAN cable pulled** | same | **an OUT-OF-PROCESS stop exists and works** |

**The LAN test is the important one.** `h1_2_ctrl` runs on the workstation and reaches the
robot over ethernet (`setup_all_robot.sh`: `H1_2_NETWORK=enp11s0`), so pulling the cable
severs controller->robot commands while our process is still alive. The robot damped anyway,
which means the safe state does **not** depend on our process behaving. Two mechanisms can
produce it and both are benign: a robot-side LowCmd-timeout watchdog, and/or our own
all-states check `lowstate->isTimeout() -> Passive` (`FSMState.h:81-86`). Either way the
observed outcome is damping.

**Judgement change: gap 3 ("no out-of-process kill") is CLOSED for harnessed work.** Pulling
the LAN cable is a physical, human-executable stop that needs no software cooperation — it is
the hardware kill the plan assumed and could not find. **G3.0's blocking requirement is
therefore SATISFIED by demonstration**, and the earlier instruction not to run G3.0 until a
power cut was established is withdrawn.

**Three caveats that remain true and belong in the session brief:**
1. **Damping is not catching.** All three paths leave the robot limp; under gravity it
   collapses. That is correct behaviour for an E-stop and is exactly why the harness/gantry
   stays mandatory. It is a safe stop, not a save.
2. **The timeout latency is unmeasured.** Between losing commands and damping, the robot runs
   on its last command for some interval. Worth timing once (flight-recorder timestamps or
   video) so the operator knows whether it is ~50 ms or ~500 ms.
3. **A HUNG process is not the same as a killed one.** `Ctrl+C` and cable-pull both stop
   commands reaching the robot. A wedged `h1_2_ctrl` whose DDS publisher thread keeps
   re-sending the last command would NOT trip a command-timeout watchdog. Untested; the cable
   pull covers it operationally (it cuts the link regardless of what the process is doing), so
   this is a known-unknown, not a blocker.

**Net: keep `p` as primary (it is the cleanest, commanded stop), the remote's `LT+B` as the
operator's equivalent, and the LAN cable as the true out-of-process backstop.** Ctrl+C is a
verified-safe fallback rather than the hazard this section originally described.

**Practical: the keyboard E-stop requires terminal focus.** `Keyboard` polls `fileno(stdin)`
via `select()` (`deploy/include/isaaclab/devices/keyboard/keyboard.h`), so if the keyboard
operator's window loses focus their `p` is silently dead. Focus check belongs on the pre-run
checklist, and both stops must be fired for real during G3.0 while the policy is not in control.

| Gate | What is tested | Pass criterion | Unblocks |
|---|---|---|---|
| **G3.0** | Dry run without policy: FixStand under harness, **E-stop chain exercised per the E-STOP section above — fire `p`->Passive AND the remote's `LT+B` for real, verify keyboard terminal focus**, E2 re-checked on the day. | All steps rehearsed; recorder produces a CSV. **Out-of-process stop requirement: SATISFIED 2026-07-23** — see "E-STOP: verified behaviour" (LAN-cable pull demonstrated on hardware; Ctrl+C also verified safe, contrary to the original warning below). | G3.1 |
| **G3.1** | Harnessed stand: FixStand -> `h` at cmd 0, >= 30 s, 3 repetitions. Split mode, pinned period. | Calm stand, no filter engagement; twitch level comparable to the G2.5 bridge-with-noise run (not the clean run). | G3.2 |
| **G3.2** | Harnessed first steps: held vx 0.2-0.3 bouts with slack harness. | 3 consecutive clean bouts, no filter engagement, no harness catches. | G3.3 |
| **G3.3** | Free walk: room walking >= 2 min continuous, repeated across >= 2 sessions; gentle turns and stops; arms held. | No falls, no filter engagements, arms calm, from-stand entries clean. **STAGE D REQUIRED TIER.** | Stage D close; G3.4/G3.5 |
| **G3.4** | Stretch A, live arms: repeat G3.1 -> G3.3 with `hold_joint_ids` reduced/removed (the M2-calmed policy driving arms). | Same bars as G3.1-G3.3. | reporting |
| **G3.5** | Stretch B, cadence authority: un-pin the period (HL-owned), walk at 2-3 speeds. | Stable walking; measured period varies with speed consistent with the candidate's sim map. **STRETCH TIER.** | reporting |

#### New tool: `scripts/bridge_session.py`

Scripted LIVE-bridge sessions — launches `/opt/unitree_mujoco` + the real `h1_2_ctrl`,
drives the FSM and velocity keys over a FIFO, releases the elastic band via XTEST (a GLFW
keypress on the mujoco window, nothing else can toggle it), then runs
`deploy_gate_analyzer.py`. The counterpart to `bridge_replica.py`, not a replacement:
the replica re-implements the controller in python and is structurally blind to C++ bugs,
which is precisely what this task needed to validate. Needs `python-xlib` (installed in
`unitree_mjlab_h1_2_rl`). Removes the need to hand-drive sim-to-sim checks.

**It characterized Defect 1 (keyboard latch) exactly, on its first run.**
`Keyboard::_read()` clears `_key` to `""` after an 80 ms `select()` timeout, so a velocity
key must be **re-sent continuously** or `keyboard_velocity_commands` reads `""` and returns
`[0,0,0]`. A real keyboard's X key-repeat does this; a one-shot piped char does not. FSM
keys (`i`/`o`/`p`) are exempt because the transition fires on the transient — which is why
the first automated run changed states correctly and logged `cmd [0,0,0]` for the entire
session. The script re-sends at 50 Hz. Also found: a stale `h1_2_ctrl` holds the DDS lowcmd
channel and the next one silently sits in Passive writing no CSV, so the script pkills both
binaries before every run.

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
   the hierarchy-benefit roadmap (research KB) `#6`. Not re-described here.

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
