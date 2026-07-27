# A1a stage D: sim-to-real deployment validation plan (v1, APPROVED 2026-07-14)

## Current status dashboard (updated 2026-07-23 — read this first)

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
   pre-fix run. → "A0 CLAMP PORT" section below.
2. **A persistent ~3-5° backward pitch lean, present even at stand (sim ~0), is a
   SEPARATE finding** (independent of the filter — one run shows the full lean with zero
   filter engagement). **DECOMPOSED 2026-07-23 (WL-B0b): it is a near-even sum of TWO
   causes, not one.** (A) a constant **−0.031 rad (1.8°) encoder→attitude map error**
   (the model's FK from encoders to body attitude, not the IMU: an independent statics
   estimate `p_tau` sides with the IMU to 0.008 rad and against the encoder kinematics by
   0.020-0.040) — **still open**, awaiting an inclinometer session (procedure + decision
   rules below); (B) **excess knee droop from unmodelled leg mass — RESOLVED 2026-07-24**:
   the real robot weighs 73.70 kg with an 18.60 kg leg vs the model's 66.98 kg / 15.40 kg
   leg (the earlier "~22% load excess, origin uncertain" framing is withdrawn — it was a
   stacked measurement error, corrected once the real mass and the robot's own `tau_est`
   were used). Fix staged for the next model version (leg mass + foot-sole geometry,
   `docs/adr/0006-leg-mass-foot-geometry-nominal-correction.md`), held until after the
   inclinometer session so it can bundle with the Component-A fix rather than rebasing
   twice. FALSIFIED along the way: IMU mounting offset, attitude-estimator convention, lab
   floor slope, pelvis-vs-torso frame, symmetric CoM shift, harness down-force, and head
   mass. → "THE BACKWARD LEAN, DECOMPOSED" section below for the full method (three
   independent pitch estimates) and the 2026-07-24 correction.

E-stop chain was also corrected this session: `p`→Passive is the verified primary stop;
Ctrl+C is **not** a verified E-stop (no signal handler exists) — see "E-STOP chain"
below. → "ROOT CAUSE OF THE STUMBLING", "FIRST REAL-HARDWARE RUN", and "Head-off run"
sections below for full detail; hand-off prompts WL-B0a/WL-B0b in `worklines.md`.

**A1 track (WL-B1):** deploy candidate LOCKED = `arm4d` (`ll_energy_coef=0.05`) —
the only A1 candidate that passes the live bridge (2026-07-22, all directions incl.
0→w step-from-stand). D2/old keeper/arm3/arm4c all fail live on **twitch, not
clearance** (action_rate splits cleanly: A0 0.57-0.75 vs A1 0.92-1.28, no overlap).
`arm5` (ankle-roll) fails catastrophically with a now-understood mechanism (bilateral
ankle-roll+hip-roll saturation decaying a lateral command to zero). Headless replica
sweep confirms arm4d 6/6 clean, matching live. **Open**: full live G2 battery on arm4d
has not been run yet (only the smoothness spot-check above). → "Post-fix bridge session
(2026-07-22)" and "Headless candidate sweep" sections below.

**Both tracks share:** the `gait_phase_cmd` fix (2026-07-21, "Defect 0" below) — the
prior finding that the deploy gait clock was permanently dead under keyboard control,
which voids every pre-2026-07-21 live bridge result for both A0 and A1. Confirmed
working on real hardware (joystick path) in the 2026-07-23 A0 session too.

**Not yet done / still open, either track:** WL-B1's full live gate battery on arm4d;
Component A of the backward lean (the 1.8° encoder→attitude map error — awaiting a
15-min inclinometer session, decision rules already written); the Model v3 leg-mass +
foot-geometry fix (staged, held until after that session, `docs/adr/0006`); the
repeat/second session for G3.3 "repeatable" plus the user's qualitative sign-off that the
stumbling is gone; the keyboard-latch fix (Defect 1, downgraded, characterized exactly
by `bridge_session.py` but not fixed). Resolved since the last pass: the out-of-process
E-stop question (LAN-cable pull verified, see "E-STOP: verified behaviour" below) and
Defect 2 (the ONNX re-export — confirmed correct as of the 2026-07-22 G2 battery).

The sections below are the full, dated, chronological record (decisions, phase
definitions, bridge post-mortems, defect-by-defect diagnosis, every hardware session)
that this dashboard summarizes — go there for evidence, numbers, and mechanism detail.

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

1. **Safety-filter terminal behavior — ⚠ SUPERSEDED. The "no port needed" verdict below
   was WRONG and the clamp was ported 2026-07-23** (see "A0 CLAMP PORT" section). It
   rested entirely on bridge/sim captures, and the pre-fix live-bridge A/B run of
   2026-07-23 proves *why* that was unsound: in sim, `trig_joint` is 0.0000 in every
   segment because MuJoCo silently enforces the joint range, so the bridge can never
   observe this defect no matter how long it runs. Kept below as the reasoning record.
   Original heading: **MEASURED 2026-07-20, no port recommended.**
   `State_RLBase.cpp:101-144`: any policy-controlled joint outside `h1_2_limits.h` sets
   `joint_hold`, which ramps `alpha` to 1 and drives `q_cmd = (1-alpha)*action +
   alpha*q_meas` on all 27 joints — the same chasing hold the 2026-07-16 post-mortem
   identified as degenerating to damping-only torque and *completing* a recoverable A1
   stumble. The A1 fix (per-joint clamp) was scoped "State_RLHRL only", so A0 still
   carries the mechanism as written.

   Analyzed the freshest A0 flight-recorder session on disk
   (`logs/deploy_safety/2026-07-20_14-41-43.csv`, 81.8 s, single continuous FSM entry, no
   `_hrl.csv` companion → A0-only) with `scripts/safety_analyzer.py`, then broke it out by
   phase (script not checked in — ad hoc numpy pass) because the whole-session number
   mixes a real fall in with normal walking:

   | phase | t range | trig_joint duty | worst margin |
   |---|---|---|---|
   | startup/takeover (band→free) | 0-3 s | 19.0% | 0.19 rad (ankle_pitch, transient) |
   | clean walking 1 | 3-51 s | **3.6%** | 0.02 rad (hip_roll/ankle_roll/ankle_pitch/hip_yaw) |
   | clean walking 2 | 53-62.7 s | 0.1% | 0.08 rad (ankle_roll) |
   | distress cluster (pre-fall) | 67.7-73.9 s | 4.8% | 0.08 rad (ankle_roll) |
   | fall tail | 73.9-81.8 s | 96.6% | pitch 0.51, roll 1.5 rad (toppled) |

   **Verdict: A0's steady-state gait does not ride its joint stops the way A1a's did**
   (~3.6% duty cycle at ≤0.02 rad margins, vs A1a's reported 40% at 0.01-0.2 rad every
   stride — roughly an order of magnitude lower on both duty cycle and margin). **Leave
   `State_RLBase`'s whole-body chasing hold as-is; no clamp port needed on this evidence.**
   The G3.0 terminal-behavior design question (ratchet/no-release once triggered) is still
   open generically but is not urgently implicated by A0's normal joint-limit path.

   **Separate finding, flagged for the user (not a fix, an open question):** this same
   session ends in a **real, previously unreported fall** at t≈74 s. Fine-grained trace:
   pitch climbs smoothly from -0.09 to +0.15 rad over 73.0-73.7 s with *no* trigger active,
   then from 74.0-74.3 s pitch reaches 0.51 rad **and roll jumps from ~0.1 to ~1.5 rad** —
   the robot topples sideways. `trig_tilt`/`trig_fall` engage only at 74.15-74.25 s, i.e.
   *after* the tilt limit was already exceeded and the roll excursion was already
   underway — unlike the A1a mechanism (hold engaged at a recoverable tilt of 0.196 and
   arguably caused the fall), here the filter looks like it caught an already-committed
   loss of balance, not created one. Root cause is open: the base `SafetyLogger` CSV logs
   no commanded velocity, so there is no way to tell from this file alone what command was
   active during the 67.7-74 s "distress cluster" that preceded it. **Ask the user**
   whether they recall what was being commanded in that session around t=67-74 s (e.g. a
   fast turn, a push, a speed change) — this is worth a source, not just a shrug, before
   the first real-hardware attempt.

2. **`h1_2_limits.h` vs. the sim model — AUDITED 2026-07-20, wider than the known knee
   issue.** Cross-checked `h1_2_limits.h` against a 5-source consensus: 3 independently
   vendored H1-2 URDFs from separate repos in this workspace
   (`h12_locomotion_rma/isaaclab`, `unitree_h12_rma_book/isaaclab`,
   `corelllab/h12_adaptive_policy`, all using this project's joint-naming convention —
   single `elbow` + `wrist_roll/pitch/yaw`, not the older 2-DOF-elbow variant some other
   URDFs in the workspace use) plus the mjlab training scene XML
   (`src/assets/robots/unitree_h1_2/xmls/h1_2.xml`) plus the live `/opt` bridge scene XML
   (`h1_2_handless.xml`) — all five agree with each other on every one of the 27 joints.
   **15 of 27 joints in `h1_2_limits.h` disagree with that consensus**, not just the knee:

   | joint | h1_2_limits.h | consensus (URDF×3 + both scene XMLs) |
   |---|---|---|
   | left_knee | (-0.26, 2.05) | **(-0.12, 2.19)** |
   | right_knee | (-0.26, 2.05) | **(-0.12, 2.19)** |
   | waist_yaw | (-3.14, 1.57) | **(-2.35, 2.35)** |
   | left_shoulder_yaw | (-3.01, 2.66) | **(-2.66, 3.01)** (= consensus right value) |
   | right_shoulder_yaw | (-2.66, 3.01) | **(-3.01, 2.66)** (= consensus left value) |
   | right_shoulder_pitch | (-1.57, 3.14) | **(-3.14, 1.57)** (mirrored, should = left) |
   | left_elbow | (-2.53, 1.60) | **(-0.95, 3.18)** |
   | right_elbow | (-1.60, 2.53) | **(-0.95, 3.18)** |
   | left/right_wrist_roll | (-2.967, 2.967) | **(-3.01, 2.75)** / **(-2.75, 3.01)** |
   | left/right_wrist_pitch | (-0.471, 0.349) | **(-0.4625, 0.4625)** |
   | left/right_wrist_yaw | (-1.012, 1.012) | **(-1.27, 1.27)** |

   (Legs other than the knee, ankles, hip yaw/roll/pitch, left_shoulder_pitch/roll,
   right_shoulder_roll match the consensus exactly — 12/27 joints are correct as-is.)
   Pattern: the two shoulder_yaw rows look **swapped between L/R**; right_shoulder_pitch
   looks **mirrored** (should equal the left value, since pitch is a sagittal-plane axis
   that doesn't mirror sign in this URDF's convention, unlike roll/yaw which correctly do
   mirror); the elbow/wrist rows look like they come from a different reference
   altogether (consistent range *widths* but shifted centers/signs). This is
   **BLOCKING for A0 hardware**: item 1's whole-body hold reads exactly these limits, so a
   too-narrow entry (e.g. waist's positive side, 1.57 vs the true 2.35) is a false-positive
   hold risk on a perfectly safe pose, and a too-wide entry (e.g. waist's negative side,
   -3.14 vs the true -2.35) is a false-negative gap that lets the real joint travel past its
   mechanical limit with no safety response at all.
   **FIXED 2026-07-21 (approved).** All 15 rows replaced with the consensus values;
   `h1_2_ctrl` (both `State_RLBase` and `State_RLHRL`, which share this header) rebuilt
   clean, no other changes needed. Source of the original error: hand-entered from a
   Unitree documentation page rather than the URDF/MJCF, per the user.

   **A1 connection, worth re-checking:** `State_RLHRL` reads the same header for its
   per-joint clamp. Of the A1a post-mortem's "ankle/hip/knee stops grazed 0.01-0.2 rad
   every stride" claim, ankle and hip were already correct in the old header, but **knee
   was wrong in exactly the implicated direction** — old positive bound 2.05 vs the true
   2.19, a 0.14 rad gap squarely inside that 0.01-0.2 rad graze band. So some fraction of
   A1a's historical knee-trigger rate may have been a false-positive against a too-narrow
   limit, not a real graze. Not re-measured here (A1 keeper is WL-B1, out of this track's
   scope) — flag for WL-B1 to re-run its own `trig_joint` measurement against the fixed
   header before concluding anything about the live gait itself.

3. **Formal G2 recording pass for A0 — headless portion DONE 2026-07-20, live portion
   prepped for the user's session.** Headless pre-checks (no live bridge needed):
   - `onnx_parity.py` on the deployed checkpoint
     (`h1_2_velocity_v2/2026-07-15_08-29-26_a0_v2_optB_rs20_baseline/model_10000.pt`):
     **PASS**, max_abs_diff 1.24e-05.
   - `bridge_replica.py --policy a0`, both scenes (`scene.xml` vendor, `scene_stress.xml`),
     delay 0/2/4 ms, `--step-vx 1.0 --walk-vx 0.5` (covers the historical vx=1.0-from-stand
     killer + G2.2's 0.5 hold + G2.7's stress-scene requirement in one pass each):
     **6/6 PASS**, no falls, swing clearance 0.05-0.12 m throughout, qvel_rms stays bounded
     even at 4 ms on the vendor scene.
   - G1.1 reference gap filled: the plan only had mjlab ss numbers at vx 0.5/1.0. Ran the
     missing 0.3 point: **ss_err_vx 0.041, t90 0.54 s, 0 falls** (same checkpoint, same
     harness) — G2.2 now has a number to check parity against at all three held speeds
     the A0 gate table asks for.
   - G2.4 split-mode fixture prepared: `deploy.yaml.g2_4_split_test` (new file, same dir as
     `deploy.yaml`) — a `deploy.yaml` copy + `hold_joint_ids: [12..26]`, loaded via
     `H1_2_DEPLOY_CFG=deploy.yaml.g2_4_split_test` before launching `h1_2_sim`. Mirrors the
     existing `deploy.yaml.w1_legacy_test` fixture convention (live `deploy.yaml` never
     touched). Needed because `deploy_real.yaml` already has the same `hold_joint_ids` but
     drives via joystick obs, not the keyboard obs this sim session needs.
   - **Tooling gap FIXED 2026-07-21** (approved). `safety_logger.h` (shared, both FSMs)
     now logs `cmd_vx/vy/wz` (the active command, re-derived in `State_RLBase::run()` from
     whichever term the deploy config actually uses — keyboard for sim, joystick for real
     — since A0 has no cached command member the way A1 does) and `ach_vx/vy/vz` (ground-
     truth body-frame base velocity from the sim bridge's `SportModeState`, the same
     privileged source `State_RLHRL` already reads for its goal state — **SIM ONLY**,
     reads 0 on real hardware, same estimator gap as E1). `h1_2_ctrl` rebuilt clean.
     `deploy_gate_analyzer.py` now auto-detects the schema (`s0` column present =
     `_hrl.csv`, else the base `SafetyLogger` schema) and handles both — A0's CSV is
     readable by the tool for the first time, using triggers (not a height column) for
     fall detection and a live-derived act_rate (base CSV has no pre-computed one).
     Verified against a synthetic base-schema CSV (segmenting, err_vx/vy, stride proxy,
     the `sim_only_velocity` all-zero warning) and regression-checked against a real
     existing `_hrl.csv` — **not yet exercised against a live-captured base CSV**, since
     no session has been recorded with the rebuilt binary yet; that's the first thing to
     sanity-check in the next live/bridge session, same caveat as the G2.4 fixture.
     Pre-2026-07-21 base CSVs (no `cmd_vx` column) are no longer readable by
     `deploy_gate_analyzer.py` — it now fails with an explicit message pointing at
     `safety_analyzer.py` instead of a raw traceback.
   - **Live-session key scripts** (H1-2 keyboard map: `i`=FixStand, `o`=A0 walk,
     `p`=Passive; inside `o`: `0`=stop, `1`-`9`=held vx 0.1-0.9, `w`=vx **1.0** (edge,
     hardcoded, within the trained/clamped range), `s`=vx **-0.5** (only backward preset —
     see caveat), `a`=vy **+0.5**, `d`=vy **-0.5**, `q`=wz **+0.5**, `e`=wz **-0.5**; each
     key press SETS the full [vx,vy,wz] triple, so combined walk+turn isn't reachable with
     one key):
     - **G2.1** (stand 60 s): `i` → `o` (enters at cmd 0) → wait 60 s → `p`.
     - **G2.2** (held-command parity, vendor scene): `o` → `3`(≥30s)→`0`(~3s)→
       `5`(≥30s)→`0`→`w`(≥30s)→`0`→`s`(~15s)→`0`→`a`(~15s)→`0`→`d`(~15s)→`0`→
       `q`(~15s)→`0`→`e`(~15s)→`0`.
       **Caveat:** the plan text asks for backward -0.3 and strafe ±0.3, but the keyboard
       only has `s`=-0.5 and `a`/`d`=±0.5 (no ±0.3 preset exists) — this run is a stricter
       (larger-magnitude) test than the plan literally asked for, and there is no mjlab
       reference at exactly -0.3/±0.3 to compare against either (G1.1 only benchmarks vx).
       Score these three as pass/fail (0 falls, direction held, arms calm), not against a
       numeric parity bar.
     - **G2.3** (steps): `0`(settle)→`5`(≥5s)→`0`(≥3s)→ re-enter stand →`w`(≥5s, the
       0→1.0-from-stand killer)→`0`→`5`(≥5s, walk)→`q`(≥5s, turn — vx drops per the
       single-key-sets-triple behavior above)→`0`(stop).
     - **G2.4**: same three sequences as G2.1-G2.3, but launch with
       `H1_2_DEPLOY_CFG=deploy.yaml.g2_4_split_test` first (see fixture above).
     - **G2.7**: edit `robot_scene:` in `/opt/unitree_mujoco/simulate/config.yaml` from
       `scene.xml` to `scene_stress.xml`, restart the bridge, repeat G2.2's `5`(≥30s) hold
       and G2.3's step sequence; revert the config line after.
     Flight recorder is auto-on via `H1_2_SAFETY_LOG` (set by `h1_2_sim`/`h1_2_real`); run
     `python scripts/safety_analyzer.py <base>` after each session for the trigger/violation
     numbers, and `deploy_gate_analyzer.py` does not apply to A0 CSVs (see tooling gap
     above).

4. **Arms: split-vs-live — RECOMMENDATION (2026-07-20).** Decision 7 ("split first, live
   later") was written because the A1 keeper's arms were wild. A0-optB's arms are already
   calm in sim (bench `ub_arm_vel` 0.143 and `pose_dev` 0.0004, vs the A1a keeper's
   0.50-0.64 and ~70x). **Recommend: split first for the very first session (G3.1-G3.3)
   regardless** — not because A0's arms are risky, but because split is also the locked
   hardware config and cuts the actuated-joint failure surface while balance/legs are
   still the unknown for a first-ever A0 hardware attempt. **But do not gate live arms
   (G3.4) behind a full stage-D close** (repeatable free walk across ≥2 sessions) the way
   decision 7 implicitly does for A1 — A0's arm risk doesn't warrant that long a validation
   window; try G3.4 within the same campaign once the first clean G3.3 free-walk lands.
   This is a recommendation only; the call is the user's.

### First A0 live bridge session (2026-07-20, the user) — G2.1 informally OK, G2.2 VOID, 2 defects found

**What was run:** G2.1-G2.3 in one session. 60 s stand from reset with `o` active, no band
(passes the G2.1 bar informally). Then strafe (`a`/`d`) and yaw (`q`/`e`): **no movement on
any of the four**, and the two strafe keys appeared switched. Then `w` (G2.3): fell on the
first attempt, no falls on the retries.

> **ROOT CAUSE FOUND 2026-07-21 (supersedes the defect-1 reading below): `gait_phase` is
> pinned to (0,0) in every keyboard bridge session.** See "Defect 0" immediately below. The
> keyboard-latching issue (defect 1) is real code but is NOT the explanation — the user held `d`
> for 15 s and the robot leaned and stayed leaning, which proves the command reached the
> policy. Kept below as a genuine secondary robustness defect, downgraded.

**Defect 0 (THE root cause): the deploy `gait_phase` clock is permanently zero under
keyboard control, so no policy ever receives its gait clock.**
`deploy/include/isaaclab/envs/mdp/observations/observations.h:125-150` — `gait_phase`
integrates `global_phase` correctly, but computes its stand-mask by calling
**`isaaclab::mdp::velocity_commands(env, params)`**, which (same file, lines 110-123) reads
the **JOYSTICK** (`joystick->ly()/lx()/rx()`). In a keyboard-driven sim bridge session the
joystick is neutral, so `cmd_norm ~ 0 < 0.1` on every tick and the term returns **(0, 0)
forever** — regardless of what the keyboard is commanding. The policy's 2-dim clock
observation is dead for the entire session.

**Why this explains everything observed, and why sim never caught it:**
- The project already measured A0's sensitivity to exactly this: the 2026-07-02
  phase-scramble test found **A0 is near-perfectly phase-slaved to its clock obs**
  (match 0.973; **zero clock -> err_vx 0.075 -> 0.41, stride 0.57 -> 0.23**). A zeroed clock
  is a known, measured, catastrophic degradation - not a subtle one.
- cmd 0 stand is UNAFFECTED (zeroing is correct there) -> the 60 s stand passed. ✓
- `w` at vx 1.0 still shuffles forward but badly degraded -> **the first-attempt fall**. ✓
- vy 0.5 / wz 0.5 are far weaker demands; without the clock the policy leans instead of
  initiating a step -> "**leaned right for 15 s but didn't make a step**", and no turning. ✓
- **mjlab cannot reproduce it** (the phase obs is correct there): a held-command matrix run
  2026-07-21 across optB-rs8 / rs20 / fric-kl01 / energy at held vy 0.5 and held wz 0.5 shows
  **all four essentially identical and all stepping** (stride 0.571-0.572, gait_match
  0.95-0.97, `ss_err_vy` 0.109-0.118, `err_yaw` 0.063-0.095, 0 falls). ✓
- **`scripts/bridge_replica.py:144-146` gates the phase on the command it actually sends**
  (`pha = zeros if norm(cmd) < 0.1 else [sin, cos]`) — i.e. the replica is CORRECT and the
  live C++ is not. **This finally explains the standing replica-vs-live contradiction**
  (every headless pre-check passes at 0/2/4 ms; live sessions fail), open since 2026-07-16. ✓

**The rs20 hypothesis is FALSIFIED** (the user's, reasonable from the evidence he had: the only
bridge-working policy was the only rs8 one). The mjlab held-command matrix above shows rs20,
fric-kl01 and energy strafe and turn exactly as well as optB-rs8. The apparent policy split on
the bridge is idiosyncratic degradation under a dead clock, not a training-side defect.

**MAJOR IMPLICATION FOR WL-B1 / WL-D (escalate):** `State_RLHRL` uses this same `gait_phase`
term (it writes the period at `State_RLHRL.cpp:271-276` and reads it back at 299), so **every
A1 keyboard bridge session also ran with a dead clock** — and the A1 LL is *more* phase-slaved
than A0 by construction (cadence-entrained, gait_match 0.94-0.96). This is now the prime
suspect for: (a) the 2026-07-17 A1 held-0.5 failures (4 falls / ~6 attempts) that were routed
to **WL-D as a "chronic gait/clearance shortfall"**; (b) **G2.6's "BOTH cadence modes fell"** —
pinned and HL-owned alike, which is exactly what a zeroed output does to both; (c) the swing
clearance gap measured live. **Do not act on the WL-D clearance routing until the A1 bridge
runs are repeated against a fixed `gait_phase`.**

**Fix constraint (load-bearing):** the defect lives in `deploy/include/isaaclab/`, which is
**off-limits** per the standing no-isaaclab-edits rule. Fix it **robot-locally**, mirroring
the pattern already used for `keyboard_velocity_commands` (registered in
`deploy/robots/h1_2/src/State_RLBase.cpp:25`): register a robot-local phase term that takes
its stand-mask from the command term the config actually uses, and point the deploy YAMLs at
it. Note WL-B0 already fixed this exact class of bug for the telemetry path on 2026-07-21
(`State_RLBase.cpp:161`, `use_keyboard ? keyboard_velocity_commands : velocity_commands`) —
the same branch was simply never applied to `gait_phase`. Spec-first; deploy/ is WL-B's file.

#### Defect 0 FIXED 2026-07-21 (robot-local), confirmed live in the bridge for A0 AND A1

**What shipped.** New robot-local header `deploy/robots/h1_2/include/h1_2_observations.h`,
included by both `State_RLBase.cpp` and `State_RLHRL.cpp` (so registration cannot depend on
which object files the linker keeps). It holds:

- `keyboard_velocity_commands` — **moved here unchanged** from `State_RLBase.cpp` (it is now
  needed by two call sites, so the doc pointer for the key map is this header, not the .cpp).
- `uses_keyboard_commands(env)` — reads the deploy config to decide keyboard vs joystick.
  Handles both YAML layouts (A0 lists terms at the top level of `observations:`, A1 nests
  them in `policy:`/`command:` groups). `State_RLBase`'s WL-B0 telemetry branch now calls it
  instead of its own hand-rolled top-level lookup.
- `gait_phase_cmd` — a **new term name** (a second `gait_phase` registration would silently
  overwrite the shared one in `observations_map()` under static-init order, so it is not
  safe to shadow). Byte-identical to `isaaclab::mdp::gait_phase` — same `env->global_phase`
  accumulator, same `period` param, same sin/cos, same `<0.1` zeroing — except the
  stand-mask reads the command source the config actually uses. Also caches its last output
  in `isaaclab::g_gait_phase_obs` for the flight recorder.

The four live yamls (`velocity{,_hrl}/v0/params/deploy{,_real}.yaml`) plus the `g26_pinned`
and `g2_4_split_test` fixtures now name `gait_phase_cmd`; the two `w1_legacy_test` fixtures
deliberately still name `gait_phase` (that is what they are for) and still load, because the
shared term was not touched. **Obs dims are unchanged** (A0 92, A1 policy 89 + command 3) and
no ONNX was re-exported or swapped.

`State_RLHRL` binds the clock's `params` node **once in the ctor** to whichever term name the
yaml uses, and writes/reads the HL cadence through that node — so a legacy fixture keeps
working instead of silently retuning a key nothing is bound to.

**Trap found while implementing (worth knowing):** yaml-cpp's **non-const** `operator[]`
INSERTS an undefined child for a missing key. The source probe runs from inside the
`ObservationManager`'s term loop, so a non-const lookup mutated `observations` mid-iteration
and threw `YAML::InvalidNode` on the next term. Every probe node is now `const`. The
pre-existing WL-B0 line at `State_RLBase.cpp:161` had the same shape (outside iteration, so
harmless) and was replaced by the shared helper.

**New permanent instrument.** `safety_logger.h` logs `phase_sin,phase_cos` next to the
`cmd_*`/`ach_*` columns (both FSMs), and `deploy_gate_analyzer.py` reports **`phase_alive`**
per segment = fraction of ticks with a non-zero clock. Read: ~0.0 on a cmd-0 segment is
correct (stand-mask), ~1.0 on a commanded segment is a working clock, **0.0 on a commanded
segment is this bug**. The analyzer prints an explicit WARNING in that case, and a NOTE on
CSVs old enough to lack the columns.

**Verified (2026-07-21):**
- Build clean (`h1_2_ctrl`), all 9 deploy yamls parse with every term registered and
  unchanged dims/order.
- Standalone regression test `deploy/robots/h1_2/test/phase_obs_test.cpp` (not in CMake;
  build/run line in its header) drives the real `ObservationManager` over the real yamls
  with a stub articulation + a real `Keyboard` on a piped stdin. **11/11 pass** with key `w`
  and again with key `0`: source detection correct on all four live configs; joystick path
  unchanged (neutral → masked, forward → alive, phase advances); **keyboard command +
  neutral joystick → clock ALIVE for both A0 and A1**; and a negative control confirming the
  untouched shared `isaaclab::mdp::gait_phase` still returns (0,0) under the same conditions.
- **Live in the actual C++ bridge**, headless-driven (unitree_mujoco + `h1_2_ctrl` with keys
  piped into stdin; `i` → `o`/`h` → hold `5` 12 s → `0`). `phase_alive`: A0 **0.0 / 0.999 /
  0.0** and A1 **0.0 / 1.0 / 0.0** across the stand/hold-0.5/stop segments; sin/cos ride the
  unit circle at a 0.625 s measured period against the 0.6 s yaml value; A1's `_hrl.csv`
  `period` column still varies (0.35-1.0), i.e. the HL cadence write survives the rename.

**What that live run does NOT show.** It ran with the bridge's default elastic band engaged
and no GUI interaction, so the robot was suspended: `act_rate` ~0.001 and achieved vx ~0.01
under a held 0.5 command, and A1's tilt trigger was latched (`alpha_max` 1.0) from the first
segment. **Nothing in it is a locomotion or gate result** — it validates the clock and the
instrument only. G2.x still needs a real operator session.

**A1 status (the WL-D escalation):** the A1/`State_RLHRL` path **is** fixed and confirmed
alive by the same live capture. So the 2026-07-17 A1 held-0.5 failures and G2.6's "both
cadence modes fell" were both produced under a dead clock and are **not interpretable as a
gait/clearance shortfall**. Those A1 bridge runs are worth repeating on this build before
WL-D acts on the clearance routing.

**Defect 1 (DOWNGRADED - real, but not the cause): the keyboard command is MOMENTARY, not
latched.**
`keyboard.h:108/133-135` — `select()` uses an 80 ms timeout and clears `_key = ""` whenever
no byte arrives in that window; `State_RLBase.cpp:52-55` maps the *current* key each tick
and returns `{0,0,0}` when it is empty. So a **tapped** key commands its velocity for ~80 ms
(~4 control ticks at 50 Hz) and then reverts to zero — invisible at the 0.5 m/s strafe/yaw
magnitudes, while a **held** `w` at 1.0 m/s survives because terminal auto-repeat (~30 ms)
keeps re-arming `_key`. **Corrected 2026-07-21:** this is NOT what the user
observed — he held `d` for 15 s and the robot leaned and stayed leaning, so the command was
sustained (terminal auto-repeat works) and reached the policy. Defect 0 above is the actual
cause. Defect 1 remains worth fixing as a robustness/protocol issue, not as the explanation.
**Note the documented intent is violated:** the comment at `State_RLBase.cpp:32-33` calls the
number keys "held forward-speed presets for the deploy-gate battery (held-command tests)",
but nothing latches, so G2.2's ">= 30 s held" is only achievable by physically holding a key
for 30 s and trusting terminal auto-repeat. Also note the first ~250-500 ms of any hold has a
dropout (auto-repeat initial delay > the 80 ms clear), so a "step from stand" is really a
short pulse, a gap, then the step — which matters for G2.3's transient tests.
**Proposed fix (WL-B0's call, needs approval — deploy/ is WL-B's file):** latch the last
non-zero command until `0` (stop) or a new key, matching the documented preset semantics.
Small and well-scoped, but it changes deploy behavior, so spec-first.

**Defect 2 (VOIDS the session's walk gates): the deployed A0 ONNX is the WRONG RUN.**
Weight-compared the deployed `velocity/v0/exported/policy.onnx` against every A0 candidate:
it is **bit-identical (max weight diff 0.000e+00) to
`2026-07-18_01-56-14_a0_baseline_energy0p05_s42/model_10000.pt`** — the **WL-D "A0+energy"
experimental arm** — NOT `a0_v2_optB_rs20_baseline`, which it differs from by 0.48 max weight.
The ONNX metadata could not catch this: it records `run_path = local` (the expected value for
any `--checkpoint-file` export) and its gains/scales/joint order are optB-lockstep, so the
file looks correct on inspection. **This is the second instance of the same class of error**
(see the 2026-07-17 G2.0 pre-check item 2, where the wrong checkpoint was staged for the A1
pair) — the staging step has no provenance check.
**Consequence:** G2.2 is a PARITY gate against *the same checkpoint's* mjlab numbers, so any
walk result from this session is uninterpretable and the gate must be re-run. G2.1 (stand)
is only weakly affected but should be re-recorded on the final candidate.
**Not the cause of defect 1:** A0+energy's sim tracking is indistinguishable from the
baseline's (err_vy 0.113 vs 0.111, err_yaw 0.086 vs 0.089), so it cannot explain the missing
strafe/yaw motion.
**Silver lining, and a real decision for the user:** A0+energy is arguably the *better* deploy
candidate — CoT 0.440 vs 0.541, power 134 vs 166 W, action_rate 0.571 vs 0.644, `ub_arm_vel`
0.118 vs 0.143, at equal tracking and 0 falls (table (i)). Calmer and more efficient is
exactly what a first hardware session wants. But it must be a **deliberate** choice with
recorded provenance, not an accident. **Decide the candidate, then re-export and re-run G2.**

**The `w` fall is expected-risk, not a new finding:** `w` commands vx = 1.0, i.e. the
`0 -> 1.0 from stand` case that G2.3 itself labels "the historical bridge killer", at the top
edge of the trained range (`lin_vel_x [-0.5, 1.0]`). Combined with the defect-1 pulse-gap-step
profile, a first-attempt fall is unsurprising. "Twisted from `e`" is unlikely to be the cause
(`e` produced no motion), but entry attitude did matter in the A1 sessions (40.2 deg -> stumble),
so re-test from a clean reset. Use the number presets (`3` then `5`) before `w`, per the
existing session guidance.

**Actions (WL-B0):** (1) settle the candidate (baseline vs A0+energy) with the user; (2) re-export
with provenance recorded and add a staging-time identity check (the weight-compare above is
~20 lines and should become a scripted gate, cf. the "generic tooling" convention);
(3) spec the keyboard latch fix; (4) re-run G2.1-G2.3 on the settled build; (5) only then
judge the `w` fall.

### Post-fix bridge session (2026-07-22, the user) — A0 CONFIRMED GOOD, A1 fails on twitch; candidate LOCKED

**A0: the fix is validated behaviourally.** With `gait_phase_cmd` live, **every** flat policy
The user had previously tested now responds to `q`/`e`/`a`/`d`, walks "way smoother", and visibly
lifts its legs — including the ones that failed before. His words: it behaves like the earliest
sim-to-sim attempts, before the accumulated changes. This closes Defect 0 behaviourally (the
2026-07-21 headless capture only proved the clock and the instrument, with the robot banded).
It also retro-explains the WL-E "A0 barely lifts its feet on the vendor plant" observation as
at least partly clock-dead, not plant — **note ADR-0005 Amendment 2 item 1 measured swing apex
in `bridge_replica.py` (correct phase) so that measurement stands, but any LIVE-bridge foot-drag
impression from before 2026-07-21 is void.**

**A0 deploy candidate LOCKED (the user, 2026-07-22): `a0_v2_optB_rs20_baseline`.** Rationale: a clean
A0-vs-A1 architectural comparison with **no CoT term injected**, i.e. the A0+energy arm is
rejected as the deploy candidate *precisely because* its `cost_of_transport_penalty` would
confound the energy axis of the RQ2 comparison — the A0-comparison-cleanliness rule applied
correctly. (A0+energy stays a legitimate WL-D result and a future option; it is not the control.)
**ACTION: the staged `velocity/v0/exported/policy.onnx` is still bit-identical to
`a0_baseline_energy0p05_s42` (defect 2) — re-export from the rs20 baseline and add the staging
identity check before the G2 re-run.**

**A1: all four tested candidates FAIL, and the cause is twitch, not clearance.** The user tested D2,
the old keeper, arm4c (foot clearance) and arm3 (stand-still) on the fixed build: all "really
unsettled and shaky — that's mainly the reason they fall". The 2026-07-20 benchmark set ranks
every run by `action_rate` and the split is total, with **no overlap**:

| tier | runs | action_rate | ub_arm_vel |
|---|---|---|---|
| **A0 (all)** | energy 0.571, corrnoise 0.623, **rs20 0.644**, widedr 0.751 | **0.57-0.75** | 0.12-0.22 |
| **A1 (all)** | arm4d 0.924, arm5/arm2b 1.066, arm3 1.090, D2 1.126, arm4c 1.151, ... fix0p8 1.273 | **0.92-1.28** | 0.30-0.48 |

So A1 is intrinsically 1.5-2x twitchier than A0 at the action level — the long-known "keeper is
~10x twitchier in the bridge" residual (post-mortem issue 2), now quantified in sim. On a bridge
carrying explicit 500 Hz PD + 2-4 ms latency, high-frequency action IS the destabilising
mechanism (the W2 energy-injection finding), so this is the expected failure mode, not a new one.

**Two consequences for WL-D's clearance routing, which is now doubly weakened:** (i) the live
evidence behind it was collected under a dead clock and is void; (ii) **arm4c (foot clearance)
has now been tested live on the FIXED build and still failed** — raising clearance does not fix
the A1 bridge instability, and arm4c is in fact one of the *twitchier* arms (1.151 vs D2's 1.126).
Clearance is not the A1 deploy blocker. **Smoothness is.**

**Recommended next A1 bridge test (not yet run): `arm4d` (`ll_energy_coef=0.05`).** It is the
**calmest A1 policy in the entire batch by a clear margin** — action_rate 0.924 (18% below D2,
next-best 1.066), lowest A1 `ub_arm_vel` 0.300 (vs D2 0.413), best A1 `orient_dev` 0.028 (equal
to A0), at unchanged tracking and 0 falls in sim. It was never tested on the bridge. `arm5`
(ankle-roll, 1.066, best probe errors in the batch) is the second candidate. If neither survives,
the lever is a dedicated smoothness arm (raise `ll_action_rate_coef` above the keeper's 0.02) —
a WL-D training arm, not a deploy fix.

**All pre-2026-07-21 A1 bridge results are void** (dead clock): the 2026-07-17 held-0.5 falls,
G2.6's "both cadence modes fell", and the live swing-clearance impressions. A1 needs its gate
data re-baselined on the fixed build regardless of candidate.

**A1 candidate results, live GUI bridge, fixed build (2026-07-22, the user) — arm4d PASSES:**

| candidate | sim action_rate | live bridge result |
|---|---|---|
| **arm4d `ll_energy_coef=0.05`** | **0.924** | **PASS — no fall, incl. 0->w step-from-stand and all directions** |
| arm5 ankle_roll | 1.066 | **CATASTROPHIC — fell at ZERO command, one leg kicked violently** |
| arm3 stand-still | 1.090 | fail (shaky) |
| D2 control / old keeper | 1.126 | fail (shaky) |
| arm4c foot clearance | 1.151 | fail (shaky) |

Reads: (i) **arm4d is the A1 deploy candidate** — it was predicted from the action-rate
ranking (calmest A1 in the batch) and the prediction held. The smoothness hypothesis is now
supported by both the sim ranking and the live outcome. (ii) **arm5 is a sim-invisible
failure mode and the more important finding**: it had the batch's BEST goal-probe errors
(hl 0.041 / ll 0.057), best `height_dev`, 0 sim falls — and it cannot stand still on the
bridge. **The deterministic benchmark cannot see whatever this is.** Leading hypothesis: the
ankle_roll posture anchor drives a joint whose range is only ~+-15 deg toward its limit, and
`State_RLHRL`'s per-joint command clamp fires (check `trig_joint`; note `h1_2_limits.h` was
corrected for 15/27 joints on 2026-07-21). Until it is explained, treat "good sim probe
numbers" as NOT predictive of bridge stability. (iii) arm4c failing again on the fixed build
confirms clearance is not the blocker.

### Headless candidate sweep (2026-07-22, WL-B1) — arm4d confirmed clean, arm5 mechanism found

**`bridge_replica.py` extended** (robot-local script, no isaaclab touched): held lateral
(`--walk-vy`) and yaw (`--walk-wz`) command phases added after the existing vx walk, plus a
second gentler step-from-stand (`--step-vx2`, default 0.5, ahead of the existing 1.0
`--step-vx`) — the chain is now `stand -> step0.5vx -> stop -> step1.0vx -> stop -> walkVx ->
stop -> walkVy -> stop -> walkWz` (`--skip-lateral` restores the old vx-only chain for quick
smoke checks). `--onnx-dir` already existed (2026-07-15) and needed no change to point at an
arbitrary candidate's exported ONNX pair instead of the staged `deploy/.../exported/`.

**Two real bugs found and fixed in the replica while wiring this up** (not just the new
phases — both would have silently corrupted the sweep):

1. **Per-joint command clamp was diagnosed but never applied.** The instrument added to
   check `trig_joint` (State_RLHRL.cpp:373-376: the raw policy target gets clipped to
   `h1_2_joint_limits` before it reaches the motor) originally only *measured* whether the
   target was out of range and kept driving the PD off the **unclamped** target — not what
   the real bridge/hardware ever does once a joint rides its stop. Fixed: `pol()` now clips
   `tgt` to `JOINT_LIMITS` (a python mirror of `h1_2_limits.h`, corrected 2026-07-21 values,
   kept in sync by hand) exactly like the C++ path, and the trig_joint duty-cycle diagnostic
   reads the pre-clip value. This materially changed results (see below) — the pre-fix
   replica could not have found the arm5 mechanism at all.
2. **`c` / `hl_cadence_source` / `cadence_period_range` were read from the shared
   `deploy.yaml`, not the loaded candidate's own ONNX metadata.** `deploy.yaml`'s `hrl:`
   block describes whichever checkpoint is currently staged, but `--onnx-dir` swaps in a
   *different* candidate — these HRL structure fields are baked per-candidate at export
   time and can legitimately disagree (caught via a crash: `fix0p8` trained a **fixed 0.8 s
   clock**, `hl_cadence_source=random`, HL action dim 3 with no period channel, while the
   staged `deploy.yaml` has `hl_cadence_source=hl` → `IndexError` reading a non-existent 4th
   action dim). Fixed: `c`, `hl_owns_period`, `cadence_period_range` now all come from
   `high_level.onnx`'s own metadata; `deploy.yaml`'s `pin_period` remains the one legitimate
   operator override.

Also added, matching what the task asked to report: **`act_rate`** (mean `|Δaction|` at each
policy tick, the project's standard smoothness metric) and **`trig_joint_duty`** + top-3
offending joints per phase (a duty-cycle proxy for the C++ flight recorder's `trig_joint`
column, since no live flight-recorder CSV exists for a headless run).

**Candidates swept:** all 10 named (D2 control, arm2a/2b, arm3, arm4a/4b/4c/4d, arm5,
fix0p8), `model_10000` from each `logs/rsl_rl/h1_2_velocity_a1_v2/` run, ONNX-exported fresh
(none had a prior export) via `play.py --export-onnx`. Both scenes (`scene.xml` vendor,
`scene_stress.xml`), delay 0/2/4 ms, all three command axes plus both step-from-stand cases
= 60 runs total.

**Pass/fail matrix** (fall = height < 0.7 m or |pitch| > 0.5 rad; `F@phase` names where):

| candidate | vendor d0 | vendor d2 | vendor d4 | stress d0 | stress d2 | stress d4 | pass/6 |
|---|---|---|---|---|---|---|---|
| D2 (control) | PASS | F@stop4 | F@stand | PASS | F@stop4 | F@stop4 | 2/6 |
| arm2a | PASS | PASS | PASS | F@stop4 | F@walk_wz | F@walk_wz | 3/6 |
| arm2b | F@stand | F@takeover | F@takeover | F@stop4 | PASS | PASS | 2/6 |
| arm3 | PASS | PASS | F@stop2 | F@stop4 | F@stop4 | F@stop4 | 2/6 |
| arm4a | PASS | PASS | F@takeover | PASS | PASS | PASS | 5/6 |
| arm4b | F@stop4 | F@stop4 | F@stop4 | F@stop4 | PASS | F@stop4 | 1/6 |
| arm4c | PASS | PASS | F@stand | PASS | PASS | PASS | 5/6 |
| **arm4d** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **PASS** | **6/6** |
| arm5 | F@stop4 | PASS | F@step0.5vx | F@stop4 | F@stop4 | F@stop4 | 1/6 |
| fix0p8 | PASS | PASS | F@walk_vy | PASS | PASS | PASS | 5/6 |

`stop4` = the settle-to-zero phase immediately after the held-vy (strafe) command, the phase
this sweep's lateral extension newly exercises.

**Smoothness summary** (mean over every phase reached, all 6 conditions per candidate):

| candidate | replica pass/6 | replica mean act_rate | replica mean clr | replica mean trig_joint | sim act_rate (2026-07-20 bench) |
|---|---|---|---|---|---|
| D2 | 2/6 | 0.153 | 0.048 | 0.337 | 1.126 |
| arm2a | 3/6 | 0.151 | 0.032 | 0.793 | 1.070 |
| arm2b | 2/6 | 0.161 | 0.047 | 0.456 | 1.070 |
| arm3 | 2/6 | 0.138 | 0.043 | 0.490 | 1.090 |
| arm4a | 5/6 | 0.130 | 0.046 | 0.498 | 1.130 |
| arm4b | 1/6 | 0.201 | 0.030 | 0.591 | 1.260 |
| arm4c | 5/6 | 0.158 | 0.054 | 0.620 | 1.150 |
| **arm4d** | **6/6** | **0.118** | 0.022 | 0.355 | **0.920** |
| arm5 | 1/6 | 0.148 | 0.041 | **0.289 (lowest)** | 1.066 |
| fix0p8 | 5/6 | 0.191 | 0.058 | 0.720 | 1.273 |

Reads: (i) **arm4d is confirmed the clean sweep winner** — 6/6 pass, lowest replica act_rate
*and* lowest original sim act_rate, matching the live GUI PASS exactly. (ii) **arm5 is the
worst replica performer (1/6) despite having the LOWEST mean trig_joint_duty of the whole
batch (0.289)** — clamp *frequency* does not predict instability here; see the mechanism
below. (iii) fix0p8 (highest sim act_rate in the whole project, 1.273) passes 5/6 in the
replica despite the highest trig_joint_duty (0.720) — action-rate and clamp-duty are each
*directionally* informative but neither is a clean discriminator on its own; arm4d winning on
every axis simultaneously is what makes it the confident pick, not any single number. (iv)
arm4c's replica clearance (0.054) is the batch's best, consistent with its sim stride/clearance
lever (table g, 0.649 vs D2 0.352) actually reaching the bridge.

**Item 3 — the arm5 mechanism, found.** Hypothesis going in: the ankle_roll posture anchor
(range only ±0.2618 rad / ±15°) drives that joint against its limit at baseline standing.
That specific form is **false** — arm5's `stand` phase (first cmd-0 entry, free, no band)
shows `trig_joint_duty = 0.0` in every one of the 6 runs, and never falls there. The refined,
data-supported mechanism, reproduced identically in 5 of 6 runs:

| run | phase where it falls | trig_joint_duty | top joints | pitch at fall | height at fall |
|---|---|---|---|---|---|
| vendor d0 | stop4 | 0.953 | left_ankle_roll 0.953, right_ankle_roll 0.727, left_hip_roll 0.513 | 0.881 | 0.166 |
| vendor d2 | — (PASS) | 0.38 at stop4 | left_ankle_roll 0.333 | — | — |
| vendor d4 | step0.5vx (earlier) | 0.993 | right/left_ankle_roll 0.92/0.917 | −1.289 | 0.110 |
| stress d0 | stop4 | 0.960 | left_ankle_roll 0.960, right_ankle_roll 0.860, left_hip_roll 0.613 | 0.799 | 0.161 |
| stress d2 | stop4 | 0.960 | left_ankle_roll 0.960, right_ankle_roll 0.887, left_hip_roll 0.640 | 0.792 | 0.161 |
| stress d4 | stop4 | 0.960 | (same signature) | 0.792 | 0.161 |

**Read: arm5 is stable at a cold stand, through the vx step/walk phases (trig_joint stays in
the ordinary 0.1-0.5 range there, same as every other candidate), and through the held-vy
walk itself — it fails specifically when SETTLING BACK TO ZERO right after the lateral (vy)
hold**, with BOTH ankle_rolls and the hip_roll saturating simultaneously (duty 0.95-0.96, vs.
0.0-0.5 everywhere else) immediately before a violent roll excursion (pitch flips sign,
height collapses to ~0.16 m — a full topple, consistent with the user's "one leg kicked
violently"). This is the classic signature of losing frontal-plane (roll) balance authority:
once both ankle-roll actuators are simultaneously clamped at the mechanical stop, the LL has
no roll-correction authority left exactly when arresting lateral momentum needs it most. The
one exception (vendor d2, PASS) shows the same phase only reaching 0.38 duty — sub-threshold,
not a different mechanism (and the vendor-d4 case falls even earlier, during the very first
vx step, consistent with the known general 4 ms latency chaos rather than this mechanism).
**This is a refinement of the original hypothesis, not a confirmation of its literal form**:
it is not "arm5 always rides its ankle_roll limit," it is "arm5's ankle_roll anchor leaves no
margin for the specific roll-recovery demand of decaying a lateral velocity command to zero."

**Item 4 — does the replica reproduce the live results, and does it need the fixes to do so?**
**No, not with the original vx-only, non-clamping replica** — a smoke test on arm5 before
today's two fixes and the vy/wz extension showed 0 falls across the full old vx-only chain
(fixstand/takeover/stand/step/stop/walk), i.e. the tool that existed this morning could not
have predicted arm5's live failure at all. **With both fixes and the lateral extension, the
replica now reproduces a severe, mechanistically well-characterized arm5 failure in 5/6
conditions**, matching the live catastrophic fail directionally and in character (violent,
roll-dominated, not a symmetric gait degradation). **arm4d matches exactly** (6/6 replica
pass, live PASS). **One live result still does not reproduce cleanly: arm4c.** Live it was
reported "shaky"/failing alongside D2/arm3/the old keeper; the replica shows arm4c at 5/6
pass — nearly as clean as arm4d, and its best-in-batch clearance number shows up correctly.
**Flagged loudly, per the task's ask**: this is a **replica fidelity gap in the opposite
direction** from arm5 — the replica under-predicts arm4c's live instability. Plausible
reasons (not verified): "shaky" may describe a human operator's qualitative read of a wobbly
but non-falling gait, which the replica's binary fall threshold (height/pitch only) would
never flag; or live GUI command sequencing (rapid manual key changes, no fixed hold
durations) exercises a transient this fixed synthetic phase chain does not. D2/arm3 (also
"shaky" live) land at 2/6 in the replica, i.e. directionally consistent (mostly failing) even
if not identically characterized.

**Deploy-viability verdict (headless evidence only, live GUI is authoritative):** **arm4d
remains the sole confirmed deploy-viable A1 candidate** — 6/6 clean in the extended replica,
matching its live PASS. **arm5 is confirmed not deploy-viable**, and now has a named,
reproducible failure mechanism (bilateral ankle_roll + hip_roll saturation on lateral-command
decay) rather than being merely "sim-invisible." D2/arm2a/arm2b/arm3/arm4b are not
deploy-viable on this evidence (majority-fail in the replica, and D2/arm3 already failed
live). arm4a/arm4c/fix0p8 look good in the replica (5/6 each) but **arm4c's live "shaky" fail
is the standing counter-evidence and should not be waved off by this headless sweep**;
arm4a/fix0p8 have no live data either way. Recommendation: **no new live candidate to try
beyond arm4d** on this evidence — the smoothness lever (arm4d) is validated end-to-end
(sim ranking -> replica -> live), and the arm4c/D2/arm3 discrepancy argues for trusting live
GUI results over the replica whenever they conflict, not the other way around.

**Incidental A0 aside (out of this task's scope, flagged for WL-B0 not chased further):** a
regression smoke test of the extended replica on the A0 path (`--policy a0`, whatever is
currently staged in `velocity/v0/exported/` — still the wrong energy-arm ONNX per unresolved
defect 2 above, not the locked rs20 baseline) also fell at `stop4` (2 ms delay, vendor scene)
in this same sweep. Not investigated — the A0 candidate is locked and out of scope for WL-B1 —
but worth a note that the stop4/lateral-decay transient this sweep newly exercises is not
obviously A1-only, and WL-B0's pending re-export + live G2 re-run should keep an eye on it
once run against the correct candidate.

### A0 FULL G2 BATTERY — PASSED 2026-07-22 (correct re-exported rs20 policy)

Three captures, `logs/deploy_safety/`: **14-14-30** = G2.1-G2.3, **14-32-45** = G2.4 (split),
**14-42-26** = G2.7 (stress scene). `[DEPLOY-GATE]` analyzer output, `any_fall: false` in all
three. The user: "no fall at all and the policy looked pretty good".

**G2.2 held-command PARITY (the gate that matters). Bar: achieved vx within +-0.05 m/s of the
same checkpoint's mjlab value; stride within +-0.05 s.** mjlab rs20 reference achieved =
cmd - ss_err (0.259 / 0.445 / 0.923 at 0.3 / 0.5 / 1.0):

| cmd vx | mjlab | G2.2 bridge | delta | G2.4 split | delta | G2.7 stress | delta |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.259 | 0.269 | **+0.010** | 0.266 | **+0.007** | 0.250 | -0.009 |
| 0.5 | 0.445 | 0.443 | **-0.002** | 0.442 | **-0.003** | 0.417 | -0.028 |
| 1.0 | 0.923 | 0.930 | **+0.007** | 0.917 | **-0.006** | 0.843 | -0.080 (advisory) |

**Parity is excellent** — within 0.010 m/s at every speed on both vendor-scene sessions, an
order of magnitude inside the bar. Stride 0.575-0.585 vs mjlab 0.590 (delta ~0.01) also passes.
Stress-scene degradation is monotonic and expected (harsher plant); G2.7's blocking bar is
0 falls at 0.5, which is met, and its 1.0 row is explicitly advisory.

**Gate verdict — the full A0 blocking set is now GREEN:**

| gate | result |
|---|---|
| G2.1 stand | ✅ 71.6 s (and 86.8 s in G2.4), no fall |
| G2.2 held walk parity | ✅ all 3 speeds + backward -0.5, strafe +-0.5, yaw +-0.5, all >=25 s, 0 falls |
| G2.3 command steps | ✅ incl. **0 -> 1.0 from stand, the historical bridge killer**, repeated, 0 falls |
| G2.4 split mode | ✅ full G2.1-G2.3 repeat with `hold_joint_ids` 12-26, 0 falls, cleanest stand of the three |
| G2.5 / G2.6 | N/A for A0 (no estimator-fed goal channel; fixed 0.6 s clock) |
| G2.7 stress scene | ✅ 0 falls at 0.5 (blocking) and at 1.0; steps clean |

**Two collateral confirmations.** (i) **The gait-clock fix is validated under real operation**:
`phase_alive` is ~0.996-1.000 in every commanded segment and 0.000 at cmd 0 (correct zeroing) —
exactly the signature the instrument was added to prove. (ii) **The Defect-1 / item-1 safety-filter
concern is empirically closed for A0**: `trig_joint_frac` = 0.000 and `alpha_max` = 0.0 in every
commanded segment across all three sessions, i.e. A0's gait never grazes its joint limits while
walking, so the un-ported whole-body chasing hold never arms. (The `alpha_max` 1.0 /
`trig_joint_frac` 0.019-0.117 seen in the FIRST segment of sessions 1 and 3 sits in the
pre-takeover reset window at cmd 0 with `phase_alive` 0 — confirm it is pre-`o` before citing it,
but it is outside every commanded segment.)

**Anomaly to carry forward — `stride_s` is unreliable, it reports exact 2:1 harmonics.** Values
cluster at ~0.577-0.585 but jump to ~1.155-1.162 in scattered segments (session 1: cmd 0.3,
-0.5, vy -0.5, wz -0.5; session 2: vy -0.5 only; session 3: wz +0.5 only). **The same command
gives a different reading in different sessions** (cmd 0.3 -> 1.155 in session 1 but 0.579 in
session 2), so this is touchdown-detection, not gait: the metric divides by counted touchdowns,
so a missed or doubled contact halves or doubles it. Same class as the double-tap finding that
motivated Arm 10. Judge stride on the 0.577-0.585 cluster, and treat any single stride number
from a live capture as suspect until the Arm-10 per-foot touchdown probe lands.

**Unblocked: A0 IS CLEAR FOR HARDWARE.** Phase G2 is complete on the locked candidate, and
**E1/E2 were already resolved 2026-07-20** (see "E1/E2 result" below): **E2 PASS** (specific-force
mean 0.084 over 80 s, nowhere near the -7 fall trigger, so no false-trigger risk and no
sign-convention fix needed), and **E1 FAIL but MOOT for A0** — the failure is structural
(`SportModeState` is only populated while Unitree's own motion service owns the robot, which is
incompatible with running our low-level controller at all), and A0's actor never consumes base
velocity from any source, so it has no velocity dependency to satisfy. E1 only binds the eventual
A1 hardware attempt, which is downstream.

**The only remaining item before the first harnessed A0 hardware session is G3.0 itself** (dry run
with no policy: FixStand under harness, E-stop chain rehearsed, E2 re-checked on the day). Two
open choices to make going in, neither blocking: split-vs-live arms (recommendation stands:
split for session 1) and whether to take the Defect-1 keyboard latch fix first (protocol
robustness for the >=30 s holds, not a safety item).

### FIRST REAL-HARDWARE RUN (2026-07-23, A0 rs20, split arms) — "drunk stumbling", diagnosed

Two attempts, flight recorder `logs/deploy_safety/2026-07-23_10-39-09` (313 s) and
`10-49-28` (186 s), plus `read_all_joints` IMU log
`~/ramlab_ws/trajectories/all_joints_2026-07-23_10-38-55.csv` (819 s). The user: "really like a
drunk robot stumbling around", operated mainly at slow velocities. **No fall flagged; safety
filter essentially silent** (run 2: `trig_joint` 76 rows and `alpha`!=0 149 rows out of 92834
= 0.08%/0.16%; run 1: zero of both).

**Exposure is small — size the conclusions accordingly.** Commanded motion (`|cmd|>=0.1`) was
only **34 s (run 1) and 12 s (run 2)**, i.e. 5-6% of each session; the rest is stand at cmd~0.

**What is NOT wrong (each measured against the 2026-07-22 bridge run that passed G2):**

| metric (during commanded walk) | BRIDGE (worked) | REAL run 1 | REAL run 2 |
|---|---|---|---|
| leg joint tracking err (mean abs) | 0.1152 rad | 0.1266 | 0.1258 |
| action rate | 0.00202 | 0.00171 | 0.00219 |
| \|roll\| mean | 0.0196 | 0.0151 | 0.0202 |
| pitch std (oscillation amplitude) | 0.0162 | 0.0165 | 0.0305 |

So: **not actuator tracking collapse, not twitchiness, not roll instability, and not a
gait-clock failure.** The A1-style smoothness problem does NOT appear on A0 hardware.

**The gait-clock fix is CONFIRMED WORKING on the joystick path** (first real-hardware
evidence): across both runs, `cmd>=0.1 & phase DEAD` = **0.0% / 0.1%**, and
`cmd<0.1 & phase alive` = 0.0% / 0.1%. The clock is alive exactly when commanded and zeroed
exactly when not. `gait_phase_cmd`'s source selection works under `deploy_real.yaml` +
`velocity_commands` as designed.

**WHAT IS WRONG: a persistent PITCH OFFSET that sim never has, present AT STAND.**

| condition | BRIDGE | REAL run 1 | REAL run 2 |
|---|---|---|---|
| pitch, stand (cmd~0) | -0.0003 | **-0.0608** | **-0.0826** |
| pitch, walk | +0.0014 | **-0.0654** | -0.0538 |
| walk pitch range | [-0.074, +0.119] | [-0.122, **-0.017**] | [-0.141, +0.018] |

Reads: (i) the offset is **~0.054-0.083 rad (3.1-4.7 deg), one-signed** — in run 1 the pitch
never crosses zero during the entire walk. (ii) It is **already present while standing**
(stand-to-walk delta is only -0.005 in run 1), so **walking does not cause it; the robot
starts from a leaning posture and walks out of it.** (iii) The oscillation *about* that
offset is normal (std matches the bridge), so this is a DC posture bias, not an instability.
(iv) `projected_gravity` is one of A0's 7 obs terms, so a constant 3-5 deg attitude bias
means the policy runs permanently offset from its trained distribution — a plausible
mechanism for "drunk" without any single dramatic failure.

**Secondary finding: a left/right leg asymmetry on real that the bridge does not show.**
Stand-time `cmd - measured` per joint: real `L_hip_pitch` +0.010 / `R_hip_pitch` -0.062
(run 1) and +0.008 / -0.078 (run 2); `L_ankle_pitch` +0.022 vs `R_ankle_pitch` -0.106 (run 2).
The bridge is far more symmetric (L/R hip_pitch -0.037 / -0.004). Independent of the pitch
offset and worth its own look — note this is A0, so it is NOT the A1 asymmetry the user has been
tracking, but the two may share a hardware-side cause.

**Cause NOT yet established — candidates, cheapest first.** Note the static `read_all_joints`
IMU attitude (mean pitch +0.0119 rad = +0.68 deg over near-static rows) does **not** show a
mounting bias of the required size or sign, so a naive "IMU is tilted" explanation is
already weak. Remaining: (a) **FixStand-vs-policy nominal-pose mismatch** —
`config/config.yaml` FixStand holds `hip_pitch -0.3 / ankle_pitch -0.2` while the policy's
`default_joint_pos` is `hip_pitch -0.2 / ankle_pitch -0.3`, a 0.1 rad disagreement in both,
so the handover starts the policy from a posture it does not consider nominal; (b) real CoM
forward of the model (battery/cabling/hands) beyond the +-5 cm `base_com` DR; (c) real ankle
/ foot compliance under load; (d) an attitude-convention or estimator offset between the
robot's IMU frame and what training assumed.

**Recommended next step (cheap, safe, no walking): a STAND-ONLY diagnostic.** Enter the
policy at cmd 0 under harness and log 60 s, then compare measured pitch against the bridge's
-0.0003. The offset is fully visible at stand, so the entire question can be settled without
the robot taking a single step, and candidate (a) can be tested by simply aligning FixStand's
`qs` with the policy's `default_joint_pos` and re-measuring.

### Head-off run + head-mass hypothesis FALSIFIED (2026-07-23, `11-39-01`)

The user removed the head (it fouled the harness) and judged the result "quite nice". Tempting
story: the head is **unmodelled mass** — in `src/assets/robots/unitree_h1_2/xmls/h1_2.xml` the
head exists only as a collision sphere (`head_collision`, pos `0.05 0 0.7`) with **no inertial**,
and every H1-2 model in the workspace agrees (ours, `/opt/unitree_mujoco`'s
`h1_2_handless.xml`, and `h12_locomotion_rma`'s: all 66.98-67.37 kg, none with head mass;
note the project's own CoT normalization assumes 75 kg, an ~8 kg gap). **That story is WRONG
and the data kills it:**

| run | stand pitch | walk pitch | walk_s |
|---|---|---|---|
| BRIDGE (sim) | -0.0003 | +0.0014 | 495 |
| REAL head ON run 1 | -0.0608 | -0.0654 | 34 |
| REAL head ON run 2 | -0.0826 | -0.0538 | 12 |
| **REAL head OFF** | **-0.0913** ¹ | -0.0535 | **97** |

¹ **Superseded 2026-07-23 (WL-B0b): -0.0717.** The -0.0913 stand mean includes rows where the
whole-body hold was engaged (run 3 is the run with `trig_joint` 4.56% and `alpha` reaching
1.00, almost all of it at stand). Filtering `alpha==0` gives -0.0717, mid-pack rather than the
largest. **The "head-off leans MORE" argument therefore no longer holds**; the head-mass
hypothesis stays falsified on the CoM arithmetic alone (1.4 mm vs the ~55 mm required), and
head-off is still no better than run 1. See "THE BACKWARD LEAN, DECOMPOSED" below.

**Head-off has the LARGEST backward lean of all three real runs.** Removing the head did not
reduce it. Arithmetic agrees: an unmodelled 2 kg head at x=+0.05 m shifts whole-body CoM only
**~1.4 mm**, while a 3.5 deg lean corresponds to ~**55 mm** of CoM travel — off by ~40x. The
head is unmodelled, but it is not the cause of the lean.

**What actually improved: walking duration tripled (97 s vs 34/12 s).** The head was
physically fouling the harness; removing it fixed a *rig interference* problem, not a
dynamics one. This is the correct read of "looked quite nice" — and a reminder that the
head-removal test changed two things at once (mass AND harness geometry), with the pitch data
showing the harness was the operative one.

**The backward lean is therefore INTRINSIC to real-robot + policy, present in every real run
at -0.054 to -0.091 rad and absent in sim (~0).** Direction confirmed by the user: leaning
**backwards**. It also explains his directional report — "forward and turning way more stable
than sideways or backwards, and the robot has to take a stabilisation step backwards": a
sustained backward lean consumes the backward stability margin, so backward/lateral motion
starts already near the limit and provokes backward catch-steps, while forward walking moves
*into* the lean and recovers margin.

**Top remaining candidate (untested, cheap): the FixStand-vs-policy nominal-pose mismatch.**
`config/config.yaml` FixStand holds `hip_pitch -0.3 / ankle_pitch -0.2`; the policy's
`default_joint_pos` is `hip_pitch -0.2 / ankle_pitch -0.3` — 0.1 rad disagreement on both, in
opposite directions, which is exactly a sagittal posture offset. Test: align FixStand `qs`
with `default_joint_pos`, re-enter at cmd 0, re-measure stand pitch against the bridge's
-0.0003. No walking required — the lean is fully visible at stand.

**SAFETY CORRECTION — the A0 whole-body hold DOES arm on real hardware.** The earlier
"empirically closed" verdict (from the 2026-07-22 bridge captures, where `trig_joint_frac`
was 0 in every commanded segment) does **not** hold on the robot. Head-off run:
`trig_joint` fires **4.56% of rows**, `alpha` reaches **1.00** (full whole-body hold), with
`trig_tilt` and `trig_fall` both zero. Crucially it is almost entirely **at stand**
(12675 of 12712 engaged rows at cmd~0; only 37 while commanded). So A0 grazes a joint limit
while standing, and the un-ported chasing hold engages fully. **Item 1 of the A0 track is
RE-OPENED**: identify which joint, confirm whether this is policy-active stand or the
pre-takeover window, and re-evaluate porting the per-joint clamp before G3.3. Note the
`h1_2_limits.h` correction (15/27 joints, 2026-07-21) landed after the bridge runs, so the
bound being grazed may itself be newly-correct.

**Ctrl+C / LAN-detach behaviour (the user, 2026-07-23): "behaves similar to when `p` is
pressed".** If confirmed, the H1-2 firmware damps on LowCmd timeout rather than holding the
last command, which would make the "stiff statue" concern in the E-STOP section too
pessimistic and make Ctrl+C an acceptable fallback. **Not yet deliberately verified** — this
is exactly the G3.0 timeout experiment; run it harnessed with the policy inactive and record
the result, then amend the E-STOP section either way.

### ROOT CAUSE OF THE STUMBLING (2026-07-23): the A0 safety filter fires on ~0.1 deg overshoots

The user's safety warnings from the live sessions (head-off run `11-40-57`, and `10-51` earlier):

```
joint 11 q=0.282  out of [-0.262, 0.262]     right_ankle_roll   overshoot 0.020 rad (1.1 deg)
joint  8 q=0.435  out of [-3.140, 0.430]     right_hip_roll     overshoot 0.005 rad (0.3 deg)
joint 11 q=0.264  out of [-0.262, 0.262]     right_ankle_roll   overshoot 0.002 rad (0.1 deg)
joint 10 q=-0.901 out of [-0.897, 0.524]     right_ankle_pitch  overshoot 0.004 rad (0.2 deg)
joint  5 q=-0.271 out of [-0.262, 0.262]     left_ankle_roll    overshoot 0.009 rad (0.5 deg)
```

**The limits are CORRECT — sim and deploy agree exactly** (`h1_2.xml` ankle_pitch
`-0.897334, 0.523598` / ankle_roll `+-0.261799`; `h1_2_limits.h` `-0.8973, 0.5236` /
`+-0.2618`; right_hip_roll `-3.14, 0.43` both). So this is **not** a limits-table bug and not
a sim-vs-real range mismatch. The difference is **what happens at the limit**:

- **In sim (MuJoCo):** the joint range is a hard constraint the simulator enforces silently.
  The policy may push against a stop; the joint just stays put and walking continues. It is
  free, and the A1a post-mortem already documented that this gait "rides its ankle/hip/knee
  stops by 0.01-0.2 rad every stride".
- **On hardware:** real compliance, momentum and encoder noise let the measured `q` cross the
  bound by a hair — and `State_RLBase.cpp:101-144` responds by ramping `alpha` to 1.0 and
  holding **all 27 joints** at their measured positions, i.e. the chasing hold that degenerates
  to damping-only torque. A **0.002 rad (0.1 deg)** excursion on one ankle takes the whole
  robot into near-passive mid-stride.

**That is the "drunk stumbling".** It matches the measured behaviour: `trig_joint` on 4.56%
of rows with `alpha` reaching 1.00, `trig_tilt` and `trig_fall` both zero. Ankle roll
(`+-0.2618` = only **+-15 deg** of travel) is the joint with the least margin and it dominates
the triggers — the same narrow joint that made WL-D arm 5 (ankle_roll posture anchor) fail
catastrophically on the A1 bridge.

**This is the SAME defect already diagnosed and FIXED for A1 on 2026-07-16, which was
deliberately not ported to A0.** The post-mortem's own words: "the any-joint-out -> hold-all-27
response turns a normal stride into a fall", fixed by clamping the offending joint's COMMAND to
its limit (firmware-like) instead of feeding the hold ramp, scoped "State_RLHRL only, A0
untouched". **A0 has been carrying the un-fixed variant onto real hardware.**

My 2026-07-22 verdict that this item was "empirically closed for A0" was **wrong** — it rested
on bridge captures where A0 happened not to graze its stops (`trig_joint_frac` 0 in every
commanded segment). Real hardware grazes constantly. Bridge non-observation was not evidence
of absence.

**ACTION (highest priority, blocks G3.3 and probably explains most of the stumbling): port the
per-joint command clamp from `State_RLHRL` to `State_RLBase`.** The code already exists and is
proven; it is a scoped copy, not a new design. Spec-first per the deploy rules, then re-run a
short bridge check and repeat the hardware walk.

**Separately, the FixStand-pose hypothesis is WITHDRAWN.** The user's question — can the FixStand
hold pose influence walking? — is correct: it cannot. After `o`, FixStand's `qs` command
nothing; the policy's targets are `default_joint_pos + action*scale`. The 0.1 rad
FixStand-vs-nominal disagreement can only produce a handover transient, never a sustained lean.
It may still be worth aligning for a cleaner takeover, but it is not the lean's cause.
**The backward lean remains unexplained and is independent of the filter triggers** (run 1
shows the lean with ZERO filter engagement), so it is a second, separate finding — not
necessarily the thing that made the robot stumble.

### THE BACKWARD LEAN, DECOMPOSED (2026-07-23, WL-B0b): two causes, not one

Investigation of the ~3-5 deg backward stand lean, using only data already on disk (the
three real flight recorders, the 2026-07-22 bridge baseline, and the concurrent
`read_all_joints` log `all_joints_2026-07-23_10-38-55.csv`, which turns out to overlap runs
1 and 2 in time and carries `tau_est`). Analysis scripts are throwaway; every number below
is reproducible from those four files plus `h1_2.xml`.

**Method: three INDEPENDENT estimates of the pelvis pitch.**

1. `p_IMU`: from the logged quaternion. This is literally the policy's own observation
   (`safety_logger.h:110` writes the same `root_quat_w` that `unitree_articulation.h:32`
   turns into `projected_gravity_b`).
2. `p_kin`: from the joint encoders alone, by solving for the base orientation that puts
   both foot soles flat on a level floor (MuJoCo FK on `h1_2.xml`). Uses no IMU.
3. `p_tau`: from statics. The ankle-pitch torque sum fixes the CoP, hence the CoM, hence
   the pitch: `tau_ank_sum = M*g*(x_com - x_ankle)`. This is statically determinate (no
   double-support load-split ambiguity) and uses neither the IMU nor the feet-flat
   assumption, only the encoders, the model's mass distribution, and the measured torque.

All three agree to 1 mrad on the bridge, which validates the machinery:

| run (quiet stand: \|cmd\|<0.05, max\|dq\|<0.1, alpha==0) | `p_IMU` | `p_kin` | `p_tau` | resid = IMU-kin |
|---|---|---|---|---|
| BRIDGE 2026-07-22 (n=8344) | +0.0002 | -0.0007 | -0.0001 | **+0.0009** |
| REAL run 1 `10-39-09` (n=264806) | -0.0549 | -0.0258 | -0.0455 | **-0.0291** |
| REAL run 2 `10-49-28` (n=60772) | -0.0767 | -0.0359 | -0.0686 | **-0.0408** |
| REAL run 3 `11-39-01` head-off (n=177700) | -0.0717 | -0.0405 | -0.0800 | **-0.0312** |
| REAL run 4 `16-50-23` POST-FIX (n=101697) | -0.0640 | -0.0359 | -0.0694 | **-0.0282** |

**Run 4 is the WL-B0a post-clamp-fix session** (225.8 s of free walking, `alpha` max 0.000,
the filter never engaging once), added after the fact. It reproduces every number: same
residual, same knee excess, same load. Two things follow. First, **the lean is confirmed
independent of the safety filter** on a run where the filter is provably inert, which was
predicted here and is no longer an inference from run 1 alone. Second, it is a **separate
session hours later with a re-rigged harness**, which matters for the open question below.

(Run 3's earlier numbers were noisy because its stand rows include the whole-body hold;
filtering `alpha==0` makes it consistent with the other two. This also revises the
head-off stand pitch from -0.0913 to **-0.0717**: it is no longer the largest lean, so the
"head-off leans MORE" argument that falsified the head-mass story is weaker than reported,
though the CoM arithmetic that falsified it, 1.4 mm vs the 55 mm required, still stands
and head-off remains no better than run 1.)

**The lean splits, almost exactly in half, into two independent gaps.**

```
reported lean  =  encoder->attitude map error  +  genuine leg-configuration lean
   -0.0549     =         -0.0291               +        -0.0258        (run 1)
   -0.0767     =         -0.0408               +        -0.0359        (run 2)
   -0.0717     =         -0.0312               +        -0.0405        (run 3)
   -0.0640     =         -0.0282               +        -0.0359        (run 4, post-fix)
```

#### Component A: a constant -0.031 rad (1.8 deg) encoder-to-attitude map error

The IMU and the encoder-based kinematics disagree by a **constant** on hardware and agree
in sim. Constant across:

- **Posture**, over a 0.43 rad (25 deg) span of actual pitch, sampled across the whole
  819 s session including the pre-policy loaded phases: regression
  `p_IMU = 1.085 * p_kin - 0.0339`, r = 0.996, n = 3894.
- **Heading**, over 207 deg of yaw. Residual by yaw bin: -0.0303 (yaw +1.0), -0.0439
  (+1.5), -0.0337 (+2.0), -0.0323 (+2.5), and run 3 at yaw -1.28, which is 187 deg from
  run 1's +1.98, gives -0.0312.
- **Phase**: -0.0310 pre-policy (loaded FixStand-like), -0.0339 policy run 1, -0.0392
  policy run 2. Not a policy artifact.

**Which side is wrong is decided by `p_tau`, which shares no assumption with either.**
It sides with the IMU: |`p_tau` - `p_IMU`| = 0.009 / 0.008 / 0.008 across the three runs,
versus |`p_tau` - `p_kin`| = 0.020 / 0.033 / 0.040. So the IMU is telling the truth and
**the model's mapping from joint encoders to body attitude is what is off by 1.8 deg.**

Physical candidates for that 1.8 deg, not separated by this data:
- **Foot-sole geometry.** The soles in `h1_2.xml:84-90` are hand-authored capsules at a
  constant `z=-0.035`, i.e. a plane exactly parallel to the `ankle_roll_link` frame. A 1.8
  deg real sole pitch, or an 8 mm heel-vs-toe pad difference over the 250 mm foot, produces
  exactly this.
- **Joint encoder zeros**: 0.6 deg each across hip_pitch + knee + ankle_pitch.
- **Structural compliance downstream of the encoder**: the residual does correlate with
  load (r = +0.73 vs knee torque, +0.81 vs ankle torque, slope ~5e-4 rad/Nm, i.e. an
  effective ~1000 Nm/rad per ankle). Partly confounded, since load and posture co-vary,
  but it accounts for the 8.5% regression slope excess and the +-0.006 run-to-run spread.

#### Component B: excess knee flexion, from ~22% more standing load than the model has

The leg-configuration lean is **entirely the knee**. Achieved joint angle minus nominal, at
quiet stand:

| | knee L | knee R | hip_pitch L | hip_pitch R | ankle_pitch L | ankle_pitch R |
|---|---|---|---|---|---|---|
| BRIDGE | -0.004 | +0.006 | -0.007 | -0.029 | +0.011 | +0.024 |
| REAL run 1 | **+0.035** | **+0.032** | -0.009 | -0.007 | +0.008 | -0.007 |
| REAL run 2 | **+0.040** | **+0.027** | -0.004 | +0.001 | +0.011 | -0.002 |
| REAL run 3 | **+0.041** | **+0.029** | -0.003 | +0.001 | +0.004 | +0.009 |

Hips and ankles sit within 0.011 rad of nominal on hardware. The knee sits 0.027 to 0.041
rad more flexed, and since `p_kin = -(hip_pitch + knee + ankle_pitch)` with feet flat, that
knee excess **is** the leg-configuration lean.

The knee is more flexed because it droops more under load, and it droops more because the
legs carry more. Knee droop (`cmd - meas`, so PD tracking lag) is 0.089/0.061 rad L/R in
the bridge and 0.098-0.123 / 0.157-0.172 on hardware. Total knee torque is 45.2 Nm in the
bridge and 76.4 / 82.2 / 85.4 Nm on hardware, 1.7-1.9x.

**The PD law itself is honest**, so those torques are real: the robot's own `tau_est` in
the concurrent trajectory log matches `kp*(cmd-meas)` (run 1 quiet window: right knee -48.2
vs -47.0 Nm, left ankle +6.8 vs +5.1, right ankle +1.2 vs +3.6). The actuators are not
under-delivering.

**Measured standing load: ~800 N, versus the model's 657 N.** Per leg, the three sagittal
torques (hip_pitch, knee, ankle_pitch) give three moment balances in the three unknowns
(f_z, f_x, x_cop), which is exactly determined. Solving:

| run | f_z L | f_z R | SUM f_z | implied kg | SUM f_x (should be ~0) |
|---|---|---|---|---|---|
| BRIDGE | 324.7 | 317.6 | **642.4 N** | 65.5 | -0.3 |
| REAL run 1 | 359.1 | 449.2 | **808.3 N** | 82.4 | -10.0 |
| REAL run 2 | 364.2 | 436.4 | **800.7 N** | 81.6 | -13.3 |
| REAL run 3 | 376.6 | 417.0 | **793.6 N** | 80.9 | -2.9 |
| REAL run 4 POST-FIX | 392.9 | 400.8 | **793.7 N** | 80.9 | -0.9 |

The bridge validates the method to -2.2% and `SUM f_x ~ 0` validates it on every run. The
four real runs agree with each other to **0.7 kg std**, across two sessions six hours apart
with the head removed in between and the harness re-rigged. **The legs carry ~140 N (21%)
more than the 66.98 kg model weighs.** Arithmetic: +22% load on a kp=300 knee adds 0.089*0.22 = 0.020
rad of droop, which covers roughly half to three quarters of the 0.027-0.041 rad knee
excess. The rest is the L/R load split (f_z R / f_z L = 1.11-1.25) which the model does not
have.

**Zero-load reference measured (2026-07-23, `all_joints_2026-07-23_16-33-35_passive_hanging.csv`,
34.8 s, Passive, robot suspended clear of the ground).** This was step 2 of the closing
procedure below; it is now **DONE** and it closes a hole in the argument above, because the
whole load result is inferred from torques:

- **`tau_est` has no meaningful zero offset.** Across all 12 leg joints, mean `|tau_est|` =
  **0.240 Nm**, max 1.97 Nm, per-joint p95 <= 0.94 Nm (arms: 0.114 Nm). The GRF solve infers
  ~800 N from torques of order 30-50 Nm through a ~0.105 m lever, so a sub-1 Nm bias is worth
  at most ~10 N per joint. **The 21% load excess is therefore not a torque-sensor artifact.**
- The legs hang straight (every leg joint within 0.04 rad of zero) at zero torque, which
  confirms the robot really was suspended rather than partly standing.
- The IMU-vs-accelerometer check passes again at zero load: -0.0153 vs -0.0166 rad.
- **Free-hang attitude is NOT a usable level reference**, so do not treat it as one: this run
  hangs at -0.0153 rad while the passive segments of the morning session hang at +0.085 to
  +0.117 rad. Free-hang pitch is set by harness rigging and CoM, not by the pelvis frame. It
  does show that the IMU is not parked at a large constant negative reading in all conditions:
  the standing lean is a real ~3 deg posture change relative to free hang.

**Whether that 140 N is unmodelled robot mass or harness down-force is still NOT resolved,
but the balance has shifted toward mass.** Four runs across two sessions six hours apart, with
the head removed in between, the harness re-rigged, and a changed safety-filter code path,
return 82.4 / 81.6 / 80.9 / 80.9 kg: a 0.7 kg spread. A harness down-force would have to
reproduce to under a kilogram across two independent riggings, which is implausible. The
remaining honest caveat is that the harness demonstrably
couples large and varying vertical force *within* a session: the same solver applied to
short `tau_est` windows returns 4.8 kg (t 760-800, robot visibly hoisted, so the estimator
correctly detects harness support), 60.1 kg, 69.0 kg, and 76.8-83.3 kg. Long-run averages
wash that out, which is why the four full-run numbers agree so tightly, but it means the
harness is never fully absent. **A scale settles it in one minute and is the one thing that
would make this conclusive**; the reading to expect if it is mass is ~81 kg, versus 66.98 kg
in `h1_2.xml`.

#### Hypothesis table

| # | hypothesis | mechanism | discriminating test | measured | survives? |
|---|---|---|---|---|---|
| 1 | **IMU frame / mounting pitch offset** | IMU pitched at its mount, biasing `projected_gravity`; policy holds a compensating lean | (a) reported quat pitch vs raw accelerometer `-atan2(acc_x, acc_z)`; (b) `p_tau`, which uses neither IMU nor feet-flat | (a) agree to <=0.0016 rad on all 3 runs (-0.0601/-0.0593, -0.0804/-0.0791, -0.0904/-0.0888); (b) `p_tau` is 0.008-0.009 from `p_IMU` but 0.020-0.040 from `p_kin` | **NO.** The IMU is corroborated by statics. E2 could not have caught a 1.9 deg pitch mount error, correct, but there isn't one. Also self-refuting a priori: the logged pitch IS the policy's observation, so a mount bias would show as the reported pitch returning to ~0, not sitting at -0.06 |
| 1b | **encoder->attitude map error** (spun out of 1) | model's FK from encoders to attitude is off by a constant | constancy vs posture, heading, phase; and which side `p_tau` picks | **-0.031 +- 0.006 rad**, constant over 25 deg of posture (r=0.996) and 207 deg of heading; +0.0009 in the bridge | **YES.** Component A, ~50% of the lean |
| 2 | **mass / CoM mismatch beyond the head** | real CoM forward/heavier than model | (a) the "8 kg gap"; (b) symmetric-mass test via L/R droop; (c) direct GRF solve | (a) `mass_g = 75.0*9.81` at `rewards.py:708` is the **CoT normalizer, not a mass claim**: red herring; (b) droop ratio real/sim is 0.9-1.3x left but 2.5-3.2x right, so no symmetric mass change fits; (c) legs carry 800 N vs 657 N | **PARTLY.** A CoM *shift* is rejected (would be symmetric). A ~22% load excess is measured and is the driver of Component B, but its origin (robot mass vs harness) is open |
| 3 | **ankle / structural compliance under load** | deflection downstream of the encoder | residual vs joint load | r = +0.73 (knee tau), +0.81 (ankle tau), ~5e-4 rad/Nm | **PARTLY.** Contributes to Component A's spread and its 8.5% slope excess, but the bulk of Component A is load-independent (phase means -0.031/-0.034/-0.039 at very different loads) |
| 4 | **attitude-estimator convention** | quaternion order or Euler convention mismatch | reported quat pitch vs raw accelerometer; index order at `unitree_articulation.h:29-34` | agree to <=0.0016 rad; order is (w,x,y,z) both sides; bridge residual +0.001 | **NO. Falsified** |
| 5 | **lab floor slope** (added) | feet flat on a sloped floor tilts the pelvis invisibly to level-floor FK | residual must vary as `-s*cos(yaw - yaw0)` | residual -0.030/-0.044/-0.034/-0.032 over 207 deg of yaw, including near-opposite headings | **NO. Falsified** |
| 6 | **pelvis-vs-torso frame** (added) | training uses `root_link_quat_w` (pelvis free joint, mjlab `entity/data.py:584`); deploy uses the vendor `imu_state` quaternion | model's only pelvis-to-torso joint is `torso_joint`, `axis="0 0 1"` | yaw-only, so pitch is shared exactly in the model | **NO mechanism**, but it identifies that the pelvis-frame-to-IMU pitch relation has never been calibrated on hardware, which is exactly Component A |
| 7 | **harness applying a pitch moment / down-force** (added) | strap pulling back at the torso; 50 N at 1.2 m = 60 Nm = 91 mm of equivalent CoM shift, more than the 33 mm needed | a load-bearing harness must UNLOAD the legs; and a rigging-dependent force cannot reproduce across re-riggings | legs carry 1.7-1.9x more knee torque, not less; head-off run demonstrably changed harness interaction (walk time 34/12 s to 97 s) yet its lean is mid-pack; **the 4 full-run loads agree to 0.7 kg across two sessions and two riggings** | **WEAK.** Downgraded 2026-07-23 by run 4. The harness does couple large vertical force within a session (5 kg to 83 kg in short windows), but it cannot explain a 0.7 kg-reproducible 140 N offset |
| 8 | **excess knee droop** (added) | achieved posture = command - droop; `p_kin = -(hip+knee+ankle)` | achieved joint angle vs nominal, per joint | knee **+0.027 to +0.041** rad on hardware vs ~0 in sim; hips/ankles within 0.011 | **YES.** Component B, ~50% of the lean |

Also worth recording: **the policy is actively fighting the lean and losing.** The commanded
leg-configuration pitch, `-(sum of commanded hip_pitch + knee + ankle_pitch)/2`, is -0.188
in the bridge but **+0.021 / +0.153 / +0.229** on the three real runs. The policy commands a
0.2 to 0.4 rad more forward-pitching posture on hardware and still lands 3-4 deg back. So
this is a saturated feedback loop, not an unobserved disturbance.

And the practical consequence, quantified: the CoP sits at +33 mm ahead of the ankle in the
bridge and at **+14/-7/-14 mm** on hardware. The foot spans -90 to +180 mm about the ankle,
so the backward CoP margin falls from 123 mm to 76-90 mm, a **27-38% loss**. That is the
direct account of the user's report that backward and lateral motion are worse and that the
robot takes backward catch-steps.

#### Ranked verdict

**No single cause. The lean is a near-even sum of two, and a fix that addresses one will
halve it, not remove it.**

1. **Component A, the -0.031 rad encoder-to-attitude map error.** Best-supported single
   item: measured constant, sim-absent, right sign, 40-57% of the lean, and the only one
   with a one-line deploy mitigation. Its physical origin (sole geometry vs encoder zeros vs
   compliance) is not yet separated.
   *Falsified by*: a digital inclinometer on the pelvis reading ~1.8 deg forward of
   `imu_rpy_p`, i.e. agreeing with `p_kin`. That would mean the IMU is biased after all and
   would invert the ranking.
2. **Component B, excess knee droop under a ~21% higher standing load.** Now measured on four
   runs across two sessions to a 0.7 kg spread, with the torque sensor verified unbiased at
   zero load, so the 140 N itself is solid. What is still open is only its *origin*: most
   likely unmodelled robot mass (~81 kg real vs 66.98 kg in `h1_2.xml`), with harness
   down-force downgraded to weak.
   *Falsified by*: a scale showing the standing robot weighs ~67 kg, which would leave the
   800 N unexplained and send the knee droop looking for another cause.

Explicitly NOT the cause, do not re-open without new evidence: the attitude estimator's
convention, a lab floor slope, a pelvis-vs-torso frame mismatch, a symmetric CoM shift, and
(still) the head mass.

#### Hardware measurement to close this, one short session (~15 min, policy NOT running)

Needs a digital inclinometer (0.1 deg) and, if available, a crane or platform scale.

1. **Floor.** Inclinometer on the floor where the robot stands, along the robot's forward
   axis and at 90 deg. Expect <0.3 deg. Closes the slope question physically.
2. ~~**Zero-load reference.**~~ **DONE 2026-07-23** (`all_joints_2026-07-23_16-33-35_passive_hanging.csv`):
   `tau_est` zero offset is under 1 Nm on every joint, so the load result is not a sensor
   artifact. Free-hang attitude turned out **not** to be a usable level reference. Skip this
   step; see the zero-load paragraph above.
3. **The key measurement.** FixStand (`h`), feet down, and **verify the strap is visibly
   slack** (this matters: the GRF solve shows the harness carries anywhere from 5 kg to 83
   kg). Record 60 s of `read_all_joints`. Then, without moving the robot, read the
   inclinometer on: (a) a machined horizontal face of the **pelvis** (this is the frame
   training uses, `root_link_quat_w`), (b) the **torso** top face, (c) the **sole or top
   machined face of one foot**. Write all three down next to the concurrent `imu_rpy_p`.
4. **Weight.** Weigh the robot on the crane/platform scale, or read the gantry load cell
   with the harness taking full load.

Decision rules, decided before the session:

- **Pelvis reading within 0.5 deg of `imu_rpy_p`** -> IMU honest, error is model-side
  (foot geometry or encoder zeros). Immediate mitigation to test: a **-0.031 rad
  `ankle_pitch` offset** in the deploy `offset` vector, which rotates the body forward by
  exactly that amount relative to the foot. Proper fix is the sole geometry in `h1_2.xml`,
  which is a retrain.
- **Pelvis reading ~1.8 deg forward of `imu_rpy_p`** -> IMU mount offset after all. Fix is a
  fixed +0.031 rad pitch rotation applied to the IMU quaternion in
  `unitree_articulation.h`, before `projected_gravity_b` is formed.
- **Foot sole not level while the feet are flat down** -> sole/ankle geometry is the specific
  culprit, and the number read is the correction.
- **Scale reads ~80 kg** -> confirms the load finding, Component B needs a model mass fix,
  which is a v2-style rebase and therefore the user's call.

#### Does this affect A1?

**Yes, both components, identically, and fixing it helps both architectures.**

- A1's LL consumes the same `projected_gravity` (`velocity_hrl/v0/params/deploy_real.yaml`,
  `observations.policy.projected_gravity`) and the same `default_joint_pos` / action `offset`
  vectors as A0, on the same hardware. Component A is a plant/observation property, not a
  policy property, so it applies unchanged.
- Component B is pure plant. A1's LL runs the same kp=300 knee against the same load.
- Under `hl_velocity_goals_only=True` the A1 HL pins orientation and height targets to
  nominal, so a 1.8 deg attitude error is a constant offset on a pinned nominal that neither
  level can see or correct.
- **For RQ2 comparison cleanliness this is shared, not a confound**: both A0 and A1 inherit
  the same bias, so an A0-vs-A1 hardware comparison is not invalidated by it. But it depresses
  both, and it eats the backward stability margin that A1 needs more than A0 does, given A1's
  known smoothness gap (action_rate 0.92-1.28 vs A0 0.57-0.75). Worth closing before the A1
  hardware track, not after.

**Relation to the stumbling.** Independent. Run 1 shows the full lean with zero filter
engagement, and the safety-filter defect is a separate finding with its own fix (WL-B0a,
ported 2026-07-23). Fixing the lean will not fix the stumbling and vice versa, but both
narrow the same margin.

#### CORRECTION 2026-07-24 (real mass + torque): Component B is unmodelled LEG mass, no load mystery

Two facts from the user closed Component B and corrected an overstatement above.

**Real robot mass = 73.70 kg; one leg (incl. all 3 hip motors) = 18.60 kg.** The model
(`h1_2.xml`) is 66.98 kg with a 15.40 kg leg. **The entire 6.7 kg gap is in the legs**
(+3.20 kg/leg); the non-leg remainder (pelvis + torso + arms) is off by only +0.31 kg. So the
mass error is neither uniform body scaling nor a torso-CoM shift, it is specifically the legs
being ~21% heavier than modelled, with everything above the hips already correct. Raising the
6 leg-link masses per side by x1.21 brings the model to 73.4 kg.

**The "~21% / ~140 N load excess of uncertain origin (possibly harness)" claim above is
WITHDRAWN. It was my error, from two stacked mistakes:**
- the GRF solve used the model's 15.40 kg legs, not the real 18.60 kg, and
- it used `kp*(cmd-meas)` for torque, which **over-reads the real knee torque by ~1.3x**: the
  robot's own `tau_est` is 0.71-0.73x my number at the knees (matched window, run 1:
  `kp*err` knee pair -76.7 Nm vs `tau_est` -55.5 Nm).

Redone with the real leg mass **and** `tau_est`, the cleanest static stand window (head-off
`11-38-59.csv`, t0-30, `|SUM f_x|` = 1.3 N, pitch -0.030) reads **74.2 kg**, i.e. the real
weight. Other clean windows scatter 75-85 kg with a systematic `SUM f_x` ~ -7 N that flags a
residual non-static/contact-point offset, so the tightest-`f_x` window is the trustworthy one.
**There is no unexplained standing load and no harness-down-force term.** Component B is just
the 6.7 kg heavier legs drooping more on the kp=300 knee. Hypothesis-table rows 2 and 7 should
be read through this: the excess load is real robot mass, not harness, and the fix is a leg-mass
correction + retrain, not a scale session.

**This also resolves the harness/FixStand question.** The user: FixStand with the strap slack just
topples, like sim with the band off. Correct, and expected: FixStand is an open-loop joint hold,
not a balancer. The balancing controller is the **policy** (Velocity `o` at cmd 0), which is what
every stand window here already uses, and under it the measured load equals full body weight, so
the strap is effectively slack under the policy. Consequence for the procedure below: **the
"FixStand + slack strap, 60 s" step is void**; Component A's inclinometer read must be taken
under the policy at cmd 0 (spotted), not in FixStand. No scale is needed, the mass is known.

**Component A is unaffected and its verdict is now stronger.** Re-running `p_tau` with the
robot's own `tau_est` and the real leg mass still sides with the IMU (|p_tau - p_IMU| =
0.003-0.006) against the encoder kinematics (|p_tau - p_kin| = 0.027-0.036) on every window.
The IMU is honest; the -0.031 rad error is in the model's encoder->attitude map (foot-sole
geometry or encoder zeros). **On IMU recalibration**: do NOT zero the IMU at stand, the robot
really is pitched back and a zeroed IMU would feed the policy a lie and deepen the lean. The
only correct IMU fix, if a mount offset is ever found by inclinometer (none is in evidence: at
passive hang IMU pitch -0.0153 equals accelerometer-implied -0.0166 to 0.0013 rad), is a fixed
quaternion pre-rotation in `unitree_articulation.h:29-34` before `projected_gravity_b`. A
deliberate constant pitch bias in `projected_gravity` is available as a posture-trim knob but
fights the symptom.

#### MODEL FIX (STAGED, held for the new version): leg masses 15.40 -> 18.60 kg/leg; foot-geometry lead; head, friction, DR

**STATUS 2026-07-24: staged, NOT applied. See `docs/adr/0006-leg-mass-foot-geometry-nominal-correction.md`
(Model v3, PROPOSED).** The edit was made, verified, then reverted at the user's call. It is HELD until after the inclinometer session and bundled into the next model
version together with the foot-geometry fix. Rationale: the ongoing WL-D gait-shape refinement
runs must stay comparable to the current `arm4d` batch (all on 15.4 kg legs), so the training
plant must not be rebased mid-batch. No hardware-bound training runs happen before the new
version, so applying the mass (and foot) change to that version carries no sim-to-real transfer
issue (the user, 2026-07-24). Exact recipe to re-apply is below.

**Two plants, both to set to 73.4 kg when the new version is cut (they serve different roles):**
- **Training model** `src/assets/robots/unitree_h1_2/xmls/h1_2.xml` (loaded via `H1_2_XML`)
  and its unused twin `scene_h1_2.xml`: this is where the mass must change for the *policy to
  learn* the heavier legs. This is the primary one.
- **Bridge test plant** `/opt/unitree_mujoco/unitree_robots/h1_2/h1_2_handless.xml`: the MuJoCo
  model the C++ controller runs against in the G2 sim-to-sim gates, the stand-in for hardware.
  For a faithful bridge it should get the real mass too, else the bridge tests a lighter robot
  than both training and reality. (It is an external clone, not version-controlled with this
  repo; note the edit there separately.) The deployed C++ and gain vectors carry no masses and
  need no change either way.

**Re-apply recipe** (`$CLAUDE_JOB_DIR/tmp/fix_legmass.py` logic): scale the 6 leg links/side by
**1.20787** (= 18.6/15.399) on `mass` and all 3 `diaginertia` values, leaving `ipos`/`quat`
unchanged so no CoM shift is introduced. Per-link mass: hip_yaw 2.829->3.417, hip_pitch
2.920->3.527, hip_roll 4.962->5.993, knee 3.839->4.637, ankle_pitch 0.102->0.123, ankle_roll
0.747->0.902. Verified total 73.386 kg, one leg 18.600, torso/pelvis/arms untouched, compiles.

**Head is negligible for the lean, and the user's 73.70 kg is head-OFF.** The head sits at
x=+0.05 m in the torso, so removing it moves whole-body CoM *backward* (same direction as the
lean), by 1.3 mm (2 kg head) to 4.9 mm (8 kg), against the ~55 mm a 3.5 deg lean needs. So
head-off cannot fix the lean and if anything deepens it by <10%, which matches the data
(head-off runs 3/4 lean the same as head-on 1/2). Bookkeeping: the real head-*on* robot is
73.70 + head, consistent with runs 1/2 reading a couple kg heavier than 3/4.

**The gap: real one leg incl. all 3 hip motors = 18.60 kg; the model is 15.399 kg** (new total
would be 73.386 kg vs real 73.70; the residual 0.31 kg is the non-leg remainder, within
measurement).

- **This is a manufacturer-spec-vs-reality gap, not a repo error, and it is universal.** The
  15.399 kg leg is byte-identical across all five sims (ours, `/opt` bridge, isaac-gym,
  RMA-27dof, RMA-book) AND the official Unitree URDFs (`/opt/unitree_mujoco`, the mybotshop
  `h1_description`, RMA), AND both isaac repos (`h1v2-Isaac`, which is the same URDF). So the
  HF `unitreerobotics/unitree_model` H1-2 will read 15.399 too. Every downstream H1-2 sim
  inherits the official CAD value, which underweights the real leg by 3.2 kg (cabling,
  connectors, as-built vs nominal component mass).
- **Not an H1-vs-H1-2 swap** (the user's hypothesis, tested): H1-v1 legs are 10.9 kg with a
  different per-link split (2.24/4.15/2.23/1.72/0.55), lighter, not heavier. The 15.4 is
  genuinely H1-2.
- **Assumption to flag**: uniform scaling assumes the missing 3.2 kg is distributed like the
  CAD leg. If the real excess is concentrated (e.g. a heavier hip actuator or added
  cabling near the pelvis), the per-link split is wrong even though the total and CoM-height
  are right. Also, "one leg with all 3 hip motors" is taken to be the 6 URDF leg links; if
  The user's physical cut point differs, the leg/non-leg split shifts (the whole-robot 73.70 is
  independent and agrees, so the total is solid regardless).
- **REBASE CONSEQUENCE (why it is held): this changes the training plant, so it invalidates
  existing checkpoints for clean comparison.** The deployed A0 (`rs20_baseline`) and A1
  (`arm4d`) and the whole WL-D batch were trained on 15.4 kg legs; applying it mid-batch would
  confound the gait-refinement comparisons. Treat the new-version cut like the Model-v2/ADR-0005
  rebase: version tag + ADR entry, bundled with the foot-geometry fix, after the inclinometer.

**Foot geometry is the CoP / push-off lead, separate from mass.** Ours is the **only** model
using hand-authored flat foot capsules (7 per foot, all at z=-0.035, a plane exactly parallel
to the ankle frame, 0.00 deg by construction); every reference (opt bridge, isaac-gym, both
RMA, h1v2-Isaac) uses the actual **foot mesh** for collision. Two consequences:
- For **Component A** (the 1.8 deg encoder->attitude error): a real sole inclined ~2-3 deg in
  the ankle frame would produce exactly the observed bias, and our flat capsule cannot. The
  inclinometer read (foot sole vs pelvis, under the policy) settles it; if the sole is
  inclined, the fix is to switch foot collision to the mesh, or tilt the capsule plane.
  (A direct mesh-vertex fit was inconclusive here: the `ankle_roll` visual mesh frame does not
  cleanly expose the sole plane, so this needs the physical read, not more desk analysis.)
- For **CoP transfer during push-off** (WL-D arm 6): CoP rolls heel->toe along the sole, so its
  trajectory and the push-off moment arm are set by sole shape, which our idealized flat plane
  gets wrong. **The leg-mass fix barely touches this**: the added mass is proximal (hip_roll
  +1.03, knee +0.80 kg) while the foot/ankle gained only +0.18 kg total, so push-off CoP is a
  foot-geometry question, not a mass question. If push-off CoP fidelity matters for the thesis
  claim, switching to mesh foot collision is the relevant change, tracked here for WL-D.

**Friction / armature are not the static lean.** At a static stand the hold torque equals the
gravity torque; frictionloss and armature are velocity/acceleration terms and vanish at rest,
so neither adds to the ~74 kg standing load (that is pure mass, now corrected). Friction only
creates a PD deadband of +-7 to +-20 mrad (2-6 Nm Coulomb at kp=300), against a measured knee
droop of 100-170 mrad that `droop = tau/kp` reproduces from gravity alone. Where friction and
armature (and the now-heavier real legs) *do* bite is walking/swing dynamics, a separate
sim2real gap plausibly feeding the stumbling, not the DC lean.

**On DR (the user's question): only *biased* DR helps; zero-mean IMU/encoder noise does not.** Both
components are biases, and a policy averages zero-mean sensor noise out and still centers on the
wrong mean. What each needs: Component B wants a **mass-magnitude** channel (per-leg mass or a
base payload band; `base_com` DR is +-5 cm of *position* only and never covered this) - though
correcting the nominal, as just done, is the primary fix and DR only adds robustness around the
corrected center. Component A wants a **biased `projected_gravity` pitch offset** (or randomized
foot-sole angle) so the policy cannot equate "gravity=0" with "upright" and must use encoders +
contact to stand truly level. Both slot into the A2 RMA e_t (currently motor-strength + damping)
as extra channels. Ordering: fix the two known-wrong nominals (leg mass done; foot geometry
pending the inclinometer), then widen DR with biased channels, not zero-mean noise.

### A0 CLAMP PORT (WL-B0a, 2026-07-23/24): landed, and confirmed on the robot

**What changed** (`deploy/robots/h1_2/src/State_RLBase.cpp`, robot-local only — the shared
`deploy/include/FSM/State_RLBase.h` and `deploy/include/isaaclab/` were not touched):

1. The measured-position loop that set `joint_hold` is deleted. Joint violations no longer
   feed the hold ramp; `alpha` is now a pure tilt/fall indicator for A0.
2. The write loop clamps each commanded joint to its `h1_2_limits.h` bound before it
   reaches the motor, exactly as `State_RLHRL` has since 2026-07-16. Clamp-before-
   `hold_ids`-override ordering matches HRL, so the two safety blocks are now equivalent.
3. `trig_joint` in the flight log changes meaning to **"a command was clamped this tick,
   no hold effect"** (the A1 semantics since 2026-07-16). Recorded in `safety_logger.h`;
   an A0 CSV dated before 2026-07-23 carries the OLD meaning and the two eras must not be
   pooled.
4. Rate-limited console warning (the user's call — silence would hide a genuinely pinned
   joint): one line per 1000 ticks naming the worst offender. **`control_dt` is 0.001, so
   the loop is 1 kHz, not the 500 Hz previously assumed** (measured from the logger meta).

**Not changed:** tilt trigger, fall trigger, `H1_2_RAMP_CYCLES`, the
`q_cmd = (1-α)·action + α·q_meas` chasing-hold form, the limits table, the split-deploy
`hold_joint_ids` path, `State_RLHRL`. The G3.0 chasing-vs-latched terminal-behavior
question is still open and untouched.

**FixStand alignment** (same session, separate item): `config/config.yaml` FixStand `qs`
legs went `[0,−0.3,0,0.5,−0.2,0]` → `[0,−0.2,0,0.5,−0.3,0]`, now byte-equal to the policy
`default_joint_pos`; mirrored by hand in `bridge_replica.py` `FIX_Q`. Both distributions
close to zero net pitch, so this only cleans the takeover transient. **It is NOT a lean
cause** and is not offered as one.

#### Hardware result (the one that counts)

| run | dur | `trig_joint` | rows α>0 | rows α=1.0 | tilt / fall |
|---|---|---|---|---|---|
| `11-39-01` **pre-fix** (head-off) | 276.3 s | 4.56% | 12712 (4.60%) | **12463** | 0% / 0% |
| `10-49-28` pre-fix | 92.8 s | 0.08% | 149 | 0 (α peaked 0.76) | 0% / 0% |
| `16-50-23` **post-fix**, all directions | 225.8 s | **11.75%** | **0** | **0** | 0% / 0% |

The post-fix run walks **3 min 45 s continuously in all directions with the safety filter
never engaging once**, across 26544 clamp-active rows. Note the direction of the
`trig_joint` change: it *rises* 4.56% → 11.75%, because the column now counts commands
past a stop rather than measured excursions — the policy asks for out-of-range targets far
more often than the joints actually got there. That is the precondition the old code turned
into a whole-body hold.

**Honest limits of this evidence.** It is one post-fix session versus one pre-fix session,
not a controlled A/B on hardware, and the sessions differ in more than the binary (head
off/on, different command sequences). The claim it supports is narrow and sufficient:
**the joint path no longer engages the hold**, which is exactly what changed. Whether the
"drunk stumbling" is fully gone is the user's qualitative call from the session, not something
these CSVs measure.

**Unintended hardware positive control for the tilt path.** Runs `16-34-34` and `16-35-28`
were launched against a build that still had a *temporary* `H1_2_TILT_LIMIT` of 0.02 rad
(left in the shared build dir by the verification work; the user caught it, reverted, rebuilt,
and re-ran as `16-50-23`). Those two runs show `trig_tilt` 98.9%/100% with `alpha` pinned at
1.000 — i.e. the tilt trigger and the ramped whole-body hold demonstrably still work **on
the real robot** after the port. Process lesson recorded: threshold-modified test binaries
must go in a separate build dir, never the one hardware sessions launch from.

#### Bridge + replica verification

- **Replica** (`bridge_replica.py`, both scenes × 0/2/4 ms): 6/6 pass, 0 falls. Caveat
  stated up front: the replica has applied the per-joint command clamp for *both* policies
  since 2026-07-22 and never modelled A0's whole-body hold, so it is already an image of
  post-fix behavior and **cannot discriminate pre- vs post-port**. It is a regression check
  here, nothing stronger.
- **Live bridge, new tooling** (`scripts/bridge_session.py`, see below): post-fix A0, 62 s,
  5327 joint-only-clamp rows (8.54%), **max `alpha` on those rows 0.000**, zero rows with
  `alpha`>0 in the whole file, 0 falls.
- **Pre-fix binary, same scripted session:** `trig_joint` **0.0000 in every segment**. The
  sim bridge's measured joints never leave range at all, while the policy commands past a
  stop 1.8-38% of ticks. **Neither the bridge nor the replica can reproduce the hardware
  trigger** — only real compliance lets measured `q` follow an out-of-range command across
  the bound. This is the mechanism behind "bridge non-observation was not evidence of
  absence", now measured rather than inferred, and it means this defect class is only ever
  *confirmable* on hardware.
- **G2.2 parity, bridge-vs-bridge pre→post** (the comparison decision 1 calls honest, since
  the vendor plant ≠ mjlab nominal by construction): achieved vx differs by
  **0.002 / 0.001 / 0.003 m/s** at commanded 0.3 / 0.5 / 1.0, against **0.001 m/s**
  run-to-run noise between two identical post-fix runs. Stride 0.573-0.585 s throughout,
  `act_rate` 0.001, 0 falls in all three runs. No tracking regression from the clamp.
  Against mjlab G1.1 steady-state the bridge tracks *closer* to command than mjlab does
  (bridge ss_err 0.027/0.043/0.056 vs mjlab 0.041/0.055/0.077); all three are inside the
  ladder's own ±0.05 bar.
- **Deliberate tilt/fall positive controls in the bridge** (temporary threshold builds,
  reverted): tilt → `trig_tilt` 100%, `alpha` ramped 0.02 → 1.000 over exactly 50 ticks
  (`H1_2_RAMP_CYCLES`) and stayed pinned. Fall → `trig_fall` 96.3%, `alpha` 1.000, first
  reached at t=0.133 s (60-tick detect window + 50-tick ramp = 0.110 s expected).
- **Fixtures:** `deploy.yaml.w1_legacy_test` and `deploy.yaml.g2_4_split_test` present and
  parsing in **both** `velocity/v0/params` and `velocity_hrl/v0/params` (the BadFile rule).

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

#### Hardware script for the next A0 session (G3.3 repeatability sign-off)

The 2026-07-23 post-fix run already met G3.3's *required* duration (3 min 45 s ≥ 2 min).
What is missing is **repeatability** (a second, independent session) and the user's explicit
qualitative verdict that the stumbling is gone. This is that session.

**Before launching — three preconditions, all cheap, all learned the hard way:**

1. `git diff deploy/robots/h1_2/include/h1_2_limits.h` must be **empty**. A threshold left
   at a test value is invisible at runtime and produced the `16-34-34` wrong-tilt run.
   Confirm `H1_2_TILT_LIMIT 0.44` / `H1_2_FALL_ACC_THRESH -7.0` by eye.
2. `pgrep -af h1_2_ctrl` must be **empty**. A stale controller holds the DDS lowcmd channel;
   the new one logs `The other process is using the lowcmd channel`, sits in Passive, and
   writes no CSV — easy to mistake for a dead policy.
3. Rebuild and note the binary timestamp, so a session can be tied to a build later.

```bash
cd ~/ramlab_ws/src/unitree_rl_mjlab/deploy/robots/h1_2
git diff include/h1_2_limits.h        # MUST be empty
pgrep -af h1_2_ctrl                   # MUST be empty
(cd build && make -j8) && ls -la build/h1_2_ctrl
source ~/ramlab_ws/setup_all_robot.sh # domain 0, enp11s0, deploy_real.yaml
h1_2_real                             # sets H1_2_SAFETY_LOG automatically
```

**On the robot the commands are the GAMEPAD, not the keyboard** (`deploy_real.yaml` uses
`velocity_commands`): `LT + up` = FixStand, `RT + A` = Velocity (A0), `LT + B` = Passive,
then the sticks (`ly` = vx, `lx` = vy, `rx` = yaw). Harness/gantry mandatory — the filter
damps, it does not catch.

| step | action | what to watch |
|---|---|---|
| 1 | `LT + up` → FixStand, let the 3 s ramp finish | posture should now match the policy nominal (the aligned `qs`); no visible pose jump at the next step |
| 2 | `RT + A` → Velocity, sticks neutral, stand ≥ 60 s | console must stay quiet apart from `[Safety] Clamp:` lines; **any `[Safety] Tilt:` or `[Safety] Fall:` line is the abort signal** |
| 3 | forward walk, ≥ 2 min continuous, free around the room | the qualitative question: is the "drunk stumbling" gone? |
| 4 | backward, both strafes, both yaw directions | the lean makes backward/sideways worst (WL-B0b), so this is where residual trouble shows |
| 5 | `LT + B` → Passive | clean damped stop |

**The `[Safety] Clamp:` lines are EXPECTED and are not a fault.** They report the new
rate-limited clamp counter (one line/s, worst offender). On `16-50-23` the clamp was active
11.75% of ticks through a clean 3¾-minute walk. What matters is that **no `Tilt:` or `Fall:`
line appears**, since those are the only remaining paths to a whole-body hold.

**Afterwards**, from the repo root with the session's CSV:

```bash
python scripts/deploy_gate_analyzer.py logs/deploy_safety/<TS>.csv
```

Sign-off criteria, all offline-checkable: `alpha` max **0.000** across the session;
`trig_tilt` and `trig_fall` both **0**; no fall; ≥ 2 min continuous in Velocity. Report
`trig_joint` (expect roughly 5-15%) as an observation, not a pass/fail — it now means
"clamp active", and a *rise* over the pre-fix number is the expected direction.

**If a lean-related observation shows up, record it and leave it** — the backward lean is
WL-B0b's, already decomposed, and is not this session's question.

### A0 gate ladder

Bars are the A1 track's bars unless noted; parity references are A0's own mjlab numbers.

| Gate | A0 status / what to do | Reference numbers |
|---|---|---|
| **G1.1** | Bench + holds exist for `a0_v2_optB_rs20_baseline`; vx=0.3 point filled 2026-07-20. Re-record only if the deployed ONNX is re-exported from a different run. | err_vx 0.085, CoT 0.537, 0 falls; ss@0.3 **0.041** (t90 0.54s), ss@0.5 **0.055** (t90 0.6 s), ss@1.0 **0.077** (t90 0.9 s) |
| **G1.2** | ✅ PASS (re-confirmed 2026-07-20). Single net, `velocity/v0/exported/policy.onnx`, no goal-scale metadata to check. | max abs diff 1.24e-05 < 1e-4 |
| **G2.0** | ✅ informally (loads and runs live); headless replica also 6/6 clean 2026-07-20 (both scenes, 0/2/4 ms). Live G2.0 folds into the G2.1 session. | no dim/FSM errors |
| **G2.1** | ✅ **PASS live 2026-07-23** (scripted, `bridge_session.py`): 12.9 s stand at cmd 0, `act_rate` 0.000, no fall, `alpha` 0 throughout (`trig_joint` 4.8% = command clamps only). Re-run longer than 60 s if a formal sign-off is wanted. | action-rate within ~1.5x A0's mjlab stand level |
| **G2.2** | ✅ **PASS live 2026-07-23** (scripted; holds were ~11.5 s each, not the 30 s the A1 ladder asks — sufficient given t90 ≤ 0.9 s, re-run longer if formal). Achieved ss vx **0.273 / 0.457 / 0.944** at 0.3/0.5/1.0 → ss_err 0.027/0.043/0.056, all inside ±0.05 of G1.1. Stride 0.573-0.585 s. 0 falls. Pre→post clamp delta 0.002/0.001/0.003 m/s vs 0.001 run-to-run noise. Backward/strafe/yaw not in this scripted run (keyboard presets are vx-only). | achieved vx within +-0.05 m/s of G1.1; stride within +-0.05 s |
| **G2.3** | ✅ **PASS live 2026-07-23**: 0→0.3→0→0.5→0→1.0→0 including the 1.0-from-stand killer, 0 falls, clean return to stand each time. Headless replica also 6/6 clean (both scenes, 0/2/4 ms). | 0 falls, clean return to stand |
| **G2.4** | **Live session pending.** Sim split-mode fixture prepared: `deploy.yaml.g2_4_split_test` (launch with `H1_2_DEPLOY_CFG` set to it). | same bars as G2.1-G2.3 |
| **G2.5** | **N/A for A0** (no estimator-fed goal channel). | - |
| **G2.6** | **N/A for A0** (fixed 0.6 s clock). | - |
| **G2.7** | Headless replica on `scene_stress.xml`: 0/2/4 ms, step 1.0 + walk 0.5, all clean, 0 falls. **Live session pending** (key script ready). | 0 falls at 0.5 (BLOCKING); 1.0 advisory |
| **E1/E2** | **E2 ✅ PASS (2026-07-20, real robot).** **E1 FAIL — no odometry data at all** (see below); does not block A0. | per Phase E |
| **G3.0-G3.3** | `h1_2_limits.h` fix landed 2026-07-21; live G2.1-G2.3 cleared 2026-07-23; the clamp port landed and is hardware-confirmed (225.8 s free walk, all directions, filter never engaged). **G3.3's required tier — ≥2 min continuous free walk — is met by that run at 3 min 45 s**, pending the user's qualitative sign-off that the stumbling is gone and a second session for "repeatable". Terminal-behavior design question (chasing vs latched hold) still open but no longer urgent for the joint path. **Open: the 2026-07-20 unexplained fall (item 1).** | G3.3 = repeatable free walk |

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
