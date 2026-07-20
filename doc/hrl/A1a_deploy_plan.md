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
  place unsettled ~~= the known TD3-HL hold degeneracy (training thread)~~ **[cause
  FALSIFIED 2026-07-16, WL-C: that "degeneracy" was a sim-EVAL artifact and never
  existed in training, so it cannot explain a bridge symptom — the stepping is now
  UNEXPLAINED, treat as open]**. vx=1.0 press
  fell via the safety filter's joint freeze (post-mortem issue 3, fixed; rebuilt).
  Re-run pending with the clamp-filter build; use number keys (5 = held 0.5) before w.
  **G2.0 must be re-checked (2026-07-16):** the deployed ONNX pair was re-exported and
  the goal scale is now pinned from its metadata — confirm `[HRL] goal scale pinned from
  ONNX metadata [0.75, 0.5, 1, 1, 1, 1, 0.2]` at FSM entry (this C++ path has not yet run
  live). See `.claude/docs/deployment.md`.
- **Replica permanent (2026-07-16):** `scripts/bridge_replica.py` = the headless bridge
  replica (A0 + HRL, both scenes, delay knob, band, full chain, reads deployed YAML +
  exported ONNX). Pre-session sanity + plant/latency A/B instrument. Validated: A0 and
  keeper both pass the chain at 2 ms on the vendor plant.
- **G2.0 pre-check (2026-07-17, headless, WL-B) — found 3 issues, 1 still open:**
  1. Bug (fixed): a variable name collision in `bridge_replica.py` (`c` = HL period,
     shadowed every tick by the swing-clearance instrument's per-contact loop variable,
     also `c`) crashed the script outright on any `--policy hrl` run. Renamed to `con`.
  2. Process error (caught + fixed): re-exporting the goal-scale metadata fix, I sourced
     the wrong checkpoint — `A1a_deploy_plan.md`/`worklines.md` name the 2026-07-09
     run as "the keeper", but the ONNX actually staged in `exported/` (confirmed via its
     `run_path` metadata) is the newer arm-calm/rs20 candidate
     (`2026-07-14_16-03-57_a1_td3_pose0p5shw16-4_ar0p02_cadhl_cot0p2_kl0p01_rs20_s42`,
     **this is the arm-weighted policy, and it IS deployable** — nothing blocks it). I
     briefly overwrote the staged pair with the wrong checkpoint's export (weight diff
     up to 2.15 vs. the original); caught via a weight-identity check, restored, and
     re-exported the correct checkpoint (weight-identical to what was staged, max diff
     0.0; parity LL 2.7e-05 / HL 3.2e-06). **Docs still calling the 07-09 run "the
     keeper" are stale** — the arm-calm/rs20 run is what is actually staged today.
  3. **Open finding — latency x goal-scale interaction:** `bridge_replica.py`'s own
     `gscale` was still deriving from `deploy.yaml` ranges (legacy path), not the ONNX
     `goal_scale` metadata the C++ fix now reads — so the tool wasn't exercising the
     same scale the real deploy does (only yaw differed: legacy 0.5 vs trained 1.0).
     Fixed to read the metadata (mirrors `hrl::GoalSpace::freeze_scale`). Re-running the
     chain with the TRUE trained yaw scale (1.0): **0 ms and 2 ms delay still pass
     cleanly** (calmer than before), but **4 ms delay now FALLS during takeover**
     (pitch 1.04 rad, height collapses to 0.41 m; deterministic, reproduced twice) —
     this passed under the old, incorrectly-halved yaw scale. Bridge delay is
     quantized to the 500 Hz control tick, so 0/2/4 ms are the only 3 distinct buckets
     in this range (2.2-2.8 ms all alias to the "2 ms" bucket, 3.0-4.0 ms to "4 ms").
     A yaw-scale-only isolation sweep at the 4 ms bucket is **non-monotonic**: 0.5/0.6/
     0.9 pass, 0.7/0.8/1.0 fail (at different phases each) — a knife-edge sensitivity,
     not a clean trend, consistent with the W2 latency/PD-energy-injection mechanism
     (small parameter changes tip the closed loop in/out of instability). The stress
     (harsh/damped) scene is fully robust at 4 ms with the corrected scale. Documented
     real bridge latency is ~2-4 ms, i.e. this failure mode sits right at the edge of
     what the live bridge may actually exhibit — **treat takeover (the `h` key, first
     ~3s) as the highest-risk moment in the next session** until this is resolved or
     better characterized.
     **Startup-ramp mitigation tested headlessly (2026-07-17) — result: unreliable, not
     adopted.** Linearly ramping the HL goal-scale multiplier 0->1 after takeover begins:
     0.3/0.5/0.75/1.0 s all still fell (0.5 s even fell harder, later, in "stand");
     2.0 s passed the FULL chain; **2.5 s (more conservative) then failed at "stop"**, a
     phase every shorter/no-ramp run had passed. Not measurement noise (the sim is
     deterministic) — the same latency/PD chaos as the W2 post-mortem, and it means a
     hand-tuned ramp duration cannot be trusted as a fix: passing at one value gives no
     assurance about nearby values. Do not adopt a fixed-duration ramp off a single
     passing sweep point. Open options: (a) a slew-rate limit on |delta V*| per HL tick
     instead of a time-based scale ramp (bounds step-discontinuities directly rather
     than through a clock; untested, may be more robust to this chaos); (b) treat the
     safety filter + flight recorder as the real backstop and use the live bridge to
     characterize whether actual DDS latency reaches the risky 3-4 ms band in practice
     (the replica's `--delay-ms` is a knob, not a measurement); (c) a training-side
     quiescent-|g| regularizer (WL-D territory, slower) or an actual latency reduction
     (bridge engineering, bigger lever) as root fixes instead of a deploy-side patch on
     a closed loop that is already marginal at this delay.
- **Flight-recorder bug found + fixed (2026-07-17):** `SafetyLogger::init()`
  (`safety_logger.h`) and `hrl::Telemetry::init()` (`hrl_telemetry.h`) truncated their CSV
  and reset their tick counter on EVERY FSM entry (every `h` press), not once per process
  — so a failed attempt's telemetry was silently destroyed the moment a later attempt in
  the same session was logged. This is why the first live re-run (`2026-07-17_11-40-42`)
  showed a clean no-fall file despite Liam observing a fall earlier that session: the
  fall's data no longer existed by the time it was inspected. Fixed: both loggers now
  append across re-entries within one process and tag every row with a new `entry` id;
  `deploy_gate_analyzer.py` segments on `entry` changes too and no longer drops a fallen
  segment under the 3s `MIN_SEG_S` floor (a fast fall must never be filtered out).
  Rebuilt, confirmed clean.
- **Live re-run (2026-07-17, `2026-07-17_12-01-08`, post-fix) — 9 attempts recovered,
  2 real findings:**
  1. **Entry-tilt sensitivity, Liam's hypothesis confirmed but not a clean threshold, and
     likely NOT a real hardware risk.** Per-entry tilt-at-`h`-press vs. outcome:
     0.3/1.6-13.9/20.2 deg all fine (height stayed >=1.1); **40.2 deg -> stumble, height
     dipped to 0.74**. So "not upright at entry" does predict trouble, but 20 deg alone
     was NOT enough to trigger it — a fuzzy band, not a hard cutoff. Liam's note: on the
     real robot `h` is only pressed once genuinely upright (~5 deg), well inside the
     always-fine range here, so this is a loose-sim-GUI-testing artifact (pressing `h`
     mid-lean), not a required deploy-side fix.
  2. **A second finding, corrected mechanism (Liam's read, not chaos): step-from-stand vs.
     ramped acceleration, matching the documented `bridge_replica.py` "historical bridge
     killer" case (0->1.0/0.5-from-stand), not "the same command randomly failing."** One
     entry started essentially perfectly upright (2.5 deg) and ran cleanly through a
     GRADUAL ramp cmd 0 -> 0.2 -> 0.3 -> 0.4 -> 0.5 (13.4 s, fine); after returning to 0 it
     then took 0.5 again as a DIRECT STEP from standing -> full collapse (height 0.14).
     Liam's hypothesis: insufficient foot-lift/swing clearance during the sudden
     acceleration demand, not a random latency flip. **Checked headlessly and it holds up
     well**: same replica, same scene, same delay (0 ms), same step magnitude — the HRL
     keeper's step-from-stand swing clearance is **0.050-0.057 m mean**, roughly HALF of
     A0's **0.065-0.098 m** in the identical test. A concrete, matched-condition
     difference, not proof of causation but a strong match for "foot lifting could be
     responsible." **This is a gait/reward-shaping question (WL-D territory: the
     swing-clearance metric + "raise clearance" lever already exist from WL-E, see
     `docs/adr/0005` Amendment 2 item 1) — flagged here, not implemented by WL-B.**
- **Flight-recorder bug, round 2 (2026-07-17): A0/A1 use SEPARATE `SafetyLogger`
  instances** (`deploy/include/FSM/State_RLBase.h` vs. robot-local
  `State_RLHRL.h`), so the round-1 fix (per-instance "have I run before" check) still let
  the FIRST switch from A0 to A1 in a session truncate the file — A1's own instance had
  never run before, so from ITS perspective it was still a "first entry." Fixed: the
  truncate-vs-append decision now checks whether the CSV already exists ON DISK (robust
  across different C++ objects), and the `entry` id is now a process-wide counter
  (`g_safety_logger_entry`, a C++17 inline global in `safety_logger.h`), not per-instance,
  so ids stay unique whether the retry is same-type or a cross-type switch. Rebuilt,
  confirmed.
- **Live re-run (2026-07-17, `2026-07-17_12-24-16`) — recorded BEFORE the round-2 fix
  above, so the A0 portion of this session was confirmed lost (only `entry=0` present,
  spanning what is actually just the A1 portion — A0's data was overwritten the moment
  `h` was first pressed, exactly the round-2 bug). Liam confirmed operator workflow: fall
  -> reset before retrying; no fall -> sometimes stop/restart without a reset — so despite
  the single `entry` id, the by-command segments plus the safety filter's own discrete
  `trig_fall` rising edges (a cleaner signal than height dips, since it marks genuine
  alpha->1 escalations) give a trustworthy, largely-independent-trials failure count:
  **0.5 m/s: 4 confirmed falls out of ~6 attempts (only ONE clean 23s pass); 1.0 m/s: 2/2
  falls.** Critically, **the final step of a deliberate gradual ramp (0.2->0.3->0.4 all
  clean) ALSO fell exactly at 0.5** — this revises the step-vs-ramp framing from the
  previous entry: a controlled, gradual buildup fails too, so it is not purely a
  step-transient/foot-clearance-at-the-moment-of-demand story. Read: **0.5+ m/s looks
  marginal for this candidate in this bridge regardless of how the speed is reached** —
  more consistent with a chronic gait/clearance shortfall (still fits Liam's swing-lift
  hypothesis and the measured clearance gap vs A0) than a pure transient-shock mechanism.
  Same WL-D routing as above; not a WL-B fix.
- **W1 back-compat regression: PASS (2026-07-17).** Pre-velgoal 7-dim absolute-mode
  checkpoint (`2026-06-18_10-10-27_a1_td3_relabel_absolute_no_warmstart_7k/model_7000.pt`)
  exported clean (parity LL 1.1e-05 / HL 2.3e-05), swapped into `exported/` alongside a
  legacy-style `deploy.yaml.w1_legacy_test` fixture (no velgoal/cadence keys, loaded via
  `H1_2_DEPLOY_CFG` so the live config was never touched), ran the live bridge: no dim
  errors, stood at cmd 0 for 19.1s (`2026-07-17_16-07-57`), no fall. Absent-key back-compat
  confirmed still working. Live candidate restored and verified (`hl_target_mode=delta`,
  `hl_velocity_goals_only=True` — the correct arm-calm/rs20 pair, not left mid-swap).
- **G2.6 cadence modes: BOTH fell (2026-07-17)** — HL-owned (`2026-07-17_16-09-52`) and
  pinned 0.625s (`2026-07-17_16-12-02`), held vx 0.5. HL-owned fell repeatedly and clearly
  (4 episodes, height down to 0.03-0.2). Pinned looked clean in `_hrl.csv` alone (height
  stayed 1.22) but the base safety CSV's `trig_fall` shows TWO real filter engagements
  (alpha->1.0) at t=11.89s (recoverable stumble, height held) and t=20.45s (likely an
  actual collapse, right at the very end) — **found only by cross-checking the raw
  `trig_fall` column, not the HRL telemetry**, because `hrl_telemetry.h`'s buffer only
  flushed every ~10s (or on a clean state exit), losing the tail on an abrupt session end
  while the safety CSV's ~2.5s cadence kept it. **Fixed: `FLUSH_ROWS` 512->128 (~2.5s),
  matching the safety logger; rebuilt.** Net read: cadence source (HL-owned vs pinned)
  does NOT rescue held-0.5 stability — consistent with, and reinforcing, the existing
  WL-D routing (chronic gait/clearance shortfall at 0.5+ m/s, not a cadence-mechanism
  question). No further WL-B action; not re-testing until WL-D's fix lands.

Operational gated checklist for `A1a_plan.md` Plan v2 stage D. Grilled 2026-07-14.
Execution is gate-by-gate; no gate starts before its blockers pass. The battery
(phases G1+G2) is checkpoint-agnostic: it runs once now on the current keeper
(pipeline shakedown) and re-runs cheaply on every M2 candidate. Hardware (G3) only
on the M2 winner.

**Scope note (2026-07-20): this doc now covers TWO deploy tracks.** The A1 keeper track
(everything below, as written) and the **A0-first hardware track** (Liam's call: the
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

## A0-first hardware track (added 2026-07-20, Liam's call; owned by WL-B)

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
| **E1/E2** | Run as written. E1's bar is softer for A0 (estimator quality is not in A0's control loop), but E2 (IMU convention / false fall trigger) is unchanged and still blocking. | per Phase E |
| **G3.0-G3.3** | As written, with item 1's terminal-behavior decision made first. | G3.3 = repeatable free walk |

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
