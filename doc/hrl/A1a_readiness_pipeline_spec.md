# Hardware-readiness validation pipeline — analysis + spec (v2, grilled 2026-08-04)

Status: **analysis complete, spec decided in a grilling session, awaiting go-ahead to
implement. No code written yet.**
Scope: the 8-task hand-off (provenance gate → chained one-command verdict).
v1 → v2: all open decisions closed; F6 added; the plant choice reversed on new provenance.

---

# Part 1 — Analysis

Six findings from the existing 2026-08-03 captures and the bridge model files. Four of
them **retract claims currently written into `A1a_deploy_plan.md`, `hrl-infra.md`,
`deployment.md`, the deploy journal or auto-memory**, so they change what the pipeline
gates on. Every number is recomputed from the committed CSVs, not quoted.

## F1 (task 2) — the "10 ms tail" is a printf artifact. `run()` sustains ~990 Hz. The 1 kHz assumption is RESTORED, not falsified.

Two defects stacked.

**(a) The `t` column is not a clock.** `safety_logger.h:133` writes `tick_ * dt_` with
`dt_ = control_dt = 0.001` (confirmed in `*_meta.json`) and `tick_` incremented once per
`record()` call (`:127`, before the decimation early-return). `t` is **iteration index ×
nominal dt** — a counter. No dt statistic from it can measure the loop rate; the quantity
is circular by construction.

**(b) Past t = 10 s the column is quantised to 10 ms by `std::setprecision(4)`**
(`:55`) — 4 *significant* digits, so above 10 s only two decimals survive. Splitting
`2026-08-03_15-18-58_ctrlB_a0.csv` at t = 10 s:

| window | rows | observed dt set |
|---|---|---|
| t < 10 s | 5 909 | **{1 ms: 1818, 2 ms: 4090}** — nothing else |
| t ≥ 10 s | 19 837 | **{0 ms: 15934, 10 ms: 3902}** — nothing else |

Below 10 s the file shows exactly what `log_every_ = round((1/500)/0.001) = 2` predicts
(2 ms quiet, 1 ms when a trigger bypasses decimation) — **no tail at all**. Above 10 s
there are 3903 distinct values spanning 10.00→49.02 s; (49.02−10.00)/0.01+1 = 3903
exactly. The "p90 = p99 = max = 10.00 ms" is the 0.01 s print bin; the duplicate rows are
2 ms samples collapsing into it.

**The real rate, measured.** `bridge_session.py` holds each key for an exact wall-clock
duration; the CSV records the same segment in synthetic time. That ratio is the rate. All
four 2026-08-03 runs used `--seq "0:8,3:15,5:15,0:8"`:

| segment | scripted wall | synthetic (all 4 runs) | ratio |
|---|---|---|---|
| 0.3 hold | 15.0 s | 14.81 – 14.87 s | **0.989** |
| 0.5 hold | 15.0 s | 14.84 – 14.87 s | **0.990** |
| final 0 | 8.0 s | 7.89 – 7.94 s | **0.988** |

**`run()` executes at 989 ± 2 Hz in every run, including the two that fell.** A 10 % duty
of 100 Hz stalls would have made synthetic time lag wall time ~2x; it lags by 1 %.

**(c) The "mean rate 942 / 525 / 700 Hz" figures were `trig_joint` in disguise.**
Triggered ticks bypass decimation (`:131-132`), so logged/total = f_trig + 0.5(1 − f_trig).
Measured logged fractions A0 0.525, keeper 0.700, rolloverfix 0.942 → implied f_trig
0.05 / 0.40 / 0.885, tracking the reported clamp rates in ordering and magnitude.

**Withdrawn:** the journal's "the 1 kHz assumption is falsified by measurement" and
hrl-infra's "the deploy platform does NOT sustain 1 kHz" — and with them the leading
hypothesis for task 4. Corroborated independently: the bridge publishes `lowstate` (IMU
included) from a `RecurrentThread` at **1000 µs = 1 kHz**
(`/opt/unitree_mujoco/simulate/src/unitree_sdk2_bridge.h:170-171`), so the accelerometer
is sampled at least as fast as the 200 Hz sim rung.

## F2 (task 4) — the estimator's bridge "error" is a broken ground-truth reference, not an estimator failure.

`h1_2_handless.xml:331,340`:

```xml
<framepos    name="frame_pos" objtype="site" objname="imu" />
<framelinvel name="frame_vel" objtype="body" objname="pelvis" />
```

The bridge publishes **pelvis body** velocity (the 2026-07-15 retarget `deployment.md`
documents) but **imu-site** position. `State_RLHRL.cpp:302-304` asserts both are at the
site, and `gt_dvel` is built on it (`:408`):

```cpp
const Eigen::Vector3f gt_dvel = rz(psi, v_gt_b - lev) - gt_vel0_;   // lev = w_T x r
```

It subtracts a lever-arm term from a signal that has none. With `L_i ≡ Rz(ψ_i)(w_i × r)`:

```
gt_logged = Δv_pelvis          − (L_i − L_0)
est       = Rz(ψ_i)·dv_int     − (L_i − L_0)
est − gt_logged = Rz(ψ_i)·dv_int − Δv_pelvis
```

**The lever correction cancels**, so the logged residual scores the *uncorrected*,
site-frame reconstruction — the rung the 2026-08-01 bench called the dominant error term.

| | vx | vy |
|---|---|---|
| bench, waist-corrected @200 Hz (what the estimator implements) | 0.0198 | 0.0255 |
| bench, **lever arm (site−pelvis)** | 0.0497 | **0.2266** |
| bench, grav-removed uncorrected | 0.0604 | **0.2275** |
| **bridge, keeper arm4d (measured)** | **0.1145** | **0.2054** |

**vy matches the uncorrected rung (0.205 vs 0.227), not the corrected one (0.026)** — an
8x discriminator landing on the predicted value.

**Honest limit:** vx (0.115) sits ~2x above its rung (0.050–0.060) and is **not**
explained by this. A second contributor may remain on vx. The spec re-measures after the
reference is fixed rather than assuming it away.

## F3 (task 5) — the height bias is two identified static errors plus a remainder.

Measured over the upright portion: bias `est_h − gt_h` = **+0.0331 ± 0.0052** (keeper) and
**+0.0336 ± 0.0152** (rolloverfix); correlation with `est_h` +0.31 / −0.12, i.e. no
consistent pose dependence → static, as the hand-off states.

`est_h = nominal_h_ + [P(q) − P(q_def)]`, `P = −lowest_foot_z` (pelvis above sole);
`gt_h` = imu-site world z. So

```
est_h − gt_h = [1.03004 − P(q_def)]  +  [P(q) − P_true(q)]
                 anchor error             FK/sole model error
```

| term | value |
|---|---|
| `P(q_def)` = `fk_h_nominal_` (recomputed from the shipped FK) | 1.00236 m |
| training nominal pelvis z (1.3076 − 0.27756) | 1.03004 m |
| **anchor error** — the two constants reference **different poses** | **+0.0277 m** |
| sole model: bridge mesh foot −0.058 vs training capsule −0.045 (MuJoCo AABB, both models) | **−0.0130 m** |
| **predicted** | **+0.0147 m** |
| observed | +0.0325 / +0.0331 m |

**Dominant identified contributor: `nominal_root_height` (1.3076) references the training
nominal standing pose (pelvis 1.030 m), while `fk_h_nominal_` is evaluated at the deploy
`default_joint_pos`, a crouched PD-hold pose (knee 0.5, hip pitch −0.2) whose
pelvis-above-sole is only 1.002 m.** The anchor identity `est_h(q_nominal) = nominal_h_`
holds at the wrong pose. ~+0.018 m remains unattributed — a measurement item, not a guess.

## F4 (task 7) — `trig_joint` is ankle-roll command saturation. The limit table is correct. The bridge sees it fine; the auto-memory is stale.

Recomputing "raw policy target outside `h1_2_limits.h`" from `raw_q*` reproduces the
logged column: A0 9.6 % vs 9.6 %, keeper 57.2 % vs 57.2 %, rolloverfix 92.5 % vs 93.9 %.

| run | dominant joint | rate | limit | raw p1 / p99 |
|---|---|---|---|---|
| A0 | right/left ankle_pitch | 5.2 / 4.8 % | [−0.90, +0.52] | −0.71/+0.90, −0.52/+1.00 |
| keeper arm4d | right/left **ankle_roll** | 35.0 / 31.1 % | **[−0.26, +0.26]** | −0.55/+0.60, −0.56/+0.58 |
| rolloverfix0p5 | right/left **ankle_roll** | 74.9 / 58.0 % | **[−0.26, +0.26]** | −0.70/+1.51, −0.58/+0.65 |

`h1_2_limits.h`'s ankle-roll ±0.26 **is identical to the training XML's ±0.261799**
(`h1_2.xml:82,123`) — not the knee discrepancy `deployment.md` flags. **The policies
genuinely command 2.1–2.7x past the mechanical stop.**

Root cause: MuJoCo silently clamps `qpos` at the joint range, so a policy commanding
±0.6 rad of ankle roll receives the ±0.26 result and is never charged for the difference;
no training reward term penalises commanding outside range. A0's gait stays inside; the
HRL gaits ride the stop every stride — which is why A1 got the per-joint command clamp
instead of A0's whole-body hold on 2026-07-16.

**Auto-memory `deploy_sim_blind_to_joint_hold` is stale and must be corrected.** Its claim
(the bridge structurally cannot observe joint-limit defects) was true under the
**pre-2026-07-16 semantics**, where `trig_joint` meant "a *measured* joint left range" —
something MuJoCo never produces. Under the current semantics it scores the *commanded*
target, which MuJoCo does not clamp: 10 % / 57 % / 93 % is the bridge doing exactly what
the memory says it cannot.

## F5 (task 1) — provenance surface, and the deploy dir is an unlabelled mutable slot.

All four ONNX in `velocity_hrl/v0/exported/` carry `run_path='local'` — useless, as the
hand-off states. `goal_scale` present. Dims: HL **94** (`hl_obs_vel: true`), LL **96**
(= 89 + goal_dim 7).

**Run dirs already hold byte-identical exports.** Verified now: deployed
`high_level.onnx` md5 `c83da701` and `low_level.onnx` `c11d591f` both match
`logs/rsl_rl/h1_2_velocity_a1_v2/2026-07-17_21-46-14_a1a_cot0p2_cad0p5_energy0p05_s42/`,
and A0's `policy.onnx` `8a3871d5` matches
`logs/rsl_rl/h1_2_velocity_v2/2026-07-15_08-29-26_a0_v2_optB_rs20_baseline/`. So the
deploy dir currently holds the keeper — **and a reverse md5 index over
`logs/rsl_rl/**/*.onnx` can NAME an unknown deployed file**, which is what was missing on
2026-08-03 ("matched no run export I checked by md5"). No re-export needed.

**`onnx_parity.py` is structurally blind to the 2026-08-03 defect.** It exports the
checkpoint to a *temp dir* and compares torch against its own fresh export
(`:157-194`). It never opens the deployed file. W4 validates the export *function*; a
correct function plus a stale deployed file passes it clean.

**Filenames are hardcoded** — `exported/high_level.onnx`, `exported/low_level.onnx`
(`State_RLHRL.cpp:206-207`), no YAML redirect. So the deploy dir is a single mutable slot
with no record of its contents: exactly the mechanism that parked `rolloverfix0p5` there
on 2026-07-29. A0's dir compounds it — **four distinct files of identical size (863446 B)**,
so size cannot distinguish them, only md5 can.

## F6 (new, 2026-08-04) — the bridge scenes are misnamed, and the base-damper argument is falsified.

**Provenance correction (the user).** `h1_2_handless_stress.xml` / `scene_stress.xml` is
the **official Unitree-shipped scene**. `h1_2_handless.xml`, whose own header calls itself
"BRIDGE PLANT = VENDOR REFERENCE ... closer to real hardware", is a **hand-made
training-proximate scene**. The "vendor" label is a misnomer that has propagated into the
XML comment, `deployment.md` ("Plant = vendor reference"), and `A1a_deploy_plan.md` W2.
**It must be corrected as part of this work** — it will keep misleading readers, and it
was the basis of my own initial (wrong) recommendation.

**Falsified by the user's own A/B: the fictional 6-DOF base damper is not the
discriminator.** The shipped scene leaves the floating base with no override
(`h1_2_handless_stress.xml:42`) so it inherits `damping="1" armature="0.1"
frictionloss="0.2"` on all six base DOF, while the training-proximate scene explicitly
zeroes them (`h1_2_handless.xml:51`). I attributed the shipped scene's calmness to that
damper. The user zeroed the floating base in the shipped scene and observed **no
meaningful behavioural change**. The calmness therefore comes from the **joint** values
(damping 1 vs 0.001 — a 1000x difference, absorbing oscillation exactly where twitch
lives), not the base. Mechanism right in kind, wrong in location.

**Consequence:** real harmonic-drive joints are not frictionless, so the shipped scene has
a genuine claim to being the better hardware proxy. Zeroing its floating base is adopted
anyway — it removes an unphysical term at no behavioural cost, which is the ideal evidence
for a change.

**Observation to carry forward (the user).** At cmd 0 the shipped scene shows the
pelvis/torso **drifting left/back**, while the training-proximate scene shows the robot
stepping. Reading: a standing bias has to go somewhere — the damped plant suppresses the
corrective stepping so the bias integrates into drift; the lively plant lets it step. Same
defect, two plant responses. A *backward* bias at zero command is the bridge-side echo of
WL-B0b's hardware backward lean. The *left* component sits alongside the still-open L/R
leg asymmetry. Ruled out this session: `joint_offset` is absent from both sim YAMLs
(= inert), and `default_joint_pos` is L/R symmetric — so it is not deploy-side asymmetry.
**A cmd-0 drift metric is added to the pipeline** to put a number on it.

---

# Part 2 — Spec (all decisions closed)

## S0. Deliverable

```bash
python scripts/deploy_readiness.py <TaskID> --checkpoint-file <pt> \
    [--policy-dir deploy/robots/h1_2/config/policy/velocity_hrl/v0] \
    [--stages provenance,sim,replica,bridge,analyze] [--tag <name>] \
    [--refresh-control]
```

**Fully chained and fail-closed** (decided): one command runs provenance → sim → replica →
bridge (both arms) → analyze, unattended. **Hard rule: the pipeline never reports GO from a
stage it could not score.** Falls are read from CSV evidence (height, `trig_fall`,
`trig_tilt`), never from FSM transitions — the 2026-08-03 process failure.

**Exit codes:** `0` GO · `3` GO-WITH-CAVEAT · `1` NO-GO · `2` infrastructure (a stage
could not run or could not be scored — never silently a pass).

**Failure handling:** a provenance failure aborts immediately, before any process starts.
Every later stage runs to completion even after a failure, so one invocation yields the
full evidence set — including the A0 control arm, which is worthless if the chain stopped
at the candidate's fall.

New file `scripts/deploy_readiness.py`; an orchestrator with **no metric maths of its
own** — every number comes from an existing tool's stdout marker.

## S1. Stage `provenance` — blocking, first, ~2 s, no GPU

**Staging is owned by the pipeline** (decided). It copies the named checkpoint's exports
into `exported/`, re-verifies md5 after the copy, and writes `exported/PROVENANCE.json`
(source run, checkpoint path, per-file md5, timestamp, staged_by). This is the actual cure
for 2026-08-03: the deploy dir cannot hold a stranger, because the tool that runs the
bridge is the tool that filled it, and the record travels with the files.

| check | rule |
|---|---|
| P1 identify | md5 each deployed ONNX; reverse-index `logs/rsl_rl/**/*.onnx`; **name the source run**. Never trust `run_path` (writes `local`) |
| P1b fallback | no md5 match → numerical identity vs the candidate checkpoint via `onnx_parity.py --onnx-dir` (new flag, scores the **deployed** file — closes the W4 blindness). Pass with a loud warning. Neither → NO-GO, printing both md5s |
| P1c manifest | cross-check live md5s against `PROVENANCE.json`; flag drift |
| P2 dims | HL == 94 if `hrl.hl_obs_vel` else 92; LL == 89 + goal_dim(`goal_components`) |
| P3 metadata | `goal_scale` present on both (else the C++ silently takes the legacy range-derived path) |
| P4 structure | `c`, `hl_target_mode`, `hl_velocity_goals_only`, `cadence_period_range`, `goal_components` equal in ONNX metadata and deploy YAML |
| P5 parity | `onnx_parity.py` max abs diff < 1e-4 |
| P6 scene | assert `/opt/unitree_mujoco/simulate/config.yaml` `robot_scene` == the expected shipped scene; mismatch = **exit 2**, no mutation of `/opt` at run time |

**The A0 control ONNX gets the same P1–P5 treatment** — an unverified control arm cannot
underwrite an attribution.

## S2. Stage `sim` — blocking, `play.py` only

| gate | command | marker | bar |
|---|---|---|---|
| S2.1 bench | `--num-envs 64 --eval-steps 600 --eval-seeds 2` | `[BENCH]` | **`fall_rate == 0`** |
| S2.2 leg action rate | `--diagnose-action-rate` | `[ARDIAG]` | **report only** |
| S2.3 holds | `--eval-cmd-vx 0.5`, `1.0` | `[BENCH]`, `[HOLDDIAG]` | `fall_rate == 0`; `ss_err`, `t90_s`, per-env bwd/fwd split recorded |
| S2.4 estimator | `--check-vel-increment 480 --eval-seeds 2` | `[VELINC]` | bands below |

Whole-body `act_rate` is recorded but **flagged misleading** under `hold_joint_ids:
[12..26]`; only the leg group is quoted in the verdict.

## S3. Stage `replica` — blocking

`bridge_replica.py`, shipped scene, `--delay-ms 0/2/4`. Bar: no fall in any cell.

## S4. Stage `bridge` — blocking

**Plant: shipped Unitree scene only** (decided), with its floating base zeroed. The
training-proximate scene is retired as a gate.

> **Cost recorded at decision time.** This retires the more sensitive twitch detector, and
> **every existing reference number was measured on the retired plant** — leg action rates
> 0.5908 / 0.7552 / 0.8125, arm4d's live pass, the G2.x parity bars. They **do not
> transfer**. Until re-baselined on the shipped scene, action rate is reported with **no
> reference** and explicitly marked so; quoting the old numbers against a shipped-scene
> measurement would be the same cross-instrument error F1–F2 exist to correct. First run
> on the shipped scene establishes the new A0 and keeper baselines.

**Sequence — full G2.2 + G2.3** (decided), ~253 s/arm, ~8.5 min both arms:

```
0:8, 3:30, 0:6, 5:30, 0:6, w:30, 0:8,      # G2.2 forward parity + G2.3 1.0-from-stand
s:15, 0:6, a:15, 0:6, d:15, 0:6,           # backward, strafe both
q:15, 0:6, e:15, 0:6,                      # yaw both
5:12, q:8, 0:10                            # G2.3 walk -> turn -> stop
```

**Deviation on record:** the keyboard map (`h1_2_observations.h:36-53`) has no ±0.3
lateral preset — `s` = −0.5, `a`/`d` = ±0.5, `q`/`e` = ±0.5. So G2.2's backward/strafe
legs run at **±0.5, stricter than the gate specifies**. Not silently weakened; not
"fixed" by editing the key map.

**Transition windows (task 6).** Every command change opens a scored 1.5 s window
(sub-scoring inside the existing `_segment_bounds`, not a new segmenter).
- **blocking:** no fall in any transition window (grounded — the observed 2026-08-03 failure)
- **blocking:** zero safety-filter engagement session-wide (`alpha.max() == 0`,
  `trig_tilt`/`trig_fall` all zero)
- **report only:** peak `|ach_v − cmd|` overshoot per transition, with A0 alongside on the
  identical sequence. **No numeric overshoot bar** — none is grounded.

**Band-release guard.** The band is a silent GLFW toggle (`main.cc:622-631`, no logging)
routed by X input focus; an iconified window swallows it and yields a session that looks
perfect with the robot hanging on the harness.
- *Preflight:* drop the forced window raise (so a backgrounded window still works), set
  focus, **assert via `get_input_focus` that it landed**, abort if not.
- *Post-hoc:* over any ≥10 s hold at cmd ≥ 0.3, integrate `|ach_vx|`. Walking covers
  ~3 m, band-held ~0 — an enormous margin needing no tuned velocity threshold. Under
  0.5 m ⇒ **BAND-NOT-RELEASED, exit 2**, never NO-GO: a harness artifact is not a policy
  verdict. The `3:30` hold satisfies the detector.

**A0 control arm, cached** (decided: no time expiry). Fingerprint = md5(A0 `policy.onnx`)
+ md5(`h1_2_ctrl`) + md5(deploy YAML) + md5(scene XML) + sequence string. Match ⇒ reuse,
showing the original capture timestamp in the report; any change ⇒ re-run.
`--refresh-control` forces it. *Default I chose where you did not specify:* the binary,
YAML, scene and sequence are included because a rebuilt binary is a **different shared
layer** — precisely what the control vouches for. Every C++ edit in this work rebuilds it,
so the cache is cold through implementation and warm during candidate iteration.

**cmd-0 drift metric (new).** Over each stand segment, integrate base displacement and
heading change, reporting magnitude **and direction**. Report-only. Quantifies F6's
left/back drift and is the same instrument that would put a number on the hardware lean.

## S5. Analyzer defects (task 3)

**D1 — HRL sessions analyzed with the wrong file.** `bridge_session.py:189` always hands
`<base>.csv` to the analyzer even for `--policy hrl`, so the richer `_hrl.csv` (height,
`act_rate`, `period`, `est_*`/`gt_*`) is never read. The analyzer's own `is_hrl` detection
is fine. **Fix:** for `--policy hrl`, analyze `<base>_hrl.csv` **and** `<base>.csv` — they
are complementary (the base CSV carries triggers and the A0-comparable columns).

**D2 — `ZeroDivisionError` in `stride_proxy`.** `analyze_base_csv` computes
`dt = median(diff(t))` = **0.0** because 71 % of rows share a printed timestamp (F1b);
`stride_proxy` then evaluates `int(2.0/dt)`. **Fix, two parts:**
1. *Analyzer:* take `dt` from `<base>_meta.json`'s `control_dt` × the decimation factor —
   the file already declares its own timing and is called "the single source of truth for
   the analyzer" (`safety_logger.h:68`). Never re-derive dt from `t`. Guard: `dt <= 0` ⇒
   `stride_proxy` returns NaN rather than raising.
2. *Logger (`safety_logger.h:133`, 1 line):* emit `t` with enough digits to stay
   ms-resolved past 10 s instead of the global 4-significant-digit setting. Every other
   float keeps 4 sig digits (the 2026-07-27 file-size decision untouched).

## S6. Height anchor (task 5)

Derive the anchor so the identity holds by construction:

```cpp
// est_h must equal gt_h at default_joint_pos, so the anchor is the FK pelvis height at
// THAT pose plus the fixed torso->imu lever — not a separately measured standing height
// referenced to the training nominal pose.
nominal_h_ = fk_h_nominal_ + kImuOffsetT.z();
```

`nominal_root_height` stays in the YAML as an override **and cross-check**: disagreement
> 5 mm logs a loud warning with both numbers — that warning would have surfaced this on
2026-08-03 instead of after two bridge runs.

**Sole offset: do nothing yet, measure first** (decided). Pre-registered prediction —
removing the +0.0277 anchor term leaves **+0.005 m**; pass bar **≤ 5 mm** on a 30 s cmd-0
bridge stand. A bridge-side correction is introduced **only if the measurement demands
it**. Rejected on principle: changing `base_state.h`'s sole constant to match the bridge
mesh — the deploy FK must mirror the real robot.

Also fix the stale provenance comment at `base_state.h:24-25` (the bridge loads
`/opt/unitree_mujoco/unitree_robots/h1_2/h1_2_handless*.xml`, which has **mesh** feet, not
the in-repo `scene_h1_2.xml`).

## S7. Estimator reference (task 4)

`State_RLHRL.cpp:408` — `v_gt_b` is the **pelvis** velocity in the torso frame, so:

```cpp
const Eigen::Vector3f gt_dvel = rz(psi, v_gt_b) - gt_vel0_;   // no lever subtraction
```

with `gt_vel0_ = rz(psi, v_gt_b)` at window start. Fix the wrong frame comment at
`:302-304`. **Touches only the telemetry reference** — the estimator and everything the
policy consumes are untouched.

**Pre-registered reading of the re-measurement** (agreed before the run):
- ≤ 0.05 both axes ⇒ the 2026-08-01 ship-band conclusion is **confirmed on the bridge**,
  provisional status lifted;
- vy ≤ 0.05 but vx ~0.1 ⇒ F2's unexplained vx term is real and gets its own investigation;
- both high ⇒ F2 is wrong and the rate question reopens — with F1's instrument, not the
  synthetic clock.

## S8. `trig_joint` reporting (task 7)

No change to the clamp. Analyzer additions:
- per-joint clamp **rate** and **magnitude** (`max`/`p99` distance past the stop, rad) —
  magnitude sets firmware torque against a stop and the rate alone hides it;
- an explicit block stating **"acceptable rate is UNKNOWN"**, the three measured reference
  points (A0 10 % hardware-proven, keeper 57 % bridge-passed, rolloverfix 87-93 % fell),
  and the dominant joint.

Reported, **never gated**.

## S9. Gates

**Blocking:**
1. provenance P1–P6
2. `fall_rate == 0` on the bench and both holds
3. zero safety-filter engagement in every bridge phase
4. no fall in any bridge phase **including transition windows**
5. estimator error, **both instruments**, against pre-registered bands:

| instrument | GO | CAVEAT (exit 3) | NO-GO |
|---|---|---|---|
| sim `[VELINC]` | ≤ 0.05 | 0.05 – 0.15 | > 0.15 |
| bridge est-vs-gt | ≤ 0.075 | 0.075 – 0.225 | > 0.225 |

Metric `max(vx, vy)`. Bridge bands = **1.5x sim**.

**Provenance of these numbers, because the order matters.** 2x sim (0.10 / 0.30) was
pre-registered 2026-08-04 *before* the S7 re-measurement, precisely so it could not be
fitted to the keeper. The measurement then came in at **vx 0.0139 / vy 0.0288** and the
band was **tightened to 1.5x afterwards**. That is not the pre-registration hazard, and
the distinction is directional: loosening a band after a result fails would be fitting the
gate to the candidate; tightening one after a result passes makes the gate strictly harder
and the incumbent still clears it (0.0288 vs 0.075 = 2.6x margin).

Two reasons for the tightening, one of them empirical:
- The widening existed to absorb plant, PD latency and DDS jitter, on the assumption those
  would inflate the bridge figure relative to sim. **The measurement falsified that** —
  post-fix the bridge (0.0139 / 0.0288) is indistinguishable from sim (0.020 / 0.025), and
  vx is better. Rate never justified widening either (F1: the bridge publishes at 1 kHz).
- What still argues against going to a flat 1:1 with sim is **n = 1**: one policy, one
  sequence, one plant, and no run-to-run spread established. 1.5x is headroom for that
  variance, not for a real degradation.

**Revisit to 1:1 (≤0.05) once several post-fix runs establish the spread** — at which
point the margin can be sized from measured variance instead of from caution.

**Reported, never gated:** leg action rate (**no valid reference until re-baselined on the
shipped scene**), transition overshoot vs A0, `trig_joint` rate + magnitude, whole-body
`act_rate` with its caveat, stride period, `ss_err`/`t90`, cmd-0 drift.

**Surfaced as UNKNOWN:** the leg action rate that is actually unsafe; the acceptable
`trig_joint` rate.

## S10. Test plan

| # | test | criterion |
|---|---|---|
| T1 | `pytest` before/after | **59/59** green (baseline verified 2026-08-04; the journal's "55/55" is stale) |
| T2 | new tests: provenance (md5 mismatch ⇒ NO-GO; reverse index names the run), dim check 94 vs 92, analyzer dt-from-meta, transition-window scoring, band-release detector | in `tests/`, CPU-only |
| T3 | `scripts/check_test_sensitivity.py` | every new test proven able to fail; the 2026-08-03 defects (parked ONNX, dt=0) re-introduced and caught |
| T4 | analyzer vs the 4 committed 2026-08-03 CSVs | no exception; A0 scores clean; both HRL runs report their falls |
| T5 | height anchor | cmd-0 bridge stand, residual ≤ 5 mm |
| T6 | estimator reference | bridge re-measure against the S7 pre-registered reading |
| T7 | full pipeline on the keeper | GO/NO-GO end to end; A0 control arm runs and caches |

Smokes use `WANDB_MODE=disabled`; local logs deleted after.

## S11. Sequencing

1. S5 analyzer/logger fixes + S7 reference fix (unblock measurement)
2. S1 provenance gate + staging/manifest (highest priority, standalone value)
3. S6 height anchor + T5
4. **F6 rename**: correct the "vendor" misnomer in `h1_2_handless.xml`'s header,
   `deployment.md`, `A1a_deploy_plan.md` W2; zero the shipped scene's floating base
5. S4 sequence + transition scoring + band guard + A0 cache + drift metric; S8 reporting
6. S0 orchestrator + T7
7. Report to the `A1a_deploy_plan.md` gate ladder + a WL-B line in `worklines.md`;
   narrative and dated results to the journal via `/sync-docs`; correct the four
   retractions (F1 in `hrl-infra.md` + journal; F2's provisional-status reason; F4's
   auto-memory; F6's misnomer across three files).
