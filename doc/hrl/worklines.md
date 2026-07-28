# Worklines plan (started 2026-07-16, grilled) — parallel tracks + delegation

This chat (the planning chat) owns this file, sequencing, and hand-offs; implementation
runs in delegated chats/agents (see `.claude/skills/delegate`).

**This file is the registry**: current status + who owns what + delegation history.
Full evidence/numbers/mechanism detail for each workline lives in its canonical doc —
`A1a_deploy_plan.md` for WL-B, `A1a_plan.md` for WL-D, the A1 findings ledger (research KB) for WL-C,
ADR-0005 for WL-E — this file summarizes and points there, it does not restate.

## Decisions locked (2026-07-16 grill)

1. **Sequencing: WL-C verification gates the WL-D batch.** The HIRO-relabel hold-bias
   suspect gets verified FIRST (cheap: code analysis + one relabel-off control run).
   If verification drags past ~2 days, WL-D launches anyway, downgraded to
   lever-validation probes. (Resolved 2026-07-16 — see WL-C verdict below.)
2. **WL-D levers**: `ll_cadence_coef` sweep; `hl_cot_coef` refinement; stand-still
   gating term; A0 gait terms mirrored into the LL intrinsic one at a time; lower-body
   posture/ankle-roll anchor; heel-to-toe roll-over/push-off (research-first). All
   implemented and run — see `A1a_plan.md` "WL-D batch results" / "Arm 6" / "Arm 10".
3. **Budget: full 10k every arm** (10001 iters, 4096 envs, seed 42, `model_10000`),
   R-round protocol: aggregate bench + `--eval-cmd-vx` holds + `[HOLDDIAG]` + goal
   probe + replay look. Naming per the deviation-tagging convention.
4. **Delegation: `/delegate` skill** (hand-off prompt generator + model picker) plus
   background subagents from the planning chat for small verifiable tasks.

## Worklines status

| WL | What | Owner chat | State |
|---|---|---|---|
| **A** | Planning, sequencing, doc/gate ownership, hand-offs | THIS chat | live |
| **B** | Deploy pipeline (plan: `A1a_deploy_plan.md`). Split 2026-07-20 into **B0 = A0-first hardware (PRIORITY)** and **B1 = A1 keeper** — same shared `deploy/` tree, one bridge, one flight recorder; kept inside WL-B rather than a new workline for exactly that reason. Both tracks' full gate ladders, defect history, and hardware-session logs live in `A1a_deploy_plan.md`; only a rolled-up status is kept here. | WL-B chat(s); ONE deploy chat at a time (shared `deploy/` tree) | **B0 (A0-first):** first real-hardware session 2026-07-23 found two separate causes of the "drunk stumbling" and disentangled them. (1) The safety-filter whole-body joint-limit hold arms on real hardware (fires on overshoots as small as 0.002 rad) — the same defect fixed for A1 on 2026-07-16, never ported to A0. **Ported + hardware-confirmed 2026-07-23 (WL-B0a)**: 225.8 s of continuous free walking, filter never engaged once. (2) A persistent ~3-5° backward stand lean, independent of the filter — **decomposed 2026-07-23/24 (WL-B0b) into two near-even causes** via three independent pitch estimates (IMU, kinematics, statics): a constant encoder→attitude map error and excess knee droop from unmodelled leg mass. **BOTH NOW DIAGNOSED; no cause remains unidentified.** (a) The **2026-07-28 inclinometer session vindicated the IMU by direct measurement** (pelvis −6.80° vs IMU −6.30°, agreeing to 0.50°, while the encoder feet-flat FK said −3.16°; torso and pelvis read identically, killing the pelvis-vs-torso question physically). Assumption-free core: with true attitude + measured encoders the model puts the foot **2.74° toe-up while it is provably flat on a flat floor**. That splits into **~0.9° foot-sole wedge** (both soles measured thicker at the FRONT → both feet toe-up 0.74°/1.11°, tilting the body backward; the flat capsule sole cannot represent it) + **~1.3° leg-encoder zero error** (hardware re-zero, no retrain). Refined model over 2 sessions: `p_IMU = 1.132*p_kin − 0.0394` (r=0.973). **Do NOT recalibrate the IMU** — it is the honest sensor. (b) Leg mass **resolved 2026-07-24**: real leg 18.60 kg vs the model's 15.40 kg. Model fix (leg mass + sole geometry) staged in `docs/adr/0006`, held so it does not rebase the plant mid-WL-D-batch. Also resolved: the E-stop out-of-process-kill question (LAN-cable pull verified on hardware). **Open for B0:** the leg-joint zero-point re-calibration + re-measured stand (free, next action — expect residual −0.048 → ~−0.016 rad, the wedge floor); the **L/R leg asymmetry** (wedge checked, does NOT explain it: differential 0.375°, wrong sign); a second G3.3 walk + the user's qualitative sign-off. Full detail → `A1a_deploy_plan.md`. **B1 (A1 keeper):** candidate LOCKED = `arm4d` (`ll_energy_coef=0.05`) — passes the live bridge (2026-07-22, all directions) and **replicates on a second seed (2026-07-24, WL-D combo batch)**; D2/old keeper/arm3/arm4c/arm5 all ruled out (twitch, not clearance — see WL-D row). **Open for B1:** full live G2 gate battery on arm4d (not yet run). Full detail → `A1a_deploy_plan.md`. |
| **C** | HL held-command fix: verify HIRO-relabel hold-bias suspect, fix if real | closed | **CLOSED 2026-07-16 — FALSIFIED.** See verdict below; full evidence → the A1 findings ledger (research KB). |
| **D** | Reward/gait lever batch (energy, gait quality, symmetry) | WL-D chat(s) | **Batch winner: `arm4d` (`ll_energy_coef=0.05`)** — CoT −48%, calmest A1 in the batch, now the B1 deploy candidate (see WL-B row — validated live 2026-07-22). S4′'s honest disconfirmer fired: A0+energy matches the energy saving without regressing tracking, so "the hierarchy buys energy" is unsupported by this batch. Arms 6 (heel-toe push-off, 2 formulations + a sign-bug found/fixed) and 10 (L/R gait symmetry, 2 formulations) shipped and trained; neither clears the winner bar cleanly (tracking cost / no measurable effect). **Arm 6 update (2026-07-23): a second, cross-formulation ankle-roll (inversion/eversion) confound was found alongside BOTH formulations' pitch motion (worst case ratio 1.78, neither formulation reads/rewards ankle_roll at all - a whole-body-policy side-channel, not a reward-shape bug) - an `ll_posture_anchor_ankle_roll=True` anchor test on formulation A (coef=0.1) killed it cleanly (roll/pitch ratio 1.59→0.006) while barely touching the pitch motion (+6.3°→+5.8°) and improving tracking/CoT too. Current best arm-6 candidate: formulation A + coef=0.1 + ankle-roll anchor; visibility on replay not yet re-checked, and the anchor not yet tested against formulation B's own (worse) confound.** Full batch tables (g)/(h)/(i) + Arm 6/Arm 10 sagas → `A1a_plan.md`. **Combo batch CLOSED 2026-07-24** (9 combination arms + arm4d seed-2, one consolidated table in `A1a_plan.md` "WL-D combo batch"): **arm4d REPLICATES on seed 123** (CoT 0.489 vs 0.462, act 0.934 vs 0.923, vx 0.060 vs 0.063 — all inside noise; the deploy candidate and the energy claim no longer rest on one seed, and the 2-seed spread is now measured at ±5.8% CoT / ±1.2% act). **No combo beats arm4d**; combinations were non-additive in all 9 cases. Only genuinely promising thread: **combo1 (`ll_action_rate_coef` 0.02→0.05) cuts act 0.923→0.837**, the lowest of any A1 and ~8x the seed spread, but still ~30% above A0's 0.644 — blocker narrowed, NOT closed; costs +12% CoT and lateral tracking, and it made `ub_arm_vel` slightly WORSE (0.300→0.320). Energy 0.10 (−7.1% CoT) is within the ±5.8% seed spread → NOT resolvable on n=1 (guard passed though: entrainment intact, cadence_rew 0.4486). **combo6 (all six A0 LL rewards mirrored at A0's weights) falsifies "copy A0's rewards to get A0's smoothness"**: act 0.951 (no better than arm4d) while probe HL/LL err vx tripled to 0.181/0.225 and holds fell below A0 — A1's action-rate gap is not a missing-reward-terms problem. Two follow-up findings (combos 7-9): **combo7 (arm4d + `ll_stand_still_coef`) fixes zero-command stepping** (touchdowns 142→128 floor, phantom goal halved) at ~zero deploy cost — arm4d that also stands cleanly, a candidate to seed+bridge; and **footclear CAUSES stepping-in-place** (combos 8/9 step 234/340 vs 128 floor, not rescued by stand_still) → doubly ruled out. Next probe: `ll_action_rate_coef` sweep past 0.05 (0.08/0.12) on the arm4d base at ≥2 seeds, plus `ll_joint_acc_coef` (new, implemented 2026-07-24) aimed at the arms-heavy action-rate floor. |
| **E** | Training-nominal realism + IsaacGym parity audit | closed | **CLOSED 2026-07-17.** See verdict below; full evidence → ADR-0005 Amendment 2. |

## WL-C verdict (2026-07-16): FALSIFIED — the held-command "HL hold degeneracy" never existed

Full evidence + numbers → the A1 findings ledger (research KB) row "WL-C: the held-command 'HL hold
degeneracy' is an EVAL-TOOLING ARTIFACT". Summary:

1. **HIRO relabeling is NOT the cause** (near-no-op: `relabel_frac` 0.03–0.09 across
   iters 1000–10000; instrumented `_relabel` on real keeper windows shows it does not
   shrink `|g|` (0.652→0.621) and preserves the HL's pull (`corr(g_vx, cmd−achieved)`
   0.827→0.798)).
2. **The real cause: the hold eval itself.** `play.py:297-299` pins the hold command by
   collapsing the twist ranges to a point; `goal_space.py:79-83` derives the HIRO goal
   scale from those same live ranges → scale hits the `1e-3` floor → `V* = s_t + 1e-3·g`
   → the A1 goal channel is **inert in every `--eval-cmd-vx` run**. A0 has no goal
   space, hence "isolated to the hierarchy". Restoring the training scale on the
   unchanged keeper: ss@0.5 0.304→**0.044**, ss@1.0 0.882→**0.097** (A0-optB-rs20:
   0.055/0.077). **A1 holds as well as A0. There is no HL hold degeneracy.**
3. Every pre-2026-07-16 A1 hold number in `A1a_plan.md` is void; the "sustained-command
   under-tracking = deploy risk" read is void.

**F2 SHIPPED 2026-07-16.** The goal scale is now baked into the checkpoint at `save()`
(like the normalizer) and pinned at `load()`, instead of re-derived from the replay
env's command ranges — 75 lines, 3 files (`goal_space.py`, `hrl_runner.py`, `play.py`).
Pre-2026-07-16 checkpoints get a `play.py` absence-shim (prints `[SHIM] ...`). The
deployed keeper ONNX pair was re-exported with the metadata.

**Deploy-side consequence (the user, 2026-07-16): `deploy.yaml`'s command ranges drove
two unrelated things** — the joystick clamp (a safety limit, should follow the operator)
and the HIRO goal-scale decode (trained policy semantics, should NOT follow the
operator). Narrowing the range for safety (e.g. `-0.25..0.5`) was silently halving the
A1 goal scale too — the deploy twin of the sim bug F2 fixed. **Shipped**:
`State_RLHRL::ensure_models_loaded` now pins the goal scale from HL ONNX metadata via
`hrl::GoalSpace::freeze_scale` before any range is read, so `deploy.yaml`'s ranges are
now a pure operator safety clamp (log line `[HRL] goal scale pinned from ONNX metadata
[...]` at FSM entry). Two carve-outs: legacy pre-2026-07-16 ONNX (no metadata) fall back
to the old range-derived path with a loud warning; `hl_target_mode: absolute` still
derives `center()` from the ranges (deliberately — the ONNX `goal_center` is in the
training frame, adopting it would inject a 0.29 m height offset; irrelevant to the
deployed `delta` mode).

**Relabel-off control** (`2026-07-16_11-06-48_..._norelabel_s42`, W&B `vg8x3gjf`,
keeper-identical except `relabeling=none`), `model_10000`, 64×600×2, holds with training
scale restored:

| run | agg err_vx | err_vy | err_yaw | fall | act_rate | CoT | stride | ss@0.5 (t90) | ss@1.0 (t90) |
|---|---|---|---|---|---|---|---|---|---|
| keeper (hiro) | 0.081 | 0.077 | 0.234 | 0 | 1.20 | 0.815 | 0.615 | 0.044 (0.56s) | 0.097 (0.76s) |
| control (none) | 0.065 | 0.070 | 0.150 | 0 | 1.09 | 0.933 | 0.357 | 0.030 (0.35s) | 0.069 (0.70s) |
| A0-optB-rs20 | 0.085 | — | — | 0 | — | 0.537 | 0.592 | 0.055 (0.6s) | 0.077 (0.9s) |

1 seed each — do not call relabel-off "better". Load-bearing point: both hold fine (≈A0),
confirming holds were never broken by relabeling. `relabel_frac` ≈0.03–0.09 means HIRO
relabeling is close to a no-op here — reportable, but dropping it is a structure change
and the user's call, not WL-C's.

## WL-E verdict (2026-07-17): friction does NOT stall at kl 0.01; foot drag is training-side

Full evidence → ADR-0005 Amendment 2. Summary:

1. **Defect quantified**: A0-optB swing apex is plant-independent (0.061–0.068 m across
   vendor/harsh/training-nominal plants) and ~35% under the trained 0.10 m target — a
   reward-shaping gap (WL-D territory), not a plant fix.
2. **Friction control** (`a0_v2_optB_fric0p1_kl01_s42`, 10k/4096/s42): converges from
   scratch (mild ~1.5–2x slower lift-off, no stall), bench err_vx/vy/yaw
   0.083/0.108/0.087, 0 falls, CoT 0.550 — at least baseline quality. **ADR-0005's "fric
   stalls" attribution is FALSIFIED** (it was the kl-0.005 + torso/arm-hold confound,
   pre-kl-0.01).
3. **Parameter audit** (mjlab vs unitree_rl_gym vs CorelLab): all references train
   friction-free too; top-3 transfer-gap suspects (each falsifiable, folded into WL-D as
   arms 7–9): (i) implicit servo vs explicit discrete-PD actuation, (ii) perturbation-DR
   strength, (iii) obs-noise temporal structure (white vs correlated).
4. **RULED (2026-07-17, the user)**: nominal stays friction 0; friction 0.1 stays a
   sanctioned option for a later rebase window. Audit follow-ups ran as WL-D arms 7–9
   (`A1a_plan.md` table (i): arm7 explicit-PD is a hard training failure/needs a config
   fix and retrain; arm8 wide-DR regresses tracking in sim, needs a bridge-replica
   verdict; arm9 correlated-noise is free/within noise).

## Hand-off prompt: WL-B0a — A0 safety-filter clamp port + FixStand pose alignment (✅ DONE 2026-07-23, kept for the record)

Model: **opus/fable class.** The port is mechanical (proven code exists), but it is the safety
path on a real robot with people beside it — a wrong clamp index or a broken tilt/fall path is
exactly the "silently wrong answer poisons downstream work" case. Paste into a fresh chat:

```
Port the per-joint command clamp from State_RLHRL to State_RLBase (A0), and align the FixStand
hold pose with the policy nominal. Spec-first per CLAUDE.md: present the diff for approval
BEFORE it goes near hardware. This is safety-path code.

Read first: doc/hrl/A1a_deploy_plan.md sections "ROOT CAUSE OF THE STUMBLING (2026-07-23)"
(evidence + exact defect) and the "Bridge post-mortem" Issue 3 entry (the 2026-07-16 A1
diagnosis + the mechanism correction explaining WHY the chasing hold degenerates to
damping-only torque). Do NOT re-derive any of it.

Settled context (do not re-litigate):
- A0's filter (deploy/robots/h1_2/src/State_RLBase.cpp:101-144) ramps alpha to 1.0 and holds
  ALL 27 joints when ANY policy-controlled joint leaves h1_2_limits.h. On hardware it fires on
  overshoots as small as 0.002 rad (0.1 deg). Measured: trig_joint on 4.56% of rows, alpha
  reaching 1.00, trig_tilt and trig_fall both ZERO.
- The limits are CORRECT — sim and deploy agree exactly. Not a limits-table bug.
- The identical defect was fixed for A1 on 2026-07-16 and scoped "State_RLHRL only, A0
  untouched". You are PORTING that existing fix, not designing a new one.
- ankle_roll (+-0.2618 rad = +-15 deg) has the least margin and dominates the triggers.

Tasks, in order:
1. Read the State_RLHRL clamp and write a short spec of the port: joint violations clamp the
   offending joint's COMMAND to its limit (firmware-like); tilt and fall KEEP the ramped
   whole-body hold; trig_joint in the flight log changes meaning to "clamp active". State
   explicitly what is NOT changed. STOP for approval.
2. Implement robot-locally in deploy/robots/h1_2/ ONLY — never deploy/include/isaaclab/
   (standing rule). Build h1_2_ctrl clean.
3. FixStand alignment (smaller, second): config/config.yaml FixStand qs holds hip_pitch -0.3 /
   ankle_pitch -0.2 while the policy default_joint_pos is hip_pitch -0.2 / ankle_pitch -0.3 —
   0.1 rad on both, opposite directions. **This is NOT the cause of the backward lean** (that
   is a separate workline) and must not be presented as one; it only cleans up the takeover
   transient. Propose aligned values, get approval before changing a pose used on hardware.
4. Verify: scripts/bridge_replica.py both scenes at 0/2/4 ms, then a live-bridge G2.1-G2.3
   re-run. Objective pass criteria:
   - joint-only violations produce a CLAMP, never alpha>0 (assert alpha stays 0 while
     trig_joint is active and tilt/fall are clear);
   - tilt and fall STILL produce the ramped whole-body hold (test deliberately);
   - G2.2 parity holds vs the 2026-07-22 baseline: achieved vx within 0.010 m/s of mjlab at
     0.3/0.5/1.0, 0 falls;
   - fixtures deploy.yaml.w1_legacy_test and deploy.yaml.g2_4_split_test still load. NOTE:
     every fixture must exist in BOTH velocity/v0/params and velocity_hrl/v0/params or
     h1_2_ctrl dies at startup with yaml-cpp BadFile (see .claude/docs/deployment.md).
5. Write the exact key script for the user's hardware re-run. Report into A1a_deploy_plan.md
   (A0-first track, house style, honest reads) + sync the WL-B row in doc/hrl/worklines.md.

Do not touch: training code (src/), reward terms, td3.py, the A1 deploy config
(velocity_hrl/v0), and the backward-lean question (separate workline — if you see lean
evidence, record it, do not chase it).
```

## Hand-off prompt: WL-B0b — the real-robot backward lean (CLOSED 2026-07-28, kept for provenance)

**DELIVERED.** Result: the lean is a near-even sum of TWO causes (a −0.031 rad
encoder→attitude map error, and excess knee droop under a ~22% higher standing load).
Full hypothesis table, arithmetic, and the closing hardware procedure live in
`A1a_deploy_plan.md` → "THE BACKWARD LEAN, DECOMPOSED". The prompt below is kept only as
provenance for what was asked.

Model: **opus/fable class.** Cross-system mechanism diagnosis (training model <-> deploy <->
hardware), several live hypotheses, no cheap oracle. **the user explicitly wants the reasoning and
the discriminating evidence, not just a verdict** — the prompt enforces that. Paste into a
fresh chat:

```
Investigate why the real H1-2 stands and walks with a persistent BACKWARD pitch lean that does
not exist in simulation. This is an INVESTIGATION, not an implementation task. Spec-first per
CLAUDE.md; change no constants without approval.

**Reporting requirement (explicit, from the user): do NOT return only a verdict.** For EVERY
hypothesis you consider, report: the mechanism, the discriminating measurement, the NUMBER you
measured, and whether it survived. Include the hypotheses you REJECTED and why — those are as
valuable as the surviving one. Show your arithmetic wherever magnitudes decide the argument.

Read first: doc/hrl/A1a_deploy_plan.md sections "FIRST REAL-HARDWARE RUN (2026-07-23)",
"Head-off run + head-mass hypothesis FALSIFIED", and "ROOT CAUSE OF THE STUMBLING" (the filter
defect — SEPARATELY delegated, do not work on it).

THE OBSERVATION: stand pitch -0.0608 / -0.0826 / -0.0913 rad across three real runs (3.5-5.2
deg BACKWARD, confirmed visually by the user) vs -0.0003 in the bridge. Walk pitch -0.0535 to
-0.0654 vs +0.0014. Oscillation amplitude about the offset MATCHES sim (std ~0.016), so this
is a DC posture bias, not an instability. The user's behavioural report fits: forward walking and
turning stable, backward and sideways worse, robot takes backward stabilisation steps.

ALREADY RULED OUT — do not re-derive (you MAY re-open with better evidence):
- **Head mass: FALSIFIED.** Head is unmodelled (collision sphere, no inertial — in ours AND
  every reference model, all ~67 kg). But head-OFF has the LARGEST lean (-0.0913), and 2 kg at
  x=+0.05 m shifts CoM ~1.4 mm where 3.5 deg needs ~55 mm. Removing the head fixed HARNESS
  INTERFERENCE (walk time 34/12 s -> 97 s), not the lean.
- **FixStand pose mismatch: WITHDRAWN.** FixStand's qs stop commanding once the policy takes
  over, so it cannot cause a sustained lean (the user's argument; it is correct).
- **Gait clock: WORKING** — phase alive exactly when |cmd|>=0.1 on hardware.
- **Filter triggers: INDEPENDENT** — run 1 shows the full lean with ZERO filter engagement.
- **Static IMU bias: WEAK** — read_all_joints static mean pitch +0.0119 rad (+0.68 deg), wrong
  size AND wrong sign.

DATA (on disk; no new hardware session needed to start):
- logs/deploy_safety/2026-07-23_{10-39-09,10-49-28,11-39-01}.csv (real; 11-39-01 = head off)
- logs/deploy_safety/2026-07-22_14-14-30.csv (bridge, worked — the comparison baseline)
- ~/ramlab_ws/trajectories/all_joints_2026-07-23_10-38-55.csv (IMU rpy/gyro/acc + joints;
  vel_*/body_height are all-zero — the known E1 structural gap, not a bug to chase)

CANDIDATE HYPOTHESES (starting set, not exhaustive — add your own):
1. **IMU frame / mounting-orientation offset. STRONGEST remaining, and note E2 CANNOT have
   caught it**: E2 validated a_world_z = R*a_imu.z - 9.81 ~ 0, which is insensitive to a small
   PITCH mounting error (cos 4 deg = 0.998). A 3-5 deg IMU pitch offset biases
   projected_gravity — an actual A0 observation — and the policy would hold a compensating
   lean as its equilibrium. Decisive test: robot physically level (spirit level on the
   pelvis), read imu_rpy_p and the deploy projected_gravity, compare against the sim IMU site
   (h1_2.xml: site name="imu" pos="-0.04452 -0.01891 0.27756" in torso_link) — position AND
   orientation convention.
2. **Mass / CoM mismatch beyond the head.** Model totals 66.98 kg while the project's own CoT
   normalization assumes 75 kg (mdp/rewards.py mass_g = 75.0*9.81) — an ~8 kg unexplained gap
   worth resolving regardless. base_com DR was only +-5 cm.
3. **Ankle compliance under static load** — check whether the sagittal offset is consistent
   with real ankle deflection; real vs sim ankle_pitch tracking already differs.
4. **Attitude-estimator convention** between the robot quaternion and what training assumed.

REQUIRED OUTPUT:
- The hypothesis table (mechanism / test / measured number / survived?).
- A ranked verdict naming the single best-supported cause, AND what would falsify it.
- If a hardware measurement is needed, the EXACT procedure (what to level, what to log, how
  long) so the user can run it in one short session.
- Whether the lean should affect A1 too (A1's LL consumes the same projected_gravity), i.e.
  does fixing it help both architectures.
- Report into doc/hrl/A1a_deploy_plan.md as a new subsection + sync the WL-B row.

Do not touch: the safety-filter clamp port (separately delegated), training code, reward
terms, td3.py.
```

## Completed hand-off prompts (archived — full prompt text superseded by results in canonical docs)

Kept as a delegation-history record (what was asked, who ran it, where it landed), not as
task specs to re-run. If a follow-up round is needed, regenerate via `/delegate` from
current doc state rather than reusing this text — the canonical docs have moved on.

| Hand-off | Launched | Result location |
|---|---|---|
| WL-C verification (HIRO-relabel suspect vs `td3.py`) | 2026-07-16 | verdict above; the A1 findings ledger (research KB) |
| WL-D reward/gait lever batch (arms 1–9, 15 runs) | 2026-07-17 | `A1a_plan.md` "WL-D batch results" tables (g)/(h)/(i) |
| WL-D arm 6 (heel-toe roll-over/push-off, probe-first, both formulations) | 2026-07-17 | `A1a_plan.md` "Arm 6" |
| WL-D arm 10 (L/R gait symmetry, probe-first, both formulations) | 2026-07-20 | `A1a_plan.md` "Arm 10" |
| WL-D combo batch (9 combination arms + arm4d seed-2) + action-rate decomposition | 2026-07-23/24 | `A1a_plan.md` "WL-D combo batch" + "Action-rate decomposition"/"Stepping-in-place"/"root cause" |
| WL-E (training-nominal realism + IsaacGym parity audit) | 2026-07-16 | verdict above; ADR-0005 Amendment 2 |
| WL-B0 (A0-first hardware track, items 1–5: joint-graze audit, `h1_2_limits.h` fix, G2 pre-checks, arms recommendation, E1/E2 prep) | 2026-07-16 | `A1a_deploy_plan.md` "A0-first hardware track" |
| WL-B (original A1-keeper pipeline validation next-steps, pre-track-split) | 2026-07-16 | superseded by the B0/B1 split (2026-07-20); current status in the WL-B row above |
| WL-B0/B1 headless candidate sweep (replica bridge, lateral+yaw extension) | 2026-07-22 | `A1a_deploy_plan.md` "Headless candidate sweep (2026-07-22)" |
| WL-B0a (A0 safety-filter clamp port + FixStand pose alignment) | 2026-07-23 | `A1a_deploy_plan.md` "A0 CLAMP PORT"; hardware-confirmed (225.8 s free walk, filter never engaged), live G2.1-G2.3 cleared, new `scripts/bridge_session.py` |
| WL-B0b (real-robot backward lean, investigation) | 2026-07-23, closed 2026-07-28 | deploy journal in the research KB (`wiki/architectures/a1a-deploy-journal.md`); 8-hypothesis table, 7 falsified. **Closed by the 2026-07-28 inclinometer session**: IMU vindicated (−6.80° measured vs −6.30° reported), error split ~0.9° foot-sole wedge + ~1.3° encoder zeros, and leg mass measured (18.60 kg/leg). Fixes: hardware re-zero (free) + `docs/adr/0006` Model v3 |

## Standing rule for the planning chat (WL-A)

When this chat catches itself deep in implementation (multi-file edits, debugging
loops, running trainings), stop and emit a hand-off instead: that work is usually too
easy for the expensive model and burns the planning context. Small, objectively
verifiable one-offs may go to background subagents (sonnet) directly.
