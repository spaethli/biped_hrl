# Worklines plan (started 2026-07-16, grilled) — parallel tracks + delegation

This chat (the planning chat) owns this file, sequencing, and hand-offs; implementation
runs in delegated chats/agents (see `.claude/skills/delegate`).

**This file is the registry**: current status + who owns what + delegation history.
Full evidence/numbers/mechanism detail for each workline lives in its canonical doc —
`A1a_deploy_plan.md` for WL-B, `A1a_plan.md` for WL-D, the A1 findings ledger (research KB) for WL-C,
ADR-0005 for WL-E — this file summarizes and points there, it does not restate.

> **Closed verdicts and delivered hand-off prompts moved out (2026-07-28).** The full
> WL-C and WL-E verdicts, and the WL-B0a/WL-B0b hand-off prompts kept for provenance,
> now live in the author's research knowledge base. This file is the live registry:
> who owns what, what is in flight, and the standing rules.

## Decisions locked (2026-07-16 grill)

1. **Sequencing: WL-C verification gates the WL-D batch.** The HIRO-relabel hold-bias
   suspect gets verified FIRST (cheap: code analysis + one relabel-off control run).
   If verification drags past ~2 days, WL-D launches anyway, downgraded to
   lever-validation probes. (Resolved 2026-07-16 — WL-C verdict, research KB.)
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
| **C** | HL held-command fix: verify HIRO-relabel hold-bias suspect, fix if real | closed | **CLOSED 2026-07-16 — FALSIFIED.** FALSIFIED. Full verdict and evidence → the worklines archive + A1 findings ledger (research KB). |
| **D** | Reward/gait lever batch (energy, gait quality, symmetry) | WL-D chat(s) | **Batch winner: `arm4d` (`ll_energy_coef=0.05`)** — CoT −48%, calmest A1 in the batch, now the B1 deploy candidate (see WL-B row — validated live 2026-07-22). S4′'s honest disconfirmer fired: A0+energy matches the energy saving without regressing tracking, so "the hierarchy buys energy" is unsupported by this batch. Arms 6 (heel-toe push-off, 2 formulations + a sign-bug found/fixed) and 10 (L/R gait symmetry, 2 formulations) shipped and trained; neither clears the winner bar cleanly (tracking cost / no measurable effect). **Arm 6 update (2026-07-23): a second, cross-formulation ankle-roll (inversion/eversion) confound was found alongside BOTH formulations' pitch motion (worst case ratio 1.78, neither formulation reads/rewards ankle_roll at all - a whole-body-policy side-channel, not a reward-shape bug) - an `ll_posture_anchor_ankle_roll=True` anchor test on formulation A (coef=0.1) killed it cleanly (roll/pitch ratio 1.59→0.006) while barely touching the pitch motion (+6.3°→+5.8°) and improving tracking/CoT too. Current best arm-6 candidate: formulation A + coef=0.1 + ankle-roll anchor; visibility on replay not yet re-checked, and the anchor not yet tested against formulation B's own (worse) confound.** Full batch tables (g)/(h)/(i) + Arm 6/Arm 10 sagas → `A1a_plan.md`. **Combo batch CLOSED 2026-07-24** (9 combination arms + arm4d seed-2, one consolidated table in `A1a_plan.md` "WL-D combo batch"): **arm4d REPLICATES on seed 123** (CoT 0.489 vs 0.462, act 0.934 vs 0.923, vx 0.060 vs 0.063 — all inside noise; the deploy candidate and the energy claim no longer rest on one seed, and the 2-seed spread is now measured at ±5.8% CoT / ±1.2% act). **No combo beats arm4d**; combinations were non-additive in all 9 cases. Only genuinely promising thread: **combo1 (`ll_action_rate_coef` 0.02→0.05) cuts act 0.923→0.837**, the lowest of any A1 and ~8x the seed spread, but still ~30% above A0's 0.644 — blocker narrowed, NOT closed; costs +12% CoT and lateral tracking, and it made `ub_arm_vel` slightly WORSE (0.300→0.320). Energy 0.10 (−7.1% CoT) is within the ±5.8% seed spread → NOT resolvable on n=1 (guard passed though: entrainment intact, cadence_rew 0.4486). **combo6 (all six A0 LL rewards mirrored at A0's weights) falsifies "copy A0's rewards to get A0's smoothness"**: act 0.951 (no better than arm4d) while probe HL/LL err vx tripled to 0.181/0.225 and holds fell below A0 — A1's action-rate gap is not a missing-reward-terms problem. Two follow-up findings (combos 7-9): **combo7 (arm4d + `ll_stand_still_coef`) fixes zero-command stepping** (touchdowns 142→128 floor, phantom goal halved) at ~zero deploy cost — arm4d that also stands cleanly, a candidate to seed+bridge; and **footclear CAUSES stepping-in-place** (combos 8/9 step 234/340 vs 128 floor, not rescued by stand_still) → doubly ruled out. Next probe: `ll_action_rate_coef` sweep past 0.05 (0.08/0.12) on the arm4d base at ≥2 seeds, plus `ll_joint_acc_coef` (new, implemented 2026-07-24) aimed at the arms-heavy action-rate floor. |
| **E** | Training-nominal realism + IsaacGym parity audit | closed | **CLOSED 2026-07-17.** Full verdict → the worklines archive (research KB); decision → ADR-0005 Amendment 2. |

## Standing rule for the planning chat (WL-A)

When this chat catches itself deep in implementation (multi-file edits, debugging
loops, running trainings), stop and emit a hand-off instead: that work is usually too
easy for the expensive model and burns the planning context. Small, objectively
verifiable one-offs may go to background subagents (sonnet) directly.
