# A1a — HL enrichment plan (cost-of-transport via commanded cadence)

Operational staged plan for A1a, the "give the HL a real job" effort. Design rationale and the
locked feature spec live in `docs/adr/0004-a1a-energy-cadence-hl-enrichment.md`; domain terms
(Goal, Gait reference, Cost of transport) in `CONTEXT.md`; backlog context in
`doc/hrl/hierarchy_benefit_roadmap.md` (Track A). This file tracks execution + live status.

## Thesis hook
The HL earns its keep by choosing a **speed-dependent gait cadence** to minimize **cost of
transport**, an authority the flat `(vx,vy,wz)` command structurally cannot express. Claim:
A1a holds A0 tracking at lower CoT, where a flat `A0+energy` policy regresses tracking. Feeds
A2 (A-RMA supplies adaptivity) and A3 (gait library plugs into the same cadence channel).

## Design (locked; rationale in ADR-0004)
- **HL reward** = tracking + (−`hl_cot_coef`·CoT); `CoT = window energy / (actual walked
  distance + small floor)`, engaged when commanded linear speed > 0.1. HL-only; the LL stays a
  pure tracker (the decoupling claim). Power `P = Σ_j |qfrc_actuator_j · joint_vel_j|`.
- **Lever** = cadence: HL commands `period` (s) in `cadence_period_range`; the LL entrains via
  the phase-clock obs (`mdp.phase`) + the `feet_gait` term in the LL intrinsic
  (`ll_cadence_coef=0.5`). R2 routing: cadence is a *gait reference*, not a goal-space component.
- `period` = stride period (same-foot touchdown interval); 0.6 s ≈ 200 steps/min ≈ 2x human.

## Config additions (`config/h1_2_a1/rl_cfg.py`, all default to current A1)
`hl_cadence: bool=False`, `cadence_period_range: tuple=(0.5, 1.4)`, `ll_cadence_coef: float=0.5`,
`hl_cot_coef: float=0.0`. With the defaults the A1 code path is byte-identical to today.

## Prerequisite metrics (`play.py` benchmark `[BENCH]`)
`mech_power_w`, `cot`, `stride_period_s`. Added once; reused by every stage.

## Stages

> **Model-v1 notice (ADR-0005, 2026-07-07):** every run, checkpoint, baseline table, and
> CoT map in this section and in Results ran on the Model-v1 nominal plant. Per ADR-0005
> they are **historical/mechanism evidence only** — no thesis-scorable A1a result may be
> v1-based. The completion plan on Model v2 is the **Plan v2** section below.

| Stage | Action | Success criterion | Status |
|---|---|---|---|
| **M0** | Add `mech_power_w` / `cot` / `stride_period_s` to the benchmark (`play.py`) | metrics print, no NaN, A0 baseline captured | ✅ 2026-06-30 |
| **S0** | A0 baseline cadence + CoT; CoT-vs-period probe | baseline captured; CoT(period) curve **deferred to S2** (A0 can't vary cadence) | 🟡 baseline done |
| **S1a** | LL-side cadence machinery in the **training** loop (period/phase buffers, `mdp.phase` + `feet_gait` rewire, `ll_cadence` intrinsic, random-period source) | smoke: new path runs + `cadence_rew` active; defaults inert (`cadence_rew=0`, intrinsic unchanged) | ✅ 2026-06-30 |
| **S1b** | Cadence in the **inference** path (`get_inference_policy` + benchmark) + `--eval-cadence-period` pin for the CoT(period) sweep | smoke: cadence ckpt benchmarks via inference path, period pin applied, no error (entrainment accuracy is S2's, untrained LL stride 0.76 vs cmd 0.60) | ✅ 2026-06-30 |
| **S1c** | CoT HL reward + learned-cadence action dim (`goal_dim+1`) | phase continuity on mid-window period change; CoT gate/floor numerics; action dim grows only when on | ✅ 2026-07-02 (18/18 smoke + v2-ckpt compat; see S1c note) |
| **S2d** | **d(T) duty schedule** (`cadence_swing_time=0.31`, `cadence_duty_range=(0.56,0.70)`) + slow-band retrain, range (0.35, 1.3) | slow-end ceiling moves past 0.8 s; fast band unregressed; tracking + survival ≥ v2 | 🟡 code ✅ (7/7 smoke). **v5 (scratch + wide band) FAILED** (clock-ignoring walker; curriculum failure). **v6 (resume v2 + τ=0.31 + (0.35,1.3)) FAILED** — d(T) re-scored v2's locked 0.7–0.8 gait (floor departs at 0.70) + 30% unfollowable episodes → lock LOST above 0.5 s (see v6 entry). → **v7 staged (τ=0.32, band (0.35, 1.0)): ceiling UNMOVED** — 0.35–0.65 lock preserved (v6's damage fixed), but 0.8 follows worse than the v3 control (stride 0.580 vs 0.700) and the slow band stays closed. Stage 2 not triggered. **d(T) parked after 3 attempts** (see v7 entry); keeper = **v2 `model_5000`**, envelope (0.35, 1.0). **v8 repro control ✅ (implementation exonerated)**: exact v2 recipe on the post-S1c/d(T) code reproduces v2's entrainment (stride 0.343/0.488/0.627/0.691 at cmd 0.35–0.8 vs v2 0.343/0.486/0.620/0.719; match ≤0.95; 0 falls; CoT slightly better). Cells ≥0.8 are extrapolation for BOTH runs (trained band (0.5, 0.7)): v8's achieved ceiling landed at ~0.69 vs v2's ~0.78 (v3, trained on the band, still lost 1.0 at 0.532) — **the followable envelope/ceiling is a per-checkpoint property**; re-measure it (and the CoT map) before freezing any other LL for S3. v5/v6/v7 failures = training-setup effects, NOT code bugs. |
| **S2** | Train A1a LL to follow a commanded **stride period** | achieved `stride_period_s` tracks command; tracking + survival ≥ A1 | ✅ **entrained** (2026-07-02, post eval-bug fix): **v2 already follows 0.35→0.8 near-1:1** at fixed vx (err_vx ~0.06, 0 falls); CoT swings 1.11→0.60 with period → **GO**. The v2/v3/v4 "no entrainment" NO-GO verdicts were an **eval artifact** (`structure_keys` bug, see correction below); only v1's training collapse was real. |
| **S3** | Frozen-LL learned HL (TD3) + `hl_cot_coef>0` | HL drives `cot` below fixed-0.6 baseline at A1-level tracking; converged period = `CoT(period)` min, speed-dependent | 🟡 **mechanism GO, optimum NOT reached** (2026-07-03, `a1a_s3_cadence_hl`, frozen v2 + `hl_target_mode=absolute` + `hl_cot_coef=1.0`): CoT < fixed-0.6 at every speed (−8…−22%), 0 falls, gait_match 0.94, tracking held — but the HL sat on a near-natural **flat ~0.65 stride** (NOT the map's long-stride-at-low-speed optimum) and tracks vel worse than the oracle (err_vx 0.17–0.30 vs 0.045–0.205). Coef too weak vs tracking → the S5 lever. See Results. |
| **S4** | Co-trained A1a vs A0 / A0+energy / A1 (≥2 seeds) | A1a `cot` < A0 at A0-level tracking + fall_rate; A0+energy regresses tracking. Honest disconfirmer logged if A0+energy matches | 🔴 **co-trained learned-cadence FAILS** (2026-07-04, `a1a_s4_cotrain_cot1_*`, full arch: TD3+HIRO+delta+velobs). At `hl_cot_coef=1.0` the HL drifts to a short expensive cadence (~0.48), CoT **+30–47% ABOVE** its own fixed-0.6 baseline — ar 0.02 vs 0.05 identical (ar is NOT the lever). NB tracking is NOT regressed: err_vx ~0.3 at fixed-vx matches the A1 keeper (0.25) — the "5× regression" was a fixed-vx-vs-random-command benchmark artifact. A0/A0+energy controls not built (approach fails first). See Results + S5. From-scratch retry (2026-07-06/07, `expOH_cot{00,10}`): cot0 walks but is an energy hog with a self-narrowed band; cot1 = penalty domination again (see Results). **Redefined → Plan v2 stage S4′** (scored on Model v2; staged/frozen default via gate G). |
| **S5** | Sweep `hl_cot_coef` | fall_rate flat as weight rises; cap below any rise (suicide-attractor guard) | 🔴 **NEGATIVE — higher coef destabilizes, doesn't help** (2026-07-04, `a1a_s5_cot{2,4,8}_ar05`). Raising coef makes cadence *shorter* (0.30) not longer, **collapses entrainment** (gait_match 0.9→0.5, LL ignores clock), **degrades tracking** (eval err_vx→1.65, train err_vx~4 at coef 8), CoT never beats fixed-0.6. fall_rate stays 0 in eval (no suicide attractor) but ep_len/fell_over erode in train. **S3's frozen-LL win does NOT transfer to co-training at any coef** — penalty-domination-style collapse (ADR warned raw energy would; normalized CoT at high weight reintroduces it). See Results. **Redefined → Plan v2 stage F**: the coef sweep that answers S3's "coef too weak" diagnosis runs against a FROZEN LL (where entrainment cannot collapse), not co-trained. |
| **S6** | OOD proxy: Narrow→Wide DR / push / terrain (secondary) | exploratory; big edge not expected pre-A2. S4 is the load-bearing result | ⬜ unchanged, after S4′ |

## Plan v2 — completing S4+S5 on Model v2 (grilled 2026-07-07)

Decisions locked in the 2026-07-07 grill session (user), superseding the S4/S5 rows above:

- **Sequencing: Model v2 strictly first** (ADR-0005). No further v1 training runs at all;
  the pasted 4-arm cluster round re-launches on v2 after the validation gate. The
  mechanism findings above (kernel chain, penalty domination, band self-narrowing, the
  staged/frozen win) inform the v2 arms but are not re-scored.
- **S4 subject (gated):** the **staged/frozen A1a is the default** deliverable;
  co-training is promoted back only if an R3 coef holds tracking (bench err_vx ≈ R1's
  ~0.10) AND cuts CoT clearly below R1 toward the fixed-0.6 reference.
- **S4 claim (tiered):** primary = **within-architecture ΔCoT** (vs the same system on a
  fixed-0.6 clock, at equal tracking — the authority a flat command cannot express);
  stretch = absolute CoT ≤ A0-v2, scored only if the LL-efficiency gap closes en route.
- **S5 redefined = stage F:** the `hl_cot_coef` sweep runs against a **frozen** LL (where
  entrainment structurally cannot collapse; only HL tracking can erode) — the 2026-07-04
  co-trained sweep answered a different, failed question.
- **Frozen-LL target-mode rule** (S3 lesson): match the HL that trained the LL — `delta`
  for R2's LL (TD3-delta-trained), `absolute` for oracle-produced LLs.
- **ar reconciliation resolved by the rebase:** all v2 runs are from-scratch at the keeper
  `ll_action_rate_coef=0.02` + `ll_posture_coef=0.5` (incl. the 2026-07-07 hip yaw/roll
  anchor); the v2-LL's 0.05 launch override dies with the v1 lineage.
- Followable envelope + CoT(period,vx) map are **per-checkpoint properties**: re-measure
  on every v2 LL before freezing/scoring (v1 maps are historical).

| Stage | Action | Success criterion | Status |
|---|---|---|---|
| **P0** | Commit+push the working tree (full-exp kernel `goal_space.py`, hip yaw/roll posture anchor, launcher `HL_ALGO`/`HL_COT` knobs, ADR-0005, doc syncs) | cluster pulls a repo containing everything the arms need | ✅ 2026-07-07 (`c21ee2d`, incl. the Model v2 implementation) |
| **V2** | Implement Model v2 + run the ADR-0005 validation gate: smoke → A0-v2 retrain → tracking/falls within noise of v1-A0 → v1↔v2 cross-eval (thesis-reportable modeling gap) → v2-A0 ONNX through the C++ bridge. Re-capture the A0 baseline table (CoT/stride) as the new S4′ reference | all four gate checks pass | 🟡 checks 1–3 ✅ 2026-07-08 (final v2 config per `398e88a`: arm hold gains + derived scales, torso/legs v1, frictionloss 0 — narrower than the ADR draft). Gate 2: `a0_v2_baseline` `model_10000` within noise of v1-A0 (err_vx 0.097 vs 0.093, fall 0=0). Gate 3: v1-in-v2 cross-eval = tracking preserved, **+12% CoT / +14% power** (arm plant change only; v2-in-v1 direction moot, legs unchanged). New baseline table → Results. Check 4 ✅ 2026-07-08: v2 ONNX walks in the C++ bridge (arms held, gain/scale lockstep verified 27/27 vs both YAMLs); the observed w-from-stand fall + dirty gait are a **pre-existing bridge-plant mismatch** (`scene_h1_2.xml` joint defaults armature 0.1/frictionloss 0.2/damping 1 vs training nominal 0.025–0.002/0/0 — the policy passes the same vx=1.0-step-from-stand test in mjlab with fall_rate 0.0/128 eps), accepted as-is until deploy (user decision 2026-07-08; see `.claude/docs/deployment.md`). **V2 GATE CLOSED** |
| **R1** | cot0 config on v2 (from-scratch exp kernel, TD3+HIRO+delta+velobs, `source=hl`, `cot=0`), seeds 42+123 | walks ≈ v1-cot0 (err_vx ~0.10, 0 falls); the ≥2-seed co-train reference pair | ✅ 2026-07-09: both seeds walk (err_vx 0.107/0.113, 0 falls), short stride again (0.41/0.35); **seed spread is large on energy** (CoT 1.42 vs 1.79) — energy claims need both seeds. Own period is WORSE than pinned 0.6 (ΔCoT +8.7%): without CoT pressure the HL's cadence is a liability |
| **R2** | `source=random` co-train on v2 (band-open fix; doubles as the staged-lineage frozen-LL producer) | LL envelope covers ≥ (0.35, 0.8) on the fixed-vx GRID — band does NOT self-narrow to the HL's visited periods | ✅ 2026-07-09 **PASSES**: full-band entrainment (stride 0.346/0.490/0.614/0.738/0.827 at cmd 0.35→1.0, vx=0.5; match 0.72–0.86; 0 falls) — the random source fixes the self-narrowing. **Valid frozen-LL producer for F** (delta-trained → F uses delta) |
| **R3** | co-train `hl_cot_coef` 0.2 and 0.5, seed 42 | tracks (err_vx ≈ R1) AND CoT clearly < R1 toward fixed-0.6 → gate G promotes co-training | 🟡 2026-07-09 **half-pass at 0.2**: CoT 1.139 = clearly < R1 (−20/−36%) AND beats its own fixed-0.6 (**−7.6%**, the first co-trained ΔCoT win; v1's penalty domination is gone at this coef) — but tracking err_vx 0.146 vs R1's 0.107/0.113 (+0.035): NOT ≈ R1. 0.5: err_vx 0.203, abs CoT no better than R1 → 0.5 is past the boundary |
| **R4** | clean-kernel ablation (oracle HL, posture/ar off) | exp kernel alone suffices to walk from scratch | ✅ 2026-07-09: kernel alone gives the round's **best tracking** (err_vx 0.071, ll_err_vx 0.033) but a **wild, undeployable gait** (act_rate 33, power 2282 W, stride 0.13, match 0.52) → the kernel is load-bearing for from-scratch convergence; posture+ar are load-bearing for gait quality. Both halves confirmed |
| **G** | Gate on R3 (criterion above) | co-train promoted as S4′ subject, else staged/frozen default | ✅ 2026-07-09 strict criterion NOT met (R3-0.2 tracking regression) → staged/frozen default. **OVERRIDDEN by user 2026-07-13: co-training IS the A1a line** (the hierarchy is the architecture's point); tracking re-prioritized in A2 |
| **F** (=S5) | Freeze the best entrained v2-lineage LL (**R2's `model_10000`**, envelope validated) → re-measure its CoT(period,vx) map → S3-style HL retrain at `hl_cot_coef` ∈ {1, 2, 4}, **`hl_target_mode=delta`** (R2's LL is delta-trained) | HL leaves the flat stride and approaches the map's speed-dependent optimum at held tracking; cap the coef below any tracking erosion | ⏸️ **parked 2026-07-13** (user: co-train is the line; revisit only if the co-train deploy stalls) |
| **D** | **Real-robot deployment prep** on the co-train keeper (cot0.2 velgoal DIR): (1) calm the arm swing — `ub_arm_vel` 2.7–3.5× A0, lever = per-joint LL posture weights (`ll_posture_weights`, shoulders 16 / elbow+wrist 4); (2) gait shaping (deferred until a replay names a defect; menu: `ll_cadence_coef` → mirror an A0 term into the intrinsic → duty/swing last); then ONNX → C++ bridge → H1-2 (battery: dual-scene, parity, held-command walk ≥30 s, command steps, safety-envelope + safety-filter audit) | deployable walk: arms calm, gait clean; held-command direction must hold; tracking magnitude secondary until A2 | 🟡 2026-07-14/15: **arms ✅** (D1+D2, table f: pose_dev 10× down, arm_vel 0.35–0.41, replay-confirmed); **held commands ❌ — root-caused to a velocity-hold HL degeneracy** (table f reads; duration/heading/zero-point/dither all falsified). Next: verify the HIRO-relabeling suspect in `td3.py` before more training arms |
| **S4′** | Scored comparison, ≥2 seeds, `num_envs=4096` fixed: A1a (gate winner) vs **A0-v2** vs **A0+energy** vs **A1-v2** (cot0 config minus cadence/CoT). A0+energy = new per-step command-gated CoT-analog term `-w·P/(m g·max(‖v_cmd‖, ε))` in `mdp/rewards.py` (fair pressure: same normalization the HL feels); 3-coef mini-grid at 1 seed, best coef gets seed 2. Protocol: deterministic bench + fixed-vx GRID + goal probe | primary: ΔCoT < 0 vs own fixed-0.6 at equal tracking, while A0+energy fails to match the saving or regresses tracking (**honest disconfirmer logged if it matches**); stretch: CoT ≤ A0-v2 | ⬜ after G/F |

Parked/out of scope: d(T) stays parked (3 failed attempts, see S2d); S6 exploratory after S4′.

**R-round launched on the cluster 2026-07-08** (all 6 runs, post-gate: R1 seeds 42+123,
R2, R3 cot 0.2+0.5, R4).

**A0-v2 baseline (2026-07-08, `h1_2_velocity_v2/2026-07-08_07-06-18_a0_v2_baseline/model_10000.pt`,
64×600×2 seeds) — the S4′ reference:**

| eval | err_vx | err_vy | err_yaw | fall | act_rate | power (W) | CoT | stride (s) | match |
|---|---|---|---|---|---|---|---|---|---|
| aggregate | 0.097 | 0.118 | 0.098 | 0.0 | 0.61 | 162 ± 6 | **0.532 ± 0.011** | 0.585 ± 0.003 | 0.946 |
| fixed vx=0.5 | 0.083 | 0.085 | 0.093 | 0.0 | 0.64 | 172 | **0.524** | 0.573 | 0.964 |
| v1-A0 ckpt in v2 plant (cross-eval) | 0.097 | 0.113 | 0.087 | 0.0 | 0.66 | 186 ± 5 | 0.595 | 0.583 | 0.948 |

Reads: (i) gate 2 passes — v2-A0 matches v1-A0 tracking/falls (v1 table above: 0.093/0.112/
0.090/0.0) with the same ~0.58 stride. (ii) The v1→v2 modeling gap for A0 is **energy-only**:
the v1 checkpoint under the v2 plant keeps tracking but pays +12% CoT / +14% power (final v2
kept legs + friction at v1, so only the upper-body gains/scales moved — consistent). (iii)
v2-A0 is *more* efficient than v1-A0's fixed-0.5 CoT (0.524 vs 0.577) — the absolute-CoT
stretch target for A1a moved down, not up.

**R-round results (2026-07-09, `h1_2_velocity_a1_v2/2026-07-08_*`, all `model_10000`,
64×600×2 seeds).** Aggregate bench (A0-v2 row above is the reference):

> ⚠ **2026-07-16: the `err_yaw` column is biased HIGH for these A1 rows** (~17%, the size of
> the keeper's 0.234→0.195 re-measure). Pre-F2 the aggregate bench ran `scale_yaw` 0.5 vs the
> trained 1.0, so the HL's yaw goals were halved. `err_vx`/`err_vy`/CoT/stride/power/match are
> **unaffected** (those scales were already correct) — only yaw. The bias is uniform across A1
> rows so the within-A1 ranking holds; A1-vs-A0 yaw comparisons overstate A1's error. Re-run
> post-F2 to fix (no retrain). See the (f)/(e) boxes + `A1_findings.md` WL-C.

| run | err_vx | err_vy | err_yaw ⚠ | fall | act | power (W) | CoT | stride | match |
|---|---|---|---|---|---|---|---|---|---|
| R1 s42 (cot0) | 0.107 | 0.071 | 0.204 | 0.0 | 1.31 | 439 | 1.417 | 0.407 | 0.92 |
| R1 s123 (cot0) | 0.113 | 0.086 | 0.194 | 0.0 | 1.49 | 537 | 1.789 | 0.351 | 0.90 |
| R2 (random src) | 0.132 | 0.121 | 0.196 | 1e-5 | 1.27 | 354 | 1.198 | 0.624 | 0.85 |
| R3 cot0.2 | 0.146 | 0.130 | 0.175 | 0.0 | 1.23 | 340 | **1.139** | 0.551 | 0.90 |
| R3 cot0.5 | 0.203 | 0.130 | 0.226 | 0.0 | 1.43 | 511 | 1.406 | 0.647 | 0.91 |
| R4 kernel-only | **0.071** | 0.046 | 0.451 | 0.0 | 33.0 | 2282 | 6.76 | 0.131 | 0.52 |

Own-HL-period vs pinned `--eval-cadence-period 0.6` (same checkpoint — the within-arch ΔCoT):

| run | own CoT | fixed-0.6 CoT | ΔCoT | own err_vx | pinned err_vx |
|---|---|---|---|---|---|
| R1 s42 | 1.417 | 1.304 | +8.7% | 0.107 | 0.143 |
| R3 cot0.2 | 1.139 | 1.233 | **−7.6%** | 0.146 | 0.156 |
| R3 cot0.5 | 1.406 | 1.740 | −19.2% | 0.203 | 0.222 |

R2 fixed-vx=0.5 GRID (envelope check): stride 0.346/0.490/0.614/0.738/0.827 at cmd
0.35/0.5/0.65/0.8/1.0 (match 0.72–0.86, CoT 1.71→1.14, 0 falls) — **no self-narrowing**;
achieved ceiling ~0.83.

Probe: every co-trained run is per-window sloppy vs the oracle (hl_err_vx 0.19–0.49,
ll_err_vx 0.32–0.46 vs R4's ll 0.033) — the learned-HL window errors, not the LL kernel,
bound tracking; consistent with the S3 finding that the HL is the wall. Reads: (i) R3-0.2 is
the **first co-trained CoT win** (beats own fixed-0.6 by −7.6% with pinned-vs-own tracking
equal) but gives up 0.035 err_vx vs R1 → gate G strict criterion not met; 0.5 is past the
boundary (tracking degrades, absolute CoT gain gone). (ii) The v1 penalty-domination cliff at
coef 1.0 has moved to a graded trade at 0.2–0.5 under the full-exp kernel — coef curve is
tame now, not catastrophic. (iii) R1 seed spread on energy (CoT 1.42/1.79) → any co-train
energy claim needs both seeds. (iv) All A1a CoT ≥ 1.14 vs A0-v2 0.53 — the absolute tier
stays out of reach pre-F, as expected.

**kl 0.01 + velocity-goals-only + directed-distance follow-up (2026-07-09/10, all
`model_10000`, 64×600×2 seeds; all A1a runs torso-200 so kl reads are plant-clean).**

*(a) kl lever + directed/undirected, full-goal HL (cluster `..._s42kl01[_unproj]`):*

| run (full-goal) | vx | vy | yaw | act | cot | stride | orient | height |
|---|---|---|---|---|---|---|---|---|
| R3a cot0.2 **kl005** dir | 0.146 | 0.130 | 0.175 | 1.23 | 1.139 | 0.551 | 0.282 | 0.257 |
| cot0.2 **kl01** dir | 0.063 | 0.065 | 0.429 | 1.33 | 1.233 | 0.371 | 0.094 | 0.045 |
| cot0.3 kl01 dir | 0.069 | 0.067 | 0.198 | 1.24 | 0.921 | 0.400 | 0.080 | 0.013 |
| cot0.5 kl01 dir | 0.077 | 0.086 | 0.322 | 1.39 | 0.902 | 0.413 | 0.091 | 0.188 |
| cot0.2 kl01 **UNDIR** | 0.069 | 0.072 | 0.350 | 1.18 | 0.830 | 0.434 | 0.070 | 0.022 |

**kl 0.005→0.01 is the dominant lever** (R3a→cot0.2-kl01, everything else fixed): err_vx
**0.146→0.063**, height_dev **0.257→0.045** — dwarfs the distance question; confirms the
optB "kl 0.01 required" call reaches A1a too.

*(b) directed vs undirected, velocity-goals-only local (`..._cot0p2_kl0p01[_undir]_s42`):*

| run (velgoal) | vx | vy | yaw | act | cot | stride | match | orient | height | ubdev |
|---|---|---|---|---|---|---|---|---|---|---|
| cot0.2 **DIR** | 0.080 | 0.076 | 0.221 | 1.18 | 0.812 | **0.625** | 0.93 | 0.050 | 0.006 | 0.033 |
| cot0.2 **UNDIR** | 0.061 | 0.078 | 0.150 | 1.14 | 0.855 | 0.353 | 0.95 | 0.035 | 0.005 | 0.022 |

Reads: (i) **velocity-goals-only fixes the posture sag** — height_dev **0.005–0.006** vs
full-goal 0.045–0.257; the biggest walk-quality gain in the set. (ii) Directed's robust
signature is **longer strides** (0.55–0.63 vs undirected 0.35–0.43, since it rewards
distance-along-command/joule); its tracking edge is inconsistent (wins vx by 0.006 full-goal,
loses by 0.019 velgoal). No sideways exploit in either UNDIR run (vy ~0.07) at cot 0.2.
**Decision (user, 2026-07-10): keep DIRECTED distance** — the longer stride is theoretically
the better-CoT gait, and the small tracking cost is acceptable. Follow-up: directed already
buys efficiency, so **lower the CoT coef** (the LL-side energy pressure) next rather than
raise it.

*(c) A0 plant, torso-v1(200) vs full-v2 optB(300) — the deploy-fidelity gain is free:*

| A0 run | vx | vy | yaw | act | cot | stride | orient | height |
|---|---|---|---|---|---|---|---|---|
| torso200 kl005 (`a0_v2_baseline`) | 0.097 | 0.118 | 0.098 | 0.61 | 0.532 | 0.585 | 0.035 | 0.024 |
| torso300 kl01 (`a0_v2_optB_baseline`) | 0.090 | 0.109 | 0.098 | 0.63 | 0.526 | 0.589 | 0.029 | 0.025 |

Torso 200→300 (true deploy hold gain) costs **nothing** in sim — all within noise. **DECIDED
2026-07-10 (ADR-0005 finalized): full-v2 optB (`a0_v2_optB_baseline` `model_10000`) is THE
default A0-v2 baseline** that all A1a/A2 work rebases on; torso-v1 200/2.5 retired. Constants
+ all 4 deploy YAMLs + A0/A1 `desired_kl=0.01` are in lockstep in the working tree.

**cot0.1 + fixed-0.8 clock control + fixed-command failure (2026-07-11/13, all `model_10000`,
64×600×2; local `h1_2_velocity_a1_v2/2026-07-11_*`).**

*(d) velgoal+kl01 aggregate bench (extends table b; fix0p8 = `source=random` pinned
`(0.8, 0.8)` via temporary rl_cfg edit — the tyro tuple CLI override is broken — `cot=0`,
so it is the same architecture minus learned cadence + CoT; cot0.1 W&B `tj8ft6me`):*

| run (velgoal) | vx | vy | yaw | act | power | cot | stride | match | height |
|---|---|---|---|---|---|---|---|---|---|
| cot0.1 DIR | 0.064 | 0.061 | 0.146 | 1.16 | 318 | 0.945 | 0.418 | 0.94 | 0.007 |
| cot0.2 DIR (table b) | 0.080 | 0.076 | 0.221 | 1.18 | — | 0.812 | 0.625 | 0.93 | 0.006 |
| fix0p8 control | 0.094 | 0.105 | 0.143 | 1.17 | **208** | **0.662** | 0.783 | 0.94 | 0.005 |

cot0.1 own-vs-pinned-0.6: CoT 0.945 vs 0.844 (**own +11.9% worse**, the R1-cot0 pattern):
coef 0.1 is too weak to hold the long stride (0.418 vs cot0.2's 0.625); tracking returns to
the kl01 level. **Coef axis mapped: 0.1 tracks-but-wastes, 0.2 = the trade sweet spot, 0.5
past the boundary.** Probe (cot0.1): hl_vx 0.142 / ll_vx 0.224, no saturation. Pinned-0.8 on
cot0.2 DIR: CoT 0.769 @ achieved stride 0.657 (band-limited follow, −5.3% vs own) — the
gradient pointing past 0.625 is what motivated the control.

*(e) fixed-command evals (`--eval-cmd-vx`, the treadmill/deploy case).*

> ⚠ **2026-07-16: the original (e) was 100% eval artifact and is REPLACED below.** Every
> number in it came from `--eval-cmd-vx`, which zeroed the A1 goal scale (see the (f) box +
> `A1_findings.md` WL-C). ~~old: fix0p8 0.674@0.5 / 1.106@1.0 · cot0.2 DIR 0.360 / 0.878 ·
> A0 0.127@1.0~~ — **there was no sustained-command failure.**

**Table (e) — RE-MEASURED post-F2 (2026-07-16), `model_10000`, 64×600×2, steady-state
`ss_err` (last 2/3):**

| run @ fixed cmd | ss vx@0.5 | ss vy@0.5 | ss vx@1.0 | ss vy@1.0 | agg vx | agg CoT | stride |
|---|---|---|---|---|---|---|---|
| fix0p8 (const 0.8 clock, cot0) | 0.048 | 0.077 | 0.078 | 0.087 | 0.094 | **0.663** | **0.781** |
| cot0.2 DIR (keeper) | 0.043 | 0.062 | 0.097 | 0.087 | 0.079 | 0.818 | 0.625 |
| cot0.1 | **0.036** | **0.038** | **0.044** | **0.040** | **0.063** | 0.948 | 0.418 |
| A0 optB rs20 | 0.057 | 0.084 | 0.076 | 0.082 | 0.083 | 0.537 | 0.591 |

Reads (rewritten 2026-07-16 — the old (e) reads are all void): (i) **fix0p8 walks sustained
commands fine** (0.048 / 0.078 ≈ A0's 0.057 / 0.076) and **the "sideways crab walk" was the
artifact too** (vy@0.5 0.270 → **0.077**): with the goal channel inert the HL could not correct
vy, so the robot drifted. fix0p8 is therefore a **valid best-constant-clock control after
all** — and it keeps its CoT win (**0.663, the best of any A1**, −19% vs the keeper) at the
longest stride in the set (0.781 > even A0's 0.591). (ii) **This un-confounds the A1a premise
question and the answer is uncomfortable**: a *pinned 0.8 s clock* beats every HL-cadence
policy on CoT while matching them on tracking — i.e. the HL's cadence authority is not paying
for itself on the metric it exists to optimize. The 2026-07-13 dismissal of fix0p8 ("not a
valid control until root-caused") no longer applies; **stage F / gate G's logic deserves a
re-read** (1 seed — worth a seed before any decision). (iii) The `hl_cot_coef` axis survives
on CoT (0.1 → 0.948 "tracks but wastes", 0.2 → 0.818) but **cot0.1 is now the best tracker in
the whole project** (agg 0.063, ss@1.0 0.044 — beats A0 everywhere) — the "tracks-but-wastes"
label was half-measured. (iv) Upper-body energy signature stands (user hypothesis, motion
proxy): A1a `ub_arm_vel` 0.37–0.58 vs A0 0.14; fix0p8 still the lowest-power A1.
**Decision (user, 2026-07-13): co-training stays the A1a line** — that decision was made on
the strength of the (now-void) claim that fix0p8 "does not walk"; it stands as the user's
call, but the evidence under it has changed and (ii) should go back to Liam.
~~Fixed-command eval joins the bench protocol; the sustained-command under-tracking is a
known deploy risk~~ → **there is no sustained-command under-tracking**; the hold eval stays in
the protocol (it is the deploy regime), now that it measures what it claims to.

**Stage D: arm-calm runs + held-command root cause (2026-07-14/15, all `model_10000`,
64×600×2; D1 = `2026-07-14_11-39-53_..._pose0p5shw16-4...s42` W&B `qcxbn7yt`, D2 =
`2026-07-14_16-03-57_..._shw16-4_..._rs20_s42`, A0 ref = `h1_2_velocity_v2/2026-07-15_08-29-26_a0_v2_optB_rs20_baseline`).**

*(f) arm weights (shoulders ×16, elbow+wrist ×4 in the LL posture anchor) ± the
`resampling_time_range` (3,20) training change (rs20; bench stays pinned (3,8)).
Holds report steady-state err (last 2/3) since 2026-07-14:*

> ⚠ **2026-07-16: the `ss@0.5` / `ss@1.0` columns below are INVALID for the A1 rows.**
> `--eval-cmd-vx` collapses the twist ranges to a point, which drives the derived HIRO
> goal scale to its `1e-3` floor → `V* ≈ s_t`, i.e. the A1 goal channel is inert during
> the hold eval (`play.py:297-299` × `goal_space.py:79-83`). A0 rows are unaffected (no
> goal space). Re-measured on the **unchanged** keeper with the training scale restored:
> **ss@0.5 0.304→0.044 (t90 0.56 s), ss@1.0 0.882→0.097 (t90 0.76 s)** vs A0-optB-rs20
> 0.055/0.077 → **A1 holds ≥ A0; the "TD3-HL velocity-hold degeneracy" does not exist.**
> Read (iii) below is void. Full verdict → `A1_findings.md` WL-C row + `worklines.md` WL-C.
>
> **FIXED 2026-07-16 (F2): the goal scale is now baked into the checkpoint** and pinned at
> load, so `--eval-cmd-vx` can no longer reach it; pre-bake checkpoints (keeper/D1/D2) get a
> `play.py` absence shim that pins the trained scale (prints `[SHIM] ...`). **The whole table
> below has been RE-MEASURED post-F2** (2026-07-16, same checkpoints, no retrain).

**Table (f) — RE-MEASURED post-F2 (2026-07-16), `model_10000`, 64×600×2. Supersedes the
pre-F2 rows** (kept in the strikethrough line below only to show the size of the artifact):

| run | agg vx | agg yaw | agg CoT | stride | pose_dev | arm_vel | fall | ss@0.5 (t90) | ss@1.0 (t90) |
|---|---|---|---|---|---|---|---|---|---|
| keeper (table b/e ref) | 0.079 | 0.194 | 0.818 | **0.625** | 0.032 | 0.567 | 0 | 0.043 (0.56 s) | 0.097 (0.77 s) |
| D1 (weights) | 0.066 | **0.121** | 1.036 | 0.403 | **0.003** | **0.366** | 0 | **0.030** (0.45 s) | **0.039** (0.73 s) |
| D2 (weights+rs20) | **0.062** | 0.124 | 0.882 | 0.352 | **0.003** | 0.420 | 0 | 0.034 (0.40 s) | 0.042 (0.73 s) |
| CTRL (norelabel, WL-C) | 0.067 | 0.133 | 0.933 | 0.360 | 0.017 | 0.581 | 0 | 0.031 (0.35 s) | 0.069 (0.70 s) |
| A0-optB-rs20 | 0.083 | **0.085** | **0.537** | 0.591 | **0.000** | **0.141** | 0 | 0.057 (0.56 s) | 0.076 (0.88 s) |

~~pre-F2 (ARTIFACT, do not cite): keeper 0.354/0.869 · D1 0.376/1.033 · D2 0.470/0.851~~

Goal probe @ held vx=1.0 (post-F2, 64×600×2), the same checkpoints that "proved" the
degeneracy: `|g|vx` **0.10–0.22, sat ≈0** (was 0.98 / 95% saturated), `hl_err_vx` 0.04–0.14,
`ll_err_vx` 0.05–0.10, `end_err_vx` 0.06–0.13 — HL asks correctly, LL delivers, nothing
saturates.

Reads (rewritten 2026-07-16): (i) **arm lever validated, and it is stronger than we thought**
— pose_dev 10× down (0.032→0.003), arm_vel −35%, aggregate vx/yaw *improve*, replay-confirmed
calmer; and now **D1/D2 are the BEST holders in the set** (ss@1.0 0.039/0.042). The artifact
had inverted this: D1's 1.033 was the worst number in the old table and is really the best.
Cost is unchanged and is now the *only* cost: stride/CoT regress to the short-stride regime
(CoT 0.88–1.04 vs keeper 0.818; 1 seed). (ii) **Every A1 run holds at least as well as A0**
(ss@0.5 0.030–0.043 vs A0 0.057; ss@1.0 0.039–0.097 vs 0.076), all 0 falls, all t90 ≤ A0's.
**A1 held-command tracking is not a deploy risk and never was** — the 2026-07-13 "whole velgoal
line under-tracks sustained commands" read is dead. (iii) rs20 (D2 vs D1) is a wash on holds
(0.034/0.042 vs 0.030/0.039) — as expected once the failure it targeted turned out to be an
eval bug; its rationale is void, it costs nothing, keep or revert on other grounds. (iv) **The
real remaining A1-vs-A0 gap is energy, not tracking**: CoT 0.82–1.04 vs A0 0.537, arm_vel
0.37–0.58 vs 0.14, yaw 0.12–0.19 vs 0.085. That is exactly WL-D's brief. (v) Deploy-candidate
implication: **D1/D2 dominate the keeper on holds + arms + tracking** and only lose on CoT/
stride — WL-B's candidate choice should be revisited (Liam's call; the keeper was picked
before the arm lever existed).

## Results

**A0 baseline** (`model_10000.pt`, 600 steps x 64 envs x 2 seeds, 2026-06-30) — the S4 reference:

| err_vx | err_vy | err_yaw | fall_rate | act_rate | CoT (J/m) | power (W) | stride (s) |
|---|---|---|---|---|---|---|---|
| 0.093 | 0.112 | 0.090 | 0.000 | 0.66 | **444.2 ± 0.8** | 187 ± 8 | **0.581 ± 0.002** |

Reads: (i) the metrics are stable across seeds (CoT std 0.8, stride std 0.002), so they are a
usable comparator. (ii) A0's *achieved* stride 0.58 s ≈ its 0.6 s reward target, i.e. A0 does
follow its gait reward (contradicts the earlier guess that it might ignore it; the 0.41 s from a
50-step smoke was contact-chatter noise, gone at 600 steps). (iii) The CoT-vs-period **go/no-go
cannot be answered on A0** (A0 only walks at ~0.58 s; feeding it an off-target phase clock is
OOD and unreliable). The definitive `CoT(period)` curve is an **S2** deliverable, once the LL can
actually walk at a commanded cadence. Proceeding to S1 rests on the literature U-curve + A0
sitting at an interior (not extreme) cadence, not on a measured curve yet.

**S2 v1 (FAILED, 2026-06-30)** — `a1a_s2_cadence_sweep`, wide range 0.5–1.4, period resampled
per HL window. The commanded cadence changed every `c=8` steps (~0.16 s), several times per
stride, which is unfollowable; the `feet_gait` reward fought the warm-started walker and
degraded it. Training `ep_len 40` (vs A0's ~300), `fell_over 103`; the good-looking
`error_vel_xy 0.062` was only over the ~40 pre-fall steps (a misread I corrected). Eval sweep:
`stride_s` flat ~0.32 s for every commanded period (no entrainment), `CoT` flat ~2.4 (4x A0),
power 5x A0. **Verdict: NO-GO-yet — the training never gave a followable cadence, so the lever
is untested.** Fix (v2): resample period **per episode** (held constant within an episode) and
start with a **narrow range (0.5, 0.7)** around A0's 0.58 s so the warm-start can entrain, then
widen. Re-sweep eval within the trained band (0.5–0.7).

**S2 v2 (2026-07-01) [SUPERSEDED — the "no entrainment" read was the eval bug, see S2
correction 2026-07-02]** — `a1a_s2_cadence_v2`, per-episode period, narrow (0.5, 0.7). Fixed v1's
collapse: eval (deterministic) is a **healthy walker** — err_vx 0.11 / err_yaw 0.15 (≈A0), 0
falls, full episodes, CoT 0.96. But **no real entrainment**: `stride_s` flat at 0.573 (= natural
~A0 gait) for every commanded period. The narrow band straddles the natural cadence, so the LL
scores high `cadence_rew` (0.84) *without varying cadence* — no gradient to entrain. (The scary
training `error_vel_xy 0.745` was a stochastic-policy artifact; `action std 0.93`. Deterministic
tracks fine.) **Go/no-go still open** — CoT is flat only because actual cadence never moved. Fix
(v3): widen the range (curriculum from v2) so the natural gait can't satisfy the extremes and
entrainment is forced. Watch the `v=f·L` coupling (off-natural cadence may cost tracking).

**S2 v3 (2026-07-01) [SUPERSEDED — same eval bug; the `v=f·L`-wall diagnosis and the v4
coef-crank are retracted]** — `a1a_s2_cadence_v3`, resume v2, wide (0.35, 1.0), coef 0.5, +3000 it.
Survival + tracking fully recovered (eval err_vx **0.08** < A0, 0 falls, full episodes), but
**still no entrainment**: deterministic stride flat at **~0.55 s for every commanded period
0.35→1.0** (Δ from +0.21 to −0.45), CoT flat ~1.05. Diagnosis: the `v=f·L` conflict — velocity
tracking (goal weighted 3×) pins the LL to its inherited natural velocity→period mapping;
`ll_cadence_coef=0.5` is outweighed, so the LL tracks velocity and eats the small period
penalty. (Note: 0.55 s is the *average* of the natural mapping across the −0.5..1.0 m/s range,
not a single optimum — so real CoT is reclaimable *if* the LL can be made to follow an override.)
**v4 diagnostic:** resume v3, `ll_cadence_coef=2.0` (`a1a_s2_cadence_v4_coef2`) — does hard weight
force entrainment (→ map the period-vs-tracking trade), or does it pin/march (→ pivot the HL job)?
Clean read needs a **fixed-velocity** period sweep (aggregate averages the natural mapping).

**S2 correction (2026-07-02) — eval bug found; ENTRAINMENT CONFIRMED, GO.** Every eval/replay
of v2–v4 ran with the cadence channel **off**: `play.py` rebuilds the runner cfg from defaults
and restores only whitelisted `structure_keys` from `params/agent.yaml`, and `hl_cadence` was
missing from the list → no `hrl_period`/`hrl_phase` buffers, `mdp.phase` fell back to the fixed
0.6 s clock, `--eval-cadence-period` silently inert behind `if self.hl_cadence`. Exposed by (i)
`gait_match` identical to 3 decimals (0.812) across seven commanded periods (impossible if the
schedule varied) and (ii) the phase-scramble test: **A0 is near-perfectly phase-slaved to its
clock obs** (match 0.973; zero/random clock → err_vx 0.075→0.41/0.53, stride 0.57→0.23), so the
clock input is load-bearing, not ignored. Fix: `hl_cadence`, `cadence_period_range`,
`ll_cadence_coef`, `hl_cot_coef` added to `structure_keys` (`scripts/play.py:170`).

**Corrected fixed-velocity sweep (vx=0.5, 64 envs × 600 steps, 1 seed):**

| cmd period | v2 stride | v2 CoT | v2 err_vx | v3 stride | v3 CoT |
|---|---|---|---|---|---|
| 0.35 | **0.343** | 1.110 | 0.069 | **0.343** | 1.392 |
| 0.50 | **0.486** | 1.051 | 0.064 | **0.485** | 1.139 |
| 0.65 | **0.620** | 0.859 | 0.064 | **0.618** | 0.872 |
| 0.80 | **0.719** | 0.684 | 0.065 | **0.700** | 0.626 |
| 1.00 | 0.782 | 0.601 | 0.093 | 0.532 (lock lost) | 0.698 |

Reads: (i) **near-1:1 entrainment 0.35→0.8** with tracking preserved and 0 falls — v2 (trained
only 0.5–0.7) generalizes far outside its band, consistent with phase-slaving; v4's coef-crank
was never needed. (ii) **CoT swings ~45% with period at constant speed** (1.11→0.60) — the
natural 0.58 s gait is NOT energy-optimal at 0.5 m/s; the optimum sits at long strides near the
entrainment edge (v3 loses lock at 1.0). The cadence lever moves energy → **S1c/S3 premise
validated**. (iii) Caveat: A0 at the same fixed 0.5 m/s has CoT **0.577** — the A1a LL is not
yet more efficient than A0 overall; frame the S4 claim within-architecture or improve LL
efficiency. (iv) Lessons → `A1_findings.md` ledger (structure_keys coupling; effect-asserting
smokes; a metric that cannot vary = broken plumbing).

**CoT(period, vx) map (2026-07-02, v2, 64×600×1 seed, fixed-command eval).** Entrainment is
robust across the whole speed range (stride ≈ command 0.35→0.8 at every vx; ceiling ~0.8 for a
commanded 1.0; **0 falls in all 20 cells**). The two decision-relevant grids:

| CoT: P \ vx | 0.2 | 0.5 | 0.8 | 1.0 |
|---|---|---|---|---|
| 0.35 | 1.813 | 1.098 | 0.924 | 0.892 |
| 0.50 | 1.814 | 1.053 | 0.929 | 0.905 |
| 0.65 | 1.406 | 0.854 | 0.763 | 0.762 |
| 0.80 | 1.035 | 0.685 | 0.643 | 0.657 |
| 1.00 | **0.751** | **0.587** | **0.608** | 0.649 |
| 1.15 ⚠ | 0.740 | 0.703 | — | — |
| 1.30 ⚠ | 0.908 | 0.808 | — | — |

⚠ = past the entrainment ceiling (slow-end probe, 2026-07-02): lock degrades (match
0.87→0.60), achieved stride regresses to 0.65–0.72, and CoT *rises* (power ~2×) as the
phase-slaved LL fights the unfollowable clock. (The re-measured P=1.0 cells replicated the
map within ~0.01 CoT — the eval is seed-robust at this magnitude.)

| err_vx: P \ vx | 0.2 | 0.5 | 0.8 | 1.0 |
|---|---|---|---|---|
| 0.50 | 0.045 | 0.068 | 0.106 | **0.135** |
| 0.65 | 0.044 | 0.062 | 0.111 | 0.166 |
| 1.00 | 0.063 | 0.088 | 0.150 | **0.205** |

Reads: (i) longer strides are cheaper at every speed, but the saving **shrinks with speed**
(0.65→1.0 saves 47% CoT at 0.2 m/s, only 15% at 1.0 m/s) and the curve is ~flat 0.8→1.0 at
high speed. (ii) The **tracking cost of long strides grows with speed** (at 1.0 m/s err_vx
0.135→0.205 going P 0.5→1.0; at 0.2 m/s long strides are ~free). So the composite optimum
(tracking + CoT) is **speed-dependent** — long strides at low speed, shorter at high speed —
the human-like cadence–speed relation emerging from the trade, and a genuinely non-trivial
speed-conditional mapping for the HL to learn (the richer-HL-job claim, now with data).
(iii) `cadence_period_range` for S1c: (0.35, 1.0) is followable everywhere; achieved ceiling
~0.8.

**Slow-end probe (2026-07-02, v2, commanded 1.0/1.15/1.3 at vx 0.2/0.5).** The achieved-stride
ceiling is **~0.8 s regardless of command**: past ~1.0 the lock degrades (match 0.87→0.60) and
the stride regresses (0.79→0.65–0.72) — no falls, it just stops obeying. Crucially **CoT RISES
past the ceiling** (0.59→0.81 at 0.5 m/s; power ~2×): a phase-slaved LL fighting an unfollowable
clock burns energy, so the CoT(period) optimum is interior, at the entrainment edge (~0.8 s
achieved). → **(0.35, 1.0) is the complete S1c command envelope; nothing to gain past 1.0** with
this LL. **Caveat: the interior optimum is conditional on the fixed 0.56 duty** — the
past-ceiling CoT rise measures the LL fighting an unfollowable clock, not an inherent cost of
long strides; under a `d(T)` schedule the low-speed optimum could sit past 1.0 s, so a small
`d(T)` pilot at vx=0.2 would either extend the CoT map or confirm the interior optimum honestly. Slow-stride enablers (period-dependent duty factor — the fixed 0.56 duty demands
minimal double support at long T, the unnatural slow gait; possibly a slow-end curriculum;
note training-length is NOT the lever: v3 = v2 + 3000 wide-band iters and its slow end got
*worse*) are filed as future work, off the A1a critical path.

**Gait geometry — why the slow end is hard (the user's step-cycle model, 2026-07-02).** Per step:
`T_step = 1/f + s` (swing/single-support time + double-stance time), `x = v·T_step` →
`s = x/v − 1/f`. At fixed speed, a longer commanded period forces a longer step `x`, which the
gait must absorb as longer swing (`1/f`, one-legged balance — hard) or longer double stance
(`s` — cheap). The `feet_gait` schedule (duty 0.56, offsets [0, 0.5]) **fixes the split**:
`1/f = 0.44·T_stride`, `s = 0.06·T_stride`, so swing absorbs **88%** of any period increase —
the hard way. At the achieved ceiling (~0.8 s stride) swing is ~0.35 s ≈ 1.2× the inverted-
pendulum time constant (√(0.88/9.81) ≈ 0.3 s). Natural slow walking instead grows the double-
stance share (duty ↑ with T) — hence the period-dependent duty factor `d(T)` as the future
slow-stride enabler.

**How humans do it (corroboration + a principled `d(T)` target).** Humans absorb slower walking
almost entirely in **double stance, not swing**: duty rises from ~0.6 at normal speed to
~0.65–0.7+ when slow, double-support share grows from ~20% to ~25–30%+, while **single-support
(swing) time stays ~constant near 0.4 s** across a wide speed range (duty < 0.5 = the run
transition, flight phase). So humans hold `1/f` fixed and let `s` take all the variation — the
opposite of our fixed 0.44/0.06 split. Principled fix target: **hold swing ≈ const**, i.e.
`d(T) ≈ 1 − τ_swing/T_stride` with `τ_swing ≈ 0.30–0.35 s` → d ≈ 0.56 at T=0.7 (matches the
trained gait) rising to ~0.70 at T=1.1, keeping single-support near the pendulum time constant
(√(0.88/9.81) ≈ 0.3 s) across the range instead of blowing past it. **Implemented 2026-07-03**
(S2d): `cadence_swing_time=0.31` — the *continuity* calibration (swing of the trained duty-0.56
gait at T=0.7, so the floor departs exactly at 0.70 s; the pendulum constant 0.30 was the
runner-up, difference negligible) — with `cadence_duty_range=(0.56, 0.70)` as the shared clamp
for `feet_gait` AND play's `gait_match` (single source of truth via structure_keys). v5
(`a1a_s2_cadence_v5_duty`) trains from scratch (A0 warm start, range (0.35, 1.3), 5001 it,
seed 42) as an independent measurement — not resumed from v2, so the d(T) effect isn't
confounded by 5000 fixed-duty iterations. Success = ceiling past 0.8 s + fast band unregressed;
then re-measure the CoT(period, vx) slow cells (the interior-optimum caveat above).

**S1c implemented (2026-07-02).** New cfg field `hl_cadence_source: "random" | "hl"` (default
`random` = the S2 per-episode source, so every pre-S1c checkpoint restores with unchanged HL
shapes). With `"hl"` (requires `hl_algorithm=td3`, loud error otherwise) the TD3 HL emits the
stride period as one extra tanh action dim: HL action = `goal_dim+1` (actor/critics/replay
buffer sized accordingly; HIRO relabel only relabels the goal dims and carries the period column
through), mapped affinely from [-1,1] to `cadence_period_range` and written to `hrl_period` at
each fire. Phase integrates incrementally, so a per-window period change never jumps the clock;
the per-episode random resample is disabled when the HL owns the period. CoT enters the HL
*window* reward: per-env command-gated energy/distance accumulated per step; at window close
`R_HL += -hl_cot_coef * E_win / (m g * max(d_win, d_floor))`, `d_floor = 0.1*c*dt` (a stuck
robot under command is expensive but finite; an all-gated-off window is exactly 0). New logs
`hl/cot_pen`, `hl/period_mean`; ONNX metadata gains `hl_cadence_source` + `cadence_period_range`
(the C++ side mapping is S3+ work, once there is a trained cadence HL worth deploying).
`hl_cadence_source` was added to the play.py `structure_keys` at creation (the S2 eval-bug
lesson). Tests 18/18 + v2 compat: dims grow only when on; relabel passthrough; endpoint map
(g=-1 -> 0.35, g=+1 -> 1.0); CoT gate exact-0 / finite-negative through the real `learn()`;
phase continuity across a 0.4->0.9 pin change; v2 `model_5000` benchmarks cleanly through the
updated play path (restored as `random`, gait_match 0.91).

**S2d v5 (2026-07-03, FAILED entrainment)** — `a1a_s2_cadence_v5_duty`: from-scratch A0
warm start + d(T) (swing 0.31) + the full (0.35, 1.3) band from iteration 0 (5001 it, seed 42;
chosen deliberately as a v2-independent measurement). Training was healthy (ep_len 1000, 0
falls, cadence_rew plateau raw ~0.64) but the fixed-vx sweep shows **no entrainment**: stride
flat ~0.39 s for every command 0.5→1.3 (v2: 0.49→0.78), gait_match ~0.65, CoT flat ~0.74.
NOT an eval artifact: training raw cadence reward (0.64) == eval gait_match (0.65) — the two
independent measurements agree the clock was ignored (this cross-check is the standard now).
Notably tracking is *better* than v2 (err_vx 0.041–0.051, 0 falls) and the natural gait it
settled on is fast (~0.4 s) and fairly efficient (CoT 0.73 vs v2's 1.05 at P=0.5) — a good
walker that treats the clock as noise. **Read: curriculum failure, not d(T)** — v2 entrained
because its narrow band (0.5–0.7) straddled the warm-start's natural 0.58 s (following = small
change, immediate reward), then widening preserved the lock (v3); giving the fresh walker
mostly-unfollowable periods from step 0 leaves no entrainment gradient and clock-ignoring is
the stable compromise. **Fix (v6)**: resume the entrained v2 `model_5000` with d(T) + (0.35,
1.3), +3000 it — v3 (same resume/budget, fixed duty) is the clean control for d(T)'s effect on
the slow cells.

**S2d v6 (2026-07-03, FAILED — d(T) at resume LOST the mid-band lock)** —
`a1a_s2_cadence_v6_duty_resume` (resume v2 `model_5000`, τ=0.31, (0.35, 1.3), +3000 it).
Fixed-vx sweep: still locks 0.35–0.5 (stride 0.343/0.487) but **regresses to a ~0.42 s
clock-ignoring gait from 0.65 up** (0.466/0.426/0.423 at cmd 0.65/0.8/1.0; v2 followed to
0.78, v3 to 0.70). 0 falls, err_vx ~0.065, CoT flat ~0.80; training raw cadence_rew 0.64 ==
eval match (real, not plumbing). **Diagnosis: the schedule re-scored the already-locked gait**
— with τ=0.31 the floor departs at T=0.70, so v2's best-following 0.7–0.8 region went from
rewarded (duty 0.56) to penalized (wants 0.60–0.61) the moment training resumed, while ~30%
of episodes (the 1.0–1.3 slice) were unfollowable noise; the gradient chose the clock-ignoring
compromise. Confound noted: v6 ≠ v3 in band too (1.3 vs 1.0), so attribution is d(T)-rescoring
and/or the extra unfollowable slice — both point to the same fix. **v7 (staged, running)**:
τ=**0.32** (floor departs at 0.32/0.44 ≈ 0.73 s; duty at 0.8 shifts only 0.56→0.60) + band
back to **(0.35, 1.0)** (no unfollowable-beyond-ceiling dilution), resume v2, +3000 it.
Stage 2 (widen to 1.3) only if the stage-1 ceiling moves past 0.8.

**S2d v7 (2026-07-03, stage 1 — ceiling UNMOVED; d(T) program PARKED)** —
`a1a_s2_cadence_v7_duty_staged` (resume v2, τ=0.32 → floor departs 0.73 s, band (0.35, 1.0),
+3000 it). Sweep (vx=0.5): locks 0.35–0.65 near-1:1 (0.342/0.488/0.609 — v6's damage fixed by
the recalibration + clean band), 0 falls, healthy general bench (err_vx 0.087, match 0.88).
But **0.8 follows worse than the v3 fixed-duty control** (stride 0.580 vs v3 0.700 / v2 0.719;
caveat: match there is high 0.85 and CoT/power are the best 0.8-cell yet (0.64, 210 W) —
consistent with partial following + contact chatter biasing the footfall interval down), and
beyond-band regresses to ~0.44 (v2 extrapolated to 0.78). **Cross-attempt conclusion (v5, v6,
v7):** the phase-slaved LL follows the clock *rate* readily but does not adopt a different
*stance shape* when the reward schedule offers one — every duty re-grade costs some following
(v6 badly, v7 mildly at 0.8) and none opened the slow band. The "swing-time is the binding
constraint, duty absorbs it" hypothesis is behaviorally unconfirmed at these budgets/curricula
(remaining untried: duty-annealing curriculum mid-run; longer budgets; explicit duty obs).
**Keeper: v2 `model_5000`** (canonical cadence LL; followable envelope (0.35, 1.0); its
CoT(period,vx) map = the S3 answer key). The interior-optimum caveat stands unresolved — the
slow cells past 1.0 remain unmeasured under a followed d(T) gait.

**S3 (2026-07-03, `a1a_s3_cadence_hl`, `logs/rsl_rl/h1_2_velocity_a1/2026-07-03_16-17-10_*`)** —
frozen v2 `model_5000` + TD3 HL, `hl_cadence_source=hl`, `hl_target_mode=absolute` (the oracle-
trained frozen LL only knows absolute command-range V\*; delta would feed it OOD targets),
`hl_cot_coef=1.0`, seed 42, 5001 it. Converged clean: ep_len 1000, 0 falls, `hl/act_abs_mean`
0.78 (unsaturated), `metrics/cot` 0.91→0.805. Fixed-vx sweep (64×600×2 seeds), HL's own period
vs `--eval-cadence-period 0.6` (same frozen LL):

| vx | HL CoT | HL stride | gait_match | HL err_vx | falls | fixed-0.6 CoT | ΔCoT |
|---|---|---|---|---|---|---|---|
| 0.2 | 2.239 | 0.615 | 0.944 | 0.166 | 0 | 2.758 | **−18.8%** |
| 0.5 | 1.036 | 0.608 | 0.927 | 0.235 | 0 | 1.127 | **−8.0%** |
| 0.8 | 0.711 | 0.697 | 0.928 | 0.301 | 0 | 0.915 | **−22.3%** |
| 1.0 | 0.748 | 0.665 | 0.924 | 0.270 | 0 | 0.904 | **−17.3%** |

Reads: (i) **mechanism GO** — the learned HL drives CoT below the fixed-0.6 clock at every speed
(−8…−22%), tracking held/slightly better, 0 falls; entrainment real (gait_match 0.94 agrees with
training `ll/cadence_rew`, the truth cross-check). (ii) **Optimum NOT reached** — the HL sat on a
near-natural **flat ~0.61–0.70 stride at all speeds**, NOT the map's speed-dependent long-strides-
at-low-speed structure (vx=0.2 could hit CoT ~0.75 at P=1.0; HL got 2.24). (iii) Absolute CoT sits
well above the oracle map because the **learned TD3 HL tracks velocity worse than the oracle**
(err_vx 0.17–0.30 vs the map's 0.045–0.205) — fewer metres per joule inflate CoT. **Diagnosis:**
`hl_cot_coef=1.0` is too weak vs the window tracking term (~10–14 vs CoT ~0.8) → S5 coef sweep is
the lever, deferred to the full arch. **Recipe note:** v2 (this frozen LL) trained at
`ll_action_rate_coef=0.05` — an explicit launch override, NOT the rl_cfg/`train_h1_2_a1.sh`
default (both 0.02) nor the documented keeper (0.02, the value that cured the `fell_over` spike).
Inert here (LL frozen) but **load-bearing for S4** (co-trained LL): reconcile before S4.

**S4 + S5 (2026-07-04) — the fully co-trained learned-cadence A1a FAILS; the working A1a is the
staged/frozen one.** Full/correct arch (TD3+HIRO, delta/directional, velobs, warm-start A0, cadence
source=hl, 10001 it). Runs: `a1a_s4_cotrain_cot1_ar05_s42`, `_ar02_s42`, `a1a_s5_cot{2,4,8}_ar05_s42`.
Fixed-vx benchmark, HL period vs `--eval-cadence-period 0.6`:

| coef (ar) | HL stride | gait_match | ΔCoT vs fixed-0.6 | err_vx | fall_rate |
|---|---|---|---|---|---|
| 1.0 (0.05) | 0.47 | 0.92 | **+33%** | 0.44 | 0 |
| 1.0 (0.02) | 0.43 | 0.90 | **+40%** | 0.50 | 0 |
| 2.0 (0.05) | 0.31 | 0.55 | −3% | 0.67 | 0 |
| 4.0 (0.05) | 0.33 | 0.50 | 0% | 0.64 | 0 |
| 8.0 (0.05) | 0.28 | 0.50 | 0% | 1.65 | 0 |

Reads: (i) **at coef 1 the HL drifts to a short (~0.45) EXPENSIVE cadence** — CoT +30–47% above its own
fixed-0.6 baseline (opposite of S3's −8…−22% frozen-LL win). (ii) **Raising coef makes it worse:**
cadence shorter (→0.30, natural), **entrainment collapses** (gait_match 0.9→0.5 — LL ignores the
clock, confirmed by train `cadence_rew` 0.30→0.24), **tracking degrades** (eval err_vx→1.65, train
err_vx~4 at coef 8), CoT never beats fixed-0.6. No suicide attractor (eval fall_rate 0) but train
ep_len/fell_over erode. (iii) **ar 0.02 ≈ 0.05** — ar is not the lever (and the earlier "5× tracking
regression" was a fixed-vx-vs-random-command **benchmark artifact**: the A1 keeper itself gives
err_vx 0.25 / err_yaw 0.82 at fixed-vx=0.5). **Root cause:** S3 worked because the FROZEN LL is a
stable entraining tracker the HL just picks cadence against; in co-training the CoT pressure makes the
JOINT policy abandon entrainment + tracking (penalty-domination, as the ADR warned raw energy would).
**Conclusion: the A1a that works = the STAGED/frozen architecture (S2 random-cadence+oracle entrained
LL → freeze → S3 learned-cadence HL), NOT fully-joint co-training.** Forward options open (rework the
CoT reward vs adopt the staged frozen-LL A1a as the deliverable).

**From-scratch track (2026-07-06/07).** `ll_goal_kernel=exp` + true-terminal `fell_over`
trains the A1a LL from scratch (oracle HL, cadence on, random source) to better-than-A0
tracking — warm-start no longer required; failure chain, the full-exp kernel fix, and the
TD3+HIRO co-train pair (cot0 walks ≈A0 linear tracking; cot1 = penalty domination; band
self-narrows to the HL's visited periods) → `A1_findings.md` (from-scratch rows).
`CoT(period, vx)` grid of the from-scratch oracle LL (1 seed/cell, 0 falls in all 20;
v2 map above is the comparator): **optimum P≈0.8 at every speed** (its entrainment edge),
err_vx 0.020–0.073 grid-wide (v2: 0.045–0.205), in-band CoT ≤ v2 (e.g. 0.87 vs 1.05 @
0.5/0.5) but the P=1.0 long-stride cells stay out of reach (lock lost at speed; only
~0.78 achieved at vx 0.2, CoT there 1.66 vs v2's 0.75):

| CoT: P \ vx | 0.2 | 0.5 | 0.8 | 1.0 |
|---|---|---|---|---|
| 0.35 | 2.07 | 1.08 | 0.92 | 0.91 |
| 0.50 | 1.50 | 0.87 | 0.77 | 0.75 |
| 0.65 | 1.37 | 0.81 | 0.73 | 0.73 |
| 0.80 | **1.25** | **0.76** | **0.66** | **0.66** |
| 1.00 | 1.66 | 0.96 | 0.70 | 0.78 |

**From-scratch co-train pair, key rows (2026-07-06/07, `a1a_scratch_td3_expOH_cot{00,10}_s42`,
Model v1 — historical).** Aggregate bench + fixed-vx GRID:

| run | err_vx | fall | CoT | stride (unpinned) | match | GRID envelope |
|---|---|---|---|---|---|---|
| cot0 | **0.100** | 0.0 | **2.81** | 0.369 | 0.92 | follows to ~0.5 only; match ≤0.77 and falls rise past P=0.65 |
| cot1 | 0.769 | 0.0 | 1.06 | 0.635 | 0.85 | flat ~0.60 stride at every speed; err_vx ≥0.72 everywhere |

Replay observations (user): cot0 = very short steps, torso slightly twisted, legs not
parallel; cot1 = no usable tracking, walks sideways (long side strides) even under a
pinned forward command; the from-scratch oracle LL walks cleanly but with a **~20° hip
twist** (upper body straight, legs not parallel to travel). The hip twist is why hip
yaw/roll joined the posture anchor (2026-07-07, `hrl_runner.py` — the goal space is
heading-invariant, nothing else aligned the legs). Reads: (i) cot0 is the first
from-scratch co-trained A1a that walks, but nothing penalizes its energy (CoT ~3× the
oracle LL) and (ii) its **cadence band = its HL's visited band** (the HL parked at ~0.36
all run, so the LL never learned long periods) — the band-coverage lesson as
*self*-curriculum; R2's random-period source is the fix under test. (iii) cot1 lowers CoT
only by abandoning the command — penalty domination reproduces from scratch at coef 1.0.

## Open knobs (set by data, not guessed)
- `cadence_period_range` — settled at `(0.35, 1.0)` on v1 (S2 envelope); re-confirm on the
  first entrained v2 LL (per-checkpoint property).
- `hl_cot_coef` — co-train boundary from R3 (0.2/0.5); frozen-arch value from F ({1,2,4},
  capped below tracking erosion).
- A0+energy coef — 3-point mini-grid at S4′, chosen favorably for the control.

## Honest caveats
- The **definitive `CoT(period)` curve needs the cadence-capable LL (S2+)**; the warm-start LL
  only knows ~0.6 s, so S0's period probe is local/limited.
- If `A0+energy` matches A1a in S4′, the in-sim efficiency benefit is **not** from the hierarchy;
  report it and lean on the OOD/A2 case. Stated up front to avoid motivated reading.
- **v1/v2 boundary (ADR-0005):** nothing above the Plan v2 section is thesis-scorable; v1
  numbers may be cited only as mechanism evidence, never compared against v2 runs.
- The absolute-CoT gap vs A0 (v1: A0 0.577 vs cadence LLs ≥1.0 at fixed vx=0.5) is an
  LL-efficiency limitation orthogonal to the hierarchy claim; if it persists on v2 it is
  reported as future work, not hidden by the within-architecture framing.

## Stage D (WL-D, 2026-07-17): reward/gait lever batch — implementation + launch record

Per `worklines.md` WL-D (gate released 2026-07-16 by WL-C). Two decisions from Liam
before launch: (1) promote D1/D2's arm weights to the A1 default (done — see below);
(2) keep arms 1-2 (cadence HL) even though fix0p8 beats every HL-cadence policy on CoT
at equal tracking — Liam's call: cadence is "the main reason for hierarchy... otherwise
there is no real benefit... not even training is faster," so the arms stay, plus a
fix0p8+weights control re-run on the new base (needed since table (e)'s fix0p8 predates
the arm-weight promotion). Open question raised alongside this (not yet investigated):
why the A0 baseline CoT (0.537) is still well below every A1a variant even after the
arm-calming — see the reads in the WL-D chat transcript for a first-pass hypothesis
(arm-swing energy, self-chosen-cadence cost, and the LL never seeing A0's efficiency/
quality shaping terms — exactly what arm 4 tests).

**Base config promoted** (`config/h1_2_a1/rl_cfg.py`): `ll_posture_weights` default
changed from `None` to `{shoulder: 16, elbow: 4, wrist: 4, ankle_roll: 4 (inert unless
arm 5's flag is set)}` — D1's temp rl_cfg edit, now permanent. Every WL-D arm trains on
this base; D1 (`2026-07-14_11-39-53_..._pose0p5shw16-4...s42`, `model_10000`) **is**
this base config unmodified and serves as the batch's zero-point (no retrain needed).

**Implementation (commit `6828ca4`):**
- New shared `mdp.mech_power()` helper (`src/tasks/velocity/mdp/rewards.py`) and
  `mdp.cost_of_transport_penalty()` — a stateless per-step CoT analog
  `-w·P/(m·g·max(|v_cmd|, floor))`, clamped `max_cot=10.0` (catastrophe bound, same
  pattern as the LL posture/action-rate clamps — see below). Wired as an inert-by-
  default (`weight=0.0`) A0 reward term (`velocity_env_cfg.py`) — this **is** the
  parked S4' "A0+energy" control, now buildable — and as the arm-4d LL-intrinsic
  mirror (`ll_energy_coef`).
- Arm 3 (`ll_stand_still_coef`) and arm 4a-4c (`ll_angmom_coef`, `ll_footslip_coef`,
  `ll_footclear_coef`) mirror A0's `stand_still` / `angular_momentum_penalty` /
  `feet_slip` / `feet_clearance` into the LL intrinsic (`hrl_runner.py`), same pattern
  as the existing posture/action-rate additions. All default 0.0 (off).
- Arm 5 (`ll_posture_anchor_ankle_roll`): extends the LL posture anchor's joint regex
  set (`hrl_runner.py` anchor_patterns) to include `.*ankle_roll.*` when set; weight
  comes from the `ankle_roll` entry already in the promoted `ll_posture_weights` (4.0 -
  the joint's range is only +-15 deg, far narrower than the shoulders).
- Arm 7 (explicit-PD actuation): no new actuator code needed — mjlab already ships
  `IdealPdActuatorCfg` (explicit torque = kp*(q_des-q)+kd*(qdot_des-qdot) via a
  `<motor>` actuator) as an alternative to the implicit `<position>` servo
  (`BuiltinPositionActuatorCfg`). New `H1_2_ARTICULATION_EXPLICIT_PD` +
  `get_h1_2_robot_cfg(explicit_pd=True)` + task `Unitree-H1_2-Flat-ExplicitPD`.
- Arm 8 (wide DR): `apply_wide_dr()` bundles push x/y +-0.5->+-1.5, new
  `dr.body_mass` event on `torso_link` (add, -1..+3 kg), foot-friction floor 0.3->0.1.
  Task `Unitree-H1_2-Flat-WideDR`.
- Arm 9 (correlated noise): new `LowPassNoiseModel`/`LowPassNoiseModelCfg`
  (`src/tasks/velocity/mdp/noise.py`) - a first-order EMA filter over the same i.i.d.
  innovations `Unoise` draws, rescaled so the filtered process's stationary std matches
  the original (unfiltered) amplitude (~2Hz cutoff). Swaps the actor's `base_ang_vel`
  and `joint_vel` noise terms. Task `Unitree-H1_2-Flat-CorrNoise`.
- Launchers: `train_h1_2_a1a_LL_rewards.sh` (new, dedicated - this batch's baseline is
  cadence-HL + cot0.2 on the arm-calmed base, which is NOT `train_h1_2_a1.sh`'s own
  tagless default, so a separate script avoids re-specifying 3+ flags on every one of
  11 sbatch invocations); `train_h1_2.sh` extended with `ENV_VARIANT` (arms 7/8/9) and
  `ENERGY_COEF` (A0+energy) knobs.
- **Catastrophe clamp + power dedupe (2026-07-17, same commit day, follow-up)**:
  `cost_of_transport_penalty`'s ratio is clamped (`max_cot=10.0`) - unlike the posture/
  action-rate penalties (clamped since the 2026-06-30 `it9624` crash: unbounded L2 let
  a sim blow-up spike the LL PPO update to ~-5e5 in one step), the CoT term had no cap
  on the numerator, only a denominator floor. Deliberately **NOT** applied to the HL's
  own `hl_cot_coef` window computation in `hrl_runner.py` (Liam's call, 2026-07-17):
  that path feeds a TD3 replay buffer (off-policy, one bad transition diluted across a
  500k-capacity buffer and clipped double-Q), not an on-policy PPO rollout whose GAE
  backward-recursion lets one outlier corrupt many timesteps' advantages at once - the
  failure mode the clamp guards against doesn't transfer with the same urgency. The
  power SUB-formula (only) is deduped: `hrl_runner.py`'s window-CoT now calls
  `mdp.mech_power()` instead of an inline copy; the window/floor/signed-distance logic
  stays a deliberately separate formula (achieved distance, not commanded speed;
  window-integrated, not per-step) - not unified, since it answers a different question.
- All new coefficients default to 0.0/off (byte-identical baseline unless swept); every
  mechanism individually smoke-tested (`WANDB_MODE=disabled`, small envs/iters) before
  commit, including the comma-syntax tuple CLI override for `cadence_period_range`
  (`--agent.cadence-period-range 0.8,0.8`) - confirmed WORKING 2026-07-17, so the old
  "tyro tuple override broken" note (2026-07-03) was a space-vs-comma syntax mistake,
  not a real limitation; no scalar-flag workaround was needed for the fix0p8 control.

**Launched 2026-07-17** (15 runs, `sbatch --export=ALL,...`, 10001 it / 4096 envs /
seed 42 each): the fix0p8+weights control; arm 1 (`ll_cadence_coef` 1.0/2.0); arm 2
(`hl_cot_coef` 0.15/0.25); arm 3 (`ll_stand_still_coef=1.0`); arm 4a-d
(`ll_angmom_coef=0.025`, `ll_footslip_coef=0.25`, `ll_footclear_coef=1.0`,
`ll_energy_coef=0.05` - first-guess values matched to each mirrored term's own A0
weight magnitude, NOT tuned, expect a second pass); arm 5 (`ll_posture_anchor_ankle_roll`);
arm 7/8/9 (`ENV_VARIANT=explicit_pd|wide_dr|corr_noise`); A0+energy
(`ENERGY_COEF=0.05`). Results pending.

### Arm 6 — heel-to-toe roll-over / ankle push-off (researched -> approved 2026-07-17; formulation A trains)

Two prior-work lines ground this: **Siekmann et al., "Sim-to-Real Learning of All
Common Bipedal Gaits via Periodic Reward Composition"** (arXiv:2011.01387) - the origin
of the clock/phase-indexed-indicator mechanism this codebase's `feet_gait` term already
implements, generalizable from a binary stance/swing indicator to *any* phase-indexed
reference (foot orientation, joint power); and the biomechanics literature on ankle
push-off (Winter-style gait analysis) - **ankle plantarflexion push-off generates >80%
of the positive mechanical power in late stance**, concentrated in the last ~15-20% of
stance (the "A2 burst"), the dominant propulsive event in human walking.

**Formulation A - foot-pitch phase-locking** (a *matching* problem - a real target
trajectory exists -> exp kernel, consistent with `track_linear_velocity`/
`variable_posture`). Reuses `feet_gait`'s existing phase math (`period`, `offset`,
`duty` threshold):

```
phi_i = leg_phase_i / duty   (in [0,1), scheduled-stance progress)
theta_ref(phi) = theta_hs + (theta_to - theta_hs) * (1 - cos(pi * phi^k)) / 2
r_pitch = 1[|cmd| > cmd_thresh] * exp( -(1/sigma^2) * sum_i 1[sched_stance_i] * (theta_i - theta_ref(phi_i))^2 )
```

using the `ankle_pitch` joint angle directly as the foot-pitch proxy (no new sensor).
`k>1` concentrates the rotation late in stance, matching the real ankle-angle curve
(flat through midstance, rapid near push-off) rather than a naive symmetric raised-
cosine. Gate pins (2026-07-17 review): (i) **command-gated** like every other gait
term - the phase clock keeps running at stand, so an ungated r_pitch rewards
ankle-marching in place and fights the arm-3 stand-still term; (ii) the stance gate
is the **scheduled** window (`leg_phase < threshold`), not actual contact - `phi` is
only well-defined on the schedule, and `feet_gait` already pushes contact to match
it; (iii) `duty` must read the SAME threshold source `feet_gait` uses (it becomes a
per-env tensor if the parked d(T) schedule ever un-parks - do not hardcode 0.55 twice).

**Formulation B - ankle push-off power burst** (a *maximization/bonus* problem - no
upper target -> must be bounded-ABOVE, not a raw unbounded ratio, per the same
clamp-discipline that motivated the `cost_of_transport_penalty` clamp above):

```
P_i = tau_ankle_pitch_i * theta_dot_ankle_pitch_i     (signed per-joint power - a new
                                                       2-line expression, NOT mech_power,
                                                       which is abs-summed over all joints)
g_i = 1[ phi_i in [1-w, 1) ] * 1[ contact_i ]         (w ~ 0.15-0.2, terminal stance;
                                                       schedule AND actual contact)
r_pushoff = sum_i g_i * ( 1 - exp( -ReLU(P_i) / P_scale ) )
```

saturates toward 1 per foot as push-off power grows, instead of rewarding unbounded
torque-cranking during the gated window (the failure mode this project already hit
once, when an uncapped HL action std blew up chasing a weak gradient). `ReLU` so only
forward-delivering push-off power is rewarded, not absorption/eccentric work. Gate pin
(2026-07-17 review): B requires **schedule AND actual contact** (unlike A) - phase-only
gating lets the LL harvest the bonus by driving the ankle in the air after an early
liftoff; power without ground contact is thrash, not propulsion. **Known watch item
(monitor in replay, not a redesign):** ReLU keeps the positive half of any ankle
oscillation inside the window, so dithering nets reward; the saturation caps it at the
same ceiling as genuine push-off and the window is short, so this is a replay check,
not a blocker.

Both reuse `feet_gait`'s phase math; no new sensors needed. A harder third option
(center-of-pressure
progression heel->toe, using the foot's 7 individual collision sub-geoms) was
considered and parked: `feet_ground_contact` nets each foot's contact into one scalar
(`reduce="netforce"`), so this would need new per-geom contact sensing - flagged as a
stretch goal, not specced.

**Constants probe (REQUIRED first, offline, no cluster):** one instrumented replay
pass on the batch-base checkpoint fixes every guessed constant at once: (i) the
`ankle_pitch` sign convention (axis `0 1 0`, range `[-0.897, 0.524]`, nominal standing
`-0.3`) - which direction is dorsiflexion vs. plantarflexion, read at a heel-strike vs.
toe-off instant; (ii) measured `theta_hs`/`theta_to` from the actual angles at those
events (an exp kernel imprints a wrong reference HARD - do not hand-pick these); (iii)
peak terminal-stance ankle-pitch power -> `P_scale`. Only `sigma` and `k` remain free
knobs after the probe.

**Decision (planning-chat review + Liam go, 2026-07-17):** implement BOTH terms as
LL-intrinsic coefficients defaulting to 0.0 (byte-identical baseline, matching the
batch discipline), run the probe, then train **formulation A only** as the arm-6 slot,
on the same batch base as the other arms (arm-calmed cadence-HL + cot0.2, i.e. the
promoted default config - clean batch member; a winner consolidation retrains
everything anyway). B stays implemented-but-untrained until A's
read. A shapes the roll-over character (the named defect); B rewards the propulsive
event and sits in direct tension with the CoT objective (it pays for ankle power the
CoT term taxes - theoretically fine, push-off should cut contralateral collision
absorption which `mech_power` also counts as cost, but harder to predict).

**Status:** spec amended + approved 2026-07-17 (gates pinned, probe-first); probe run,
both formulations implemented, formulation A trained + benched (2026-07-17/18). **Does
NOT clear the winner bar as trained at `coef=0.5`** (tracking regresses) — a coef sweep
is the natural next step; not yet run (out of this hand-off's one-run scope).

**Constants probe (2026-07-17, D2 checkpoint, 64 envs x 900 steps, `vx=1.0` pinned, duty
read live off the loaded runner — never a second hardcoded copy):** sign convention:
more-negative `ankle_pitch` = plantarflexion (push-off), confirmed empirically (correlates
with positive signed ankle power in the same window, not just assumed from the URDF axis).
`theta_hs = -0.253 rad` (std 0.125, n=6528 scheduled heel-strikes), `theta_to = -0.275 rad`
(std 0.073, n=6592 scheduled toe-offs); nominal standing -0.300. Peak terminal-stance power
(per-cycle, ReLU'd, `w=0.175`, n=4707 cycles) is heavily right-skewed: median 8.7W / p90
39.7W / max 149.3W — `P_scale` uses **p90 (40W), not the median**: at the median, today's
already-typical cycles would sit near saturation and kill the improvement gradient; p90
keeps the typical cycle (~9W) at only ~20% of saturation while the best cycles (40-150W)
sit at 63-98%. Free knobs (not probe-measured): `sigma=0.15 rad` (anchored to
`env_cfgs.py`'s existing `std_walking` ankle_pitch tolerance), `k=2`.

**Implementation:** `mdp.ankle_pushoff_pitchref` (A) + `mdp.ankle_pushoff_power` (B) added
to `rewards.py`, both reusing a factored-out `_gait_schedule()` helper (pulled out of
`feet_gait`'s inline phase/duty math, gate pin iii — verified byte-identical) so the duty
threshold is never a second hardcoded copy. `ll_pitchref_coef`/`ll_pushoff_coef` + their
shape params (`theta_hs`/`theta_to`/`sigma`/`k`, `w`/`p_scale`) added to `HrlRunnerCfg`
with the probe-derived defaults above; wired into `hrl_runner.py`'s LL intrinsic next to
the cadence term, gated on `hl_cadence` (same guard as arm 1, since both read
`env.hrl_phase`/`hrl_period`). Both default to 0.0; smoke-verified byte-identical baseline
at 0, sane nonzero values independently and together, and correctly inert (no crash) when
`hl_cadence=False`.

**Training (2026-07-17/18, LOCAL not cluster):** `a1a_cot0p2_cad0p5_pitchref0p5_s42`,
formulation A only (B stays untrained per the approved decision), `ll_pitchref_coef=0.5`
— **not probe-derived**, there is no A0 term to mirror the magnitude from, so it's matched
to `ll_cadence_coef`'s own weight as the closest analog (same phase-gated, bounded-[0,1]
kernel family) — flag this the same way arms 1/2/4's first-guess magnitudes are flagged:
untuned, expect a second pass. 10001 it / 4096 envs / seed 42, `model_10000`, 0 falls
throughout training (W&B `8gs72pzd`).

**Caveat on the comparator:** the cluster's own 15-run batch control hadn't synced to this
local machine at bench time, so **D2** (`2026-07-14_16-03-57_..._rs20_s42`) stands in — same
base-config lineage (arm-calmed weights + cadence-HL + cot0.2) but a different
checkpoint/seed lineage than whatever the cluster batch's own arm-6 slot converges to.
Re-diff against the cluster batch control once it lands.

**Bench (`model_10000`, 64x600x2, vs D2):**

| run | agg vx | agg vy | agg CoT | power (W) | fall | ss@0.5 (t90) | ss@1.0 (t90) | HL err vx | LL err vx |
|---|---|---|---|---|---|---|---|---|---|
| D2 (pre-arm6) | 0.064 | 0.055 | 0.898 | 316.5 | 0 | 0.034 (0.40s) | 0.041 (0.73s) | 0.055 | 0.064 |
| arm6-A (pitchref 0.5) | 0.099 | 0.082 | **0.707** | **231.5** | 0 | 0.066 (0.65s) | 0.098 (1.02s) | 0.114 | 0.139 |

**Roll-over character (replay, `vx=1.0` pinned, 64x900):**

| run | theta_hs | theta_to | ROM | RMSE to ref (k=2) | terminal-window dither |
|---|---|---|---|---|---|
| D2 (pre-arm6) | -0.252 ± 0.126 | -0.278 ± 0.073 | -0.026 | 0.114 rad | 0.330/step |
| arm6-A (pitchref 0.5) | -0.238 ± 0.027 | -0.280 ± 0.025 | -0.042 | **0.031 rad** | **0.254/step** |

Reads (honest, 2026-07-17/18): (i) **the roll-over character DID improve as intended** —
RMSE to the raised-cosine reference dropped 3.7x (0.114→0.031 rad), heel-strike angle
variance dropped 4.7x (std 0.126→0.027), and ROM grew 65% (-0.026→-0.042 rad): the LL now
reaches a materially more consistent, larger-swing ankle trajectory. The formulation-B
watch item (terminal-window ankle dither) went DOWN, not up, under A (0.330→0.254
sign-changes/step) — no early warning sign for B's later gate. (ii) **but tracking
regresses substantially and does NOT clear the winner bar** ("improves without regressing
holds/tracking/falls"): aggregate err_vx +55%, err_vy +49%; hold `ss_err` roughly doubles
at both 0.5 and 1.0 m/s with slower `t90`. The goal probe shows this is a real LL-capacity
conflict, not noise: LL reach error on vx nearly doubles (0.064→0.139) and the
realized/requested follow-through ratio collapses (0.743→0.130) — the LL now under-reaches
the HL's velocity delta far more than before. (iii) energy is a genuine, unambiguous win —
CoT 0.898→0.707 (-21%), power 316W→231W (-27%), plausibly because scheduled ankle push-off
recovers propulsion that used to come from elsewhere in the gait. (iv) falls: no
regression, 0 in every eval cell for both checkpoints. **Read: `coef=0.5` is too strong**
(never tuned — see the training note above) — the roll-over term is winning a real budget
fight against velocity tracking inside the shared LL capacity. A coef sweep (e.g.
0.1-0.3) is the natural next step before any verdict on formulation A; not run here (out
of this hand-off's scope — one training run, formulation A only, per the approved
decision). Planning-chat call on whether/how to sweep.

**Correction (2026-07-19-20, Liam's visual replay at pinned `vx=0.9`): the roll-over is
NOT visible** — reconciled with, not contradicting, the RMSE-to-reference win above. The
probe-measured reference itself has almost no amplitude: `theta_hs -0.253 -> theta_to
-0.275` is a ROM of ~0.022 rad (**1.3 degrees**), because it was measured off the
PRE-arm-6 gait, which barely rolled over in the first place (probe methodology,
2026-07-17: "measured... do not hand-pick" per the spec, applied to a gait that didn't
have the target behavior yet). Formulation A can therefore only train tighter conformance
to that near-flat reference — it structurally cannot induce a bigger, more human-like
push-off swing: post-training ROM only grew from ~1.5 to ~2.4 degrees (still far under
Winter-style human ankle push-off ROM, tens of degrees). The 3.7x RMSE improvement is
real (tighter, lower-variance tracking of the tiny target) but not what "fixes the named
defect" looks like visually. **Net read, revised: arm 6 formulation A, as specced and
probe-calibrated, does not fix the visual roll-over defect and costs real tracking —
weaker candidate than the other WL-D arms.** The crux is a chicken-and-egg limit in the
probe-first methodology itself (measuring the reference off a gait that doesn't yet do
the thing the reward is meant to induce) — whether to hand-pick a larger,
biomechanically-anchored reference instead (reopening the spec) is a planning-chat
decision, out of this hand-off's scope.
