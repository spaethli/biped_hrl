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

| Stage | Action | Success criterion | Status |
|---|---|---|---|
| **M0** | Add `mech_power_w` / `cot` / `stride_period_s` to the benchmark (`play.py`) | metrics print, no NaN, A0 baseline captured | ✅ 2026-06-30 |
| **S0** | A0 baseline cadence + CoT; CoT-vs-period probe | baseline captured; CoT(period) curve **deferred to S2** (A0 can't vary cadence) | 🟡 baseline done |
| **S1a** | LL-side cadence machinery in the **training** loop (period/phase buffers, `mdp.phase` + `feet_gait` rewire, `ll_cadence` intrinsic, random-period source) | smoke: new path runs + `cadence_rew` active; defaults inert (`cadence_rew=0`, intrinsic unchanged) | ✅ 2026-06-30 |
| **S1b** | Cadence in the **inference** path (`get_inference_policy` + benchmark) + `--eval-cadence-period` pin for the CoT(period) sweep | smoke: cadence ckpt benchmarks via inference path, period pin applied, no error (entrainment accuracy is S2's, untrained LL stride 0.76 vs cmd 0.60) | ✅ 2026-06-30 |
| **S1c** | CoT HL reward + learned-cadence action dim (`goal_dim+1`) | phase continuity on mid-window period change; CoT gate/floor numerics; action dim grows only when on | ⬜ |
| **S2** | Train A1a LL under **per-episode** random-period HL | achieved `stride_period_s` tracks commanded period; velocity tracking **and survival** ≥ current A1 | 🔴 v1 FAILED → 🟡 v2: healthy walker (err 0.11, 0 falls, CoT 0.96) but **no entrainment** (stride flat 0.573 = natural; range too narrow). → 🟢 v3 running (`a1a_s2_cadence_v3`, resume v2, wide 0.35–1.0, coef 0.5). |
| **S3** | Frozen-LL learned HL (TD3) + `hl_cot_coef>0` | HL drives `cot` below fixed-0.6 baseline at A1-level tracking; converged period = `CoT(period)` min, speed-dependent | ⬜ |
| **S4** | Co-trained A1a vs A0 / A0+energy / A1 (≥2 seeds) | A1a `cot` < A0 at A0-level tracking + fall_rate; A0+energy regresses tracking. Honest disconfirmer logged if A0+energy matches | ⬜ |
| **S5** | Sweep `hl_cot_coef` | fall_rate flat as weight rises; cap below any rise (suicide-attractor guard) | ⬜ |
| **S6** | OOD proxy: Narrow→Wide DR / push / terrain (secondary) | exploratory; big edge not expected pre-A2. S4 is the load-bearing result | ⬜ |

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

**S2 v2 (2026-07-01)** — `a1a_s2_cadence_v2`, per-episode period, narrow (0.5, 0.7). Fixed v1's
collapse: eval (deterministic) is a **healthy walker** — err_vx 0.11 / err_yaw 0.15 (≈A0), 0
falls, full episodes, CoT 0.96. But **no real entrainment**: `stride_s` flat at 0.573 (= natural
~A0 gait) for every commanded period. The narrow band straddles the natural cadence, so the LL
scores high `cadence_rew` (0.84) *without varying cadence* — no gradient to entrain. (The scary
training `error_vel_xy 0.745` was a stochastic-policy artifact; `action std 0.93`. Deterministic
tracks fine.) **Go/no-go still open** — CoT is flat only because actual cadence never moved. Fix
(v3): widen the range (curriculum from v2) so the natural gait can't satisfy the extremes and
entrainment is forced. Watch the `v=f·L` coupling (off-natural cadence may cost tracking).

## Open knobs (set by data, not guessed)
- `cadence_period_range` — from S0 (A0 walkable band ∩ `CoT(period)` minimum). Start `(0.5, 1.4)`.
- `hl_cot_coef` — bounded by S5 (highest weight that keeps fall_rate flat).

## Honest caveats
- The **definitive `CoT(period)` curve needs the cadence-capable LL (S2+)**; the warm-start LL
  only knows ~0.6 s, so S0's period probe is local/limited.
- If `A0+energy` matches A1a in S4, the in-sim efficiency benefit is **not** from the hierarchy;
  report it and lean on the OOD/A2 case. Stated up front to avoid motivated reading.
