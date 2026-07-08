# A2 — HIRO + A-RMA (design spec)

**Status:** spec / pre-implementation (2026-06-24). Task ID `Unitree-H1_2-Flat-A2`.
Reuses the shared HRL machinery in `.claude/docs/hrl-infra.md` (co-train loop, goal
space, warm-start, reward decomp, ONNX export) — read that first; this doc only
describes what A2 *adds* on top of A1.

## 1. What A2 is

A2 = **A1's HIRO hierarchy, unchanged, with A-RMA bolted onto the low level.** The
high level is fixed to A1's best config (TD3, `absolute` target mode); the only new
mechanism is an **extrinsics latent `z`** on the LL actor plus the 3-phase A-RMA
training schedule around the existing LL PPO. A2 differs from A1 by *only* the
encoder/latent/adaptation — so the clean RMA-mechanism comparison is **cell 4 vs
cell 2** of the 2×2 (both under wide DR; see §8), not A2-vs-A1-as-is.

Paper basis: A-RMA (Kumar et al., IROS 2022, arXiv:2205.15299) — RMA's 2 phases +
a 3rd model-free fine-tune of the base policy against the *imperfect* estimated
extrinsics. The Phase-3 fine-tune is A-RMA's actual contribution and is **not**
present in any workspace RMA repo (all stop at Phase 2).

## 2. Locked decisions (2026-06-24)

- **Scope: full 3-phase A-RMA** (Phase 1 encoder + Phase 2 adaptation + Phase 3 fine-tune).
- **HL = A1's best config (TD3, `absolute`).** *Co-trained* with the LL+encoder in
  Phase 1 (the standard A1 co-train loop — HL adapts to its own LL; all four 2×2 cells
  use the identical loop so the A1/A2 rows differ only by the encoder). **Frozen** only
  in Phase 3 (with φ). Supersedes the earlier "HL held fixed / frozen throughout"
  reading. No new freeze-HL machinery needed.
- **Evaluation = 2×2 factorial** (architecture × DR width), ADR 0001 — see §8.
- **e_t = parameter DR** (§3): friction + base CoM + per-joint `pd_gains` (motor
  strength) + per-joint `dof_damping` + actuation delay. **Base mass OUT.** Motor
  strength = **`pd_gains` (Kp/Kd)**, not effort limits. Actuation delay via mjlab's
  native `delay_max_lag` (quasi-static per episode). **Force perturbations = optional
  later extension, NOT initial scope** (§4.1). `e_t` read live from model fields.
- **Latent `z ∈ ℝ⁸`**; **adaptation φ = 1D-CNN over k=50 history** (≈1 s @ 50 Hz),
  ported from `corelllab/h12_locomotion_rma`.
- **φ input = proprio + action + goal** (goal included, sys-ID disambiguation;
  command/`z`/`e_t` excluded), validated by a φ±goal ablation.
- **Asymmetric LL critic:** actor gets `z=μ(e_t)`, critic gets raw `e_t`; `μ` trained
  via the actor path only.
- **Warm-start = converged A0 for all four cells** (z cols fresh); identical init.
- **Phase 3:** LL actor + critic trainable, critic stays privileged on `e_t`, HL + φ
  frozen, same wide DR, A1 PPO hyperparams.
- **Wide DR applied full-strength from iter 0**; reactive ramp only if S2 smoke shows
  low iter-0 ep_len.

## 3. Privileged environment factor vector e_t (baseline)

Sourced from the per-episode DR the env already samples (`velocity_env_cfg.py:186`)
plus two **config-only** additions (no new event code — the mjlab DR functions already
exist). Fixed ordering (an `RmaEtSpec`, mjlab-native):

| Component | Dims | Source (mjlab) | Status |
|---|---|---|---|
| Foot/ground friction | 1 | `dr.geom_friction` (narrow, exists) | reuse |
| Base CoM offset (x,y,z) | 3 | `dr.body_com_offset` (narrow, exists) | reuse |
| Per-joint motor strength = **PD gains (Kp/Kd)** | 27 | `dr.pd_gains` (wide, config-only) | add config |
| Per-joint **passive joint damping** (`dof_damping`) | 27 | `dr.joint_damping` (wide, config-only) | add config |
| **Actuation delay** (per-env command lag) | 1 | mjlab actuator `delay_max_lag` + `DelayBuffer` (wide, config-only) | add config |

- **Motor strength → `pd_gains` (Kp/Kd)** (decided 2026-06-24), the classic RMA motor-
  model randomization, not `effort_limits`.
- **Per-joint granularity** (27 each; mjlab `shared_random=False`) — free, matches real
  per-joint variation; the encoder compresses to `z=8` regardless.
- **Base mass is deliberately OUT** — not in the current DR set (only `base_com` is);
  keeping it out avoids silently widening narrow-DR.
- **"Damping" = passive joint damping (`dof_damping`)**, distinct from the controller
  Kd (which lives in `pd_gains`) — see glossary.
- **Actuation delay is in the wide-DR baseline** (decided 2026-06-24): config-only via
  mjlab's native actuator `delay_max_lag`/`DelayBuffer` (Isaac's `DelayedPDActuator`
  equivalent — no porting). Configure **quasi-static per episode** (resample lag at
  reset, not per physics step, via `delay_update_period`/`delay_hold_prob`) so φ can
  infer a stable value. Highest-value sim2real extrinsic; lets RMA *adapt to* latency,
  not just tolerate it. e_t carries the per-env lag (1 dim; revisit per-actuator-group
  in S0).

**Reference ranges (added 2026-07-07).** The OpenHomie-based H1-2 stack
(`corelllab/h12_loco_manipulation`, HomieRL config in its wandb logs) is a working
sim2real recipe for exactly our extrinsics; use it as the starting point for the wide
ranges in S0: `kp_range [0.9, 1.1]`, `kd_range [0.9, 1.1]` (motor strength),
`friction_range [0.1, 3.0]`, actuation delay ON, `actuation_offset [-0.05, 0.05] rad`,
`joint_injection [-0.05, 0.05]` (torque noise), `link_mass ×[0.8, 1.2]`,
`payload_mass [-10, +15] kg`, `com_displacement [-0.1, 0.1] m`. Their extras
(payload/link mass, torque injection) are candidates for the later e_t extension, not
baseline. Note for the normalization + any effort DR: the Unitree datasheet motor
limits are hip ~220, knee ~360, waist ~220, ankle ~75x2, shoulder ~120, elbow ~120,
wrist ~30 N.m; the URDF values we use as `effort_limit` (hip 200, knee 300, ankle 40,
shoulder 40/18, elbow 18, wrist 19) are much more conservative. Effort stays a
capacity knob, not a DR extrinsic (decided 2026-06-24), but the datasheet numbers are
the ceiling if effort is ever revisited.

Exact dims/ordering finalized in S0; `e_t` dim is **derived from the spec**, never
hardcoded (mirrors how `goal_dim` is derived from `goal_components`). The encoder
input is normalized to roughly `[-1,1]` per component using each DR range. All these
DR funcs write to per-env model fields → `e_t` is read live at obs time (no cache).

### 4.1 OPTIONAL LATER EXTENSION — external force perturbations

> Not in initial scope. Reason: force/payload adaptation only matters once the robot
> is asked to **carry something**, which is not currently planned. Documented here so
> it can be switched on later without redesign.

If a payload/manipulation scenario is added, extend `e_t` with a force block —
torso + L/R wrist forces (spherical-sampled), the approach validated on real H1-2 in
`corelllab/h12_locomotion_rma` (e_t = torso(3) + L/R wrist(3+3) = 9, ±100 N,
resample p≈0.01) and `corelllab/h12_rma`. Design hooks to keep this cheap:
- `RmaEtSpec` is a dataclass with per-block slices → append a `force` block, `dim`
  re-derives, encoder input dim follows automatically.
- Add a `push_force` EventTerm applying the sampled force to `torso_link` /
  `*_wrist_*_link` each step (corelllab's `gym_et_builder.py` is the reference).
- Adaptation φ and LL obs need no structural change (only the e_t/encoder-input dim
  grows). z stays 8-dim.

This is a **config-flagged add-on** (e.g. `--agent.rma.force-perturbations`), default
off → byte-identical to the param-DR baseline (RQ2-safe, same discipline as
`GoalStateNoise`).

## 4. Architecture

**LL actor obs** = `proprio∖command ⊕ goal ⊕ z` — A1's LL obs with `z (8)` appended.
HL obs unchanged. **LL critic is asymmetric** (decided 2026-06-24): it takes the raw
`e_t` appended to A1's existing privileged critic obs (`base_lin_vel` + foot terms) —
not `z`. So the encoder `μ` is trained **only through the actor's policy-gradient
path** (learns an encoding useful for acting); the critic sees exact extrinsics for a
lower-variance value. Privileged, never deployed.

- **Env-factor encoder μ** (Phase 1): MLP `e_t → 256 → z(8)`, trained jointly with
  LL PPO (gradients flow z → LL actor). New module `rl/hrl/rma/env_factor_encoder.py`.
- **Adaptation module φ** (Phase 2): **port `Adaptation1DCNN` from
  `corelllab/h12_locomotion_rma/rma/adaptation_module.py`** ~verbatim (pure torch,
  paper-faithful: 2-layer MLP embed→32; Conv1d k=8/s=4, k=5/s=1, k=5/s=1; history
  **k=50** ≈ 1 s @ 50 Hz; latent 8; flat 96).
- **φ input = history of (LL proprio obs `policy` group + LL action + goal)**, k=50.
  **Goal is included** (decided 2026-06-24) — sys-ID view: the goal contextualizes the
  proprio response so extrinsics are more identifiable ("commanded fast, moved slow →
  weak motor"), and the goal is the architecture's sanctioned structured channel
  (HL-produced at deploy → no distribution shift; deployable). **Command, `z`, `e_t`
  stay excluded.** Caveat: the regression target `z=μ(e_t)` is goal-independent, so the
  goal only helps via disambiguation — validated by the φ±goal ablation (§7). Impl note
  (S3): `policy` already contains `last_action`, so don't double-feed the action.
- **History buffer**: a `(num_envs, k, policy+action+goal)` ring buffer the runner
  fills each LL step (analogous to how the runner writes `env.hrl_goal`); zero-padded /
  cleared on episode reset.

## 5. Three phases

1. **Phase 1 — joint co-train of HL + LL + encoder.** Run the standard A1 co-train
   loop (HL TD3 co-trained, LL PPO every step) with the LL actor augmented by
   `z = μ(e_t)`; μ optimized with the LL PPO loss. Warm-start the LL from A0
   (gap-aware; `z` cols init fresh, same trick as the goal cols). Output: teacher
   policy + encoder (+ co-trained HL).
2. **Phase 2 — adaptation distillation.** Roll out the frozen Phase-1 policy; φ
   regresses `ẑ = φ(history)` to the teacher's `z = μ(e_t)` via MSE. No privileged
   access at φ's input → deployable. Reference: corelllab `phase2_runner.py`.
3. **Phase 3 — A-RMA fine-tune.** Reload Phase-1 LL, **freeze the HL and φ**, swap the
   actor's extrinsics channel `z ← ẑ`, and continue PPO under the **same wide DR** so
   the policy adapts to φ's *imperfect* estimate. **LL actor + critic both trainable**
   (the critic re-adapts to the ẑ-conditioned actor; a frozen Phase-1 critic would be
   stale); the **critic stays privileged on raw `e_t`** (still in sim — never used `z`
   anyway). Inherits A1's LL PPO hyperparams (`entropy_coef=0.005`). This is the
   headline; only the actor's extrinsics input changed vs Phase 1.

## 6. Implementation stages (each gated on go-ahead)

| Stage | Deliverable |
|---|---|
| **S0** | Privileged-e_t plumbing: `RmaEtSpec`, **config** the `dr.pd_gains` + `dr.joint_damping` events (no new DR code), an `mdp.rma_extrinsics` obs term, + the φ history ring buffer. **Confirmed (2026-06-24):** mjlab DR funcs route through `_randomize_model_field`, which writes sampled values into per-env model fields (`geom_friction`, `body_ipos`, body mass, actuator gains). So `e_t` is read **directly from live model fields at obs time** (deviation-from-nominal, normalized) — **no separate sample-time cache buffer needed**. |
| **S1** | `config/h1_2_a2/{__init__,env_cfgs,rl_cfg}.py` registering `Unitree-H1_2-Flat-A2` (runner `A2Runner`); obs-group restructure adds `extrinsics`; enables the A2 DR set. |
| **S2** | `A2Runner(HierarchicalRunner)` — builds z=μ(e_t), concats into LL actor obs, joint-optimizes μ with LL PPO; gap-aware warm-start extended for z cols; save/load adds `encoder`/`adaptation` keys. **(Phase 1)** |
| **S3** | Phase-2 trainer (port `Adaptation1DCNN` + corelllab phase-2 loop to mjlab). **(Phase 2)** |
| **S4** | Phase-3 fine-tune path (`--agent.rma-phase 3`): load Phase-1 LL + frozen φ, feed ẑ, continue **LL-only** PPO. HL frozen via `hl.eval_mode()` + skip `hl.update()` (small runner branch; not the full `freeze_ll` path). **(Phase 3)** |
| **S5** | Play / benchmark / ONNX: `get_inference_policy` runs φ (ẑ from history); export 3 nets (HL, LL base, φ). **Deploy implication (follow-up, not blocking sim):** the C++ two-ONNX HIRO stack must (a) run 3 nets, (b) maintain φ's rolling **50-step (proprio+action+goal) ring buffer** onboard, (c) route the HL goal into *both* the LL and φ. |

## 7. Test plan (defined up front)

- **S0/S1 smoke:** env builds; LL obs dim = derived (+z 8); `e_t` term returns finite,
  per-env-distinct values; 50-iter run no NaN. Delete smoke runs after.
- **S2 (Phase 1):** **wide DR applied full-strength from iter 0** (no curriculum).
  Stability gate = **high iter-0 ep_len** (warm-start intact — same A1 diagnostic); if
  iter-0 ep_len is low (early falls under the harsher new extrinsics: weak Kp / high
  damping / 5-step delay), **contingency = ramp** the *new* spreads (Kp/Kd, damping,
  `delay_max_lag`) from ~0 to full over the first chunk of Phase 1, holding friction/CoM
  at A1 levels; the ramp must be **identical across wide-DR cells 2 & 4** and reach full
  strength well before the end. LL then converges to ≈A1 tracking under DR. (RMA-
  mechanism control is the **blind A1 cell under identical DR** — §8's 2×2, not a
  z-zeroed hack.)
- **S3 (Phase 2):** MSE(ẑ, z) drops to a plateau; ẑ correlates with held-out e_t.
  **φ±goal ablation:** train φ with and without the goal channel; compare Phase-2 MSE
  and downstream `fall_rate` to confirm goal-in-φ earns its place (else it was neutral
  and the proprio trace alone sufficed).
- **S4 (headline):** A-RMA-finetuned policy **beats Phase-1-with-ẑ** (the imperfect-
  estimator gap) on the deterministic benchmark (`fall_rate`, err_vx/vy/yaw) under
  held-out DR. ≥2 seeds, `num_envs` fixed. Judge by the benchmark scorecard, not
  training err.

## 8. RQ2 / comparison cleanliness

A2 introduces heavier DR + a privileged channel vs A0/A1. To attribute effects
cleanly we run a **2×2 factorial** (architecture × DR width), decided 2026-06-24
(ADR 0001):

| | **narrow DR** (A1's current set) | **wide DR** (+motor-strength, +damping) |
|---|---|---|
| **A1** (HIRO, no encoder) | cell 1 — existing A1 baseline | cell 2 — retrain A1 @ wide DR |
| **A2** (HIRO + A-RMA) | cell 3 — A2 @ narrow DR | cell 4 — main A2 |

Wide DR = narrow (friction, base CoM, push) + per-joint `pd_gains` + per-joint
`dof_damping` + actuation delay. Narrow DR = A1's current set.

- **DR-width main effect** (cols): 2 vs 1, 4 vs 3. **RMA-mechanism main effect**
  (rows): 3 vs 1, 4 vs 2. **Interaction** (4−3) vs (2−1) — *does RMA help more as DR
  widens?* — is the headline result.
- **A2's network is identical across cells 3 & 4:** `e_t` always covers the full
  parameter set; in narrow-DR cells the un-randomized dims sit at nominal (constant),
  so cells differ only by DR ranges, not net shape.
- **Identical LL init across all four cells:** every cell warm-starts the LL from the
  same converged A0 (gap-aware; A2 cells additionally init `z` cols fresh) → rows stay
  symmetric, no init confound.
- **Equal-compute fairness:** match total environment steps across all four cells
  (A2 spends Phase-1 + Phase-3; A1 gets a comparably long budget) → policy-quality-at-
  equal-compute, not "A2 trained longer."
- **Eval on a held-out *wide* DR set for all four cells** (incl. narrow-trained), using
  `fall_rate` + err_vx/vy/yaw; ≥2 seeds, `num_envs` fixed. Robustness is measured vs
  unseen randomization, where RMA earns its keep.

Env reward terms, goal space, `c`, and gamma derivation are unchanged across all cells.
A1's #8b `GoalStateNoise` (goal-channel *estimator* noise) is **orthogonal** to A2's
extrinsics DR and stays **off in all four cells** (its default) so it isn't a 2×2
confound — A2 is about *dynamics* extrinsics, not goal estimation.

## 9. Reuse ledger

| Need | From |
|---|---|
| Hierarchy, co-train loop, warm-start, reward decomp, ONNX | A1 / `hrl-infra.md` |
| Adaptation 1D-CNN + Phase-2 distillation loop | `corelllab/h12_locomotion_rma` (port to mjlab) |
| DR events (friction, CoM, push) | `velocity_env_cfg.py` |
| e_t spec idea, force-perturbation extension | `corelllab/h12_rma`, `h12_locomotion_rma` |
| Deploy (MuJoCo/ROS2) reference for S5 | `corelllab/h12_adaptive_policy`, `h12_ros2_controller` |

None of the workspace RMA repos is drop-in (all legged_gym/Isaac Gym + rsl_rl, not
mjlab; all flat RMA without HIRO; none implement Phase 3). See
`.claude/docs/` survey + memory `rma_repos_workspace` for the full assessment.
