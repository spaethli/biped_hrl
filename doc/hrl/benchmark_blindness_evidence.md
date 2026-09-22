# Benchmark blindness: the verified evidence base

Every number the thesis's "What the Benchmark Cannot See" chapter rests on, with its
primary source, so the chapter can be written without a re-verification pass. Verified
2026-09-21/22 by reading the cited files, not from narrative.

**Two claims carry corrections that must travel with them.** They are marked ⚠ and the
correction is stated next to the claim, not in a footnote.

The organising claim: a simulation benchmark scores the regime it samples. A policy's
quality can live in a reserve, a transient, or a bifurcation that the sampled regime never
exercises, and in that case the benchmark is not noisy about it, it is silent about it.

---

## §6.1 The instrument, and what it measures reliably

State the benchmark's validated authority before its boundary, or the chapter reads as an
apology rather than as characterisation.

| Quantity | Value | Source |
|---|---|---|
| Re-measuring the SAME checkpoint | reproduces to **1.6-4.2%** | `standing_metrics_replicate_spread` memory |
| 64-env measurement floor | **~2.6%** | `bench_num_envs_is_the_measurement` memory |
| Commanded ankle range, sim vs hardware | agree to **a few percent** | `docs/adr/0009`, Evidence tables |

The third is the strongest single statement of the bench's validity: a quantity defined
identically in sim and on the robot, measured in both, agreeing. Use it to establish that
what follows is not "simulation is unreliable".

⚠ Keep this separate from §6.3. This is **measurement** noise (same checkpoint, re-scored).
§6.3 is **training** noise (same config and seed, re-trained), and it is far larger.

---

## §6.2 When the harness changes what it measures

**The strongest instance. Lead with it.**

The coupling is one line. `play.py:297-299` collapses the twist ranges to a point to pin a
hold command (`lin_vel_x=(vx,vx)`); `goal_space.py:79-83` derived the HIRO goal scale from
those same live ranges (`half = max((hi-lo)/2, 1e-3)`). So the scale went
`[0.75, 0.5, 1.0]` to `[1e-3, 1e-3, 1e-3]`, and `V* = s_t + 1e-3·g ≈ s_t` for **any** `g`.
The high level's action was disconnected from the low level, and the composed system was
the "driftless random walk" that had been observed and attributed to the hierarchy.

Cured on the **unchanged checkpoint** (no retrain, the fix is where the scale is read):

| | before | after |
|---|---|---|
| `\|g\|_vx` (saturation) | 0.98 (95.6% saturated) | **0.219** (4.6%) |
| `hl_err_vx` | 0.912 | **0.132** |
| hold `ss_err_vx` @ 0.5 m/s | 0.304 (`t90` never) | **0.044** (0.56 s) |
| hold `ss_err_vx` @ 1.0 m/s | 0.882 (`t90` never) | **0.097** (0.76 s) |
| direction split | 38 fwd / 19 bwd | **64 / 64 fwd** |

**A0 was structurally immune**: it reads the command directly and has no goal space. That
is why the defect looked "isolated to the hierarchy" and why four other hypotheses
(duration, heading, exact-zero commands, eval dither) all failed to move it.

**And the ranking inverted.** D1/D2 went from the set's *worst* holders (0.376/1.033 and
0.470/0.851) to its *best* (0.030/0.039 and 0.034/0.042).

**It is one of a family.** The findings ledger calls it "third of the family", and two more
were found in the same period: a missing `hl_obs_vel` absence-shim that mis-built every
pre-velobs HL network (94-dim vs the saved 92), and a `--diagnose-goals` loop that never
advanced `env.hrl_phase`, so every probe reading before 2026-07-15 ran on a frozen,
de-entrained, flattering gait clock.

**Root lesson, in the ledger's own words:** deriving the HL's action scale from a mutable
env config couples policy semantics to eval and deploy knobs. The scale belongs in the
checkpoint, like the normalizer. Fix shipped as `GoalSpace.freeze_scale` + save/load +
ONNX metadata.

*Source:* the A1 findings ledger (research KB), WL-C row; `doc/hrl/HRL_plan.md` A1 row;
`.claude/docs/hrl-infra.md:381`.

---

## §6.3 When the effect is smaller than the training noise

Two accidental config-identical, same-seed-42 replicate pairs (`repro1/repro2`,
`jacc1/jacc2`, from a flag that never applied). Max/min ratio:

| metric | walking | standing |
|---|---|---|
| `act_legs` | 1.08-1.10x | 1.21-2.23x |
| `ajit` | 1.02-1.31x | **1.19-4.15x** |
| `ajit_p95` | — | **1.00-5.16x** |
| `jacc_legs` | 1.03-1.08x | 1.74-2.43x |
| touchdowns | — | **2.30-4.22x** |
| `\|g\|vy` | — | 1.37-3.57x |
| `ss_err_vx` (hold) | **1.38-3.32x** | — |
| aggregate `err_vx` | **1.12-1.94x** | — |

**Sharpest case:** `repro1` tracks the 0.5 m/s hold (`ss_err_vx` 0.0653, `t90` 0.58 s);
`repro2` never reaches 90% of it (0.2167, `t90` undefined). Same launch command.

**Counterintuitive and worth stating:** seed variation is **smaller** than same-seed
non-determinism. Standing `ajit` across seeds 42/123 spans 2.68x at `rse` 0.15 and 1.40x at
0.20, against the 4.15x same-seed spread. A second seed is still worth running, but seed
choice is not the dominant noise source.

**Mechanism:** standing is a near-marginal equilibrium. Quiet double-support versus a limit
cycle of corrective steps is a bifurcation, and GPU non-determinism lands either side.
Walking is a limit cycle either way, so its metrics are continuous in the perturbation.

**What it cost:** `ll_stand_still_coef` 2.0 became a null result (10 of 12 cells inside
their band), and the r=-0.997 smoothness/tracking frontier lost its precision, since its
tracking axis has a 1.94x replicate spread and it was read to three decimals across five
n=1 points.

*Source:* `standing_metrics_replicate_spread` memory (2026-08-27/28).

---

## §6.4 When the benchmark never exercises the property

### Instance A: Model v3, ADR-0009

Every pre-registered metric **improved** under v3:

| metric | v2 | v3 |
|---|---|---|
| `err_vx` | 0.0941 | **0.0903** |
| `act_legs` | 0.600 | **0.567** |
| `jacc_legs_p95` | 78.6 | **62.1** |
| `fall_rate` | 0 | 0 |

The cost, invisible to all of them: **2.1-2.4x of the commanded ankle roll range**, the
joint that rejects lateral disturbance.

| arm | plant | roll span L / R |
|---|---|---|
| v2 anchor (sim) | v2 | 0.345 / 0.331 |
| v3 (sim) | v3 | **0.154 / 0.141** |
| v2 (hardware, 2 sessions) | — | 0.413/0.505, 0.436/0.373 |
| v3 (hardware) | — | **0.192 / 0.223** |

Operator verdict: v2 stood better, resisted pushes better, walked better; **v3 was too
unstable to walk at all**.

**The quotable sentence:** *"A reserve is invisible to any metric collected in the regime
where the reserve is not needed."*

⚠ **Correction 2026-09-07 — do NOT state the mechanism.** The ADR originally explained this
as "sim standing applies no lateral disturbance" and prescribed adding a disturbance term.
**That is false.** `push_robot` (`push_by_setting_velocity`) fires every **5-6 s** in
training with linear `y` in **±0.5 m/s** and roll rate **±0.52 rad/s**, and `play.py`
removes only `base_mass` from the event set, so the bench applies it too. The ADR says
literally: *"Do not cite 'sim applies no lateral disturbance' — it is false."* The
conclusion stands; the mechanism is **open**. What is missing is a metric scoring the
**response** to the push that already happens (recovery time, peak ankle-roll demand,
lateral excursion), not pooled statistics over a rollout where pushes are a small minority
of steps.

⚠ Second caveat from the same ADR: *"The step from 'less range' to 'worse push rejection'
is an inference."* It rests on the operator's hardware observation, n=1 session for v3, one
training seed per arm.

*Source:* `docs/adr/0009-revert-model-v3-action-clip.md`.

### Instance B: A1a versus the payload-DR arms, ADR-0013

Base cell, 0 kg, pooling `data/2026-09-09-wp3-baseline-arms/{all,nodr,cotcap}_bench.json`:

| arm | `ub_arm_vel` | `ub_pose_dev` | `act_legs` | `err_vx` |
|---|---|---|---|---|
| A0_DR (flat) | 0.120 / 0.124 | 0.00034 | 0.578 / 0.580 | 0.086 / 0.086 |
| **A1a** | 0.307 / 0.316 | 0.0083 / 0.0114 | **0.618 / 0.627** | **0.127 / 0.126** |
| A1a_DR | 0.322 / 0.331 | 0.0110 / 0.0103 | 0.588 / 0.614 | 0.117 / 0.134 |
| A1a_DR_cotcap | 0.320 / 0.305 | 0.0106 / 0.0109 | 0.582 / 0.582 | 0.104 / 0.112 |

A1a sits **inside** the DR arms' range on both upper-body metrics and is the **worst** of
the four hierarchical arms on `act_legs` and `err_vx`. The large upper-body gap (2.7x
`ub_arm_vel`, 30x `ub_pose_dev`) is hierarchy-versus-flat, and A1a is on the hierarchical
side of it.

**ADR-0013's own framing, worth quoting:** *"Once is an accident; twice is a property of the
benchmark."*

⚠ **Narrowed by ADR-0013's Amendment (2026-09-18):** the hardware result *"does not
strengthen as a DR effect. It holds for seed 123 only."* So the benchmark's blindness is
real, but what it was blind to is "one seed of two DR recipes behaved differently on
hardware", **not** "A1a is better than the DR arms". State it that way.

*Source:* `docs/adr/0013-keep-a1a-as-the-deploy-policy.md` + its 2026-09-18 amendment.

---

## §6.5 When the control differs on the axis being claimed

The A0 anchor trained at `rel_standing_envs` **0.05**; A1a trained at **0.12-0.20**. Every
"A1a stands quieter than A0" ratio from 2026-09-05 on was therefore measured against a
baseline that had seen roughly a third as much standing, **on the exact metric the claim
was about**.

Re-benched with a matched A0 (2026-09-21, `data/2026-09-21-rse-matched-baseline/`):

| | result |
|---|---|
| A0 itself, 0.12 vs 0.05 | **0.80x** as loud |
| deployed A1a vs A0 | 0.94x → **1.17x** (crosses from a win to a loss) |

**The valuable half: the matched control SEPARATES real from confounded.** Standing
`ajit_p95` moved only **1.01x** between the two A0 fractions while A1a sits at **10-18x**,
so *that* gap is architectural and survives matching. Run the control precisely so the two
kinds can be told apart.

**Why it stayed hidden, and this is the generalisable part:** `rel_standing_envs` is set
per run on the **launch command line**, not in a config file. The repo default is 0.05 for
both architectures and is byte-identical to upstream. A reader diffing the two config trees
would find them identical. **The tell is a lever that was added for one architecture:
nobody retrains the baseline when a new knob lands, so the baseline silently freezes at the
old default.**

*Source:* `control_must_match_on_the_claimed_axis` memory; `velocity_env_cfg.py:187`;
keeper `params/env.yaml:1757`.

---

## §6.6 When the stopping rule watches the wrong quantity

ADR-0004's S5 pre-registered guard on `hl_cot_coef` was **`fall_rate` must stay flat**.

Amended 2026-09-01: the usable cap is **5-10**, and coefficients **10 and 20 hold
`fall_rate` at 0 while running ~5x A0's `err_vx`**. The ADR's own words: *"the arm walks
smoothly and slowly rather than falling, so S5's stated guard would have passed it."*

**Rule:** pair every smoothness or energy weight guard with a **tracking floor**
(`err_vx` / `ss_err_vx`).

**Companion instance — stand pitch across policies.** ADR-0006 (2026-08-03): stillness
predicts the lean at **r = -0.96** across three A0 variants, at slope **-0.0739 deg per
stillness-point**; a policy that fidgets settles into a less-drooped stance and reads less
lean. Any cross-policy lean comparison must report stillness alongside or it is void. This
caught a live over-claim on 2026-09-21 within one turn, and the corrected result is in
`docs/adr/0006` Amendment (2026-09-21): the lean is an **A0 seed effect**, flat spanning
3.87° across two seeds *with* its calibration enabled against 0.87° across six hierarchical
recipe/seed combinations *without* it, and 88% of the widest gap unexplained by stillness.
Scored by `scripts/score_stand_attitude.py`, which enforces the rule in code.

*Source:* `docs/adr/0004` Amendments (2026-09-01), lines 133-135, 180-182;
`docs/adr/0006` (2026-08-03 confound + 2026-09-21 amendment).

---

## §6.7 The rules

1. **Report the confounder with the metric, not after it.** (§6.6)
2. **Pair every magnitude guard with an error floor.** `fall_rate` alone passed a policy at
   5x the baseline's tracking error. (§6.6)
3. **Split regimes; never pool.** A joystick hardware session is 86-95% standing while the
   sim bench is ~95% walking; pooling reads a session as 4.5x smoother than it is.
4. **Match the control on the axis the claim names.** Diff both runs' `params/` yaml on the
   knob the metric is named after, not just the eval protocol. (§6.5)
5. **Attach a measured noise floor to every threshold.** A standing effect below ~2x is
   unresolvable at n=1. (§6.3)
6. **A coefficient is calibrated against a REWARD, not a task.** The CoT denominator changed
   on 2026-07-10 from an undirected speed integral to a signed projection; the 0.2 keeper and
   the sweep that ruled out higher values both predate it and then rode ~40 runs unchanged.
   Re-swept, the **sign flips**. Reformulating a reward term reverts every weight tuned
   against it, and every sweep that ruled out neighbours, to unvalidated.
7. **Score the signal the consumer actually receives.** WL-F: the HL consumes a `c=8`
   averaged velocity estimate, and that averaging cancels the policy coupling the whole
   workline rested on. Raw per-step error spans **7.62x** across commands (0.0703→0.5359)
   while the HL-consumed error is nearly flat at **1.55x** (0.0506→0.0782). Right shape
   (+0.213 vs +0.0006 correlation), right size, **wrong consumer**. Before investing in a
   better estimator, score the rung the consumer reads; the two disagree in direction.

---

## Figures

None of the ten figures in `scripts/plot_thesis_figures.py` cover this chapter; they are all
hardware Run bundles. `f_chain` (training bench → bridge → hardware) is the nearest relative
and is worth **reusing in §6.1** as the positive statement of cross-plant agreement.

Four new figures, specified against data that already exists:

| id | shows | form | data |
|---|---|---|---|
| `f_blind_v3` | §6.4A. Four pre-registered metrics all improving under v3 (normalised to v2=1.0) beside the ankle roll span collapsing to 0.45x, in sim and on hardware | grouped bars, two panels sharing a normalised y axis, a horizontal line at 1.0 | `docs/adr/0009` tables (hand-entered; the source captures are gitignored) |
| `f_blind_scale` | §6.2. The same unchanged checkpoint before and after the goal-scale fix, on the four hold metrics plus the 38/19 → 64/64 direction split | paired before/after dumbbells, log x for the hold errors | A1 findings ledger WL-C row |
| `f_blind_band` | §6.3. Replicate spread by regime: walking ratios clustered near 1.0, standing ratios fanning to 4-5x, with the seed-to-seed spread overlaid to show it is smaller | dot plot, metric on y, ratio on x, log scale | `standing_metrics_replicate_spread` memory table |
| `f_blind_control` | §6.5. `rel_standing_envs` 0.05 vs 0.12 for A0 and A1a, on the metric that moves (`act_legs`, crossing parity) and the one that does not (`ajit_p95`, 1.01x vs 10-18x) | two panels, parity line at 1.0 | `data/2026-09-21-rse-matched-baseline/*.log` |

`f_blind_control` is the only one whose source is machine-readable today; the other three
are small enough to carry their numbers in the figure script as literals, the way
`test_stand_attitude.py` carries its golden values, with the source cited per series.

⚠ `scripts/plot_thesis_figures.py` was under concurrent edit by another session on
2026-09-21, so these were NOT added to it. Add them there once that work lands, rather than
starting a second figure script, so the whole thesis keeps one pgf/CSV convention.
