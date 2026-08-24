# Parked hand-off prompts (specced 2026-08-07/09, blocked on WL-F)

Three worklines were fully specced during the 2026-08-07 grill and then parked behind
**WL-F** (HL velocity estimator) when the 2026-08-09 hardware session showed leg odometry
is a closed feedback loop. WL-F is taking longer than expected, so the prompts are written
down here rather than held in a chat.

**Status: ready to deliver, not delivered.** Copy the block, hand it to a fresh chat, then
move the prompt out of this file and record the delegation in `worklines.md`.

**Sequencing.** WL-T lands first (WL-S rebases onto its instrument). WL-V can run in
parallel with either — it is bridge-serial and touches no training code. None of the three
needs WL-F to *finish*, but WL-S's arms should be trained on whatever HL velocity source
WL-F settles on, or they will need re-running.

---

## Shared facts all three prompts depend on

**Leg-only is the deploy-relevant lens.** `hold_joint_ids: [12..26]` freezes waist and both
arms on hardware, so their share of `action_rate` is deploy-irrelevant. `[BENCH]` now emits
`act_legs` (legs-only commanded action rate) and `jacc`/`jacc_legs`/`jacc_arms` (+`_p95`,
realized joint acceleration).

**Rank on `act_legs` AND `jacc_legs_p95`, never `action_rate` alone.** The realized-motion
gap is much larger than the commanded one (vs A0: `jacc` 1.78-2.63x, `action_rate`
1.30-1.67x), A0 and A1 have opposite acceleration profiles (A0 leg p95/mean 2.39 vs A1
~1.17 — only p95 reveals it), and combo1 is **best** on `action_rate` while **worst of 13**
on `jacc_legs_p95`.

**Anchors** (`g_legs` = `act_legs`): A0 deployed **0.5960**, A0 `fric0p1_kl01` 0.5859,
`full_jacc` 0.6970, keeper/arm4d 0.7570, blind HL 0.8124, keeper+jitter 0.8363.
`err_vx`: A0 0.0838, jitter keeper 0.0736, blind 0.1219.

**Noise floor: ±2.6% at 64 envs**; 1 env is bimodal, not merely noisy. Rankings inside ~3%
are not resolved at n=1 seed. `play.py` eval paths now default `num_envs` to 64; numbers
taken at different env counts are never comparable — check the `num_envs` field `[BENCH]`
records.

**Battery data** (10 reward-mirror arms, bench + velinc, with a README on reuse caveats):
`~/biped_hrl_wiki/raw/engineering-journal/data/2026-08-06-wld-mirror-bench/`.

---

## WL-V — validate the reward-mirror battery on the bridge

**Model: sonnet.** Well-scoped, fixed protocol, objective outputs, no mechanism reasoning.

### Task

Bridge-validate three arms of the reward-mirror battery, after making bridge sessions
survive a fall. Per CLAUDE.md: analysis → spec → approval → implement → test.

### Read first

- `doc/hrl/A1a_deploy_plan.md`, "HOW TO RUN THE GATES"
- `~/biped_hrl_wiki/raw/engineering-journal/data/2026-08-06-wld-mirror-bench/README.md`
  — the sim stage is already measured there; cite it when skipping that stage
- `~/biped_hrl_wiki/raw/engineering-journal/data/2026-08-09-hl-vel-estimator-hardware/README.md`
  — why a bridge GO is **not** evidence of deployability

### Tasks in order

**1. Fall-detect-and-restand in `scripts/bridge_session.py`.** Today the script presses FSM
keys on a fixed schedule with no fall detection anywhere in the run loop, so one fall costs
every later phase — the 2026-08-06 candidate session was 82% ground rows after falling at
t=16 s. On `gt_h < 0.9`, drive `p` → `i` → re-enter the policy before the next command
phase.

Three constraints: **sim-bridge only** (an autonomous re-stand of a fallen H1-2 is not
safe), **default off**, and **it must not soften any gate** — "no fall in any bridge phase"
stays blocking. Recovery buys per-phase evidence, not forgiveness. A run with three
recovered falls is still NO-GO; you just also learn which three phases.

*Verifiable output:* a session that falls deliberately (any known-bad checkpoint) produces
scored data for every phase after the fall, and the verdict is still NO-GO.

**2. Bridge three arms, n ≥ 2 each.** `full_jacc`, `full_jacc_noEnergy`, `full_noEnergy`
(checkpoint paths in the battery CSV). The spread is deliberate: `full_jacc` is
best-legs/short-stride (0.697, 0.365 s), `full_noEnergy` is worse-legs/long-stride (0.733,
0.506 s), so the pair doubles as a stride-vs-smoothness discriminator.

**n ≥ 2 is not optional.** The `pin_period 0.625` discriminator ran twice on the same
config and same policy and disagreed with itself — r1 never fell in 49.5 s, r2 fell at
26.12 s. A single bridge run cannot support a verdict.

```bash
python scripts/deploy_readiness.py Unitree-H1_2-Flat-A1 \
  --checkpoint-file <pt> --tag <name> --stages provenance,bridge,analyze
```

Freeze `--seq` (never pass it explicitly) — it is part of the A0-control cache fingerprint,
along with the A0 ONNX, the `h1_2_ctrl` binary, A0's `deploy.yaml`, the scene XML and
`--bridge-cfg`. There is no time expiry; the control is for **attribution**, so let it stay
cached. Omitting the `sim` stage silently drops its gates (it does not raise INFRA), which
is defensible only because the battery directory holds the passing measurements — cite it
in the run notes.

### Pass criteria

Per arm: `fall_rate` 0 in every phase across both runs, zero safety-filter engagement,
estimator inside the pre-registered bands. Report `act_legs`, `jacc_legs_p95`, transition
overshoot, `trig_joint` rate and stride period as **reported-never-gated**.

### The caveat that must appear in the writeup

**A bridge GO is evidence about the reward set, not about deployability.** The bridge
computes leg odometry from clean sim state and understates the noise the HL sees by 3-6x
(matched runs: 0.054 / 0.117 sim vs 0.316 on the robot). These three arms are all clean-
trained HLs. Do not let a GO here read as "ready for hardware".

### Reporting + do-not-touch

Dated entry in the A1a deploy journal (research KB), numbers into a new
`raw/engineering-journal/data/` directory, one status line in `worklines.md`.
Do not touch: `src/tasks/velocity/rl/` (WL-S/WL-F), `scripts/play.py` eval protocols
(WL-T), `deploy/include/isaaclab/`. Never `git checkout` or revert anything under
`deploy/` — uncommitted work lives there.

---

## WL-S — smoothness arms

**Model: opus for LCP (a silently wrong gradient penalty fails quietly and poisons every
downstream comparison); sonnet for the coefficient arms.**

### Task

Close, or honestly bound, the A1 leg-smoothness gap to A0 (`act_legs` 0.697 vs 0.596 for
the best battery arm). Per CLAUDE.md: analysis → spec → approval → implement → test.

### Read first

- `~/biped_hrl_wiki/wiki/reference/ll-reward-tuning.md` — every coefficient, its validated
  value, and what has already been rejected. **Two corrections it does not yet carry**, both
  established 2026-08-09: (i) `ll_joint_acc_coef` is "closed as negative at both doses"
  only on the **arm4d base** — on the full-mirror base, 2.5e-7 is the batch leader on both
  `action_rate` (0.8597) and `g_legs` (0.6970), and the two jacc-trained arms are the
  lowest-`jacc` of 13; (ii) the "arms-heavy floor" framing was chasing joints that are held
  on hardware.
- `~/biped_hrl_wiki/wiki/papers/lcp-smooth-humanoid-locomotion-chen-2025.md` and
  `wiki/concepts/policy-smoothness-regularization.md`

### Arms, in priority order

1. **`ll_joint_acc_coef` sweep on the full-mirror base**: 1e-7 and 5e-7 (2.5e-7 exists and
   leads). Now well-motivated on its own target metric — score `jacc_legs_p95` first.
   Watch zero-command standing: the historical failure was touchdowns 142 → 5325.
2. **`ll_action_rate_coef` 0.07 and 0.10** on the full-mirror base (all battery arms carry
   0.05). **Pre-register the stop rule and expect this one to fail**: combo1 at 0.05 is
   already best on `action_rate` and worst of 13 on `jacc_legs_p95`, so pushing the
   commanded-rate penalty harder is expected to trade realized smoothness for commanded
   smoothness. If `jacc_legs_p95` regresses, the lever is exhausted — say so and stop.
3. **LCP**, LL only, on the full-mirror base. A local subclass of `rsl_rl.algorithms.PPO`
   overriding `update()` — add `E[||∇_s log π(a|s)||²]` as an auxiliary loss beside the
   existing `loss = surrogate + value_loss_coef*value - entropy_coef*entropy`. Do not edit
   site-packages. Sweep the penalty coefficient; it is the only candidate that attacks the
   mechanism rather than adding another competing scalar to the same objective, and it is
   validated on a Unitree H1.
4. **Cadence arms**, both distinct: `hl_cadence=False` (A0's fixed 0.6 s clock, no cadence
   channel, no `feet_gait` term) and `hl_cadence=True, source=random,
   cadence_period_range=(0.625, 0.625)` (keeps entrainment, pins the period). Plus a floor
   raise to `(0.5, 1.0)`.

   **Scope note, already established:** period pinning is **falsified for standing** — the
   2026-08-07 hardware session was pinned at 0.625 and shook anyway. These arms are in
   scope only for the walk-phase cadence-floor fall (the HL slamming cadence to the 0.35
   floor). And fixed period is falsified as an *action-rate* lever: pinning at A0's own
   0.6 s clock still leaves +39% over A0.

**Implementation gotcha:** `cadence_period_range` cannot be set from the CLI — the tyro
tuple override raises "Unrecognized options" under `--agent` (confirmed 2026-07-03). Range
changes are code edits to the default.

**Do not spend arms on:** raising `ll_cadence_coef` to get bigger steps (tested — stride
0.352 → 0.438 → 0.496 while `action_rate` went 1.13 → 1.21 → 1.28 and CoT 0.897 → 1.413);
mirroring more A0 reward terms (combo6 ran all six at A0's weights: `action_rate` 0.951 and
tripled probe error).

### Protocol

```bash
conda activate unitree_mjlab_h1_2_rl
python scripts/train.py Unitree-H1_2-Flat-A1 --env.scene.num-envs 4096 \
    --agent.max-iterations 10001 --agent.run-name <name>
```

Never `conda run`. `max-iterations` is N+1 so the last checkpoint is `model_10000.pt`. On
resume, **repeat every structure flag** from the original launch — `--agent.resume` does not
restore them from the checkpoint's `agent.yaml` the way `play.py` does. Match the base
structure to the keeper lineage by reading its `params/agent.yaml` first; launching with
bare task defaults has silently produced non-comparable runs twice.

One seed per arm is accepted for now (user's call, 2026-08-07) — so **do not order arms
inside ~3%**.

### Pass criteria

`act_legs` and `jacc_legs_p95` against the anchors above, `fall_rate` 0, zero-command
touchdowns not regressed, and tracking not worse than blind's `err_vx` 0.1219. An arm that
improves `act_legs` while regressing `jacc_legs_p95` has not helped — report it as such.

### RQ2 note requiring the owner's decision

LCP on A1 only is a confound against A0. An A0 LCP control was **deferred** (user, 2026-08-07:
"only if it's a huge benefit"). If the A1 LCP gain is large, raise it before drawing any
hierarchy conclusion from the comparison.

### Reporting + do-not-touch

Dated entry in `a1a-experiment-journal.md` (research KB) + a data directory; status line in
`worklines.md`; the two `ll-reward-tuning.md` corrections above. `pytest` green before and
after; new tests proven able to fail via `scripts/check_test_sensitivity.py`.
Do not touch: `scripts/deploy_readiness.py`, `scripts/bridge_session.py`,
`logs/deploy_safety/` (WL-V); `scripts/play.py` eval protocols (WL-T); `deploy/` at all.

---

## WL-T — start/stop transition instrument, then the rs8 ablation

**Model: sonnet.** Contained addition to an existing tool with an objective output.

### Task

Build the instrument that can see start/stop transition falls, then use it to settle the
command-resampling question. Per CLAUDE.md: analysis → spec → approval → implement → test.

### Why the instrument comes first

The user's observation is that start/stop transitions "cause falling regularly". Nothing
currently measures that: the aggregate bench resamples commands every 3-8 s **regardless of
the training setting**, and `--eval-cmd-vx` holds pin the command and never transition. So
rs8-vs-rs20 has been measured twice and read as a wash both times — A1 D1 (rs8) `act` 1.14
vs D2 (rs20) 1.13, and the A0 `legmass_rs20`/`legmass_rs8` pair agreeing "to within 0.001 on
every tracking metric" — on a bench that structurally cannot see the failure being claimed.
Training rs8 first would just produce a third such wash.

### Tasks in order

**1. `--eval-cmd-transitions` in `scripts/play.py`.** Scripted 0 → v → 0 at a fixed cadence
with pinned commands (no resampling), scoring per transition: falls, velocity overshoot,
settling time, and `act_legs`/`jacc_legs_p95` inside the transition window vs the steady
window. Reuse the existing `--eval-cmd-vx` command-pinning machinery rather than adding a
parallel path.

*Verifiable output:* run it on A0 and on the keeper. A0 is the control — if A0 shows the
same per-transition fall rate as A1, the phenomenon is not A1-specific and the ablation
below is pointless. Report that either way.

**2. rs8 vs rs20 ablation** against the new metric, one seed each, everything else matched.
Training currently uses `(3.0, 20.0)`; `(3.0, 8.0)` is the pre-2026-07-14 value **and** what
the bench already uses (`config/h1_2/env_cfgs.py`, play branch), so the change also removes
a train/eval distribution mismatch. Say explicitly whether that mismatch, not the resample
rate itself, is what moves the numbers.

### Pass criteria

The instrument resolves a difference where the aggregate bench cannot: report per-transition
fall counts with the 64-env noise floor (±2.6%) stated, and A0 as the control in every table.
A null result is a perfectly good outcome — two prior measurements already say wash, and
confirming that on an instrument that *could* have seen a difference is worth more than a
third ambiguous number.

### Reporting + do-not-touch

Dated entry in `a1a-experiment-journal.md` (research KB) + a data directory; status line in
`worklines.md`; document the new flag in `.claude/docs/hrl-infra.md` alongside the other
probes. `pytest` green; new tests proven able to fail.
Do not touch: `src/tasks/velocity/rl/` (WL-S/WL-F), the deploy pipeline scripts (WL-V),
`deploy/` at all.
