---
name: analyze-runs
description: Locate, benchmark, probe, and interpret newly-finished training runs in logs/rsl_rl/, then sync the results into the docs. Use whenever the user says they have new results / new logs / new runs / finished trainings (e.g. "I have the results for X", "the runs are done", "analyze/benchmark these runs", "new checkpoints in the logs folder") — run the full procedure: find the run dirs, pick the newest checkpoint, run the deterministic benchmark (+ the goal probe for A1), present comparison tables with an honest read vs the existing baselines, and offer to sync into doc/hrl + memory.
---

# Analyze new training runs (benchmark + probe + sync)

The standing procedure for "I have new results/logs". Do every step in order; don't skip
the read or the sync. Match the conventions already in `CLAUDE.md` and the canonical docs.

## 0. Environment

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate unitree_mjlab_h1_2_rl
```
Run `python scripts/play.py` directly (NOT `conda run`). All commands from the repo root.

## 1. Locate the runs

If the user named the runs, use those. Otherwise list the newest dirs and confirm which
are new (don't re-benchmark ones already in the results tables):

```bash
ls -dt logs/rsl_rl/h1_2_velocity/*/      # A0 (flat)
ls -dt logs/rsl_rl/h1_2_velocity_a1/*/   # A1 (hierarchy)
```

## 2. Map each run dir → task ID

| log dir | run-name contains `lean` | task ID |
|---|---|---|
| `h1_2_velocity/` (A0) | no  | `Unitree-H1_2-Flat` |
| `h1_2_velocity/` (A0) | yes | `Unitree-H1_2-Flat-Lean` |
| `h1_2_velocity_a1/` (A1) | no  | `Unitree-H1_2-Flat-A1` |
| `h1_2_velocity_a1/` (A1) | yes | `Unitree-H1_2-Flat-A1-Lean` |

A1 HL structure (`hl_algorithm`, `hl_target_mode`, `hl_reward_mode`, goal space) is
**auto-restored** by play.py from `params/agent.yaml` — do NOT pass HL flags. Peek at it
only to label the run in the table (`absolute` vs `delta`=directional goals, etc.):
```bash
grep -E "hl_target_mode|hl_reward_mode|hl_algorithm|relabeling" <run>/params/agent.yaml
```

## 3. Pick the newest checkpoint (highest model number — never `ls -t`, never guess)

```bash
ls <run>/*.pt | sed 's/.*model_//;s/.pt//' | sort -n | tail -1
```

## 4. Benchmark every run (the canonical comparator)

```bash
python scripts/play.py <TASK> --checkpoint-file <run>/model_<N>.pt \
  --num-envs 64 --eval-steps 600 --eval-seeds 2 2>&1 | grep -E "BENCH\]"
```
Always **≥2 seeds** (same-config runs diverge — GPU nondeterminism + RL chaos; hold
`num_envs` fixed within a comparison set). Survival metric is `fall_rate`, NOT `ep_len`.

## 5. Probe every A1 run (HL-vs-LL error decomposition)

```bash
python scripts/play.py <A1_TASK> --checkpoint-file <run>/model_<N>.pt \
  --diagnose-goals 600 --eval-seeds 2 2>&1 | grep -E "GOALDIAG\]"
```
Skip for A0 (no hierarchy). Key fields: `hl_err_*` (HL goal error) vs `ll_err_*` (LL
reach error) per axis, `gabs_*`/`sat>0.95` (goal saturation), `vx follow ratio`
(<<1 or negative = LL under-reaches / wrong direction).

## 6. Present the read

**First: pull in the comparison that already exists — don't start a fresh isolated table.**
Before building tables, look for comparable runs already on the record:
- earlier in **this chat** (runs already benchmarked/probed this session),
- in **memory** (e.g. `a1_reward_routing`, `a1_goal_probe`, and the MEMORY.md index),
- in the **canonical results tables**: `doc/hrl/A1_HIRO.md` "Current results" in this
  repo, and in the research KB (`~/biped_hrl_wiki`) the A1 findings ledger
  (`wiki/architectures/a1-findings-ledger.md`), the A1a experiment journal, and the
  hierarchy-benefit roadmap's Track F results.

A run is "comparable" if it shares the axis under study (same task family / lean variant /
goal mode / init / reward set). When you find them, **extend that table with the new
rows** and compare the new runs *against* the existing ones — same columns, same metrics,
so the deltas are read off directly. Don't re-run a checkpoint already in a results table;
cite its stored numbers. Carry over the variant-label columns (`goals`, `init`,
`reward set`) so old and new rows stay distinguishable. If a new run's "same-config"
sibling already exists, put them adjacent and call out the run-to-run spread.

Two markdown tables, numbers from the `[BENCH]`/`[GOALDIAG]` JSON (extended with the
prior comparable rows):

- **Benchmark:** `err_vx | err_vy | err_yaw | fall_rate | act_rate | orient_dev | height_dev`
  (+ a `goals`/`init` label column when comparing A1 variants).
- **Probe (A1):** `HL err vx | LL err vx | HL err yaw | LL err yaw | gabs/sat | follow ratio`.

Then interpret honestly (cite the actual field/number — don't state assumptions as fact):
- Compare against the A0 baseline (~err_vx 0.09 / vy 0.11 / yaw 0.10, act_rate ~0.6 shaped)
  and the standing A1 best in `doc/hrl/A1_HIRO.md` / the KB's Track F results.
- Decompose A1: is the wall the HL (high `hl_err`) or the LL (high `ll_err`)?
  Saturation? Init contamination (jerky `act_rate` inherited from a wild warm-start)?
- Flag run-to-run variance explicitly when a "same-config" run lands far from its sibling.
- State the bottom line in one sentence.

## 7. Sync (close the loop)

Invoke the **`sync-docs`** skill to route the results to their canonical homes. Since
2026-07-28 that is a two-destination split: **current status and the standing results
table** stay in this repo (`doc/hrl/A1_HIRO.md`, the `HRL_plan.md` dashboard), while the
**chronological record** — what was tried, what failed, why, with run names, log dirs and
W&B ids — goes to the research KB (`~/biped_hrl_wiki`), which must never be condensed.
Numbers live in exactly one table; everything else links. Personal-identifier redaction
gate applies to the repo only. Confirm with the user before large doc rewrites.

## Notes / gotchas
- `gamma_hi`, goal dim, etc. are derived — don't hardcode. See `CLAUDE.md` "Load-bearing gotchas".
- A1 LL trains only on the intrinsic L2 goal-distance reward (env reward terms are
  diagnostic-only for A1) — see the `a1_reward_routing` memory before reasoning about what
  a reward term does to an A1 run.
- Lean variants: `semi-lean` keeps the `action_rate_l2`/`joint_acc_l2` smoothness floor,
  `true-lean` zeros it too (fair A0-vs-A1 smoothness read). Label which one in the table.
