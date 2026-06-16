# Plan: A1 goal-achievability probe — decompose the tracking error (HL vs LL)

> Standalone plan to start in a NEW chat. Status as of 2026-06-16: the translational
> (vx/vy) tracking wall has survived every HL-side lever (warm-start, LR split, freeze-LL,
> velocity-obs F4, 5× reward, HIRO relabel) AND the c-sweep {4,8,12} — so neither HL data
> quantity nor temporal resolution is the limiter. The wall is **structural**. This probe
> pinpoints WHERE in the hierarchy it lives. (The user also has a replay-based structural
> hypothesis to fold in at the start of that session — get it before finalizing scope.)

## Context / why

End tracking error decomposes exactly:

  command − achieved  =  (command − V*)  +  (V* − achieved)
                          └ HL goal error ┘   └ LL reach error ┘

where, per HL window, `V*` is the absolute target the HL sets, `achieved` is the realized
state at window end. So measuring `V*` and `achieved` at window boundaries tells us whether
the vx/vy gap is **the HL setting the wrong velocity target** (command→goal map broken) or
**the LL failing to reach the target the HL set** (goal-scale/reachability or LL competence).
The frozen-LL run (competent LL, HL still failed) *hints* HL goal error dominates — this
probe quantifies it per axis and settles it.

## What to measure (per HL window, deterministic rollout)

At each HL fire (every `c` steps): record `s_fire = goal_space.extract(uenv)`,
`V* = hl.act_inference(uenv, obs, s_fire)`, `command = command_manager.get_command("twist")`,
raw goal `g` (= the HL actor output, |g| diagnostic). At the NEXT fire (window end):
`achieved = goal_space.extract(uenv)`. Goal layout (DEFAULT_GOAL_COMPONENTS): velocity =
dims[0:3], orientation = dims[3:6], height = dims[6:7]. Velocity is the only `is_task`
component, so `command` aligns with `V*[:, 0:3]`.

Aggregate (mean over windows × envs), per axis:
- **HL goal error** (velocity only): `|command − V*[:, 0:3]|` per axis (vx, vy, yaw). "Is the
  HL even asking for the commanded velocity?"
- **LL reach error** (all components): `|V* − achieved|` per component. "Does the LL deliver
  the target the HL set?" (For velocity report per axis; orient/height as norms.)
- **Sanity:** these two should sum (per axis, in signed form) to the end tracking error the
  benchmark reports (~0.39–0.47 vx). Verify the decomposition closes.
- **Goal aggressiveness:** mean `|g|` per axis and the realized-vs-requested delta ratio
  `(achieved − s_fire) / (V* − s_fire)` per component — if ≪ 1 the LL under-reaches; if `|g|`
  saturates near 1 the HL is demanding full-scale deltas every window (possibly unreachable
  in `c` steps → a goal-scale problem).

## Implementation

Add a `--diagnose-goals` mode to `scripts/play.py` (sibling to the validated `--eval-steps`
benchmark; reuse the same checkpoint/structure-restore loading). The current `get_inference_policy`
closure hides `V*`, so write a small dedicated rollout loop in this branch that mirrors the
runner's goal wiring (`HierarchicalRunner.get_inference_policy` / `learn`): every `c` steps
fire `runner.hl.act_inference`, write `obs["goal"] = V* − state` and `uenv.hrl_goal`, step the
LL deterministically, and capture the per-window quantities above. Deterministic (no
exploration noise), 64 envs / ~600 steps, multi-seed like the benchmark. Print a per-axis
decomposition table + a `[GOALDIAG] {json}` line. ~50 lines; no other files; reuse
`runner.goal_space` (extract/scale), `runner.hl`, `runner.c`.

## Verdict logic

- **HL goal error dominates** (HL targets a velocity far from `command`) → the command→goal
  map is the wall; next work is HL-side (goal representation / HL learning), NOT the LL.
- **LL reach error dominates** (HL targets the right velocity, LL under-reaches) → goal-scale/
  reachability or LL competence; check `|g|` saturation and the realized/requested ratio →
  candidate fixes: cap/retune `GoalSpace.scale`, lengthen the reach window, or LL retraining.
- **Both moderate** → compounding; tackle the larger first.

## Verification / success

- The signed decomposition closes to the benchmark's end tracking error (within noise) — this
  validates the probe itself.
- Run on the three c-sweep checkpoints (`a1_td3_relabel_c{4,8,12}_3k/model_3000.pt`) and the
  c=8 relabel run; the diagnosis should be consistent across them (structural, not c-dependent).
- Optional: run on the oracle-HL checkpoint as a control — oracle HL goal error ≈ 0 by
  construction, so any residual there is pure LL reach error (a clean LL-competence baseline).

## Caveat (carry in)

Single-seed-per-config still applies — for any firm HL-vs-LL attribution, confirm the dominant
term holds across 2–3 seeds. See `memory/a1_m4_td3.md` (c-sweep verdict) and
`memory/feedback-a0-comparison-cleanliness.md`.
