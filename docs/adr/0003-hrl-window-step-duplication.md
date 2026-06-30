# HIRO window-step is intentionally duplicated, not unified behind one module

**Status:** accepted

## Context

The HIRO goal-wiring kernel (fire the high level every `c` steps for a target, then each
step set `delta = target - estimate` into `uenv.hrl_goal` and `obs["goal"]`, step the low
level) appears at three call sites:

- `HierarchicalRunner.learn()` (`src/tasks/velocity/rl/hrl/hrl_runner.py`, the training loop),
- `HierarchicalRunner.get_inference_policy()` (same file, the play/deploy path),
- the `--diagnose-goals` probe (`scripts/play.py`).

An architecture review proposed deepening this into a single stateful module
(`GoalWindow`) so the cadence + fire + wiring lived behind one seam and the probe could
not drift from training.

On inspection the shared kernel is only ~4 lines plus a `% c` modulo. Everything that
differs across the three sites is real and load-bearing: estimator noise (`state_n`,
`noise_off`), `target_obs` (delta vs absolute mode), `begin_window`, `record_step`, the
goal-distance reward, the privileged-clean-state vs estimate split, and learning vs frozen
low level. The probe is also a *deliberate* clean-eval mirror (noise-free, captures
`s_fire`/`v_star`/`achieved`, no reward, no learning), not an accidental copy.

Unifying the three required an injected `on_fire` callback, a side-effect contract for the
privileged reward target, and an estimate-legal vs privileged split written into the
object. That is more machinery than the duplication it removes.

## Decision

Keep the window-step logic duplicated inline at the three call sites. Do **not** introduce
a `GoalWindow` (or equivalent stateful window controller).

The one invariant worth protecting if a future change touches this code is that
`uenv.hrl_goal` and `obs["goal"]` must both receive the *same* `delta`, and that `delta`
is computed against the **estimate** (`state_n` in training, the sportmode reading in
deploy), never the privileged clean state. That can be captured with a trivial wiring
helper if it ever becomes a real source of bugs; it does not justify a stateful module.

## Consequences

- The probe and the training loop stay independently editable. The cost is that a change
  to the goal-wiring convention (`delta = target - estimate`, fire every `c`) must be
  applied at all three sites by hand; this is low-frequency and easy to catch.
- Future architecture reviews should not re-suggest unifying these three sites into one
  module. The shallow shared kernel does not pay for the callback indirection and the
  privileged/estimate split the unification would require.
- If the dual-write (`uenv.hrl_goal` + `obs["goal"]`) ever causes a real bug, the
  proportionate fix is a small `set_goal(uenv, obs, delta)` helper, not a window object.
