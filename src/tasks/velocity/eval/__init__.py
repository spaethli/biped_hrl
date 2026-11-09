"""play.py's measurement probes: each emits one `[TAG] {json}` line that downstream parsers read.

A probe is a module in `probes/` with `run(ctx) -> dict`. Most drive the shared seed/reset/step
skeleton in `rollout.py` as an observer; `--diagnose-goals` and `--probe-lean` keep their own loops
(the first drives the high level itself, ADR-0003). `addons.py` holds the observers any rollout
probe can attach: the schedule replay, the trace, the reward-term table, the reward dump and the
commanded joint-limit check.
"""
