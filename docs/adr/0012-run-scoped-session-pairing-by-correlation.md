# Score hardware per FSM Run, and pair logs by correlation rather than by recording interval

**Status:** accepted (2026-09-14, owner's decision). Changes how every hardware number is
produced from here on. Introduces `scripts/bundle_hardware_run.py` and adds `--out-dir` to
`scripts/bench_flight_recorder.py`. Supersedes the matching rule recorded in `CONTEXT.md`'s
_Session pair_ entry.

Two coupled decisions. **(1)** The unit a hardware metric is scored over is a **Run**, one
continuous occupancy of an RL state, identified by the _Flight recorder_'s `entry` column,
not a file. **(2)** A Run is matched to its _Joint telemetry log_ by **cross-correlating a
shared measured joint against every candidate** with offset and rate fitted together;
overlapping recording intervals are demoted to a sanity flag. Both were forced by the
2026-09-14 session, where the previous rules silently produce wrong numbers rather than no
numbers.

## Context

**Why Run-scoped.** The _Flight recorder_ writes one CSV per controller *process*. Returning
to Passive and entering the policy again appends a new `entry` to the same file. On
2026-09-14 that gave 12 Runs across 8 files, and `15-00-37.csv` alone holds three Runs
spanning two deliberate experimental conditions (a broom disturbance, then a 7.5 kg
backpack). Scoring that file whole averages them into one meaningless row. Nothing in the
tooling previously had a concept below the file.

A corollary worth recording, because it silently corrected an operator label: since
`State_RLHRL::ensure_models_loaded()` returns early once `ll_runner_` exists, the ONNX
sessions load **once per process and never on re-entry**. Every Run inside one file therefore
ran the *same policy*. A session log claiming two different policies in one file is wrong by
construction, and one did.

**Why correlation-first.** The interval rule proposes the telemetry log whose
`[filename, mtime]` window overlaps the flight session's. On 2026-09-14 it excludes the
correct partner for **4 of 12 Runs**: the two loggers are started and stopped by hand and
independently, so a real pair's intervals need not overlap at all. Correlating every
candidate recovers 11 of 12.

The correlation must fit **offset and rate together**. A constant lag is not merely
imprecise here, it is disqualifying: the two clocks differ by -3337 to -6613 ppm on this
session, which over a 170 s Run is ~0.5 s of drift against a ~0.78 s stride period. A
correctly-offset pair scored at a fixed lag ends up a full stride out of phase and correlates
at nothing, so the right answer gets rejected. This is why the first implementation attempt
paired only 3 of 12 and the fix was to route through `fit_alignment`.

## Considered options

- **Keep the interval proposal.** Honest, but leaves 4 Runs without torque, and therefore
  without mechanical power or _Cost of transport_, for no reason other than operator timing.
- **Widen the interval window.** The pad would be an unprincipled constant, and the observed
  spread (-6 s to +1028 s) has no natural cut.
- **Correlate everything, intervals as a flag.** Chosen.

## Consequences

Correlation alone can be fooled: a quasi-periodic walking knee plus a free rate parameter can
match the *wrong* walking bout. Two guards, both fail-closed:

- A fitted rate beyond 2% is rejected as degenerate, not a clock. A 12.8 s Run (shorter than
  the 20 s coarse-alignment window) produced a "pair" at rate 1.2 with a 0.0 ms residual.
- A pair whose wall-clock offset sits more than 300 s from the session median is **refused by
  default** and reported for a human. On 2026-09-14 this caught `14-20-04`'s two Runs matching
  at correlation 0.98 and 0.92 against a log offset by +1028 s where every other pair in the
  day sat within ±113 s. Those Runs are bundled without torque rather than with numbers that
  might be someone else's.

Metrics are written into the Run's own bundle directory, because several Runs sliced from one
process share the `filename[:19]` label the bench derives and would otherwise overwrite each
other.
