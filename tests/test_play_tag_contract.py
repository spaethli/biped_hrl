"""The [BENCH] key contract: what the probe emits, as downstream scorers read it by name.

`bench.SEED_KEYS` is the declared per-seed key list (the bench fails closed at runtime if its
computed dict drifts from it). This pins it against a [BENCH]/[BENCH_SEED] line RECORDED from a
real run, so a key added, dropped or renamed in the bench without a matching change to the
recorded contract (and therefore to the parsers: bench_flight_recorder, plot_radar, the data/
drivers) fails here, on CPU.
"""

import json
from pathlib import Path

from src.tasks.velocity.eval.probes.bench import SEED_KEYS

FIX = json.loads((Path(__file__).parent / "fixtures/play_tags/bench_keys.json").read_text())
META = {"label", "hl_algorithm", "goal_components", "c"}


def test_seed_keys_match_the_recorded_bench_seed_line():
  assert set(FIX["BENCH_SEED"]) == META | {"seed"} | set(SEED_KEYS)
  assert len(SEED_KEYS) == len(set(SEED_KEYS))


def test_bench_line_is_seed_keys_with_std_twins_plus_run_fields():
  expect = META | set(SEED_KEYS) | {f"{k}_std" for k in SEED_KEYS} | {"eval_steps", "num_envs", "eval_seeds"}
  assert set(FIX["BENCH"]) == expect
