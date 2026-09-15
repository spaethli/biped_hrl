"""A0 (``State_RLBase``) deploy I/O contract, 2026-09-15.

``algorithms.h`` builds each ORT input tensor from the observation vector's data pointer but
the ONNX's declared size, so a model wider than the vector the deploy yaml builds reads
ADJACENT HEAP as observations and the robot goes limp via the safety hold, with nothing in
the log naming the cause. The HRL state has had a fail-closed check since WP5d; A0 had none,
while its slot is the one filled by hand (2026-09-14: ``A0_DR_s123`` copied in under a
PROVENANCE.json naming a different run).

Three layers, each able to fail on its own:
  * the staged export against every A0 deploy yaml, on the CPU -- catches a wrong file before
    it reaches the robot at all;
  * the pure C++ decision (``include/obs_contract.h``), compiled and run here;
  * that the FSM actually calls it, at LOAD, and refuses rather than warns.
"""

import re
import shutil
import subprocess
from pathlib import Path

import onnx
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy/robots/h1_2"
A0 = DEPLOY / "config/policy/velocity/v0"
A0_YAMLS = sorted((A0 / "params").glob("deploy*.yaml"))
A0_CPP = DEPLOY / "src/State_RLBase.cpp"


def _io(path: Path) -> tuple[list[tuple[str, int]], int]:
  m = onnx.load(str(path))
  flat = lambda t: __import__("math").prod(d.dim_value for d in t.type.tensor_type.shape.dim)
  return [(i.name, flat(i)) for i in m.graph.input], flat(m.graph.output[0])


def _width(terms: dict) -> int:
  """A deploy group's width, summed from each term's per-element scale list -- the same way
  the C++ ObservationManager builds it (and _dims in test_adapt_encoder_contract)."""
  return sum(len(t["scale"]) for t in terms.values() if isinstance(t, dict) and "scale" in t)


def test_the_a0_yamls_were_found():
  """A glob that matches nothing makes every parametrized case below vanish, and a suite
  with zero cases reports green."""
  assert A0_YAMLS, f"no deploy*.yaml under {A0 / 'params'}"


@pytest.mark.parametrize("yaml_path", A0_YAMLS, ids=lambda p: p.name)
def test_the_staged_a0_policy_matches_the_deploy_yaml(yaml_path):
  """The CPU twin of the C++ load check, run against the file ACTUALLY in the slot."""
  inputs, n_out = _io(A0 / "exported/policy.onnx")
  cfg = yaml.safe_load(yaml_path.read_text())
  obs = cfg["observations"]
  # ObservationManager's "only one input" trick: a flat term list becomes ONE group named
  # "obs". An ONNX input by any other name is not a size error -- OrtRunner::act THROWS it
  # from the policy thread, which has no handler (std::terminate, fail-dark).
  single = all(isinstance(v, dict) and "params" in v for v in obs.values())
  groups = {"obs": _width(obs)} if single else {g: _width(t) for g, t in obs.items()}
  for name, size in inputs:
    assert name in groups, f"policy.onnx input {name!r} has no group; yaml builds {list(groups)}"
    assert size == groups[name], (
      f"{yaml_path.name}: policy.onnx input {name!r} is {size} wide but the yaml builds "
      f"{groups[name]} -- on the robot ORT would read {size - groups[name]} floats of heap")
  acts = sum(len(a["scale"]) for a in cfg["actions"].values())
  assert n_out == acts, f"{yaml_path.name}: policy.onnx emits {n_out} actions, yaml maps {acts}"


@pytest.mark.skipif(shutil.which("g++") is None, reason="no g++ on this machine")
def test_the_cpp_contract_unit_test_passes(tmp_path):
  """Built OUTSIDE deploy/robots/h1_2/build/ (the hardware build dir), as its header says.
  Run here so the C++ half is exercised by the suite rather than by remembering to."""
  exe = tmp_path / "obs_contract_test"
  subprocess.run(["g++", "-std=c++17", "test/obs_contract_test.cpp", "-Iinclude", "-o", str(exe)],
                 cwd=DEPLOY, check=True, capture_output=True, timeout=120)
  r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
  assert r.returncode == 0, r.stdout + r.stderr


def _constructor(cpp: str) -> str:
  return cpp.split("State_RLBase::State_RLBase(", 1)[1].split("\nvoid State_RLBase::run()", 1)[0]


def test_the_a0_contract_is_checked_at_load_and_refuses():
  """Worth nothing if the FSM stops calling it, or calls it and only warns. Asserts the SHAPE:
  the call sits in the constructor and the first statement under the failure branch is the
  throw -- substring matching passes on `spdlog::warn(...); if (false) throw`, the degradation
  the mutation harness caught in the HRL twin of this test."""
  ctor = _constructor(A0_CPP.read_text())
  assert "h1_2::io_contract_violation(" in ctor, "the A0 contract check is not called at load"
  branch = ctor.split("if (!violation.empty())", 1)[1]
  first = next(ln.strip() for ln in branch.splitlines()
               if ln.strip() and not ln.strip().startswith("//"))
  assert first.startswith("throw "), f"a violated contract must refuse to load; got {first!r}"


def test_the_a0_contract_is_not_evaluated_in_the_control_loop():
  """run() is the 1 kHz loop. Throwing there is fail-dark, and the check calls compute(),
  which re-runs every observation term -- gait_phase_cmd included -- so evaluating it outside
  the constructor would perturb the policy's own inputs."""
  cpp = A0_CPP.read_text()
  run = cpp.split("\nvoid State_RLBase::run()", 1)[1]
  assert "io_contract_violation" not in run and "observation_manager->compute()" not in run
