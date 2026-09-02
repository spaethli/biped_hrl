"""WP5d: the sim/hardware contract for H-adapt's adaptation module ``phi``.

The HL trained in WP5 Phase 1 reads a privileged latent ``e`` that exists only in the
simulator. On hardware that channel is ``z_hat = phi(history)``, and phi is fed a window of
the HL's own observation vector. Two sides therefore have to agree on a 4600-float tensor,
in two languages, with no runtime error if they disagree: ``algorithms.h`` sizes the ORT
input from the ONNX declared shape and never compares it against the vector that was built,
so a mismatch reads adjacent heap and the robot goes limp via the safety hold.

These tests make that agreement executable on the CPU-only path.
"""

import re
from pathlib import Path

import onnx
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy/robots/h1_2"
HRL_PARAMS = DEPLOY / "config/policy/velocity_hrl/v0/params"
HRL_YAMLS = sorted(HRL_PARAMS.glob("deploy*.yaml"))
STATE_CPP = DEPLOY / "src/State_RLHRL.cpp"
ENCODER_H = DEPLOY / "include/hrl/adapt_encoder.h"

from src.tasks.velocity.mdp.observations import E_NAMES  # noqa: E402
from src.tasks.velocity.rl.hrl.hrl_runner import HierarchicalRunner  # noqa: E402


def _dims(cfg: dict, group: str) -> int:
  """Width of a deploy observation group, summed from each term's per-element scale list —
  the same way the C++ observation manager builds it."""
  return sum(len(t["scale"]) for t in cfg["observations"][group].values())


@pytest.fixture(scope="module")
def deploy_cfg() -> dict:
  return yaml.safe_load((HRL_PARAMS / "deploy_est.yaml").read_text())


class _Phi(torch.nn.Module):
  def __init__(self, n: int):
    super().__init__()
    self.register_buffer("z", torch.zeros(1, n))

  def forward(self, x):
    return self.z + 0.0 * x.sum()


def _export(tmp_path, **over):
  kw = dict(
    history_len=50, input_dim=92, z_names=E_NAMES,
    z_clip_lo=[0.0, -0.05, -0.05, -0.05, 0.3],
    z_clip_hi=[12.0, 0.05, 0.05, 0.05, 1.6],
    z_cold=[0.0, 0.0, 0.0, 0.0, 0.95],
    policy_dim=89, command_dim=3,
  )
  kw.update(over)
  out = str(tmp_path / "adapt_encoder.onnx")
  HierarchicalRunner.export_adapt_encoder_onnx(_Phi(len(kw["z_names"])), out, **kw)
  return out


def _meta(path: str) -> dict[str, str]:
  return {e.key: e.value for e in onnx.load(path).metadata_props}


# --- the yaml flag: one key, exactly once, off by default ---------------------


@pytest.mark.parametrize("yaml_path", HRL_YAMLS, ids=lambda p: p.name)
def test_hl_obs_e_appears_exactly_once(yaml_path):
  """yaml-cpp takes the FIRST occurrence of a duplicate key while PyYAML takes the LAST, so
  a second ``hl_obs_e`` appended below the ``hrl:`` block would be a silent no-op on the
  robot AND would read as active from Python — the failure would point the wrong way."""
  n = len(re.findall(r"^\s*hl_obs_e\s*:", yaml_path.read_text(), re.M))
  assert n == 1, f"{yaml_path.name} declares hl_obs_e {n} times"


@pytest.mark.parametrize("yaml_path", HRL_YAMLS, ids=lambda p: p.name)
def test_hl_obs_e_is_inside_the_hrl_block_and_defaults_off(yaml_path):
  """It has to be reachable as ``hrl.hl_obs_e`` (a key at file scope parses fine and is
  simply never read), and it must default OFF so every pre-WP5d checkpoint still deploys
  byte-identically."""
  cfg = yaml.safe_load(yaml_path.read_text())
  assert "hl_obs_e" in cfg["hrl"], f"{yaml_path.name}: hl_obs_e is not under hrl:"
  assert cfg["hrl"]["hl_obs_e"] is False


# --- the 92-float per-step vector --------------------------------------------


def test_phi_frame_is_the_deploy_policy_and_command_widths(deploy_cfg, tmp_path):
  """phi's per-step vector is ``obs[policy] ++ obs[command]``, the leading columns of the
  HL's own input. The exported layout string must name the widths this deploy actually
  builds, or the deploy refuses at load."""
  pol, cmd = _dims(deploy_cfg, "policy"), _dims(deploy_cfg, "command")
  assert (pol, cmd) == (89, 3), f"deploy obs widths changed: policy={pol} command={cmd}"
  meta = _meta(_export(tmp_path, policy_dim=pol, command_dim=cmd, input_dim=pol + cmd))
  assert meta["phi_input_layout"] == f"policy{pol}+command{cmd}"
  assert float(meta["phi_input_dim"]) == pol + cmd


def test_e_names_match_the_latent_the_env_actually_builds():
  """``E_NAMES`` is baked into the ONNX as ``z_names``; if it drifts from the order
  ``env_latent_e`` concatenates, the HL and the estimator label the same columns
  differently and nothing errors."""
  src = (REPO / "src/tasks/velocity/mdp/observations.py").read_text()
  body = src.split("def env_latent_e", 1)[1].split("\ndef ", 1)[0]
  cat = re.search(r"torch\.cat\(\[(.*?)\], dim=-1\)", body, re.S).group(1)
  order = [p.strip().split(".")[0] for p in cat.split(",") if p.strip()]
  assert order == ["payload", "com_delta", "friction"]
  assert E_NAMES == ("payload_kg", "com_dx", "com_dy", "com_dz", "friction")


# --- the exporter is the single writer of the contract -----------------------


def test_export_writes_every_metadata_key_the_deploy_reads(tmp_path):
  """The C++ loader reads phi's contract by name. Both sides are read from source here, so
  adding a key on one side without the other fails rather than silently defaulting."""
  cpp = STATE_CPP.read_text()
  block = cpp.split("read_encoder_meta", 1)[1].split("\n}", 1)[0]
  wanted = set(re.findall(r'(?:str|vec|num)\("([a-z_]+)"\)', block))
  assert wanted, "could not parse the encoder metadata reads out of State_RLHRL.cpp"
  written = set(_meta(_export(tmp_path)))
  assert wanted <= written, f"deploy reads keys the exporter never writes: {wanted - written}"


def test_export_refuses_a_cold_start_prior_outside_the_clamp(tmp_path):
  """The zero vector is NOT a nominal latent: payload and CoM read as deltas, but friction
  reads as an absolute coefficient whose support is 0.3-1.6, so a zero fill hands the HL a
  frictionless floor — off-distribution, in the worst direction, for the whole 1.0 s the
  history takes to fill."""
  with pytest.raises(ValueError, match="training support"):
    _export(tmp_path, z_cold=[0.0, 0.0, 0.0, 0.0, 0.0])


def test_export_refuses_an_input_dim_that_is_not_policy_plus_command(tmp_path):
  with pytest.raises(ValueError, match="phi input_dim"):
    _export(tmp_path, input_dim=94)


def test_export_refuses_a_scale_that_the_metadata_cannot_represent(tmp_path):
  """Metadata float lists serialise at 3 decimals. A normalisation scale of 4e-4 would be
  written ``0.000`` — the latent component would be silently muted on the robot while every
  dimension check still passed."""
  with pytest.raises(ValueError, match="rounds to 0.000"):
    _export(tmp_path, z_scale=[0.0004, 1.0, 1.0, 1.0, 1.0])


def test_export_refuses_bounds_that_do_not_cover_every_component(tmp_path):
  with pytest.raises(ValueError, match="z_names"):
    _export(tmp_path, z_clip_hi=[12.0, 0.05, 0.05, 0.05])


# --- fail-closed: every session's built vector is checked --------------------


def test_every_ort_session_is_length_checked_before_it_runs():
  """``algorithms.h`` sizes the input tensor from the ONNX shape and never compares it with
  the vector that was built, so a short vector reads adjacent HEAP as observations and the
  robot goes limp via the safety hold. Every ``act()`` in the HRL state must therefore be
  preceded by a ``check_input``; this catches a fourth session added without one."""
  cpp = STATE_CPP.read_text()
  acts = re.findall(r"(\w+)_runner_->act\(", cpp)
  guarded = set(re.findall(r'check_input\("(\w+)\.onnx"', cpp))
  expect = {"hl": "high_level", "ll": "low_level", "enc": "adapt_encoder"}
  for who in acts:
    assert expect.get(who) in guarded, f"{who}_runner_->act() runs unguarded"


def test_the_load_time_contract_check_is_actually_called():
  """The whole load-time decision lives in one pure function so it can be tested offline;
  that is worth nothing if the FSM stops calling it. A deleted call would leave a deploy
  that loads any encoder whose tensor happens to be the right total size — including one
  exported against the A0 column permutation."""
  cpp = STATE_CPP.read_text()
  assert "hrl::check_deploy_contract(" in cpp, "the load-time contract check is never called"
  # Assert the SHAPE of the guard, not just that the word "throw" appears nearby: the first
  # statement inside `if (!bad.empty())` must be the throw. Substring-matching passes happily
  # on `spdlog::warn(...); if (false) throw ...`, which is exactly the degradation this test
  # is supposed to catch (the mutation harness found that hole).
  body = cpp.split("if (!bad.empty()) {", 1)[1].split("\n        }", 1)[0]
  first = next(ln.strip() for ln in body.splitlines()
               if ln.strip() and not ln.strip().startswith("//"))
  assert first.startswith("throw "), (
    f"a failed contract check must refuse to load; first statement is {first!r}")


def test_the_runtime_guard_latches_instead_of_throwing():
  """The policy thread is a bare ``std::thread`` with no handler: an escaping exception is
  ``std::terminate``, which kills the process and stops lowcmd entirely — fail-DARK, not
  fail-closed. At run time the guard must latch a flag the registered check turns into a
  Passive transition."""
  cpp = STATE_CPP.read_text()
  hdr = (DEPLOY / "include/FSM/State_RLHRL.h").read_text()
  guard = hdr.split("bool check_input(", 1)[1].split("\n    }", 1)[0]
  assert "throw" not in guard, "check_input must not throw from the policy thread"
  assert "dim_fault_ = true" in guard
  assert re.search(r"return dim_fault_;", cpp), "dim_fault_ is never registered as an FSM check"


def test_window_refuses_a_partial_buffer():
  """The cold-start invariant, at its source: phi is never called on a partially filled
  history. If ``window()`` zero-padded instead, phi would run on an input shape it never
  saw in training and emit a confident, arbitrary z_hat."""
  src = ENCODER_H.read_text()
  body = src.split("bool window(", 1)[1].split("\n    }", 1)[0]
  assert "if (!full()) return false;" in body


# The ring buffer's runtime behaviour (fill, wrap, oldest-first order, reset, clamp) is
# pinned by deploy/robots/h1_2/test/adapt_encoder_test.cpp -- header-only, no robot, no
# onnxruntime. Build it OUTSIDE the hardware build dir; see that file's header.
