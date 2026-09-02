"""ADR-0010: the coupled mounted-payload event. Four seams, each independently wrong-able.

1. **Composition + ordering.** ``Operation.add`` writes ``default + random``, so a later
   ``add`` on ``body_ipos`` ERASES an earlier write. The payload event must therefore run
   last and compose on the CURRENT value. When this breaks nothing crashes: the CoM shift
   silently vanishes and ``env_latent_e`` keeps reporting a plausible number.
2. **Physical consistency.** The rank-1 pseudo-inertia update must agree with the
   independent closed form (per-part parallel axis to the combined CoM), and must leave a
   valid inertia tensor across the whole sampled range.
3. **Eval-pin parity.** ``--eval-payload-kg m`` must produce what training samples at that
   ``m`` with ``d = d_mean``; otherwise eval physics sits outside the training distribution.
4. **``e`` readback fidelity.** ``env_latent_e`` is unchanged by ADR-0010 and must keep
   reporting realized totals -- payload column = added mass, com columns = TOTAL delta.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.tasks.velocity.mdp.observations import env_latent_e  # noqa: E402
from src.tasks.velocity.mdp.payload_inertia import (  # noqa: E402
  MOUNT_D_MEAN,
  MOUNT_D_RANGES,
  add_point_mass,
  eval_pin_params,
  payload_mount,
  principal_axes,
)

BID, GID = 1, 0
M0, C0 = 17.789, torch.tensor([0.00049, 0.0028, 0.20484])
I0 = torch.tensor([0.4873, 0.4096, 0.1278])          # torso principal moments, h1_2.xml


def _fake_env(n=4, ipos_residual=None):
  """Minimal stand-in exposing exactly what payload_mount / env_latent_e touch."""
  model = SimpleNamespace(
    body_mass=torch.full((n, 2), M0),
    body_ipos=C0.repeat(n, 2, 1).clone(),
    body_inertia=I0.repeat(n, 2, 1).clone(),
    body_iquat=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 2, 1).clone(),
    geom_friction=torch.full((n, 1, 3), 0.8),
  )
  if ipos_residual is not None:          # stands in for base_com having already run
    model.body_ipos[:, BID, :] += ipos_residual
  defaults = {
    "body_mass": torch.full((2,), M0),
    "body_ipos": C0.repeat(2, 1).clone(),
  }
  asset = SimpleNamespace(indexing=SimpleNamespace(
    body_ids=torch.tensor([0, BID]), geom_ids=torch.tensor([GID])))
  return SimpleNamespace(
    num_envs=n, device="cpu", scene={"robot": asset},
    sim=SimpleNamespace(model=model, get_default_field=lambda f: defaults[f]),
  )


def _cfg():
  return SimpleNamespace(name="robot", body_ids=slice(1, 2), geom_ids=slice(None))


# --------------------------------------------------------------- 2. physical consistency
def _closed_form(mass, com, i_c, m_p, p):
  """Independent route: combine CoMs, then parallel-axis EACH part to the combined CoM."""
  m_new = mass + m_p
  c_new = (mass * com + m_p * p) / m_new
  eye = torch.eye(3)
  r1, r2 = com - c_new, p - c_new
  return (m_new, c_new,
          i_c
          + mass * ((r1 @ r1) * eye - torch.outer(r1, r1))
          + m_p * ((r2 @ r2) * eye - torch.outer(r2, r2)))


def test_rank1_update_matches_the_independent_closed_form():
  p = C0 + torch.tensor(MOUNT_D_MEAN)
  for m_p in (0.0, 3.0, 12.0):
    m1, c1, i1 = add_point_mass(
      torch.tensor([M0]), C0[None, :], torch.diag(I0)[None], torch.tensor([m_p]), p[None, :])
    em, ec, ei = _closed_form(M0, C0, torch.diag(I0), m_p, p)
    assert m1.item() == pytest.approx(em, rel=1e-6)
    assert torch.allclose(c1[0], ec, atol=1e-6)
    assert torch.allclose(i1[0], ei, atol=1e-6), f"m_p={m_p}"


def test_zero_payload_is_an_exact_no_op():
  m1, c1, i1 = add_point_mass(
    torch.tensor([M0]), C0[None, :], torch.diag(I0)[None],
    torch.zeros(1), torch.zeros(1, 3))
  assert m1.item() == pytest.approx(M0)
  assert torch.allclose(c1[0], C0, atol=1e-6)
  assert torch.allclose(i1[0], torch.diag(I0), atol=1e-6)


def test_inertia_stays_valid_across_the_whole_sampled_range():
  """Positive-definite AND satisfying the triangle inequality, for extremes of (m, d)."""
  for m_p in (0.0, 12.0):
    for dx in MOUNT_D_RANGES[0]:
      for dz in MOUNT_D_RANGES[2]:
        p = (C0 + torch.tensor([dx, 0.03, dz]))[None, :]
        _, _, i1 = add_point_mass(
          torch.tensor([M0]), C0[None, :], torch.diag(I0)[None], torch.tensor([m_p]), p)
        mom, axes = principal_axes(i1)
        assert (mom > 0).all(), f"non-PD at m_p={m_p} d=({dx},{dz})"
        a, b, c = sorted(mom[0].tolist())
        assert a + b >= c - 1e-6, "violates the inertia triangle inequality"
        assert torch.det(axes)[0] > 0, "principal axes are a reflection, not a rotation"


# --------------------------------------------------------------- 1. composition + ordering
def test_payload_shift_composes_on_top_of_the_base_com_residual():
  """The residual must SURVIVE: this is the silent-overwrite class."""
  residual = torch.tensor([0.01, -0.02, 0.03])
  env = _fake_env(ipos_residual=residual)
  payload_mount(env, None, _cfg(), mass_range=(12.0, 12.0),
                d_ranges=tuple((v, v) for v in MOUNT_D_MEAN))

  m_p = 12.0
  p = C0 + torch.tensor(MOUNT_D_MEAN)          # anchored to the DEFAULT com, not the residual
  expect = ((M0 * (C0 + residual)) + m_p * p) / (M0 + m_p)
  got = env.sim.model.body_ipos[:, BID, :]
  assert torch.allclose(got, expect.expand_as(got), atol=1e-6), (
    f"residual lost: got {got[0].tolist()}, expected {expect.tolist()}")
  # and it must differ from the residual-free result, or the test proves nothing
  env0 = _fake_env()
  payload_mount(env0, None, _cfg(), mass_range=(12.0, 12.0),
                d_ranges=tuple((v, v) for v in MOUNT_D_MEAN))
  assert not torch.allclose(got, env0.sim.model.body_ipos[:, BID, :], atol=1e-6)


def test_payload_position_is_anchored_to_the_default_com_not_the_current_one():
  """The pack is bolted to the shell; base_com is uncertainty, not a displacement."""
  d = tuple((v, v) for v in MOUNT_D_MEAN)
  residual = torch.tensor([0.05, 0.0, 0.0])
  m_p, expect_p = 12.0, C0 + torch.tensor(MOUNT_D_MEAN)

  def _recover_p(env, torso_com):
    """Invert the CoM combination to recover where the pack was actually placed."""
    c_new = env.sim.model.body_ipos[0, BID, :]
    return ((M0 + m_p) * c_new - M0 * torso_com) / m_p

  a, b = _fake_env(), _fake_env(ipos_residual=residual)
  payload_mount(a, None, _cfg(), mass_range=(m_p, m_p), d_ranges=d)
  payload_mount(b, None, _cfg(), mass_range=(m_p, m_p), d_ranges=d)

  # The pack sits at the same physical place in both, even though the torso CoM differs.
  # (The resulting COMBINED CoM and inertia legitimately differ -- both parts' lever arms
  # to the combined CoM change -- so only the recovered p is invariant.)
  assert torch.allclose(_recover_p(a, C0), expect_p, atol=1e-6)
  assert torch.allclose(_recover_p(b, C0 + residual), expect_p, atol=1e-6)
  # anchoring to the CURRENT com instead would drag the pack by the whole residual
  assert not torch.allclose(_recover_p(b, C0 + residual), expect_p + residual, atol=1e-3)


def test_apply_payload_dr_rejects_an_order_that_would_erase_the_shift():
  from src.tasks.velocity.config.h1_2.env_cfgs import apply_payload_dr

  cfg = SimpleNamespace(events={
    "payload_mount": object(),                       # already present, i.e. BEFORE base_com
    "base_com": SimpleNamespace(params={"asset_cfg": _cfg()}),
  })
  with pytest.raises(ValueError, match="must run AFTER base_com"):
    apply_payload_dr(cfg)


# --------------------------------------------------------------------- 3. eval-pin parity
def test_eval_pin_reproduces_what_training_samples_at_that_mass():
  kg = 8.0
  pin = eval_pin_params(_cfg(), kg)
  a = _fake_env()
  payload_mount(a, None, _cfg(), mass_range=pin["mass_range"], d_ranges=pin["d_ranges"])
  b = _fake_env()   # training path, degenerate at the same point of the ray
  payload_mount(b, None, _cfg(), mass_range=(kg, kg),
                d_ranges=tuple((v, v) for v in MOUNT_D_MEAN))
  for f in ("body_mass", "body_ipos", "body_inertia"):
    assert torch.allclose(getattr(a.sim.model, f), getattr(b.sim.model, f), atol=1e-6), f
  assert a.sim.model.body_mass[0, BID].item() == pytest.approx(M0 + kg)
  # the pin must move the CoM: pinning mass alone is the defect this guards
  assert abs(a.sim.model.body_ipos[0, BID, 2].item() - C0[2].item()) > 1e-3


# ------------------------------------------------------------------- 4. e readback fidelity
def test_env_latent_e_reports_realized_totals_after_the_coupled_event():
  residual = torch.tensor([0.0, 0.0, 0.02])
  env = _fake_env(ipos_residual=residual)
  payload_mount(env, None, _cfg(), mass_range=(12.0, 12.0),
                d_ranges=tuple((v, v) for v in MOUNT_D_MEAN))
  e = env_latent_e(env, _cfg(), _cfg())
  assert e.shape[-1] == 5
  assert e[0, 0].item() == pytest.approx(12.0, abs=1e-5), "payload column != added mass"
  com_total = (env.sim.model.body_ipos[0, BID, :] - C0)
  assert torch.allclose(e[0, 1:4], com_total, atol=1e-6), "com columns != TOTAL delta"
  assert e[0, 3].item() < -0.05, "payload CoM drop not visible in e"
  assert e[0, 4].item() == pytest.approx(0.8)
