"""Mounted-payload randomization: mass, CoM shift and inertia from ONE sampled mount.

ADR-0010. The deployment rig is a weighted backpack in front of the torso, so payload mass
never arrives without a CoM shift and an added rotational inertia -- they are one physical
fact. The previous DR sampled mass (``base_mass``) and CoM (``base_com``) as two independent
events, which could draw 12 kg with zero CoM shift and could *not* draw the configuration the
robot will actually be in (only 4.8 kg stays inside the trained +-0.05 m CoM support).

Adding a point mass is an exact rank-1 update of the pseudo-inertia matrix, so nothing here
is an approximation::

    J = [[Sigma, h ],     h = m*c      Sigma = 1/2 tr(I_o) I3 - I_o
         [h^T,   m ]]                 I_o   = I_c + m (c.c I3 - c c^T)

    J' = J + m_p [p;1][p;1]^T         (payload m_p at body-frame position p)

``add_point_mass`` and ``principal_axes`` are pure torch over full 3x3 inertia matrices --
no quaternion convention, no sim state, no env -- so they are unit-testable on CPU with
hand-computed expectations. The event wrapper below owns the quaternion round-trip (via
mjlab's own helpers, so the wxyz convention lives in exactly one place) and the sim
plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg

# Sampling ranges (ADR-0010). Measured rig: d = (0.1625, 0, -0.2348) m from the torso CoM.
# d_x spans the cable-cover ambiguity (0.1425-0.1865 across extreme readings of where the
# cover sits); d_z spans how low the pack hangs. Randomized rather than fixed so `com` stays
# a non-deterministic function of `m` -- a fixed d would collapse the payload block of the
# privileged latent `e` to one degree of freedom.
PAYLOAD_MASS_RANGE = (0.0, 12.0)
MOUNT_D_RANGES = ((0.13, 0.19), (-0.03, 0.03), (-0.28, -0.18))
MOUNT_D_MEAN = tuple((lo + hi) / 2.0 for lo, hi in MOUNT_D_RANGES)  # (0.16, 0.0, -0.23)


def eval_pin_params(asset_cfg, kg: float) -> dict:
  """Params for ``payload_mount`` that pin THE RIG loaded to ``kg`` -- the eval counterpart.

  Single source of truth for ``play.py --eval-payload-kg``, so the pinned eval physics cannot
  drift from what training samples at the same mass. Degenerate ranges on both the mass and
  the lever arm: mass ``(kg, kg)`` and ``d`` fixed at :data:`MOUNT_D_MEAN`, i.e. one point ON
  the deployment ray. Pinning mass alone would place eval outside the training distribution.
  """
  return {
    "asset_cfg": asset_cfg,
    "mass_range": (kg, kg),
    "d_ranges": tuple((v, v) for v in MOUNT_D_MEAN),
  }


def _sym(m: torch.Tensor) -> torch.Tensor:
  """Re-symmetrize; guards float drift before eigendecomposition."""
  return 0.5 * (m + m.transpose(-1, -2))


def add_point_mass(
  mass: torch.Tensor,
  com: torch.Tensor,
  inertia: torch.Tensor,
  m_p: torch.Tensor,
  p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Attach a point mass ``m_p`` at body-frame position ``p``. Exact, not linearized.

  Args:
    mass: ``[B]`` body mass.
    com: ``[B, 3]`` body CoM in the body frame (MuJoCo ``body_ipos``).
    inertia: ``[B, 3, 3]`` rotational inertia **about the CoM**, in body-frame axes.
    m_p: ``[B]`` payload mass. ``0`` is a well-defined no-op.
    p: ``[B, 3]`` payload position in the body frame.

  Returns:
    ``(mass', com', inertia')`` in the same conventions. ``inertia'`` is again about the
    NEW CoM -- the parallel-axis shift between the two CoMs is part of the update, which is
    the step it is easiest to get wrong (and the one the test mutation targets).
  """
  eye = torch.eye(3, dtype=inertia.dtype, device=inertia.device).expand_as(inertia)
  cc = (com * com).sum(-1)[:, None, None]
  c_outer = com[:, :, None] * com[:, None, :]
  # About the CoM -> about the body origin, then to the second-moment form.
  i_o = inertia + mass[:, None, None] * (cc * eye - c_outer)
  sigma = 0.5 * i_o.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * eye - i_o
  h = mass[:, None] * com

  # The rank-1 update itself.
  sigma = sigma + m_p[:, None, None] * (p[:, :, None] * p[:, None, :])
  h = h + m_p[:, None] * p
  mass_new = mass + m_p

  com_new = h / mass_new[:, None].clamp(min=1e-12)
  i_o_new = sigma.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * eye - sigma
  cc_new = (com_new * com_new).sum(-1)[:, None, None]
  c_outer_new = com_new[:, :, None] * com_new[:, None, :]
  inertia_new = i_o_new - mass_new[:, None, None] * (cc_new * eye - c_outer_new)
  return mass_new, com_new, _sym(inertia_new)


def principal_axes(inertia: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Diagonalize ``[B, 3, 3]`` -> ``([B, 3]`` moments, ``[B, 3, 3]`` proper rotation``)``.

  ``torch.linalg.eigh`` returns an orthonormal basis that may be a reflection; MuJoCo's
  ``body_iquat`` must be a rotation, so a negative-determinant basis has one axis flipped.
  """
  moments, axes = torch.linalg.eigh(_sym(inertia))
  flip = torch.det(axes) < 0
  axes = torch.where(flip[:, None, None], axes * torch.tensor(
    [-1.0, 1.0, 1.0], dtype=axes.dtype, device=axes.device), axes)
  return moments, axes


@requires_model_fields(
  "body_mass", "body_ipos", "body_inertia", "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def payload_mount(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg,
  mass_range: tuple[float, float] = PAYLOAD_MASS_RANGE,
  d_ranges: tuple[tuple[float, float], ...] = MOUNT_D_RANGES,
) -> None:
  """Startup event: sample a mounted payload and DERIVE mass, CoM and inertia from it.

  ⚠ **Must be registered LAST among the events that write ``body_ipos``.**
  ``Operation.add`` has ``uses_defaults=True`` (``dr/_types.py:94-99``), so the DR engine
  writes ``default + random`` rather than ``current + random``: a later ``add`` on the same
  field ERASES this event's write. It fails silently -- the CoM shift vanishes and
  ``env_latent_e`` still reports a plausible number -- so ``apply_payload_dr`` asserts the
  ordering at config time. This function deliberately writes ``body_ipos`` itself, composing
  on the CURRENT value so the ``base_com`` residual survives.

  The payload position is anchored to the **default** torso CoM (``p = ipos_default + d``),
  not the current one: the pack is bolted to the shell, whereas ``base_com`` represents
  uncertainty *about where the CoM is*, not a real displacement of the shell. This also makes
  ``p`` independent of event order; only the resulting combined CoM depends on ``base_com``.
  """
  from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix

  asset = env.scene[asset_cfg.name]
  bid = asset.indexing.body_ids[asset_cfg.body_ids][0]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device)
  n = env_ids.numel()

  def _u(lo: float, hi: float) -> torch.Tensor:
    return torch.rand(n, device=env.device) * (hi - lo) + lo

  m_p = _u(*mass_range)
  d = torch.stack([_u(*r) for r in d_ranges], dim=-1)                       # [n, 3]
  p = env.sim.get_default_field("body_ipos")[bid].to(env.device) + d        # [n, 3]

  model = env.sim.model
  mass = model.body_mass[env_ids, bid]                                      # [n]
  com = model.body_ipos[env_ids, bid, :]                                    # [n, 3]
  quat = model.body_iquat[env_ids, bid, :]                                  # [n, 4] wxyz
  rot = matrix_from_quat(quat)                                              # [n, 3, 3]
  # body_inertia holds PRINCIPAL moments in the body_iquat frame; lift to a body-frame tensor.
  i_c = rot @ torch.diag_embed(model.body_inertia[env_ids, bid, :]) @ rot.transpose(-1, -2)

  mass_new, com_new, i_new = add_point_mass(mass, com, i_c, m_p, p)
  moments, axes = principal_axes(i_new)

  model.body_mass[env_ids, bid] = mass_new
  model.body_ipos[env_ids, bid, :] = com_new
  model.body_inertia[env_ids, bid, :] = moments
  model.body_iquat[env_ids, bid, :] = quat_from_matrix(axes)
