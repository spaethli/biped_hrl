"""Parity contract for the simulated leg-odometry estimator (WL-F, 2026-08-09).

``src/tasks/velocity/rl/hrl/leg_odom.py`` is a transcription of the DEPLOYED estimator,
``deploy/robots/h1_2/include/hrl/base_state.h`` (``leg_fk`` / ``foot_site_b`` /
``leg_odom_velocity``) plus the window accumulator at ``State_RLHRL.cpp:667/444``. The
whole value of the WL-F change rests on that transcription being exact: a policy trained
against a subtly different estimator learns to compensate for an error the robot does not
have, and nothing downstream would reveal it — sim would simply look fine.

The vectors below were PRINTED BY THE C++ ITSELF (`cpp_ref.cpp` in the WL-F evidence
directory compiles against the unmodified shipped header). They are not recomputed from
the Python, so this file cannot drift into restating the code it guards: change the frame
convention, the link table, the rotation order, the stance rule or the ``w x p`` term and
these numbers stop matching. Each row is
``(q_prev[12], q[12], w_P[3], grav_P[3]) -> v[3]`` at the sim physics step 0.005.

Tolerance is float32 round-off: the C++ computes in float, the fixture stores 9 significant
digits, and ``leg_odom_velocity`` divides a ~1e-3 displacement by dt.
"""

import torch

from src.tasks.velocity.rl.hrl.leg_odom import (
  LegOdometry,
  foot_site_b,
  leg_odom_velocity,
)

PHYSICS_DT = 0.005  # velocity_env_cfg.py sim.mujoco.timestep

GOLDEN = (
  ([0.687385738, -0.293799132, -0.505729854, -0.472703546, 0.108360007, 0.152871519, 0.74322319, 0.245083332, 0.39825058, 0.245711803, 0.396343708, 0.738090813],
   [0.702991903, -0.308570832, -0.524139464, -0.459646106, 0.109643124, 0.171123922, 0.741700232, 0.262032866, 0.393207431, 0.231910735, 0.412037462, 0.719162405],
   [-1.1806668, -0.625492811, -1.47483504], [-0.100015387, -0.0503096841, -0.993713081], [0.291322559, 2.0434463, 0.133518711]),
  ([0.491661936, 0.203350931, 0.652679861, 0.0902357101, 0.543870866, -0.719219267, 0.48997575, 0.689305127, -0.217552662, 0.30551672, -0.593096673, 0.532298267],
   [0.497918397, 0.215743437, 0.667566895, 0.108821616, 0.552818298, -0.713520229, 0.498673886, 0.688009083, -0.224529281, 0.303102493, -0.583909094, 0.552058876],
   [0.71160686, 0.530621052, -0.544938803], [0.0478264652, 0.143733233, -0.988460183], [4.4259572, -0.584413886, -3.02878094]),
  ([-0.52653718, -0.757041156, 0.48059234, 0.645956039, -0.760518074, -0.0132042887, 0.0420083068, 0.154185578, -0.716867924, 0.632143199, 0.365225881, 0.509360015],
   [-0.531988919, -0.745242357, 0.48853159, 0.658831179, -0.74453944, 7.16820359e-05, 0.0584066138, 0.17318213, -0.710623145, 0.644622803, 0.349335641, 0.498861164],
   [0.930568278, -0.361964375, 0.000668227673], [-0.197078303, 0.0971832722, -0.975559175], [1.53302789, -2.57644796, 0.39079082]),
  ([-0.0240876209, -0.135644436, 0.221858889, 0.414863974, 0.0920366272, 0.309528649, 0.66121465, -0.760619521, 0.539805353, -0.186871633, -0.32196638, -0.789924681],
   [-0.0353296176, -0.14529568, 0.220583126, 0.413238913, 0.100417018, 0.296650767, 0.662472665, -0.773909807, 0.550557911, -0.169744805, -0.317586631, -0.803917348],
   [0.713811815, -0.0311198831, -0.187327713], [-0.0618844479, -0.0610951856, -0.996211648], [-2.32722282, -0.449248135, -0.0254894197]),
  ([0.557762265, 0.657755554, -0.185842037, -0.295206547, 0.109430604, -0.499491125, -0.598653495, 0.300153255, 0.479370803, 0.117658518, 0.757167995, 0.214486986],
   [0.557492256, 0.638317168, -0.195864558, -0.301321149, 0.121298067, -0.481955141, -0.614632607, 0.309569359, 0.498428285, 0.12589395, 0.775238574, 0.231600881],
   [-0.0137557089, -0.269907206, 1.16526508], [-0.073748447, -0.0804688334, -0.994025171], [-4.17267179, -0.277418464, 1.42474794]),
  ([0.430573374, -0.264562845, -0.667701364, 0.637693107, 0.459735692, 0.347944826, -0.0896156803, 0.401427269, 0.252637297, -0.222884327, -0.194683224, 0.239411071],
   [0.439142585, -0.264405668, -0.67867583, 0.62749207, 0.471447706, 0.347751737, -0.0730119348, 0.419242144, 0.2539666, -0.232784629, -0.185848743, 0.234108627],
   [-1.26522207, -0.00405469537, -1.33308959], [-0.19815895, -0.13276647, -0.97113651], [2.42013717, 1.24337614, -0.449780494]),
)


def _stack(rows, idx):
  return torch.tensor([r[idx] for r in rows], dtype=torch.float32)


def _feet(q):
  return torch.stack([foot_site_b(q, 0), foot_site_b(q, 1)], dim=1)


def test_matches_cpp_leg_odom_velocity_on_golden_vectors():
  """The batched python estimator reproduces the shipped C++ header, row for row."""
  q_prev, q = _stack(GOLDEN, 0), _stack(GOLDEN, 1)
  w_P, grav_P, want = _stack(GOLDEN, 2), _stack(GOLDEN, 3), _stack(GOLDEN, 4)

  got = leg_odom_velocity(_feet(q), _feet(q_prev), PHYSICS_DT, w_P, grav_P)

  assert torch.allclose(got, want, atol=2e-4, rtol=1e-4), (
    f"python leg odometry diverged from the deployed C++:\nmax |d| = "
    f"{(got - want).abs().max().item():.3e}\ngot  {got}\nwant {want}"
  )


def test_stance_selection_is_the_foot_further_along_gravity():
  """The C++ picks ``stance = (p[0].dot(grav) >= p[1].dot(grav)) ? 0 : 1`` and differences
  PER FOOT. Swapping that choice silently injects the other leg's motion, which the golden
  vectors above catch only when the two feet happen to differ enough -- so pin the rule."""
  grav = torch.tensor([[0.0, 0.0, -1.0]])
  # foot 0 lower (more negative z) -> further along gravity -> larger dot -> stance 0.
  p = torch.tensor([[[0.0, 0.1, -0.95], [0.0, -0.1, -0.60]]])
  p_prev = torch.tensor([[[0.0, 0.1, -0.95], [0.1, -0.1, -0.60]]])
  w = torch.zeros(1, 3)

  v = leg_odom_velocity(p, p_prev, PHYSICS_DT, w, grav)

  assert torch.allclose(v, torch.zeros(1, 3), atol=1e-6), (
    "the still foot (0) is the lower one and must be chosen as stance; "
    f"got {v} -- foot 1's 0.1 m jump leaked in"
  )


def test_omega_cross_p_term_is_present():
  """``v = -d(p)/dt - w x p``. With no foot motion the estimate is pure ``-w x p``; on
  hardware this term carries most of the signal (zeroing it drops the replay correlation
  against the robot from ~0.9 to ~0.3), so a dropped cross product must fail loudly."""
  p = torch.tensor([[[0.0, 0.1, -0.9], [0.0, -0.1, -0.5]]])
  w = torch.tensor([[0.0, 0.0, 2.0]])
  grav = torch.tensor([[0.0, 0.0, -1.0]])

  v = leg_odom_velocity(p, p, PHYSICS_DT, w, grav)

  # stance = foot 0 at (0, 0.1, -0.9); -w x p = -(0,0,2) x (0,0.1,-0.9) = (0.2, 0, 0).
  assert torch.allclose(v, torch.tensor([[0.2, 0.0, 0.0]]), atol=1e-6), v


def test_window_average_telescopes_and_latches_at_the_fire():
  """``fire()`` returns ``sum/n`` over the window and holds it until the next fire, which
  is what ``hl_vel_lo_`` does (State_RLHRL.cpp:444). With a constant-velocity foot and no
  rotation the c-average must equal that velocity exactly."""
  odom = LegOdometry(1, "cpu")
  q = torch.zeros(1, 12)
  w, grav = torch.zeros(1, 3), torch.tensor([[0.0, 0.0, -1.0]])
  for _ in range(8):
    odom.step(q, PHYSICS_DT, w, grav)
    q = q + torch.tensor([[0.0, 0.001] + [0.0] * 10])  # move a hip pitch a fixed amount

  first = odom.fire()
  held = odom.value.clone()
  odom.step(q, PHYSICS_DT, w, grav)

  assert torch.equal(odom.value, held), "the latched value must not move between fires"
  assert first.abs().sum() > 0.0, "sanity: the window must have accumulated something"


def test_reset_drops_the_cross_episode_difference():
  """A reset rewrites the joints to defaults; differencing across it would read as an
  enormous velocity spike."""
  odom = LegOdometry(2, "cpu")
  w, grav = torch.zeros(2, 3), torch.tensor([[0.0, 0.0, -1.0]] * 2)
  odom.step(torch.zeros(2, 12), PHYSICS_DT, w, grav)
  odom.reset(torch.tensor([0]))

  jumped = torch.zeros(2, 12)
  jumped[:, [3, 9]] = 1.2  # BOTH knees: bending one only raises that foot, and the
  # stance rule would then correctly pick the other, still foot and report 0.
  v = odom.step(jumped, PHYSICS_DT, w, grav)

  assert torch.allclose(v[0], torch.zeros(3)), "reset env must not difference across reset"
  assert v[1].abs().sum() > 1.0, "sanity: the un-reset env DOES see the jump"


def test_nominal_stance_width_is_the_hip_offset_pair():
  """Pins the 0.326 m nominal stance width quoted in doc/hrl/A1a_plan.md and ADR-0009's
  correction, so a link-table or hip-offset edit cannot silently move the reference the
  stance-width metric (`play.py --eval-steps ...` -> `stance_w_*`) is read against.

  0.326 is not a free constant: it is twice the 0.163 m hip lateral offset, and it holds at
  ANY symmetric leg pose with hip_roll = 0, because only hip_roll and hip_yaw move a foot
  laterally. Both poses below are checked for that reason -- straight-legged and the
  crouched default the policies actually train at (hip_pitch -0.2, knee 0.5, ankle -0.3).
  """
  import pytest
  import torch
  from src.tasks.velocity.rl.hrl.leg_odom import foot_sites_b

  def width(q):
    p = foot_sites_b(q)
    return float((p[0, 0, 1] - p[0, 1, 1]).abs())

  straight = torch.zeros(1, 12)
  assert width(straight) == pytest.approx(0.326, abs=1e-3)

  crouched = torch.zeros(1, 12)
  for leg in (0, 1):  # sdk order per leg: yaw, pitch, roll, knee, ankle_pitch, ankle_roll
    crouched[0, leg * 6 + 1] = -0.2
    crouched[0, leg * 6 + 3] = 0.5
    crouched[0, leg * 6 + 4] = -0.3
  assert width(crouched) == pytest.approx(0.326, abs=1e-3), \
    "hip_pitch/knee/ankle_pitch are sagittal; they must not change stance width"


def test_stance_width_grows_with_symmetric_hip_roll():
  """The sensitivity the stance-width finding rests on (~1.63 m/rad near 0).

  Guards the SIGN and rough scale, not a fitted number: if abduction stopped widening the
  stance, every reading of `stance_w_*` would invert in meaning while still looking
  plausible.
  """
  import pytest
  import torch
  from src.tasks.velocity.rl.hrl.leg_odom import foot_sites_b

  def width(hip_roll):
    q = torch.zeros(1, 12)
    for leg, sgn in ((0, 1.0), (1, -1.0)):
      q[0, leg * 6 + 1] = -0.2
      q[0, leg * 6 + 2] = sgn * hip_roll
      q[0, leg * 6 + 3] = 0.5
      q[0, leg * 6 + 4] = -0.3
    p = foot_sites_b(q)
    return float((p[0, 0, 1] - p[0, 1, 1]).abs())

  w0, w5, w10 = width(0.0), width(0.05), width(0.10)
  assert w0 < w5 < w10, "abduction must widen the stance"
  assert (w5 - w0) / 0.05 == pytest.approx(1.63, rel=0.15)
