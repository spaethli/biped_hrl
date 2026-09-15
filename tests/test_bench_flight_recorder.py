"""Flight-recorder -> `[BENCH]` scoring (``scripts/bench_flight_recorder.py``).

Two classes of guard, because the tool can be wrong in two independent ways.

**Golden values.** The commanded ankle range is the number ADR-0009 rests on, and it is the
one quantity validated against both plants (sim and hardware agree to a few percent). The
fixture under ``tests/fixtures/flight_recorder`` is the policy-step decode of the three
2026-08-27 A0 sessions, extracted losslessly (the recorder writes 4 significant digits), so
this test runs everywhere rather than skipping wherever the 470 MB captures are absent.

**Seam invariants.** Every rate metric is differenced at a rate this tool has to RECOVER,
and every torque metric depends on an alignment it has to FIT. Both have flipped verdicts on
this project before, so each is pinned on synthetic data where the right answer is known by
construction: a stride decode cannot be checked against a stride decode.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests/fixtures/flight_recorder"
sys.path.insert(0, str(REPO / "scripts"))

import bench_flight_recorder as bfr  # noqa: E402

# ADR-0009 "Evidence", hardware table. p1..p99 of the post-clip commanded target on the
# change-detected policy steps of the WHOLE session -- not row-rate (0.4488/0.5100 on the
# first session) and not alpha-gated (0.4273 on its right ankle). Both alternatives were
# measured and neither reproduces the ADR.
ADR_0009_ROLL_SPAN = {
    "2026-08-27_13-42-55": (0.4132, 0.5048),
    "2026-08-27_13-51-52": (0.1923, 0.2233),
    "2026-08-27_13-56-37": (0.4360, 0.3728),
}


def load_fixture(session):
    df = pd.read_csv(FIX / f"{session}.steps.csv.gz")
    meta = json.loads((FIX / f"{session}.meta.json").read_text())
    steps = df[[f"raw_q{j}" for j in range(12)]].to_numpy(float)
    joints = {j["slot"]: j for j in meta["joints"]}
    return steps, joints


@pytest.mark.parametrize("session", sorted(ADR_0009_ROLL_SPAN))
def test_ankle_roll_span_reproduces_adr_0009(session):
    """The headline hardware number, to the digit."""
    steps, joints = load_fixture(session)
    jr = bfr.joint_ranges(steps, joints, bfr.LEG_SLOTS)
    exp_l, exp_r = ADR_0009_ROLL_SPAN[session]
    assert jr["left_ankle_roll"]["span"] == pytest.approx(exp_l, abs=5e-5)
    assert jr["right_ankle_roll"]["span"] == pytest.approx(exp_r, abs=5e-5)


def test_span_is_p1_to_p99_not_min_to_max():
    """min..max is a different quantity: one transient excursion is not authority.

    Guards the percentile choice specifically -- on this session min..max is 27% wider than
    p1..p99, so a silent switch would inflate every authority claim.
    """
    steps, joints = load_fixture("2026-08-27_13-42-55")
    jr = bfr.joint_ranges(steps, joints, bfr.LEG_SLOTS)
    x = steps[:, 5]
    assert jr["left_ankle_roll"]["span"] < (x.max() - x.min()) * 0.95


def test_headroom_is_distance_to_the_nearer_bound():
    """`headroom` ~ 0 with pinned > 0 is a clip; ~1 action-sigma is a penalty (ADR-0009).

    The two clipped arms sat ~1.8 sigma off the bound and v2 sat 0.19 sigma from it, so the
    sign and scale of this number carry the whole clip-vs-penalty verdict.
    """
    steps, joints = load_fixture("2026-08-27_13-42-55")
    jr = bfr.joint_ranges(steps, joints, bfr.LEG_SLOTS)
    r = jr["left_ankle_roll"]
    j = joints[5]
    assert r["headroom"] == pytest.approx(min(j["max"] - r["p99"], r["p1"] - j["min"]), abs=5e-5)
    # v2 rides its ankle stops: pinned is nonzero and headroom is near zero
    assert r["pinned"] > 0.0
    assert abs(r["headroom"]) < 0.05


def test_pinned_any_leg_is_a_union_not_a_mean():
    """The ADR reports a per-STEP fraction with any leg joint on a bound.

    A per-joint mean over the 12 slots is ~3x smaller and would read as a different robot.
    """
    steps, joints = load_fixture("2026-08-27_13-42-55")
    any_frac = bfr.pinned_any(steps, joints, bfr.LEG_SLOTS)
    jr = bfr.joint_ranges(steps, joints, bfr.LEG_SLOTS)
    mean_frac = float(np.mean([v["pinned"] for v in jr.values()]))
    assert any_frac == pytest.approx(0.1215, abs=5e-4)
    assert any_frac > 3 * mean_frac


def test_policy_step_decode_recovers_50hz_from_the_500hz_log():
    """The recorder decimates to ~500 Hz; the decode must recover the 50 Hz policy steps.

    Run on a real 30 s raw-rate slice with 39% non-uniform row spacing (the recorder logs
    trigger ticks undecimated on top of the decimation), because the DEFECT this guards is
    only visible on a non-uniform log.
    """
    df = pd.read_csv(FIX / "2026-08-27_13-42-55.rawslice.csv.gz")
    raw = df[[f"raw_q{j}" for j in range(27)]].to_numpy(float)
    t = df["t"].to_numpy(float)
    steps, idx = bfr.policy_steps(raw)
    hz = len(steps) / (t[-1] - t[0])
    assert 45.0 < hz < 55.0, f"decoded {hz:.2f} Hz, expected ~50"
    # the log really is ~500 Hz, i.e. the decode is doing work, not passing rows through
    assert (len(raw) / (t[-1] - t[0])) > 400.0
    assert len(steps) < len(raw) / 5


def test_policy_step_decode_tracks_changes_not_a_row_count():
    """Each decoded step must span rows that are all IDENTICAL to it.

    The rate assertion above cannot carry this on its own: at a ~500 Hz log a fixed stride
    of 10 rows also averages ~50 Hz, so it passes a rate check while being wrong. The
    mechanism is what separates them -- on this slice the true decode has 0 rows that
    disagree with their own step, and a stride-10 decode has thousands, because the log is
    not uniform. Restating the rate is exactly the test that cannot fail when the code
    breaks; this is the one that can.
    """
    df = pd.read_csv(FIX / "2026-08-27_13-42-55.rawslice.csv.gz")
    raw = df[[f"raw_q{j}" for j in range(27)]].to_numpy(float)

    def inconsistent_runs(idx):
        bounds = np.append(idx, len(raw))
        return sum(1 for i in range(len(bounds) - 1)
                   if len(raw[bounds[i]:bounds[i + 1]])
                   and not np.all(raw[bounds[i]:bounds[i + 1]] == raw[bounds[i]]))

    _, idx = bfr.policy_steps(raw)
    assert inconsistent_runs(idx) == 0, "a decoded step spans rows it does not equal"
    # and the slice really can tell the two apart -- otherwise this test proves nothing
    assert inconsistent_runs(np.arange(0, len(raw), 10)) > 100


@pytest.mark.parametrize("b_true", [-3.3, -10.3, -60.0])
def test_affine_alignment_recovers_an_injected_clock_skew(b_true):
    """A constant offset is wrong by 40 policy steps over a 308 s session.

    Synthetic: one chirp-ish signal sampled on two clocks differing by a known offset and
    rate. Recovery of BOTH terms is asserted; a constant-only fit fails the rate assertion.
    The offsets span the operator-timing range: -3.3 s is 2026-08-27, -10.3 s is
    2026-09-07 (outside the fine search's +-8 s, so only the coarse stage reaches it), and
    -60 s is a slow start nobody has measured yet but nothing prevents.
    """
    rng = np.random.default_rng(0)
    dur, m_true = 300.0, -2600e-6
    ta = np.arange(0.0, dur, 0.002)
    # A gait-like signal must carry APERIODIC structure or the lag is only identifiable
    # modulo the stride period (0.6 s), and a -3.3 s offset aliases onto -0.3 s. Real
    # sessions supply that structure the same way this fixture does: walking bursts
    # separated by quiet standing (measured 14% walking on 13-42-55).
    burst = np.zeros_like(ta)
    for start, length in [(12.0, 22.0), (70.0, 9.0), (140.0, 35.0), (230.0, 14.0)]:
        burst[(ta >= start) & (ta < start + length)] = 1.0
    knee_a = (np.sin(2 * np.pi * 1.67 * ta) * burst
              + 0.02 * rng.standard_normal(len(ta)))
    tf = ta + (b_true + m_true * ta)
    knee_f = knee_a.copy()
    fit = bfr.fit_alignment(tf, knee_f, ta, knee_a)
    assert fit is not None
    assert fit["b"] == pytest.approx(b_true, abs=0.15)
    assert fit["m"] * 1e6 == pytest.approx(m_true * 1e6, abs=400)
    assert fit["xcorr"] > 0.7


def test_regime_gate_matches_the_reward_command_threshold():
    """`||cmd_xy|| + |cmd_wz| > 0.1`, exactly mdp/rewards.py:157-159.

    NOT a 3-vector norm and NOT score_joint_hold's 1e-6 stillness gate: on the 2026-08-27
    sessions those three gates disagree by up to 2x in walking-step count, and pooling or
    mis-gating reads a session ~4x smoother than it is.
    """
    src = (REPO / "src/tasks/velocity/mdp/rewards.py").read_text()
    assert "total_command = linear_norm + angular_norm" in src, "reward gate form changed"
    assert bfr.CMD_THRESHOLD == 0.1

    cmd = np.array([[0.05, 0.0, 0.04],    # sum 0.09 -> standing
                    [0.05, 0.0, 0.06],    # sum 0.11 -> walking
                    [0.0, 0.0, 0.2],      # pure yaw -> walking
                    [0.08, 0.06, 0.0],    # sum = L2 0.10, not > 0.1 -> standing
                    [0.06, 0.0, 0.06]])   # sum 0.12 -> walking, but 3-norm 0.085 would not be
    assert list(bfr.walking_mask(cmd)) == [False, True, True, False, True]
    # the last row is the discriminating case: a 3-vector norm calls it standing, and the
    # 1e-6 stillness gate calls rows 0 and 3 walking. Only the reward form gets all five.
    assert np.linalg.norm(cmd[4]) < bfr.CMD_THRESHOLD
    assert list(np.linalg.norm(cmd, axis=1) > bfr.CMD_THRESHOLD) != list(bfr.walking_mask(cmd))
    assert list(np.abs(cmd).sum(axis=1) > 1e-6) != list(bfr.walking_mask(cmd))


def test_dominant_period_recovers_a_known_stride():
    """`stride_period_s` is sim's realized quantity, recovered from kinematics.

    Contact state is dead in both logs, so the same-foot interval comes from a leg joint's
    dominant period. Pinned on a synthetic 0.6 s oscillation with noise and a DC offset.
    """
    rng = np.random.default_rng(1)
    dt, T = 0.002, 0.5919
    x = np.arange(0, 20.0, dt)
    sig = 0.7 + 0.4 * np.sin(2 * np.pi * x / T) + 0.03 * rng.standard_normal(len(x))
    assert bfr.dominant_period(sig, dt) == pytest.approx(T, abs=0.01)
    # a flat signal has no period rather than a spurious one
    assert np.isnan(bfr.dominant_period(np.ones(5000), dt))


def test_projected_gravity_matches_the_sim_definition():
    """orient_dev is ||proj_gravity_b[:2]||, the quantity play.py reads off the robot."""
    # upright: gravity is straight down in body frame, so the xy part is zero
    assert bfr.projected_gravity_xy(1.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-9)
    # 90 deg pitch: gravity lies fully in the body xy plane
    s = np.sqrt(0.5)
    assert bfr.projected_gravity_xy(s, 0.0, s, 0.0) == pytest.approx(1.0, abs=1e-9)
    # yaw alone never tilts the robot, so it cannot move this metric
    assert bfr.projected_gravity_xy(s, 0.0, 0.0, s) == pytest.approx(0.0, abs=1e-9)


def test_every_bench_key_is_emitted_renamed_or_documented_as_omitted():
    """No BENCH key may be silently dropped.

    `plot_radar` normalizes against a baseline and fails closed on a MISSING key, but it
    cannot see a key that is present under a sim name while measuring something else. This
    pins the partition: the union of emitted, renamed and OMITTED must be the whole schema.
    """
    emitted = {
        "err_yaw", "orient_dev", "omega_xy", "act_legs_rad", "ajit_legs", "ajit_legs_p95",
        "ank_roll_span_l", "ank_roll_span_r", "ank_pitch_span_l", "ank_pitch_span_r",
        "ub_pose_dev", "ub_arm_vel", "stride_period_s", "mech_power_w", "cot",
    }
    renamed = {"err_vx", "err_vy"}  # -> err_vx_est / err_vy_est when the reference is an estimate
    bench_schema = emitted | renamed | set(bfr.OMITTED)
    play = (REPO / "scripts/play.py").read_text()
    block = play.split("seed_results.append({", 1)[1].split("})", 1)[0]
    import re
    keys = set(re.findall(r'^\s*"([a-z0-9_]+)":', block, re.M))
    assert keys - bench_schema == set(), f"unclassified BENCH key(s): {sorted(keys - bench_schema)}"
    assert emitted & set(bfr.OMITTED) == set(), "a key is both emitted and documented as omitted"


def test_omissions_each_state_a_reason():
    for k, why in bfr.OMITTED.items():
        assert len(why) > 15, f"{k} has no real reason recorded"


def test_reference_column_names_handle_the_ach_v_exception():
    """`ach_v` breaks the estimator bank's `<prefix>_<axis>` naming.

    The columns are `ach_vx`/`ach_vy`. Only the SIM branch reaches that name, because
    `ach_v*` is dead on hardware, so a naive suffix concat passes every hardware test and
    then dies on the first sim-bridge log.
    """
    assert bfr.ref_columns("ach_v") == ("ach_vx", "ach_vy")
    assert bfr.ref_columns("est_v_compl") == ("est_v_compl_x", "est_v_compl_y")
    assert bfr.ref_columns("est_v_legodom") == ("est_v_legodom_x", "est_v_legodom_y")



def load_align_fixture(session):
    f = pd.read_csv(FIX / f"{session}.align_flight.csv.gz")
    a = pd.read_csv(FIX / f"{session}.align_telemetry.csv.gz")
    return (f["t"].to_numpy(float), f["meas_q3"].to_numpy(float),
            a["time"].to_numpy(float), a["q3"].to_numpy(float))


# (b at t_telem = 0, skew band ppm) measured on the full-rate logs. Each session guards a
# different defect: 13-42-55 carries a non-affine excursion in its first 60 s that a fixed
# 0.5 s rejection gate bends the fit around (-1648 ppm at 116 ms), and 09-07's -10.2 s
# offset is outside the fine search, so without the coarse stage it fits -16021 ppm at
# correlation 0.58. The other session is unaffected by each defect, which is the point.
REAL_ALIGNMENT = {
    "2026-08-27_13-42-55": (-3.339, (-2700, -2300)),
    "2026-09-07_11-44-50": (-10.169, (-3600, -3100)),
}


@pytest.mark.parametrize("session", sorted(REAL_ALIGNMENT))
def test_alignment_on_real_session_pairs(session):
    """The two real pairs that broke the alignment, each on the defect it exposed."""
    b_exp, (m_lo, m_hi) = REAL_ALIGNMENT[session]
    fit = bfr.fit_alignment(*load_align_fixture(session))
    assert fit is not None
    assert fit["b"] == pytest.approx(b_exp, abs=0.05)
    assert m_lo < fit["m"] * 1e6 < m_hi, f"skew {fit['m'] * 1e6:+.0f} ppm"
    assert fit["resid_ms"] < 40.0
    assert fit["xcorr"] > 0.9


def _touch(path, stamp_end):
    import datetime as _dt
    import os
    path.write_text("")
    ts = _dt.datetime.strptime(stamp_end, "%Y-%m-%d %H:%M:%S").timestamp()
    os.utime(path, (ts, ts))
    return path


def test_find_pair_selects_by_recording_interval_overlap(tmp_path):
    """Neither filename is a first-sample time; both files being written at once is.

    The real 2026-09-07 case: the telemetry file starts +125 s after the flight filename,
    past any forward window sized on 2026-08-27 (+5..+68 s), yet both files were written
    together from 11:46:55 to 11:49:42. A file that ended before the flight started must
    lose, and among overlapping files the larger overlap wins.
    """
    traj = tmp_path / "traj"
    traj.mkdir()
    flight = _touch(tmp_path / "2026-09-07_11-44-50.csv", "2026-09-07 11:49:42")
    _touch(traj / "all_joints_2026-09-07_11-30-00_earlier_session.csv", "2026-09-07 11:44:30")
    _touch(traj / "all_joints_2026-09-07_11-49-30_brief_overlap.csv", "2026-09-07 11:52:00")
    _touch(traj / "all_joints_2026-09-07_11-46-55_real.csv", "2026-09-07 11:50:42")
    p, start_offset, overlap = bfr.find_pair(flight, traj)
    assert p.name == "all_joints_2026-09-07_11-46-55_real.csv"
    assert start_offset == pytest.approx(125.0)
    assert overlap == pytest.approx(167.0, abs=1.0)


def test_find_pair_refuses_a_file_started_after_the_flight_stopped(tmp_path):
    """+70 s after the flight filename sits inside any plausible forward window, but the
    flight recorder had already stopped writing: there is nothing to align to."""
    traj = tmp_path / "traj"
    traj.mkdir()
    flight = _touch(tmp_path / "2026-09-07_12-00-00.csv", "2026-09-07 12:00:40")
    _touch(traj / "all_joints_2026-09-07_12-01-10_next.csv", "2026-09-07 12:04:00")
    assert bfr.find_pair(flight, traj) is None


def test_missing_traj_dir_fails_loudly(tmp_path):
    """A typo'd directory used to glob to nothing and read as "no pair for this session"."""
    flight = _touch(tmp_path / "2026-09-07_11-44-50.csv", "2026-09-07 11:49:42")
    with pytest.raises(SystemExit, match="does not exist"):
        bfr.find_pair(flight, tmp_path / "no_such_dir")


def _energy_inputs(run_s, telem_s, speed=0.5, power_w=100.0):
    """One Run of `run_s` walking at a constant speed, paired with a `telem_s` telemetry log.

    Everything is constant, so the right answers are exact by construction: the credited
    distance is speed * run_s and the mean power is power_w, no matter how far past the Run
    the telemetry kept recording.
    """
    tf = np.arange(0.0, run_s, 1 / 500.0)
    n = len(tf)
    F = {
        "t": tf,
        "cmd": np.column_stack([np.full(n, speed), np.zeros(n), np.zeros(n)]),
        "df": pd.DataFrame({"est_v_compl_x": np.full(n, speed),
                            "est_v_compl_y": np.zeros(n)}),
    }
    ta = np.arange(0.0, telem_s, 1 / 50.0)
    m = len(ta)
    tau = np.zeros((m, 27))
    dq = np.zeros((m, 27))
    tau[:, 0] = power_w          # one joint carries all of it: sum |tau*dq| = power_w
    dq[:, 0] = 1.0
    return F, {"t": ta, "tau": tau, "dq": dq, "knee": np.zeros(m)}, {"b": 0.0, "m": 0.0}


def test_energy_ignores_telemetry_recorded_outside_the_run():
    """np.interp CLAMPS, so a Run that ENDS WALKING used to earn distance for every trailing
    telemetry sample: the held gate stayed 1 and the held speed stayed nonzero.

    Latent while a flight recorder file spanned a whole controller process; it bit under the
    Run-scoped scoring of ADR-0012, where a 186 s Run pairs with a 232 s telemetry log. On
    2026-09-14_14-48-55 it reported 45.12 m against an independently computed 11.5 m, and
    diluted mech_power_w to 79 W, the lowest of the session, by averaging in samples from
    after the robot had stopped.
    """
    F, T, align = _energy_inputs(run_s=10.0, telem_s=30.0)
    out = bfr._energy(F, T, align, "est_v_compl")
    assert out["cot_distance_m"] == pytest.approx(5.0, abs=0.05)   # 0.5 m/s * 10 s, not 30 s
    assert out["mech_power_w"] == pytest.approx(100.0, rel=1e-6)


def test_energy_is_unchanged_when_the_telemetry_fits_inside_the_run():
    """The guard must not shorten a pair that was already contained: the eight 2026-09-14
    Runs other than 14-48-55 reproduced to every printed digit after the fix, and that
    invariant is what makes the correction safe to apply retroactively."""
    F, T, align = _energy_inputs(run_s=30.0, telem_s=10.0)
    out = bfr._energy(F, T, align, "est_v_compl")
    assert out["cot_distance_m"] == pytest.approx(5.0, abs=0.05)   # bounded by the telemetry
    assert out["mech_power_w"] == pytest.approx(100.0, rel=1e-6)
