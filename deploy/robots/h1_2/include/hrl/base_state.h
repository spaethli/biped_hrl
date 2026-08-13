// A1 (HIRO) deploy base-state estimator — the geometry behind the base linear velocity
// and base height that the goal space needs and that a real H1-2 cannot read anywhere.
//
// Sibling of hrl/goal_space.h: that file mirrors the training-side goal encoding, this one
// mirrors the DEPLOYABILITY bench that decided what the robot may condition on, the
// "waist-corr" rung of scripts/play.py --check-vel-increment.
//
// WHY AN INCREMENT IS ENOUGH. The LL's goal obs is V* - s_i, and in `delta` mode
// V* = s_t0 + scale*g, so
//     V* - s_i = scale*g - (s_i - s_t0)
// and the absolute base velocity CANCELS. Deploy therefore needs only the within-window
// increment, which is re-zeroed every c steps so integration drift cannot accumulate.
// This matters because rt/sportmodestate — the only base-velocity/height source the deploy
// had — publishes ZEROS once our own low-level controller has command of the robot.
//
// FRAMES. The H1-2's IMU (and the bridge's `imu` site) sit on TORSO_LINK, behind the
// yaw-axis `torso_joint` whose anchor coincides with the pelvis origin. The goal space uses
// the PELVIS velocity, so pelvis <- torso is a pure Rz(psi) with a constant torso-frame
// lever arm r:
//     v_pelvis_P = Rz(psi) * (v_site_T - w_T x r)
// Uncorrected, that lever arm is the single largest error term (bench: vy 0.227 vs 0.025).
//
// GEOMETRY PROVENANCE. Every constant is read off
// src/assets/robots/unitree_h1_2/xmls/h1_2.xml (the training model), with the source line
// noted at each use. Corrected 2026-08-04: this note used to claim "the bridge's
// scene_h1_2.xml carries identical numbers". The in-repo scene_h1_2.xml is NOT what the
// bridge loads (see deployment.md) -- it loads
// /opt/unitree_mujoco/unitree_robots/h1_2/h1_2_handless*.xml, whose feet are a single
// MESH geom rather than the training model's seven capsules. The imu site offset and the
// link lengths ARE identical, but the sole plane is not: the bridge's lowest foot surface
// sits at -0.058 m in the ankle_roll frame vs -0.045 m here (measured via MuJoCo geom
// AABBs on both models). Keep these constants matched to the TRAINING/real robot -- the
// 13 mm difference is a bridge modelling artifact and must not be tuned away here.
#pragma once

#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace hrl
{

// Site "imu" pos, h1_2.xml:145 — in the TORSO frame (the site declares no quat, so its
// axes are torso_link's).
static const Eigen::Vector3f kImuOffsetT(-0.04452f, -0.01891f, 0.27756f);
// |g|; matches the bench's torch.tensor(uenv.cfg.sim.mujoco.gravity).norm().
static constexpr float kGravity = 9.81f;

// Rotate v about z by psi (psi < 0 gives the inverse). See the FRAMES note above:
// torso_link hangs off the pelvis through `torso_joint`, a z-axis revolute whose anchor is
// the pelvis origin and whose child body carries no offset (h1_2.xml:139-145).
inline Eigen::Vector3f rz(float psi, const Eigen::Vector3f& v)
{
    const float c = std::cos(psi), s = std::sin(psi);
    return Eigen::Vector3f(c * v.x() - s * v.y(), s * v.x() + c * v.y(), v.z());
}

// Within-window increment of the PELVIS-frame base linear velocity.
//   dv_int : gravity-free velocity increment of the imu SITE, torso frame, integrated at
//            the control rate since the window started (State_RLHRL::run(), 1 kHz).
//   psi    : waist encoder now;   w_T : gyro now (torso frame)
//   lev0   : Rz(psi_0) * (w_T(t0) x r), captured at the window start.
// Derivation: subtract v_pelvis_P(t0) from v_pelvis_P(i) and substitute
// v_site_T(i) = v_site_T(t0) + dv_int. The [Rz(psi_i) - Rz(psi_0)] * v_site_T(t0) residual
// is unmodellable without absolute velocity and is dropped; it is small while the waist
// barely turns inside one window, and the bench's reported RMS already includes it.
// Returns exactly 0 at the window start (dv_int = 0 and lev0 cancels the lever term).
inline Eigen::Vector3f base_vel_increment(float psi, const Eigen::Vector3f& w_T,
                                          const Eigen::Vector3f& dv_int,
                                          const Eigen::Vector3f& lev0)
{
    return rz(psi, dv_int - w_T.cross(kImuOffsetT)) + lev0;
}

// Lowest foot-sole z in the PELVIS frame, from the 12 leg encoders in sdk order
// (0-5 left, 6-11 right; yaw, pitch, roll, knee, ankle_pitch, ankle_roll — the chain order
// in h1_2.xml:56-95). Base height above the support plane is -this. Valid in single AND
// double support because it minimises over both feet, so no contact sensing is needed —
// which matters: unitree_hg::msg::LowState has NO foot-force field at all (its fields are
// version, mode_pr, mode_machine, tick, imu_state, motor_state[35], wireless_remote,
// reserve, crc).
//
// Link offsets (h1_2.xml, left leg; right mirrors the y sign):
//   pelvis -> left_hip_yaw_link   (0, 0.0875, -0.1632)  :56
//   hip_yaw -> hip_pitch          (0, 0.0755,  0)       :61
//   hip_pitch -> hip_roll         coincident (no pos)   :65
//   hip_roll -> knee              (0, 0, -0.4)  thigh   :70
//   knee -> ankle_pitch           (0, 0, -0.4)  shank   :75
//   ankle_pitch -> ankle_roll     (0, 0, -0.02)         :79
//   sole                          z = -0.035 - 0.01     :84-90 (foot capsule centre line,
//                                 minus the `foot_capsule` class radius, h1_2.xml:13)
// The sole term evaluates the capsule's heel/toe ENDPOINTS, not its centre: a capsule's
// lowest surface point is (segment minimum) - r, and a linear function on a segment attains
// its minimum at an endpoint, so two points are EXACT for the foot's sagittal extent. The
// lateral spread of the seven parallel capsules (|y| <= 0.03) is not modelled; measured
// against mj_kinematics that costs rms 4.4 mm / max 11 mm over a walking-envelope sweep,
// and exactly 0 at the nominal pose — which is the pose the caller anchors at.
// One leg's ankle_roll frame (origin + orientation) in the PELVIS frame, from the 6 leg
// encoders of that leg. The shared kinematic core: both consumers below start here and
// differ only in which POINT of the foot they then evaluate, so there is exactly one copy
// of the link table in this file.
// Scalar-templated core. The float instantiation below is the shipped path and is
// unchanged; base_estimators.h instantiates it at DOUBLE, because an error-state covariance
// recursion whose P spans 1e2 (inflated position) to 1e-2 carries a condition number ~1e4,
// where float32 epsilon predicts ~1e-3 relative error -- measured, and enough to break the
// C++/Python parity bar on its own (doc/hrl/h1_2_ekf_design.md §13.5 T2).
template <typename S>
inline void leg_fk_t(const S q[12], int leg, Eigen::Matrix<S, 3, 1>& p,
                     Eigen::Matrix<S, 3, 3>& R)
{
    using V3 = Eigen::Matrix<S, 3, 1>;
    const S sy = (leg == 0) ? S(1) : S(-1);
    const S* j = q + 6 * leg;
    R = Eigen::Matrix<S, 3, 3>(Eigen::AngleAxis<S>(j[0], V3::UnitZ()));
    p = V3(S(0), sy * S(0.0875), S(-0.1632));
    p += R * V3(S(0), sy * S(0.0755), S(0));
    R = R * Eigen::AngleAxis<S>(j[1], V3::UnitY());
    R = R * Eigen::AngleAxis<S>(j[2], V3::UnitX());
    p += R * V3(S(0), S(0), S(-0.4));
    R = R * Eigen::AngleAxis<S>(j[3], V3::UnitY());
    p += R * V3(S(0), S(0), S(-0.4));
    R = R * Eigen::AngleAxis<S>(j[4], V3::UnitY());
    p += R * V3(S(0), S(0), S(-0.02));
    R = R * Eigen::AngleAxis<S>(j[5], V3::UnitX());
}

inline void leg_fk(const float q[12], int leg, Eigen::Vector3f& p, Eigen::Matrix3f& R)
{
    leg_fk_t<float>(q, leg, p, R);
}

inline float lowest_foot_z(const float q[12])
{
    float lowest = 1e9f;
    for (int leg = 0; leg < 2; ++leg) {
        Eigen::Vector3f p; Eigen::Matrix3f R;
        leg_fk(q, leg, p, R);
        for (float x : {-0.08f, 0.17f})
            lowest = std::min(lowest, (p + R * Eigen::Vector3f(x, 0.0f, -0.045f)).z());
    }
    return lowest;
}

// --- leg odometry: the HL's ABSOLUTE base velocity (2026-08-06) ---------------------
// The one quantity `delta` mode does NOT cancel. base_vel_increment above is 0 at every
// window start BY CONSTRUCTION, and the HL fires exactly there, so an `hl_obs_vel` policy
// running on the IMU increment reads "stationary" at every fire — measured on the bridge
// as 3/3 falls against 0/3 on privileged truth (2026-08-05). This is the replacement, and
// it feeds the HL ONLY: `s`, and therefore the LL's goal delta, keeps the IMU increment.

// Site "left_foot"/"right_foot" pos, h1_2.xml:80,121 — in the ANKLE_ROLL frame. This is
// the point scripts/play.py's --check-leg-odometry bench differences, so it is the point
// whose error the 0.074/0.049 figures (and HlVelJitter's calibration) describe. It is NOT
// the sole point lowest_foot_z evaluates: those capsule endpoints sit up to 0.13 m away in
// x, and p_foot enters the w x p term directly, so the two are not interchangeable.
static const Eigen::Vector3f kFootSiteA(0.04f, 0.0f, -0.04f);

inline Eigen::Vector3f foot_site_b(const float q[12], int leg)
{
    Eigen::Vector3f p; Eigen::Matrix3f R;
    leg_fk(q, leg, p, R);
    return p + R * kFootSiteA;
}

// Instantaneous PELVIS-frame base linear velocity from the stance foot:
//     v = -d(p_foot)/dt - w_P x p_foot
// exact while the stance foot is fixed in the world. No integration anywhere, so unlike
// the IMU increment this cannot drift and needs no window reset.
//
// p / p_prev: both feet's site positions now and one tick ago (the caller caches them, so
// the FK runs twice per tick, not four times). Differencing is PER FOOT — never across the
// stance switch, which would inject the ~0.3 m inter-foot gap as a false velocity spike.
//
// w_P: the PELVIS angular velocity in the pelvis frame. The gyro gives the TORSO one, so
// the caller owes `Rz(psi)*w_gyro - psi_dot*z` (see FRAMES above). Do NOT rotate the return
// value: leg FK is pelvis-referenced already, unlike the torso-mounted IMU.
//
// grav_P: projected gravity in the pelvis frame; stance = the foot further along it. No
// contact sensing (LowState has no foot-force field) and none is wanted: a perfect contact
// ORACLE was measured to buy ~nothing, since 86% of steps are single-support where there is
// no attribution ambiguity, and flight is 0.13% of steps.
inline Eigen::Vector3f leg_odom_velocity(const Eigen::Vector3f p[2],
                                         const Eigen::Vector3f p_prev[2],
                                         float dt, const Eigen::Vector3f& w_P,
                                         const Eigen::Vector3f& grav_P)
{
    const int stance = (p[0].dot(grav_P) >= p[1].dot(grav_P)) ? 0 : 1;
    return -(p[stance] - p_prev[stance]) / dt - w_P.cross(p[stance]);
}

} // namespace hrl
