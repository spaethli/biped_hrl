#pragma once
// =====================================================================================
// Base-velocity estimator arms B-F, ported from scripts/replay_base_estimators.py.
//
// Arm A (the shipped leg odometry) is NOT here -- it stays in base_state.h, untouched, so
// the validated keeper path cannot be disturbed by work on the new arms (test T3).
//
//   B  complementary filter          ComplementaryFilter
//   C  EKF, position-only            BaseEkf<false>
//   C+ C + gravity tilt update       BaseEkf<false>, use_grav
//   D  C + flat-foot orientation     BaseEkf<true>
//   E  Jacobian leg odometry         jacobian_velocity()
//   F  C + vendor attitude           BaseEkf<false>, use_att
//
// PARITY (doc/hrl/h1_2_ekf_design.md §13.2). The Python harness steps on DEDUPED samples
// with the real interval; run() ticks at ~1 kHz while LowState changes at ~500 Hz. So:
// predict() every tick at dt = 1 ms (holding f constant over two 1 ms steps integrates
// identically to one 2 ms step), but the kinematic update fires ONLY on a changed sample.
// Updating every tick would count each measurement twice, shrink P about 2x too fast, and
// bias the filter toward the stationary-foot assumption.
//
// FRAMES. Filter body frame is the TORSO/IMU, as in Rotella. Output is converted to the
// PELVIS once, at the end: v_P = Rz(psi) (C v_W - w_T x r_imu). Leg FK is pelvis-referenced,
// so a foot reaches the torso frame as Rz(-psi) p^P - r_imu.
//
// No heap allocation on any path called from run(): every matrix is fixed-size, and the
// foot update solves a constant-size system with unused rows neutralised (see update_feet).
// =====================================================================================

#include "base_state.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace hrl
{

// The EKF anchors on the SOLE PLANE, not the site kFootSiteA (0.04, 0, -0.04) that leg
// odometry differences. -0.045 is the measured sole; the 5 mm matters because p_foot enters
// the w x p term directly.
static const Eigen::Vector3d kContactA(0.04, 0.0, -0.045);
// IMU site in the torso frame (doc/hrl/h1_2_kinematics.md).
static const Eigen::Vector3d kRimu(-0.04452, -0.01891, 0.27756);
static const Eigen::Vector3d kGravityW(0.0, 0.0, -9.81);
// Double twin of base_state.h's kFootSiteA, the point arm A differences. Arm E uses it so
// its Jacobian describes the same point the incumbent's error figures were measured at.
static const Eigen::Vector3d kFootSiteAd(0.04, 0.0, -0.04);

inline Eigen::Matrix3d skew3(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S <<   0.0, -v.z(),  v.y(),
         v.z(),   0.0, -v.x(),
        -v.y(),  v.x(),   0.0;
    return S;
}

// Rodrigues, with the series branch at w -> 0 so it is exact there (matches so3_exp).
inline Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w)
{
    const double th = w.norm();
    const Eigen::Matrix3d K = skew3(w);
    if (th < 1e-8)
        return Eigen::Matrix3d::Identity() + K + 0.5 * K * K;
    return Eigen::Matrix3d::Identity() + (std::sin(th) / th) * K
           + ((1.0 - std::cos(th)) / (th * th)) * (K * K);
}

inline Eigen::Vector3d so3_log(const Eigen::Matrix3d& R)
{
    const double c = std::max(-1.0, std::min(1.0, (R.trace() - 1.0) * 0.5));
    const double th = std::acos(c);
    const Eigen::Vector3d u(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    if (th < 1e-8)
        return u * 0.5;
    return (th / (2.0 * std::sin(th))) * u;
}

inline Eigen::Matrix3d rz_mat(double psi)
{
    return Eigen::Matrix3d(Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()));
}

// --- kinematic Jacobians -------------------------------------------------------------
// Geometric Jacobian of one foot point w.r.t. that leg's 6 joints, PELVIS frame. Column i
// is a_i x (p_foot - o_i) for J_v and a_i for J_w. The link table is leg_fk's; it is walked
// again here rather than shared because the Jacobian needs each joint's axis and origin as
// it goes, which leg_fk does not surface.
inline void leg_jacobian_pelvis(const double q[12], int leg, const Eigen::Vector3d& offset,
                                Eigen::Matrix<double, 3, 6>& Jv,
                                Eigen::Matrix<double, 3, 6>& Jw,
                                Eigen::Vector3d& pf)
{
    const double sy = (leg == 0) ? 1.0 : -1.0;
    const double* j = q + 6 * leg;
    Eigen::Vector3d axes[6], origins[6];

    Eigen::Matrix3d R(Eigen::AngleAxisd(j[0], Eigen::Vector3d::UnitZ()));
    Eigen::Vector3d p(0.0, sy * 0.0875, -0.1632);
    axes[0] = Eigen::Vector3d::UnitZ();
    origins[0] = p;

    p += R * Eigen::Vector3d(0.0, sy * 0.0755, 0.0);
    origins[1] = p;
    axes[1] = R * Eigen::Vector3d::UnitY();

    R = R * Eigen::AngleAxisd(j[1], Eigen::Vector3d::UnitY());
    origins[2] = p;
    axes[2] = R * Eigen::Vector3d::UnitX();

    R = R * Eigen::AngleAxisd(j[2], Eigen::Vector3d::UnitX());
    p += R * Eigen::Vector3d(0.0, 0.0, -0.4);
    origins[3] = p;
    axes[3] = R * Eigen::Vector3d::UnitY();

    R = R * Eigen::AngleAxisd(j[3], Eigen::Vector3d::UnitY());
    p += R * Eigen::Vector3d(0.0, 0.0, -0.4);
    origins[4] = p;
    axes[4] = R * Eigen::Vector3d::UnitY();

    R = R * Eigen::AngleAxisd(j[4], Eigen::Vector3d::UnitY());
    p += R * Eigen::Vector3d(0.0, 0.0, -0.02);
    origins[5] = p;
    axes[5] = R * Eigen::Vector3d::UnitX();

    R = R * Eigen::AngleAxisd(j[5], Eigen::Vector3d::UnitX());
    pf = p + R * offset;

    for (int i = 0; i < 6; ++i) {
        Jv.col(i) = axes[i].cross(pf - origins[i]);
        Jw.col(i) = axes[i];
    }
}

// Both feet's contact-frame Jacobians in the TORSO/IMU frame, 3x7 each. Column 6 is the
// WAIST -- the seventh DoF between the IMU and the feet, which Rotella's leg-only chain has
// no analogue for:  J_v^I = [Rz(-psi) J_v^P | -z x (p^I + r_imu)],  J_w^I = [Rz(-psi) J_w^P | -z].
inline void foot_jacobians_imu(const double q[12], double psi, const Eigen::Vector3d& offset,
                               Eigen::Matrix<double, 3, 7> Jv[2],
                               Eigen::Matrix<double, 3, 7> Jw[2])
{
    const Eigen::Matrix3d Rzn = rz_mat(-psi);
    const Eigen::Vector3d zhat = Eigen::Vector3d::UnitZ();
    for (int leg = 0; leg < 2; ++leg) {
        Eigen::Matrix<double, 3, 6> Jv_P, Jw_P;
        Eigen::Vector3d pf_P;
        leg_jacobian_pelvis(q, leg, offset, Jv_P, Jw_P, pf_P);
        Jv[leg].leftCols<6>() = Rzn * Jv_P;
        Jw[leg].leftCols<6>() = Rzn * Jw_P;
        const Eigen::Vector3d p_I = Rzn * pf_P - kRimu;
        Jv[leg].col(6) = -zhat.cross(p_I + kRimu);
        Jw[leg].col(6) = -zhat;
    }
}

// Feet in the torso/IMU frame, contact frame: p^I = Rz(-psi) p^P - r_imu.
inline void feet_in_imu(const double q[12], double psi, const Eigen::Vector3d& offset,
                        Eigen::Vector3d p_I[2], Eigen::Matrix3d R_I[2])
{
    const Eigen::Matrix3d Rzn = rz_mat(-psi);
    for (int leg = 0; leg < 2; ++leg) {
        Eigen::Vector3d p;
        Eigen::Matrix3d R;
        leg_fk_t<double>(q, leg, p, R);
        p_I[leg] = Rzn * (p + R * offset) - kRimu;
        R_I[leg] = Rzn * R;
    }
}

// --- contact confidence ---------------------------------------------------------------
// Continuous, from foot height alone, by FK relative to the OTHER foot -- never from the
// filter's own state, which would close a bad-state -> wrong-contact -> worse-state loop.
// "Lower" means FURTHER ALONG PROJECTED GRAVITY, not smaller body-frame z: under torso tilt
// those are different axes, and using body z reported both feet planted 100% of a stepping
// session (2026-08-12).
inline void contact_alpha(const Eigen::Vector3d p_I[2], const Eigen::Vector3d& g_I,
                          double alpha[2], double hyst = 0.04)
{
    const double a0 = p_I[0].dot(g_I), a1 = p_I[1].dot(g_I);
    const double hi = std::max(a0, a1);
    alpha[0] = std::max(0.0, std::min(1.0, 1.0 - (hi - a0) / hyst));
    alpha[1] = std::max(0.0, std::min(1.0, 1.0 - (hi - a1) / hyst));
}

// --- measurement covariance -----------------------------------------------------------
struct MeasCfg
{
    double sigma_q     = 1.64e-5;  // rad, per leg joint   MEASURED (655 s hang)
    double sigma_psi   = 2.61e-5;  // rad, waist encoder   MEASURED
    double sigma_slip  = 3.0e-3;   // m,  foot motion while nominally planted   [assumed]
    double sigma_model = 3.0e-3;   // m,  link-length + sole-geometry error     [assumed]
    double sigma_rot   = 2.0e-2;   // rad, foot orientation model error         [assumed]
};

struct EkfNoise
{
    double acc = 0.0145, gyro = 0.0016;   // densities per sqrt(Hz), MEASURED
    double ba = 1e-4, bw = 1e-5;
    double foot_p = 1e-3, foot_th = 1e-2;
    double att = 5e-2;
};

// R = G Sigma_q G^T + diag(slip, model, rot) + contact inflation, with G the stacked
// kinematic Jacobian (m x 13: 12 leg joints + the waist). Building it this way produces the
// two couplings a per-foot diagonal R silently drops: cross-foot (both feet share the SAME
// waist column, so their noises are correlated -- Rotella assumes they are not, which is
// false here because the IMU sits behind the waist) and position/orientation within a foot.
// Row order [L_pos, L_rot, R_pos, R_rot], rot rows only when UseOri.
template <bool UseOri>
inline void build_R(const Eigen::Matrix<double, 3, 7> Jv[2],
                    const Eigen::Matrix<double, 3, 7> Jw[2],
                    const double alpha[2], const MeasCfg& cfg,
                    Eigen::Matrix<double, UseOri ? 12 : 6, UseOri ? 12 : 6>& R)
{
    constexpr int per = UseOri ? 6 : 3;
    constexpr int M = 2 * per;
    Eigen::Matrix<double, M, 13> G = Eigen::Matrix<double, M, 13>::Zero();
    for (int leg = 0; leg < 2; ++leg) {
        const int r0 = leg * per;
        G.template block<3, 6>(r0, 6 * leg) = Jv[leg].template leftCols<6>();
        G.template block<3, 1>(r0, 12) = Jv[leg].col(6);
        if (UseOri) {
            G.template block<3, 6>(r0 + 3, 6 * leg) = Jw[leg].template leftCols<6>();
            G.template block<3, 1>(r0 + 3, 12) = Jw[leg].col(6);
        }
    }
    Eigen::Matrix<double, 13, 1> sq;
    sq.template head<12>().setConstant(cfg.sigma_q * cfg.sigma_q);
    sq(12) = cfg.sigma_psi * cfg.sigma_psi;
    R = G * sq.asDiagonal() * G.transpose();

    for (int leg = 0; leg < 2; ++leg) {
        const int r0 = leg * per;
        // A foot we are less sure about gets a larger measurement covariance; the same alpha
        // simultaneously frees its process noise (BaseEkf::predict).
        const double a = std::max(alpha[leg], 1e-3);
        const double infl = (1.0 / a - 1.0) * (1.0 / a - 1.0) * cfg.sigma_slip * cfg.sigma_slip;
        R.template block<3, 3>(r0, r0) += Eigen::Matrix3d::Identity()
            * (cfg.sigma_slip * cfg.sigma_slip + cfg.sigma_model * cfg.sigma_model + infl);
        if (UseOri)
            R.template block<3, 3>(r0 + 3, r0 + 3) +=
                Eigen::Matrix3d::Identity() * (cfg.sigma_rot * cfg.sigma_rot + infl);
    }
}

// --- the EKF ---------------------------------------------------------------------------
// Error state dx = [dr, dv, dphi, dp_L, dp_R, (dth_L, dth_R,) db_f, db_w], R^27 with the
// orientation block and R^21 without. Attitude convention R_true = R_hat exp([dphi]x)
// (right-multiplied, body-frame error), which reproduces Rotella's F_c and H exactly.
template <bool UseOri>
class BaseEkf
{
public:
    static constexpr int N = UseOri ? 27 : 21;
    static constexpr int M = UseOri ? 12 : 6;   // stacked both-feet measurement
    static constexpr int per = UseOri ? 6 : 3;
    static constexpr int i_r = 0, i_v = 3, i_phi = 6, i_p = 9;
    static constexpr int i_th = UseOri ? 15 : -1;
    static constexpr int i_bf = UseOri ? 21 : 15;
    static constexpr int i_bw = i_bf + 3;

    using MatN = Eigen::Matrix<double, N, N>;
    using VecN = Eigen::Matrix<double, N, 1>;
    using MatM = Eigen::Matrix<double, M, M>;

    BaseEkf() { reset(); }

    void reset(const Eigen::Matrix3d* R0 = nullptr, const Eigen::Vector3d* v0 = nullptr)
    {
        r_.setZero();
        v_ = v0 ? *v0 : Eigen::Vector3d::Zero();
        R_ = R0 ? *R0 : Eigen::Matrix3d::Identity();
        p_[0].setZero(); p_[1].setZero();
        Rf_[0].setIdentity(); Rf_[1].setIdentity();
        bf_.setZero(); bw_.setZero();
        P_ = MatN::Identity() * 1e-2;
        P_.template block<3, N>(i_r, 0) *= 1e2;   // absolute position is unobservable anyway
        anchored_[0] = anchored_[1] = false;
        n_reset_++;
    }

    // Seed the velocity from leg odometry (the harness's v0_pelvis warm start): without it a
    // short run scores the filter's transient rather than the filter.
    void warm_start(const Eigen::Vector3d& v_pelvis, double psi, const Eigen::Vector3d& gyro_T)
    {
        const Eigen::Vector3d v_T = rz_mat(-psi) * v_pelvis + gyro_T.cross(kRimu);
        v_ = R_ * v_T;                      // R_ = C^T maps torso -> world
        P_.template block<3, 3>(i_v, i_v) = Eigen::Matrix3d::Identity() * (0.25 * 0.25);
    }

    void predict(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, double dt,
                 const double alpha[2], const EkfNoise& s)
    {
        const Eigen::Vector3d f = acc - bf_;
        const Eigen::Vector3d w = gyro - bw_;
        const Eigen::Vector3d a_w = R_ * f + kGravityW;
        r_ += v_ * dt + 0.5 * a_w * dt * dt;
        v_ += a_w * dt;
        R_ = R_ * so3_exp(w * dt);

        MatN F = MatN::Identity();
        F.template block<3, 3>(i_r, i_v) = Eigen::Matrix3d::Identity() * dt;
        F.template block<3, 3>(i_v, i_phi) = -R_ * skew3(f) * dt;
        F.template block<3, 3>(i_v, i_bf) = -R_ * dt;
        F.template block<3, 3>(i_phi, i_phi) = so3_exp(-w * dt);
        F.template block<3, 3>(i_phi, i_bw) = -Eigen::Matrix3d::Identity() * dt;

        MatN Q = MatN::Zero();
        Q.template block<3, 3>(i_v, i_v) = Eigen::Matrix3d::Identity() * (s.acc * s.acc) * dt;
        Q.template block<3, 3>(i_phi, i_phi) = Eigen::Matrix3d::Identity() * (s.gyro * s.gyro) * dt;
        Q.template block<3, 3>(i_bf, i_bf) = Eigen::Matrix3d::Identity() * (s.ba * s.ba) * dt;
        Q.template block<3, 3>(i_bw, i_bw) = Eigen::Matrix3d::Identity() * (s.bw * s.bw) * dt;
        // A foot is free to move exactly as far as it is NOT in contact.
        for (int i = 0; i < 2; ++i) {
            const double free = (1.0 - alpha[i]) + 1e-6;
            Q.template block<3, 3>(i_p + 3 * i, i_p + 3 * i) =
                Eigen::Matrix3d::Identity() * (s.foot_p * s.foot_p + free * free) * dt;
            if (UseOri)
                Q.template block<3, 3>(i_th + 3 * i, i_th + 3 * i) =
                    Eigen::Matrix3d::Identity() * (s.foot_th * s.foot_th + free * free) * dt;
        }
        P_ = F * P_ * F.transpose() + Q;
    }

    // Both feet in ONE update, so the cross-foot waist correlation in R_meas is actually
    // used (updating foot-at-a-time discards those off-diagonal blocks).
    //
    // Rows for a foot that is not in contact are NEUTRALISED rather than removed: H row = 0,
    // innovation = 0, R row/col = e_i. That leaves K's column exactly zero and the Mahalanobis
    // term unchanged, so the result is identical to solving the smaller system -- but the
    // system stays a compile-time size and nothing allocates inside run().
    bool update_feet(const Eigen::Vector3d s_p[2], const Eigen::Matrix3d* s_R,
                     const double alpha[2], const MatM& R_meas, double gate = 25.0)
    {
        Eigen::Matrix<double, M, N> H = Eigen::Matrix<double, M, N>::Zero();
        Eigen::Matrix<double, M, 1> e = Eigen::Matrix<double, M, 1>::Zero();
        bool used[M];
        for (int i = 0; i < M; ++i) used[i] = false;
        int m_used = 0;

        for (int i = 0; i < 2; ++i) {
            if (alpha[i] <= 1e-3)
                continue;
            if (!anchored_[i]) {
                // First contact: place the foot where FK says it is rather than fighting a
                // stale pose. Avoids one large transient innovation per touchdown.
                p_[i] = r_ + R_ * s_p[i];
                if (UseOri && s_R)
                    Rf_[i] = R_ * s_R[i];
                anchored_[i] = true;
                continue;
            }
            const int r0 = i * per;
            e.template segment<3>(r0) = s_p[i] - h_pos(i);
            H.template block<3, N>(r0, 0) = H_pos(i);
            for (int k = 0; k < 3; ++k) used[r0 + k] = true;
            m_used += 3;
            if (UseOri && s_R) {
                e.template segment<3>(r0 + 3) = so3_log(h_rot(i).transpose() * s_R[i]);
                H.template block<3, N>(r0 + 3, 0) = H_rot(i);
                for (int k = 0; k < 3; ++k) used[r0 + 3 + k] = true;
                m_used += 3;
            }
        }
        if (m_used == 0)
            return false;

        MatM Rm = R_meas;
        for (int i = 0; i < M; ++i) {
            if (used[i])
                continue;
            Rm.row(i).setZero();
            Rm.col(i).setZero();
            Rm(i, i) = 1.0;
        }
        MatM S = H * P_ * H.transpose() + Rm;
        Eigen::FullPivLU<MatM> lu(S);
        if (!lu.isInvertible())
            return false;
        const MatM Sinv = lu.inverse();
        if (e.dot(Sinv * e) > gate * static_cast<double>(m_used))
            return false;                                   // Mahalanobis gate
        const Eigen::Matrix<double, N, M> K = P_ * H.transpose() * Sinv;
        inject(K * e);
        const MatN IKH = MatN::Identity() - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * Rm * K.transpose();
        return true;
    }

    // Accelerometer as a direct TILT observation, applied only while near-static. NOT in
    // Rotella or Bloesch: in the strapdown form the accelerometer is an input, never a
    // measurement, so attitude is corrected only through the weak (C(p-r))x dphi term. On
    // this robot that proved far too weak -- the position-only arm's tilt error ran to
    // 72.8 deg, which leaks gravity into horizontal specific force. Yaw is untouched by
    // construction: [h]x is rank 2 with its nullspace along h.
    bool update_gravity(const Eigen::Vector3d& acc, double sigma = 0.05, double tol = 0.5)
    {
        const double n = acc.norm();
        if (std::fabs(n - 9.81) > tol)
            return false;
        const Eigen::Vector3d m = acc / n;
        const Eigen::Vector3d h = R_.transpose() * Eigen::Vector3d::UnitZ();
        Eigen::Matrix<double, 3, N> H = Eigen::Matrix<double, 3, N>::Zero();
        H.template block<3, 3>(0, i_phi) = skew3(h);
        apply3(H, m - h, sigma * sigma);
        return true;
    }

    // Arm F only: the vendor quaternion as a direct attitude measurement.
    void update_attitude(const Eigen::Matrix3d& R_meas, double sigma)
    {
        Eigen::Matrix<double, 3, N> H = Eigen::Matrix<double, 3, N>::Zero();
        H.template block<3, 3>(0, i_phi) = Eigen::Matrix3d::Identity();
        apply3(H, so3_log(R_.transpose() * R_meas), sigma * sigma);
    }

    // World velocity -> torso -> pelvis:  v_P = Rz(psi) (C v_W - w_T x r_imu).
    Eigen::Vector3d velocity_pelvis(double psi, const Eigen::Vector3d& gyro_T) const
    {
        const Eigen::Vector3d v_T = R_.transpose() * v_;
        return rz_mat(psi) * (v_T - (gyro_T - bw_).cross(kRimu));
    }

    bool finite() const { return v_.allFinite() && R_.allFinite() && P_.allFinite(); }
    int n_reset() const { return n_reset_; }
    const Eigen::Matrix3d& attitude() const { return R_; }

private:
    Eigen::Vector3d h_pos(int i) const { return R_.transpose() * (p_[i] - r_); }
    Eigen::Matrix3d h_rot(int i) const { return R_.transpose() * Rf_[i]; }

    Eigen::Matrix<double, 3, N> H_pos(int i) const
    {
        Eigen::Matrix<double, 3, N> H = Eigen::Matrix<double, 3, N>::Zero();
        const Eigen::Matrix3d C = R_.transpose();
        H.template block<3, 3>(0, i_r) = -C;
        H.template block<3, 3>(0, i_phi) = skew3(h_pos(i));
        H.template block<3, 3>(0, i_p + 3 * i) = C;
        return H;
    }

    Eigen::Matrix<double, 3, N> H_rot(int i) const
    {
        Eigen::Matrix<double, 3, N> H = Eigen::Matrix<double, 3, N>::Zero();
        H.template block<3, 3>(0, i_phi) = -h_rot(i).transpose();
        if (UseOri)
            H.template block<3, 3>(0, i_th + 3 * i) = Eigen::Matrix3d::Identity();
        return H;
    }

    void apply3(const Eigen::Matrix<double, 3, N>& H, const Eigen::Vector3d& e, double var)
    {
        const Eigen::Matrix3d Rm = Eigen::Matrix3d::Identity() * var;
        const Eigen::Matrix3d S = H * P_ * H.transpose() + Rm;
        const Eigen::Matrix<double, N, 3> K = P_ * H.transpose() * S.inverse();
        inject(K * e);
        const MatN IKH = MatN::Identity() - K * H;
        P_ = IKH * P_ * IKH.transpose() + K * Rm * K.transpose();
    }

    void inject(const VecN& dx)
    {
        r_ += dx.template segment<3>(i_r);
        v_ += dx.template segment<3>(i_v);
        R_ = R_ * so3_exp(dx.template segment<3>(i_phi));
        for (int i = 0; i < 2; ++i) {
            p_[i] += dx.template segment<3>(i_p + 3 * i);
            if (UseOri)
                Rf_[i] = Rf_[i] * so3_exp(dx.template segment<3>(i_th + 3 * i));
        }
        bf_ += dx.template segment<3>(i_bf);
        bw_ += dx.template segment<3>(i_bw);
    }

    Eigen::Vector3d r_, v_, bf_, bw_;
    Eigen::Matrix3d R_, Rf_[2];
    Eigen::Vector3d p_[2];
    MatN P_;
    bool anchored_[2] = {false, false};
    int n_reset_ = 0;
};

// --- arm B: complementary filter --------------------------------------------------------
// High-pass the gravity-free accelerometer, low-pass the leg odometry, in the PELVIS frame.
// One time constant, no states, no contact logic. If the EKF cannot beat this it has not
// earned 27 states.
class ComplementaryFilter
{
public:
    explicit ComplementaryFilter(double tau = 0.20) : tau_(tau) { v_.setZero(); }

    void reset() { v_.setZero(); }

    // a_free_P: gravity-removed specific force in the PELVIS frame. v_legodom: arm A's
    // reading, or nullptr when it has none this tick (stance switch).
    const Eigen::Vector3d& step(const Eigen::Vector3d& a_free_P,
                                const Eigen::Vector3d* v_legodom, double dt)
    {
        if (dt <= 0.0)
            return v_;
        const Eigen::Vector3d pred = v_ + a_free_P * dt;
        if (v_legodom && v_legodom->allFinite()) {
            const double a = dt / (tau_ + dt);
            v_ = (1.0 - a) * pred + a * (*v_legodom);
        } else {
            v_ = pred;
        }
        return v_;
    }

private:
    double tau_;
    Eigen::Vector3d v_;
};

// --- arm E: Jacobian leg odometry --------------------------------------------------------
// v = -(J qdot) - w_P x p_stance. Same information as arm A with the position differencing
// removed -- but dq is differentiated on the MOTOR BOARD, so this measures where the
// differentiation happens, not its absence.
inline Eigen::Vector3d jacobian_velocity(const double q[12], const double dq[12],
                                         const Eigen::Vector3d& w_P,
                                         const Eigen::Vector3d& grav_P,
                                         const Eigen::Vector3d& offset = kFootSiteAd)
{
    Eigen::Vector3d pf[2];
    Eigen::Matrix<double, 3, 6> Jv[2], Jw[2];
    for (int leg = 0; leg < 2; ++leg)
        leg_jacobian_pelvis(q, leg, offset, Jv[leg], Jw[leg], pf[leg]);
    const int stance = (pf[0].dot(grav_P) >= pf[1].dot(grav_P)) ? 0 : 1;
    Eigen::Matrix<double, 6, 1> qd;
    for (int k = 0; k < 6; ++k)
        qd(k) = dq[6 * stance + k];
    return -(Jv[stance] * qd) - w_P.cross(pf[stance]);
}

// --- the bank: all seven arms, one call ---------------------------------------------------
// Owned by both State_RLHRL (A1) and State_RLBase (A0). A0's observation vector has no base
// linear velocity term, so there every arm is passive and nothing selects one; A1 routes
// out[est_arm] into obs["hl_vel"]. Output order is kEstArmNames.
class EstimatorBank
{
public:
    // q/dq: 13 joints, encoder-zero offset already applied. acc/gyro: torso frame, gravity
    // INCLUDED in acc. g_T: unit gravity direction in the torso frame (= C * (0,0,-1)).
    // R_wt: torso -> world. v_legodom: arm A from the caller's own guarded path, or nullptr
    // to have the bank difference it internally (it keeps its own previous foot positions).
    // dt: the loop period; predict runs every call, the kinematic update only when the
    // sensor block actually changed.
    void step(const float q[13], const float dq[13], const Eigen::Vector3f& acc,
              const Eigen::Vector3f& gyro, const Eigen::Vector3f& g_T,
              const Eigen::Matrix3f& R_wt, const Eigen::Vector3f* v_legodom, float dt,
              Eigen::Vector3d out[7])
    {
        double q_d[13], dq_d[13], key[19];
        for (int k = 0; k < 13; ++k) { q_d[k] = q[k]; dq_d[k] = dq[k]; }
        for (int k = 0; k < 3; ++k) { key[k] = acc[k]; key[3 + k] = gyro[k]; }
        for (int k = 0; k < 13; ++k) key[6 + k] = q_d[k];
        bool fresh = !key_valid_;
        for (int k = 0; k < 19 && !fresh; ++k) fresh = (key[k] != prev_key_[k]);
        for (int k = 0; k < 19; ++k) prev_key_[k] = key[k];
        key_valid_ = true;

        const double psi = q_d[12];
        const Eigen::Vector3d acc_d = acc.cast<double>(), gyro_d = gyro.cast<double>();
        const Eigen::Vector3d g_I = g_T.cast<double>();
        const Eigen::Matrix3d Rw = R_wt.cast<double>();
        Eigen::Vector3d w_P = rz_mat(psi) * gyro_d;
        w_P.z() -= dq_d[12];
        const Eigen::Vector3d grav_P = rz_mat(psi) * g_I;

        // Arm A: the caller's value if it owns one, else differenced here.
        Eigen::Vector3d v_a = Eigen::Vector3d::Zero();
        bool a_ok = false;
        if (v_legodom) {
            v_a = v_legodom->cast<double>();
            a_ok = true;
        } else {
            Eigen::Vector3d p_now[2];
            for (int lg = 0; lg < 2; ++lg) {
                Eigen::Vector3d p; Eigen::Matrix3d R;
                leg_fk_t<double>(q_d, lg, p, R);
                p_now[lg] = p + R * kFootSiteAd;
            }
            const int st = (p_now[0].dot(grav_P) >= p_now[1].dot(grav_P)) ? 0 : 1;
            if (a_prev_valid_ && st == a_prev_stance_ && dt > 0.0f) {
                v_a = -(p_now[st] - a_prev_[st]) / dt - w_P.cross(p_now[st]);
                a_ok = true;
            }
            a_prev_[0] = p_now[0]; a_prev_[1] = p_now[1];
            a_prev_stance_ = st; a_prev_valid_ = true;
        }

        // Attitude is available on tick ONE, so reset immediately -- predicting even once
        // from an identity attitude leaks the full gravity vector into horizontal specific
        // force. The velocity warm start has to wait for arm A, which needs a previous foot
        // position, so it lands on the next tick; both happen BEFORE that tick's predict, so
        // the filter never propagates an un-warm-started velocity.
        if (!reset_done_) {
            ekf_c_.reset(&Rw); ekf_cg_.reset(&Rw); ekf_att_.reset(&Rw); ekf_rot_.reset(&Rw);
            reset_done_ = true;
        }
        if (!primed_ && a_ok) {
            ekf_c_.warm_start(v_a, psi, gyro_d);
            ekf_cg_.warm_start(v_a, psi, gyro_d);
            ekf_att_.warm_start(v_a, psi, gyro_d);
            ekf_rot_.warm_start(v_a, psi, gyro_d);
            primed_ = true;
        }

        Eigen::Vector3d p_I[2];
        Eigen::Matrix3d R_I[2];
        feet_in_imu(q_d, psi, kContactA, p_I, R_I);
        double alpha[2];
        contact_alpha(p_I, g_I, alpha);

        ekf_c_.predict(acc_d, gyro_d, dt, alpha, noise_);
        ekf_cg_.predict(acc_d, gyro_d, dt, alpha, noise_);
        ekf_rot_.predict(acc_d, gyro_d, dt, alpha, noise_);
        ekf_att_.predict(acc_d, gyro_d, dt, alpha, noise_);
        if (fresh) {
            Eigen::Matrix<double, 3, 7> Jv[2], Jw[2];
            foot_jacobians_imu(q_d, psi, kContactA, Jv, Jw);
            Eigen::Matrix<double, 6, 6> R6;
            Eigen::Matrix<double, 12, 12> R12;
            build_R<false>(Jv, Jw, alpha, meas_, R6);
            build_R<true>(Jv, Jw, alpha, meas_, R12);
            ekf_cg_.update_gravity(acc_d);
            ekf_att_.update_attitude(Rw, noise_.att);
            ekf_c_.update_feet(p_I, nullptr, alpha, R6);
            ekf_cg_.update_feet(p_I, nullptr, alpha, R6);
            ekf_rot_.update_feet(p_I, R_I, alpha, R12);
            ekf_att_.update_feet(p_I, nullptr, alpha, R6);
        }

        const Eigen::Vector3d a_free_P = rz_mat(psi) * (acc_d + g_I * 9.81);
        out[0] = v_a;
        out[1] = compl_.step(a_free_P, a_ok ? &v_a : nullptr, dt);
        out[2] = ekf_c_.velocity_pelvis(psi, gyro_d);
        out[3] = ekf_cg_.velocity_pelvis(psi, gyro_d);
        out[4] = ekf_rot_.velocity_pelvis(psi, gyro_d);
        out[5] = jacobian_velocity(q_d, dq_d, w_P, grav_P);
        out[6] = ekf_att_.velocity_pelvis(psi, gyro_d);
    }

    // Explicit warm start, for the parity gate. Online the bank cannot do this on tick one
    // -- arm A needs a previous foot position -- but the offline harness can, because it
    // sees the whole session. Driving both from the SAME initial condition is what isolates
    // a porting error from that initialisation-convention difference, which the divergent
    // position-only arms amplify (see tests/test_estimator_parity.py).
    void prime(const Eigen::Vector3d& v_pelvis, double psi, const Eigen::Vector3d& gyro,
               const Eigen::Matrix3d& R_wt)
    {
        ekf_c_.reset(&R_wt); ekf_cg_.reset(&R_wt);
        ekf_att_.reset(&R_wt); ekf_rot_.reset(&R_wt);
        ekf_c_.warm_start(v_pelvis, psi, gyro);
        ekf_cg_.warm_start(v_pelvis, psi, gyro);
        ekf_att_.warm_start(v_pelvis, psi, gyro);
        ekf_rot_.warm_start(v_pelvis, psi, gyro);
        reset_done_ = true;
        primed_ = true;
    }

    // Re-prime across an FSM re-entry: the cached foot positions are from before the exit,
    // so the first difference would span the whole gap and read as an enormous velocity.
    void reprime()
    {
        a_prev_valid_ = false;
        key_valid_ = false;
        primed_ = false;
        reset_done_ = false;
        compl_.reset();
    }

private:
    BaseEkf<false> ekf_c_, ekf_cg_, ekf_att_;
    BaseEkf<true> ekf_rot_;
    ComplementaryFilter compl_{0.20};
    EkfNoise noise_{};
    MeasCfg meas_{};
    Eigen::Vector3d a_prev_[2];
    int a_prev_stance_{-1};
    bool a_prev_valid_{false}, key_valid_{false}, primed_{false}, reset_done_{false};
    double prev_key_[19] = {0};
};

} // namespace hrl
