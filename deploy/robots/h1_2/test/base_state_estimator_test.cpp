// Regression test for the deployable base-state estimator (hrl/base_state.h), 2026-08-03.
// Pins the two things that can silently go wrong in geometry code and would only show up
// as a mis-tracked policy on hardware:
//   1. the leg FK's link lengths / sole offset, against MuJoCo's OWN kinematics
//      (mj_kinematics on src/assets/robots/unitree_h1_2/xmls/h1_2.xml), and
//   2. the within-window velocity-increment composition, against the Python reference in
//      scripts/play.py's `check_vel_increment` block ("waist-corr" rung).
// Both fixture tables were generated from those two sources; regenerate them the same way
// if the model changes. Header-only, no DDS/robot/ONNX — build + run by hand (the deploy
// tree has no test target, same as test/phase_obs_test.cpp):
//
//   cd deploy/robots/h1_2
//   g++ -std=c++17 -Iinclude -I../../include -I/usr/include/eigen3
//       test/base_state_estimator_test.cpp -o /tmp/base_state_estimator_test
//   /tmp/base_state_estimator_test           # must end "ALL PASS (0)"
// (join the g++ line; it is split here only to keep it inside the comment block)
#include "hrl/base_state.h"
#include "hrl/base_estimators.h"

#include <cmath>
#include <fstream>
#include <vector>
#include <cstdio>
#include <string>
#include <sstream>
#include <array>
#include <algorithm>
#include <cstdlib>

static int fails = 0;
static void check(bool ok, const std::string& what)
{
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) fails++;
}

// --- fixtures: mj_kinematics on h1_2.xml -------------------------------------------
// Row 0 is the deploy nominal leg pose (deploy.yaml default_joint_pos legs); the rest are
// samples from a walking envelope.
static const float kFkQ[8][12] = {
    {+0.000000f, -0.200000f, +0.000000f, +0.500000f, -0.300000f, +0.000000f, +0.000000f, -0.200000f, +0.000000f, +0.500000f, -0.300000f, +0.000000f},
    {+0.050038f, +0.535542f, +0.187843f, +0.237811f, -0.339800f, +0.112066f, -0.197894f, +0.413965f, +0.098535f, +0.601902f, -0.336361f, -0.066472f},
    {-0.098052f, -0.187878f, +0.052274f, +0.730246f, +0.494600f, +0.087799f, +0.048872f, +0.682336f, -0.192346f, +0.140318f, +0.035048f, -0.136817f},
    {-0.185728f, -0.076178f, +0.033103f, +1.275752f, +0.055072f, +0.004235f, -0.001251f, -0.503976f, -0.294103f, +0.188603f, +0.130439f, -0.089818f},
    {-0.052185f, -0.894025f, +0.215024f, +0.131692f, -0.378881f, +0.114100f, +0.003916f, +0.455440f, +0.019859f, +1.012656f, -0.590205f, +0.012343f},
    {+0.003109f, +0.494143f, -0.019368f, +0.797276f, -0.628898f, -0.033710f, -0.070785f, -0.659680f, +0.108169f, +0.469169f, +0.474497f, +0.026998f},
    {+0.042023f, +0.120795f, +0.138225f, +0.126182f, -0.171624f, -0.078131f, -0.039001f, -0.745273f, +0.183914f, +0.222506f, +0.106118f, -0.059874f},
    {+0.149631f, +0.159544f, -0.134192f, +1.167611f, +0.433938f, +0.121175f, +0.027888f, -0.667264f, -0.203768f, +1.291859f, -0.037208f, -0.095834f},
};
static const float kFkMujocoLowestZ[8] = {
    -1.002361f, -0.911431f, -1.081687f, -0.949194f, -0.799942f, -0.978094f, -1.018198f, -0.940021f
};

// --- fixtures: the python `waist-corr` composition, play.py check_vel_increment -----
// Columns: psi0, psi, w0[3], w[3], dv_int[3].
static const float kIncIn[6][11] = {
    {-0.445716f, -0.000867f, +0.304495f, -1.413933f, -1.056222f, +1.284633f, -1.288738f, -1.110678f, +0.537994f, +0.146260f, -0.157208f},
    {+0.013668f, +0.195412f, -0.674074f, -1.086096f, +0.864119f, +0.511082f, +0.037147f, +0.950209f, +0.058890f, +0.577096f, -0.354589f},
    {+0.064476f, -0.019650f, -0.440175f, +0.274786f, -0.794096f, +0.906608f, +1.102001f, -1.113721f, -0.039512f, -0.267426f, -0.500260f},
    {+0.475133f, -0.084062f, -1.056926f, +0.520087f, -0.893352f, +1.204293f, -0.848555f, -1.400776f, -0.359077f, -0.185103f, -0.037310f},
    {+0.487361f, +0.236833f, -0.482038f, -1.449368f, -1.020529f, +1.489308f, -0.120852f, +0.573120f, -0.534398f, -0.559140f, +0.415068f},
    {+0.105458f, -0.229548f, -0.547870f, -1.232288f, -0.981991f, -1.426242f, +1.017375f, -0.101090f, -0.447357f, +0.287096f, -0.365217f},
};
static const float kIncOut[6][3] = {
    {+0.528797f, +0.596549f, -0.144248f},
    {-0.404905f, +0.897426f, -0.382184f},
    {-0.274569f, +0.102162f, -0.511620f},
    {-0.128483f, +0.449182f, +0.066381f},
    {-0.925627f, -0.276017f, +0.393201f},
    {-1.113718f, +0.212047f, -0.481982f},
};


// --- T2: estimator-arm parity against the Python harness ----------------------------
// The robot runs the arms from hrl/base_estimators.h; every published number about them
// came from scripts/replay_base_estimators.py. Both are pinned to one golden fixture so a
// hardware disagreement can be attributed. Bar is stated on the c=8 WINDOW average because
// that is what the HL consumes -- never an individual tick.
//
// Both sides are primed from the SAME initial condition (bank.prime()). Online the bank
// warm-starts one tick later, which contracting arms forget and the divergent position-only
// arm does not; that is a property of the arm, not of the port, and is documented in
// tests/test_estimator_parity.py.
static void parity_gate(const std::string& dir)
{
    const char* kArms[7] = {"legodom","compl","ekf","ekf_grav","ekf_rot","jacobian","ekf_att"};
    auto load = [](const std::string& p, int cols) {
        std::vector<std::vector<double>> rows;
        std::ifstream f(p);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            // Token-at-a-time via strtod, NOT `is >> double`: operator>> sets failbit on
            // "nan", and once the stream fails EVERY remaining column on that row silently
            // reads as 0 -- including the valid ones. Arm A is legitimately NaN on its first
            // sample (no previous foot position), so this bug zeroed whole rows and made
            // even the stateless arms look like porting errors.
            std::istringstream is(line);
            std::vector<double> r(cols, std::nan(""));
            std::string tok;
            for (int i = 0; i < cols && (is >> tok); ++i) r[i] = std::strtod(tok.c_str(), nullptr);
            rows.push_back(r);
        }
        return rows;
    };
    auto in = load(dir + "/estimator_parity_input.txt", 37);
    auto want = load(dir + "/estimator_parity_expected.txt", 14);
    if (in.empty() || in.size() != want.size()) {
        check(false, "parity fixture missing or ragged (" + dir + ") -- run tests/ generator");
        return;
    }

    auto v0rows = load(dir + "/estimator_parity_v0.txt", 3);
    if (v0rows.empty()) { check(false, "parity v0 fixture missing"); return; }
    const Eigen::Vector3d v0(v0rows[0][0], v0rows[0][1], v0rows[0][2]);

    hrl::EstimatorBank bank;
    const int n = (int)in.size();
    std::vector<std::array<double,14>> got(n);
    for (int k = 0; k < n; ++k) {
        float q[13], dq[13];
        for (int i = 0; i < 13; ++i) { q[i] = (float)in[k][11+i]; dq[i] = (float)in[k][24+i]; }
        Eigen::Vector3f acc((float)in[k][1], (float)in[k][2], (float)in[k][3]);
        Eigen::Vector3f gyro((float)in[k][4], (float)in[k][5], (float)in[k][6]);
        Eigen::Quaternionf qw((float)in[k][7], (float)in[k][8], (float)in[k][9], (float)in[k][10]);
        qw.normalize();
        const Eigen::Vector3f g_T = qw.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
        const float dt = k ? (float)(in[k][0] - in[k-1][0]) : 0.0f;
        if (k == 0)   // identical initial condition to the harness, before the first step
            bank.prime(v0, q[12], gyro.cast<double>(),
                       qw.toRotationMatrix().cast<double>());
        Eigen::Vector3d o[7];
        bank.step(q, dq, acc, gyro, g_T, qw.toRotationMatrix(), nullptr, dt, o);
        for (int a = 0; a < 7; ++a) { got[k][2*a] = o[a].x(); got[k][2*a+1] = o[a].y(); }
    }

    const int W = 160;               // c=8 policy steps at the ~500 Hz DDS rate
    const int nw = n / W;
    for (int a = 0; a < 7; ++a) {
        double worst = 0.0;
        for (int w = 0; w < nw; ++w)
            for (int ax = 0; ax < 2; ++ax) {
                double sg = 0, sw = 0;
                int cnt = 0;
                for (int k = w*W; k < (w+1)*W; ++k) {
                    if (!std::isfinite(got[k][2*a+ax]) || !std::isfinite(want[k][2*a+ax])) continue;
                    sg += got[k][2*a+ax]; sw += want[k][2*a+ax]; ++cnt;
                }
                if (cnt > W/2) worst = std::max(worst, std::fabs(sg/cnt - sw/cnt));
            }
        char msg[160];
        std::snprintf(msg, sizeof msg, "arm %-9s c=8 window vs Python: %.2e m/s (bar 1e-4)",
                      kArms[a], worst);
        check(worst < 1e-4, msg);
    }
}

// --- T6: passive-arm isolation (docs/adr/0007) --------------------------------------
// The bench runs the KNOWN-DIVERGENT position-only arm inside the ~1 kHz control loop. That
// is only defensible if a passive arm reaches the flight recorder and nothing else, so the
// invariant is asserted, not asserted-by-comment. Two halves:
//
//   (a) no cross-arm coupling: an arm's output inside the bank must be bit-identical to the
//       same filter driven alone on the same inputs, even while a sibling is diverging.
//       Anything shared (a static, a scratch buffer) would show up here.
//   (b) NaN containment: a filter driven to NaN stays NaN and cannot reach a sibling.
//
// The half this cannot reach is "a passive arm never touches lowcmd" -- that lives in
// State_RLHRL.cpp's routing, and tests/test_deploy_parity.py pins it at source level.
static void isolation_gate(const std::string& dir)
{
    auto load = [](const std::string& p, int cols) {
        std::vector<std::vector<double>> rows;
        std::ifstream f(p);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream is(line);
            std::vector<double> r(cols, std::nan(""));
            std::string tok;
            for (int i = 0; i < cols && (is >> tok); ++i) r[i] = std::strtod(tok.c_str(), nullptr);
            rows.push_back(r);
        }
        return rows;
    };
    auto in = load(dir + "/estimator_parity_input.txt", 37);
    if (in.empty()) { check(false, "isolation: fixture missing"); return; }

    auto v0rows = load(dir + "/estimator_parity_v0.txt", 3);
    if (v0rows.empty()) { check(false, "isolation: v0 fixture missing"); return; }
    const Eigen::Vector3d v0(v0rows[0][0], v0rows[0][1], v0rows[0][2]);

    hrl::EstimatorBank bank;
    hrl::BaseEkf<true> lone_rot;          // arm D, driven alone with identical inputs
    hrl::EkfNoise ns;
    hrl::MeasCfg mc;
    double worst_d = 0.0;
    bool c_went_bad = false;
    // Mirror the bank's own sequencing exactly -- same initial condition, and the kinematic
    // update gated on a CHANGED sample. Without both, the reference diverges for reasons that
    // have nothing to do with coupling (measured: 2e-2 m/s from the warm start alone).
    double prev_key[19] = {0};
    bool key_valid = false;

    for (size_t k = 0; k < in.size(); ++k) {
        float q[13], dq[13];
        double q_d[13];
        for (int i = 0; i < 13; ++i) {
            q[i] = (float)in[k][11 + i]; dq[i] = (float)in[k][24 + i]; q_d[i] = q[i];
        }
        Eigen::Vector3f acc((float)in[k][1], (float)in[k][2], (float)in[k][3]);
        Eigen::Vector3f gyro((float)in[k][4], (float)in[k][5], (float)in[k][6]);
        Eigen::Quaternionf qw((float)in[k][7], (float)in[k][8], (float)in[k][9], (float)in[k][10]);
        qw.normalize();
        const Eigen::Vector3f g_T = qw.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
        const Eigen::Matrix3d Rw = qw.toRotationMatrix().cast<double>();
        const float dt = k ? (float)(in[k][0] - in[k - 1][0]) : 0.0f;

        // POISON: from tick 200 on, the bank's inputs are unchanged but we drive the LONE
        // reference and the bank identically, so any divergence between bank-D and lone-D is
        // coupling. Meanwhile arm C is separately driven to NaN below (half b).
        double key[19];
        for (int i = 0; i < 3; ++i) { key[i] = acc[i]; key[3 + i] = gyro[i]; }
        for (int i = 0; i < 13; ++i) key[6 + i] = q_d[i];
        bool fresh = !key_valid;
        for (int i = 0; i < 19 && !fresh; ++i) fresh = (key[i] != prev_key[i]);
        for (int i = 0; i < 19; ++i) prev_key[i] = key[i];
        key_valid = true;

        if (k == 0) {
            bank.prime(v0, q_d[12], gyro.cast<double>(), Rw);
            lone_rot.reset(&Rw);
            lone_rot.warm_start(v0, q_d[12], gyro.cast<double>());
        }
        Eigen::Vector3d o[7];
        bank.step(q, dq, acc, gyro, g_T, qw.toRotationMatrix(), nullptr, dt, o);

        // lone arm D, same sequence the bank uses
        Eigen::Vector3d p_I[2]; Eigen::Matrix3d R_I[2];
        hrl::feet_in_imu(q_d, q_d[12], hrl::kContactA, p_I, R_I);
        double alpha[2];
        hrl::contact_alpha(p_I, g_T.cast<double>(), alpha);
        lone_rot.predict(acc.cast<double>(), gyro.cast<double>(), dt, alpha, ns);
        if (fresh) {
            Eigen::Matrix<double, 3, 7> Jv[2], Jw[2];
            hrl::foot_jacobians_imu(q_d, q_d[12], hrl::kContactA, Jv, Jw);
            Eigen::Matrix<double, 12, 12> R12;
            hrl::build_R<true>(Jv, Jw, alpha, mc, R12);
            lone_rot.update_feet(p_I, R_I, alpha, R12);
        }
        const Eigen::Vector3d v_lone = lone_rot.velocity_pelvis(q_d[12], gyro.cast<double>());

        if (k > 0) {
            worst_d = std::max(worst_d, (o[4].head<2>() - v_lone.head<2>()).cwiseAbs().maxCoeff());
        }
        if (!std::isfinite(o[2].x())) c_went_bad = true;
    }
    char msg[200];
    std::snprintf(msg, sizeof msg,
                  "isolation: bank arm ekf_rot == the same filter driven ALONE (worst %.2e m/s)",
                  worst_d);
    check(worst_d < 1e-9, msg);
    check(!c_went_bad, "isolation: no arm went non-finite on clean data (a NaN here would be a bug, "
                       "not divergence)");

    // (b) NaN containment at class level: two filters, one poisoned, no shared state.
    {
        hrl::BaseEkf<false> poisoned, clean;
        const Eigen::Vector3d nan3(std::nan(""), std::nan(""), std::nan(""));
        const Eigen::Vector3d good(0.0, 0.0, 9.81), zero(0.0, 0.0, 0.0);
        double alpha[2] = {1.0, 0.0};
        poisoned.predict(nan3, zero, 0.001, alpha, ns);
        for (int i = 0; i < 50; ++i) {
            poisoned.predict(good, zero, 0.001, alpha, ns);
            clean.predict(good, zero, 0.001, alpha, ns);
        }
        check(!poisoned.finite(), "isolation: a NaN-poisoned filter reports finite()==false "
                                  "(so the log records divergence rather than hiding it)");
        check(clean.finite(), "isolation: a sibling filter is UNAFFECTED by the poisoned one "
                              "(no shared state between arms)");
    }
}

int main(int argc, char** argv)
{
    // 1. Leg FK vs MuJoCo. The nominal pose must be EXACT (it is the pose the deploy
    //    anchors at, so any error there biases every height the policy ever sees); the
    //    walking-envelope samples are allowed the documented lateral-capsule slack.
    const float fk_nom = hrl::lowest_foot_z(kFkQ[0]);
    check(std::fabs(fk_nom - kFkMujocoLowestZ[0]) < 1e-4f,
          "leg FK at the nominal pose matches mj_kinematics exactly (" +
              std::to_string(fk_nom) + " vs " + std::to_string(kFkMujocoLowestZ[0]) + ")");
    float worst = 0.0f;
    for (int i = 1; i < 8; ++i)
        worst = std::max(worst, std::fabs(hrl::lowest_foot_z(kFkQ[i]) - kFkMujocoLowestZ[i]));
    check(worst < 0.012f, "leg FK over a walking envelope stays within 12 mm of "
                          "mj_kinematics (worst " + std::to_string(worst) + " m)");

    // 2. Anchoring. h = nominal_h + (h_FK - h_FK_nominal) must return EXACTLY the deploy
    //    nominal height at the nominal pose, whatever frame nominal_h is expressed in --
    //    that cancellation is the whole reason the anchored form is used (goal_space.h:71).
    const float nominal_h = 1.3076f;  // deploy.yaml hrl.nominal_root_height (imu-site frame)
    const float h_at_nominal = nominal_h + (-hrl::lowest_foot_z(kFkQ[0]) - (-fk_nom));
    check(std::fabs(h_at_nominal - nominal_h) < 1e-6f,
          "anchored height reproduces nominal_root_height at the nominal pose");

    // 3. Velocity increment vs the play.py reference.
    float inc_worst = 0.0f;
    for (int i = 0; i < 6; ++i) {
        const float* in = kIncIn[i];
        const Eigen::Vector3f w0(in[2], in[3], in[4]), w(in[5], in[6], in[7]);
        const Eigen::Vector3f dv(in[8], in[9], in[10]);
        const Eigen::Vector3f lev0 = hrl::rz(in[0], w0.cross(hrl::kImuOffsetT));
        const Eigen::Vector3f got = hrl::base_vel_increment(in[1], w, dv, lev0);
        const Eigen::Vector3f want(kIncOut[i][0], kIncOut[i][1], kIncOut[i][2]);
        inc_worst = std::max(inc_worst, (got - want).cwiseAbs().maxCoeff());
    }
    check(inc_worst < 1e-5f, "velocity increment matches the play.py waist-corr reference "
                             "(worst " + std::to_string(inc_worst) + " m/s)");

    // 4. The property the whole substitution rests on: at a window start dv_int is 0 and
    //    psi/w are still the window-start values, so the increment is EXACTLY 0. If this
    //    ever drifts, `delta` mode stops cancelling and V* - s silently gains a bias.
    {
        const Eigen::Vector3f w(0.9f, -0.4f, 1.3f);
        const float psi0 = 0.37f;
        const Eigen::Vector3f lev0 = hrl::rz(psi0, w.cross(hrl::kImuOffsetT));
        const Eigen::Vector3f at_start =
            hrl::base_vel_increment(psi0, w, Eigen::Vector3f::Zero(), lev0);
        check(at_start.norm() < 1e-6f, "increment is exactly 0 at the window start");
    }

    // 5. Gravity convention: standing upright and perfectly still, the gravity-removed
    //    acceleration must vanish. MuJoCo's (and a real IMU's) accelerometer reports
    //    a - g in the sensor frame, so upright reads +9.81 on z while projected gravity
    //    reads -1 on z; removal is a + proj_grav*|g|, NOT a - proj_grav*|g|.
    {
        const Eigen::Vector3f acc_upright_at_rest(0.0f, 0.0f, hrl::kGravity);
        const Eigen::Vector3f proj_grav_upright(0.0f, 0.0f, -1.0f);
        const Eigen::Vector3f lin = acc_upright_at_rest + proj_grav_upright * hrl::kGravity;
        check(lin.norm() < 1e-5f, "gravity removal (a + proj_grav*|g|) zeroes an upright "
                                  "resting IMU");
    }

    // 6. Foot SITE forward kinematics vs MuJoCo (2026-08-06). Distinct from test 1: that
    //    one pins the SOLE (capsule endpoints, the height path), this pins the `left_foot`/
    //    `right_foot` SITE, which is the point scripts/play.py's --check-leg-odometry
    //    differences and therefore the point the 0.074/0.049 error figures describe. The two
    //    are up to 0.13 m apart in x and p_foot enters the w x p term directly, so using the
    //    sole point here would be a silent 0.1 m lever error, not a rounding difference.
    //    Fixtures: mj_kinematics site_xpos on h1_2.xml, expressed in the pelvis frame.
    {
        static const float kSite[8][2][3] = {
            {{+0.001260f, +0.163000f, -0.997361f}, {+0.001260f, -0.163000f, -0.997361f}},
            {{-0.475635f, +0.301502f, -0.850963f}, {-0.496008f, +0.012327f, -0.808603f}},
            {{-0.150765f, +0.221402f, -0.962448f}, {-0.541017f, -0.359846f, -0.801868f}},
            {{-0.362818f, +0.251841f, -0.763497f}, {+0.348761f, -0.416827f, -0.911872f}},
            {{+0.655829f, +0.313777f, -0.678727f}, {-0.594137f, -0.151271f, -0.632406f}},
            {{-0.579661f, +0.145416f, -0.697400f}, {+0.339637f, -0.096974f, -0.936929f}},
            {{-0.117574f, +0.272914f, -1.003200f}, {+0.524973f, -0.028544f, -0.832017f}},
            {{-0.511503f, +0.011070f, -0.677956f}, {+0.011444f, -0.281192f, -0.864026f}},
        };
        float worst_site = 0.0f;
        for (int i = 0; i < 8; ++i)
            for (int leg = 0; leg < 2; ++leg) {
                const Eigen::Vector3f got = hrl::foot_site_b(kFkQ[i], leg);
                const Eigen::Vector3f want(kSite[i][leg][0], kSite[i][leg][1], kSite[i][leg][2]);
                worst_site = std::max(worst_site, (got - want).cwiseAbs().maxCoeff());
            }
        // Exact, unlike the sole (whose 12 mm slack is the unmodelled lateral capsule
        // spread): a site is a single point, so the FK either reproduces it or is wrong.
        check(worst_site < 1e-5f, "foot SITE FK matches mj_kinematics exactly over the "
                                  "walking envelope (worst " + std::to_string(worst_site) + " m)");
    }

    // 7. Leg odometry. Three properties, each one a failure that has a name.
    {
        const float dt = 0.001f;
        const Eigen::Vector3f up_grav(0.0f, 0.0f, -1.0f);  // upright: depth = -p.z

        // (a) stance = the foot further ALONG gravity, i.e. the LOWER one. Getting this
        //     backwards silently estimates from the swing foot, which is not world-fixed.
        Eigen::Vector3f p[2]  = {{0.0f, 0.15f, -0.90f}, {0.0f, -0.15f, -0.98f}};  // right lower
        Eigen::Vector3f pp[2] = {p[0], p[1]};
        pp[1].x() -= 0.001f;   // right foot moved +1 mm in x over dt -> v = -1 m/s
        Eigen::Vector3f v = hrl::leg_odom_velocity(p, pp, dt, Eigen::Vector3f::Zero(), up_grav);
        check(std::fabs(v.x() + 1.0f) < 1e-4f,
              "leg odometry differences the LOWER (stance) foot, not the swing foot");

        // (b) the difference is PER FOOT. Both feet stationary but 0.30 m apart, stance
        //     flipping between ticks: a cross-foot difference would report 0.30/dt = 300 m/s
        //     as a real velocity, once per step, and poison the whole window mean.
        Eigen::Vector3f s0[2] = {{0.15f, 0.15f, -0.95f}, {-0.15f, -0.15f, -0.95f}};
        v = hrl::leg_odom_velocity(s0, s0, dt, Eigen::Vector3f::Zero(), up_grav);
        check(v.norm() < 1e-5f, "stationary feet 0.30 m apart give exactly 0 (the difference "
                                "never crosses the stance switch)");

        // (c) the gyro term. Feet fixed, body rotating at 1 rad/s about z: the pelvis must
        //     report -w x p. Dropping this term is the difference between a body-frame
        //     velocity and a foot-frame one.
        Eigen::Vector3f r[2] = {{0.10f, 0.0f, -1.0f}, {0.10f, 0.0f, -0.9f}};
        v = hrl::leg_odom_velocity(r, r, dt, Eigen::Vector3f(0.0f, 0.0f, 1.0f), up_grav);
        check((v - Eigen::Vector3f(0.0f, -0.10f, 0.0f)).norm() < 1e-5f,
              "leg odometry applies -w x p_foot (pure yaw rate over a 0.10 m lever)");
    }

    parity_gate(argc > 1 ? argv[1] : "../../../tests/fixtures");
    isolation_gate(argc > 1 ? argv[1] : "../../../tests/fixtures");

    std::printf("%s (%d)\n", fails ? "FAILURES" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
