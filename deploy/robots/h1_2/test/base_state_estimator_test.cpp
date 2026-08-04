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

#include <cmath>
#include <cstdio>
#include <string>

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

int main()
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

    std::printf("%s (%d)\n", fails ? "FAILURES" : "ALL PASS", fails);
    return fails ? 1 : 0;
}
