// Regression test for defect 0 (dead gait clock under keyboard control), 2026-07-21.
// Drives the REAL ObservationManager over the REAL deploy yamls with a stub articulation
// (no DDS, no robot) and a real Keyboard reading a piped stdin. Not wired into CMake --
// the deploy tree has no test target; build + run it by hand:
//
//   cd deploy/robots/h1_2 && make -C build            # the lib this links against
//   g++ -std=c++17 test/phase_obs_test.cpp -o /tmp/phase_obs_test \
//     -Iinclude -I../../include -I/usr/include/eigen3 -I/opt/unitree_robotics/include \
//     -I/opt/unitree_robotics/include/ddscxx -I/opt/ros/humble/include/iceoryx/v2.0.5 \
//     -I../../thirdparty/onnxruntime-linux-x64-1.22.0/include \
//     -L/opt/unitree_robotics/lib -Lbuild -lh1_2_controller_lib -lunitree_sdk2 -lddsc \
//     -lddscxx -lrt -lpthread -lboost_program_options -lyaml-cpp -lfmt \
//     ../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
//   export LD_LIBRARY_PATH=/opt/unitree_robotics/lib:$PWD/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib
//   for k in w 0; do yes $k | tr -d '\n' | head -c 1000000 \
//       | /tmp/phase_obs_test config/policy; done   # both must end "ALL PASS (0)"
//
// `w` = a held command (clock must be alive), `0` = stop (clock must stay stand-masked).
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "h1_2_observations.h"
#include <iostream>
#include <cmath>

std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

static int fails = 0;
static void check(bool ok, const std::string& what)
{
    std::cout << (ok ? "  PASS  " : "  FAIL  ") << what << "\n";
    if (!ok) fails++;
}

struct StubArticulation : isaaclab::Articulation
{
    void update() override {}
};

static std::unique_ptr<isaaclab::ManagerBasedRLEnv> make_env(const std::string& yaml,
                                                             unitree::common::UnitreeJoystick* js)
{
    auto robot = std::make_shared<StubArticulation>();
    robot->data.joystick = js;
    robot->data.root_quat_w = Eigen::Quaternionf::Identity();
    robot->data.root_ang_vel_b.setZero();
    robot->data.projected_gravity_b = Eigen::Vector3f(0, 0, -1);
    return std::make_unique<isaaclab::ManagerBasedRLEnv>(YAML::LoadFile(yaml), robot);
}

// Fraction of N obs computes where the phase term came out non-zero (analyzer's phase_alive).
static float alive_frac(isaaclab::ManagerBasedRLEnv* env, const char* group, int n)
{
    int live = 0;
    for (int i = 0; i < n; i++) {
        env->observation_manager->compute_group(group);
        if (isaaclab::g_gait_phase_obs[0] != 0.0f || isaaclab::g_gait_phase_obs[1] != 0.0f) live++;
    }
    return (float)live / n;
}

int main(int argc, char** argv)
{
    std::cout << std::unitbuf;  // abort-safe output
    const std::string A0  = std::string(argv[1]) + "/velocity/v0/params/";
    const std::string A1  = std::string(argv[1]) + "/velocity_hrl/v0/params/";

    // Keyboard first: keyboard_velocity_commands derefs FSMState::keyboard, and the
    // ObservationManager ctor evaluates every term once. stdin is a pipe fed with 'w'.
    FSMState::keyboard = std::make_shared<Keyboard>();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const std::string key = FSMState::keyboard->key();
    std::cout << "  (keyboard reads '" << key << "')\n";
    const float want = (key == "w") ? 1.0f : 0.0f;  // "0" (stop) -> stand-masked
    check(key == "w" || key == "0", "test harness: a key is being read from stdin");

    // --- 1. command-source detection over all four live configs -------------------------
    unitree::common::UnitreeJoystick js;
    {
        auto e = make_env(A0 + "deploy.yaml", &js);
        check(isaaclab::uses_keyboard_commands(e.get()), "A0 deploy.yaml -> keyboard (flat yaml)");
        auto e2 = make_env(A0 + "deploy_real.yaml", &js);
        check(!isaaclab::uses_keyboard_commands(e2.get()), "A0 deploy_real.yaml -> joystick");
        auto e3 = make_env(A1 + "deploy.yaml", &js);
        check(isaaclab::uses_keyboard_commands(e3.get()), "A1 deploy.yaml -> keyboard (nested group)");
        auto e4 = make_env(A1 + "deploy_real.yaml", &js);
        check(!isaaclab::uses_keyboard_commands(e4.get()), "A1 deploy_real.yaml -> joystick");
    }

    // --- 2. joystick path (real robot) unchanged ----------------------------------------
    {
        js.lx(0); js.ly(0); js.rx(0);
        auto e = make_env(A0 + "deploy_real.yaml", &js);
        check(alive_frac(e.get(), "obs", 50) == 0.0f, "joystick neutral -> phase stand-masked");
        // Axis is a smoothed input (smooth=0.03 per update); ramp it like a real stick.
        for (int i = 0; i < 400; i++) js.ly(1.0f);
        check(alive_frac(e.get(), "obs", 50) == 1.0f, "joystick fwd -> phase alive");
        // and it actually advances (not a frozen constant)
        e->observation_manager->compute_group("obs");
        float s0 = isaaclab::g_gait_phase_obs[0];
        e->observation_manager->compute_group("obs");
        check(isaaclab::g_gait_phase_obs[0] != s0, "phase advances between steps");
        for (int i = 0; i < 400; i++) js.ly(0.0f);  // ramp back to neutral
    }

    // --- 3. THE BUG: keyboard config, neutral joystick, key held -------------------------
    // stdin is a pipe fed with the held key by the caller.
    {
        auto e = make_env(A0 + "deploy.yaml", &js);           // keyboard config
        check(alive_frac(e.get(), "obs", 50) == want,
              "A0 keyboard '" + key + "' + neutral joystick -> phase alive==" + std::to_string(want));
        auto e1 = make_env(A1 + "deploy.yaml", &js);          // keyboard config, nested
        check(alive_frac(e1.get(), "policy", 50) == want,
              "A1 keyboard '" + key + "' + neutral joystick -> phase alive==" + std::to_string(want));

        // negative control: the untouched shared term still reproduces the bug
        auto e2 = make_env(A0 + "deploy.yaml.w1_legacy_test", &js);
        int live = 0;
        for (int i = 0; i < 50; i++) {
            auto o = isaaclab::mdp::gait_phase(e2.get(), YAML::Load("{period: 0.6}"));
            if (o[0] != 0.0f || o[1] != 0.0f) live++;
        }
        check(live == 0, "control: shared isaaclab::mdp::gait_phase still dead (bug reproduced)");
    }

    // --- 4. stand-mask preserved: no key -> zero -----------------------------------------
    std::cout << (fails ? "\nFAILURES: " : "\nALL PASS (") << fails << (fails ? "\n" : ")\n");
    std::cout.flush();
    _exit(fails ? 1 : 0);  // Keyboard's read thread is never joined; skip static dtors
}
