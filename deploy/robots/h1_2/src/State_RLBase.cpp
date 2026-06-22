#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

// [SAFETY FILTER] set to 0 to revert to unfiltered policy output
#define SAFETY_FILTER 1
#if SAFETY_FILTER
#  include "h1_2_limits.h"
#  include <cmath>
#  include <algorithm>
#endif

namespace isaaclab
{
// Keyboard velocity commands — copied from deploy/robots/g1/src/State_RLBase.cpp.
// To activate: change "velocity_commands" → "keyboard_velocity_commands" in
// config/policy/velocity/v0/params/deploy.yaml (observations section).
//
// Key mapping (same as G1 and h1v2-Isaac convention):
//   w/s = forward/backward   a/d = strafe left/right   q/e = turn left/right
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    // Here you can change the input velocity of the keyboard
    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", { 1.0f,  0.0f,  0.0f}}, 
        {"s", {-1.0f,  0.0f,  0.0f}},
        {"a", { 0.0f,  1.0f,  0.0f}},
        {"d", { 0.0f, -1.0f,  0.0f}},
        {"q", { 0.0f,  0.0f,  1.0f}},
        {"e", { 0.0f,  0.0f, -1.0f}},
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    auto it = key_commands.find(key);
    if (it != key_commands.end()) cmd = it->second;
    return cmd;
}
} // namespace isaaclab

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // H1_2_DEPLOY_CFG selects sim (keyboard) vs real (gamepad) params.
    // Set automatically by setup_all_mujoco.sh / setup_all_robot.sh.
    const char* cfg_env = std::getenv("H1_2_DEPLOY_CFG");
    std::string deploy_cfg = cfg_env ? cfg_env : "deploy.yaml";
    spdlog::info("Using deploy config: {}", deploy_cfg);
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / deploy_cfg),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();

#if SAFETY_FILTER
    // IMU tilt check — uses snapshot already captured by pre_run(), no extra lock needed
    const auto& q = env->robot->data.root_quat_w;
    float pitch = std::asin(std::clamp(2.0f*(q.w()*q.y() - q.z()*q.x()), -1.0f, 1.0f));
    float roll  = std::atan2(2.0f*(q.w()*q.x() + q.y()*q.z()), 1.0f - 2.0f*(q.x()*q.x() + q.y()*q.y()));
    bool tilt_safety = (std::abs(pitch) > H1_2_TILT_LIMIT || std::abs(roll) > H1_2_TILT_LIMIT);
    if (tilt_safety)
        spdlog::warn("[Safety] Tilt: pitch={:.2f} roll={:.2f} rad", pitch, roll);

    // Joint limit check — any policy-controlled joint out of range triggers whole-robot hold
    bool joint_hold = false;
    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); i++) {
        int jid = (int)env->robot->data.joint_ids_map[i];
        float q_meas = env->robot->data.joint_pos[i];
        if (q_meas < h1_2_joint_limits[jid].min || q_meas > h1_2_joint_limits[jid].max) {
            joint_hold = true;
            spdlog::warn("[Safety] Hold: joint {} q={:.3f} out of [{:.3f}, {:.3f}]",
                jid, q_meas, h1_2_joint_limits[jid].min, h1_2_joint_limits[jid].max);
            break;
        }
    }

    // Vertical acceleration fall detection — catches pelvis sinking while body stays upright.
    // Read accelerometer under the lowstate mutex (3 floats, negligible lock time).
    Eigen::Vector3f lin_acc_b;
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        lin_acc_b = Eigen::Vector3f(
            lowstate->msg_.imu_state().accelerometer()[0],
            lowstate->msg_.imu_state().accelerometer()[1],
            lowstate->msg_.imu_state().accelerometer()[2]
        );
    }
    // Rotate to world frame and remove gravity: 0 = standing still, -9.81 = free-fall
    float a_world_z = (env->robot->data.root_quat_w * lin_acc_b).z() - 9.81f;
    // Count consecutive ticks below threshold; reset immediately when condition clears
    if (a_world_z < H1_2_FALL_ACC_THRESH) fall_acc_counter_ = std::min(fall_acc_counter_ + 1, H1_2_FALL_ACC_WINDOW);
    else                                  fall_acc_counter_ = std::max(fall_acc_counter_ - 1, 0);
    bool fall_detected = (fall_acc_counter_ >= H1_2_FALL_ACC_WINDOW);
    if (fall_detected)
        spdlog::warn("[Safety] Fall: a_world_z={:.2f} m/s²", a_world_z);

    // Ramp hold counter up/down over H1_2_RAMP_CYCLES ticks to avoid torque spikes
    bool hold_active = joint_hold || tilt_safety || fall_detected;
    if (hold_active) hold_counter_ = std::min(hold_counter_ + 1, H1_2_RAMP_CYCLES);
    else             hold_counter_ = std::max(hold_counter_ - 1, 0);
    float alpha = static_cast<float>(hold_counter_) / H1_2_RAMP_CYCLES;

    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); i++) {
        int jid = (int)env->robot->data.joint_ids_map[i];
        float q_meas = env->robot->data.joint_pos[i];
        // alpha=0: pure policy output  |  alpha=1: hold at current measured position
        float q_cmd = (1.0f - alpha) * action[i] + alpha * q_meas;
        lowcmd->msg_.motor_cmd()[jid].q() = q_cmd;
    }

#ifdef STATE_RLBASE_HAS_SAFETY_LOGGER
    // Flight recorder: logs raw policy intent (action) vs measured state + triggers.
    if (safety_logger_.enabled()) {
        float quat[4] = { q.w(), q.x(), q.y(), q.z() };
        float acc[3]  = { lin_acc_b.x(), lin_acc_b.y(), lin_acc_b.z() };
        safety_logger_.record(action,
                              env->robot->data.joint_pos.data(),
                              env->robot->data.joint_vel.data(),
                              quat, acc, alpha, joint_hold, tilt_safety, fall_detected);
    }
#endif
#else
    // Original unfiltered policy output
    for(int i(0); i < (int)env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[(int)env->robot->data.joint_ids_map[i]].q() = action[i];
    }
#endif
}