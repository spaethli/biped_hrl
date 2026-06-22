#include "FSM/State_RLHRL.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <filesystem>
#include <stdexcept>

// [SAFETY FILTER] master switch lives in FSM/State_RLHRL.h (gates filter + logger).
#if SAFETY_FILTER
#  include "h1_2_limits.h"
#  include <cmath>
#  include <algorithm>
#endif

State_RLHRL::State_RLHRL(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // H1_2_DEPLOY_CFG selects sim (keyboard) vs real (gamepad) params (same as State_RLBase).
    const char* cfg_env = std::getenv("H1_2_DEPLOY_CFG");
    std::string deploy_cfg = cfg_env ? cfg_env : "deploy.yaml";
    spdlog::info("[HRL] Using deploy config: {}", deploy_cfg);
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / deploy_cfg),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );

    // ONNX live here; loaded lazily on first enter() (see ensure_models_loaded).
    exported_dir_ = policy_dir / "exported";

    // HRL structure from deploy.yaml (`hrl:` block) — switch checkpoints by editing this,
    // no recompile. Must match how the policy was trained.
    auto hrl = env->cfg["hrl"];
    c_ = hrl["c"].as<int>();
    hl_target_mode_ = hrl["hl_target_mode"].as<std::string>();
    // hl_algorithm: "oracle" -> analytic V* (command + nominal), no high_level.onnx;
    // anything else (default) -> learned HL from high_level.onnx. Default keeps old behavior.
    oracle_ = hrl["hl_algorithm"] && hrl["hl_algorithm"].as<std::string>() == "oracle";
    auto goal_components = hrl["goal_components"].as<std::vector<std::string>>();
    float nominal_h = hrl["nominal_root_height"].as<float>();
    goal_space_ = std::make_unique<hrl::GoalSpace>(goal_components, nominal_h);
    target_ = Eigen::VectorXf::Zero(goal_space_->dim());
    // Optional goal-state noise (default off) to emulate real-robot estimator noise in sim.
    state_noise_std_ = goal_space_->noise_std(hrl["state_noise"]);
    if (state_noise_std_.cwiseAbs().sum() > 0.0f) {
        std::string s;
        for (int i = 0; i < state_noise_std_.size(); ++i)
            s += std::to_string(state_noise_std_[i]) + " ";
        spdlog::warn("[HRL] goal-state noise ENABLED (per-dim std): [{}] — emulating "
                     "estimator noise; clear hrl.state_noise for clean runs", s);
    }

    // Ground-truth base velocity + height for the goal-space state (the LowState IMU has
    // neither). Provided by the MuJoCo sim bridge's SportModeState publisher
    // (rt/sportmodestate). SIM-ONLY privileged signal; a real robot needs a state estimator.
    highstate_ = std::make_shared<unitree::robot::go2::subscription::SportModeState>();
    spdlog::info("[HRL] hl_algorithm={} c={} hl_target_mode={} goal_dim={} (HighState=rt/sportmodestate)",
                 oracle_ ? "oracle" : "learned", c_, hl_target_mode_, goal_space_->dim());

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLHRL::ensure_models_loaded()
{
    if (ll_runner_) return;  // already loaded (entered before)

    // The LL ONNX is always needed. The HL ONNX is needed only for a learned HL; the oracle
    // computes V* analytically (command + nominal), so it requires no high_level.onnx.
    const std::filesystem::path hl_path = exported_dir_ / "high_level.onnx";
    const std::filesystem::path ll_path = exported_dir_ / "low_level.onnx";
    if (!std::filesystem::exists(ll_path)) {
        throw std::runtime_error("[HRL] missing ONNX '" + ll_path.string() +
                                 "'. The A1 deploy needs low_level.onnx in exported/.");
    }
    ll_runner_ = std::make_unique<isaaclab::OrtRunner>(ll_path.string());

    if (oracle_) {
        spdlog::info("[HRL] Oracle HL (analytic V*); loaded low_level.onnx from {}",
                     exported_dir_.string());
        return;
    }

    if (!std::filesystem::exists(hl_path)) {
        throw std::runtime_error("[HRL] missing ONNX '" + hl_path.string() +
                                 "'. A learned HL needs high_level.onnx in exported/ "
                                 "(or set hrl.hl_algorithm: oracle).");
    }
    hl_runner_ = std::make_unique<isaaclab::OrtRunner>(hl_path.string());

    // Sanity: the exported HL output dim must equal the configured goal dim.
    if ((int)hl_runner_->get_action().size() != goal_space_->dim()) {
        throw std::runtime_error(
            "[HRL] high_level.onnx output dim " +
            std::to_string(hl_runner_->get_action().size()) + " != goal_dim " +
            std::to_string(goal_space_->dim()) + " (goal_components/deploy.yaml mismatch).");
    }
    spdlog::info("[HRL] Loaded high_level.onnx + low_level.onnx from {}", exported_dir_.string());
}

void State_RLHRL::policy_step()
{
    env->robot->update();
    auto obs = env->observation_manager->compute();
    const auto& policy = obs.at("policy");    // proprio (command removed)
    const auto& command = obs.at("command");  // twist command (HL input only)

    // Goal-space state s: base lin vel (world->body via the IMU quat) + ang_vel/orientation
    // (IMU) + height (sim HighState). frame_vel is published in the WORLD frame; rotate it.
    const Eigen::Vector3f v_world = highstate_->velocity();
    const Eigen::Vector3f lin_vel_b = env->robot->data.root_quat_w.conjugate() * v_world;
    const float height = highstate_->position().z();
    Eigen::VectorXf s = goal_space_->state(env.get(), lin_vel_b, height);

    // Emulate real-robot state-estimator noise on the goal feedback (no-op if std all 0).
    if (state_noise_std_.size() == s.size()) {
        std::normal_distribution<float> ndist(0.0f, 1.0f);
        for (int i = 0; i < s.size(); ++i)
            if (state_noise_std_[i] > 0.0f) s[i] += state_noise_std_[i] * ndist(rng_);
    }

    // High level fires at the start of each window -> new absolute target V* (held for c steps).
    if (step_ % c_ == 0) {
        if (oracle_) {
            // Analytic target: commanded twist + nominal pose (no network).
            const Eigen::Vector3f cmd(command[0], command[1], command[2]);
            target_ = goal_space_->oracle_target(cmd);
        } else {
            std::vector<float> hl_in = policy;
            hl_in.insert(hl_in.end(), command.begin(), command.end());
            const auto g_vec = hl_runner_->act({{"obs", hl_in}});
            const Eigen::VectorXf g =
                Eigen::Map<const Eigen::VectorXf>(g_vec.data(), (int)g_vec.size());
            target_ = goal_space_->to_target(env.get(), s, g, hl_target_mode_);
        }
    }
    ++step_;

    // Low level observes the remaining delta V* - s and acts.
    const Eigen::VectorXf delta = target_ - s;
    std::vector<float> ll_in = policy;
    ll_in.insert(ll_in.end(), delta.data(), delta.data() + delta.size());
    const auto action = ll_runner_->act({{"obs", ll_in}});
    env->action_manager->process_action(action);
}

void State_RLHRL::run()
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
    Eigen::Vector3f lin_acc_b;
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        lin_acc_b = Eigen::Vector3f(
            lowstate->msg_.imu_state().accelerometer()[0],
            lowstate->msg_.imu_state().accelerometer()[1],
            lowstate->msg_.imu_state().accelerometer()[2]
        );
    }
    float a_world_z = (env->robot->data.root_quat_w * lin_acc_b).z() - 9.81f;
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

    if (safety_logger_.enabled()) {
        float quat[4] = { q.w(), q.x(), q.y(), q.z() };
        float acc[3]  = { lin_acc_b.x(), lin_acc_b.y(), lin_acc_b.z() };
        safety_logger_.record(action,
                              env->robot->data.joint_pos.data(),
                              env->robot->data.joint_vel.data(),
                              quat, acc, alpha, joint_hold, tilt_safety, fall_detected);
    }
#else
    for(int i(0); i < (int)env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[(int)env->robot->data.joint_ids_map[i]].q() = action[i];
    }
#endif
}
