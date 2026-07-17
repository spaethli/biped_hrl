#include "FSM/State_RLHRL.h"
#include <algorithm>
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

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

    // Keeper structure (2026-07-14): velocity-goals-only HL, HL lin-vel input, HL-owned
    // cadence. All default off -> pre-velgoal checkpoints run byte-identical.
    hl_obs_vel_ = hrl["hl_obs_vel"] && hrl["hl_obs_vel"].as<bool>();
    velgoal_ = hrl["hl_velocity_goals_only"] && hrl["hl_velocity_goals_only"].as<bool>();
    const bool cadence = hrl["hl_cadence"] && hrl["hl_cadence"].as<bool>();
    const std::string cad_src =
        hrl["hl_cadence_source"] ? hrl["hl_cadence_source"].as<std::string>() : "hl";
    cadence_dim_ = (cadence && cad_src == "hl" && !oracle_) ? 1 : 0;
    if (cadence) {
        auto r = hrl["cadence_period_range"];
        period_lo_ = r[0].as<float>();
        period_hi_ = r[1].as<float>();
    }
    pin_period_ = hrl["pin_period"] ? hrl["pin_period"].as<float>() : 0.0f;
    if (hl_obs_vel_ || velgoal_) goal_space_->task_dim();  // enforce velocity-first prefix
    // The LL entrains to its gait_phase obs clock. The C++ term integrates
    // global_phase += dt/period from its YAML params node, which aliases env->cfg
    // (yaml-cpp handles share storage) — so writing this key retunes the clock
    // phase-continuously, exactly like training's hrl_period buffer. Pin now if asked;
    // the HL rewrites it per window unless pinned (see policy_step).
    if (pin_period_ > 0.0f)
        env->cfg["observations"]["policy"]["gait_phase"]["params"]["period"] = pin_period_;
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
    spdlog::info("[HRL] hl_algorithm={} c={} hl_target_mode={} goal_dim={} velgoal={} "
                 "hl_obs_vel={} cadence_dim={} period_range=[{},{}] pin_period={} "
                 "(HighState=rt/sportmodestate)",
                 oracle_ ? "oracle" : "learned", c_, hl_target_mode_, goal_space_->dim(),
                 velgoal_, hl_obs_vel_, cadence_dim_, period_lo_, period_hi_, pin_period_);

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

// One-time probe of an ONNX file's flattened input dim. OrtRunner::act builds its input
// tensor from the MODEL's size, so feeding a short obs vector reads out of bounds
// silently — this hard check at load is the only place a wiring mismatch fails loudly.
static int64_t onnx_input_dim(const std::filesystem::path& path)
{
    Ort::Env ort_env(ORT_LOGGING_LEVEL_ERROR, "hrl_dim_probe");
    Ort::SessionOptions so;
    Ort::Session session(ort_env, path.c_str(), so);
    auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    int64_t n = 1;
    for (auto d : shape) n *= d;  // [1, obs_dim]
    return n;
}

// One-time read of a comma-separated float list from an ONNX file's custom metadata (written
// by HierarchicalRunner._attach_hrl_metadata). Empty = key absent (pre-2026-07-16 exports).
// Robot-local, same throwaway-session pattern as onnx_input_dim: OrtRunner lives in the
// shared isaaclab header and keeps its Ort::Session private, so we open our own at load. Load
// path only — never the control loop.
static std::vector<float> onnx_metadata_floats(const std::filesystem::path& path, const char* key)
{
    Ort::Env ort_env(ORT_LOGGING_LEVEL_ERROR, "hrl_meta_probe");
    Ort::SessionOptions so;
    Ort::Session session(ort_env, path.c_str(), so);
    Ort::AllocatorWithDefaultOptions alloc;
    auto val = session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, alloc);
    std::vector<float> out;
    if (!val) return out;  // absent
    std::stringstream ss(val.get());
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stof(tok));
    }
    return out;
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

    // Input-dim guards (see onnx_input_dim). Obs sizes from a live compute.
    const auto obs = env->observation_manager->compute();
    const int policy_dim = (int)obs.at("policy").size();
    const int command_dim = (int)obs.at("command").size();
    const int64_t ll_in = onnx_input_dim(ll_path);
    if (ll_in != policy_dim + goal_space_->dim()) {
        throw std::runtime_error(
            "[HRL] low_level.onnx input dim " + std::to_string(ll_in) + " != policy(" +
            std::to_string(policy_dim) + ") + goal_dim(" +
            std::to_string(goal_space_->dim()) + ") — obs/goal_components mismatch.");
    }

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

    const int64_t hl_in = onnx_input_dim(hl_path);
    const int hl_in_expect = policy_dim + command_dim + (hl_obs_vel_ ? 2 : 0);
    if (hl_in != hl_in_expect) {
        throw std::runtime_error(
            "[HRL] high_level.onnx input dim " + std::to_string(hl_in) + " != policy+" +
            "command" + (hl_obs_vel_ ? "+hl_vel(2)" : "") + " = " +
            std::to_string(hl_in_expect) + " — check hrl.hl_obs_vel in deploy yaml.");
    }

    // The exported HL emits the learned goal cols (all, or velocity-only) + the
    // period dim when it owns the cadence.
    const int hl_out_expect =
        (velgoal_ ? goal_space_->task_dim() : goal_space_->dim()) + cadence_dim_;
    if ((int)hl_runner_->get_action().size() != hl_out_expect) {
        throw std::runtime_error(
            "[HRL] high_level.onnx output dim " +
            std::to_string(hl_runner_->get_action().size()) + " != expected " +
            std::to_string(hl_out_expect) +
            " (check hrl.hl_velocity_goals_only / hl_cadence / goal_components).");
    }
    // Pin the g -> V* decode (V* = ref + scale*g) to what the HL TRAINED with, so that
    // `commands.base_velocity.ranges` in deploy.yaml is free to be whatever the operator
    // wants. Those ranges are the joystick safety clamp (isaaclab observations.h
    // `velocity_commands`); until now GoalSpace ALSO derived scale/center from them, so
    // narrowing them for safety (e.g. lin_vel_x -0.5..1.0 -> -0.25..0.5) silently halved
    // scale_vx 0.75 -> 0.375 and gutted the HL's goal authority. A0 has no goal space and was
    // never affected. See doc/hrl/A1_findings.md (WL-C) + .claude/docs/deployment.md.
    const auto meta_scale = onnx_metadata_floats(hl_path, "goal_scale");
    if (meta_scale.empty()) {
        spdlog::warn(
            "[HRL] high_level.onnx carries no 'goal_scale' metadata (pre-2026-07-16 export) "
            "-> falling back to deriving the goal scale from deploy.yaml "
            "commands.base_velocity.ranges. Those ranges MUST then equal the ranges the "
            "policy was TRAINED with, and narrowing them for safety WILL silently rescale "
            "the HL's goals. Re-export the policy (play.py --export-onnx) to fix.");
    } else {
        goal_space_->freeze_scale(
            Eigen::Map<const Eigen::VectorXf>(meta_scale.data(), (Eigen::Index)meta_scale.size()));
        std::ostringstream s;
        for (size_t i = 0; i < meta_scale.size(); ++i) s << (i ? ", " : "") << meta_scale[i];
        spdlog::info("[HRL] goal scale pinned from ONNX metadata [{}] — deploy.yaml command "
                     "ranges are the operator clamp only (safe to narrow).", s.str());
    }
    // `absolute` mode also decodes g against GoalSpace::center(), which stays range-derived
    // (its height column must remain this deploy's imu-site nominal, not training's pelvis z
    // — see goal_space.h). So the clamp is only free to move in `delta` mode.
    if (hl_target_mode_ == "absolute") {
        spdlog::warn("[HRL] hl_target_mode=absolute: the absolute target CENTER is still "
                     "derived from deploy.yaml commands.base_velocity.ranges, so those "
                     "ranges must equal the TRAINED ranges — do NOT narrow them for safety "
                     "with this mode (delta mode is unaffected).");
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
            // Same (noised) estimate the LL's goal uses — training feeds the HL the
            // identical reading (td3._state_vec order: policy, command, hl_vel).
            if (hl_obs_vel_) { hl_in.push_back(s[0]); hl_in.push_back(s[1]); }
            const auto g_vec = hl_runner_->act({{"obs", hl_in}});
            const int gd = velgoal_ ? goal_space_->task_dim() : goal_space_->dim();
            const Eigen::VectorXf g = Eigen::Map<const Eigen::VectorXf>(g_vec.data(), gd);
            target_ = goal_space_->to_target(env.get(), s, g, hl_target_mode_, velgoal_);
            // HL-owned stride period: extra tanh dim -> affine map to the range, written
            // to the gait_phase clock for this window (phase-continuous; the term
            // integrates incrementally). Pinned clock was set once in the ctor.
            if (cadence_dim_ && pin_period_ <= 0.0f) {
                const float period =
                    period_lo_ + (g_vec[gd] + 1.0f) * 0.5f * (period_hi_ - period_lo_);
                env->cfg["observations"]["policy"]["gait_phase"]["params"]["period"] = period;
            }
        }
    }
    ++step_;

    // Low level observes the remaining delta V* - s and acts.
    const Eigen::VectorXf delta = target_ - s;
    std::vector<float> ll_in = policy;
    ll_in.insert(ll_in.end(), delta.data(), delta.data() + delta.size());
    const auto action = ll_runner_->act({{"obs", ll_in}});
    env->action_manager->process_action(action);

    if (telemetry_.enabled()) {
        float ar = 0.0f;  // mean |Δaction| vs the previous policy step
        if (last_action_.size() == action.size()) {
            for (size_t i = 0; i < action.size(); ++i)
                ar += std::abs(action[i] - last_action_[i]);
            ar /= action.size();
        }
        last_action_ = action;
        // The live phase clock in every mode (yaml default / ctor pin / HL-written).
        const float period =
            env->cfg["observations"]["policy"]["gait_phase"]["params"]["period"].as<float>();
        telemetry_.record(step_ * (float)env->step_dt, command.data(), s, target_, period,
                          env->robot->data.joint_pos[1], env->robot->data.joint_pos[7], ar);
    }
}

void State_RLHRL::run()
{
    auto action = env->action_manager->processed_actions();

    // Split deploy (ADR-0005): joints listed in hold_joint_ids track default_joint_pos
    // instead of the policy action (upper body held; obs still see all joints).
    static const auto hold_ids = env->cfg["hold_joint_ids"]
        ? env->cfg["hold_joint_ids"].as<std::vector<int>>() : std::vector<int>{};

#if SAFETY_FILTER
    // IMU tilt check — uses snapshot already captured by pre_run(), no extra lock needed
    const auto& q = env->robot->data.root_quat_w;
    float pitch = std::asin(std::clamp(2.0f*(q.w()*q.y() - q.z()*q.x()), -1.0f, 1.0f));
    float roll  = std::atan2(2.0f*(q.w()*q.x() + q.y()*q.z()), 1.0f - 2.0f*(q.x()*q.x() + q.y()*q.y()));
    bool tilt_safety = (std::abs(pitch) > H1_2_TILT_LIMIT || std::abs(roll) > H1_2_TILT_LIMIT);
    if (tilt_safety)
        spdlog::warn("[Safety] Tilt: pitch={:.2f} roll={:.2f} rad", pitch, roll);

    // Joint limits: per-joint COMMAND CLAMP, not a whole-body hold (changed 2026-07-16).
    // The A1a gait rides its ankle/hip/knee stops by 0.01-0.2 rad every stride; the old
    // any-joint->freeze response held all 27 joints mid-step and CAUSED the fall it
    // guarded against (bridge session 2026-07-16: alpha=1.0 at tilt 0.196, fall after).
    // Real firmware clamps the offending command; tilt/fall below keep the ramped hold.
    bool joint_hold = false;  // joint violations no longer feed the hold ramp

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

    bool any_clamp = false;
    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); i++) {
        int jid = (int)env->robot->data.joint_ids_map[i];
        float q_meas = env->robot->data.joint_pos[i];
        // alpha=0: pure policy output  |  alpha=1: hold at current measured position
        float q_cmd = (1.0f - alpha) * action[i] + alpha * q_meas;
        // per-joint safety clamp (commands past the mechanical stop are pointless and
        // trip the real firmware; measured grazes are the plant's business, not ours)
        const float lo = h1_2_joint_limits[jid].min, hi = h1_2_joint_limits[jid].max;
        if (q_cmd < lo || q_cmd > hi) { q_cmd = std::clamp(q_cmd, lo, hi); any_clamp = true; }
        if (std::find(hold_ids.begin(), hold_ids.end(), jid) != hold_ids.end())
            q_cmd = env->robot->data.default_joint_pos[i];
        lowcmd->msg_.motor_cmd()[jid].q() = q_cmd;
    }
    joint_hold = any_clamp;  // recorded as trig_joint in the flight log (no hold effect)

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
        int jid = (int)env->robot->data.joint_ids_map[i];
        bool held = std::find(hold_ids.begin(), hold_ids.end(), jid) != hold_ids.end();
        lowcmd->msg_.motor_cmd()[jid].q() = held ? env->robot->data.default_joint_pos[i] : action[i];
    }
#endif
}
