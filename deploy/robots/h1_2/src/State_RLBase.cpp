#include "FSM/State_RLBase.h"
#include <algorithm>
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "h1_2_observations.h"  // robot-local terms: keyboard_velocity_commands, gait_phase_cmd

// [SAFETY FILTER] set to 0 to revert to unfiltered policy output
#define SAFETY_FILTER 1
#if SAFETY_FILTER
#  include "h1_2_limits.h"
#  include <cmath>
#  include <algorithm>
#  include <unitree/dds_wrapper/robots/go2/go2.h>  // go2::subscription::SportModeState (sim ground truth)
#endif

// keyboard_velocity_commands + gait_phase_cmd moved to include/h1_2_observations.h
// (2026-07-21) so State_RLHRL gets the same registrations without depending on which
// object files the linker keeps.

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

    // Joint limits: per-joint COMMAND CLAMP, not a whole-body hold (ported from State_RLHRL
    // 2026-07-23, where it landed 2026-07-16). On hardware real compliance and encoder noise
    // let the MEASURED q cross a bound by a hair on normal strides (0.002-0.02 rad observed);
    // the old any-joint-out -> hold-all-27 response then drove the chasing hold, which
    // degenerates to damping-only torque and takes the whole robot near-passive mid-stance.
    // Measured on the 2026-07-23 hardware runs: trig_joint on 4.56% of rows with alpha
    // reaching 1.00, while trig_tilt and trig_fall were both ZERO. Real firmware clamps the
    // offending command; tilt/fall below keep the ramped whole-body hold.
    bool joint_hold = false;  // joint violations no longer feed the hold ramp

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

    bool any_clamp = false;
    int tick_jid = -1; float tick_over = 0.0f, tick_cmd = 0.0f;  // worst offender this tick
    for (int i = 0; i < (int)env->robot->data.joint_ids_map.size(); i++) {
        int jid = (int)env->robot->data.joint_ids_map[i];
        float q_meas = env->robot->data.joint_pos[i];
        // alpha=0: pure policy output  |  alpha=1: hold at current measured position
        float q_cmd = (1.0f - alpha) * action[i] + alpha * q_meas;
        // per-joint safety clamp (commands past the mechanical stop are pointless and
        // trip the real firmware; measured grazes are the plant's business, not ours)
        const float lo = h1_2_joint_limits[jid].min, hi = h1_2_joint_limits[jid].max;
        if (q_cmd < lo || q_cmd > hi) {
            float over = (q_cmd < lo) ? (lo - q_cmd) : (q_cmd - hi);
            if (over > tick_over) { tick_over = over; tick_jid = jid; tick_cmd = q_cmd; }
            q_cmd = std::clamp(q_cmd, lo, hi);
            any_clamp = true;
        }
        if (std::find(hold_ids.begin(), hold_ids.end(), jid) != hold_ids.end())
            q_cmd = env->robot->data.default_joint_pos[i];
        lowcmd->msg_.motor_cmd()[jid].q() = q_cmd;
    }
    joint_hold = any_clamp;  // recorded as trig_joint in the flight log (no hold effect)

    // Rate-limited console notification (user's call 2026-07-23): clamping is normal-gait
    // behaviour at 500 Hz so a per-event warn would spam, but full silence would hide a
    // joint that is genuinely pinned. One line per second with the worst offender since the
    // last line. Statics, not members: State_RLBase.h is the SHARED deploy header (g1/go2/
    // a2 build against it) and this fix is scoped robot-local.
    static int warn_tick = 0, clamp_ticks = 0, warn_jid = -1;
    static float warn_over = 0.0f, warn_cmd = 0.0f;
    if (any_clamp) {
        clamp_ticks++;
        if (tick_over > warn_over) { warn_over = tick_over; warn_jid = tick_jid; warn_cmd = tick_cmd; }
    }
    if (++warn_tick >= 500) {  // 500 Hz control loop
        if (clamp_ticks > 0)
            spdlog::warn("[Safety] Clamp: {}/500 ticks, worst joint {} cmd={:.3f} over by {:.3f} rad",
                clamp_ticks, warn_jid, warn_cmd, warn_over);
        warn_tick = 0; clamp_ticks = 0; warn_jid = -1; warn_over = 0.0f; warn_cmd = 0.0f;
    }

#ifdef STATE_RLBASE_HAS_SAFETY_LOGGER
    // Flight recorder: logs raw policy intent (action) vs measured state + triggers.
    if (safety_logger_.enabled()) {
        float quat[4] = { q.w(), q.x(), q.y(), q.z() };
        float acc[3]  = { lin_acc_b.x(), lin_acc_b.y(), lin_acc_b.z() };

        // Commanded twist (2026-07-21, WL-B0): whichever command term this deploy config
        // actually uses (keyboard = sim bridge, joystick = real robot). A0 has no member
        // holding this (unlike A1's cached `command` obs), so re-derive it directly — same
        // free function the observation manager would call, cheap (map lookup + 3 clamps).
        static const bool use_keyboard = isaaclab::uses_keyboard_commands(env.get());
        const auto cmd_vec = use_keyboard
            ? isaaclab::keyboard_velocity_commands(env.get(), YAML::Node())
            : isaaclab::mdp::velocity_commands(env.get(), YAML::Node());
        const float cmd[3] = { cmd_vec[0], cmd_vec[1], cmd_vec[2] };

        // Ground-truth body-frame base velocity (SIM ONLY — same SportModeState source A1
        // already reads for its goal state; ~0 on real hardware, see E1). A0's actor never
        // consumes this; it exists here purely for flight-recorder analysis (sim-to-sim
        // achieved-vx, and correlating a fall with the command active at the time).
        static auto highstate = std::make_shared<unitree::robot::go2::subscription::SportModeState>();
        const Eigen::Vector3f v_world = highstate->velocity();
        const Eigen::Vector3f lin_vel_b = env->robot->data.root_quat_w.conjugate() * v_world;
        const float ach_vel[3] = { lin_vel_b.x(), lin_vel_b.y(), lin_vel_b.z() };

        // Gait clock exactly as the policy saw it last policy step (defect 0, 2026-07-21):
        // (0,0) while standing is correct, (0,0) under a non-zero command means the clock
        // is dead. The instrument that proves the fix live.
        safety_logger_.record(action,
                              env->robot->data.joint_pos.data(),
                              env->robot->data.joint_vel.data(),
                              quat, acc, alpha, joint_hold, tilt_safety, fall_detected,
                              cmd, ach_vel, isaaclab::g_gait_phase_obs);
    }
#endif
#else
    // Original unfiltered policy output
    for(int i(0); i < (int)env->robot->data.joint_ids_map.size(); i++) {
        int jid = (int)env->robot->data.joint_ids_map[i];
        bool held = std::find(hold_ids.begin(), hold_ids.end(), jid) != hold_ids.end();
        lowcmd->msg_.motor_cmd()[jid].q() = held ? env->robot->data.default_joint_pos[i] : action[i];
    }
#endif
}