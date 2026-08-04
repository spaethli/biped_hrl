// A1 (HIRO) two-level deploy state. Unlike State_RLBase (one ONNX, command inside the
// obs), this runs the hierarchy: a high level fires every `c` steps and emits a bounded
// goal g -> absolute target V* (held over the window); the low level observes the
// remaining delta V* - s (command removed) and acts. Velocity/height for the goal-space
// state s come from the sim HighState (SportModeState) — see State_RLHRL.cpp — or, opt-in
// (`hrl.base_vel_from_imu` / `hrl.base_height_from_fk`), from the deployable IMU/FK
// estimator described below, which is what a real H1-2 can actually run on.
//
// A0 path (State_RLBase) is untouched; pick this state via config.yaml `type: RLHRL`.
#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "hrl/base_state.h"
#include "hrl/goal_space.h"
#include "hrl/hrl_telemetry.h"

#include <unitree/dds_wrapper/robots/go2/go2.h>  // go2::subscription::SportModeState

#include <array>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <cstdlib>

// [SAFETY FILTER] master switch for the HRL state (mirrors State_RLBase). When 1,
// the position-hold filter is active AND the flight recorder logs automatically if
// H1_2_SAFETY_LOG is set (the launch script sets it). When 0, neither is compiled in.
// ON since 2026-07-14 (deploy plan): every bridge gate runs the hardware build.
#define SAFETY_FILTER 1
#if SAFETY_FILTER
#  include "safety_logger.h"
#endif

class State_RLHRL : public FSMState
{
public:
    State_RLHRL(int state_mode, std::string state_string);

    void enter()
    {
        // Load the two ONNX on first entry (not in the ctor): both Velocity (A0) and
        // VelocityHRL states are constructed at startup, so a ctor load would crash the
        // whole binary — incl. A0 — if the HRL ONNX aren't placed yet. Lazy load keeps A0
        // independent; entering VelocityHRL is what requires the files.
        ensure_models_loaded();

        // set gains (same as State_RLBase)
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        // Re-prime the base-state estimator for this entry. Without it a re-entry keeps the
        // dv_ left over from the last tick before exit(), plus the lev0_/gt_vel0_ captured
        // in the previous entry's window, until the next window start clears them — up to
        // c-1 policy steps of a bogus velocity increment (bounded by |w||r| ~ 0.4 m/s),
        // which the LL would see as a real goal error. step_ deliberately does NOT reset
        // here (telemetry continuity), so the first partial window after a re-entry is a
        // little short; that window's target_ is stale from before the exit anyway.
        {
            const float psi = sdk_slot_[12] >= 0
                ? env->robot->data.joint_pos[sdk_slot_[12]] : 0.0f;
            const Eigen::Vector3f lev =
                env->robot->data.root_ang_vel_b.cross(hrl::kImuOffsetT);
            std::lock_guard<std::mutex> lock(est_mu_);
            dv_.setZero();
            lev0_ = hrl::rz(psi, lev);
            gt_vel0_ = hrl::rz(psi, (env->robot->data.root_quat_w.conjugate()
                                     * highstate_->velocity()) - lev);
        }

        // step_ drives the telemetry timestamp (t = step_*step_dt) and the HL firing
        // schedule (step_ % c_); only zero it on the FIRST entry this process, so re-
        // entries (retries within one session) don't collide their `t` back onto an
        // earlier attempt's (2026-07-17: this + the truncate-on-init bug were silently
        // destroying failed attempts' telemetry before a later successful retry).
        if (first_entry_) {
          step_ = 0;
          first_entry_ = false;
        }

        // Deploy-gate telemetry (W3); same output base as the flight recorder.
        if (const char* sp = std::getenv("H1_2_SAFETY_LOG"))
            telemetry_.init(sp, goal_space_->dim());
        last_action_.clear();

#if SAFETY_FILTER
        // Opt-in flight recorder: enabled only if H1_2_SAFETY_LOG is set (launch script).
        if (const char* sp = std::getenv("H1_2_SAFETY_LOG"))
            safety_logger_.init(sp, env->robot->data.joint_ids_map,
                                H1_2_TILT_LIMIT, H1_2_FALL_ACC_THRESH, H1_2_CONTROL_DT);
#endif

        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                policy_step();

                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        telemetry_.flush();
#if SAFETY_FILTER
        safety_logger_.flush(); // write any buffered rows to disk
#endif
    }

private:
    // Load high_level.onnx + low_level.onnx (once); guards both files exist + dim match.
    void ensure_models_loaded();
    // One HRL control step: HL fires every c steps (holds V*), LL acts on V* - s.
    void policy_step();

    std::filesystem::path exported_dir_;  // resolved in ctor; ONNX loaded lazily from here

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<isaaclab::OrtRunner> hl_runner_;  // high_level.onnx: policy++command -> g
    std::unique_ptr<isaaclab::OrtRunner> ll_runner_;  // low_level.onnx:  policy++delta   -> action
    std::unique_ptr<hrl::GoalSpace> goal_space_;
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> highstate_;

    int c_{8};
    std::string hl_target_mode_{"absolute"};
    bool oracle_{false};  // hl_algorithm==oracle: V* computed analytically, no high_level.onnx
    // Keeper-structure knobs (deploy.yaml `hrl:` block; absent key = old behavior, so
    // pre-velgoal 7-dim checkpoints run unchanged). Mirror config/h1_2_a1/rl_cfg.py names.
    bool hl_obs_vel_{false};   // HL input = policy ++ command ++ (vx,vy estimate)
    bool velgoal_{false};      // hl_velocity_goals_only: HL emits the velocity goal cols only
    int cadence_dim_{0};       // 1 = HL owns the stride period (hl_cadence, source=hl, learned)
    float period_lo_{0.35f}, period_hi_{1.0f};  // cadence_period_range (tanh affine map)
    float pin_period_{0.0f};   // >0 = freeze the LL phase clock at this period (bring-up pin)
    // The gait-clock term's `params` node (aliases env->cfg): writing ["period"] retunes
    // the LL clock live. Bound in the ctor to whichever term name the yaml uses.
    YAML::Node phase_params_;
    Eigen::VectorXf target_;  // held window target V* (refreshed by the HL every c steps)
    long step_{0};
    bool first_entry_{true};  // gates the step_ reset to the process's first enter() only

    // Optional synthetic noise on the goal-space state s (per-dim std; 0 = off). Lets us
    // mimic the real-robot estimator noise on velocity/height in sim (where s is exact).
    Eigen::VectorXf state_noise_std_;
    std::mt19937 rng_{std::random_device{}()};

    // Deploy-gate telemetry (plan W3): 50 Hz cmd/s/V*/period CSV, on when
    // H1_2_SAFETY_LOG is set (independent of SAFETY_FILTER). Flushed in exit().
    hrl::Telemetry telemetry_;
    std::vector<float> last_action_;  // for the logged action-rate scalar

    // Cached for the SafetyLogger CSV (2026-07-21, WL-B0): computed in policy_step() at
    // step_dt cadence, read by run() at the 1kHz control loop, same split as last_action_.
    Eigen::Vector3f last_cmd_{0, 0, 0};
    Eigen::Vector3f last_lin_vel_b_{0, 0, 0};  // sim-only ground truth, ~0 on real hardware

    // --- Deployable base-state estimator (2026-08-03) --------------------------------
    // rt/sportmodestate is the only source of base velocity/height today, and it publishes
    // ZEROS once our own low-level controller has command of a real H1-2 (E1). Two
    // replacements, both opt-in from deploy.yaml's `hrl:` block; an ABSENT key keeps the
    // SportModeState path, so every existing config/checkpoint replays byte-identically:
    //   base_vel_from_imu   -> the WITHIN-WINDOW base-velocity increment, integrated from
    //                          the torso IMU in run() at 1 kHz. Legitimate only because
    //                          `delta` mode cancels absolute velocity:
    //                          V* - s_i = scale*g - (s_i - s_t0).
    //   base_height_from_fk -> lowest-foot forward kinematics off the leg encoders.
    // Both are computed UNCONDITIONALLY so the telemetry can score them against the sim
    // bridge's ground truth (permanent regression guard) even while the policy is still
    // being fed the privileged signal.
    bool vel_from_imu_{false};
    bool height_from_fk_{false};
    float nominal_h_{0.0f};        // hrl.nominal_root_height (imu-site referenced, 1.3076)
    float fk_h_nominal_{0.0f};     // leg FK evaluated at default_joint_pos -> the anchor
    std::array<int, 13> sdk_slot_; // sdk motor id (0-11 legs, 12 waist) -> articulation slot

    // dv_ is the ONLY state shared across the two threads: run() (1 kHz control loop)
    // integrates the gravity-free IMU acceleration into it; policy_step() (50 Hz policy
    // thread) reads it and zeroes it at each HL window start. Three floats under est_mu_.
    std::mutex est_mu_;
    Eigen::Vector3f dv_{0, 0, 0};
    // Policy-thread-only (never touched by run(), so no lock): the window-start lever-arm
    // term and the window-start ground-truth pelvis velocity.
    Eigen::Vector3f lev0_{0, 0, 0};
    Eigen::Vector3f gt_vel0_{0, 0, 0};

    std::thread policy_thread;
    bool policy_thread_running = false;

    // [SAFETY FILTER] — mirrors State_RLBase (duplicated, not shared, to leave A0 untouched)
    int hold_counter_{0};
    int fall_acc_counter_{0};
#if SAFETY_FILTER
    SafetyLogger safety_logger_;
#endif
};

REGISTER_FSM(State_RLHRL)
