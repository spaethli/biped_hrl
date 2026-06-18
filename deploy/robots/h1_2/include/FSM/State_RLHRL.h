// A1 (HIRO) two-level deploy state. Unlike State_RLBase (one ONNX, command inside the
// obs), this runs the hierarchy: a high level fires every `c` steps and emits a bounded
// goal g -> absolute target V* (held over the window); the low level observes the
// remaining delta V* - s (command removed) and acts. Velocity/height for the goal-space
// state s come from the sim HighState (SportModeState) — see State_RLHRL.cpp.
//
// A0 path (State_RLBase) is untouched; pick this state via config.yaml `type: RLHRL`.
#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "hrl/goal_space.h"

#include <unitree/dds_wrapper/robots/go2/go2.h>  // go2::subscription::SportModeState

#include <filesystem>
#include <memory>
#include <random>
#include <thread>

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
        step_ = 0;
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
    Eigen::VectorXf target_;  // held window target V* (refreshed by the HL every c steps)
    long step_{0};

    // Optional synthetic noise on the goal-space state s (per-dim std; 0 = off). Lets us
    // mimic the real-robot estimator noise on velocity/height in sim (where s is exact).
    Eigen::VectorXf state_noise_std_;
    std::mt19937 rng_{std::random_device{}()};

    std::thread policy_thread;
    bool policy_thread_running = false;

    // [SAFETY FILTER] — mirrors State_RLBase (duplicated, not shared, to leave A0 untouched)
    int hold_counter_{0};
    int fall_acc_counter_{0};
};

REGISTER_FSM(State_RLHRL)
