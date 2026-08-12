// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <cstdlib>
// [SAFETY FILTER] flight recorder is robot-local (h1_2 only). The guard keeps
// other robots (g1, a2, go2…) — which share this header but lack the file on
// their include path — building unchanged. Remove this block when reverting.
#if __has_include("safety_logger.h")
#  include "safety_logger.h"
#  define STATE_RLBASE_HAS_SAFETY_LOGGER 1
#endif

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

#ifdef STATE_RLBASE_HAS_SAFETY_LOGGER
        // Opt-in flight recorder: enabled only if H1_2_SAFETY_LOG is set.
        if (const char* sp = std::getenv("H1_2_SAFETY_LOG"))
            // with_estimator=true since 2026-08-12: A0 is the only policy that stands
            // still, so it is the only source of a known-zero velocity reference for the
            // offline estimator comparison. Inside __has_include("safety_logger.h"), which
            // only h1_2 provides, so g1/a2/go2/r1 are unaffected.
            safety_logger_.init(sp, env->robot->data.joint_ids_map,
                                H1_2_TILT_LIMIT, H1_2_FALL_ACC_THRESH, H1_2_CONTROL_DT,
                                /*with_estimator=*/true,
                                env->cfg["joint_offset"]
                                  ? env->cfg["joint_offset"].as<std::vector<float>>()
                                  : std::vector<float>{});
#endif

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
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
#ifdef STATE_RLBASE_HAS_SAFETY_LOGGER
        safety_logger_.flush(); // write any buffered rows to disk
#endif
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // [SAFETY FILTER] — remove these lines when reverting
    int hold_counter_{0};
    int fall_acc_counter_{0};
#ifdef STATE_RLBASE_HAS_SAFETY_LOGGER
    SafetyLogger safety_logger_;
    // DDS-rate sensor snapshot, filled in run() inside the fall detector's existing lock
    // (2026-08-12). A0 does not consume it; it exists so the only genuinely-still regime on
    // this robot can be replayed offline against a known v == 0.
    EstSample est_sample_{};
#endif
};

REGISTER_FSM(State_RLBase)
