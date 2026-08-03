// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;
    }

    // Per-joint encoder-zero correction (sdk order, same indexing as joint_ids_map):
    // true angle = reported angle + joint_offset[i]. Empty (the default) is a no-op;
    // only an h1_2 real-robot deploy config populates it (ADR-0006 Spec B). Set by the
    // caller from deploy_real.yaml's `joint_offset` before this articulation starts
    // running, so g1/go2/a2/r1 — which never set it — are unaffected.
    std::vector<float> joint_offset;

    void update() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        // base_angular_velocity
        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // joint positions and velocities. joint_pos lands in TRUE joint coordinates (see
        // joint_offset above) so it stays consistent with projected_gravity_b for every
        // downstream consumer (observations, the safety clamp's q_meas, the hold).
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            const float offset = (i < (int)joint_offset.size()) ? joint_offset[i] : 0.0f;
            data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q() + offset;
            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }
    }

    LowStatePtr lowstate;
};

}