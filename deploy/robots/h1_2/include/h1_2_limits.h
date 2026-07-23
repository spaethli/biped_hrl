#pragma once
#include <array>

struct H12JointLimit { float min; float max; };

// Safety filter thresholds — single source of truth for run() and the logger meta.
static constexpr float H1_2_TILT_LIMIT      = 0.44f; // rad, ~25 deg
static constexpr float H1_2_FALL_ACC_THRESH = -7.0f; // m/s² world-frame vertical
static constexpr int   H1_2_FALL_ACC_WINDOW = 60;    // ticks
static constexpr int   H1_2_RAMP_CYCLES     = 50;    // ticks
static constexpr float H1_2_CONTROL_DT      = 0.001f;// FSM run loop period (1000 Hz)

// Joint names indexed by hardware joint ID (used by the safety flight recorder).
static const std::array<const char*, 27> h1_2_joint_names = {{
    "left_hip_yaw",   "left_hip_pitch",  "left_hip_roll",  "left_knee",
    "left_ankle_pitch","left_ankle_roll",
    "right_hip_yaw",  "right_hip_pitch", "right_hip_roll", "right_knee",
    "right_ankle_pitch","right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow",
    "left_wrist_roll","left_wrist_pitch","left_wrist_yaw",
    "right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow",
    "right_wrist_roll","right_wrist_pitch","right_wrist_yaw",
}};

// Position limits for all 27 H1-2 joints, indexed by hardware joint ID.
// Re-audited 2026-07-20/21 (WL-B0) against a 5-source consensus (3 independently
// vendored H1-2 URDFs + the mjlab training scene XML + the live /opt bridge scene XML,
// all identical to each other): 15/27 rows below were wrong (originally hand-entered
// from a Unitree documentation page, not the URDF/XML) — waist, both elbows, both
// wrist_roll/pitch/yaw, both shoulder_yaws (previously swapped L/R), and
// right_shoulder_pitch (previously mirrored; pitch does not mirror sign in this URDF's
// convention, unlike roll/yaw). See doc/hrl/A1a_deploy_plan.md "A0-first hardware track"
// item 2 for the full comparison table.
static const std::array<H12JointLimit, 27> h1_2_joint_limits = {{
    {-0.43f,   0.43f  },  //  0 LEFT_HIP_YAW
    {-3.14f,   2.50f  },  //  1 LEFT_HIP_PITCH
    {-0.43f,   3.14f  },  //  2 LEFT_HIP_ROLL
    {-0.12f,   2.19f  },  //  3 LEFT_KNEE
    {-0.8973f, 0.5236f},  //  4 LEFT_ANKLE_PITCH
    {-0.2618f, 0.2618f},  //  5 LEFT_ANKLE_ROLL
    {-0.43f,   0.43f  },  //  6 RIGHT_HIP_YAW
    {-3.14f,   2.50f  },  //  7 RIGHT_HIP_PITCH
    {-3.14f,   0.43f  },  //  8 RIGHT_HIP_ROLL
    {-0.12f,   2.19f  },  //  9 RIGHT_KNEE
    {-0.8973f, 0.5236f},  // 10 RIGHT_ANKLE_PITCH
    {-0.2618f, 0.2618f},  // 11 RIGHT_ANKLE_ROLL
    {-2.35f,   2.35f  },  // 12 WAIST_YAW
    {-3.14f,   1.57f  },  // 13 LEFT_SHOULDER_PITCH
    {-0.38f,   3.40f  },  // 14 LEFT_SHOULDER_ROLL
    {-2.66f,   3.01f  },  // 15 LEFT_SHOULDER_YAW
    {-0.95f,   3.18f  },  // 16 LEFT_ELBOW
    {-3.01f,   2.75f  },  // 17 LEFT_WRIST_ROLL
    {-0.4625f, 0.4625f},  // 18 LEFT_WRIST_PITCH
    {-1.27f,   1.27f  },  // 19 LEFT_WRIST_YAW
    {-3.14f,   1.57f  },  // 20 RIGHT_SHOULDER_PITCH
    {-3.40f,   0.38f  },  // 21 RIGHT_SHOULDER_ROLL
    {-3.01f,   2.66f  },  // 22 RIGHT_SHOULDER_YAW
    {-0.95f,   3.18f  },  // 23 RIGHT_ELBOW
    {-2.75f,   3.01f  },  // 24 RIGHT_WRIST_ROLL
    {-0.4625f, 0.4625f},  // 25 RIGHT_WRIST_PITCH
    {-1.27f,   1.27f  },  // 26 RIGHT_WRIST_YAW
}};
