#pragma once
#include <array>

struct H12JointLimit { float min; float max; };

// Position limits for all 27 H1-2 joints, indexed by hardware joint ID.
// Matches the limits used in the invDyn safety filter.
static const std::array<H12JointLimit, 27> h1_2_joint_limits = {{
    {-0.43f,   0.43f  },  //  0 LEFT_HIP_YAW
    {-3.14f,   2.50f  },  //  1 LEFT_HIP_PITCH
    {-0.43f,   3.14f  },  //  2 LEFT_HIP_ROLL
    {-0.26f,   2.05f  },  //  3 LEFT_KNEE
    {-0.8973f, 0.5236f},  //  4 LEFT_ANKLE_PITCH
    {-0.2618f, 0.2618f},  //  5 LEFT_ANKLE_ROLL
    {-0.43f,   0.43f  },  //  6 RIGHT_HIP_YAW
    {-3.14f,   2.50f  },  //  7 RIGHT_HIP_PITCH
    {-3.14f,   0.43f  },  //  8 RIGHT_HIP_ROLL
    {-0.26f,   2.05f  },  //  9 RIGHT_KNEE
    {-0.8973f, 0.5236f},  // 10 RIGHT_ANKLE_PITCH
    {-0.2618f, 0.2618f},  // 11 RIGHT_ANKLE_ROLL
    {-3.14f,   1.57f  },  // 12 WAIST_YAW
    {-3.14f,   1.57f  },  // 13 LEFT_SHOULDER_PITCH
    {-0.38f,   3.40f  },  // 14 LEFT_SHOULDER_ROLL
    {-3.01f,   2.66f  },  // 15 LEFT_SHOULDER_YAW
    {-2.53f,   1.60f  },  // 16 LEFT_ELBOW
    {-2.967f,  2.967f },  // 17 LEFT_WRIST_ROLL
    {-0.471f,  0.349f },  // 18 LEFT_WRIST_PITCH
    {-1.012f,  1.012f },  // 19 LEFT_WRIST_YAW
    {-1.57f,   3.14f  },  // 20 RIGHT_SHOULDER_PITCH
    {-3.40f,   0.38f  },  // 21 RIGHT_SHOULDER_ROLL
    {-2.66f,   3.01f  },  // 22 RIGHT_SHOULDER_YAW
    {-1.60f,   2.53f  },  // 23 RIGHT_ELBOW
    {-2.967f,  2.967f },  // 24 RIGHT_WRIST_ROLL
    {-0.471f,  0.349f },  // 25 RIGHT_WRIST_PITCH
    {-1.012f,  1.012f },  // 26 RIGHT_WRIST_YAW
}};
