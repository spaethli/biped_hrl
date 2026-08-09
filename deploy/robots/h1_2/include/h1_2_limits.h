#pragma once
#include <array>

struct H12JointLimit { float min; float max; };
struct H12JointTrip  { float tau; float dq; };  // |tau_est| Nm, |dq| rad/s trip thresholds

// Safety filter thresholds — single source of truth for run() and the logger meta.
static constexpr float H1_2_TILT_LIMIT      = 0.44f; // rad, ~25 deg
static constexpr float H1_2_FALL_ACC_THRESH = -7.0f; // m/s² world-frame vertical
static constexpr int   H1_2_FALL_ACC_WINDOW = 60;    // ticks
static constexpr int   H1_2_RAMP_CYCLES     = 50;    // ticks
static constexpr float H1_2_CONTROL_DT      = 0.001f;// FSM run loop period (1000 Hz)

// Does the per-joint torque/velocity trip ACTUATE (ramped hold at the measured position),
// or only detect and log? 0 = detect + log only. DEFAULT 0 SINCE 2026-08-05, ON EVIDENCE:
// with the hold enabled, the trip fired BEFORE the fall in 5 of 6 bridge falls (lead 0.17
// to 3.2 s, with the ramp fully engaged first), and the shipped-plant 30 s cell went from
// 0/2 falls with no filter to 2/4 (ground truth) and 4/4 (estimator) with it. That is the
// 2026-07-16 finding repeating: the whole-body position hold "held all 27 joints mid-step
// and CAUSED the fall it guarded against". Scoping the hold PER JOINT did not remove the
// hazard -- pinning a single leg joint mid-stride is enough to put the robot down.
// Detection is kept on because it costs nothing and is pure signal: trig_torque/trig_dq/
// trip_jid/trip_ratio still record what fired and how close every joint came.
// Flip to 1 only once a bridge battery shows the thresholds never firing during normal
// walking; while it fires at all, enabling this trades a rare failure for a frequent one.
#define H1_2_TRIP_HOLD 0

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

// Per-joint torque / joint-velocity TRIP thresholds (2026-08-05). A joint over either one
// gets its own ramped hold in State_RLHRL::run(), per joint, exactly like the position
// clamp above and unlike the whole-body tilt/fall hold.
//
// WHY THESE VALUES. Sized at 1.3x the MEASURED HEALTHY-WALKING maximum per joint group,
// over the UNION of two sources: 389 s of real-robot walking pooled from 10 hardware
// sessions (2026-07-20 .. 2026-08-03, leg-active samples with the IMU tilt envelope under
// 12 deg for +-1 s so stumbles and falls are excluded) and 103 s of non-falling bridge
// walking. Sizing off the STAND instead would be badly wrong: the stand sits at
// 36 Nm / 0.06 rad/s, which every stride exceeds by 5-350x.
//
// WHY THE UNION, learned the hard way the same day. The first cut used the hardware pool
// alone. That pool is A0-DOMINATED, and A1 is measurably twitchier (action rate 0.92-1.28
// vs A0 0.57-0.75), so the A0-sized table held right_hip_yaw during the A1 candidate's own
// normal bridge gait -- a fall cause, not a fall guard. The two sources disagree most on
// hip_yaw torque (hardware 44 Nm, bridge 174 Nm) and on the held waist/arm joints, and the
// binding source is noted per group by scripts/measure_joint_trip_thresholds.py. COST ON
// RECORD: raising hip_yaw torque 60 -> 230 Nm to clear the bridge means the 2026-08-05
// splay no longer trips RIGHT hip yaw on torque. It still trips on four other joints,
// first-sample detection is unchanged, and left hip yaw still catches it on velocity.
//
// WHY PER JOINT AND NOT WHOLE BODY. Walking and the failure overlap in whole-body terms --
// walking peaks at 296 Nm / 21.5 rad/s (knee, ankle) while the 2026-08-05 leg splay peaked
// at 330 Nm / 15.2 rad/s (hip) -- so no single whole-body threshold separates them. Per
// joint they separate cleanly, because the loud joints in walking (knee, ankle) are not the
// joints that splay (hip, waist). Scored against this table: ZERO trips across all 389 s of
// walking, and the splay caught on its FIRST anomalous sample (5 joints at once: both hip
// yaws, both hip pitches, right hip roll), 260-380 ms of dwell.
//
// The joints whose threshold sits above anything the splay produced (knee, ankles,
// elbow/wrist) are deliberately still covered: they are runaway backstops for a failure
// that goes through THOSE joints, not dead entries.
//
// NOTE these are NOT the model's actuatorfrcrange. Real tau_est routinely reads past it --
// ankle pitch walks at 95.7 Nm against a +-60 Nm rating -- so the XML ratings are not a
// trustworthy ceiling and the measurement is the authority.
// Recompute with scripts/measure_joint_trip_thresholds.py after any gait/plant change.
static const std::array<H12JointTrip, 27> h1_2_joint_trips = {{
    { 230.0f,  10.0f},  //  0 LEFT_HIP_YAW
    { 200.0f,   9.5f},  //  1 LEFT_HIP_PITCH
    { 265.0f,  11.0f},  //  2 LEFT_HIP_ROLL
    { 385.0f,  21.0f},  //  3 LEFT_KNEE
    { 125.0f,  28.0f},  //  4 LEFT_ANKLE_PITCH
    {  45.0f,  27.0f},  //  5 LEFT_ANKLE_ROLL
    { 230.0f,  10.0f},  //  6 RIGHT_HIP_YAW
    { 200.0f,   9.5f},  //  7 RIGHT_HIP_PITCH
    { 265.0f,  11.0f},  //  8 RIGHT_HIP_ROLL
    { 385.0f,  21.0f},  //  9 RIGHT_KNEE
    { 125.0f,  28.0f},  // 10 RIGHT_ANKLE_PITCH
    {  45.0f,  27.0f},  // 11 RIGHT_ANKLE_ROLL
    {  80.0f,   7.0f},  // 12 WAIST_YAW
    {  40.0f,   5.5f},  // 13 LEFT_SHOULDER_PITCH
    {  40.0f,   5.5f},  // 14 LEFT_SHOULDER_ROLL
    {  40.0f,   5.5f},  // 15 LEFT_SHOULDER_YAW
    {  25.0f,   7.5f},  // 16 LEFT_ELBOW
    {  25.0f,   7.5f},  // 17 LEFT_WRIST_ROLL
    {  25.0f,   7.5f},  // 18 LEFT_WRIST_PITCH
    {  25.0f,   7.5f},  // 19 LEFT_WRIST_YAW
    {  40.0f,   5.5f},  // 20 RIGHT_SHOULDER_PITCH
    {  40.0f,   5.5f},  // 21 RIGHT_SHOULDER_ROLL
    {  40.0f,   5.5f},  // 22 RIGHT_SHOULDER_YAW
    {  25.0f,   7.5f},  // 23 RIGHT_ELBOW
    {  25.0f,   7.5f},  // 24 RIGHT_WRIST_ROLL
    {  25.0f,   7.5f},  // 25 RIGHT_WRIST_PITCH
    {  25.0f,   7.5f},  // 26 RIGHT_WRIST_YAW
}};
