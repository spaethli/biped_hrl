#pragma once
// [SAFETY FILTER] flight recorder — delete this file when reverting.
//
// Two-file output (base path from H1_2_SAFETY_LOG):
//   <base>.csv        standard CSV, one header row + one row per control tick
//                      (decimated to ~500 Hz; a tick with trig_joint/trig_tilt/
//                      trig_fall set is NEVER decimated, so every raw joint-limit
//                      violation and every tilt/fall event is still captured at
//                      the full control rate)
//   <base>_meta.json  static config written once at init (limits, names, dt)
//
// Hot-path safe: record() only appends a formatted line to an in-RAM string
// buffer (no syscall). The file is written on periodic flush (~5 s) and on exit.
// Floats are logged at 4 significant digits (2026-07-27, file-size reduction) --
// plenty for joint angles/velocities, well past the sensor noise floor.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "h1_2_limits.h"

// Process-wide (NOT per-instance) entry counter: State_RLBase (A0) and State_RLHRL (A1)
// each own a SEPARATE SafetyLogger object, so a per-instance counter would restart at 0
// the first time the OTHER FSM type is entered, colliding with entries already written
// by the first type to the same shared file. 2026-07-17 finding (2nd round): a session
// that ran A0 first, then switched to A1, needs entry ids that stay unique across that
// switch, not just across repeated entries into the same type.
inline int g_safety_logger_entry = -1;

class SafetyLogger
{
public:
    // Enabled only if base_path is non-empty (H1_2_SAFETY_LOG set).
    bool enabled() const { return !base_.empty(); }

    // Called once per State ENTRY (every `h`/`o` press), not once per process, and A0/A1
    // use SEPARATE SafetyLogger instances (see g_safety_logger_entry above). Whether to
    // truncate is decided by whether the file already exists on disk (robust across BOTH
    // repeated entries into one FSM type AND switching between A0 and A1), not by this
    // instance's own memory of having run before -- an instance that has never run yet
    // (e.g. A1's, the first time you press `h` after already having used A0) must still
    // append, not wipe out what the other type already wrote this session.
    void init(const std::string& base_path, const std::vector<float>& joint_ids_map,
              float tilt_limit, float fall_acc_thresh, float control_dt)
    {
        base_ = base_path;
        if (base_.empty()) return;
        n_ = (int)joint_ids_map.size();
        dt_ = control_dt;
        buf_ << std::setprecision(4);  // significant digits, not decimal places
        // Decimate quiet ticks to ~500 Hz (control loop runs at 1/control_dt, e.g. 1000 Hz).
        // record() ignores this factor and always logs a tick when a safety trigger fires,
        // so this only shrinks the file -- it never drops a joint clamp or a tilt/fall event.
        constexpr float kTargetLogHz = 500.0f;
        log_every_ = std::max(1, (int)std::lround((1.0f / kTargetLogHz) / control_dt));
        g_safety_logger_entry++;  // every entry gets a new id, incl. the first (starts at 0)
        std::ifstream existing(base_ + ".csv");
        bool first_entry = !existing.good();
        existing.close();
        if (!first_entry) return;  // re-entry (this type or the other): append, keep tick_
        tick_ = 0;

        // --- meta json (single source of truth for the analyzer) ---
        std::ofstream meta(base_ + "_meta.json");
        meta << "{\n";
        meta << "  \"num_joints\": " << n_ << ",\n";
        meta << "  \"control_dt\": " << control_dt << ",\n";
        meta << "  \"tilt_limit\": " << tilt_limit << ",\n";
        meta << "  \"fall_acc_thresh\": " << fall_acc_thresh << ",\n";
        meta << "  \"joints\": [\n";
        for (int i = 0; i < n_; i++) {
            int jid = (int)joint_ids_map[i];
            meta << "    {\"slot\": " << i << ", \"hw_id\": " << jid
                 << ", \"name\": \"" << h1_2_joint_names[jid] << "\""
                 << ", \"min\": " << h1_2_joint_limits[jid].min
                 << ", \"max\": " << h1_2_joint_limits[jid].max << "}"
                 << (i < n_ - 1 ? "," : "") << "\n";
        }
        meta << "  ]\n}\n";
        meta.close();

        // --- csv header ---
        buf_.str("");
        buf_ << "t";
        for (int i = 0; i < n_; i++) buf_ << ",raw_q" << i;
        for (int i = 0; i < n_; i++) buf_ << ",meas_q" << i;
        for (int i = 0; i < n_; i++) buf_ << ",meas_dq" << i;
        buf_ << ",quat_w,quat_x,quat_y,quat_z,acc_x,acc_y,acc_z";
        // cmd_* = commanded twist (keyboard/joystick, both FSMs). ach_* = ground-truth body-
        // frame base velocity from the sim bridge's SportModeState (2026-07-21, WL-B0) — SIM
        // ONLY, added for sim-to-sim validation (e.g. A0 achieved-vx) and for correlating a
        // fall with what was commanded at the time (the flight recorder previously had no
        // command column at all). Reads 0 on real hardware: same estimator gap as E1, the
        // SportModeState publisher goes silent once a custom low-level controller has command.
        buf_ << ",cmd_vx,cmd_vy,cmd_wz,ach_vx,ach_vy,ach_vz";
        // phase_sin/phase_cos = the gait-clock observation AS THE POLICY SAW IT (post
        // stand-mask), from the robot-local gait_phase_cmd term (2026-07-21, defect 0).
        // (0,0) under cmd 0 is correct; (0,0) while a command is held means the clock is
        // dead, which is precisely the bug the joystick-hardcoded shared term caused.
        buf_ << ",phase_sin,phase_cos";
        // trig_joint MEANING (A1 since 2026-07-16, A0 since 2026-07-23): "at least one
        // commanded joint was clamped to its h1_2_limits.h bound this tick", with NO hold
        // effect. It no longer contributes to alpha, which is now a pure tilt/fall
        // indicator for both FSM types. Older CSVs mean the opposite ("a MEASURED joint
        // left range -> whole-body hold engaged"), so trig_joint=1 with alpha>0 in an A0
        // capture dated before 2026-07-23 is the OLD semantics — do not mix the two eras.
        buf_ << ",alpha,trig_joint,trig_tilt,trig_fall,entry\n";
        // Truncate/create the file and write the header now (first entry only).
        std::ofstream(base_ + ".csv", std::ios::trunc) << buf_.str();
        buf_.str("");
        rows_since_flush_ = 0;
    }

    // Append one tick. Cheap: string formatting only, no file I/O until flush.
    void record(const std::vector<float>& raw_q,
                const float* meas_q, const float* meas_dq,
                const float quat[4], const float acc[3],
                float alpha, bool trig_joint, bool trig_tilt, bool trig_fall,
                const float cmd[3], const float ach_vel[3], const float phase[2])
    {
        if (base_.empty()) return;
        long this_tick = tick_++;
        // alpha > 0 covers the hold ramp-down tail (up to H1_2_RAMP_CYCLES ticks after
        // trig_joint/trig_tilt/trig_fall all clear) -- without it, decimation could drop
        // rows from that tail and undercount engaged_ticks/engaged_time_s downstream.
        bool trig = trig_joint || trig_tilt || trig_fall || alpha > 0.0f;
        if (!trig && (this_tick % log_every_) != 0) return;  // decimated quiet tick, skip
        buf_ << (this_tick * dt_);
        for (int i = 0; i < n_; i++) buf_ << ',' << raw_q[i];
        for (int i = 0; i < n_; i++) buf_ << ',' << meas_q[i];
        for (int i = 0; i < n_; i++) buf_ << ',' << meas_dq[i];
        buf_ << ',' << quat[0] << ',' << quat[1] << ',' << quat[2] << ',' << quat[3];
        buf_ << ',' << acc[0]  << ',' << acc[1]  << ',' << acc[2];
        buf_ << ',' << cmd[0] << ',' << cmd[1] << ',' << cmd[2];
        buf_ << ',' << ach_vel[0] << ',' << ach_vel[1] << ',' << ach_vel[2];
        buf_ << ',' << phase[0] << ',' << phase[1];
        buf_ << ',' << alpha << ',' << (trig_joint?1:0) << ',' << (trig_tilt?1:0)
             << ',' << (trig_fall?1:0) << ',' << g_safety_logger_entry << '\n';
        // Periodic flush as crash insurance (~5 s at the quiet-tick 500 Hz logged rate; more
        // often during a safety-trigger burst, since those log every tick). Infrequent
        // blocking write.
        if (++rows_since_flush_ >= 2500) flush();
    }

    // Append the RAM buffer to the CSV and clear it. Called periodically and on exit.
    void flush()
    {
        if (base_.empty() || buf_.tellp() <= 0) return;
        std::ofstream(base_ + ".csv", std::ios::app) << buf_.str();
        buf_.str("");
        rows_since_flush_ = 0;
    }

private:
    std::string base_;
    int n_{0};
    float dt_{0.001f};
    long tick_{0};
    int log_every_{1};
    std::ostringstream buf_;
    int rows_since_flush_{0};
};
