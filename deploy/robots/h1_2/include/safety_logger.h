#pragma once
// [SAFETY FILTER] flight recorder — delete this file when reverting.
//
// Two-file output (base path from H1_2_SAFETY_LOG):
//   <base>.csv        standard CSV, one header row + one row per control tick
//   <base>_meta.json  static config written once at init (limits, names, dt)
//
// Hot-path safe: record() only appends a formatted line to an in-RAM string
// buffer (no syscall). The file is written on periodic flush (~5 s) and on exit.

#include <array>
#include <fstream>
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
                float alpha, bool trig_joint, bool trig_tilt, bool trig_fall)
    {
        if (base_.empty()) return;
        buf_ << (tick_++ * dt_);
        for (int i = 0; i < n_; i++) buf_ << ',' << raw_q[i];
        for (int i = 0; i < n_; i++) buf_ << ',' << meas_q[i];
        for (int i = 0; i < n_; i++) buf_ << ',' << meas_dq[i];
        buf_ << ',' << quat[0] << ',' << quat[1] << ',' << quat[2] << ',' << quat[3];
        buf_ << ',' << acc[0]  << ',' << acc[1]  << ',' << acc[2];
        buf_ << ',' << alpha << ',' << (trig_joint?1:0) << ',' << (trig_tilt?1:0)
             << ',' << (trig_fall?1:0) << ',' << g_safety_logger_entry << '\n';
        // Periodic flush as crash insurance (~5 s at 500 Hz). Infrequent blocking write.
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
    std::ostringstream buf_;
    int rows_since_flush_{0};
};
