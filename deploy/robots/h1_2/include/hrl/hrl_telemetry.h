// HRL deploy-gate telemetry (plan W3) — buffered per-policy-step CSV from State_RLHRL.
// Logs what the safety flight recorder (1 kHz, joint-space) cannot: the command, the
// goal-space state estimate s, the held target V*, the commanded stride period, and a
// hip-pitch pair for the analyzer's stride-period proxy. 50 Hz x ~20 floats — RAM
// buffered, appended to disk every FLUSH_ROWS (so a Ctrl-C kill loses <=10 s, not the
// session; lesson from 2026-07-15). Consumed by scripts/deploy_gate_analyzer.py.
#pragma once

#include <eigen3/Eigen/Dense>
#include <cstdio>
#include <string>
#include <vector>

namespace hrl
{

class Telemetry
{
public:
    // ~2.5 s at 50 Hz -- was 512 (~10 s), matching the safety_logger's flush cadence.
    // 2026-07-17 finding: a session that ended abruptly (not a clean FSM exit) lost the
    // final ~2-10 s of _hrl.csv telemetry -- including, in one case, the actual fall --
    // while the safety CSV's tighter cadence still had it. exit() calls flush() too, so
    // this only matters for non-clean process ends (crash/kill), which is exactly when
    // the data matters most.
    static constexpr size_t FLUSH_ROWS = 128;

    // base: output path base (H1_2_SAFETY_LOG); writes <base>_hrl.csv. goal_dim sizes
    // the s/target columns. Disabled (no-op) until init() is called. Called once per
    // STATE ENTRY, not once per process: only the first entry this process truncates/
    // writes the header; later re-entries append, so an earlier attempt (e.g. one that
    // fell) survives a later successful attempt in the same session instead of being
    // silently overwritten (2026-07-17 finding). Each row also carries an `entry` id,
    // since `t` alone (only advances while actively in this state) won't show a gap
    // between separate attempts.
    // z_dim > 0 (WP5d) appends the H-adapt latent columns; 0 keeps the pre-WP5d header.
    void init(const std::string& base, int goal_dim, int z_dim = 0)
    {
        bool first_entry = path_.empty() || path_ != base + "_hrl.csv";
        path_ = base + "_hrl.csv";
        goal_dim_ = goal_dim;
        z_dim_ = z_dim;
        rows_.clear();
        rows_.reserve(FLUSH_ROWS);
        entry_++;
        enabled_ = true;
        if (!first_entry) return;  // re-entry: keep the existing file, just bump entry_
        FILE* f = std::fopen(path_.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "t,cmd_vx,cmd_vy,cmd_wz");
        for (int i = 0; i < goal_dim_; ++i) std::fprintf(f, ",s%d", i);
        for (int i = 0; i < goal_dim_; ++i) std::fprintf(f, ",tgt%d", i);
        std::fprintf(f, ",period,hip_pitch_l,hip_pitch_r,act_rate,entry");
        // Deployable-estimator regression guard (2026-08-03). est_* is what the real robot
        // can compute for itself (IMU-integrated velocity increment + leg-FK height);
        // gt_* is the sim bridge's SportModeState ground truth. Logged side by side EVEN
        // WHEN the estimator is not feeding the policy (hrl.base_vel_from_imu /
        // base_height_from_fk absent), so any bridge session doubles as a regression check.
        // est_vx/est_vy and gt_vx/gt_vy are WITHIN-WINDOW INCREMENTS of the pelvis-frame
        // base velocity (the quantity `delta` mode actually needs), reset at every HL
        // window start; est_h/gt_h are absolute heights. On real hardware gt_* read 0
        // (rt/sportmodestate goes silent) — only a bridge run scores this pair.
        std::fprintf(f, ",est_vx,est_vy,est_h,gt_vx,gt_vy,gt_h");
        // Leg odometry (2026-08-06), the HL's ABSOLUTE velocity. Logged UNCONDITIONALLY,
        // like the pair above, so any bridge session scores it whether or not it is feeding
        // the HL -- which is what makes the open-loop rung measurable without a special
        // build. lo_* is the window-latched c-average (held constant across the window, so
        // score it on FIRE rows: the est_vx==0 rows, whose modal gap is c). gt_a* is the
        // matching ABSOLUTE pelvis-frame ground truth -- gt_vx/gt_vy above are within-window
        // INCREMENTS and cannot score an absolute estimator. Both read 0 on real hardware.
        std::fprintf(f, ",lo_vx,lo_vy,gt_avx,gt_avy");
        // WP5d: what the HL actually read for the env latent, per fire, plus whether it was
        // an estimate at all (z_valid 0 = still on the cold-start prior) and how many
        // components the sanity clamp bit. With the payload KNOWN per WP7 cell, these columns
        // ARE the mechanism check -- and they make "phi does not transfer" a reading in the
        // data rather than an inference from the robot's behaviour.
        for (int i = 0; i < z_dim_; ++i) std::fprintf(f, ",z%d", i);
        if (z_dim_) std::fprintf(f, ",z_valid,z_clipped");
        std::fprintf(f, "\n");
        std::fclose(f);
    }

    bool enabled() const { return enabled_; }

    // est/gt: {vx, vy, h}; lo: {lo_vx, lo_vy, gt_avx, gt_avy} — see the header note in init().
    // z/z_valid/z_clipped are ignored unless init() was given z_dim > 0.
    void record(float t, const float* cmd, const Eigen::VectorXf& s,
                const Eigen::VectorXf& target, float period,
                float hip_pitch_l, float hip_pitch_r, float act_rate,
                const float est[3], const float gt[3], const float lo[4],
                const float* z = nullptr, bool z_valid = false, int z_clipped = 0)
    {
        if (!enabled_) return;
        std::vector<float> row;
        row.reserve(18 + 2 * goal_dim_);
        row.push_back(t);
        row.insert(row.end(), cmd, cmd + 3);
        row.insert(row.end(), s.data(), s.data() + goal_dim_);
        row.insert(row.end(), target.data(), target.data() + goal_dim_);
        row.push_back(period);
        row.push_back(hip_pitch_l);
        row.push_back(hip_pitch_r);
        row.push_back(act_rate);
        row.push_back((float)entry_);
        row.insert(row.end(), est, est + 3);
        row.insert(row.end(), gt, gt + 3);
        row.insert(row.end(), lo, lo + 4);
        if (z_dim_) {
            for (int i = 0; i < z_dim_; ++i) row.push_back(z ? z[i] : 0.0f);
            row.push_back(z_valid ? 1.0f : 0.0f);
            row.push_back((float)z_clipped);
        }
        rows_.push_back(std::move(row));
        if (rows_.size() >= FLUSH_ROWS) flush();
    }

    void flush()
    {
        if (!enabled_ || rows_.empty()) return;
        FILE* f = std::fopen(path_.c_str(), "a");
        if (!f) return;
        for (const auto& r : rows_) {
            for (size_t i = 0; i < r.size(); ++i)
                std::fprintf(f, i ? ",%.6g" : "%.6g", r[i]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        rows_.clear();
    }

private:
    bool enabled_{false};
    std::string path_;
    int goal_dim_{0};
    int z_dim_{0};
    int entry_{-1};  // pre-incremented in init() -> first entry logs as 0
    std::vector<std::vector<float>> rows_;
};

} // namespace hrl
