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
    static constexpr size_t FLUSH_ROWS = 512;  // ~10 s at 50 Hz

    // base: output path base (H1_2_SAFETY_LOG); writes <base>_hrl.csv. goal_dim sizes
    // the s/target columns. Disabled (no-op) until init() is called.
    void init(const std::string& base, int goal_dim)
    {
        path_ = base + "_hrl.csv";
        goal_dim_ = goal_dim;
        rows_.clear();
        rows_.reserve(FLUSH_ROWS);
        FILE* f = std::fopen(path_.c_str(), "w");
        if (!f) return;
        std::fprintf(f, "t,cmd_vx,cmd_vy,cmd_wz");
        for (int i = 0; i < goal_dim_; ++i) std::fprintf(f, ",s%d", i);
        for (int i = 0; i < goal_dim_; ++i) std::fprintf(f, ",tgt%d", i);
        std::fprintf(f, ",period,hip_pitch_l,hip_pitch_r,act_rate\n");
        std::fclose(f);
        enabled_ = true;
    }

    bool enabled() const { return enabled_; }

    void record(float t, const float* cmd, const Eigen::VectorXf& s,
                const Eigen::VectorXf& target, float period,
                float hip_pitch_l, float hip_pitch_r, float act_rate)
    {
        if (!enabled_) return;
        std::vector<float> row;
        row.reserve(7 + 2 * goal_dim_);
        row.push_back(t);
        row.insert(row.end(), cmd, cmd + 3);
        row.insert(row.end(), s.data(), s.data() + goal_dim_);
        row.insert(row.end(), target.data(), target.data() + goal_dim_);
        row.push_back(period);
        row.push_back(hip_pitch_l);
        row.push_back(hip_pitch_r);
        row.push_back(act_rate);
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
    std::vector<std::vector<float>> rows_;
};

} // namespace hrl
