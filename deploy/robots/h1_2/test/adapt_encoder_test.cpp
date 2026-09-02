// WP5d unit test for the H-adapt deploy plumbing that can be exercised without a robot:
// the history ring buffer, the flattened window's ORDER, and the latent decode + sanity
// clamp. Header-only and dependency-free, so unlike the other two tests here it needs no
// controller lib, no DDS and no onnxruntime:
//
//   cd deploy/robots/h1_2
//   g++ -std=c++17 test/adapt_encoder_test.cpp -Iinclude -o <somewhere OUTSIDE build/> && run it
//
// (Never build into build/ -- that is the hardware build dir.)
//
// What it pins, and why each one can actually fail:
//  * window() refuses a partial buffer  -> the cold-start invariant. If it returned a
//    zero-padded window instead, phi would silently run on an input shape it never saw in
//    training and emit a confident, arbitrary z_hat.
//  * oldest-first order across a wrap    -> a reversed or rotated time axis is a perfectly
//    valid tensor of the right size that returns garbage; no length check can see it.
//  * push() splices policy ++ command    -> the A1 HL column order (command LAST). The A0
//    flat observation is the same 92 floats with command mid-vector.
//  * reset() on re-entry                 -> a buffer carried across an FSM exit describes a
//    robot that was doing something else, usually falling.
//  * decode + clamp                      -> a wrong z_hat must never be OUT-OF-DISTRIBUTION
//    wrong, and the clamp count is the readout that says phi does not transfer.
#include "hrl/adapt_encoder.h"

#include <cmath>
#include <iostream>

static int fails = 0;
static void check(bool ok, const std::string& what)
{
    std::cout << (ok ? "  PASS  " : "  FAIL  ") << what << "\n";
    if (!ok) fails++;
}

// Frame f = [f, f+0.1, ..., f+0.1*(D-1)], split policy(D-1) ++ command(1) so the test also
// covers the two-part splice the caller actually uses.
static void push_frame(hrl::HistoryBuffer& h, int D, float f)
{
    std::vector<float> policy, command;
    for (int d = 0; d < D - 1; ++d) policy.push_back(f + 0.1f * d);
    command.push_back(f + 0.1f * (D - 1));
    h.push(policy, command);
}

int main()
{
    const int H = 4, D = 3;
    hrl::HistoryBuffer h;
    h.configure(H, D);
    std::vector<float> win;

    check(!h.full() && h.size() == 0, "fresh buffer is empty");
    check(!h.window(win), "window() refuses an empty buffer");

    for (int i = 0; i < H - 1; ++i) push_frame(h, D, (float)i);
    check(!h.full() && h.size() == H - 1, "partial buffer reports its fill");
    check(!h.window(win), "window() refuses a PARTIAL buffer (cold-start invariant)");

    push_frame(h, D, (float)(H - 1));
    check(h.full(), "buffer full after H pushes");
    check(h.window(win), "window() succeeds once full");
    check((int)win.size() == H * D, "window is H*D floats");
    {
        bool ok = true;
        for (int t = 0; t < H; ++t)
            for (int d = 0; d < D; ++d)
                ok &= std::fabs(win[t * D + d] - ((float)t + 0.1f * d)) < 1e-6f;
        check(ok, "row-major [1,H,D], OLDEST frame first, before any wrap");
    }
    {   // the splice: last column of each row is the command piece
        bool ok = true;
        for (int t = 0; t < H; ++t) ok &= std::fabs(win[t * D + (D - 1)] - ((float)t + 0.1f * (D - 1))) < 1e-6f;
        check(ok, "push() splices policy ++ command, command LAST");
    }

    // Wrap twice: the window must still start at the oldest surviving frame.
    for (int i = H; i < H + 2 * H + 1; ++i) push_frame(h, D, (float)i);
    check(h.window(win), "window() still succeeds after wrapping");
    {
        const float oldest = (float)(H + 2 * H + 1 - H);
        bool ok = true;
        for (int t = 0; t < H; ++t)
            for (int d = 0; d < D; ++d)
                ok &= std::fabs(win[t * D + d] - (oldest + (float)t + 0.1f * d)) < 1e-6f;
        check(ok, "after a wrap the window is the last H pushes, oldest first");
    }

    h.reset();
    check(!h.full() && h.size() == 0 && !h.window(win),
          "reset() empties it -> an FSM re-entry cannot reuse pre-exit history");

    // A frame of the wrong total length is dropped rather than mis-spliced.
    {
        hrl::HistoryBuffer g;
        g.configure(2, 3);
        g.push({1.0f, 2.0f}, {3.0f, 4.0f});   // 4 != 3
        check(g.size() == 0, "push() drops a wrong-length frame instead of mis-splicing it");
    }

    // --- metadata contract -----------------------------------------------------------
    hrl::AdaptEncoderMeta m;
    m.history_len = 50; m.input_dim = 92; m.z_dim = 5;
    m.time_order = "oldest_first";
    m.input_layout = "policy89+command3";
    m.z_names = "payload_kg,com_dx,com_dy,com_dz,friction";
    m.scale = {1, 1, 1, 1, 1};
    m.center = {0, 0, 0, 0, 0};
    m.clip_lo = {0.0f, -0.05f, -0.05f, -0.05f, 0.3f};
    m.clip_hi = {12.0f, 0.05f, 0.05f, 0.05f, 1.6f};
    m.cold = {0.0f, 0.0f, 0.0f, 0.0f, 0.95f};
    check(m.validate().empty(), "a complete metadata block validates");
    {
        auto bad = m; bad.time_order = "newest_first";
        check(!bad.validate().empty(), "a reversed time axis is REFUSED at load");
        bad = m; bad.input_layout = "";
        check(!bad.validate().empty(), "a missing layout string is REFUSED");
        bad = m; bad.clip_hi = {12.0f, 0.05f, 0.05f, 0.05f};
        check(!bad.validate().empty(), "a short bound vector is REFUSED");
        bad = m; bad.cold = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        check(!bad.validate().empty(),
              "z_cold outside the clamp is REFUSED (friction 0.0 is a FRICTIONLESS FLOOR, "
              "off-distribution -- the trap a zero-vector 'nominal' walks into)");
    }

    // --- decode + clamp --------------------------------------------------------------
    std::vector<float> z;
    check(hrl::decode_latent(m, {5.0f, 0.01f, 0.0f, 0.0f, 0.8f}, z) == 0
          && std::fabs(z[0] - 5.0f) < 1e-6f && std::fabs(z[4] - 0.8f) < 1e-6f,
          "in-range latent passes through unclamped");
    check(hrl::decode_latent(m, {99.0f, 0.0f, 0.0f, 0.0f, -3.0f}, z) == 2
          && std::fabs(z[0] - 12.0f) < 1e-6f && std::fabs(z[4] - 0.3f) < 1e-6f,
          "out-of-range components are clamped to the training support and COUNTED");
    {   // a normalised export: z_hat = center + scale*raw must be applied, not assumed away
        auto n = m;
        n.center = {6.0f, 0, 0, 0, 0.95f};
        n.scale = {6.0f, 0.05f, 0.05f, 0.05f, 0.65f};
        hrl::decode_latent(n, {1.0f, 0.0f, 0.0f, 0.0f, -1.0f}, z);
        check(std::fabs(z[0] - 12.0f) < 1e-6f && std::fabs(z[4] - 0.3f) < 1e-6f,
              "z_center/z_scale are applied (a normalised phi un-normalises correctly)");
    }

    // --- the load-time deploy contract (the highest-severity item) ---------------------
  // algorithms.h sizes the ORT input tensor from the ONNX declared shape and NEVER compares
  // it with the vector that was built, so every disagreement below would otherwise be read
  // as adjacent HEAP -> erratic actions -> the safety hold ramps alpha to 1 -> PD error 0 ->
  // damping-only torque, i.e. a LIMP robot, with a symptom pointing at actuation.
  {
    const int POL = 89, CMD = 3, H = 50, D = 92, Z = 5;
    const long long IN = (long long)H * D;
    check(hrl::check_deploy_contract(m, POL, CMD, IN, Z).empty(),
          "a matching export is accepted");
    check(!hrl::check_deploy_contract(m, POL, CMD, IN - D, Z).empty(),
          "REFUSED: the ONNX accepts H-1 frames while the metadata claims H");
    check(!hrl::check_deploy_contract(m, POL, CMD, IN, Z + 1).empty(),
          "REFUSED: the ONNX emits a wider latent than z_dim");
    {
      auto bad = m; bad.input_layout = "policy6+command86";
      check(!hrl::check_deploy_contract(bad, POL, CMD, IN, Z).empty(),
            "REFUSED: the A0 flat column order -- SAME 92 floats, command mid-vector, so "
            "only the layout string can catch it");
    }
    {
      auto bad = m; bad.input_dim = 91;
      check(!hrl::check_deploy_contract(bad, POL, CMD, (long long)H * 91, Z).empty(),
            "REFUSED: a frame width that is not policy+command");
    }
    {
      auto bad = m; bad.time_order = "newest_first";
      check(!hrl::check_deploy_contract(bad, POL, CMD, IN, Z).empty(),
            "REFUSED: a reversed time axis (a valid tensor that returns garbage)");
    }
    check(!hrl::check_deploy_contract(m, 47, CMD, IN, Z).empty(),
          "REFUSED: a deploy whose own policy group is a different width than phi trained on");
  }

  std::cout << (fails ? "FAILURES: " : "ALL PASS (") << fails << ")\n";
    return fails ? 1 : 0;
}
