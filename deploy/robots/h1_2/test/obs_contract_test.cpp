// Unit test for the A0 load-time I/O contract (include/obs_contract.h, 2026-09-15).
// Header-only and dependency-free, like adapt_encoder_test.cpp:
//
//   cd deploy/robots/h1_2
//   g++ -std=c++17 test/obs_contract_test.cpp -Iinclude -o <somewhere OUTSIDE build/> && run it
//
// (Never build into build/ -- that is the hardware build dir.)
//
// The shipped A0 export declares input `obs` [1, 92] and output `actions` [1, 27]; the cases
// below are that contract and each way a hand-copied file can break it.
#include "obs_contract.h"

#include <iostream>

static int fails = 0;
static void check(bool ok, const std::string& what)
{
    std::cout << (ok ? "  PASS  " : "  FAIL  ") << what << "\n";
    if (!ok) fails++;
}
static bool has(const std::string& s, const std::string& needle)
{
    return s.find(needle) != std::string::npos;
}

int main()
{
    using h1_2::io_contract_violation;
    const h1_2::ObsMap obs92{{"obs", std::vector<float>(92)}};
    const std::vector<h1_2::OnnxTensor> in92{{"obs", 92}};

    std::cout << "A0 I/O contract\n";

    check(io_contract_violation(in92, obs92, 27, 27).empty(),
          "the shipped A0 contract (obs 92 -> actions 27) passes");

    // The case the header exists for: a model wider than the built vector.
    {
        const auto v = io_contract_violation({{"obs", 99}}, obs92, 27, 27);
        check(!v.empty(), "a 99-wide model against a 92-float obs is refused");
        check(has(v, "ADJACENT HEAP") && has(v, "7 floats"),
              "  ...and the message names the heap read and its size (7)");
    }

    // Narrower model: no heap read, but the policy sees a truncated input. Still wrong.
    {
        const auto v = io_contract_violation({{"obs", 89}}, obs92, 27, 27);
        check(!v.empty(), "an 89-wide model against a 92-float obs is refused");
        check(has(v, "drop the last 3"), "  ...and the message says what is dropped (3)");
    }

    // An input name with no matching group. OrtRunner::act THROWS this from the policy thread.
    {
        const auto v = io_contract_violation({{"policy", 92}}, obs92, 27, 27);
        check(!v.empty(), "an input named for a group the yaml never builds is refused");
        check(has(v, "'policy'") && has(v, "built: obs"),
              "  ...and the message names both the wanted and the built groups");
    }

    // Output side: process_action slices past the end of a short action vector.
    check(!io_contract_violation(in92, obs92, 12, 27).empty(),
          "a 12-action model (legs only) against a 27-joint action manager is refused");
    check(!io_contract_violation(in92, obs92, 29, 27).empty(),
          "a 29-action model is refused too (a mismatch in either direction)");

    // Multi-input models are checked input by input, not only the first.
    {
        const h1_2::ObsMap two{{"obs", std::vector<float>(92)}, {"hist", std::vector<float>(10)}};
        check(io_contract_violation({{"obs", 92}, {"hist", 10}}, two, 27, 27).empty(),
              "a matching two-input model passes");
        check(!io_contract_violation({{"obs", 92}, {"hist", 11}}, two, 27, 27).empty(),
              "a mismatch on the SECOND input is still caught");
    }

    check(!io_contract_violation({}, obs92, 27, 27).empty(), "a model with no inputs is refused");

    std::cout << (fails ? "\nFAILED " : "\nall passed, ") << fails << " failure(s)\n";
    return fails ? 1 : 0;
}
