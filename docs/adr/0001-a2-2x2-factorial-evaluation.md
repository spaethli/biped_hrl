# A2 evaluated as a 2×2 factorial (architecture × DR width)

**Status:** accepted (2026-06-24)

To attribute A2's effect cleanly we evaluate it as a 2×2 factorial — architecture
{A1 = HIRO no encoder, A2 = HIRO + A-RMA} × DR width {narrow = A1's current set, wide
= + per-joint motor-strength + damping} — rather than the simpler A2-vs-A1-as-is
comparison. This costs four training configs (A1 retrained @ wide DR, A2 @ narrow DR,
plus the two anchors) but is the only design that separates the **DR-width main
effect** from the **RMA-mechanism main effect** and exposes the **interaction** (does
RMA help *more* as DR widens?), which is A2's headline thesis claim.

## Considered Options

- **A2 vs A1-as-is** (rejected): bundles "more DR" and "RMA" into one delta — can't say
  which caused any improvement.
- **z-zeroed ablation** (rejected): A1 never saw the wide DR, so a z-zeroed A2 differs
  from A1 in DR exposure too — not a clean mechanism test.
- **Single blind-under-wide-DR control** (subsumed): this is cell 2 of the 2×2; the
  full grid adds the narrow-DR row for the interaction term at modest extra cost.

## Consequences

- A2's encoder `e_t` must cover the **full** parameter set in every cell; in narrow-DR
  cells the un-randomized dims sit at nominal so the network is identical across cells
  (only DR ranges differ).
- All four cells are trained to **equal total environment steps** (A2 spends Phase-1 +
  Phase-3; A1 gets a matching budget) and evaluated on a **held-out wide-DR** set
  (`fall_rate` + err_vx/vy/yaw, ≥2 seeds) — robustness measured vs unseen randomization.
