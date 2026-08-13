# All base-velocity estimators run on-robot in parallel, selected fail-closed

**Status:** accepted (2026-08-13). Implements §13 of `doc/hrl/h1_2_ekf_design.md`.

Seven base-velocity estimator arms were scored offline against hardware logs (WL-G). Taking
them to the robot could either deploy one arm per session or run all of them at once. We run
**all seven every control tick, log all seven, and let exactly one — named by a required
`base_estimator:` config key — feed `obs["hl_vel"]`.** A missing or unknown key aborts at
startup rather than falling back to a default.

## Considered options

**One arm per session** (the literal reading of "3 runs each") would need nine or more
sessions, and each arm would see different sensor data under its own per-restart encoder
offset — reintroducing exactly the confounds the offline replay design exists to remove.
Running all arms in parallel gives every arm byte-identical input in three sessions, and the
selected arm still exercises the closed loop.

**Defaulting the selector to `legodom`** would keep every existing deploy config booting
untouched. We rejected it: a missing key silently selecting a signal is the precise failure
mode that splayed the keeper on 2026-08-05, where absent `base_vel_from_imu` /
`base_height_from_fk` keys routed the goal state to a dead `sportmodestate`. No bridge gate
catches that class, so the fix has to be startup validation that refuses to run.

## Consequences

* **Every HRL deploy config must carry the key or the controller will not start.** That is the
  intent, not an oversight. All configs under `config/policy/velocity_hrl/v0/params/` are
  updated in the same change; any config outside that set will fail loudly.
* **A0 is exempt.** Its observation vector has no base linear velocity term, so on `State_RLBase`
  every arm is passive and no key is required.
* **A known-divergent arm runs inside the ~1 kHz loop.** Arm C diverges on real data (72.8° tilt,
  6.3 m drift). It is safe only because arms are fully isolated: a passive arm reaches the flight
  recorder and nothing else — never the selected estimate, `lowcmd`, or the safety filter. That
  isolation is a load-bearing invariant, not an implementation detail, and it is asserted by test
  T6 in §13.5.
* **Divergence is logged raw, never reset**, so the failure stays visible as evidence. Downstream
  consumers must tolerate NaN columns.
* Prediction runs every tick at a fixed 1 ms, but the kinematic **update fires only when the DDS
  sample actually changes**. Updating every tick would count each measurement twice, shrink the
  covariance about twice as fast as it should, and bias the filter toward the stationary-foot
  assumption — making the deployed arms measurably more zero-biased than the arms that were
  scored. See §13.2.
