# Motion-capture alignment: method, maths, and the marker measurement protocol

How `scripts/mocap_align.py` turns a Vicon take into ground-truth pelvis velocity for one
hardware Run, how its accuracy is checked, and how to measure a marker layout for the optional
IMU-free **geometric frame**. Terms (Run, Session pair, Motion-capture ground truth) are in
`CONTEXT.md`; the chronology of how this method came about is in the engineering journal.

## Commands

```bash
# ground truth for one bundled Run (gyro frame, the default)
python scripts/mocap_align.py <bundle> <bundle>/mocap.csv --lever 0 0 0.32

# + the geometric frame check (raw C3D of the SAME take, a marker template)
python scripts/mocap_align.py <bundle> <bundle>/mocap.csv --lever 0 0 0.32 \
    --c3d <take>.c3d --marker-template doc/hrl/mocap/h1_2_torso_marker_template.json
#   --frame geometric   expresses the output in the template frame instead (IMU-free)

# build a template from tape measurements (protocol below)
python scripts/mocap_marker_template.py doc/hrl/mocap/h1_2_torso_markers_tape.json \
    <take1>.c3d <take2>.c3d ... -o doc/hrl/mocap/h1_2_torso_marker_template.json

python scripts/mocap_align.py --selftest      # clock, dropouts, frame, lever, geometric frame
```

The C3D reader is the optional `c3d` package (`pip install -e '.[mocap]'`); nothing else needs it.

### Full hardware-session workflow

```bash
# 1. scan the new date (dashes), then confirm labels
python scripts/bundle_hardware_run.py --scan YYYY-MM-DD

# 2. bundle; optional per-Run key "trajectory": "<all_joints stem>" when the knee
#    correlation cannot pair the telemetry
python scripts/bundle_hardware_run.py --labels labels.json

# 3. mocap, per bundle. The bundler does NOT copy the Vicon export: copy the take
#    (take-to-Run map in the capture session's Read_me) in as mocap.csv first
cp <capture_session>/Tn.csv logs/robot_logs/<bundle>/mocap.csv
python scripts/mocap_align.py logs/robot_logs/<bundle>/ logs/robot_logs/<bundle>/mocap.csv \
    --lever 0 0 0.32 \
    --c3d <capture_session>/Tn.c3d --marker-template doc/hrl/mocap/h1_2_torso_marker_template.json
#    (--c3d/--marker-template optional); then read run.json -> mocap_align.flags

# 4. score; picks up mocap_aligned.csv beside the flight recorder, so run AFTER step 3
#    (re-score if mocap is added later; --no-mocap scores against the estimator)
python scripts/bench_flight_recorder.py logs/robot_logs/<bundle>/<flight>.csv --robot \
    --out-dir logs/robot_logs/<bundle>/ --traj-dir logs/robot_logs/<bundle>/

# 5. cadence/settling over both sessions; bundle dirs use underscores (YYYY_MM_DD-*).
#    UNQUOTED: this script takes shell-expanded paths (a quoted glob matches nothing and
#    writes an empty json); plot_thesis_figures.py below takes quoted globs
python scripts/analyze_cadence_settling.py \
    --bundles logs/robot_logs/2026_09_14-*/ logs/robot_logs/<YYYY_MM_DD>-*/ \
    --out data/<new-session>/cadence_settling.json

# 6. figures over both sessions; --cadence-json is required (its default is the 09-14 file)
python scripts/plot_thesis_figures.py \
    --bundles "logs/robot_logs/2026_09_14-*/" "logs/robot_logs/<YYYY_MM_DD>-*/" \
    --cadence-json data/<new-session>/cadence_settling.json
```

Quick check of a new session before mocap is ready: run steps 4 and 6 only, with the new
date's glob alone. `f_track`, `f_hier`, `f_power` and `f_smooth` do not need the cadence json;
`f_cadence`, `f_settle` and the residual-speed line of `f_stand` need steps 3 and 5. Raw
flight recorder files that hold several Runs need `--entry` in step 4; bundles are already one
Run each.

## Frames and signals

| Symbol | Meaning | Source |
|---|---|---|
| $W$ | Vicon world | |
| $M$ | Vicon rigid body of the marker cluster (re-defined per take) | export |
| $T$ | torso / IMU frame, $x$ forward, $y$ left, $z$ up | `est_gyro`, `acc_*` |
| $P$ | pelvis frame, $R_{PT} = R_z(\psi)$, same origin as $T$ | `h1_2_kinematics.md` |
| $G$ | geometric torso frame of a marker template | template JSON |
| $R_{WM}(t),\ p_W(t)$ | cluster attitude and origin | export `q*`, `*_mm` |
| $\omega^W_{M}(t)$ | exported angular rate, **in the world frame** | export `w*_deg_s` |
| $\omega_g(t),\ f(t)$ | gyro and accelerometer, frame $T$ | flight recorder |
| $\psi(t)$ | waist yaw | `meas_q12` |
| $x_k(t)$ | raw position of marker $k$ | C3D |

The unknowns are a clock map (mocap time to `t_wall`), the constant rotation $R_{MT}$ between
torso and rigid body, and the lever arm $r$ (pelvis origin to cluster origin, frame $T$), which
is measured, not fitted. Every alignment signal is an angular one: fitting on the linear
velocity that is being scored would absorb the estimator's own lag and cross-axis error.

## 1. Clean samples

A mocap row is refused if any pose or rate value is non-finite or `markers_used < 3`. A hole is
a time step $\Delta t_i > 1.5\,\mathrm{median}(\Delta t)$. With $\mathcal{B}$ the set of refused
times and hole intervals, a row is clean if

$$C(t) = \big[\ \mathrm{dist}(t, \mathcal{B}) > \varepsilon\ \big],\qquad \varepsilon = 0.2\ \mathrm{s}.$$

Nothing interpolates across a hole; ground truth is NaN wherever the aligned flight row does not
fall on clean mocap.

## 2. Time: offset and rate from $|\omega|$

The magnitude is invariant to the unknown rotation, so both sides reduce to scalars,
$s_m(t) = \lVert R_{WM}^\top \omega^W_M \rVert$ and $s_f(t) = \lVert \omega_g \rVert$, each smoothed
by a 20 ms boxcar. The flight recorder's clock is **`t_wall`**, never `t` (a tick counter: 0.27-1 %
slow and bent by up to 1.76 s). The map is

$$t_\text{wall} = t_m + b + m\,t_m .$$

**Whole-take offset.** On a 20 ms grid, with mocap mask $c_k = C(t_k)$ and flight-overlap mask
$v_{k+L}$, the masked normalized cross-correlation over every lag $L$ is

$$\rho(L) = \frac{\sum_k w_k (A_k - \bar A_L)(F_{k+L} - \bar F_L)}
{\sqrt{\sum_k w_k (A_k - \bar A_L)^2 \sum_k w_k (F_{k+L} - \bar F_L)^2}},
\qquad w_k = c_k\, v_{k+L},$$

with $\bar A_L, \bar F_L$ the $w$-weighted means (all sums by FFT). Lags whose overlap
$\sum_k w_k$ is below $\min(20\,\mathrm{s},\ 0.8\,T_\text{clean},\ 0.8\,T_\text{slice})$ are
excluded, and $L^* = \arg\max \rho$.

**Per-window lags.** Each clean stretch is cut into windows $j$ of at most 20 s. Each finds
$\ell_j = \arg\max_\ell \rho_j(\ell)$ on a 5 ms grid within $L^* \pm 1$ s, refined by a parabola
through the peak and its neighbours:

$$\hat\ell_j = \ell_k + \frac{\Delta}{2}\,\frac{\rho_{k-1} - \rho_{k+1}}{\rho_{k-1} - 2\rho_k + \rho_{k+1}} .$$

A window votes if $\rho_j \ge 0.8$ and is kept if $|\hat\ell_j - \mathrm{median}(\hat\ell)| \le 50$ ms
(confident wrong answers sit one standing step period, about 0.9 s, away). With at least 3 kept
windows spanning more than 20 s, $(b, m)$ is the weighted least-squares line through
$(\tau_j, \hat\ell_j)$ with weights $\sqrt{T_j}$; otherwise $m = 0$ and $b$ is the weighted mean.

## 3. Frame: orthogonal Procrustes on $\omega$

With both rate vectors smoothed (100 ms boxcar, each on its own continuous grid) and sampled on
the clean aligned flight rows, $w_i = R_{WM}^\top \omega^W_M$ (frame $M$) and $g_i = \omega_g$
(frame $T$):

$$R_{MT} = \arg\min_{R \in SO(3)} \sum_i \lVert w_i - R\,g_i \rVert^2,\qquad
\sum_i w_i g_i^\top = U \Sigma V^\top,\qquad
R_{MT} = U\,\mathrm{diag}\big(1, 1, \det(UV^\top)\big)\,V^\top .$$

The $\det$ term removes reflections. **Excitation** is $\sigma_3/\sigma_1$ (refused below 0.02):
rotation about a missing axis is unobservable. The **residual** is
$\mathrm{median}_i\ \angle(w_i, R_{MT}\,g_i)$ over samples with $\lVert w_i \rVert > 0.2$ rad/s
(refused above 15°). It is a goodness-of-fit check only: tilting a right $R$ by 10° raises it by
about 5°. The torso attitude is $R_{WT}(t) = R_{WM}(t)\,R_{MT}$.

With $\angle(R) = \lVert \log R \rVert$ the rotation angle, two accuracy checks, both written to
`run.json` and flagged (never refused):

* **Window agreement**: the same fit on each 20 s window gives $R_j$;
  $\mathrm{median}_j\ \angle(R_{MT}^\top R_j)$, flagged above 4°. Covers all three axes.
* **Gravity**: on quasi-static rows ($\lVert\omega_g\rVert < 0.15$ rad/s,
  $|\lVert f \rVert - g| < 0.3$ m/s², at least 200 rows) the accelerometer reads $+g$ along up:

  $$u_T(t) = R_{MT}^\top R_{WM}(t)^\top e_z,\qquad
  \theta_g = \angle\big(\overline{f},\ \overline{u_T}\big),$$

  flagged above 3°. It checks roll and pitch with a sensor the fit never used; it cannot see yaw.

## 4. Linear velocity, lever arm, waist yaw

Differentiated on the mocap grid (central differences), then rotated and lever-corrected:

$$v_W = \dot p_W,\qquad
\omega_T = R_{MT}^\top R_{WM}^\top \omega^W_M,\qquad
v_T = R_{WT}^\top v_W - \omega_T \times r,\qquad
v_P = R_z(\psi)\, v_T,$$

$$p_P = p_W - R_{WT}\, r,\qquad \omega_z^\text{gt} = \frac{d}{dt}\,\mathrm{yaw}(R_{WT}).$$

$v_P$ is interpolated onto the flight recorder's own rows and written keyed by its `t`. With
$r$ vertical ($r = (0,0,0.32)$ m on the H1-2), $\omega_T \times r = (\omega_y r_z, -\omega_x r_z, 0)$:
yaw rate does not leak into horizontal velocity.

**Lever check (report only).** Over zero-command stretches the pelvis origin barely moves, so
per 2 s window (both sides detrended against $[1, t]$)
$p_W - \bar p_W \approx (R_{WT} - \bar R_{WT})\, r$ is linear in $r$; least squares with a
leave-one-window-out jackknife SE. On the robot it reads $r_z$ 0.64-0.93 m (the robot rocks about
its feet) and $r_y$ with both signs across takes, so nothing is warned on.

## 5. Geometric frame (optional, IMU-free)

A template gives each marker's position $t_k$ in a torso-fixed frame $G$ ($x$ forward, $y$ left,
$z$ up), built from tape measurements (protocol below). Per C3D frame, Kabsch:

$$\bar x = \tfrac1n\sum_k x_k,\quad \bar t = \tfrac1n\sum_k t_k,\qquad
R_{WG} = \arg\min_{R\in SO(3)} \sum_k \lVert (x_k - \bar x) - R\,(t_k - \bar t) \rVert^2$$

(the same SVD as §3), with fit rms
$e = \big(\tfrac1n \sum_k \lVert R_{WG} t_k + \bar x - R_{WG}\bar t - x_k \rVert^2\big)^{1/2}$.
The cluster's Vicon pose is the same rigid body, so

$$R_{MG} = \mathrm{polar}\Big(\sum_j R_{WM}(t_j)^\top R_{WG}(t_j)\Big),\qquad
R_{TG} = R_{MT}^\top R_{MG},$$

with $\mathrm{polar}(A)$ the nearest rotation (SVD as above). $R_{TG}$, the rotation from the
geometric to the gyro frame, is the reported cross-check, and $R_{MG}$ replaces $R_{MT}$ under
`--frame geometric`. Refused if $e > 30$ mm or if the per-frame scatter
$\mathrm{median}_j\ \angle(R_{MG}^\top R_{WM}^\top R_{WG})$ exceeds 2° (wrong take, template or
convention). C3D frame $i$ maps to export row time $i/f_s$.

**Labelling.** Vicon's marker labels change between takes, so the assignment $\pi$ is recomputed
per take from the median inter-marker distances $d_{ab}$, which do not depend on labels:
$\pi^* = \arg\min_\pi \sum_{a<b} (\lVert t_a - t_b\rVert - d_{\pi(a)\pi(b)})^2$. Distances cannot
tell a layout from its mirror image, so the six best $\pi$ are re-ranked by the Kabsch fit rms $e$.

**Template construction** (`mocap_marker_template.py`). With face widths $W_f$, per-marker tape
values $(u_k, v_k, o_k)$ and the viewer convention:

$$\text{front: } t_k = \Big(\tfrac D2 + o_k,\ \ u_k - \tfrac{W_f}{2},\ \ v_k\Big),\qquad
\text{back: } t_k = \Big(-\tfrac D2 - o_k,\ \ \tfrac{W_b}{2} - u_k + \delta_y,\ \ v_k + \delta_z\Big).$$

(With the robot convention the front $y$ becomes $W_f/2 - u_k$; behind the robot both conventions
coincide.) The face separation $D$ and the back face's offsets $\delta_z, \delta_y$ cannot be
taped and are fitted to the Vicon distances, with a grid of starting points (the cost has local
minima). Diagnostics written into the template:

* **Layout gap** $\Delta_k = \mathrm{median}_t\big(R_{WG}^\top (x_k - p)\big) - t_k$, where the
  markers really are in the taped frame; per-number tape error $\sigma = \lVert\Delta\rVert_\text{rms}/\sqrt3$.
* **Frame uncertainty**: Monte Carlo over every tape number perturbed by $\mathcal N(0, \sigma^2)$,
  refit, and $\mathrm{median}\ \angle(R_{WG}^\top R'_{WG})$ against the nominal template.
* **Check distances** (e.g. front plane to lower-back plane along $x$), measured against the markers.

## 6. Grading the onboard gyro

The gyro-fitted $R_{MT}$ absorbs a constant mounting rotation and the clock absorbs a latency,
so those two are not gradable against mocap without a gyro-free frame and clock. Scale, bias and
noise are gradable. Per torso axis $a$, on clean samples, with both signals low-passed by a
0.25 s boxcar and $\tilde\omega = R_{MT}^\top R_{WM}^\top\omega^W_M$:

$$\omega_{g,a} = k_a\,\tilde\omega_a + b_a + \epsilon .$$

The ordinary fit (gyro on mocap) is biased low by any timing, filter or reference error, and the
inverse fit (mocap on gyro) is biased high, so the two bracket $k_a$. Bias and noise come from
standstill: mean and standard deviation of $\omega_g$ where
$\lVert\tilde\omega\rVert < 0.03$ rad/s for at least 1 s. The mocap reference can come from the
export or from raw markers (§5 attitudes, differentiated). Results: journal 2026-09-17.

## 7. Measuring a marker layout

Measure every marker at its **centre**, on the faces it sits on, against the torso's own edges.

1. **Decide the left/right convention before measuring and write it in the JSON.**
   `"viewer"` (recommended, used on 2026-09-17): stand facing the face you are measuring, and
   left/right are **your** left and right. On the front face your left edge is the robot's right.
   `"robot"`: left/right are the robot's own. Behind the robot the two agree. The builder scores
   both and warns if the other one fits better (2026-09-17: viewer 17 mm, mirrored 68 mm).
2. **Per face**, write its `width` (edge to edge).
3. **Per marker**, write:
   * `face`: `front` or `back`;
   * `u`: horizontal distance from the face's **left edge** (by the declared convention) to the
     marker centre, measured parallel to the bottom edge;
   * `v`: height of the marker centre above that face's **bottom edge**. If both faces can be
     measured from one common bottom reference, do so: the front/back height offset is otherwise
     fitted, and that costs several degrees;
   * `offset`: distance of the marker centre in front of the face surface, perpendicular to it.
     Include the cover thickness where the base sits on a cover.
4. **Add check distances** you can measure directly, e.g. front marker plane to lower-back marker
   plane along $x$ (2026-09-17: taped 0.270 m, markers 0.278 m).
5. **Build the template** from at least two good takes and read its output: fit rms under 30 mm
   (tape quality is about 17 mm), the declared convention clearly better than the mirror, the
   per-marker gaps, and the frame uncertainty.

**What precision buys.** Median frame error from the Monte Carlo, per tape number: 3 mm gives 2.1°,
5 mm 3.5°, 10 mm 6.8°. Errors common to a whole face (all offsets, a shared height reference)
cost about 0°, while one marker's offset wrong by 10 mm costs 4.2°. Use a ruler against a square
and measure perpendicular to the edges. The 2026-09-17 tape set is good to about 10 mm per number,
i.e. a ~7° frame, so there the gyro frame (0.3-2.4° window agreement) remains the primary one.

**Measurements the geometric frame cannot replace:**
- A marker on the pelvis origin for one take measures the lever arm $r$.
- An inclinometer on the torso top edge, read against the accelerometer on the gantry, measures
  the IMU's roll/pitch mounting.
- A hardware sync between Vicon and the robot PC gives a clock independent of the gyro, and with
  it the gyro latency.

## `run.json` → `mocap_align`

| Field | Meaning |
|---|---|
| `clock`, `lag_s`, `skew_ppm`, `global_lag_s` | clock and map $(b, m)$, whole-take $L^*$ |
| `xcorr`, `windows` (kept/voting/total), `lag_spread_ms` | time-fit quality |
| `mocap_clean_frac`, `n_samples` | clean fraction of the take, flight rows with ground truth |
| `R`, `proc_angle_deg`, `excitation` | $R_{MT}$ used for the output, residual, $\sigma_3/\sigma_1$ |
| `frame_spread_deg`, `gravity_tilt_deg`, `gravity_rows` | the two accuracy checks |
| `frame`, `geom_frame` | output frame; geometric check (fit, scatter, $R_{TG}$, accelerometer vs geometric up) |
| `lever`, `pivot` | measured $r$; lever report |
| `flags` | every flag raised; the Run is written but low quality |
