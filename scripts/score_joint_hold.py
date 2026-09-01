#!/usr/bin/env python3
"""Score how much commanded joint motion the deploy hold suppresses, and what the legs
do in response, across a SET of flight-recorder sessions.

The per-session tools next door answer different questions: `safety_analyzer.py` scores
limit violations for one log, `plot_deploy_logs.py` draws one. This one is cross-session
by construction, because the quantity of interest is an INTERACTION: whether holding the
upper body costs A1 more than it costs A0. A single session cannot express that.

Two numbers per session:

  dose      the commanded upper-body motion the hold threw away, = |raw_q - meas_q| on the
            joints named by `hold_joint_ids`. `raw_q` is the raw policy intent, recorded
            BEFORE the hold substitution (State_RLBase.cpp), so every session already
            carries its own counterfactual. Split into a BIAS part (|mean of the signed
            error|, "the policy wants a different static pose") and an OSCILLATION part
            (its std, "the policy is wobbling into a joint that never moves"). The split
            matters: A0's suppressed intent is ~pure bias, A1's is ~pure oscillation, and
            only the second is a dynamic term the leg policy could have been counting on.
            A free-arm session has dose 0 by definition -- nothing was suppressed.

  response  leg action rate, = mean |delta raw_q| over slots 0-11 at the POLICY rate.
            The deploy-relevant smoothness metric (`g_legs`): hold_joint_ids freezes the
            upper body on hardware, so a whole-body action rate scores joints that cannot
            move. Also reported realised, from meas_dq, since commanded and realised
            smoothness came apart before (A0/A1 have opposite acceleration profiles).

Sessions are segmented by command regime before anything is averaged. A pooled
standing+walking score has produced a spurious winner on this project before, via error
cancellation, and `cmd == 0` is the only regime with a clean interpretation here.

Usage:
  # one cell per label, repeats collapse into mean +- spread
  python scripts/score_joint_hold.py \
      logs/deploy_safety/RUN1.csv=a0_free logs/deploy_safety/RUN2.csv=a0_free \
      logs/deploy_safety/RUN3.csv=a1_held logs/deploy_safety/RUN4.csv=a1_held

  # pre-2026-08-25 logs have no hold_joint_ids in their meta json; supply it
  python scripts/score_joint_hold.py OLD.csv=a1_held --assume-hold 12-26

Emits a per-cell table, the held/free ratio per policy, the dose-response regression,
and a one-line  [HOLD] {json}  summary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEG_SLOTS = list(range(12))


def parse_slots(spec):
    """'12-26' or '12,13,14' or '' -> list of ints."""
    out = []
    for part in filter(None, spec.split(",")):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def load(csv_path, assume_hold):
    """One session -> (dataframe, hold slots, provenance string)."""
    csv_path = Path(csv_path)
    meta_path = csv_path.with_name(csv_path.stem + "_meta.json")
    if not meta_path.exists():
        # Sessions get tagged by hand after the fact (`X.csv` -> `X_A0_fix.csv`), which
        # moves the meta to `X_meta_A0_fix.json` and breaks the canonical pairing. Fall
        # back to the sibling meta sharing this log's timestamp prefix.
        siblings = sorted(csv_path.parent.glob(f"{csv_path.name[:19]}*meta*.json"))
        if len(siblings) == 1:
            meta_path = siblings[0]
    hold, src = assume_hold, "--assume-hold"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if "hold_joint_ids" in meta:
            hold, src = meta["hold_joint_ids"], "meta.json"
        # A pre-change log is silently missing the key, which is exactly the ambiguity
        # the key was added to remove -- so say so rather than defaulting quietly.
        elif assume_hold is None:
            raise SystemExit(
                f"{csv_path.name}: meta json predates the hold_joint_ids field and no "
                f"--assume-hold given; the log cannot say whether the upper body was held")
    elif assume_hold is None:
        raise SystemExit(f"{csv_path.name}: no meta json and no --assume-hold")
    df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False)
    return df, sorted(hold), src


def policy_steps(raw):
    """Reduce a ~500 Hz-logged raw_q trace to one row per policy step.

    `raw` is (n_rows, 27) raw policy intent. The policy thread runs at 50 Hz while the
    recorder logs at ~500 Hz (safety_logger.h decimates to a 500 Hz target and then logs
    trigger ticks undecimated on top), so each policy output appears repeated across a
    VARIABLE number of rows. Every downstream rate metric differences this output, so how
    the repeats are collapsed sets what "action rate" means.

    Collapsed by detecting where `raw` actually changes, not by a fixed row stride. A
    stride assumes a uniform log, and this log is not uniform in two independent ways --
    the 500 Hz decimation means a stride of 20 rows spans 40 ms (two policy steps, not
    one), and undecimated trigger ticks perturb the spacing wherever the safety filter
    fires, which is 32-62% of rows on these sessions. Resampling on `t` was the third
    option and was rejected for the same reason: `t` is a tick counter times NOMINAL dt,
    not a clock, so it carries the same uniformity assumption the stride does while
    looking like it doesn't. Rate metrics differenced at the wrong rate have flipped two
    verdicts on this project, so the robust decode is worth the extra line.

    The residual risk is a policy emitting a bit-identical action twice in a row, which
    would merge two steps into one. That transition's true difference is zero, so merging
    drops a zero from the mean and biases the rate UP -- conservative for a test looking
    for excess motion. main() prints the implied step rate so a broken decode is visible.

    Returns (steps, idx): the (n_steps, 27) distinct outputs, and the row index each one
    started at, so a caller can map a step back to the regime it was issued in.
    """
    changed = np.any(np.diff(raw, axis=0) != 0.0, axis=1)
    idx = np.flatnonzero(np.concatenate(([True], changed)))
    return raw[idx], idx


def score(df, hold, label):
    raw = df[[f"raw_q{j}" for j in range(27)]].to_numpy(float)
    meas = df[[f"meas_q{j}" for j in range(27)]].to_numpy(float)
    dq = df[[f"meas_dq{j}" for j in range(27)]].to_numpy(float)
    cmd = df[["cmd_vx", "cmd_vy", "cmd_wz"]].to_numpy(float)
    t = df["t"].to_numpy(float)
    alpha = df["alpha"].to_numpy(float)

    # alpha > 0 means the safety filter is blending toward a position hold, so the row no
    # longer shows what the policy alone does. Scored separately as an outcome instead.
    live = alpha < 0.01
    regimes = {"still": live & (np.linalg.norm(cmd, axis=1) < 1e-6),
               "walk": live & (np.linalg.norm(cmd, axis=1) >= 1e-6)}

    out = {"label": label, "rows": int(len(df)), "hold": hold,
           "falls": int(np.diff((df["trig_fall"].to_numpy() > 0).astype(int)).clip(0).sum()),
           "alpha_max": float(alpha.max()),
           "trig_joint_pct": float(100.0 * (df["trig_joint"].to_numpy() > 0).mean())}

    # Step-decode ONCE on the full trace, then let each regime select which transitions to
    # average. Decoding inside a regime mask would difference across the gaps the mask
    # punches in the row stream, turning every regime re-entry into a spurious large step.
    steps, sidx = policy_steps(raw)
    d_all = np.abs(np.diff(steps, axis=0))
    out["step_hz"] = float(len(steps) / (t[-1] - t[0])) if t[-1] > t[0] else 0.0

    for name, m in regimes.items():
        if m.sum() < 500:
            continue
        e = (raw - meas)[m]
        # dose: only joints the hold actually suppressed contribute
        if hold:
            bias = float(np.abs(e[:, hold].mean(axis=0)).mean())
            osc = float(e[:, hold].std(axis=0).mean())
        else:
            bias = osc = 0.0
        # a transition counts for this regime only if BOTH its endpoints were issued in it
        in_reg = m[sidx]
        tsel = in_reg[:-1] & in_reg[1:]
        rsel = m[:-1] & m[1:]  # same rule for the realised (log-rate) difference
        if tsel.sum() < 20:
            continue
        d = d_all[tsel]
        # A session's pooled mean is dominated by whatever settling transient it happened
        # to contain -- measured on the 2026-08-25 block, two runs of the SAME cell read
        # 0.0166 and 0.0333 purely because one was stopped before it settled. `cmd == 0`
        # is not proof the robot is still. The FLOOR (quietest fifth of the session) is
        # the comparable quantity: the best stand this policy reached under this hold.
        # Report both -- a floor far below the mean means the session was still settling,
        # and a session whose floor never drops has not converged at all.
        w = len(d) // 5
        floor = min(d[k * w:(k + 1) * w][:, LEG_SLOTS].mean() for k in range(5)) if w else np.nan
        out[name] = {
            "dose_bias": bias, "dose_osc": osc, "dose": float(np.hypot(bias, osc)),
            "g_legs": float(d[:, LEG_SLOTS].mean()),
            "g_legs_floor": float(floor),
            "still_s": float(t[sidx][:-1][tsel][-1] - t[sidx][:-1][tsel][0]),
            "g_upper": float(d[:, 12:].mean()),
            "realised_legs": float(np.abs(np.diff(dq, axis=0))[rsel][:, LEG_SLOTS].mean()),
            "steps": int(tsel.sum()),
            "frac": float(m.mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", metavar="CSV=LABEL",
                    help="flight-recorder csv and the experimental cell it belongs to")
    ap.add_argument("--assume-hold", default=None,
                    help="hold slots for logs whose meta json predates the field, e.g. 12-26")
    ap.add_argument("--regime", default="still", choices=["still", "walk"],
                    help="command regime to score (default: still)")
    args = ap.parse_args()

    assume = parse_slots(args.assume_hold) if args.assume_hold is not None else None
    rows = []
    for spec in args.sessions:
        path, _, label = spec.partition("=")
        if not label:
            raise SystemExit(f"{spec}: expected CSV=LABEL")
        df, hold, src = load(path, assume)
        r = score(df, hold, label)
        r["file"], r["hold_src"] = Path(path).name, src
        rows.append(r)

    reg = args.regime
    scored = [r for r in rows if reg in r]
    if not scored:
        raise SystemExit(f"no session had enough rows in the '{reg}' regime")

    print(f"\nregime: {reg}    (per session)")
    print(f"  {'label':<12} {'dose':>7} {'osc':>7} {'g_legs':>8} {'floor':>8} {'still_s':>8} "
          f"{'steps':>6} {'Hz':>5} {'falls':>5} {'a_max':>6}  file")
    for r in scored:
        s = r[reg]
        # step_hz far from the policy rate means the decode is wrong and every rate below
        # it is meaningless -- flag it rather than let it pass as a quiet number.
        flag = "" if 40.0 <= r["step_hz"] <= 60.0 else "  <-- STEP RATE OFF"
        print(f"  {r['label']:<12} {s['dose']:>7.4f} {s['dose_osc']:>7.4f} "
              f"{s['g_legs']:>8.5f} {s['g_legs_floor']:>8.5f} {s['still_s']:>8.1f} "
              f"{s['steps']:>6} {r['step_hz']:>5.1f} "
              f"{r['falls']:>5} {r['alpha_max']:>6.2f}  {r['file'][:34]}{flag}")

    cells = {}
    for r in scored:
        cells.setdefault(r["label"], []).append(r[reg])
    print(f"\n  {'cell':<12} {'n':>2} {'dose':>17} {'g_legs_floor':>19}")
    summary = {}
    for label, ss in sorted(cells.items()):
        g = np.array([s["g_legs_floor"] for s in ss])
        d = np.array([s["dose"] for s in ss])
        summary[label] = {"n": len(ss), "dose": float(d.mean()), "g_legs_floor": float(g.mean()),
                          "spread": float(g.max() - g.min()),
                          "still_s": float(np.mean([x["still_s"] for x in ss]))}
        print(f"  {label:<12} {len(ss):>2} {d.mean():>9.4f} +-{d.max()-d.min():>6.4f} "
              f"{g.mean():>10.5f} +-{g.max()-g.min():>7.5f}")

    # The discriminator: held/free ratio per policy. A1 being jitterier than A0 is already
    # known and proves nothing -- only a LARGER ratio for A1 supports the hold hypothesis.
    print("\n  held/free ratio (the interaction; > A0's supports the hold hypothesis)")
    for policy in sorted({l.rsplit("_", 1)[0] for l in cells}):
        held, free = summary.get(f"{policy}_held"), summary.get(f"{policy}_free")
        if held and free and free["g_legs_floor"] > 0:
            ratio = held["g_legs_floor"] / free["g_legs_floor"]
            summary[f"{policy}_ratio"] = float(ratio)
            print(f"    {policy:<10} {ratio:>7.2f}   (held {held['g_legs_floor']:.5f} over "
                  f"{held['still_s']:.0f}s / free {free['g_legs_floor']:.5f} over "
                  f"{free['still_s']:.0f}s, n={held['n']}/{free['n']})")
        else:
            print(f"    {policy:<10} incomplete: need both {policy}_held and {policy}_free")

    # Dose-response across every cell: if the hold is causal, leg thrash rises with the
    # suppressed command on one line through both policies. A policy sitting above that
    # line at equal dose is jittery for its own reasons. Survives n=2 where the ratio does not.
    d = np.array([s["dose"] for ss in cells.values() for s in ss])
    g = np.array([s["g_legs_floor"] for ss in cells.values() for s in ss])
    if len(d) >= 3 and d.std() > 1e-9:
        slope, icpt = np.polyfit(d, g, 1)
        r = float(np.corrcoef(d, g)[0, 1])
        summary["dose_response"] = {"slope": float(slope), "intercept": float(icpt), "r": r}
        print(f"\n  dose-response  g_legs = {slope:.4f}*dose + {icpt:.5f}   r = {r:+.3f}  (n={len(d)})")
    else:
        print(f"\n  dose-response  not computed (need >=3 sessions with spread in dose)")

    print(f"\n[HOLD] {json.dumps({'regime': reg, 'cells': summary})}")


if __name__ == "__main__":
    main()
