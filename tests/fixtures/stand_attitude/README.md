# Quiet-stand attitude fixture

The two 2026-09-16 runs that `docs/adr/0006` Amendment (2026-09-21) rests on, reduced so
`tests/test_stand_attitude.py` runs everywhere. The full captures are 115-127 MB each and
`logs/` is gitignored; without this fixture the guard on the amendment's headline would skip
on every machine but one.

| file | what it is |
|---|---|
| `2026-09-16_13-03-28.stand.csv.gz` | **A0_s123, run 16** — the leaning arm (-4.35 deg) and, awkwardly for the confound, the *more still* one. |
| `2026-09-16_11-06-52.stand.csv.gz` | **A1a_DR_s42, run 11** — the hierarchical arm at matched stillness (+0.57 deg). |
| `<stem>.meta.json` | `joint_offset` only. A0 flew the 0.012/joint calibration, the hierarchy flew the all-zero no-op; that asymmetry is what the amendment turns on, and it is read per run because the deploy yaml is the *current* file, not the one that flew. |

**Columns:** only what the scorer reads (`quat_*`, `cmd_*`, `alpha`, `entry`, `meas_dq0..26`
plus `t`). The upper-body `meas_dq12..26` are kept although the statistic is legs-only, so a
legs-only-to-all-27 regression is expressible against real data rather than only synthetic.

**Reduction: every 10th row (≈500 Hz → the 50 Hz policy rate), not lossless.** Measured
against the full captures over both runs: `pitch_deg` moves ≤2.4e-4 deg, `still_pct` ≤0.028
points, `mean_maxdq` ≤8.4e-4. The test pins the ADR's FULL-RATE numbers and sets its
tolerances from those measured deltas with ~2x margin, so a real regression is still caught.

Regenerate (needs the full captures in `logs/deploy_safety/`):

```python
import json, pandas as pd
from pathlib import Path
FIX = Path("tests/fixtures/stand_attitude")
COLS = ["t","quat_w","quat_x","quat_y","quat_z","cmd_vx","cmd_vy","cmd_wz","alpha","entry"] \
       + [f"meas_dq{i}" for i in range(27)]
for stem in ["2026-09-16_13-03-28", "2026-09-16_11-06-52"]:
    pd.read_csv(f"logs/deploy_safety/{stem}.csv").iloc[::10][COLS].to_csv(
        FIX/f"{stem}.stand.csv.gz", index=False, float_format="%.6g", compression="gzip")
    off = json.loads(Path(f"logs/deploy_safety/{stem}_meta.json").read_text())["joint_offset"]
    (FIX/f"{stem}.meta.json").write_text(json.dumps({"joint_offset": off}, indent=1))
```
