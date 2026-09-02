# Flight-recorder fixture

The three A0 sessions of 2026-08-27, reduced so `tests/test_bench_flight_recorder.py` runs
everywhere. The full captures are 470 MB and are not in git; without this fixture the golden
test that guards ADR-0009's hardware numbers would skip on every machine but one.

| file | what it is |
|---|---|
| `<session>.steps.csv.gz` | the **policy-step decode** of `raw_q0..raw_q11` plus `t`. One row per distinct policy output, recovered by change-detection over all 27 columns (`score_joint_hold.policy_steps`), which is how the ADR number was computed. |
| `<session>.meta.json` | `control_dt` and the 12 leg joints' name/slot/min/max from the session's own `_meta.json`. The bounds are the pinning reference for `span`/`headroom`/`pinned`. |
| `2026-08-27_13-42-55.rawslice.csv.gz` | 30 s of **undecoded** rows (all 27 `raw_q` + `t`), from `t=210..240 s`, chosen as the window with the most **non-uniform row spacing** (39%). That is deliberate: on a perfectly uniform 500 Hz log a fixed stride of 10 rows also averages 50 Hz, so a uniform slice cannot tell a correct decode from a broken one. The 13-51-52 session never engaged the safety filter, so its rows are uniform and it made a useless fixture. |

**Lossless.** The recorder writes 4 significant digits, so `%.6g` loses nothing; the six
ankle spans reproduce bit-for-bit against the full captures.

Regenerate (needs the full captures in `logs/deploy_safety/`):

```python
import json, sys; sys.path.insert(0, "scripts")
import pandas as pd
from score_joint_hold import policy_steps
for s in ["2026-08-27_13-42-55", "2026-08-27_13-51-52", "2026-08-27_13-56-37"]:
    full = pd.read_csv(f"logs/deploy_safety/{s}.csv",
                       usecols=[f"raw_q{j}" for j in range(27)],
                       on_bad_lines="skip", low_memory=False)
    steps, sidx = policy_steps(full.to_numpy(float))
    t = pd.read_csv(f"logs/deploy_safety/{s}.csv", usecols=["t"],
                    on_bad_lines="skip", low_memory=False)["t"].to_numpy(float)
    fx = pd.DataFrame(steps[:, :12], columns=[f"raw_q{j}" for j in range(12)])
    fx.insert(0, "t", t[sidx])
    fx.to_csv(f"tests/fixtures/flight_recorder/{s}.steps.csv.gz",
              index=False, float_format="%.6g", compression="gzip")
    m = json.load(open(f"logs/deploy_safety/{s}_meta.json"))
    json.dump({"control_dt": m["control_dt"],
               "joints": [j for j in m["joints"] if j["slot"] < 12]},
              open(f"tests/fixtures/flight_recorder/{s}.meta.json", "w"), indent=1)
```
