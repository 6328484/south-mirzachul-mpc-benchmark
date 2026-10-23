"""Summary tables of the evaluation (results/tables.md, results/summary.json)."""
import json, os
import numpy as np
from figures import load_eval, keys, GROUP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABS = ["Fixed", "PI", "R-PPO", "R-TD3", "PPO", "TD3", "MPC-KF", "MPC"]
ev = load_eval()["results"]
named = ["T1", "T2", "T3", "T4", "T5", "T6"]
rand = sorted(k for k in ev if k.startswith("R"))
out, lines = {}, []


def agg(s, lab, met):
    v = np.array([ev[s][c][met] for c in keys(ev[s], lab)], float)
    return v


for s in named:
    lines.append("\n### %s\n\n| Controller | MAE (cm) | IAE (cm) | StE (cm) | Gate travel (m) | Delivery error (%%) | Failed runs | CPU per decision (ms) |\n|---|---|---|---|---|---|---|---|" % s)
    for lab in LABS:
        row = {}
        cells = []
        for met, sc in (("MAE", 100), ("IAE", 100), ("StE", 100), ("travel", 1), ("delivery_err_pct", 1)):
            v = agg(s, lab, met) * sc
            row[met] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
            cells.append("%.1f" % np.median(v) if len(v) == 1 else "%.1f (%.1f–%.1f)" % (np.median(v), v.min(), v.max())
                         if met != "delivery_err_pct" else "%.2f (%.2f–%.2f)" % (np.median(v), v.min(), v.max()))
            if len(v) == 1 and met == "delivery_err_pct":
                cells[-1] = "%.2f" % v[0]
        f = agg(s, lab, "failed"); n = len(f)
        cpu = agg(s, lab, "cpu_ms")
        row["failed"] = int(f.sum()); row["n"] = n
        cells += ["%d/%d" % (f.sum(), n), "%.1f" % np.median(cpu)]
        out.setdefault(s, {})[lab] = row
        lines.append("| %s | %s |" % (lab, " | ".join(cells)))

# random set: per controller, mean over scenarios of the per-scenario median over seeds
lines.append("\n### Random test set (20 scenarios)\n\n| Controller | mean IAE (cm) | mean MAE (cm) | worst MAE (cm) | mean delivery error (%) | failed runs | IAE ratio to MPC-KF |\n|---|---|---|---|---|---|---|")
ref = np.array([ev[s]["MPC-KF"]["IAE"] for s in rand])
for lab in LABS:
    iae, mae, dl, fails, nr = [], [], [], 0, 0
    for s in rand:
        iae.append(np.median(agg(s, lab, "IAE"))); mae.append(np.median(agg(s, lab, "MAE")))
        dl.append(np.median(agg(s, lab, "delivery_err_pct")))
        f = agg(s, lab, "failed"); fails += f.sum(); nr += len(f)
    iae, mae = np.array(iae) * 100, np.array(mae) * 100
    out.setdefault("random", {})[lab] = dict(IAE=float(iae.mean()), MAE=float(mae.mean()), MAE_worst=float(mae.max()),
                                             deliv=float(np.mean(dl)), failed=int(fails), runs=int(nr),
                                             ratio=float(np.mean(iae / (ref * 100))))
    r = out["random"][lab]
    lines.append("| %s | %.2f | %.1f | %.1f | %.2f | %d/%d | %.2f |" % (lab, r["IAE"], r["MAE"], r["MAE_worst"], r["deliv"], fails, nr, r["ratio"]))
open(os.path.join(ROOT, "results", "tables.md"), "w").write("\n".join(lines))
json.dump(out, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)
print("\n".join(lines))
