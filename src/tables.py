"""Summary tables of the evaluation (results/tables.md, results/summary.json).

A failed run (the simulation stopped before 48 h) ends with a truncated trace, so its MAE, IAE, StE, gate
travel and delivery error are not comparable with those of a completed run. Failed runs are counted in the
'Failed runs' column and are left out of the other columns: a cell is '–' when every run of that controller
failed, and a '†' marks statistics computed from the surviving runs only.
"""
import json, os
import numpy as np
from figures import load_eval, keys, GROUP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABS = ["Fixed", "PI", "R-PPO", "R-TD3", "PPO", "TD3", "MPC-KF", "MPC"]
ev = load_eval()["results"]
named = ["T1", "T2", "T3", "T4", "T5", "T6"]
rand = sorted(k for k in ev if k.startswith("R"))
METS = (("MAE", 100), ("IAE", 100), ("StE", 100), ("travel", 1), ("delivery_err_pct", 1))
out, lines = {}, []


def keys_ok(s, lab):
    """All run keys of a controller label in scenario s, and those of the runs that did not fail."""
    ks = keys(ev[s], lab)
    return ks, [c for c in ks if not ev[s][c]["failed"]]


for s in named:
    lines.append("\n### %s\n\n| Controller | MAE (cm) | IAE (cm) | StE (cm) | Gate travel (m) | Delivery error (%%) | Failed runs | CPU per decision (ms) |\n|---|---|---|---|---|---|---|---|" % s)
    for lab in LABS:
        ks, ok = keys_ok(s, lab)
        n, f = len(ks), len(ks) - len(ok)
        row, cells = dict(failed=f, n=n), []
        for met, sc in METS:
            if not ok:
                cells.append("–")
                continue
            v = np.array([ev[s][c][met] for c in ok], float) * sc
            row[met] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()), n_used=len(ok))
            dec = 2 if met == "delivery_err_pct" else 1
            if len(v) == 1:
                cells.append("%.*f" % (dec, v[0]))
            else:
                cells.append("%.*f (%.*f–%.*f)" % (dec, np.median(v), dec, v.min(), dec, v.max()))
        if ok and f:
            cells[0] += " †"
        cpu = float(np.median([ev[s][c]["cpu_ms"] for c in ks]))
        cells += ["%d/%d" % (f, n), "%.1f" % cpu]
        out.setdefault(s, {})[lab] = row
        lines.append("| %s | %s |" % (lab, " | ".join(cells)))

# random set: per controller, mean over scenarios of the per-scenario median over the surviving seeds
lines.append("\n### Random test set (20 scenarios)\n\n| Controller | mean IAE (cm) | mean MAE (cm) | worst MAE (cm) | mean delivery error (%) | failed runs | scenarios used | IAE ratio to MPC-KF |\n|---|---|---|---|---|---|---|---|")
for lab in LABS:
    iae, mae, dl, ref, fails, nr = [], [], [], [], 0, 0
    for s in rand:
        ks, ok = keys_ok(s, lab)
        fails += len(ks) - len(ok); nr += len(ks)
        if not ok:
            continue
        iae.append(np.median([ev[s][c]["IAE"] for c in ok])); mae.append(np.median([ev[s][c]["MAE"] for c in ok]))
        dl.append(np.median([ev[s][c]["delivery_err_pct"] for c in ok])); ref.append(ev[s]["MPC-KF"]["IAE"])
    if not iae:
        out.setdefault("random", {})[lab] = dict(failed=int(fails), runs=int(nr), scenarios_used=0)
        lines.append("| %s | – | – | – | – | %d/%d | 0/%d | – |" % (lab, fails, nr, len(rand)))
        continue
    iae, mae, ref = np.array(iae) * 100, np.array(mae) * 100, np.array(ref) * 100
    out.setdefault("random", {})[lab] = dict(IAE=float(iae.mean()), MAE=float(mae.mean()), MAE_worst=float(mae.max()),
                                             deliv=float(np.mean(dl)), failed=int(fails), runs=int(nr),
                                             scenarios_used=len(iae), ratio=float(np.mean(iae / ref)))
    r = out["random"][lab]
    mark = " †" if fails else ""
    lines.append("| %s%s | %.2f | %.1f | %.1f | %.2f | %d/%d | %d/%d | %.2f |" % (
        lab, mark, r["IAE"], r["MAE"], r["MAE_worst"], r["deliv"], fails, nr, len(iae), len(rand), r["ratio"]))
lines.append("\n† statistics from the surviving runs only; – every run failed. A failed run ends before 48 h, so its "
             "indicators refer to a truncated trace and are not comparable with those of completed runs.")
open(os.path.join(ROOT, "results", "tables.md"), "w").write("\n".join(lines))
json.dump(out, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)
print("\n".join(lines))
