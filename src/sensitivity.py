"""Sensitivity of MPC-KF to the prediction horizon Np and the move weight w (scenarios T1 and T4).
Writes results/sensitivity.json."""
import json, os
import numpy as np
from env import CanalEnv
from controllers import MPCKF, run_episode
from evaluate import named_scenarios, metrics, ROOT

env = CanalEnv()
S = named_scenarios(env)
out = []
for Np in (8, 16, 24):
    for w in (0.01, 0.05, 0.25):
        ctrl = MPCKF(env, Np=Np, Nc=min(6, Np), w_move=w)
        for sn in ("T1", "T4"):
            env.rng = np.random.default_rng(42)
            m = metrics(env, run_episode(env, ctrl, S[sn]))
            out.append(dict(Np=Np, w=w, scenario=sn, IAE=m["IAE"], MAE=m["MAE"], travel=m["travel"],
                            cpu_ms=m["cpu_ms"]))
            print(out[-1], flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "sensitivity.json"), "w"))
