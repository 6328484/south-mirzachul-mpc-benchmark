"""Mean return of the classical controllers on the 6 validation scenarios (reference lines of the
learning curves). Writes results/val_baselines.json."""
import json, os
import numpy as np
from env import CanalEnv
from controllers import Fixed, PI, MPC, MPCKF, run_episode
from train import VAL_SEEDS, GAINS, ROOT

env = CanalEnv()
out = {}
for C in (Fixed(), PI(GAINS["kp"], GAINS["ki"]), MPC(env), MPCKF(env)):
    R = []
    for s in VAL_SEEDS:
        env.rng = np.random.default_rng(s)
        env.reset()
        R.append(run_episode(env, C, env.sc)["return"])
    out[C.name] = float(np.mean(R))
    print(C.name, round(out[C.name], 2), np.round(R, 1), flush=True)
json.dump(out, open(os.path.join(ROOT, "results", "val_baselines.json"), "w"))
