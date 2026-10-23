"""Grid search (kp, ki, area scaling) of the common PI gains on the training scenarios (same distribution as DRL training;
seeds 1000-1015). Writes results/pi_grid.csv and results/pi_gains.json."""
import json, sys, itertools
import numpy as np
from multiprocessing import Pool
from env import CanalEnv
from controllers import PI, run_episode

KP = [3.0, 6.0]
KI = [1.5, 2.5, 4.0]
SCALED = [False]
SEEDS = range(1000, 1016)


def evaluate(args):
    kp, ki, sc = args
    env = CanalEnv()
    R, fails = [], 0
    for s in SEEDS:
        env.rng = np.random.default_rng(s)
        env.reset()
        L = run_episode(env, PI(kp, ki, sc), env.sc)
        R.append(L["return"]); fails += L["failed"]
    return kp, ki, sc, float(np.mean(R)), fails


if __name__ == "__main__":
    with Pool(2) as p:
        out = p.map(evaluate, list(itertools.product(KP, KI, SCALED)))
    out.sort(key=lambda r: -r[3])
    with open("../results/pi_grid.csv", "w") as f:
        f.write("kp,ki,area_scaled,mean_return,failures\n")
        for r in out:
            f.write("%g,%g,%d,%.3f,%d\n" % r)
    best = out[0]
    json.dump(dict(kp=best[0], ki=best[1], area_scaled=bool(best[2]), mean_return=best[3]), open("../results/pi_gains.json", "w"))
    print(out[:5])
