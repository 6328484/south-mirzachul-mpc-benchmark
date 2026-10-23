"""
Training of the residual PPO and TD3 agents on the Saint-Venant environment
===========================================================================
usage: python train.py ppo|td3 SEED [EPISODES] [pure]
Residual architecture (default): the gate move is the tuned PI move plus the agent's correction,
da = clip(da_PI + DA_MAX a, +-DA_MAX); the agent observes the 32 measurements and the PI move.
'pure': the agent alone sets da = DA_MAX a (ablation).
Training scenarios: random_scenario() with n/n0 in {0.9, 1.0, 1.1}.
Validation: 6 fixed scenarios (seeds 2000-2005), deterministic policy; the checkpoint with the best
mean validation return is kept (results/policies/<tag>.pkl); learning curve: results/logs/<tag>.json.
"""
import sys, json, time, os, pickle
import numpy as np
from env import CanalEnv, DA_MAX
from rl import PPO, TD3
from controllers import PI

R_SCALE = 0.1          # reward scaling for the learners (logged returns are unscaled)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL_SEEDS = range(2000, 2006)
GAINS = json.load(open(os.path.join(ROOT, "results", "pi_gains.json")))


class Runner:
    """Couples an agent to the environment (residual or pure)."""

    def __init__(self, env, residual):
        self.env, self.residual = env, residual
        self.pi = PI(GAINS["kp"], GAINS["ki"])

    def reset(self):
        o = self.env.reset()
        self.pi.reset(self.env)
        return self._obs(o)

    def _obs(self, o):
        if not self.residual:
            return o
        self.b = self.pi.act(self.env, None)
        return np.concatenate([o, self.b / DA_MAX])

    def step(self, a):
        a = np.clip(a, -1, 1)
        da = np.clip(self.b + DA_MAX * a, -DA_MAX, DA_MAX) if self.residual else DA_MAX * a
        o, r, d, i = self.env.step(da)
        return self._obs(o), r, d, i


def validate(run, act):
    R = []
    for s in VAL_SEEDS:
        run.env.rng = np.random.default_rng(s)
        o = run.reset(); d = False; tot = 0.0
        while not d:
            o, r, d, _ = run.step(act(o)); tot += r
        R.append(tot)
    return float(np.mean(R))


def main(algo, seed, episodes, residual=True):
    tag = "%s%s_s%d" % (algo, "" if residual else "pure", seed)
    env, venv = CanalEnv(seed=seed), CanalEnv(seed=seed)
    env.rng = np.random.default_rng(10_000 + seed)
    run, vrun = Runner(env, residual), Runner(venv, residual)
    n_obs = env.n_obs + (env.n_act if residual else 0)
    os.makedirs(os.path.join(ROOT, "results", "policies"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    log = dict(episode=[], train_return=[], val_episode=[], val_return=[], wall=[], env_steps=[],
               residual=residual)
    best, best_state, t0, steps = -np.inf, None, time.time(), 0

    def checkpoint(ep, ag, act, extra=""):
        nonlocal best, best_state
        vr = validate(vrun, act)
        log["val_episode"].append(ep); log["val_return"].append(vr)
        log["wall"].append(time.time() - t0); log["env_steps"].append(steps)
        if vr > best:
            best, best_state = vr, ag.get_state()
        print("%s ep %d val %.2f best %.2f %s %.0f s" % (tag, ep, vr, best, extra, time.time() - t0), flush=True)

    if algo == "ppo":
        ag = PPO(n_obs, env.n_act, seed=seed, log_std0=-2.0 if residual else -0.7, lr=1e-4, gamma=0.95)
        per_update, val_every, ep = 8, 40, 0
        checkpoint(0, ag, lambda o: ag.act(o, True)[0])
        while ep < episodes:
            buf = {k: [] for k in ("obs", "act", "logp", "rew", "val", "done", "val_next", "term")}
            for _ in range(per_update):
                o = run.reset(); d = False; tot = 0.0
                while not d:
                    a, lp, v = ag.act(o)
                    o2, r, d, _ = run.step(a)
                    for k, vv in zip(("obs", "act", "logp", "rew", "val", "done", "term"),
                                     (o, a, lp, r * R_SCALE, v, d, env.failed)):
                        buf[k].append(vv)
                    buf["val_next"].append(float(ag.value(o2[None])[0])); o = o2; tot += r; steps += 1
                ep += 1
                log["episode"].append(ep); log["train_return"].append(tot)
            ag.update(buf)
            if ep % val_every == 0 or ep >= episodes:
                checkpoint(ep, ag, lambda o: ag.act(o, True)[0], "std %.3f" % np.exp(ag.log_std).mean())
    else:
        ag = TD3(n_obs, env.n_act, seed=seed, expl=0.1 if residual else 0.2, lr_a=1e-4, lr_c=3e-4,
                 gamma=0.95, bc_alpha=2.5 if residual else 0.0)
        warm, val_every = 10, 10
        checkpoint(0, ag, lambda o: ag.act(o, True))
        for ep in range(1, episodes + 1):
            o = run.reset(); d = False; tot = 0.0
            while not d:
                a = env.rng.normal(0, 0.1, env.n_act) if ep <= warm else ag.act(o)
                o2, r, d, _ = run.step(a)
                ag.store(o, np.clip(a, -1, 1), r * R_SCALE, o2, float(env.failed)); o = o2; tot += r; steps += 1
                if ep > warm:
                    ag.update()
            log["episode"].append(ep); log["train_return"].append(tot)
            if ep % val_every == 0 or ep == episodes:
                checkpoint(ep, ag, lambda o: ag.act(o, True))
    log["best_val"] = best; log["total_wall_s"] = time.time() - t0; log["total_env_steps"] = steps
    with open(os.path.join(ROOT, "results", "policies", tag + ".pkl"), "wb") as f:
        pickle.dump(best_state, f)
    json.dump(log, open(os.path.join(ROOT, "results", "logs", tag + ".json"), "w"))


if __name__ == "__main__":
    algo, seed = sys.argv[1], int(sys.argv[2])
    episodes = int(sys.argv[3]) if len(sys.argv) > 3 else (800 if algo == "ppo" else 200)
    residual = not (len(sys.argv) > 4 and sys.argv[4] == "pure")
    main(algo, seed, episodes, residual)
