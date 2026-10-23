"""
Evaluation on test scenarios that are not used in training
==========================================================
Named scenarios (Table 2 of the manuscript):
  T1 scheduled decrease : Q_head 260 -> 220 m3/s (2-h ramp at 4 h), matching scheduled branch orders;
                          unscheduled: Kurgantepa +20 % at 20 h
  T2 scheduled increase : Q_head 260 -> 285 m3/s at 6 h; unscheduled: Central -15 % at 12 h,
                          YuR-18 +20 % at 30 h
  T3 unscheduled only   : constant Q_head; Right Bayaut -20 % at 5 h, Central +10 % at 18 h
  T4 T1 with n/n0 = 1.2 (roughness outside the training range)
  T5 T1 with n/n0 = 0.85 (roughness outside the training range)
  T6 T1 with measurement noise (sigma = 2 cm on every depth)
Random test set R: 20 scenarios of the training distribution (seeds 3000-3019), unseen.
Metrics (7 controlled points): MAE = max |e| (m), IAE = time-mean |e| averaged over points (m),
StE = mean |e| over the last 6 h (m), gate travel (m), delivery error = sum over offtakes of
|volume delivered - volume ordered| / volume ordered (%; the order is scaled by the
delivery/order ratio of the initial steady state), tail deviation volume (1000 m3),
CPU time per decision (ms).
"""
import sys, os, json, pickle, glob
import numpy as np
from env import CanalEnv, Scenario, DA_MAX, CTRL
from controllers import Fixed, PI, MPC, MPCKF, run_episode
from rl import PPO, TD3
from canal import Q_NORM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = dict(RB=0, Y10=1, CEN=2, KUR=3, Y18=4)


def scheduled(env, t0, dQ):
    """Scheduled branch orders matching a head change dQ (shared in proportion to the orders)."""
    share = env.maj_q0 / env.maj_q0.sum()
    out = []
    for j in range(len(env.maj_q0)):
        fac = 1.0 + dQ * share[j] / env.maj_q0[j]
        out.append((j, t0 + env.majors[j][1] / 25.0, fac, 2.0))
    return out


def named_scenarios(env):
    T1 = lambda **k: Scenario([(0, 260), (4, 260), (6, 220), (48, 220)], scheduled(env, 4, -40),
                              [(J["KUR"], 20, 1.2, 1.0)], **k)
    S = dict(
        T1=T1(name="T1"),
        T2=Scenario([(0, 260), (6, 260), (8, 285), (48, 285)], scheduled(env, 6, 25),
                    [(J["CEN"], 12, 0.85, 1.0), (J["Y18"], 30, 1.2, 1.0)], name="T2"),
        T3=Scenario([(0, 260), (48, 260)], [], [(J["RB"], 5, 0.8, 1.0), (J["CEN"], 18, 1.1, 1.0)], name="T3"),
        T4=T1(n_scale=1.2, name="T4"),
        T5=T1(n_scale=0.85, name="T5"),
        T6=T1(noise=0.02, name="T6"),
    )
    return S


def random_test(env, n=20):
    out = []
    for s in range(3000, 3000 + n):
        env.rng = np.random.default_rng(s)
        env.reset()
        out.append(env.sc)
    return out


def metrics(env, L):
    e = np.abs(L["hu"][:, :7] - env.h_tgt[:7])
    t = L["t"]
    last = t >= t[-1] - 6 * 3600
    a = L["a"]
    qd, qo = L["qd"], L["qo"]
    dt = np.diff(t)[0] if len(t) > 1 else 900.0
    # delivery reference: the order scaled by the delivery/order ratio of the initial steady state
    # (orifice offtakes do not deliver exactly their order away from the control points)
    q0 = env.op["q_ord"]; r0 = np.where(q0 > 0, env.c.actual_offtake(env.op["h"], q0) / np.maximum(q0, 1e-12), 1.0)
    vol_d, vol_o = qd.sum(0) * dt, (qo * r0).sum(0) * dt
    sel = vol_o > 0
    deliv = 100 * np.abs(vol_d - vol_o)[sel].sum() / vol_o[sel].sum()
    tail_dev = float(((L["Qtail"][1:] - L["Qtail"][0]) * dt).sum() / 1e3)
    return dict(MAE=float(e.max()), IAE=float(e.mean()), StE=float(e[last].mean()),
                MAE_pts=e.max(0).tolist(), travel=float(np.abs(np.diff(a, axis=0)).sum()),
                delivery_err_pct=float(deliv), tail_dev_1000m3=tail_dev, failed=bool(L["failed"]),
                cpu_ms=1e3 * L["cpu_per_decision"], ret=float(L["return"]))


class Agent:
    """Trained agent; residual (PI + correction) unless the policy file is a 'pure' ablation."""

    def __init__(self, algo, path, env, gains):
        st = pickle.load(open(path, "rb"))
        self.residual = "pure" not in os.path.basename(path)
        n_obs = env.n_obs + (env.n_act if self.residual else 0)
        self.ag = PPO(n_obs, env.n_act) if algo == "ppo" else TD3(n_obs, env.n_act)
        self.ag.set_state(st)
        self.pi = PI(gains["kp"], gains["ki"])

    def reset(self, env):
        self.pi.reset(env)

    def act(self, env, obs):
        if self.residual:
            b = self.pi.act(env, None)
            a = self.ag.act(np.concatenate([obs, b / DA_MAX]), True)
        else:
            b = 0.0
            a = self.ag.act(obs, True)
        a = a[0] if isinstance(a, tuple) else a
        return np.clip(b + DA_MAX * np.clip(a, -1, 1), -DA_MAX, DA_MAX)


def controllers(env):
    g = json.load(open(os.path.join(ROOT, "results", "pi_gains.json")))
    out = [("Fixed", Fixed()), ("PI", PI(g["kp"], g["ki"])), ("MPC", MPC(env)), ("MPC-KF", MPCKF(env))]
    for algo in ("ppo", "td3", "ppopure", "td3pure"):
        for p in sorted(glob.glob(os.path.join(ROOT, "results", "policies", "%s_s*.pkl" % algo))):
            seed = os.path.basename(p)[len(algo) + 2:-4]
            out.append(("%s_s%s" % (algo.upper(), seed), Agent(algo[:3], p, env, g)))
    return out


def main(which="all"):
    env = CanalEnv(seed=0)
    ctrls = controllers(env)
    res, traces = {}, {}
    scen = list(named_scenarios(env).items()) if which in ("all", "named") else []
    if which in ("all", "random"):
        scen += [("R%02d" % i, s) for i, s in enumerate(random_test(env))]
    for sname, sc in scen:
        for cname, C in ctrls:
            env.rng = np.random.default_rng(42)            # same noise realisation for all controllers
            L = run_episode(env, C, sc)
            res.setdefault(sname, {})[cname] = metrics(env, L)
            if not sname.startswith("R"):
                traces.setdefault(sname, {})[cname] = dict(t=L["t"].tolist(), hu=L["hu"].tolist(),
                                                           a=L["a"].tolist(), Qin=L["Qin"].tolist())
            m = res[sname][cname]
            print("%-4s %-8s MAE %.3f IAE %.4f StE %.4f travel %6.2f deliv %.2f%% fail %s cpu %.1f ms" %
                  (sname, cname, m["MAE"], m["IAE"], m["StE"], m["travel"], m["delivery_err_pct"],
                   m["failed"], m["cpu_ms"]), flush=True)
    suf = "" if which == "all" else "_" + which
    json.dump(dict(results=res, h_tgt=env.h_tgt.tolist(), points=[env.c.sname[k] for k in env.ci] + ["tail"]),
              open(os.path.join(ROOT, "results", "evaluation%s.json" % suf), "w"))
    if traces:
        json.dump(traces, open(os.path.join(ROOT, "results", "traces.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
