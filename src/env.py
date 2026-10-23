"""
Unsteady-flow regulation environment (Saint-Venant model of the canal)
=======================================================================
Control task: hold the depth upstream of the seven controlled check structures at its target while
the head discharge and the withdrawals of the major branches change (ASCE-style canal automation
test: scheduled changes known in advance, unscheduled changes not known).

Time: hydraulic step DT = 900 s; the controller acts every CTRL = 1800 s (2 hydraulic steps);
episode length 48 h (96 decisions).
Actuators: openings of the 7 check gates (PK145, PK400, PK623+86, PK746+62, PK923+02, PK1087+23,
PK1177+29). PK1035 is fully open; the tail gate keeps the opening that gives the normal tail depth
in the initial steady state. A gate moves by at most
DA_MAX per control interval and stays within [0.05, 1.3 max(passport height, normal opening)].
Offtakes: orifice type, q = u q_ord sqrt(h/h_ref); u is the order factor of a major branch
(Right Bayaut, YuR-10, Central, Kurgantepa, YuR-18), minor offtakes keep their order (u = 1).

Observation (measurable by SCADA, 32 values):
  8  depth errors (7 controlled points + tail), scaled by 0.1 m
  8  their change over the last control interval, scaled by 0.03 m
  7  gate openings relative to the normal opening (a/a0 - 1)
  4  head discharge now and its schedule at +1, +2, +4 h (Q/Q_norm - 1) x 5
  5  measured withdrawals of the major branches relative to normal (q/q0 - 1) x 5
"""
import numpy as np
from canal import build, Q_NORM
from gated_model import g

DT, CTRL, T_EP = 900.0, 1800.0, 48 * 3600.0
DA_MAX = 0.15                       # m per control interval
E_SCALE, W_MOVE = 0.10, 0.05        # reward: depth error scale (m) and gate-movement weight
E_CAP = 25.0                        # per-point cap of (e/E)^2 (errors above 0.5 m count as 0.5 m)
MAJOR_MIN_Q = 10.0


class Scenario:
    """Head discharge schedule (known), branch order factors (scheduled part known at its start;
    unscheduled part unknown), roughness multiplier and measurement noise."""

    def __init__(self, Qin_pts, sched, unsched, n_scale=1.0, noise=0.0, name=""):
        self.Qin_pts = np.asarray(Qin_pts, float)          # (k, 2): time [h], Q [m3/s]
        self.sched = sched                                 # list of (branch j, t_h, factor, ramp_h)
        self.unsched = unsched                             # list of (branch j, t_h, factor, ramp_h)
        self.n_scale, self.noise, self.name = n_scale, noise, name

    def Qin(self, t):
        th = t / 3600.0
        return float(np.interp(th, self.Qin_pts[:, 0], self.Qin_pts[:, 1]))

    def factors(self, t, nb):
        th = t / 3600.0
        u = np.ones(nb)
        for j, t0, f, r in list(self.sched) + list(self.unsched):
            w = np.clip((th - t0) / max(r, 1e-6), 0.0, 1.0)
            u[j] *= 1.0 + (f - 1.0) * w
        return u


def random_scenario(rng, n_choices=(0.9, 1.0, 1.1)):
    """Training distribution. A scheduled change of the head discharge (dQ in [-45, +25] m3/s,
    2-h ramp starting at 2-12 h) is matched by scheduled changes of the major-branch orders that
    start later, by a delay that grows with the distance from the head (water released in advance);
    one or two unscheduled branch changes (factor 0.8-1.2) occur at 4-40 h."""
    nb = 5
    t0 = rng.uniform(2, 12)
    dQ = rng.uniform(-45, 25) if rng.random() < 0.85 else 0.0
    pts = [(0, Q_NORM), (t0, Q_NORM), (t0 + 2, Q_NORM + dQ), (48, Q_NORM + dQ)]
    if rng.random() < 0.3:                                 # second change
        t1 = rng.uniform(t0 + 8, 36)
        dQ2 = rng.uniform(-25, 25)
        Q2 = float(np.clip(Q_NORM + dQ + dQ2, 200, 295))
        pts = pts[:3] + [(t1, Q_NORM + dQ), (t1 + 2, Q2), (48, Q2)]
    sched = []
    # branch orders follow the head change (shared in proportion to the branch orders)
    share = rng.dirichlet(np.ones(nb) * 2.0)
    return Scenario(pts, sched, _unsched(rng, nb), n_scale=float(rng.choice(n_choices)),
                    name="random"), share, dQ


def _unsched(rng, nb):
    out = []
    for _ in range(rng.integers(1, 3)):
        out.append((int(rng.integers(nb)), float(rng.uniform(4, 40)), float(rng.uniform(0.8, 1.2)),
                    float(rng.uniform(0.5, 2.0))))
    return out


class CanalEnv:
    def __init__(self, seed=0):
        op = build()
        self.op, self.c = op, op["c"]
        c = self.c
        self.n0 = c.n.copy()
        self.ci = np.array([k for k in range(c.n_struct) if not c.open_struct[k]])   # 7 controlled
        self.pts = np.append(c.u_nodes[self.ci], c.N - 1)                            # 8 points
        self.h_tgt = op["h"][self.pts].copy()
        self.a_norm = op["a"].copy()
        self.a_lo = 0.05
        self.a_hi = 1.3 * np.maximum(c.Hg[self.ci], self.a_norm[self.ci])
        self.majors = [o for o in c.offtake_list if o[3] >= MAJOR_MIN_Q]
        self.maj_seg = np.array([o[2] for o in self.majors])
        self.maj_names = [o[0] for o in self.majors]
        f = Q_NORM / 300.0
        self.maj_q0 = np.array([o[3] for o in self.majors]) * f
        self.q_minor = op["q_ord"].copy()
        for s, q in zip(self.maj_seg, self.maj_q0):
            self.q_minor[s] -= q
        self.n_obs, self.n_act = 32, len(self.ci)
        self.rng = np.random.default_rng(seed)
        self._init_cache = {}
        self.pool_area = self._pool_areas()
        self.c.max_halvings = 3

    # ------------------------------------------------------------------
    def _pool_areas(self):
        """Water-surface area of the pool upstream of each controlled gate (m2)."""
        c, h = self.c, self.op["h"]
        Tw = c.b + 2 * c.m * h
        seg_area = 0.5 * (Tw[1:] + Tw[:-1]) * np.diff(c.x)
        seg_area[c.is_struct] = 0.0
        areas, prev = [], 0
        for p in self.pts[:-1]:
            areas.append(seg_area[prev:p].sum()); prev = p + 1
        return np.array(areas)

    def initial_state(self, n_scale):
        """Steady state at Q_norm, targets held, for the given roughness multiplier."""
        key = round(n_scale, 4)
        if key not in self._init_cache:
            c = self.c
            c.n = self.n0 * n_scale
            if abs(n_scale - 1.0) < 1e-9:
                h, Q, a = self.op["h"].copy(), self.op["Q"].copy(), self.op["a"].copy()
            else:
                target = np.append(self.op["h"][c.u_nodes], self.op["h"][-1])
                h, Q = c.steady(Qin=Q_NORM, qoff=self.op["q_ord"], target=target,
                                h=self.op["h"].copy(), Q=self.op["Q"].copy(), n_steps=3000)
                u, d = c.u_nodes, c.u_nodes + 1
                dY = (c.z[u] + h[u]) - (c.z[d] + h[d])
                a = self.op["a"].copy()
                for k in self.ci:
                    a[k] = Q[u[k]] / (c.mu * c.W[k] * np.sqrt(2 * g * max(dY[k], 1e-9)))
                ke = c.n_struct          # tail gate: set so that the tail depth is also at its normal value
                a[ke] = Q[-1] / (c.mu * c.W[ke] * np.sqrt(2 * g * h[-1]))
                h, Q = c.steady(Qin=Q_NORM, qoff=self.op["q_ord"], a=a, h=h, Q=Q, n_steps=400)
            self._init_cache[key] = (h, Q, a)
        h, Q, a = self._init_cache[key]
        return h.copy(), Q.copy(), a.copy()

    def qoff(self, u, h):
        q = self.q_minor.copy()
        q[self.maj_seg] += u * self.maj_q0
        return q

    def actual_withdrawals(self, h, q):
        return self.c.actual_offtake(h, q)

    # ------------------------------------------------------------------
    def reset(self, scenario=None):
        if scenario is None:
            scenario, share, dQ = random_scenario(self.rng)
            # scheduled branch changes that match the head change (delay ~ distance / 25 km/h)
            sched = []
            for j in range(len(self.majors)):
                if abs(dQ) > 0 and share[j] > 0:
                    fac = 1.0 + dQ * share[j] / self.maj_q0[j]
                    fac = float(np.clip(fac, 0.5, 1.5))
                    delay = self.majors[j][1] / 25.0
                    sched.append((j, scenario.Qin_pts[1, 0] + delay, fac, 2.0))
            scenario.sched = sched
        self.sc = scenario
        self.c.n = self.n0 * scenario.n_scale
        self.h, self.Q, self.a = self.initial_state(scenario.n_scale)
        self.t = 0.0
        self.u = np.ones(len(self.majors))
        self.e_prev = self._err(self.h)
        self.failed = False
        self.log = dict(t=[0.0], hu=[self.h[self.pts].copy()], a=[self.a[self.ci].copy()],
                        Qin=[self.sc.Qin(0)], qd=[], qo=[], Qtail=[self.Q[-1]])
        return self.observe()

    def _err(self, h):
        return h[self.pts] - self.h_tgt

    def observe(self, noise=True):
        e = self._err(self.h)
        if noise and self.sc.noise > 0:
            e = e + self.rng.normal(0, self.sc.noise, e.shape)
        de = e - self.e_prev
        self._e_meas = e
        q = self.actual_withdrawals(self.h, self.qoff(self.u, self.h))
        qmaj = q[self.maj_seg] / self.maj_q0 - 1.0
        Qs = [self.sc.Qin(self.t + k * 3600) / Q_NORM - 1 for k in (0, 1, 2, 4)]
        o = np.concatenate([e / 0.1, de / 0.03, self.a[self.ci] / self.a_norm[self.ci] - 1.0,
                            np.array(Qs) * 5, qmaj * 5])
        return o.astype(float)

    def step(self, da):
        """da: change of the 7 check-gate openings in metres (clipped to +-DA_MAX and the limits)."""
        da = np.clip(np.asarray(da, float), -DA_MAX, DA_MAX)
        a_new = self.a.copy()
        a_new[self.ci] = np.clip(self.a[self.ci] + da, self.a_lo, self.a_hi)
        moved = a_new[self.ci] - self.a[self.ci]
        self.a = a_new
        self.e_prev = self._e_meas.copy()
        cost_e = 0.0
        n_sub = int(CTRL / DT)
        for _ in range(n_sub):
            self.t += DT
            self.u = self.sc.factors(self.t, len(self.majors))
            q = self.qoff(self.u, self.h)
            self.c.reset_diag()
            with np.errstate(all="ignore"):
                h, Q = self.c.step(self.h, self.Q, DT, self.sc.Qin(self.t), qoff=q, a=self.a)
            if (not (np.isfinite(h).all() and np.isfinite(Q).all()) or h.min() < 0.05
                    or self.c.diag["unconverged"] > 0):
                self.failed = True
                break
            self.h, self.Q = h, Q
            e = self._err(h)[:-1]
            cost_e += np.mean(np.minimum((e / E_SCALE) ** 2, E_CAP)) / n_sub
            qa = self.actual_withdrawals(h, q)
            self.log["t"].append(self.t); self.log["hu"].append(h[self.pts].copy())
            self.log["a"].append(self.a[self.ci].copy()); self.log["Qin"].append(self.sc.Qin(self.t))
            self.log["qd"].append(qa.copy()); self.log["qo"].append(q.copy())
            self.log["Qtail"].append(Q[-1])
        cost_m = W_MOVE * np.mean((moved / DA_MAX) ** 2)
        big = np.abs(self._err(self.h)[:-1]).max() > 0.5
        r = -(cost_e + cost_m) - (2.0 if big else 0.0)
        done = self.t >= T_EP - 1e-6 or self.failed
        if self.failed:
            r -= 50.0
        obs = self.observe()
        return obs, float(np.clip(r, -60, 0)), done, {}


if __name__ == "__main__":
    import time
    env = CanalEnv(seed=1)
    print("majors:", env.maj_names, np.round(env.maj_q0, 1))
    print("pool areas (km2):", np.round(env.pool_area / 1e6, 3))
    for n in (0.9, 1.0, 1.1, 1.2):
        h, Q, a = env.initial_state(n)
        print("n x%.1f  init err max %.3f m  a=%s" % (n, np.abs(h[env.pts] - env.h_tgt).max(), np.round(a[env.ci], 2)))
    o = env.reset()
    t = time.time(); R = 0; k = 0
    done = False
    while not done:
        o, r, done, _ = env.step(np.zeros(7)); R += r; k += 1
    print("zero-action episode: %d steps, return %.1f, %.2f s, failed %s" % (k, R, time.time() - t, env.failed))
    print("max |e| per point:", np.round(np.abs(np.array(env.log["hu"]) - env.h_tgt).max(0), 3))
