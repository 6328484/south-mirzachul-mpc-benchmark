"""
South Mirzachul main canal: Saint-Venant model with gated check structures (Sect. 3.1)
=====================================================================================
Data: the canal passport as reproduced in Seytov A.J. (2018), Appendix 2; data/structures.csv.

Reaches: 10 (passport); reach 1 is 6.475 km long (corrected from the chainage), 123.7 km in total.
Check structures: at the downstream ends of reaches 2..10 (PK145, PK400, ..., PK1237); there is no
structure at the end of reach 1.

Equations (Preissmann scheme, theta-implicit, Newton iteration):
  ordinary segment:  dA/dt + dQ/dx + q_off = 0
                     dQ/dt + d(Q^2/A)/dx + g A dY/dx + g A Sf = 0,   Y = z + h
  check structure (upstream node u, downstream node d, same x):
                     Q_d = Q_u - q_s                       (offtakes located at the structure)
                     Q_u = mu * a * W * sign(dY) * sqrt(2 g |dY|),  dY = Y_u - Y_d   (Eq. 4.18 in Seytov 2018)
  tail structure:    Q = mu * a * W * sqrt(2 g h)   (free outflow; the tailwater level is unknown)
  When h < a (the gate does not touch the water), h is used instead of a: the flow passes over the sill.
  mu = 0.65 (as used by Seytov 2018 for this canal), a = gate opening, W = number of spans x span width.
Manning: Sf = n^2 Q|Q| / (A^2 R^(4/3))  (y = 1/6 in Seytov 2018: C = R^(1/6)/n).
The bed is continuous (the passport gives no bed steps or sill elevations at the structures).
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from paths import data_file

g = 9.81
MU = 0.65

PASSPORT = dict(
    L=[6.475, 8.025, 25.5, 22.386, 12.276, 17.64, 11.198, 5.223, 9.006, 5.971],  # km
    Q=[300, 290, 260, 230, 101, 78, 42, 37, 25, 14],                             # Qmax, m3/s
    T=[71.2, 70.4, 68.4, 66.9, 39.4, 35.44, 26.18, 20.36, 19.72, 18.6],          # top width at Qmax, m
    b=[18, 18, 18, 18, 8, 8, 3.5, 3, 3, 3],                                      # bottom width, m
    h=[7.15, 7.0, 6.77, 6.59, 5.25, 4.48, 3.67, 3.84, 3.68, 3.40],               # depth at Qmax, m
    h_norm=[7.05, 6.89, 6.66, 6.47, 5.20, 4.38, 3.53, 3.58, 3.41, 2.50],         # depth at Qnorm, m
    h_min=[6.64, 6.48, 6.23, 6.03, 4.70, 4.00, 2.88, 2.83, 2.47, 1.79],          # depth at Qmin, m
    S=[5e-5] * 4 + [6e-5, 7e-5] + [1e-4] * 4,                                    # bed slope
)
Q_HEAD = dict(max=300.0, norm=260.0, min=196.0)


def reach_table():
    P = {k: np.array(v, float) for k, v in PASSPORT.items()}
    m = (P["T"] - P["b"]) / (2 * P["h"])
    A = (P["b"] + m * P["h"]) * P["h"]
    R = A / (P["b"] + 2 * P["h"] * np.sqrt(1 + m**2))
    n = A * R ** (2 / 3) * np.sqrt(P["S"]) / P["Q"]
    return dict(P, m=m, A=A, n=n)


def load_structures(path=None):
    st = pd.read_csv(path or data_file("structures.csv"))
    checks = st[st.kind == "check"].sort_values("km").reset_index(drop=True)
    return st, checks


class GatedCanal:
    def __init__(self, dx=1000.0, theta=0.6, mu=MU):
        t = reach_table()
        self.tab, self.theta, self.mu = t, theta, mu
        st, checks = load_structures()
        self.checks = checks
        Lm = t["L"] * 1000
        edges = np.concatenate([[0], np.cumsum(Lm)])
        self.edges = edges

        # --- nodes: every reach; a duplicate node after the ends of reaches 2..9 (internal structures)
        xs, rid, seg_struct = [0.0], [0], []
        self.struct_reach = []          # reach at whose end the structure lies
        for i in range(10):
            k = max(1, int(round(Lm[i] / dx)))
            for j in range(1, k + 1):
                xs.append(edges[i] + Lm[i] * j / k)
                rid.append(i)
                seg_struct.append(-1)
            if 1 <= i <= 8:                       # internal check structure
                xs.append(edges[i + 1]); rid.append(i + 1)
                seg_struct.append(len(self.struct_reach))
                self.struct_reach.append(i)
        # the junction node at the end of reach 1 is assigned to reach 2 (as in the earlier model)
        self.x = np.array(xs); self.rid = np.array(rid)
        self.rid[np.isclose(self.x, edges[1])] = 1
        self.N = len(self.x)
        self.seg_struct = np.array(seg_struct)            # segment -> structure index, or -1
        self.is_struct = self.seg_struct >= 0
        self.n_struct = len(self.struct_reach)            # 8 internal structures (+1 tail structure in the arrays)
        self.b = t["b"][self.rid]; self.m = t["m"][self.rid]; self.n = t["n"][self.rid]
        dxs = np.diff(self.x)
        S_seg = t["S"][self.rid[1:]]
        S_seg = np.where(self.is_struct, 0.0, S_seg)
        self.z = 100.0 - np.concatenate([[0], np.cumsum(S_seg * dxs)])
        self.dx = np.where(self.is_struct, 1.0, dxs)      # not used at the structures

        # check structures: total span width (W) and gate height
        def chk(km):
            r = checks.iloc[int(np.argmin(abs(checks.km - km)))]
            return r.spans * r.gate_w_m, r.gate_h_m, r["name"]
        self.W, self.Hg, self.sname = [], [], []
        for i in self.struct_reach + [9]:
            w, hg, nm = chk(edges[i + 1] / 1000)
            self.W.append(w); self.Hg.append(hg); self.sname.append(nm)
        self.W = np.array(self.W, float); self.Hg = np.array(self.Hg, float)
        self.a = np.full(self.n_struct + 1, np.nan)        # gate openings (set by the calibration)
        self.open_struct = np.zeros(self.n_struct + 1, bool)  # True: fully open (Y_u = Y_d)
        self.offtake_mode = "fixed"                        # "fixed" or "orifice"
        self.h_ref = None                                  # reference depth for the orifice mode
        self.u_nodes = np.where(self.is_struct)[0]         # upstream node of each internal structure
        self.q_off = self._offtakes(st)

    # ---------------- offtakes ----------------
    def _offtakes(self, st):
        off = st[st.kind == "offtake"]
        t, e = self.tab, self.edges
        drops = -np.diff(t["Q"])
        q = np.zeros(self.N - 1)
        xm = 0.5 * (self.x[1:] + self.x[:-1])
        self.offtake_list = []                      # (name, km, segment, discharge m3/s)
        self.seepage = np.zeros(self.N - 1)         # distributed losses (seepage and unlisted withdrawals)
        for i, drop in enumerate(drops):
            a_, b_ = e[i], e[i + 1]
            sel = off[(off.km * 1000 > a_ + 1e-3) & (off.km * 1000 <= b_ + 1e-3)]
            cap = sel.Q_cap_m3s.sum()
            sc = min(1.0, drop / cap) if cap > 0 else 0.0
            for _, r in sel.iterrows():
                xk = r.km * 1000
                if abs(xk - b_) < 1.0 and i >= 1:      # offtake located at a check structure
                    seg = int(np.where(self.is_struct & np.isclose(xm, b_))[0][0])
                else:
                    seg = int(np.clip(np.searchsorted(self.x, xk) - 1, 0, self.N - 2))
                    if self.is_struct[seg]:
                        seg -= 1
                q[seg] += r.Q_cap_m3s * sc
                self.offtake_list.append((r["name"], r.km, seg, r.Q_cap_m3s * sc))
            rest = drop - cap * sc
            if rest > 1e-9:                             # seepage / withdrawals not in the list
                segs = np.where((xm > a_) & (xm < b_) & ~self.is_struct)[0]
                q[segs] += rest * self.dx[segs] / self.dx[segs].sum()
                self.seepage[segs] += rest * self.dx[segs] / self.dx[segs].sum()
        return q

    # ---------------- geometry ----------------
    def A(self, h): return (self.b + self.m * h) * h
    def R(self, h): return self.A(h) / (self.b + 2 * h * np.sqrt(1 + self.m**2))
    def Sf(self, h, Q):
        A = self.A(h)
        return self.n**2 * Q * np.abs(Q) / (A**2 * self.R(h) ** (4 / 3))

    def gate_Q(self, a, W, dY):
        return self.mu * a * W * np.sign(dY) * np.sqrt(2 * g * np.abs(dY) + 1e-8)

    # ---------------- residuals ----------------
    def actual_offtake(self, h, qoff):
        if self.offtake_mode == "fixed" or self.h_ref is None:
            return qoff
        hs = 0.5 * (h[:-1] + h[1:])
        hr = 0.5 * (self.h_ref[:-1] + self.h_ref[1:])
        return qoff * np.sqrt(np.clip(hs / hr, 0.0, None))

    def _residual(self, h1, Q1, h0, Q0, dt, Qin, qoff, a, target):
        th, dx = self.theta, self.dx
        qoff = self.actual_offtake(h1, qoff)
        A1, A0 = self.A(h1), self.A(h0)
        Y1, Y0 = self.z + h1, self.z + h0
        sl, sr = slice(0, -1), slice(1, None)
        avg = lambda v: 0.5 * (v[sl] + v[sr])
        dif = lambda v: (v[sr] - v[sl]) / dx
        Rc = (avg(A1) - avg(A0)) / dt + th * dif(Q1) + (1 - th) * dif(Q0) + qoff / dx
        F1, F0 = Q1**2 / A1, Q0**2 / A0
        S1 = g * avg(A1) * dif(Y1) + g * avg(A1 * self.Sf(h1, Q1))
        S0 = g * avg(A0) * dif(Y0) + g * avg(A0 * self.Sf(h0, Q0))
        Rm = (avg(Q1) - avg(Q0)) / dt + th * (dif(F1) + S1) + (1 - th) * (dif(F0) + S0)
        # check-structure segments
        u = self.u_nodes; d = u + 1; k = self.seg_struct[u]
        Rc[u] = Q1[d] - Q1[u] + qoff[u]
        if target is None:
            # if the depth is below the opening, the flow passes over the sill: a_eff = min(a, h_u)
            Rm[u] = Q1[u] - self.gate_Q(np.minimum(a[k], np.maximum(h1[u], 0)), self.W[k], Y1[u] - Y1[d])
        else:                                   # calibration: the upstream depth is prescribed
            Rm[u] = h1[u] - target[k]
        op = self.open_struct[k]
        if op.any():                            # fully open: equal water levels
            Rm[u[op]] = Y1[u[op]] - Y1[d[op]]
        res = np.empty(2 * self.N)
        res[0] = Q1[0] - Qin
        res[1:-1:2] = Rc
        res[2:-1:2] = Rm
        ke = self.n_struct
        if target is None:
            he = max(h1[-1], 1e-6)
            res[-1] = Q1[-1] - self.mu * min(a[ke], he) * self.W[ke] * np.sqrt(2 * g * he)
        else:
            res[-1] = h1[-1] - target[ke]
        return res

    # How often a non-converged time step may be halved (4: down to dt/16). The nominal training
    # (train_nominal.py) and the benchmark optimisation (benchmark.py) were run with an earlier version of
    # this solver that had no subdivision; those scripts set 0 to reproduce the stored runs exactly.
    max_halvings = 4

    def step(self, h0, Q0, dt, Qin, qoff=None, a=None, target=None, tol=1e-7, maxit=25, _depth=0):
        """One implicit time step. If Newton does not converge (residual > 1e-6 or not finite), the step
        is automatically split into two half-steps (at most max_halvings times, i.e. down to dt/16).
        Converged steps are not affected."""
        h, Q = self._newton(h0, Q0, dt, Qin, qoff, a, target, tol, maxit)
        q_ = self.q_off if qoff is None else qoff
        a_ = self.a if a is None else a
        with np.errstate(all="ignore"):
            res = np.max(np.abs(self._residual(h, Q, h0, Q0, dt, Qin, q_, a_, target)))
            ok = np.isfinite(h).all() and np.isfinite(Q).all() and res < 1e-6
        d = self.__dict__.setdefault("diag", dict(max_res=0.0, substeps=0, unconverged=0))
        if ok or _depth >= self.max_halvings:
            d["max_res"] = max(d["max_res"], float(res) if np.isfinite(res) else np.inf)
            if not ok:
                d["unconverged"] += 1
            return h, Q
        d["substeps"] += 1
        hm, Qm = self.step(h0, Q0, dt / 2, Qin, qoff, a, target, tol, maxit, _depth + 1)
        return self.step(hm, Qm, dt / 2, Qin, qoff, a, target, tol, maxit, _depth + 1)

    def reset_diag(self):
        self.diag = dict(max_res=0.0, substeps=0, unconverged=0)

    def _newton(self, h0, Q0, dt, Qin, qoff=None, a=None, target=None, tol=1e-7, maxit=25):
        qoff = self.q_off if qoff is None else qoff
        a = self.a if a is None else a
        h, Q = h0.copy(), Q0.copy()
        N, eps = self.N, 1e-6
        for _ in range(maxit):
            r = self._residual(h, Q, h0, Q0, dt, Qin, qoff, a, target)
            rows, cols, vals = [], [], []
            for var in (0, 1):
                for c in range(3):
                    idx = np.arange(c, N, 3)
                    hp, Qp = h.copy(), Q.copy()
                    (hp if var == 0 else Qp)[idx] += eps
                    dr = (self._residual(hp, Qp, h0, Q0, dt, Qin, qoff, a, target) - r) / eps
                    for off in range(-2, 4):
                        rr = 2 * idx + off
                        ok = (rr >= 0) & (rr < 2 * N)
                        rr, cc = rr[ok], (2 * idx + var)[ok]
                        v = dr[rr]; nz = v != 0
                        rows.append(rr[nz]); cols.append(cc[nz]); vals.append(v[nz])
            J = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(2 * N, 2 * N)).tocsc()
            d = spsolve(J, -r)
            h += d[0::2]; Q += d[1::2]
            if np.max(np.abs(d)) < tol:
                break
        return h, Q

    # ---------------- initial state, steady flow ----------------
    def initial_state(self, depth=None):
        depth = self.tab["h"] if depth is None else np.asarray(depth)
        h = depth[self.rid].astype(float).copy()
        Q = self.Qhead_guess - np.concatenate([[0], np.cumsum(self.q_off)])
        return h, Q

    Qhead_guess = 300.0

    def steady(self, Qin=300.0, qoff=None, a=None, target=None, h=None, Q=None, dt=1800.0, n_steps=600):
        self.Qhead_guess = Qin
        if h is None:
            h, Q = self.initial_state()
        for _ in range(n_steps):
            hn, Qn = self.step(h, Q, dt, Qin, qoff, a, target)
            done = np.max(abs(hn - h)) < 1e-6 and np.max(abs(Qn - Q)) < 1e-5
            h, Q = hn, Qn
            if done:
                break
        return h, Q

    def calibrate(self, levels=None):
        """Find the gate openings a that give the passport depth upstream of each structure at Qmax."""
        t = self.tab
        levels = t["h"] if levels is None else np.asarray(levels)
        target = np.array([levels[i] for i in self.struct_reach] + [levels[9]])
        h, Q = self.steady(Qin=t["Q"][0], target=target)
        u, d = self.u_nodes, self.u_nodes + 1
        dY = (self.z[u] + h[u]) - (self.z[d] + h[d])
        a = np.empty(self.n_struct + 1)
        a[:-1] = Q[u] / (self.mu * self.W[:-1] * np.sqrt(2 * g * np.maximum(dY, 1e-9)))
        a[-1] = Q[-1] / (self.mu * self.W[-1] * np.sqrt(2 * g * h[-1]))
        a[self.open_struct] = np.nan
        # Structures where Qmax cannot pass at the passport depth with the passport gate dimensions
        # (a > 0.8 h): the bed and sill elevations at the structures are unknown. There the effective
        # width W is increased so that a = 0.8 h (the calibrated discharge is unchanged and a margin
        # for control remains).
        hu = np.append(h[u], h[-1])
        self.W_passport = self.W.copy()
        lim = 0.8 * hu
        fix = np.isfinite(a) & (a > lim)
        self.W[fix] = self.W[fix] * a[fix] / lim[fix]
        a[fix] = lim[fix]
        self.W_adjusted = fix
        self.a = a
        self.h_ref = h.copy()
        self.calib = dict(h=h, Q=Q, dY=np.append(dY, np.nan))
        return h, Q
