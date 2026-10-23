"""
Classical controllers
=====================
Fixed  : gates keep their initial openings (no control).
PI     : decentralised upstream-level PI control of each check gate (velocity form),
         da_k = Kp (e_k - e_k,prev) + Ki e_k  [m per control interval], e_k = measured depth error.
         Common gains, optionally scaled by the pool surface area (gain_k = gain x A_k / mean A);
         tuned by a grid search on the training scenarios (tune_pi.py).
MPC    : centralised linear model predictive control.
         Model: the Saint-Venant model linearised about the normal steady state (finite differences
         of the 30-min transition map; 264 states = depth and discharge at 132 nodes).
         Inputs: the 7 gate openings; measured disturbances: head discharge (schedule known over the
         horizon) and the 5 major-branch withdrawals (held at their measured value).
         Offset-free: the one-step prediction error of the state is assumed constant over the horizon.
         Cost: sum_k ||e_k / E||^2 + w ||da_k / DA_MAX||^2 over Np steps, Nc moves, |da| <= DA_MAX.
         The MPC receives the full state (depth and discharge at every node), an advantage the other
         controllers do not have; it is the model-based reference.
"""
import numpy as np
from scipy.optimize import minimize
from env import CTRL, DT, DA_MAX, E_SCALE, W_MOVE
from canal import Q_NORM


class Fixed:
    name = "Fixed"

    def reset(self, env):
        pass

    def act(self, env, obs):
        return np.zeros(env.n_act)


class PI:
    name = "PI"

    def __init__(self, kp=0.6, ki=0.15, area_scaled=False):
        self.kp, self.ki, self.area_scaled = kp, ki, area_scaled

    def reset(self, env):
        self.e_prev = env._e_meas[:-1].copy()
        self.s = env.pool_area / env.pool_area.mean() if self.area_scaled else np.ones(env.n_act)

    def act(self, env, obs):
        e = env._e_meas[:-1]
        da = self.s * (self.kp * (e - self.e_prev) + self.ki * e)
        self.e_prev = e.copy()
        return np.clip(da, -DA_MAX, DA_MAX)


class MPC:
    name = "MPC"

    def __init__(self, env, Np=16, Nc=6, w_move=None, eps=1e-4):
        self.Np, self.Nc = Np, Nc
        self.w = W_MOVE if w_move is None else w_move
        self._linearise(env, eps)
        self._condense()

    # ------------------------------------------------------------------
    def _transition(self, env, h, Q, a, Qin, qmaj):
        q = env.q_minor.copy()
        q[env.maj_seg] += qmaj
        c = env.c
        for _ in range(int(CTRL / DT)):
            h, Q = c.step(h, Q, DT, Qin, qoff=q, a=a)
        return np.concatenate([h, Q])

    def _linearise(self, env, eps):
        c = env.c
        c.n = env.n0.copy()
        h0, Q0, a0 = env.initial_state(1.0)
        self.x0 = np.concatenate([h0, Q0]); self.a0 = a0.copy()
        N = c.N
        base = self._transition(env, h0, Q0, a0, Q_NORM, env.maj_q0)
        nx = 2 * N
        A = np.zeros((nx, nx))
        for i in range(nx):
            x = self.x0.copy(); d = eps * (1.0 if i < N else 10.0)
            x[i] += d
            A[:, i] = (self._transition(env, x[:N], x[N:], a0, Q_NORM, env.maj_q0) - base) / d
        B = np.zeros((nx, env.n_act))
        for j, k in enumerate(env.ci):
            a = a0.copy(); a[k] += 1e-3
            B[:, j] = (self._transition(env, h0, Q0, a, Q_NORM, env.maj_q0) - base) / 1e-3
        Bd = np.zeros((nx, 1 + len(env.maj_q0)))
        Bd[:, 0] = (self._transition(env, h0, Q0, a0, Q_NORM + 1.0, env.maj_q0) - base) / 1.0
        for j in range(len(env.maj_q0)):
            qm = env.maj_q0.copy(); qm[j] += 1.0
            Bd[:, 1 + j] = (self._transition(env, h0, Q0, a0, Q_NORM, qm) - base) / 1.0
        self.A, self.B, self.Bd = A, B, Bd
        self.C = np.zeros((env.n_act, nx)); self.C[np.arange(env.n_act), env.pts[:-1]] = 1.0
        self.res_lin = float(np.abs(base - self.x0).max())       # steady state check

    def _condense(self):
        A, B, C, Np, Nc = self.A, self.B, self.C, self.Np, self.Nc
        nu = B.shape[1]
        Apow = [np.eye(A.shape[0])]
        for _ in range(Np):
            Apow.append(A @ Apow[-1])
        # Y_k (k=1..Np) = C A^k X0 + sum_{j<k} C A^(k-1-j) (B U_j + Bd D_j + w)
        self.Phi = np.vstack([C @ Apow[k] for k in range(1, Np + 1)])
        CAB = [C @ Apow[i] @ B for i in range(Np)]              # impulse responses
        self.CAB = CAB
        # U_j = U_prev + sum_{m <= min(j, Nc-1)} dU_m   -> Y = ... + G_prev U_prev + G dU
        S = [sum(CAB[i] for i in range(k)) for k in range(1, Np + 1)]  # step responses
        self.Gprev = np.vstack(S)
        G = np.zeros((Np * C.shape[0], Nc * nu))
        for k in range(1, Np + 1):
            for m in range(Nc):
                if m < k:
                    G[(k - 1) * C.shape[0]:k * C.shape[0], m * nu:(m + 1) * nu] = S[k - 1 - m]
        self.G = G
        self.CABd = [C @ Apow[i] @ self.Bd for i in range(Np)]
        self.CAw = [C @ Apow[i] for i in range(Np)]
        Wy = 1.0 / E_SCALE ** 2 / C.shape[0]
        Wu = self.w / DA_MAX ** 2 / nu
        self.H = 2 * (Wy * G.T @ G + Wu * np.eye(G.shape[1]))
        self.Wy = Wy

    # ------------------------------------------------------------------
    def reset(self, env):
        self.x_prev = None
        self.wd = np.zeros(self.A.shape[0])
        self.z_prev = np.zeros(self.G.shape[1])

    def _state(self, env):
        return np.concatenate([env.h, env.Q]) - self.x0

    def act(self, env, obs):
        X = self._state(env)
        U = env.a[env.ci] - self.a0[env.ci]
        qmaj = env.actual_withdrawals(env.h, env.qoff(env.u, env.h))[env.maj_seg] - env.maj_q0
        if self.x_prev is not None:                           # offset-free disturbance estimate
            pred = self.A @ self.x_prev + self.B @ self.u_prev + self.Bd @ self.d_prev
            self.wd = 0.7 * self.wd + 0.3 * (X - pred)
        Np, ny = self.Np, self.C.shape[0]
        D = [np.concatenate([[env.sc.Qin(env.t + (j + 0.5) * CTRL) - Q_NORM], qmaj]) for j in range(Np)]
        free = self.Phi @ X + self.Gprev @ U
        for k in range(1, Np + 1):
            acc = np.zeros(ny)
            for j in range(k):
                acc += self.CABd[k - 1 - j] @ D[j] + self.CAw[k - 1 - j] @ self.wd
            free[(k - 1) * ny:k * ny] += acc
        f = 2 * self.Wy * self.G.T @ free
        H = self.H
        z0 = np.concatenate([self.z_prev[env.n_act:], np.zeros(env.n_act)])
        res = minimize(lambda z: (0.5 * z @ H @ z + f @ z, H @ z + f), z0, jac=True,
                       method="L-BFGS-B", bounds=[(-DA_MAX, DA_MAX)] * len(z0),
                       options=dict(maxiter=200))
        z = res.x
        self.z_prev = z
        da = z[:env.n_act]
        self.x_prev, self.u_prev = X, U + da
        self.d_prev = D[0]
        return da


def run_episode(env, ctrl, scenario, policy_obs=False):
    """Run one episode; ctrl.act(env, obs) -> da. Returns the episode log and the return."""
    obs = env.reset(scenario)
    ctrl.reset(env)
    R, done, t_cpu = 0.0, False, []
    import time
    while not done:
        t0 = time.perf_counter()
        da = ctrl.act(env, obs)
        t_cpu.append(time.perf_counter() - t0)
        obs, r, done, _ = env.step(da)
        R += r
    log = {k: np.array(v) for k, v in env.log.items()}
    log["failed"] = env.failed
    log["return"] = R
    log["cpu_per_decision"] = float(np.mean(t_cpu))
    return log


class ResidualDRL:
    """Residual reinforcement-learning controller: the agent adds a learned correction to the
    tuned PI action, da = clip(da_PI + DA_MAX * a_agent, +-DA_MAX). The agent observes the 32
    canal measurements and the PI action (scaled by DA_MAX). With a_agent = 0 it is the PI."""

    def __init__(self, agent, kp, ki, name="DRL"):
        self.agent, self.pi, self.name = agent, PI(kp, ki), name

    def reset(self, env):
        self.pi.reset(env)

    def base(self, env):
        return self.pi.act(env, None)

    @staticmethod
    def obs(o, base):
        return np.concatenate([o, base / DA_MAX])

    def act(self, env, obs):
        b = self.base(env)
        a = self.agent.act(self.obs(obs, b), True)
        a = a[0] if isinstance(a, tuple) else a
        return np.clip(b + DA_MAX * np.clip(a, -1, 1), -DA_MAX, DA_MAX)


class MPCKF(MPC):
    """MPC with SCADA measurements only (practical variant).
    Measured: depths upstream of the 7 controlled checks and at the tail, gate openings, head discharge
    schedule and the withdrawals of the 5 major branches. The 264-state hydraulic state and 8 unknown
    pool disturbances (an extra withdrawal in each pool, constant model: p_{k+1} = p_k) are estimated by a
    steady-state Kalman filter on the augmented linear model; the MPC prediction holds the estimated
    disturbances constant over the horizon (offset-free MPC with disturbance model)."""
    name = "MPC-KF"

    def __init__(self, env, sig_x=0.002, sig_p=0.5, sig_y=0.01, **kw):
        super().__init__(env, **kw)
        from scipy.linalg import solve_discrete_are
        c = env.c
        h0, Q0, a0 = env.initial_state(1.0)
        base = self._transition(env, h0, Q0, a0, Q_NORM, env.maj_q0)
        segs = list(env.pts[:-1] - 1) + [c.N - 2]
        nx, npd = self.A.shape[0], len(segs)
        Bp = np.zeros((nx, npd))
        for j, s in enumerate(segs):
            q = env.q_minor.copy(); q[env.maj_seg] += env.maj_q0; q[s] += 1.0
            x = self._transition_q(env, h0, Q0, a0, Q_NORM, q)
            Bp[:, j] = x - base
        self.Bp = Bp
        Aa = np.block([[self.A, Bp], [np.zeros((npd, nx)), np.eye(npd)]])
        Ca = np.zeros((len(env.pts), nx + npd)); Ca[np.arange(len(env.pts)), env.pts] = 1.0
        Qw = np.diag(np.concatenate([np.full(nx, sig_x ** 2), np.full(npd, sig_p ** 2)]))
        Rv = np.eye(len(env.pts)) * sig_y ** 2
        P = solve_discrete_are(Aa.T, Ca.T, Qw, Rv)
        self.L = P @ Ca.T @ np.linalg.inv(Ca @ P @ Ca.T + Rv)      # filter (measurement-update) gain
        self.Aa, self.Ca, self.npd = Aa, Ca, npd
        self.CAp = [self.C @ np.linalg.matrix_power(self.A, i) @ Bp for i in range(self.Np)]

    def _transition_q(self, env, h, Q, a, Qin, q):
        c = env.c
        for _ in range(int(CTRL / DT)):
            h, Q = c.step(h, Q, DT, Qin, qoff=q, a=a)
        return np.concatenate([h, Q])

    def reset(self, env):
        super().reset(env)
        nx = self.A.shape[0]
        self.z = np.zeros(nx + self.npd)                       # estimate (deviation from normal state)
        self.started = False

    def act(self, env, obs):
        U = env.a[env.ci] - self.a0[env.ci]
        qmaj = env.actual_withdrawals(env.h, env.qoff(env.u, env.h))[env.maj_seg] - env.maj_q0
        y = env._e_meas + env.h_tgt - self.x0[env.pts]            # measured depth deviations
        if self.started:                                         # time update with the applied input
            self.z = self.Aa @ self.z
            self.z[:self.A.shape[0]] += self.B @ self.u_prev + self.Bd @ self.d_prev
        self.z = self.z + self.L @ (y - self.Ca @ self.z)        # measurement update
        self.started = True
        nx = self.A.shape[0]
        X, p = self.z[:nx], self.z[nx:]
        Np, ny = self.Np, self.C.shape[0]
        D = [np.concatenate([[env.sc.Qin(env.t + (j + 0.5) * CTRL) - Q_NORM], qmaj]) for j in range(Np)]
        free = self.Phi @ X + self.Gprev @ U
        for k in range(1, Np + 1):
            acc = np.zeros(ny)
            for j in range(k):
                acc += self.CABd[k - 1 - j] @ D[j] + self.CAp[k - 1 - j] @ p
            free[(k - 1) * ny:k * ny] += acc
        f = 2 * self.Wy * self.G.T @ free
        H = self.H
        z0 = np.concatenate([self.z_prev[env.n_act:], np.zeros(env.n_act)])
        res = minimize(lambda z: (0.5 * z @ H @ z + f @ z, H @ z + f), z0, jac=True,
                       method="L-BFGS-B", bounds=[(-DA_MAX, DA_MAX)] * len(z0), options=dict(maxiter=200))
        self.z_prev = res.x
        da = res.x[:env.n_act]
        self.u_prev, self.d_prev = U + da, D[0]
        return da
