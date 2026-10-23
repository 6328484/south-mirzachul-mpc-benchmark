"""
PPO and TD3 in NumPy (no deep-learning library is needed)
=========================================================
MLP      : fully connected, tanh hidden layers, manual back-propagation, Adam optimiser.
PPO      : clipped surrogate (Schulman et al. 2017), Gaussian policy with a state-independent log-std,
           generalised advantage estimation, separate value network.
TD3      : twin critics, delayed actor and target updates, target-policy smoothing
           (Fujimoto et al. 2018), replay buffer; optional TD3+BC-style regulariser that keeps the
           residual action near zero, actor loss = -alpha Q / mean|Q| + mean(a^2) (Fujimoto & Gu 2021).
Actions are normalised to [-1, 1]; the environment wrapper multiplies them by DA_MAX.
"""
import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
            s = np.sqrt(2.0 / (m + n)) * (out_scale if i == len(sizes) - 2 else 1.0)
            self.W.append(rng.normal(0, s, (m, n)))
            self.b.append(np.zeros(n))

    def params(self):
        return self.W + self.b

    def set_params(self, ps):
        k = len(self.W)
        self.W = [p.copy() for p in ps[:k]]; self.b = [p.copy() for p in ps[k:]]

    def forward(self, x):
        self.cache = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            self.cache.append(x)
        return x

    def backward(self, dout):
        """Gradients of sum(dout * output); returns (grads like params(), d input)."""
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            if i < len(self.W) - 1:
                d = d * (1 - self.cache[i + 1] ** 2)
            gW[i] = self.cache[i].T @ d
            gb[i] = d.sum(0)
            d = d @ self.W[i].T
        return gW + gb, d


class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, max_norm=0.5):
        self.lr, self.b1, self.b2, self.eps, self.max_norm = lr, b1, b2, eps, max_norm
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        if self.max_norm:
            n = np.sqrt(sum((g ** 2).sum() for g in grads))
            if n > self.max_norm:
                grads = [g * self.max_norm / n for g in grads]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t); vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ======================================================================================
class PPO:
    def __init__(self, n_obs, n_act, seed=0, hidden=(64, 64), lr=3e-4, gamma=0.97, lam=0.95,
                 clip=0.2, epochs=10, minibatch=64, log_std0=-0.7, ent=0.0):
        self.rng = np.random.default_rng(seed)
        self.pi = MLP([n_obs, *hidden, n_act], self.rng, out_scale=0.01)
        self.vf = MLP([n_obs, *hidden, 1], self.rng)
        self.log_std = np.full(n_act, log_std0)
        self.opt_pi = Adam(self.pi.params() + [self.log_std], lr=lr)
        self.opt_v = Adam(self.vf.params(), lr=lr * 3)
        self.gamma, self.lam, self.clip, self.epochs, self.mb, self.ent = gamma, lam, clip, epochs, minibatch, ent
        self.n_act = n_act

    def act(self, obs, deterministic=False):
        mu = self.pi.forward(obs[None])[0]
        if deterministic:
            return np.clip(mu, -1, 1), None, None
        a = mu + np.exp(self.log_std) * self.rng.normal(size=mu.shape)
        logp = self._logp(a[None], mu[None])[0]
        return a, logp, float(self.vf.forward(obs[None])[0, 0])

    def _logp(self, a, mu):
        s = np.exp(self.log_std)
        return -0.5 * (((a - mu) / s) ** 2).sum(1) - self.log_std.sum() - 0.5 * len(s) * np.log(2 * np.pi)

    def value(self, obs):
        return self.vf.forward(obs)[:, 0]

    def update(self, buf):
        O, A, LP, R, V, D = (np.array(buf[k]) for k in ("obs", "act", "logp", "rew", "val", "done"))
        Vlast = np.array(buf["val_next"])
        # GAE (episodes are concatenated; D marks the last step of an episode)
        T = len(R); adv = np.zeros(T); gae = 0.0
        for t in reversed(range(T)):
            nv = Vlast[t] if D[t] else (V[t + 1] if t + 1 < T else Vlast[t])
            nonterm = 0.0 if buf["term"][t] else 1.0
            delta = R[t] + self.gamma * nv * nonterm - V[t]
            gae = delta + self.gamma * self.lam * (0.0 if D[t] else 1.0) * gae
            adv[t] = gae
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(T); stats = []
        for _ in range(self.epochs):
            self.rng.shuffle(idx)
            for s0 in range(0, T, self.mb):
                b = idx[s0:s0 + self.mb]
                o, a, lp0, ad, rt = O[b], A[b], LP[b], adv[b], ret[b]
                mu = self.pi.forward(o)
                s = np.exp(self.log_std)
                lp = self._logp(a, mu)
                ratio = np.exp(lp - lp0)
                active = ~(((ad > 0) & (ratio > 1 + self.clip)) | ((ad < 0) & (ratio < 1 - self.clip)))
                dL_dlp = -(ad * ratio * active) / len(b)
                dmu = dL_dlp[:, None] * (a - mu) / s ** 2
                dls = (dL_dlp[:, None] * (((a - mu) / s) ** 2 - 1)).sum(0) - self.ent
                gp, _ = self.pi.backward(dmu)
                self.opt_pi.step(self.pi.params() + [self.log_std], gp + [dls])
                self.log_std = np.clip(self.log_std, -3.0, 0.5)
                v = self.vf.forward(o)[:, 0]
                gv, _ = self.vf.backward(((v - rt) / len(b))[:, None])
                self.opt_v.step(self.vf.params(), gv)
                stats.append((float(np.mean(np.abs(ratio - 1))), float(np.mean((v - rt) ** 2))))
        return dict(ratio_dev=float(np.mean([s[0] for s in stats])), v_loss=float(np.mean([s[1] for s in stats])),
                    std=float(np.exp(self.log_std).mean()))

    def get_state(self):
        return dict(pi=[p.copy() for p in self.pi.params()], log_std=self.log_std.copy(),
                    vf=[p.copy() for p in self.vf.params()])

    def set_state(self, st):
        self.pi.set_params(st["pi"]); self.log_std = st["log_std"].copy(); self.vf.set_params(st["vf"])


# ======================================================================================
class TD3:
    def __init__(self, n_obs, n_act, seed=0, hidden=(64, 64), lr_a=3e-4, lr_c=1e-3, gamma=0.97,
                 tau=0.005, batch=128, policy_noise=0.2, noise_clip=0.5, delay=2, expl=0.2,
                 buffer=200000, bc_alpha=0.0):
        self.rng = np.random.default_rng(seed)
        self.actor = MLP([n_obs, *hidden, n_act], self.rng, out_scale=0.1)
        self.q1 = MLP([n_obs + n_act, *hidden, 1], self.rng)
        self.q2 = MLP([n_obs + n_act, *hidden, 1], self.rng)
        self.actor_t = MLP([n_obs, *hidden, n_act], self.rng); self.actor_t.set_params(self.actor.params())
        self.q1_t = MLP([n_obs + n_act, *hidden, 1], self.rng); self.q1_t.set_params(self.q1.params())
        self.q2_t = MLP([n_obs + n_act, *hidden, 1], self.rng); self.q2_t.set_params(self.q2.params())
        self.opt_a = Adam(self.actor.params(), lr=lr_a, max_norm=1.0)
        self.opt_q1 = Adam(self.q1.params(), lr=lr_c, max_norm=10.0)
        self.opt_q2 = Adam(self.q2.params(), lr=lr_c, max_norm=10.0)
        self.gamma, self.tau, self.batch = gamma, tau, batch
        self.pn, self.nc, self.delay, self.expl = policy_noise, noise_clip, delay, expl
        self.n_act, self.n_obs = n_act, n_obs
        self.bc_alpha = bc_alpha        # TD3+BC-style anchor towards a = 0 (the PI action)
        self.B = dict(o=np.zeros((buffer, n_obs)), a=np.zeros((buffer, n_act)), r=np.zeros(buffer),
                      o2=np.zeros((buffer, n_obs)), d=np.zeros(buffer))
        self.size, self.ptr, self.cap, self.it = 0, 0, buffer, 0

    def _pi(self, net, o):
        return np.tanh(net.forward(o))

    def act(self, obs, deterministic=False):
        a = self._pi(self.actor, obs[None])[0]
        if not deterministic:
            a = a + self.expl * self.rng.normal(size=a.shape)
        return np.clip(a, -1, 1)

    def store(self, o, a, r, o2, d):
        i = self.ptr
        self.B["o"][i], self.B["a"][i], self.B["r"][i], self.B["o2"][i], self.B["d"][i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.cap; self.size = min(self.size + 1, self.cap)

    def update(self):
        self.it += 1
        idx = self.rng.integers(0, self.size, self.batch)
        o, a, r, o2, d = (self.B[k][idx] for k in ("o", "a", "r", "o2", "d"))
        noise = np.clip(self.pn * self.rng.normal(size=a.shape), -self.nc, self.nc)
        a2 = np.clip(self._pi(self.actor_t, o2) + noise, -1, 1)
        x2 = np.hstack([o2, a2])
        qt = np.minimum(self.q1_t.forward(x2)[:, 0], self.q2_t.forward(x2)[:, 0])
        y = r + self.gamma * (1 - d) * qt
        x = np.hstack([o, a])
        out = {}
        for q, opt in ((self.q1, self.opt_q1), (self.q2, self.opt_q2)):
            qv = q.forward(x)[:, 0]
            g, _ = q.backward(((qv - y) / self.batch)[:, None])
            opt.step(q.params(), g)
            out.setdefault("q_loss", float(np.mean((qv - y) ** 2)))
        if self.it % self.delay == 0:
            pre = self.actor.forward(o)
            ap = np.tanh(pre)
            qv = self.q1.forward(np.hstack([o, ap]))
            lam = 1.0 if self.bc_alpha <= 0 else self.bc_alpha / (np.abs(qv).mean() + 1e-6)
            _, dx = self.q1.backward(-lam * np.ones((self.batch, 1)) / self.batch)   # maximise Q1
            dreg = (2 * ap / (self.batch * self.n_act)) if self.bc_alpha > 0 else 0.0
            da = (dx[:, self.n_obs:] + dreg) * (1 - ap ** 2)
            ga, _ = self.actor.backward(da)
            self.opt_a.step(self.actor.params(), ga)
            for net, tgt in ((self.actor, self.actor_t), (self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(net.params(), tgt.params()):
                    pt *= (1 - self.tau); pt += self.tau * p
        return out

    def get_state(self):
        return dict(actor=[p.copy() for p in self.actor.params()])

    def set_state(self, st):
        self.actor.set_params(st["actor"])
