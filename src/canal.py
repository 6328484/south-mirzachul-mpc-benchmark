"""
Canal model at the normal operating point
=========================================
The one-dimensional Saint-Venant model (Preissmann scheme, Newton iteration, gated check
structures) is the one built from the canal passport in gated_model.py (reused from the
South Mirzachul repository, same authors).

Operating point used here: normal head discharge Q_norm = 260 m3/s (passport). All offtake
orders are the design (Qmax) withdrawals scaled by 260/300, so the water balance of the canal is
the passport balance at normal flow. The target depth upstream of every controlled check
structure is the passport depth at Qmax (the calibrated depth), i.e. the constant-upstream-level
operation used on the canal. The gate openings that hold these targets in the steady state at
Q_norm are found with the calibration routine of gated_model (prescribed upstream depths).
"""
import numpy as np
from gated_model import GatedCanal, g

Q_NORM = 260.0


def build(dx=1000.0):
    c = GatedCanal(dx=dx)
    c.open_struct[5] = True            # PK1035: fully open (negative head in the passport calibration)
    h_max, Q_max = c.calibrate()       # gate openings and effective widths at Qmax (passport)
    c.offtake_mode = "orifice"         # q = q_ord * sqrt(h / h_ref)
    f = Q_NORM / 300.0
    q_ord = c.q_off * f                # offtake orders at normal flow
    # steady state at Q_norm with the target depths prescribed upstream of the controlled structures
    target = h_max[np.append(c.u_nodes, c.N - 1)].copy()
    c.Qhead_guess = Q_NORM
    h0, Q0 = h_max.copy(), Q_max * f
    h, Q = c.steady(Qin=Q_NORM, qoff=q_ord, target=target, h=h0, Q=Q0, n_steps=3000)
    u, d = c.u_nodes, c.u_nodes + 1
    dY = (c.z[u] + h[u]) - (c.z[d] + h[d])
    a = np.empty(c.n_struct + 1)
    a[:-1] = Q[u] / (c.mu * c.W[:-1] * np.sqrt(2 * g * np.maximum(dY, 1e-9)))
    a[-1] = Q[-1] / (c.mu * c.W[-1] * np.sqrt(2 * g * h[-1]))
    a[c.open_struct] = np.nan
    # check: free simulation with these openings stays at the targets
    h2, Q2 = c.steady(Qin=Q_NORM, qoff=q_ord, a=a, h=h.copy(), Q=Q.copy(), n_steps=200)
    err = np.abs(h2[np.append(u, c.N - 1)] - target)
    err = err[np.isfinite(np.append(a[:-1], a[-1]))]
    return dict(c=c, h=h2, Q=Q2, a=a, q_ord=q_ord, target=target, h_max=h_max,
                steady_err=float(err.max()))


if __name__ == "__main__":
    import time
    t = time.time()
    op = build()
    c = op["c"]
    print("built in %.1f s, nodes %d, steady check error %.2e m" % (time.time() - t, c.N, op["steady_err"]))
    print("gate openings at Q_norm:", np.round(op["a"], 3))
    print("upstream depths:", np.round(op["h"][np.append(c.u_nodes, c.N - 1)], 3))
    print("tail outflow %.2f m3/s, head %.1f" % (op["Q"][-1], op["Q"][0]))
