"""Figures of the manuscript (figures/*.png, 300 dpi)."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JNAME = {"fig_canal.png": "Fig1", "fig_learning.png": "Fig2", "fig_traces_T1.png": "Fig3", "fig_gates_T1.png": "Fig4",
         "fig_summary.png": "Fig5", "fig_traces_T4.png": "Fig6"}


def save(fig, name):
    fig.savefig(F(name), dpi=300)
    if name in JNAME:
        os.makedirs(os.path.join(ROOT, "figures", "journal"), exist_ok=True)
        fig.savefig(os.path.join(ROOT, "figures", "journal", JNAME[name] + ".pdf"))
        fig.savefig(os.path.join(ROOT, "figures", "journal", JNAME[name] + ".png"), dpi=600)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)
F = lambda n: os.path.join(ROOT, "figures", n)
# controller groups: label -> key in evaluation.json (agents: prefix, one entry per seed)
GROUP = {"Fixed": "Fixed", "PI": "PI", "PPO": "PPOPURE_s", "R-PPO": "PPO_s", "R-TD3": "TD3_s",
         "TD3": "TD3PURE_s", "MPC": "MPC", "MPC-KF": "MPC-KF"}
# palette checked with the dataviz validator in the bar order PI, R-PPO, R-TD3, PPO, MPC-KF (MPC neutral)
COL = {"PI": "#eb6834", "R-PPO": "#6250d6", "R-TD3": "#eda100", "PPO": "#1baf7a", "MPC-KF": "#2a78d6",
       "MPC": "#52514e", "TD3": "#e34948", "Fixed": "#a8a7a2"}
LS = {"MPC-KF": "-", "PI": "--", "PPO": "-", "R-TD3": "-.", "R-PPO": "-.", "MPC": ":", "TD3": "-.", "Fixed": ":"}
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "legend.frameon": False})


def load_eval():
    ev = json.load(open(R("evaluation_named.json")))
    rnd = json.load(open(R("evaluation_random.json")))
    ev["results"].update(rnd["results"])
    return ev


def keys(res_s, label):
    p = GROUP[label]
    return [k for k in res_s if (k.startswith(p) if p.endswith("_s") else k == p)]


def learning_curves():
    base = json.load(open(R("val_baselines.json")))
    panels = [("PPO", "ppopure"), ("R-PPO", "ppo"), ("R-TD3", "td3"), ("TD3", "td3pure")]
    fig, axs = plt.subplots(2, 2, figsize=(6.85, 5.0), sharey=True)
    for ax, (lab, tag) in zip(axs.flat, panels):
        logs = [json.load(open(p)) for p in sorted(glob.glob(R("logs", "%s_s[0-9]*.json" % tag)))]
        n = min(len(l["val_episode"]) for l in logs)
        x = np.array(logs[0]["val_episode"][:n])
        Y = np.array([l["val_return"][:n] for l in logs])
        for y in Y:
            ax.plot(x, y, color=COL[lab], lw=0.6, alpha=0.35)
        ax.plot(x, np.median(Y, 0), color=COL[lab], lw=2, label=lab)
        for name in ("Fixed", "PI", "MPC-KF"):
            ax.axhline(base[name], color=COL[name], ls=LS[name], lw=1.3, label=name)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title("(%s) %s, %d seeds" % ("abcd"[list(axs.flat).index(ax)], lab, len(logs)), loc="left", fontsize=8)
    for ax in axs[1]:
        ax.set_xlabel("Training episodes (48 h each)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Validation return")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[n], ls=LS[n], lw=1.3) for n in ("Fixed", "PI", "MPC-KF")]
    fig.legend(h, ["Fixed gates", "PI", "MPC-KF"], ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.96)); save(fig, "fig_learning.png"); plt.close(fig)


def median_seed(ev, scen, lab):
    ks = sorted(keys(ev["results"][scen], lab), key=lambda k: ev["results"][scen][k]["IAE"])
    return ks[len(ks) // 2] if ks else None


def traces(scen="T1", pts=(0, 2, 5), show=("PI", "PPO", "MPC-KF", "MPC")):
    tr = json.load(open(R("traces.json")))[scen]
    ev = load_eval()
    h_tgt = np.array(ev["h_tgt"]); names = ev["points"]
    fig, ax = plt.subplots(len(pts) + 1, 1, figsize=(6.85, 1.75 * (len(pts) + 1)), sharex=True)
    for lab in show:
        key = median_seed(ev, scen, lab) if GROUP[lab].endswith("_s") else lab
        d = tr[key]; t = np.array(d["t"]) / 3600; hu = np.array(d["hu"])
        for i, p in enumerate(pts):
            ax[i].plot(t, 100 * (hu[:, p] - h_tgt[p]), color=COL[lab], ls=LS[lab], lw=1.5,
                       label=lab + (" (median seed)" if GROUP[lab].endswith("_s") else ""))
    t = np.array(tr["MPC"]["t"]) / 3600
    ax[-1].plot(t, tr["MPC"]["Qin"], color="#52514e", lw=1.5)
    ax[-1].set_ylabel("Head Q (m$^3$/s)")
    ax[-1].set_title("(%s) head discharge" % "abcdef"[len(pts)], loc="left", fontsize=8)
    for i, p in enumerate(pts):
        ax[i].set_ylabel("Error (cm)")
        ax[i].set_title("(%s) upstream of %s" % ("abcdef"[i], names[p].replace("Check structure ", "").split(" (")[0]), loc="left", fontsize=8)
    ax[0].legend(ncol=4, fontsize=7, loc="upper right")
    ax[-1].set_xlabel("Time (h)")
    fig.tight_layout(); save(fig, "fig_traces_%s.png" % scen); plt.close(fig)


def summary_bars(labs=("PI", "R-PPO", "R-TD3", "PPO", "MPC-KF", "MPC")):
    ev = load_eval()["results"]
    scen = ["T1", "T2", "T3", "T4", "T5", "T6"]
    fig, ax = plt.subplots(1, 2, figsize=(6.85, 3.0))
    w = 0.8 / len(labs)
    for k, met in enumerate(("IAE", "MAE")):
        for j, lab in enumerate(labs):
            vals, lo, hi = [], [], []
            for s in scen:
                v = [ev[s][c][met] for c in keys(ev[s], lab)]
                m = np.median(v)
                vals.append(100 * m); lo.append(100 * (m - min(v))); hi.append(100 * (max(v) - m))
            xs = np.arange(len(scen)) + (j - (len(labs) - 1) / 2) * w
            ax[k].bar(xs, vals, w * 0.9, color=COL[lab], label=lab, hatch="///" if lab == "MPC" else None,
                      edgecolor="white" if lab == "MPC" else None, lw=0)
            if GROUP[lab].endswith("_s"):
                ax[k].errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor="#0b0b0b", lw=0.7, capsize=1.2)
        ax[k].set_xticks(range(len(scen))); ax[k].set_xticklabels(scen)
        ax[k].set_ylabel(met + " (cm)")
        ax[k].set_yscale("log")
    hs, ls_ = ax[0].get_legend_handles_labels()
    fig.legend(hs, ls_, fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "fig_summary.png"); plt.close(fig)


def gates(scen="T1", show=("PI", "PPO", "MPC-KF"), k=2):
    tr = json.load(open(R("traces.json")))[scen]
    ev = load_eval()
    fig, ax = plt.subplots(1, 1, figsize=(6.85, 2.4))
    for lab in show:
        key = median_seed(ev, scen, lab) if GROUP[lab].endswith("_s") else lab
        d = tr[key]; t = np.array(d["t"]) / 3600; a = np.array(d["a"])
        ax.plot(t, a[:, k], color=COL[lab], ls=LS[lab], lw=1.5, label=lab)
    ax.set_xlabel("Time (h)"); ax.set_ylabel("Opening of PK623+86 (m)")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout(); save(fig, "fig_gates_%s.png" % scen); plt.close(fig)


def canal_schematic():
    """Longitudinal schematic: bed and target water surface, controlled checks, major branches."""
    from env import CanalEnv
    env = CanalEnv(); c = env.c
    x = c.x / 1000
    fig, ax = plt.subplots(figsize=(6.85, 2.6))
    ax.plot(x, c.z, color="#52514e", lw=1.2, label="Bed")
    ax.plot(x, c.z + env.op["h"], color="#2a78d6", lw=1.6, label="Water surface, normal operation (260 m$^3$/s)")
    for k, p in zip(env.ci, env.pts[:-1]):
        ax.axvline(x[p], color="#8a8984", lw=0.6, ls="--")
        ax.text(x[p] - 0.9, c.z.max() + 8.0, c.sname[k].replace("Check structure ", "").split(" (")[0],
                rotation=90, fontsize=6.5, ha="right", va="bottom", color="#0b0b0b")
    for nm, km, seg, q in env.majors:
        ax.annotate("", xy=(km, c.z[seg] - 0.6), xytext=(km, c.z[seg] + 0.2),
                    arrowprops=dict(arrowstyle="-|>", color="#eb6834", lw=1.2))
        ax.text(km + 0.8, c.z[seg] - 1.1, nm.replace(" branch", "").replace(" (Akbulak", "\n(Akbulak"),
                fontsize=6, color="#52514e")
    ax.set_xlabel("Chainage from the head works (km)"); ax.set_ylabel("Elevation (m, local datum)")
    ax.set_ylim(c.z.min() - 2.5, c.z.max() + 14)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); save(fig, "fig_canal.png"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    canal_schematic()
    learning_curves()
    traces("T1"); traces("T4"); summary_bars(); gates("T1")
