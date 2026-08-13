"""make_figs.py -- render the robotics paper figures from measured data.

Every panel is drawn from robotics.json (the output of flywheel.py), so
each figure is the measured number reported in the paper.

Usage:  python make_figs.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arm import Arm2DOF, pd_ctrl, make_reference          # noqa: E402
import identify                                          # noqa: E402
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "robotics.json")))

INK = "#1a1a1a"
WEAK = "#c1272d"     # crimson
FD = "#377eb8"       # blue
BATCH = "#7f7f7f"    # gray
ACCENT = "#e8a33d"   # amber

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.linewidth": 0.9, "figure.facecolor": "white",
})


def mean_std(per, k):
    return np.mean([p[k] for p in per]), np.std([p[k] for p in per])


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    # panel a: the arm + socket
    ax = axes[0]
    l1, l2 = 0.5, 0.45
    q1, q2 = 0.4, 1.2
    base = np.array([0.0, 0.0])
    p1 = base + l1 * np.array([np.cos(q1), np.sin(q1)])
    p2 = p1 + l2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    ax.plot([base[0], p1[0]], [base[1], p1[1]], color=INK, lw=5, solid_capstyle="round")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=WEAK, lw=5, solid_capstyle="round")
    for pt, lab in ((base, "J1"), (p1, "J2")):
        ax.plot(*pt, "o", ms=9, color=INK, zorder=5)
        ax.annotate(lab, pt, textcoords="offset points", xytext=(7, 7), fontsize=11)
    ax.plot(*p2, "*", ms=16, color=ACCENT, zorder=5)
    ax.annotate("tip", p2, textcoords="offset points", xytext=(8, -16), fontsize=11)
    # socket
    sx, sy = float(R["contact"]["contact"]["sx"]), float(R["contact"]["contact"]["sy"])
    rs = R["contact"]["contact"]["rs"]
    circ = plt.Circle((sx, sy), rs, fill=False, edgecolor=FD, lw=2.2)
    ax.add_patch(circ)
    ax.annotate("socket", (sx, sy), textcoords="offset points", xytext=(10, -8),
                color=FD, fontsize=11)
    ax.set_xlim(-0.7, 1.2); ax.set_ylim(-0.6, 1.2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("a | the plant: gravity, friction, payload, socket", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # panel b: the momentum-form weak identity
    ax = axes[1]
    ax.axis("off")
    lines = [
        ("the law (exact, per joint)", 0.96, "black", "bold"),
        (r"$\tau_1 = A_0\,a_1 + B_0\,a_2 + B_2\,c_2(2a_1{+}a_2)$", 0.80, "black", "normal"),
        (r"$\quad\; +\, G_1\cos q_1 + G_2\cos(q_1{+}q_2) + b_1 v_1 + f_{c1}\mathrm{sgn}\,v_1$", 0.68, "black", "normal"),
        ("the weak form: window both sides (W is linear)", 0.52, "black", "bold"),
        (r"$W[\tau_1] = A_0\,y_{v_1} + B_0\,y_{v_2} + B_2\,y_{c_2(2v_1+v_2)}$", 0.38, WEAK, "normal"),
        (r"$\quad\; +\, G_1 W[\cos q_1] + \cdots$  with  $y_f \equiv \lambda(f - W[f])$", 0.26, WEAK, "normal"),
        ("no derivative of any signal  |  target is the windowed torque", 0.12, "black", "italic_style"),
    ]
    for text, y, c, w in lines:
        ax.text(0.04, y, text, fontsize=13, color=c, fontweight=("bold" if w == "bold" else "normal"),
                style=("italic" if w == "italic_style" else "normal"),
                transform=ax.transAxes)
    ax.set_title("b | the identification layer", fontsize=12)
    # panel c: the flywheel
    ax = axes[2]
    ax.axis("off")
    cx = [0.12, 0.5, 0.88, 0.5, 0.12, 0.5]
    cy = [0.5, 0.84, 0.5, 0.16, 0.5, 0.5]
    labels = ["robot logs\n(q, q\u0307, \u03c4)", "streaming weak-form\nlaw identification",
              "computed-torque\nfeedforward", "better tracking:\ncleaner data",
              "", "the flywheel"]
    for i in range(4):
        box = FancyBboxPatch((cx[i] - 0.17, cy[i] - 0.14), 0.34, 0.28,
                                 boxstyle="round,pad=0.012", fc="#f7f7f7",
                                 ec=INK, lw=1.2, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(cx[i], cy[i], labels[i], ha="center", va="center",
                fontsize=10.5, transform=ax.transAxes)
    for a_, b_ in ((0, 1), (1, 2), (2, 3), (3, 0)):
        ax.annotate("", xy=(cx[b_], cy[b_]), xytext=(cx[a_], cy[a_]),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color=INK),
                    transform=ax.transAxes)
    ax.set_title("c | the operations data flywheel", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_paradigm.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    noises = list(R["law_recovery"]["noises"].keys())
    ns = np.array([float(x) for x in noises])
    w_ang, w_ang_s = [], []
    w_tau, w_tau_s = [], []
    f_ang, f_ang_s = [], []
    f_tau, f_tau_s = [], []
    b_ang, b_ang_s = [], []
    b_tau, b_tau_s = [], []
    for nf in noises:
        per = R["law_recovery"]["noises"][nf]
        w_ang.append(np.mean([p["weak"]["angle"] for p in per]))
        w_ang_s.append(np.std([p["weak"]["angle"] for p in per]))
        w_tau.append(np.mean([p["weak"]["nmse_tau"] for p in per]))
        w_tau_s.append(np.std([p["weak"]["nmse_tau"] for p in per]))
        f_ang.append(np.mean([p["fd"]["angle"] for p in per]))
        f_ang_s.append(np.std([p["fd"]["angle"] for p in per]))
        f_tau.append(np.mean([p["fd"]["nmse_tau"] for p in per]))
        f_tau_s.append(np.std([p["fd"]["nmse_tau"] for p in per]))
        b_ang.append(np.mean([p["batch"]["angle"] for p in per]))
        b_ang_s.append(np.std([p["batch"]["angle"] for p in per]))
        b_tau.append(np.mean([p["batch"]["nmse_tau"] for p in per]))
        b_tau_s.append(np.std([p["batch"]["nmse_tau"] for p in per]))
    ax = axes[0]
    ax.errorbar(ns, w_ang, yerr=w_ang_s, marker="o", lw=2, color=WEAK,
                label="weak form", capsize=3)
    ax.errorbar(ns, f_ang, yerr=f_ang_s, marker="s", lw=2, color=FD,
                label="finite differences", capsize=3)
    ax.errorbar(ns, b_ang, yerr=b_ang_s, marker="^", lw=2, color=BATCH,
                label="batch oracle", capsize=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("observational noise (fraction of signal RMS)")
    ax.set_ylabel("law direction error (degrees)")
    ax.set_title("a | the mechanism stays readable", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axes[1]
    ax.errorbar(ns, w_tau, yerr=w_tau_s, marker="o", lw=2, color=WEAK, capsize=3)
    ax.errorbar(ns, f_tau, yerr=f_tau_s, marker="s", lw=2, color=FD, capsize=3)
    ax.errorbar(ns, b_tau, yerr=b_tau_s, marker="^", lw=2, color=BATCH, capsize=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("observational noise (fraction of signal RMS)")
    ax.set_ylabel("torque-domain law-fit NMSE")
    ax.set_title("b | torque prediction from the law", fontsize=12)
    # panel c: identified vs true coefficients (representative seed, 5% noise)
    ax = axes[2]
    arm_ = Arm2DOF(mp=1.2)
    th_star = arm_.theta_star()
    x = np.arange(16)
    w = 0.35
    ax.bar(x - w / 2, th_star.reshape(-1), w, color=BATCH, label=r"true $\theta^*$")
    idf = identify.WeakFormID(80.0, 1e-3, dt=2e-3)
    logs = arm_.run(pd_ctrl(arm_, ff_model=None, ref=make_reference()), 6000,
                    noise_frac=0.05, seed=0)
    res = idf.fit(logs["q"], logs["v"], logs["tau"], t_lo=0, t_hi=3900)
    ax.bar(x + w / 2, res["law"].reshape(-1), w, color=WEAK,
           label=r"identified $\hat{\theta}$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{j}.{k}" for j in (1, 2) for k in range(8)],
                       fontsize=8, rotation=45)
    ax.set_ylabel("coefficient (SI)")
    ax.set_title("c | coefficients are the mechanism (5% noise)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_law.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    laps_w = R["flywheel"]["laps"]["weak"]
    laps_f = R["flywheel"]["laps"]["fd"]
    n_laps = len(laps_w[0])
    L = np.arange(n_laps)
    def series(laps, key):
        return np.array([[l[key] for l in seed_laps] for seed_laps in laps])
    tw = series(laps_w, "track"); tf = series(laps_f, "track")
    aw = series(laps_w, "angle"); af = series(laps_f, "angle")
    ax = axes[0]
    ax.errorbar(L, tw.mean(0), yerr=tw.std(0), marker="o", lw=2, color=WEAK,
                label="weak form", capsize=3)
    ax.errorbar(L, tf.mean(0), yerr=tf.std(0), marker="s", lw=2, color=FD,
                label="finite differences", capsize=3)
    ax.axvline(2.5, color=BATCH, ls="--", lw=1.2)
    ax.text(2.62, ax.get_ylim()[1] * 0.55, "payload +1.5 kg", fontsize=9,
            color=BATCH, rotation=90)
    ax.set_xticks(L)
    ax.set_xlabel("flywheel lap")
    ax.set_ylabel("tracking NMSE")
    ax.set_title("a | the flywheel spins (weak), stalls (FD)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax = axes[1]
    ax.errorbar(L, aw.mean(0), yerr=aw.std(0), marker="o", lw=2, color=WEAK,
                capsize=3, label="weak form")
    ax.errorbar(L, af.mean(0), yerr=af.std(0), marker="s", lw=2, color=FD,
                capsize=3, label="finite differences")
    ax.axvline(2.5, color=BATCH, ls="--", lw=1.2)
    ax.set_xticks(L)
    ax.set_xlabel("flywheel lap")
    ax.set_ylabel("law direction error (degrees)")
    ax.set_title("b | the law quality after each lap", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_flywheel.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fl = R["fleet"]
    names = fl["names"]
    am = np.array(fl["angle_mat"])
    ax = axes[0]
    im = ax.imshow(am, cmap="Reds", vmin=0, vmax=np.percentile(am, 90))
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{am[i, j]:.0f}", ha="center", va="center",
                    fontsize=8, color=INK)
    ax.set_title("a | law-vector angles across the fleet (deg)", fontsize=12)
    ax = axes[1]
    mp = fl["mp_est"]
    names_p = ["payload-1", "payload-2"]
    true_mp = {"payload-1": 1.2, "payload-2": 2.4}
    x = np.arange(2)
    ax.bar(x - 0.2, [true_mp[n] for n in names_p], 0.4, color=BATCH,
           label="true payload")
    ax.bar(x + 0.2, [mp[n] for n in names_p], 0.4, color=WEAK,
           label="read from the law")
    ax.set_xticks(x); ax.set_xticklabels(["1.2 kg payload", "2.4 kg payload"])
    ax.set_ylabel("payload mass (kg)")
    ax.set_title("b | the law reads the payload mass", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.text(0.02, 0.9, f"direction cosine: payload {fl['payload_cos']:.2f}, "
            f"wear {fl['wear_cos']:.2f}", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_fleet.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig5():
    c = R["contact"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: metric summary bars
    ax = axes[0]
    cats = ["baseline\nresidual (Nm)", "peak at\ncontact (Nm)", "detector\nSNR"]
    wv = [c["base_weak"], c["peak_weak"], c["spike_ratio_weak"]]
    fv = [c["base_fd"], c["peak_fd"], c["spike_ratio_fd"]]
    x = np.arange(3)
    ax.bar(x - 0.2, wv, 0.4, color=WEAK, label="weak form")
    ax.bar(x + 0.2, fv, 0.4, color=FD, label="finite differences")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("residual magnitude")
    ax.set_title("a | the law as a contact detector", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    # right: schematic of the mechanism
    ax = axes[1]
    ax.axis("off")
    lines = [
        ("identify once on an excitation run (no contact)", 0.86, "black", "normal"),
        ("freeze the nominal law", 0.74, "black", "normal"),
        ("then insert:", 0.62, "black", "normal"),
        (r"residual $= \tau - \hat{\theta}\!\cdot\!\Phi$", 0.50, WEAK, "bold"),
        (f"baseline {c['base_weak']:.1f} Nm  ->  {c['peak_weak']:.0f} Nm at contact", 0.36, WEAK, "normal"),
        (f"SNR {c['spike_ratio_weak']:.0f}x  (FD: {c['spike_ratio_fd']:.0f}x)", 0.24, WEAK, "normal"),
        (f"contact engaged {100*c['in_contact_frac']:.0f}% of the maneuver", 0.12, "black", "italic_style"),
    ]
    for text, y, col, w in lines:
        ax.text(0.05, y, text, fontsize=12.5, color=col,
                fontweight=("bold" if w == "bold" else "normal"),
                style=("italic" if w == "italic_style" else "normal"),
                transform=ax.transAxes)
    ax.set_title("b | contact detection, no extra sensors", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_contact.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 6
def fig6():
    de = R["data_efficiency"]
    curve = de["curve"]
    sec = np.array([x["sec"] for x in curve])
    w = np.array([x["weak"] for x in curve])
    f = np.array([x["fd"] for x in curve])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(sec, w, marker="o", lw=2, color=WEAK, label="weak form")
    ax.plot(sec, f, marker="s", lw=2, color=FD, label="finite differences")
    ax.set_xscale("log")
    ax.set_xlabel("logged data (seconds)")
    ax.set_ylabel("law direction error (degrees)")
    ax.set_title("data efficiency of law extraction (5% noise)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.annotate(f"LSTM: {de['lstm_nmse_tau1']:.3f} one-step NMSE\n"
                f"(4800 samples, 60 epochs, no mechanism)",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                fontsize=9, color=BATCH)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_data.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("figures written to", OUT)
