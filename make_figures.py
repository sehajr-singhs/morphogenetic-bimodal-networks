"""make_figures.py -- render the paper figures from measured data.

Figures are produced from the actual saved result files (results.json,
ablation.json, scaling.json) and one representative seed-0 run of the network
(for the spatial maps), so every panel is grounded in the measured numbers
reported in the manuscript.

Usage:  python make_figures.py [--out figs]
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biomaterial_net import MorphogeneticNet, make_grid, default_cfg

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figs"
os.makedirs(OUT, exist_ok=True)

R = json.load(open("results.json"))
A = json.load(open("ablation.json"))
S = json.load(open("scaling.json"))


def mean_std(s, k):
    v = s[k]
    return v["mean"], v["std"]


# ---------------------------------------------------------------- Fig 1+3:
# representative run for the spatial maps
def representative_run(seed=0, T=3000):
    pos, edges = make_grid(7, 7)
    cfg = default_cfg("weak")
    cfg["sources"] = (0, 6, 42, 48)
    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(seed))
    for _ in range(T):
        net.step(sigma=0.05, record=True)
    return net


net = representative_run()
kappa, a_ij, cbar_e, kappa0 = net._compute_kappa()
w = net._w
pos, edges = net.pos, net.edges

# ---- Fig 1: architecture / learned material state
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

# panel a: chemical corridor (the neuromodulator wind imprints a corridor)
ax = axes[0]
sc = ax.scatter(pos[:, 0], pos[:, 1], c=net.c, s=90, cmap="inferno",
                edgecolors="k", linewidths=0.4, vmin=0, vmax=3.0)
for ei, ej in edges:
    ax.plot([pos[ei, 0], pos[ej, 0]], [pos[ei, 1], pos[ej, 1]],
            color="0.75", lw=0.8, zorder=0)
ax.quiver(pos[:, 0], pos[:, 1], net.vhat[:, 0], net.vhat[:, 1],
          color="tab:blue", scale=14, width=0.0045, alpha=0.9)
ax.set_title("(a) Chemical corridor + structural vectors\n"
             "(neuromodulator drift imprints the axis)")
ax.set_xlabel("x (grid units)"); ax.set_ylabel("y")
fig.colorbar(sc, ax=ax, fraction=0.046, label="c")

# panel b: identified vs geometric coupling on the same edges
ax = axes[1]
lw = 0.5 + 2.5 * (w - w.min()) / max(w.max() - w.min(), 1e-12)
for ei, ej, l in zip(edges[:, 0], edges[:, 1], lw):
    ax.plot([pos[ei, 0], pos[ej, 0]], [pos[ei, 1], pos[ej, 1]],
            color="0.45", lw=l, zorder=1)
sc = ax.scatter(pos[:, 0], pos[:, 1], c=net.c, s=60, cmap="inferno",
                edgecolors="k", linewidths=0.4, vmin=0, vmax=3.0)
ax.set_title("(b) Identified per-edge coupling w\n(line width = |w|)")
ax.set_xlabel("x (grid units)"); ax.set_ylabel("y")

# panel c: geometric coupling a_ij (the shape) on the same edges
ax = axes[2]
lw = 0.5 + 2.5 * a_ij
for ei, ej, l in zip(edges[:, 0], edges[:, 1], lw):
    ax.plot([pos[ei, 0], pos[ej, 0]], [pos[ei, 1], pos[ej, 1]],
            color="tab:green", lw=l, zorder=1)
sc = ax.scatter(pos[:, 0], pos[:, 1], c=net.c, s=60, cmap="inferno",
                edgecolors="k", linewidths=0.4, vmin=0, vmax=3.0)
ax.set_title("(c) Geometric coupling a_ij from the angles\n"
             "(the shape is the interpretable readout)")
ax.set_xlabel("x (grid units)"); ax.set_ylabel("y")

fig.suptitle("Figure 1 | The bi-modal morphogenetic network: chemistry imprints "
             "a corridor, angles encode it as a readable shape")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/fig1_architecture.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: learning curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# panel a: physical tracking on a noisy observation segment
t = np.arange(len(net.u_true_hist)) * net.dt
ax = axes[0]
ax.plot(t[800:1200], np.array(net.u_true_hist)[800:1200, 5], "k", lw=1.2,
        label="true")
ax.plot(t[800:1200], np.array(net.u_obs_hist)[800:1200, 5], ".", ms=2.5,
        color="0.6", label="observed (5% noise)")
ax.plot(t[800:1200], np.array(net.pred_hist)[800:1200, 5], "tab:red", lw=1.0,
        label="streaming weak-form prediction")
ax.set_title("(a) Physical one-step tracking\n(held-out window)")
ax.set_xlabel("t (s)"); ax.set_ylabel("u")
ax.legend(fontsize=8)

# panel b: law-fit NMSE (streaming vs finite-difference)
nmse_y_series = np.array(net.resid2_hist) / np.maximum(
    np.array(net.y2_hist), 1e-12)
ax = axes[1]
ax.semilogy(t, nmse_y_series, "tab:blue", lw=1.4,
            label="streaming weak form")
ax.axhline(0.2, color="0.5", ls="--", lw=0.9)
ax.set_title("(b) Law-fit NMSE over time\n(weak form vs finite difference)")
ax.set_xlabel("t (s)"); ax.set_ylabel("NMSE (law fit)")
# FD identifier on the same observations for the curve
cfg_fd = default_cfg("fd")
fd = MorphogeneticNet(pos, edges, cfg_fd, np.random.default_rng(0))
for uo in net.u_obs_hist:
    fd.observe(uo, np.zeros(net.N))
nmse_fd_series = np.array(fd.resid2_hist) / np.maximum(
    np.array(fd.y2_hist), 1e-12)
ax.semilogy(t, nmse_fd_series, "tab:orange", lw=1.2,
            label="finite-difference RLS")
ax.legend(fontsize=8)

# panel c: morphogenesis (alignment self-organizes)
ax = axes[2]
al = np.array(net.align_hist)
ax.plot(t, al, "tab:green", lw=1.4)
ax.set_title("(c) Structural alignment self-organizes\n"
             "(0.5 = random, 1.0 = coherent)")
ax.set_xlabel("t (s)"); ax.set_ylabel("mean pairwise alignment")
ax.set_ylim(0.4, 1.02)

fig.suptitle("Figure 2 | Streaming identification at 5% observational noise "
             "and self-organized morphogenesis")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/fig2_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Fig 3: read-back scatter (shape vs weights vs law)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
sw = np.random.default_rng(0).choice(len(w), size=60, replace=False)

ax = axes[0]
ax.scatter(kappa, w, s=14, alpha=0.7, color="tab:blue")
ax.set_xlabel(r"true law $\kappa$"); ax.set_ylabel("identified w")
ax.set_title("(a) Weights vs law\n"
             f"corr(w, κ) = {np.corrcoef(w, kappa)[0, 1]:.2f}")

ax = axes[1]
ax.scatter(kappa, a_ij, s=14, alpha=0.7, color="tab:green")
ax.set_xlabel(r"true law $\kappa$"); ax.set_ylabel(r"geometric $a_{ij}$")
ax.set_title("(b) Shape vs law\n"
             f"corr(a_ij, κ) = {np.corrcoef(a_ij, kappa)[0, 1]:.2f}")

ax = axes[2]
ax.scatter(w, a_ij, s=14, alpha=0.7, color="tab:purple")
ax.set_xlabel("identified w"); ax.set_ylabel(r"geometric $a_{ij}$")
ax.set_title("(c) Shape vs weights\n"
             f"corr(a_ij, w) = {np.corrcoef(a_ij, w)[0, 1]:.2f}")

fig.suptitle("Figure 3 | The shape is the denoised spatial readout of the law\n"
             "(seed 0; 15-seed means in the manuscript)")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/fig3_readback.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Fig 4: ablation bar chart
variants = ["full", "no_geometry", "no_chemgate", "no_consolidation",
            "no_morphogenesis", "no_chemistry"]
labels = ["full\nmodel", "no\ngeometry", "no chem\ngate",
          "no\nconsolidation", "no\nmorphogenesis", "no\nchemistry"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

def bars(ax, key, ylabel, title, scale=1.0):
    means = [A["summary"][v][key]["mean"] * scale for v in variants]
    stds = [A["summary"][v][key]["std"] * scale for v in variants]
    ax.bar(range(len(variants)), means, yerr=stds, capsize=3,
           color=["tab:green"] + ["0.75"] * 5)
    ax.set_xticks(range(len(variants))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(title)

bars(axes[0], "nmse_y", "law-fit NMSE", "(a) Identification survives\nevery ablation")
bars(axes[1], "trans_gain", "transmission gain", "(b) Routing requires the\nlearned geometric channel")
bars(axes[2], "corr_focus", "corridor contrast", "(c) Chemical gate makes the\nshape specific")

fig.suptitle("Figure 4 | Ablation: each mechanism earns its place\n"
             "(8 seeds per variant)")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/fig4_ablation.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Fig 5: scaling
sizes = ["7x7", "11x11", "15x15"]
n_nodes = [49, 121, 225]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
for key, lab, col in [("nmse_y", "local (streaming)", "tab:blue"),
                      ("nmse_y_oracle", "global batch oracle", "tab:red"),
                      ("nmse_y_global", "streaming-global RLS", "tab:purple"),
                      ("nmse_y_fd", "finite-difference RLS", "tab:orange")]:
    m = [S["summary"][s][key]["mean"] for s in sizes]
    ax.plot(n_nodes, m, "o-", label=lab, color=col)
ax.set_xlabel("nodes N"); ax.set_ylabel("law-fit NMSE")
ax.set_title("(a) Identification advantage persists\nat scale")
ax.legend(fontsize=7)

ax = axes[1]
for key, lab, col in [("wind_align", "wind alignment", "tab:green"),
                      ("align_frac", "aligned-edge fraction", "tab:blue")]:
    m = [S["summary"][s][key]["mean"] for s in sizes]
    ax.plot(n_nodes, m, "o-", label=lab, color=col)
ax.set_xlabel("nodes N"); ax.set_ylabel("morphogenesis metric")
ax.set_title("(b) Geometry organizes better\nas the material grows")
ax.legend(fontsize=7)

ax = axes[2]
m = [S["summary"][s]["memory_ratio"] for s in sizes]
ax.plot(n_nodes, m, "o-", color="tab:red")
ax.set_xlabel("nodes N"); ax.set_ylabel("memory ratio (batch / local)")
ax.set_title("(c) Memory advantage holds\n(~590x)")

fig.suptitle("Figure 5 | Scaling with constant excitation power density "
             "(5 seeds per size)")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/fig5_scaling.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- Fig 6: routing (Dirichlet probe) map
u_l = net.route_potential(0, 42)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, title in [(axes[0], "(a) Potential field, learned geometry"),
                  (axes[1], "(b) Potential field, random angles")]:
    rng2 = np.random.default_rng(0 + 999)
    if title.startswith("(b)"):
        vh = rng2.normal(size=(net.N, 3))
        vh /= np.maximum(np.linalg.norm(vh, axis=1, keepdims=True), 1e-12)
        net.vhat = vh
        u = net.route_potential(0, 42)
    else:
        u = u_l
    tris = ax.tripcolor(pos[:, 0], pos[:, 1], u, shading="gouraud",
                        cmap="RdBu_r")
    ax.scatter(pos[:, 0], pos[:, 1], c="k", s=12, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(tris, ax=ax, fraction=0.046, label="potential")
fig.suptitle("Figure 6 | Steady-state routing: the learned shape redirects "
             "flux (source NW, sink SE)")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/fig6_routing.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 7:
# real-data validation on the SILSO sunspot benchmark
import real_benchmark as rb

RB = json.load(open("real_benchmark.json"))
meta = RB["meta"]
d, M, tau = meta["d"], meta["M"], meta["tau"]
lam, ridge, lrls = meta["lam"], meta["ridge"], meta["lam_rls"]
raw = rb.load_sunspots()
s = rb.preprocess(raw)
n = len(s)
n_train, n_val = int(0.70 * n), int(0.10 * n)
tr_lo, tr_hi = 24, n_train
va_lo, va_hi = n_train + 1, n_train + n_val
te_lo, te_hi = n_train + n_val + 1, n - 1
zm = float(s[tr_lo: tr_hi + 1].mean())
zsd = float(s[tr_lo: tr_hi + 1].std())
z = (s - zm) / zsd
lr = rb.train_weak(z, d, M, tau, lam, ridge, tr_lo, tr_hi, lam_rls=lrls,
                   drive=True, level=False)
A_traj = lr.ema_trajectory(z, tau)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) real series, split shading, sample 12-month forecast in the test era
ax = axes[0]
t_yr = np.arange(n) / 12.0 + 1749.0
ax.plot(t_yr, s, color="0.75", lw=0.7, label="SILSO monthly sunspot number")
ax.axvspan(1749.0, 1749.0 + tr_hi / 12.0, color="tab:blue", alpha=0.08)
ax.axvspan(1749.0 + (tr_hi + 1) / 12.0, 1749.0 + va_hi / 12.0,
           color="tab:green", alpha=0.10)
ax.axvspan(1749.0 + (va_hi + 1) / 12.0, 1749.0 + te_hi / 12.0,
           color="tab:red", alpha=0.08)
ax.text(1800, 0.97, "train", fontsize=8, color="tab:blue")
ax.text(1960, 0.97, "val", fontsize=8, color="tab:green")
ax.text(1998, 0.97, "test", fontsize=8, color="tab:red")
t0 = te_lo + 60
fh = 24
fc = lr.forecast(z, t0, fh, tau, A_traj=A_traj)
ax.plot(t_yr[t0 + 1: t0 + fh + 1], fc, "o-", color="tab:red", ms=3,
        lw=1.2, label="24-month weak-form forecast")
ax.axvline(t_yr[t0], color="k", lw=0.8, ls=":")
ax.set_xlabel("year"); ax.set_ylabel(r"$\sqrt{\mathrm{SN}}$")
ax.set_title("(a) Real 277-year record, one forecast")
ax.legend(fontsize=7, loc="upper left")

# (b) noise robustness: holdout law-fit NMSE vs added noise (log)
ax = axes[1]
sigmas = [float(k.replace("sigma", "")) for k in RB["identification"]]
for key, lab, col, mk in [("weak_streaming", "weak-form streaming",
                           "tab:blue", "o"),
                          ("fd_streaming", "FD streaming", "tab:orange", "s"),
                          ("batch_weak", "batch weak (SINDy)",
                           "tab:cyan", "^"),
                          ("batch_fd", "batch FD (SINDy)",
                           "tab:red", "v")]:
    ys = [RB["identification"][k][key] for k in RB["identification"]]
    ax.semilogy(sigmas, ys, f"{mk}-", label=lab, color=col)
ax.set_xlabel("added measurement noise  (fraction of signal std)")
ax.set_ylabel("holdout law-fit NMSE (signal-normalized)")
ax.set_title("(b) Identification generalizes across\nunseen decades; FD does not")
ax.legend(fontsize=7)

# (c) forecast NMSE vs horizon (log)
ax = axes[2]
hs = [int(h[1:]) for h in RB["per_horizon"]]
for key, lab, col, mk in [("weak", "weak-form law", "tab:blue", "o"),
                          ("fd_streaming", "FD law", "tab:orange", "s"),
                          ("ar", "AR(24)", "tab:green", "^"),
                          ("esn", "ESN", "tab:purple", "v"),
                          ("persistence", "persistence", "k", "d")]:
    ys = [RB["per_horizon"][f"h{h}"][key] for h in hs]
    if key == "esn":
        ys = [RB["per_horizon"][f"h{h}"][key]["mean"] for h in hs]
    ax.semilogy(hs, ys, f"{mk}-", label=lab, color=col)
ax.set_xlabel("forecast horizon (months)")
ax.set_ylabel("forecast NMSE")
ax.set_title("(c) Forecast skill, held-out\n1971-2026 real window")
ax.legend(fontsize=7)

fig.suptitle("Figure 7 | Real-data validation: SILSO sunspot benchmark "
             "(3331 months, 1749-2026; law frozen after training)")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/fig7_realdat.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 8:
# scaling to 10^4 nodes (sparse warm-started CG plant solver)
SL = json.load(open("scaling_large.json"))
sizes_l = ["22x22", "32x32", "64x64", "100x100"]
n_nodes_l = [runs[0]["N"] for runs in (SL[s] for s in sizes_l)]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

ax = axes[0]
for key, lab, col in [("nmse_y", "law-fit NMSE", "tab:blue"),
                      ("nmse", "physical 1-step NMSE", "tab:green")]:
    m = [np.mean([r[key] for r in SL[s]]) for s in sizes_l]
    ax.semilogy(n_nodes_l, m, "o-", label=lab, color=col)
ax.set_xscale("log")
ax.set_xlabel("nodes N"); ax.set_ylabel("NMSE")
ax.set_title("(a) Identification improves\nwith scale")
ax.legend(fontsize=7)

ax = axes[1]
for key, lab, col in [("corr_g_phys", "shape vs law (readability)",
                       "tab:red"),
                      ("corr_w_phys", "raw weights vs law", "0.5")]:
    m = [np.mean([r[key] for r in SL[s]]) for s in sizes_l]
    ax.plot(n_nodes_l, m, "o-", label=lab, color=col)
ax.set_xscale("log")
ax.set_ylim(0, 1)
ax.set_xlabel("nodes N"); ax.set_ylabel("Pearson r")
ax.set_title("(b) The shape stays readable:\nangles beat weights at every size")
ax.legend(fontsize=7)

ax = axes[2]
m = [np.mean([r["oracle_bytes"] / r["local_bytes"] for r in SL[s]])
     for s in sizes_l]
ax.semilogy(n_nodes_l, m, "o-", color="tab:red")
ax.set_xscale("log")
ax.set_xlabel("nodes N")
ax.set_ylabel("memory ratio (batch frame-buffer / local)")
ax.set_title("(c) Memory advantage grows\nwith scale")

ax = axes[3]
m = [np.mean([r["ms_per_step"] for r in SL[s]]) for s in sizes_l]
ax.loglog(n_nodes_l, m, "o-", color="tab:purple")
ax.set_xlabel("nodes N")
ax.set_ylabel("wall ms per simulation step")
ax.set_title("(d) Per-node cost stays flat\n(sublinear scaling)")

fig.suptitle("Figure 8 | The material at 10^4 nodes (sparse warm-started "
             "CG plant solver; 3 seeds per size)")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/fig8_scaling_large.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Fig 9:
# second real benchmark: NINO3.4 El Nino SST anomalies (NOAA/PSL)
import real_benchmark as rb

RE = json.load(open("real_benchmark_enso.json"))
meta_e = RE["meta"]
d_e, M_e, tau_e = meta_e["d"], meta_e["M"], meta_e["tau"]
lam_e, ridge_e, lrls_e = meta_e["lam"], meta_e["ridge"], meta_e["lam_rls"]
raw_e = rb.load_enso()
s_e = rb.preprocess(raw_e, "enso")
n_e = len(s_e)
n_tr_e, n_va_e = int(0.70 * n_e), int(0.10 * n_e)
tr_lo_e, tr_hi_e = d_e, n_tr_e
va_lo_e, va_hi_e = n_tr_e + 1, n_tr_e + n_va_e
te_lo_e, te_hi_e = n_tr_e + n_va_e + 1, n_e - 1
zm_e = float(s_e[tr_lo_e: tr_hi_e + 1].mean())
zsd_e = float(s_e[tr_lo_e: tr_hi_e + 1].std())
z_e = (s_e - zm_e) / zsd_e
lr_e = rb.train_weak(z_e, d_e, M_e, tau_e, lam_e, ridge_e, tr_lo_e,
                     tr_hi_e, lam_rls=lrls_e, drive=True, level=False)
A_e = lr_e.ema_trajectory(z_e, tau_e)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) series + split + sample forecast
ax = axes[0]
t_yr_e = np.arange(n_e) / 12.0 + 1948.0
ax.plot(t_yr_e, s_e, color="0.75", lw=0.7,
        label="NINO3.4 monthly SST anomaly")
ax.axvspan(1948.0, 1948.0 + tr_hi_e / 12.0, color="tab:blue", alpha=0.08)
ax.axvspan(1948.0 + (tr_hi_e + 1) / 12.0, 1948.0 + va_hi_e / 12.0,
           color="tab:green", alpha=0.10)
ax.axvspan(1948.0 + (va_hi_e + 1) / 12.0, 1948.0 + te_hi_e / 12.0,
           color="tab:red", alpha=0.08)
ax.text(1955, 0.95, "train", fontsize=8, color="tab:blue")
ax.text(1993, 0.95, "val", fontsize=8, color="tab:green")
ax.text(2004, 0.95, "test", fontsize=8, color="tab:red")
t0_e = te_lo_e + 30
fh_e = 12
fc_e = lr_e.forecast(z_e, t0_e, fh_e, tau_e, A_traj=A_e)
ax.plot(t_yr_e[t0_e + 1: t0_e + fh_e + 1], fc_e, "o-", color="tab:red",
        ms=3, lw=1.2, label="12-month weak-form forecast")
ax.axvline(t_yr_e[t0_e], color="k", lw=0.8, ls=":")
ax.set_xlabel("year"); ax.set_ylabel("SST anomaly ($^\\circ$C)")
ax.set_title("(a) Real 78-year record, one forecast")
ax.legend(fontsize=7, loc="upper left")

# (b) noise robustness
ax = axes[1]
sigmas_e = [float(k.replace("sigma", "")) for k in RE["identification"]]
for key, lab, col, mk in [("weak_streaming", "weak-form streaming",
                           "tab:blue", "o"),
                          ("fd_streaming", "FD streaming", "tab:orange", "s"),
                          ("batch_weak", "batch weak (SINDy)",
                           "tab:cyan", "^"),
                          ("batch_fd", "batch FD (SINDy)",
                           "tab:red", "v")]:
    ys = [RE["identification"][k][key] for k in RE["identification"]]
    ax.semilogy(sigmas_e, ys, f"{mk}-", label=lab, color=col)
ax.set_xlabel("added measurement noise (fraction of signal std)")
ax.set_ylabel("holdout law-fit NMSE (signal-normalized)")
ax.set_title("(b) Identification armor holds on\na second real stream")
ax.legend(fontsize=7)

# (c) low-data: weak vs deep/linear learners as training data shrinks
ax = axes[2]
fracs = [float(k.replace("frac", "")) for k in RE["low_data"]]
fracs.sort()
for key, lab, col, mk in [("weak", "weak-form law", "tab:blue", "o"),
                          ("lstm", "LSTM (trained net)", "tab:red", "s"),
                          ("ar", "AR(24)", "tab:green", "^"),
                          ("esn", "ESN", "tab:purple", "v"),
                          ("persistence", "persistence", "k", "d")]:
    ys = [RE["low_data"][f"frac{f}"]["h1"][key] for f in fracs]
    ax.plot(fracs, ys, f"{mk}-", label=lab, color=col)
ax.set_xlabel("fraction of training data")
ax.set_ylabel("1-month forecast NMSE")
ax.set_title("(c) Sample complexity: the law is learned\nfrom a few hundred months")
ax.legend(fontsize=7)

fig.suptitle("Figure 9 | Second real benchmark: NINO3.4 El Nino SST "
             "anomalies (943 months, 1948-2026; NOAA/PSL)")
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/fig9_enso.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"figures written to {OUT}/")
print(sorted(os.listdir(OUT)))
