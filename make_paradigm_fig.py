"""make_paradigm_fig.py -- the CROWN-style 3-panel "angle paradigm" figure.

Each panel is a real matplotlib axes: a clean network schematic, proper
data plots with axes and labels, and the real Flowrunner game output.
A bold CROWN-style label sits below each panel.

  (a) THE PARADIGM  -- two channels of computation per neuron: scalar
      weights (strength) and 3D angles (direction).
  (b) THE MODULATION -- the angle factor kappa = w*c*(vhat_i . vhat_j):
      how alignment steers the effective coupling, and the measured
      read-back (shape vs law, weights vs law) from the paper's
      controlled grid run.
  (c) THE SHAPE -- real Flowrunner output: the geometry grows along the
      hidden law and routes a token through it.

Usage:  python make_paradigm_fig.py [--out figs]
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#111111",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#111111",
    "axes.linewidth": 1.0,
})

BLUE = "#226999"
INK = "#1a1a1a"
GRAY = "#9a9c9e"
CORAL = "#c94f4f"


# ---------------------------------------------------------------- panel (a)
def panel_paradigm(ax):
    """A clean schematic that fills the panel: a 2x3 neuron grid, each with
    a weight to its neighbors AND a 3D angle vector. Weights: line width.
    Angles: arrows. The composition callout sits below the grid."""
    # data ranges chosen to match the panel's aspect so the drawing fills it
    ax.set_xlim(-1.15, 4.25)
    ax.set_ylim(-1.35, 4.10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 2x3 neuron grid (spread to fill the panel)
    xs = np.array([0.0, 1.7, 3.4, 0.0, 1.7, 3.4])
    ys = np.array([2.9, 2.9, 2.9, 0.9, 0.9, 0.9])
    vhat = np.array([[0.94, 0.34], [0.36, 0.93], [-0.60, 0.80],
                     [0.90, -0.44], [-0.20, -0.98], [0.72, 0.70]])

    # edges (horizontal + vertical) with width = weight
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    wgt = np.array([1.0, 0.55, 0.8, 1.0, 0.45, 0.9, 0.6])
    for (i, j), w in zip(edges, wgt):
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#b9bbbd",
                lw=0.5 + 2.4 * w, zorder=1, solid_capstyle="round")

    # neurons + angle arrows
    for (x, y, v) in zip(xs, ys, vhat):
        ax.add_patch(plt.Circle((x, y), 0.30, facecolor="#f4f6f8",
                                edgecolor=INK, lw=1.4, zorder=3))
        ax.annotate("", xy=(x + v[0] * 0.58, y + v[1] * 0.58),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=CORAL,
                                    lw=2.0, mutation_scale=14), zorder=4)

    # legend: weight line + angle arrow (top band)
    ax.plot([-0.55, 0.0], [3.85, 3.85], color="#b9bbbd", lw=3.0,
            solid_capstyle="round")
    ax.text(0.07, 3.85, r"weight $w_{ij}$ (strength)", va="center",
            fontsize=10)
    ax.annotate("", xy=(-0.55 + 0.5, 3.55), xytext=(-0.55, 3.55),
                arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=2.0,
                                mutation_scale=13))
    ax.text(0.07, 3.55, r"angle $\hat{v}_i$ (direction)", va="center",
            fontsize=10)

    # composition callout (bottom band)
    ax.text(1.7, -0.85, r"$\kappa_{ij} = w_{ij}\,c_{ij}\,"
            r"(\hat{v}_i\cdot\hat{v}_j)$", ha="center", fontsize=13,
            color=INK)


# ---------------------------------------------------------------- panel (b)
def representative_grid(seed=0, T=3000):
    """Same representative run make_figures.py uses (paper's read-back)."""
    from biomaterial_net import MorphogeneticNet, make_grid, default_cfg
    pos, edges = make_grid(7, 7)
    cfg = default_cfg("weak")
    cfg["sources"] = (0, 6, 42, 48)
    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(seed))
    for _ in range(T):
        net.step(sigma=0.05, record=True)
    return net


def panel_modulation(ax, ep):
    """Two real subplots inside the panel: (left) the alignment factor
    cos(theta); (right) measured read-back: shape vs law and weights vs
    law, each with its regression line and Pearson r."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ------- left: the coupling factor curve -------
    lax = ax.inset_axes([0.0, 0.06, 0.44, 0.88])
    th = np.linspace(0, np.pi, 400)
    lax.plot(th, np.cos(th), color=BLUE, lw=2.4)
    lax.fill_between(th, np.cos(th), 0, where=np.cos(th) > 0,
                     color=BLUE, alpha=0.12)
    lax.axhline(0, color="#555", lw=0.8)
    lax.axvline(0, color="#555", lw=0.8)
    lax.annotate("aligned\n(+1)", xy=(0.05, 1.0), xytext=(0.35, 0.85),
                 fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
    lax.annotate("perpendicular\n(0)", xy=(np.pi / 2, 0.0),
                 xytext=(1.55, 0.45), fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
    lax.set_xlim(0, np.pi)
    lax.set_ylim(-1.15, 1.2)
    lax.set_xticks([0, np.pi / 2, np.pi])
    lax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    lax.set_xlabel(r"angle $\theta$ between $\hat{v}_i$, $\hat{v}_j$",
                   fontsize=9)
    lax.set_ylabel(r"coupling factor  $\hat{v}_i\!\cdot\!\hat{v}_j$",
                   fontsize=9)
    lax.tick_params(labelsize=8)
    lax.set_title("alignment steers the connection", fontsize=10, pad=4)

    # arrow between the two
    ax.annotate("", xy=(0.50, 0.5), xytext=(0.455, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))

    # ------- right: measured read-back (paper's grid run) -------
    rax = ax.inset_axes([0.54, 0.06, 0.46, 0.88])
    grid_net = representative_grid()
    kappa_g, a_g, _, _ = grid_net._compute_kappa()
    w = grid_net._w
    rng = np.random.default_rng(0)
    n = min(len(kappa_g), 140)
    idx = rng.choice(len(kappa_g), n, replace=False)

    def regline(x, y, col, lab, r):
        p = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        rax.plot(xx, np.polyval(p, xx), color=col, lw=1.6, alpha=0.85)
        rax.scatter(x, y, s=16, alpha=0.7, color=col, edgecolors="none",
                    label=f"{lab}  (r = {r:.2f})")

    regline(kappa_g[idx], a_g[idx], BLUE, "shape $a_{ij}$",
            np.corrcoef(a_g, kappa_g)[0, 1])
    regline(kappa_g[idx], w[idx], GRAY, "weights $w_{ij}$",
            np.corrcoef(w, kappa_g)[0, 1])
    rax.set_xlabel(r"true law  $\kappa$", fontsize=9)
    rax.set_ylabel("read-back (a.u.)", fontsize=9)
    rax.tick_params(labelsize=8)
    rax.legend(fontsize=8, frameon=True, loc="upper left")
    rax.set_title("measured: the shape reads the law", fontsize=10, pad=4)


# ---------------------------------------------------------------- panel (c)
def panel_shape(ax, ep):
    """Real Flowrunner output: the shape grows along the hidden law."""
    env, net = ep["env"], ep["net"]
    pos, edges = env.pos, env.edges
    a_ij = ep["a_ij"]
    vals = np.zeros(len(pos))
    for e, (i, j) in enumerate(edges):
        vals[i] += a_ij[e]
        vals[j] += a_ij[e]
    ax.set_xlim(-0.6, env.nx - 0.4)
    ax.set_ylim(-0.6, env.ny - 0.4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    tcf = ax.tripcolor(pos[:, 0], pos[:, 1], vals, shading="gouraud",
                       cmap="viridis",
                       vmin=np.percentile(vals, 2),
                       vmax=np.percentile(vals, 98))
    ax.plot(env.path[:, 0], env.path[:, 1], "w--", lw=2.2, alpha=1.0,
            label="hidden law")
    p = np.array(ep["path"])
    ax.plot(pos[p, 0], pos[p, 1], color=CORAL, lw=2.6, alpha=1.0,
            label="token path")
    ax.scatter([pos[env.entry, 0]], [pos[env.entry, 1]], marker="o",
               s=120, facecolor="white", edgecolor=INK, lw=1.2, zorder=5)
    ax.scatter([pos[env.goal, 0]], [pos[env.goal, 1]], marker="*",
               s=320, facecolor="gold", edgecolor=INK, lw=1.2, zorder=5)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.set_title(f"steps to goal: {ep['steps']}  (oracle "
                 f"{ep.get('oracle', '--')})", fontsize=10, pad=4)
    return tcf


# ---------------------------------------------------------------- figure
def main():
    import flowgame as fg
    ep = fg.run_episode(0, morph_on=True, record=False)
    ep["a_ij"] = ep["a_ij"] if "a_ij" in ep else ep["net"]._compute_kappa()[1]
    ep["oracle"] = fg.route_oracle(ep["env"])[0]

    fig = plt.figure(figsize=(16.5, 5.8))
    gs = fig.add_gridspec(1, 3, wspace=0.28, left=0.045, right=0.985,
                          top=0.90, bottom=0.155)

    labels = [
        "(a) THE PARADIGM\nweights say how strongly, angles say where",
        "(b) THE MODULATION\n"
        r"$\kappa = w\,c\,(\hat{v}_i\cdot\hat{v}_j)$ — the angle steers "
        "the connection",
        "(c) THE SHAPE\nthe geometry grows into the law it learned",
    ]
    for axg, lab in zip(gs, labels):
        ax = fig.add_subplot(axg)
        ax.axis("off")
        ax.text(0.5, -0.045, lab, transform=ax.transAxes, ha="center",
                va="top", fontsize=11.5, color=INK, fontweight="bold",
                linespacing=1.25)

    panel_paradigm(fig.add_subplot(gs[0]))
    panel_modulation(fig.add_subplot(gs[1]), ep)
    panel_shape(fig.add_subplot(gs[2]), ep)

    fig.savefig(f"{OUT}/fig_paradigm.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    print(f"paradigm figure -> {OUT}/fig_paradigm.png")


if __name__ == "__main__":
    main()
