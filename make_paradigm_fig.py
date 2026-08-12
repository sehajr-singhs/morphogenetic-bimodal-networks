"""make_paradigm_fig.py -- the CROWN-style 3-panel "angle paradigm" figure.

Three panels, each with a diagram above and a bold label below (the
α/β-CROWN layout the site's hero figure follows):

  (a) THE PARADIGM  -- two channels of computation per neuron: scalar
      weights (strength) and 3D angles (direction). Weights say how
      strongly; angles say where.
  (b) THE MODULATION -- the angle factor kappa = w*c*(vhat_i . vhat_j):
      how alignment steers the effective coupling, with the measured
      read-back (shape vs law, weights vs law) from a real game episode.
  (c) THE SHAPE -- morphogenesis on real Flowrunner output: the geometry
      grows along the hidden law and routes a token through it.

Usage:  python make_paradigm_fig.py [--out figs]
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#111111",
})

BLUE = "#226999"      # angle channel / shape
INK = "#1a1a1a"
GRAY = "#8c8e90"
CORAL = "#d95f5f"


# ---------------------------------------------------------------- panel (a)
def panel_paradigm(ax):
    """Schematic: one neuron carries BOTH a scalar weight and a 3D angle."""
    ax.set_xlim(-0.5, 7.4)
    ax.set_ylim(-0.4, 4.6)
    ax.axis("off")

    # three neurons in a row
    xs = np.array([1.0, 3.4, 5.8])
    ys = np.array([1.2, 2.6, 1.2])
    vhat = np.array([[0.96, 0.28], [-0.35, 0.94], [0.85, -0.53]])

    # edges: weight channel (thickness) + angle factor (alignment)
    for (x1, y1, v1), (x2, y2, v2) in zip(
            zip(xs[:-1], ys[:-1], vhat[:-1]),
            zip(xs[1:], ys[1:], vhat[1:])):
        align = float(np.dot(v1, v2))
        lw = 1.0 + 3.2 * max(align, 0.0)
        ax.plot([x1, x2], [y1, y2], color=INK, lw=lw, zorder=1,
                alpha=0.85)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.42
        ax.text(mx, my, f"w = {1.0 + 0.5 * align:.1f}", ha="center",
                fontsize=9, color=INK)
        ax.text(mx, my - 0.42, f"v\u0302\u1d62\u00b7v\u0302\u2c7c = {align:+.2f}",
                ha="center", fontsize=9, color=BLUE)

    # neurons with their angle vectors
    for (x, y, v) in zip(xs, ys, vhat):
        ax.add_patch(plt.Circle((x, y), 0.34, facecolor="white",
                                edgecolor=INK, lw=1.6, zorder=3))
        ax.arrow(x, y, v[0] * 0.62, v[1] * 0.62, head_width=0.13,
                 head_length=0.15, fc=CORAL, ec=CORAL, zorder=4, lw=1.8)
    # the angle channel label
    ax.text(xs[1], ys[1] + 0.62, "v\u0302\u1d62 (3D angle)", ha="center",
            fontsize=10, color=CORAL, fontweight="bold")

    # two-channel strips
    ax.text(-0.35, 0.35, "WEIGHT channel", rotation=90, ha="center",
            va="center", fontsize=9, color=INK)
    ax.text(-0.35, 3.6, "ANGLE channel", rotation=90, ha="center",
            va="center", fontsize=9, color=CORAL, fontweight="bold")

    # the composition
    ax.text(3.6, 4.28, r"$\kappa_{ij} = w_{ij}\,c_{ij}\,(\hat{v}_i\cdot\hat{v}_j)$",
            ha="center", fontsize=13.5, color=INK)
    ax.add_patch(FancyArrowPatch((3.6, 3.62), (3.6, 2.9),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=GRAY, lw=1.4))
    ax.text(3.6, 0.55, "strength  \u00d7  direction  =  the connection",
            ha="center", fontsize=9.5, color=GRAY, style="italic")


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
    """Angle factor curve (left) + measured read-back (right)."""
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # left: the alignment factor
    th = np.linspace(0, np.pi, 300)
    ax.plot([0, 0.88], [0.78, 0.78], color=GRAY, lw=1.0)
    ax.plot([0, 0.88], [0.22, 0.22], color=GRAY, lw=1.0)
    ax.plot(th / np.pi * 0.88, 0.5 + 0.28 * np.cos(th), color=BLUE, lw=2.4)
    ax.axvline(0.0, color=INK, lw=1.0)
    ax.text(0.44, 0.88, "alignment  cos \u03b8", ha="center", fontsize=10,
            color=INK)
    ax.text(0.02, 0.52, "+1  aligned", fontsize=8.5, color=BLUE)
    ax.text(0.60, 0.185, "0  perpendicular", fontsize=8.5, color=BLUE)
    ax.text(0.02, 0.10, "\u03b8 = 0", fontsize=8, color=GRAY)
    ax.text(0.79, 0.10, "\u03b8 = \u03c0", fontsize=8, color=GRAY)
    # two little vector pairs
    for (cx, ang, lab) in [(0.44, 0.0, "aligned"), (0.44, 1.4, "weak")]:
        pass
    # arrow to measured panel
    ax.annotate("", xy=(1.02, 0.5), xytext=(0.90, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5))

    # right: measured read-back (paper's controlled 7x7 grid run)
    grid_net = representative_grid()
    kappa_g, a_g, _, _ = grid_net._compute_kappa()
    w = grid_net._w
    kappa = kappa_g
    a_ij = a_g
    r_a = np.corrcoef(a_ij, kappa)[0, 1]
    r_w = np.corrcoef(w, kappa)[0, 1]
    ax.plot([1.22, 1.98], [0.78, 0.78], color=GRAY, lw=1.0)
    ax.plot([1.22, 1.98], [0.22, 0.22], color=GRAY, lw=1.0)
    rng = np.random.default_rng(3)
    n = min(len(kappa), 220)
    idx = rng.choice(len(kappa), n, replace=False)
    ax.scatter(1.22 + (kappa[idx] - kappa.min()) /
               (kappa.max() - kappa.min()) * 0.76,
               0.5 + 0.28 * (a_ij[idx] - a_ij.min()) /
               max(a_ij.max() - a_ij.min(), 1e-12) * 2.0 - 0.28,
               s=8, alpha=0.65, color=BLUE, lw=0)
    ax.scatter(1.22 + (kappa[idx] - kappa.min()) /
               (kappa.max() - kappa.min()) * 0.76,
               0.5 + 0.28 * (w[idx] - w.min()) /
               max(w.max() - w.min(), 1e-12) * 2.0 - 0.28,
               s=8, alpha=0.5, color=GRAY, lw=0)
    ax.text(1.6, 0.90, "measured read-back", ha="center", fontsize=10,
            color=INK)
    ax.text(1.26, 0.665, f"shape vs law   r = {r_a:.2f}", fontsize=9,
            color=BLUE, fontweight="bold")
    ax.text(1.26, 0.30, f"weights vs law  r = {r_w:.2f}", fontsize=9,
            color=GRAY)


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
    ax.tripcolor(pos[:, 0], pos[:, 1], vals, shading="gouraud",
                 cmap="inferno", vmin=np.percentile(vals, 5),
                 vmax=np.percentile(vals, 95))
    ax.plot(env.path[:, 0], env.path[:, 1], "w--", lw=1.8, alpha=0.95,
            label="hidden law")
    p = np.array(ep["path"])
    ax.plot(pos[p, 0], pos[p, 1], color=CORAL, lw=2.4, alpha=0.95,
            label="token path")
    ax.scatter([pos[env.entry, 0]], [pos[env.entry, 1]], marker="o",
               s=90, facecolor="white", edgecolor=INK, zorder=5)
    ax.scatter([pos[env.goal, 0]], [pos[env.goal, 1]], marker="*",
               s=260, facecolor="gold", edgecolor=INK, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=8, loc="lower left", framealpha=0.85)
    ax.set_title(f"steps to goal: {ep['steps']}  "
                 f"(oracle {ep.get('oracle', '--')})", fontsize=9.5,
                 color=INK, pad=4)


# ---------------------------------------------------------------- figure
def main():
    import flowgame as fg
    ep = fg.run_episode(0, morph_on=True, record=False)
    ep["a_ij"] = ep["a_ij"] if "a_ij" in ep else ep["net"]._compute_kappa()[1]
    ep["oracle"] = fg.route_oracle(ep["env"])[0]

    fig = plt.figure(figsize=(16.5, 5.6))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.02, right=0.98,
                          top=0.92, bottom=0.10)

    labels = [
        "(a) THE PARADIGM\nweights say how strongly, angles say where",
        "(b) THE MODULATION\n\u03ba = w\u00b7c\u00b7(v\u0302\u1d62\u00b7v\u0302\u2c7c): "
        "the angle steers the connection",
        "(c) THE SHAPE\nthe geometry grows into the law it learned",
    ]
    for axg, lab in zip(gs, labels):
        ax = fig.add_subplot(axg)
        ax.text(0.5, -0.06, lab, transform=ax.transAxes, ha="center",
                va="top", fontsize=11.5, color=INK, fontweight="bold",
                linespacing=1.35)
        ax.axis("off")

    panel_paradigm(fig.add_subplot(gs[0]))
    panel_modulation(fig.add_subplot(gs[1]), ep)
    panel_shape(fig.add_subplot(gs[2]), ep)

    fig.savefig(f"{OUT}/fig_paradigm.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    print(f"paradigm figure -> {OUT}/fig_paradigm.png")


if __name__ == "__main__":
    main()
