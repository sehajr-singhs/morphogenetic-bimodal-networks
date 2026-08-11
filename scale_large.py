"""scale_large.py -- the bimodal network at 10^3-10^4 nodes.

The paper's scaling table stops at 225 nodes because the reference solver
factorizes the dense N x N Laplacian (O(N^3) per step). With the sparse
warm-started conjugate-gradient plant solver (cfg solver="cg", O(E) per
iteration, exact to 1e-9 -- verified identical to the dense solve on the 7x7
grid), the same network runs at 10,000 nodes at ~0.1 s/step.

Reported metrics (identical definitions to run_experiments.run_trial, minus
the global batch-oracle baselines which are O(T E^2) and irrelevant at this
scale): physical one-step NMSE, weak-form law-fit NMSE, convergence latency,
self-organization (alignment, wind alignment, corridor contrast), mechanistic
readability (shape vs law and vs weights), Dirichlet routing (learned vs
random geometry), and the runtime-memory ratio (local streaming identifier vs
frame-buffer batch estimator).

Run:  python scale_large.py --sizes 32x32,64x64,100x100 --seeds 3 --T 1500
"""
from __future__ import annotations

import json
import time

import numpy as np

from biomaterial_net import MorphogeneticNet, make_grid, default_cfg


def run_trial(seed: int, nx: int, ny: int, T: int = 1500,
              noise_frac: float = 0.05) -> dict:
    cfg = dict(default_cfg("weak"))
    cfg["solver"] = "cg"
    dt = cfg["dt"]
    pos, edges = make_grid(nx, ny)
    N, E = pos.shape[0], edges.shape[0]
    cfg["sources"] = (0, ny - 1, (nx - 1) * ny, nx * ny - 1)
    cfg["a_base"] = cfg["a_base"] * (N / 49.0)   # constant excitation power density

    t0 = time.time()
    probe = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(seed ^ 0x5EED))
    for _ in range(600):
        probe.step(sigma=0.0, record=True, learn=True)
    u_std = float(np.std(np.array(probe.u_true_hist)))
    sigma = noise_frac * u_std

    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(seed))
    for _ in range(T):
        net.step(sigma=sigma, record=True)
    wall = time.time() - t0

    u_true = np.array(net.u_true_hist)
    u_obs = np.array(net.u_obs_hist)
    pred = np.array(net.pred_hist)
    align = np.array(net.align_hist)
    vnorm = np.array(net.vnorm_hist)
    te = int(0.6 * T)
    var_eval = float(np.var(u_true[te:]))
    nmse = float(np.mean((u_true[te:] - pred[te:]) ** 2) / var_eval)
    nmse_persist = float(np.mean((u_true[te + 1:] - u_obs[te:-1]) ** 2) / var_eval)

    resid2 = np.array(net.resid2_hist)
    y2 = np.array(net.y2_hist)
    nmse_y = float(resid2[te:].mean() / max(y2[te:].mean(), 1e-12))

    nmse_y_series = resid2 / np.maximum(y2, 1e-12)
    w = 30
    roll = np.convolve(nmse_y_series, np.ones(w) / w, mode="valid")
    latency = None
    for c in np.where(roll < 0.20)[0]:
        if c >= 20 and c + 16 <= len(roll) and np.all(roll[c:c + 16] < 0.20):
            latency = c * dt
            break

    align_init = float(np.mean(align[:50]))
    align_final = float(np.mean(align[-200:]))
    vnorm_max = float(np.max(vnorm))
    c_hist = np.array(net.c_hist)
    enc_err = np.array(net.enc_err_hist)
    w2 = np.mean(np.array(net._w) ** 2)
    enc_rmse = float(np.sqrt(enc_err[-200:].mean() / max(w2, 1e-12)))

    wdir = np.zeros(3)
    wdir[: len(cfg["wind"])] = np.asarray(cfg["wind"], dtype=float)
    wdir /= np.linalg.norm(wdir)
    wind_align = float(np.abs(net.vhat @ wdir).mean())
    corr_cw = float(np.corrcoef(net.c, net.vhat @ wdir)[0, 1]) \
        if np.std(net.c) > 1e-12 else 0.0

    kappa, a_ij, cbar_e, kappa0 = net._compute_kappa()
    w = net._w
    corr_w_k = float(np.corrcoef(w, kappa)[0, 1])
    corr_g_k = float(np.corrcoef(a_ij, kappa)[0, 1])
    corr_g_w = float(np.corrcoef(a_ij, w)[0, 1])
    kappa_phys = kappa0 * (1.0 + cfg["lam_c"] * cbar_e)
    if np.std(kappa_phys) > 1e-12:
        corr_g_phys = float(np.corrcoef(a_ij, kappa_phys)[0, 1])
        corr_w_phys = float(np.corrcoef(w, kappa_phys)[0, 1])
    else:
        corr_g_phys = corr_w_phys = 0.0
    align_frac = float((a_ij > 0.8).mean())
    if np.std(cbar_e) > 1e-12:
        hi_c = cbar_e > np.median(cbar_e)
        corr_focus = float(a_ij[hi_c].mean() - a_ij[~hi_c].mean())
    else:
        corr_focus = 0.0

    # Dirichlet routing via the sparse CG probe (learned vs random geometry)
    src, sink = 0, (nx - 1) * ny
    mid = (nx // 2) * ny
    flux, cbar_e2 = net.route_probe(src, sink)
    u_l = net.route_potential(src, sink)
    mcf_l = float(np.sum(flux * cbar_e2) / max(flux.sum(), 1e-12))
    corr_flux_g = float(np.corrcoef(flux, a_ij)[0, 1])
    rng2 = np.random.default_rng(seed + 999)
    v0, vh0 = net.v.copy(), net.vhat.copy()
    net.vhat = rng2.normal(size=(N, 3))
    net.vhat /= np.maximum(np.linalg.norm(net.vhat, axis=1, keepdims=True), 1e-12)
    net.v = np.linalg.norm(v0, axis=1)[:, None] * net.vhat
    flux_r, _ = net.route_probe(src, sink)
    u_r = net.route_potential(src, sink)
    mcf_r = float(np.sum(flux_r * cbar_e2) / max(flux_r.sum(), 1e-12))
    net.v, net.vhat = v0, vh0
    focus_gain = mcf_l - mcf_r
    trans_gain = float(u_l[mid] - u_r[mid])

    local_bytes = (32 * N + 8 * E) * 8
    oracle_bytes = (T * N * 4 + T * E + E * (E + 1) // 2 + T * N) * 8

    return dict(seed=seed, nx=nx, ny=ny, N=N, E=E,
                nmse=nmse, nmse_persist=nmse_persist, nmse_y=nmse_y,
                latency=latency, wall_s=wall, ms_per_step=1e3 * wall / T,
                align_init=align_init, align_final=align_final,
                wind_align=wind_align, corr_cw=corr_cw,
                corr_w_k=corr_w_k, corr_g_k=corr_g_k, corr_g_w=corr_g_w,
                corr_g_phys=corr_g_phys, corr_w_phys=corr_w_phys,
                align_frac=align_frac, corr_focus=corr_focus,
                corr_flux_g=corr_flux_g, focus_learned=mcf_l,
                focus_random=mcf_r, focus_gain=focus_gain,
                trans_gain=trans_gain,
                vnorm_max=vnorm_max, vnorm_final=float(vnorm[-1]),
                enc_rmse=enc_rmse,
                local_bytes=local_bytes, oracle_bytes=oracle_bytes)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="22x22,32x32,64x64,100x100")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--out", type=str, default="scaling_large.json")
    args = ap.parse_args()

    import os
    sizes = [tuple(int(x) for x in s.split("x")) for s in args.sizes.split(",")]
    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    t_start = time.time()
    for (nx, ny) in sizes:
        key = f"{nx}x{ny}"
        runs = [r for r in out.get(key, []) if r["seed"] >= args.seeds]  # keep
        for sd in range(args.seeds):
            t0 = time.time()
            r = run_trial(sd, nx, ny, T=args.T)
            print(f"{key} seed{sd}: nmse {r['nmse']:.4f}  law {r['nmse_y']:.4f}  "
                  f"align {r['align_init']:.2f}->{r['align_final']:.2f}  "
                  f"corr_g_phys {r['corr_g_phys']:.3f}  "
                  f"trans_gain {r['trans_gain']:.4f}  "
                  f"({time.time()-t0:.0f}s)")
            runs.append(r)
        out[key] = runs
        # incremental save: a crash mid-sweep keeps the completed sizes
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"{key} saved -> {args.out}  (elapsed {time.time()-t_start:.0f}s)")
    print(f"\ndone -> {args.out}  (total {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
