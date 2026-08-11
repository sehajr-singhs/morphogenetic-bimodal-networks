"""
run_experiments.py -- brutal continuous-excitation test.

Subject the Morphogenetic Bi-Modal Network to a non-stationary chaotic
environment with 5% observational noise, over a multi-seed sweep, and measure:

  * Physical one-step prediction NMSE (held-out window) vs baselines
  * Weak-form-domain NMSE (how well the identified law explains the data)
  * Convergence latency
  * Structural stability (multi-seed variance of the alignment statistic)
  * Runtime memory (local streaming identifier vs global batch estimator)
  * Mechanistic interpretability: shape-read-back (corr(a_ij, kappa)),
    steady-state routing (Dirichlet probes), law decomposition

Baselines (all on the SAME recorded observations):
  * FD-RLS       : identical per-node identifier but with raw finite-difference
                   targets ("the death of differentiation": 2/dt^2 noise amp.)
  * Persistence  : predict u(t+1) = u_obs(t)
  * Global batch oracle : full per-edge conductivity matrix fit by least
                   squares on the SAME weak-form-filtered data over the
                   training window, then frozen (best-case global/batch
                   estimator)
  * Streaming-global RLS : the same full per-edge matrix fit with exponential
                   forgetting over the WHOLE trajectory -- the
                   "streaming-optimized batch" that frame-buffer batch
                   estimators could be replaced by. Measured honestly: it
                   isolates whether PER-NODE LOCALITY itself (not just
                   streaming) earns the advantage.

Tasks:
  * sweep    : default -- 15-seed sweep on the 7x7 reference grid
  * ablation : one-configuration-at-a-time removal of each mechanism
  * scale    : the full sweep at multiple grid sizes (generality check)
"""

from __future__ import annotations

import json
import time

import numpy as np

from biomaterial_net import MorphogeneticNet, make_grid, default_cfg


# ---------------------------------------------------------------- configs
def variant_cfg(name: str) -> dict:
    """Ablation configurations: the full model minus exactly one mechanism."""
    cfg = default_cfg("weak")
    if name == "full":
        return cfg
    if name == "no_geometry":            # angles exist but cannot steer flow
        cfg["lam_g"] = 0.0
    elif name == "no_chemgate":          # morphogenesis everywhere (F7)
        cfg["act_th"] = 0.0
        cfg["gate_pow"] = 0.0
    elif name == "no_consolidation":     # raw fast multicollinear weights
        cfg["consolidate"] = False
    elif name == "no_morphogenesis":     # geometry frozen at random init
        cfg["morph_on"] = False
    elif name == "no_chemistry":         # no chemical channel anywhere;
        # the gate is opened so morphogenesis still runs on identification
        # alone -- isolates the chemical imprint specifically
        cfg["lam_c"] = 0.0
        cfg["beta"] = 0.0
        cfg["D"] = 0.0
        cfg["wind"] = None
        cfg["act_th"] = 0.0
        cfg["gate_pow"] = 0.0
    else:
        raise ValueError(f"unknown variant: {name}")
    return cfg


# ---------------------------------------------------------------- trial
def run_trial(seed: int, cfg: dict, T: int = 4000, train_frac: float = 0.6,
              noise_frac: float = 0.05, nx: int = 7, ny: int = 7) -> dict:
    dt = cfg["dt"]
    cfg = dict(cfg)
    cfg["sources"] = (0, ny - 1, (nx - 1) * ny, nx * ny - 1)   # four corners
    pos, edges = make_grid(nx, ny)
    N, E = pos.shape[0], edges.shape[0]
    # hold excitation power density constant as the material grows: the 7x7
    # reference (N=49) is driven at a_base=5.0 at its four corner sources;
    # a larger material needs proportionally more drive, otherwise the
    # interior never crosses u_thr, the chemical corridor never develops, and
    # morphogenesis stays frozen (observed at 15x15: c_mean 1.21 -> 0.02).
    cfg["a_base"] = cfg["a_base"] * (N / 49.0)

    # ---- probe pass: noise-free, estimate the voltage amplitude so that the
    #      5% observational noise is relative to the actual signal scale
    probe = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(seed ^ 0x5EED))
    for _ in range(600):
        probe.step(sigma=0.0, record=True, learn=True)
    u_std = float(np.std(np.array(probe.u_true_hist)))
    sigma = noise_frac * u_std

    # ---- main run (streaming, online learning throughout)
    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(seed))
    net.enable_oracle_recording()
    for _ in range(T):
        net.step(sigma=sigma, record=True)

    u_true = np.array(net.u_true_hist)     # (T, N)
    u_obs = np.array(net.u_obs_hist)
    I = np.array(net._oracle["I"])
    pred = np.array(net.pred_hist)         # one-step-ahead physical prediction
    align = np.array(net.align_hist)
    vnorm = np.array(net.vnorm_hist)
    te = int(train_frac * T)               # eval window start

    var_eval = float(np.var(u_true[te:]))
    err2 = (u_true - pred) ** 2
    nmse = float(np.mean(err2[te:]) / var_eval)
    floor = float(sigma ** 2 / var_eval)

    # persistence baseline: predict u(t+1) = u_obs(t)
    nmse_persist = float(np.mean((u_true[te + 1:] - u_obs[te:-1]) ** 2) / var_eval)

    # weak-form-domain NMSE: how well the identified law v_i . x_i explains
    # the weak-form targets y_i (the actual system-identification metric)
    resid2 = np.array(net.resid2_hist)
    y2 = np.array(net.y2_hist)
    nmse_y = float(resid2[te:].mean() / max(y2[te:].mean(), 1e-12))

    # ---- convergence latency: time until the rolling law-fit NMSE drops
    #      below 20% (the identification has converged to the local law)
    latency = None
    nmse_y_series = resid2 / np.maximum(y2, 1e-12)
    w = 30
    roll = np.convolve(nmse_y_series, np.ones(w) / w, mode="valid")
    for c in np.where(roll < 0.20)[0]:
        if c >= 20 and c + 16 <= len(roll) and np.all(roll[c:c + 16] < 0.20):
            latency = c * dt
            break

    # ---- structural stability statistics
    align_init = float(np.mean(align[:50]))
    align_final = float(np.mean(align[-200:]))
    vnorm_max = float(np.max(vnorm))
    c_hist = np.array(net.c_hist)
    enc_err = np.array(net.enc_err_hist)
    w2 = np.mean(np.array(net._w) ** 2)
    enc_rmse = float(np.sqrt(enc_err[-200:].mean() / max(w2, 1e-12)))

    # morphogenesis orientation: does the beam field point along the
    # neuromodulator drift axis, and do corridor (high-c) nodes align more?
    if cfg["wind"] is not None:
        wdir = np.zeros(3)
        wdir[: len(cfg["wind"])] = np.asarray(cfg["wind"], dtype=float)
        wdir /= np.linalg.norm(wdir)
        wind_align = float(np.abs(net.vhat @ wdir).mean())
        corr_cw = float(np.corrcoef(net.c, net.vhat @ wdir)[0, 1]) \
            if np.std(net.c) > 1e-12 else 0.0
    else:
        wind_align, corr_cw = 0.0, 0.0

    # ---- mechanistic interpretability: read the mechanism off the SHAPE.
    #      kappa is the true material law at the final state; w is what the
    #      identifier learned; a_ij (from the 3D angles alone) is what the
    #      geometry predicts -- the shape IS the coupling map.
    kappa, a_ij, cbar_e, kappa0 = net._compute_kappa()
    w = net._w
    corr_w_k = float(np.corrcoef(w, kappa)[0, 1])           # weight-readability
    corr_g_k = float(np.corrcoef(a_ij, kappa)[0, 1])        # shape vs FULL law
    corr_g_w = float(np.corrcoef(a_ij, w)[0, 1])            # shape vs weights
    # CLEAN read-back: the shape must be compared against the part of the
    # law it did NOT create -- the environment-imposed physical law
    # kappa_phys = kappa0 * (1 + lam_c*cbar), with the geometric channel
    # removed. corr(a_ij, kappa_full) is confounded: a_ij appears inside
    # kappa, so even a random frozen geometry scores mechanically high.
    kappa_phys = kappa0 * (1.0 + cfg["lam_c"] * cbar_e)
    if np.std(kappa_phys) > 1e-12:
        corr_g_phys = float(np.corrcoef(a_ij, kappa_phys)[0, 1])
        corr_w_phys = float(np.corrcoef(w, kappa_phys)[0, 1])
    else:
        corr_g_phys = corr_w_phys = 0.0
    aligned = a_ij > 0.8
    align_frac = float(aligned.mean())                      # aligned edges
    if np.std(cbar_e) > 1e-12:
        hi_c = cbar_e > np.median(cbar_e)
        corr_focus = float(a_ij[hi_c].mean() - a_ij[~hi_c].mean())
    else:
        corr_focus = 0.0                                    # corridor contrast

    # law decomposition: how much of the identified law is carried by each
    # computational channel (geometric angulation, chemistry)
    X = np.stack([a_ij, cbar_e], axis=1)
    Xm = X - X.mean(0)
    wm = w - w.mean()
    beta, *_ = np.linalg.lstsq(Xm, wm, rcond=None)
    pred_law = Xm @ beta
    r2_law = 1.0 - float(np.var(wm - pred_law) / max(np.var(wm), 1e-12))
    std_beta = beta * Xm.std(0) / max(wm.std(), 1e-12)

    # ---- routing: does the learned geometry actually direct information
    #      flow? AXIAL steady-state Dirichlet probe along the chemical
    #      corridor (source at the upwind end 0=(0,0), sink at the downwind
    #      end (nx-1,0)), plus an ablation against the identical material
    #      with random (unlearned) angles.
    src, sink = 0, (nx - 1) * ny            # axial corridor probe
    mid = (nx // 2) * ny                    # corridor midpoint node
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
    corr_flux_lr = float(np.corrcoef(flux, flux_r)[0, 1])
    # corridor transmission: potential at the corridor midpoint -- the learned
    # geometry should push more signal down the axis
    trans_gain = float(u_l[mid] - u_r[mid])

    # ---- global batch oracle (frozen fit on the training window, evaluated
    #      on the held-out window) -- uses the SAME weak-form filtered data
    od = net.oracle_data()
    Au = od["Au"]
    AI = od["AI"]
    duf = od["duf"]                        # (T, E)
    y = cfg["lam"] * (od["u_obs"] - Au) - AI   # weak-form targets (T, N)

    inc = np.zeros((N, E))
    inc[edges[:, 0], np.arange(E)] = 1.0
    inc[edges[:, 1], np.arange(E)] = -1.0
    D_adj = inc.T @ inc                        # (E, E)

    def _eval(K, yhat_cache=None):
        yhat = np.zeros((T - te, N))
        for t in range(T - te):
            d = duf[te + t]
            yhat[t] = inc @ (d * K)
        pred_or = od["u_obs"][te:] + dt * (yhat + AI[te:])
        nmse_or = float(np.mean((od["u_true"][te:] - pred_or) ** 2) / var_eval)
        var_y_eval = float(np.var(y[te:]))
        nmse_y_or = float(np.mean((y[te:] - yhat) ** 2) / max(var_y_eval, 1e-12))
        return nmse_or, nmse_y_or

    # frozen batch oracle: uniform-weight normal equations over the training
    # window only (fast via the edge-sign adjacency)
    XtX = np.zeros((E, E))
    Xty = np.zeros(E)
    for t in range(te):
        d = duf[t]
        XtX += np.outer(d, d) * D_adj
        Xty += d * (inc.T @ y[t])
    K_batch = np.linalg.solve(XtX + 1e-6 * np.eye(E), Xty)
    nmse_oracle, nmse_y_oracle = _eval(K_batch)

    # streaming-global RLS: the SAME global per-edge model, but fit with
    # exponential forgetting over the whole trajectory -- the honest
    # "streaming-optimized batch" baseline. Reported at TWO forgetting rates:
    #   (i)  matched forgetting (same alpha as the local identifier): shows
    #        the sample-efficiency failure -- E=84 global parameters must be
    #        re-learned inside a ~34-step window;
    #   (ii) tuned slow forgetting (lambda_g = 0.25, swept; ~200-step window,
    #        still faster than the 15 s regime period): the best a global
    #        streaming model can do. The residual gap to the local identifier
    #        isolates whether PER-NODE LOCALITY itself (not just streaming)
    #        earns the advantage.
    incT = inc.T
    def _glob_accum(alpha):
        XtX = np.zeros((E, E))
        Xty = np.zeros(E)
        for t in range(T):
            d = duf[t]
            XtX = alpha * XtX + (1.0 - alpha) * np.outer(d, d) * D_adj
            Xty = alpha * Xty + (1.0 - alpha) * d * (incT @ y[t])
        ridge_g = 1e-2 * (np.trace(XtX) / E) + 1e-6
        return np.linalg.solve(XtX + ridge_g * np.eye(E), Xty)

    K_match = _glob_accum(np.exp(-cfg["lam"] * dt))
    nmse_gm, nmse_y_gm = _eval(K_match)
    K_tuned = _glob_accum(np.exp(-0.25 * dt))
    nmse_global, nmse_y_global = _eval(K_tuned)
    nmse_y_global_matched = nmse_y_gm
    nmse_global_matched = nmse_gm

    # ---- FD baseline: IDENTICAL identifier, raw finite-difference targets,
    #      run offline over the SAME recorded observations (no morphogenesis)
    cfg_fd = dict(cfg)
    cfg_fd["mode"] = "fd"
    fdnet = MorphogeneticNet(pos, edges, cfg_fd, np.random.default_rng(seed ^ 0xF00D))
    for t in range(T):
        fdnet.observe(u_obs[t], I[t])
    pred_fd = np.array(fdnet.pred_hist)
    nmse_fd = float(np.mean((u_true[te:] - pred_fd[te:]) ** 2) / var_eval)
    resid2_fd = np.array(fdnet.resid2_hist)
    y2_fd = np.array(fdnet.y2_hist)
    nmse_y_fd = float(resid2_fd[te:].mean() / max(y2_fd[te:].mean(), 1e-12))

    # ---- runtime memory accounting
    # local streaming identifier: per-node (u,c,v,vhat,Au,AI,C(10),r,z,a) +
    # per-edge scratch
    local_bytes = (32 * N + 8 * E) * 8
    # global batch estimator: full trajectory + filtered fields + normal matrix
    oracle_bytes = (T * N * 4 + T * E + E * (E + 1) // 2 + T * N) * 8

    return dict(
        seed=seed, nx=nx, ny=ny,
        nmse=nmse, nmse_persist=nmse_persist, nmse_y=nmse_y,
        nmse_fd=nmse_fd, nmse_y_fd=nmse_y_fd,
        nmse_oracle=nmse_oracle, nmse_y_oracle=nmse_y_oracle, floor=floor,
        nmse_global=nmse_global, nmse_y_global=nmse_y_global,
        nmse_global_matched=nmse_global_matched,
        nmse_y_global_matched=nmse_y_global_matched,
        latency=latency,
        align_init=align_init, align_final=align_final,
        d_align=align_final - align_init, wind_align=wind_align,
        corr_cw=corr_cw,
        corr_w_k=corr_w_k, corr_g_k=corr_g_k, corr_g_w=corr_g_w,
        corr_g_phys=corr_g_phys, corr_w_phys=corr_w_phys,
        align_frac=align_frac, corr_focus=corr_focus,
        r2_law=r2_law, beta_align=float(std_beta[0]),
        beta_chem=float(std_beta[1]),
        corr_flux_g=corr_flux_g, focus_learned=mcf_l,
        focus_random=mcf_r, focus_gain=focus_gain, corr_flux_lr=corr_flux_lr,
        trans_gain=trans_gain,
        vnorm_max=vnorm_max, vnorm_final=float(vnorm[-1]),
        enc_rmse=enc_rmse,
        c_min=float(c_hist.min()), c_max=float(c_hist.max()),
        local_bytes=local_bytes, oracle_bytes=oracle_bytes,
        u_std=u_std, sigma=sigma,
    )


def _trial_wrapper(args):
    seed, cfg, T, train_frac, noise_frac, nx, ny = args
    return run_trial(seed, cfg, T, train_frac, noise_frac, nx, ny)


def _map(jobs, workers: int):
    if workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            return pool.map(_trial_wrapper, jobs)
    return [_trial_wrapper(j) for j in jobs]


# ---------------------------------------------------------------- tasks
def sweep(seeds: int, T: int = 4000, train_frac: float = 0.6,
          noise_frac: float = 0.05, workers: int = 1, nx: int = 7,
          ny: int = 7, seed0: int = 0) -> dict:
    cfg = default_cfg("weak")
    jobs = [(s, cfg, T, train_frac, noise_frac, nx, ny)
            for s in range(seed0, seed0 + seeds)]
    return {"weak": _map(jobs, workers)}


def ablation(seeds: int, T: int = 4000, train_frac: float = 0.6,
             noise_frac: float = 0.05, workers: int = 1) -> dict:
    variants = ["full", "no_geometry", "no_chemgate", "no_consolidation",
                "no_morphogenesis", "no_chemistry"]
    out = {}
    for name in variants:
        cfg = variant_cfg(name)
        jobs = [(s, cfg, T, train_frac, noise_frac, 7, 7) for s in range(seeds)]
        out[name] = _map(jobs, workers)
    return out


def scale(seeds: int, sizes: list, T: int = 4000, train_frac: float = 0.6,
          noise_frac: float = 0.05, workers: int = 1) -> dict:
    cfg = default_cfg("weak")
    out = {}
    for (nx, ny) in sizes:
        key = f"{nx}x{ny}"
        jobs = [(s, cfg, T, train_frac, noise_frac, nx, ny) for s in range(seeds)]
        out[key] = _map(jobs, workers)
    return out


# ---------------------------------------------------------------- summary
def summarize(results: dict) -> dict:
    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        a = np.array(vals, dtype=float)
        return dict(mean=float(a.mean()), std=float(a.std()), n=len(vals))

    keys = ("nmse", "nmse_persist", "nmse_y", "nmse_fd", "nmse_y_fd",
            "nmse_oracle", "nmse_y_oracle", "nmse_global", "nmse_y_global",
            "nmse_global_matched", "nmse_y_global_matched",
            "floor", "latency",
            "align_init", "align_final", "d_align", "wind_align", "corr_cw",
            "corr_w_k", "corr_g_k", "corr_g_w", "corr_g_phys", "corr_w_phys",
            "align_frac", "corr_focus",
            "r2_law", "beta_align", "beta_chem", "corr_flux_g",
            "focus_learned", "focus_random", "focus_gain", "corr_flux_lr",
            "trans_gain",
            "vnorm_max", "vnorm_final", "enc_rmse", "c_min", "c_max",
            "local_bytes", "oracle_bytes")
    out = {}
    for name, runs in results.items():
        s = {k: stats([r[k] for r in runs]) for k in keys}
        s["n_converged"] = sum(1 for r in runs if r["latency"] is not None)
        s["n_seeds"] = len(runs)
        s["n_geo_beats_w"] = sum(1 for r in runs
                                 if r["corr_g_k"] > r["corr_w_k"])
        s["n_geo_beats_w_phys"] = sum(1 for r in runs
                                      if r["corr_g_phys"] > r["corr_w_phys"])
        s["n_trans_pos"] = sum(1 for r in runs if r["trans_gain"] > 0.0)
        lb = s["local_bytes"]["mean"] if s["local_bytes"] else 1.0
        ob = s["oracle_bytes"]["mean"] if s["oracle_bytes"] else 1.0
        s["memory_ratio"] = ob / lb
        s["x_oracle"] = (s["nmse_y_oracle"]["mean"] / s["nmse_y"]["mean"]
                         if s["nmse_y_oracle"] and s["nmse_y"] and
                         s["nmse_y"]["mean"] > 0 else None)
        s["x_fd"] = (s["nmse_y_fd"]["mean"] / s["nmse_y"]["mean"]
                     if s["nmse_y_fd"] and s["nmse_y"] and
                     s["nmse_y"]["mean"] > 0 else None)
        s["x_global"] = (s["nmse_y_global"]["mean"] / s["nmse_y"]["mean"]
                         if s["nmse_y_global"] and s["nmse_y"] and
                         s["nmse_y"]["mean"] > 0 else None)
        out[name] = s
    return out


def print_table(s: dict, title: str):
    print("=" * 80)
    print(title)
    print("=" * 80)
    rows = [
        ("Physical one-step NMSE (streaming weak-form)", "nmse", 1.0),
        ("   persistence baseline", "nmse_persist", 1.0),
        ("   global batch oracle (frozen)", "nmse_oracle", 1.0),
        ("   streaming-global RLS (tuned)", "nmse_global", 1.0),
        ("   streaming-global RLS (matched)", "nmse_global_matched", 1.0),
        ("   finite-difference RLS", "nmse_fd", 1.0),
        ("   observation-noise floor", "floor", 1.0),
        ("Weak-form NMSE  (law fit)", "nmse_y", 1.0),
        ("   global batch oracle (law fit)", "nmse_y_oracle", 1.0),
        ("   streaming-global RLS (law fit)", "nmse_y_global", 1.0),
        ("   streaming-global RLS matched (law fit)",
         "nmse_y_global_matched", 1.0),
        ("   finite-difference law fit", "nmse_y_fd", 1.0),
        ("Convergence latency (s)", "latency", 1.0),
        ("Aim coherence init -> final", "align_init", 1.0),
        ("Aim coherence rise  dA", "d_align", 1.0),
        ("Beam field . wind (drift axis)", "wind_align", 1.0),
        ("corr(c, beam . wind)", "corr_cw", 1.0),
        ("readability: corr(w, kappa)", "corr_w_k", 1.0),
        ("coupling: corr(a_ij, full kappa)", "corr_g_k", 1.0),
        ("read-back: corr(a_ij, kappa_phys)", "corr_g_phys", 1.0),
        ("read-back: corr(w, kappa_phys)", "corr_w_phys", 1.0),
        ("shape vs weights: corr(a_ij, w)", "corr_g_w", 1.0),
        ("aligned-edge fraction", "align_frac", 1.0),
        ("corridor alignment contrast", "corr_focus", 1.0),
        ("law R^2 (a_ij+chem)", "r2_law", 1.0),
        ("law beta: angulation", "beta_align", 1.0),
        ("law beta: chemistry", "beta_chem", 1.0),
        ("routing: corr(flux, a_ij)", "corr_flux_g", 1.0),
        ("flux-weighted corridor c (learned)", "focus_learned", 1.0),
        ("flux-weighted corridor c (random geom)", "focus_random", 1.0),
        ("routing gain (learned-random)", "focus_gain", 1.0),
        ("corr(flux_learned, flux_random)", "corr_flux_lr", 1.0),
        ("corridor transmission gain", "trans_gain", 1.0),
        ("Encoding RMSE (rank-1 of w)", "enc_rmse", 1.0),
        ("v-norm max (windup check)", "vnorm_max", 1.0),
        ("v-norm final (metabolic r)", "vnorm_final", 1.0),
        ("Chemical field c_min", "c_min", 1.0),
        ("Chemical field c_max", "c_max", 1.0),
        ("Local runtime memory (bytes)", "local_bytes", 1.0),
        ("Batch estimator memory (bytes)", "oracle_bytes", 1.0),
    ]
    print(f"{'metric':<46}{'mean':>14}{'std':>14}")
    print("-" * 80)
    for name, key, _ in rows:
        v = s.get(key)
        if v is None:
            print(f"{name:<46}{'-':>14}{'-':>14}")
        else:
            print(f"{name:<46}{v['mean']:>14.4f}{v['std']:>14.4f}")
    print("-" * 80)
    print(f"Seeds converged: {s['n_converged']}/{s['n_seeds']}")
    print(f"Memory ratio (batch frame-buffer / local streaming): "
          f"{s['memory_ratio']:.1f}x")
    print(f"Law-fit advantage vs frozen batch oracle: "
          f"{s['x_oracle']:.1f}x" if s["x_oracle"] else "")
    print(f"Law-fit advantage vs streaming-global RLS: "
          f"{s['x_global']:.1f}x" if s["x_global"] else "")
    print(f"Shape beats weights in {s['n_geo_beats_w']}/{s['n_seeds']} seeds; "
          f"transmission gain positive in {s['n_trans_pos']}/{s['n_seeds']}")
    print("=" * 80)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sweep", "ablation", "scale"],
                    default="sweep")
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first seed index (for chunked sweeps)")
    ap.add_argument("--T", type=int, default=4000)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", type=str, default="results.json")
    ap.add_argument("--sizes", type=str, default="11x11,15x15")
    ap.add_argument("--variants", type=str, default=None)
    args = ap.parse_args()
    import os
    workers = args.workers if args.workers > 0 else min(
        max(os.cpu_count() or 1, 1), args.seeds)

    t0 = time.time()
    if args.task == "sweep":
        results = sweep(args.seeds, T=args.T, noise_frac=args.noise,
                        workers=workers, seed0=args.seed0)
        summary = summarize(results)
        print_table(summary["weak"], "MORPHOGENETIC BI-MODAL NETWORK -- "
                    f"{args.seeds}-SEED SWEEP ({args.T} STEPS)")
    elif args.task == "ablation":
        variants = None if args.variants is None else \
            [v.strip() for v in args.variants.split(",")]
        results = ablation(args.seeds, T=args.T, noise_frac=args.noise,
                           workers=workers)
        if variants is not None:
            results = {v: results[v] for v in variants}
        summary = summarize(results)
        for name in results:
            print_table(summary[name], f"ABLATION: {name}")
    else:
        sizes = [tuple(int(x) for x in s.split("x")) for s in
                 args.sizes.split(",")]
        results = scale(args.seeds, sizes, T=args.T, noise_frac=args.noise,
                        workers=workers)
        summary = summarize(results)
        for name in results:
            print_table(summary[name], f"SCALE: {name} grid ({args.seeds} seeds)")
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_seed": results}, f, indent=2,
                  default=str)
    print(f"\nsaved -> {args.out}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
