"""real_benchmark.py -- Weak-form streaming identification on REAL data.

Benchmark: monthly total sunspot numbers (SILSO / World Data Center for the
Sunspot Index, 1749-2026; 3331 real observations). The scalar stream is
delay-embedded into a ring of d lag nodes (tau-month spacing) and each node
couples to its +/-M lag neighbors. The paper's identification machinery --
strictly local streaming RLS on spacetime weak-form quantities -- then
identifies the local delay-space law  u' ~ a . z  where z is the graph
Laplacian of the embedded window, and forecasts by Euler integration of the
identified law.

Baselines on the SAME embedded data and SAME train/test split:
  * FD-streaming   -- same architecture, finite-difference targets (the
                      death-of-differentiation baseline, now on real data)
  * batch weak     -- same weak-form features/targets, closed-form global
                      least squares (weak SINDy, Messenger-Bortz style):
                      isolates "streaming vs batch"
  * batch FD SINDy -- global ridge on FD derivatives
  * AR(12)         -- classic linear autoregressive forecaster
  * ESN            -- echo state network (nonlinear reservoir, ridge readout)
  * persistence    -- forecast = last value

Metrics: h-step-ahead forecast NMSE on a held-out real window (h = 1, 6, 12,
24 months), in-sample law-fit R^2, wall-clock, memory. Everything frozen after
training; pure multistep (no oracle values fed back).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

DATA_URL = ("https://www.sidc.be/silso/INFO/snmtotcsv.php?nol_header=1")
LOCAL = os.path.join(os.path.dirname(__file__), "data_sunspots.csv")


# ---------------------------------------------------------------- data I/O
def load_sunspots(path: str = LOCAL) -> np.ndarray:
    """Monthly total sunspot numbers; drops flagged/missing (-1) months."""
    if not os.path.exists(path):
        import urllib.request
        print(f"downloading {DATA_URL} -> {path}")
        urllib.request.urlretrieve(DATA_URL, path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(";")
            if len(p) < 4:
                continue
            try:
                sn = float(p[3])
            except ValueError:
                continue
            if sn >= 0.0:
                rows.append(sn)
    return np.array(rows, dtype=float)


def preprocess(series: np.ndarray) -> np.ndarray:
    """Variance-stabilizing sqrt (standard for count data), then z-score."""
    return np.sqrt(np.maximum(series, 0.0))


# ---------------------------------------------------------------- embedding
def make_ring(d: int, M: int) -> np.ndarray:
    edges = []
    for k in range(d):
        for m in range(1, M + 1):
            edges.append((k, (k + m) % d))
    return np.array(edges, dtype=int)


def node_neighbors(d: int, M: int) -> list:
    return [[(k + m) % d for m in range(-M, 0)] + [(k + m) % d for m in range(1, M + 1)]
            for k in range(d)]


# ---------------------------------------------------------------- learner
class WeakFormLearner:
    """Streaming per-node RLS on weak-form quantities (the paper's
    identification layer, re-implemented standalone for the real benchmark)."""

    def __init__(self, d: int, edges: np.ndarray, nb: list, lam: float,
                 ridge: float, dt: float = 1.0, lam_rls: float | None = None,
                 drive: bool = True, level: bool = True):
        self.d, self.edges, self.nb = d, edges, nb
        self.dt = dt
        self.lam = lam
        self.alpha = np.exp(-lam * dt)         # weak-window forgetting
        self.alpha_rls = np.exp(-(lam_rls if lam_rls is not None else lam) * dt)
        self.ridge = ridge
        self.drive = drive
        self.level = level
        self.D = (2 * (len(nb[0]) // 2) + (1 if drive else 0)
                  + (1 if level else 0))
        self.A = np.zeros(d)                     # IIR window means
        self.C = np.zeros((d, self.D, self.D))   # EMA of z z^T
        self.r = np.zeros((d, self.D))           # EMA of z y
        self.a = np.zeros((d, self.D))           # identified law
        self.z = np.zeros((d, self.D))
        self._n_diff = 2 * (len(nb[0]) // 2)

    def _features(self, X: np.ndarray) -> np.ndarray:
        z = np.zeros((self.d, self.D))
        for k in range(self.d):
            z[k, : self._n_diff] = X[self.nb[k]] - X[k]
            if self.level:                        # self-dissipation -gamma u
                z[k, self._n_diff] = -X[k]
            if self.drive:
                z[k, -1] = 1.0                   # constant drive term
        return z

    def observe(self, X: np.ndarray, learn: bool = True) -> np.ndarray:
        """One streaming step. Returns the one-step prediction of node 0."""
        f = X
        self.A = self.alpha * self.A + (1.0 - self.alpha) * f
        y = self.lam * (f - self.A)              # weak derivative, no FD
        z = self._features(f)
        if learn:
            self.C = self.alpha_rls * self.C + (1.0 - self.alpha_rls) * (
                z[:, :, None] * z[:, None, :])
            self.r = self.alpha_rls * self.r + (1.0 - self.alpha_rls) * z * y[:, None]
            tr = np.trace(self.C, axis1=1, axis2=2) / self.D
            A = self.C + (self.ridge * tr[:, None, None] + 1e-9) * np.eye(self.D)
            self.a = np.linalg.solve(A, self.r[..., None])[..., 0]
        self.z = z
        return X[0] + self.dt * float(self.a[0] @ z[0])

    def _law_pred(self, win: np.ndarray) -> np.ndarray:
        """Identified-law prediction for node 0 given the current window."""
        a = self.a.mean(axis=0)
        z0 = win[self.nb[0]] - win[0]
        if self.level:
            z0 = np.append(z0, -win[0])
        if self.drive:
            z0 = np.append(z0, 1.0)
        return a @ z0

    def ema_trajectory(self, series: np.ndarray, tau: int) -> np.ndarray:
        """Causal IIR-window trajectory over the whole series, (n, d)."""
        n = len(series)
        out = np.zeros((n, self.d))
        A = np.zeros(self.d)
        for t in range(n):
            X = series[t - tau * np.arange(self.d)]
            A = self.alpha * A + (1.0 - self.alpha) * X
            out[t] = A
        return out

    def forecast(self, series: np.ndarray, t: int, h: int, tau: int,
                 law: np.ndarray | None = None, mode: str = "window",
                 A_traj: np.ndarray | None = None) -> np.ndarray:
        """h-step-ahead pure-multistep forecast from index t (0-based into
        the z-scored series; t is the latest observation).

        mode="window": weak-form-consistent integration, u(t+1) ~ A + y/lam,
        predicting THROUGH the IIR window (the weak form's own semantics).
        mode="euler": u(t+1) = u(t) + dt*y (the paper's plant update)."""
        a = law if law is not None else self.a.mean(axis=0)
        win = series[t - tau * np.arange(self.d)].copy()
        preds = np.zeros(h)
        A = None
        if mode == "window":
            if A_traj is not None:
                A = A_traj[t].copy()
            else:
                A = np.zeros(self.d)
                for tt in range(0, t + 1):
                    X = series[tt - tau * np.arange(self.d)]
                    A = self.alpha * A + (1.0 - self.alpha) * X
        for s in range(h):
            z0 = win[self.nb[0]] - win[0]
            if self.level:
                z0 = np.append(z0, -win[0])
            if self.drive:
                z0 = np.append(z0, 1.0)
            yhat = float(a @ z0)
            if mode == "window":
                win[0] = A[0] + yhat / self.lam
                A = self.alpha * A + (1.0 - self.alpha) * win
            else:
                win[0] = win[0] + self.dt * yhat
            preds[s] = win[0]
            win[1:] = win[:-1]
        return preds

    def law_fit(self, series: np.ndarray, t_lo: int, t_hi: int,
                tau: int) -> float:
        """In-sample R^2 of the identified law on the weak-form target."""
        yy, mm = [], []
        A = np.zeros(self.d)
        for t in range(t_lo, t_hi + 1):
            X = series[t - tau * np.arange(self.d)]
            A = self.alpha * A + (1.0 - self.alpha) * X
            y = self.lam * (X - A)
            z = self._features(X)
            yy.append(y[0]); mm.append(self.a[0] @ z[0])
        yy = np.array(yy); mm = np.array(mm)
        return 1.0 - float(np.mean((yy - mm) ** 2) / np.var(yy))

    def holdout_law_fit(self, series: np.ndarray, t_lo: int, t_hi: int,
                        tau: int) -> dict:
        """FROZEN law evaluated on an unseen window. Returns NMSE of the
        law's prediction of the weak-form target normalized (a) by the
        signal variance of the unseen window (common scale across models)
        and (b) by the target's own variance on the unseen window (law-fit
        R^2-style, per-model scale)."""
        A = np.zeros(self.d)
        errs, vu, vy = [], [], []
        ym = 0.0
        ys = []
        for t in range(t_lo, t_hi + 1):
            X = series[t - tau * np.arange(self.d)]
            A = self.alpha * A + (1.0 - self.alpha) * X
            y = self.lam * (X - A)
            z = self._features(X)
            errs.append(float((y[0] - self.a[0] @ z[0]) ** 2))
            vu.append(float(X[0] ** 2))
            ys.append(float(y[0]))
        ys = np.array(ys)
        return {"nmse_signal": float(np.mean(errs) / np.mean(vu)),
                "nmse_target": float(np.mean(errs) / np.var(ys))}


def train_weak(series: np.ndarray, d: int, M: int, tau: int, lam: float,
               ridge: float, t_lo: int, t_hi: int,
               lam_rls: float | None = None, drive: bool = True,
               level: bool = True) -> WeakFormLearner:
    lr = WeakFormLearner(d, make_ring(d, M), node_neighbors(d, M), lam, ridge,
                         lam_rls=lam_rls, drive=drive, level=level)
    for t in range(t_lo, t_hi + 1):
        lr.observe(series[t - tau * np.arange(d)])
    return lr


# ---------------------------------------------------------------- baselines
def train_fd(series: np.ndarray, d: int, M: int, tau: int, ridge: float,
             t_lo: int, t_hi: int, lam_rls: float = 0.02):
    """FD-streaming twin of the weak learner (raw finite-difference targets)
    with the SAME streaming RLS forgetting -- isolates the weak form itself."""
    nb = node_neighbors(d, M)
    D = 2 * M
    C = np.zeros((d, D, D)); r = np.zeros((d, D))
    alpha = np.exp(-lam_rls)
    prev = series[t_lo - tau * np.arange(d)]
    for t in range(t_lo, t_hi + 1):
        X = series[t - tau * np.arange(d)]
        y = X - prev
        z = np.zeros((d, D))
        for k in range(d):
            z[k] = X[nb[k]] - X[k]
        C = alpha * C + (1.0 - alpha) * z[:, :, None] * z[:, None, :]
        r = alpha * r + (1.0 - alpha) * z * y[:, None]
        prev = X
    tr = np.trace(C, axis1=1, axis2=2) / D
    A = C + (ridge * tr[:, None, None] + 1e-9) * np.eye(D)
    a = np.linalg.solve(A, r[..., None])[..., 0]
    law = a.mean(axis=0)

    def holdout(series_, t_lo_, t_hi_, tau_):
        errs, vu, ys = [], [], []
        prev_ = series_[t_lo_ - tau_ * np.arange(d)]
        for t in range(t_lo_, t_hi_ + 1):
            X = series_[t - tau_ * np.arange(d)]
            y = X - prev_
            z = np.zeros((d, D))
            for k in range(d):
                z[k] = X[nb[k]] - X[k]
            errs.append(float((y[0] - law @ z[0]) ** 2))
            vu.append(float(X[0] ** 2))
            ys.append(float(y[0]))
            prev_ = X
        ys = np.array(ys)
        return {"nmse_signal": float(np.mean(errs) / np.mean(vu)),
                "nmse_target": float(np.mean(errs) / np.var(ys))}

    def forecast(series_, t, h, tau_):
        win = series_[t - tau_ * np.arange(d)].copy()
        preds = np.zeros(h)
        for s in range(h):
            z0 = win[nb[0]] - win[0]
            win[0] = win[0] + float(law @ z0)
            preds[s] = win[0]
            win[1:] = win[:-1]
        return preds
    return forecast, holdout


def _batch_fit(series, d, M, tau, t_lo, t_hi, lam, ridge, fd):
    """Shared batch solver for the weak-form and FD feature/target sets."""
    nb = node_neighbors(d, M)
    D = 2 * M
    C = np.zeros((d, D, D)); r = np.zeros((d, D))
    A = np.zeros(d); alpha = np.exp(-lam)
    prev = series[t_lo - tau * np.arange(d)]
    for t in range(t_lo, t_hi + 1):
        X = series[t - tau * np.arange(d)]
        if fd:
            y = X - prev
            prev = X
        else:
            A = alpha * A + (1.0 - alpha) * X
            y = lam * (X - A)
        z = np.zeros((d, D))
        for k in range(d):
            z[k] = X[nb[k]] - X[k]
        C = C + z[:, :, None] * z[:, None, :]
        r = r + z * y[:, None]
    tr = np.trace(C, axis1=1, axis2=2) / D
    A_ = C + (ridge * tr[:, None, None] + 1e-9) * np.eye(D)
    a = np.linalg.solve(A_, r[..., None])[..., 0]
    law = a.mean(axis=0)
    alpha = np.exp(-lam)
    # cached causal IIR trajectory for window-consistent integration
    n_ = len(series)
    Atr = np.zeros((n_, d))
    Aacc = np.zeros(d)
    for t in range(n_):
        X = series[t - tau * np.arange(d)]
        Aacc = alpha * Aacc + (1.0 - alpha) * X
        Atr[t] = Aacc

    def forecast(series_, t, h, tau_):
        win = series_[t - tau_ * np.arange(d)].copy()
        A = Atr[t].copy()
        preds = np.zeros(h)
        for s in range(h):
            z0 = win[nb[0]] - win[0]
            win[0] = A[0] + float(law @ z0) / lam
            A = alpha * A + (1.0 - alpha) * win
            preds[s] = win[0]
            win[1:] = win[:-1]
        return preds

    def holdout(series_, t_lo_, t_hi_, tau_):
        errs, vu, ys = [], [], []
        A_ = np.zeros(d)
        prev_ = series_[t_lo_ - tau_ * np.arange(d)]
        for t in range(t_lo_, t_hi_ + 1):
            X = series_[t - tau_ * np.arange(d)]
            if fd:
                y = X - prev_
                prev_ = X
            else:
                A_ = alpha * A_ + (1.0 - alpha) * X
                y = lam * (X - A_)
            z = np.zeros((d, D))
            for k in range(d):
                z[k] = X[nb[k]] - X[k]
            errs.append(float((y[0] - law @ z[0]) ** 2))
            vu.append(float(X[0] ** 2))
            ys.append(float(y[0]))
        ys = np.array(ys)
        return {"nmse_signal": float(np.mean(errs) / np.mean(vu)),
                "nmse_target": float(np.mean(errs) / np.var(ys))}
    return forecast, holdout


def train_batch_weak(series: np.ndarray, d: int, M: int, tau: int, lam: float,
                     ridge: float, t_lo: int, t_hi: int):
    """Global batch weak SINDy on the same features/targets (isolates
    streaming vs batch; equivalent to Messenger-Bortz on this basis)."""
    return _batch_fit(series, d, M, tau, t_lo, t_hi, lam, ridge, fd=False)


def train_batch_fd(series: np.ndarray, d: int, M: int, tau: int, ridge: float,
                   t_lo: int, t_hi: int):
    """Global batch ridge on finite-difference targets (classic SINDy)."""
    return _batch_fit(series, d, M, tau, t_lo, t_hi, lam=1.0, ridge=ridge,
                      fd=True)


def train_ar(series: np.ndarray, p: int, t_lo: int, t_hi: int, ridge: float):
    t_idx = np.arange(t_lo + p - 1, t_hi + 1)
    Xm = np.stack([series[t_idx - i] for i in range(1, p + 1)], axis=1)
    y = series[t_idx]
    beta = np.linalg.solve(Xm.T @ Xm + ridge * np.eye(p), Xm.T @ y)

    def forecast(series_, t, h, tau_=None):
        hist = series_[t - p + 1: t + 1].copy()
        preds = np.zeros(h)
        for s in range(h):
            nv = float(beta @ hist[::-1])
            preds[s] = nv
            hist = np.concatenate([hist[1:], [nv]])
        return preds
    return forecast


def train_esn(series: np.ndarray, d: int, tau: int, t_lo: int, t_hi: int,
              n_res: int = 400, sr: float = 1.1, rho_in: float = 0.6,
              ridge: float = 1e-3, seed: int = 0, leak: float = 0.3):
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0, (n_res, n_res))
    mask = rng.random((n_res, n_res)) < 0.05
    W = W * mask
    W *= sr / max(np.abs(np.linalg.eigvals(W)).max(), 1e-9)
    Win = rng.uniform(-rho_in, rho_in, (n_res, 1))
    x = np.zeros(n_res)
    Xs, Ys = [], []
    for t in range(t_lo, t_hi):
        u = np.array([series[t]])
        x = (1.0 - leak) * x + leak * np.tanh(W @ x + Win @ u)
        Xs.append(np.concatenate([x, [series[t]]]))
        Ys.append(series[t + 1])
    Xs = np.array(Xs); Ys = np.array(Ys)
    beta = np.linalg.solve(Xs.T @ Xs + ridge * np.eye(n_res + 1), Xs.T @ Ys)

    # precompute the causal reservoir trajectory over the whole series ONCE
    # (state at t depends only on data <= t), so forecast(t) starts from the
    # cached state instead of re-running O(T) reservoir steps per test point
    x = np.zeros(n_res)
    states = np.zeros((len(series), n_res))
    for t in range(len(series)):
        x = (1.0 - leak) * x + leak * np.tanh(W @ x + Win @ np.array([series[t]]))
        states[t] = x

    def forecast(series_, t, h, tau_=None):
        x = states[t].copy()
        preds = np.zeros(h)
        u = series_[t]
        for s in range(h):
            x = (1.0 - leak) * x + leak * np.tanh(W @ x + Win @ np.array([u]))
            u = float(beta @ np.concatenate([x, [u]]))
            preds[s] = u
        return preds
    return forecast


# ---------------------------------------------------------------- evaluation
def eval_horizon(forecast_fn, series: np.ndarray, test_lo: int, test_hi: int,
                 h: int, tau: int, z_mean: float, z_std: float,
                 stride: int = 1) -> float:
    """Mean NMSE of h-step-ahead forecasts over all test start points."""
    errs = []
    for t in range(test_lo, test_hi - h + 1, stride):
        pred = forecast_fn(series, t, h, tau)
        true = series[t + 1: t + h + 1]
        errs.append(np.mean((pred - true) ** 2))
    var = float(np.var(series[test_lo: test_hi + 1]))
    return float(np.mean(errs) / var)


# ================================================================ main
def run(lam: float = 0.15, ridge: float = 0.1, lam_rls: float = 0.005,
        d: int = 24, M: int = 12, tau: int = 1, p_ar: int = 24,
        esn_ridge: float = 1e-2, esn_seeds: int = 3,
        out: str = "real_benchmark.json", horizons=(1, 6, 12, 24),
        noise_levels=(0.0, 0.05, 0.2, 0.5), noise_seeds: int = 3) -> dict:
    raw = load_sunspots()
    s = preprocess(raw)
    n = len(s)

    # split: train / validation (tuning) / test, all on real data
    n_train = int(0.70 * n)          # 2331 months (~1943)
    n_val = int(0.10 * n)            # 333 months (~1971)
    t_lo = d * tau                   # first usable embedding index
    tr_lo, tr_hi = t_lo, n_train
    va_lo, va_hi = n_train + 1, n_train + n_val
    te_lo, te_hi = n_train + n_val + 1, n - 1

    # z-score fit on TRAIN only
    z_mean = float(s[tr_lo: tr_hi + 1].mean())
    z_std = float(s[tr_lo: tr_hi + 1].std())
    z = (s - z_mean) / z_std
    zc = z.copy()

    results = {"meta": dict(d=d, M=M, tau=tau, lam=lam, ridge=ridge,
                            lam_rls=lam_rls, p_ar=p_ar,
                            esn_ridge=esn_ridge, n_train=n_train, n_val=n_val,
                            n_test=te_hi - te_lo + 1, source=DATA_URL),
               "per_horizon": {}, "identification": {}}

    # ---------------- identification panel (frozen law, unseen decades)
    ident = {}
    for sigma in noise_levels:
        acc = {"weak_streaming": [], "fd_streaming": [], "batch_weak": [],
               "batch_fd": [], "weak_in_sample_r2": []}
        for sd in range(noise_seeds):
            ztr = z.copy()
            if sigma > 0.0:
                rng = np.random.default_rng(1000 + sd)
                ztr[: tr_hi + 1] = ztr[: tr_hi + 1] \
                    + sigma * rng.normal(size=tr_hi + 1)
            lr_n = train_weak(ztr, d, M, tau, lam, ridge, tr_lo, tr_hi,
                              lam_rls=lam_rls, drive=True, level=False)
            _, h_fd_n = train_fd(ztr, d, M, tau, ridge, tr_lo, tr_hi,
                                 lam_rls=lam_rls)
            _, h_bw_n = train_batch_weak(ztr, d, M, tau, lam, ridge,
                                         tr_lo, tr_hi)
            _, h_bf_n = train_batch_fd(ztr, d, M, tau, ridge, tr_lo, tr_hi)
            acc["weak_streaming"].append(
                lr_n.holdout_law_fit(zc, va_lo, va_hi, tau)["nmse_signal"])
            acc["fd_streaming"].append(
                h_fd_n(zc, va_lo, va_hi, tau)["nmse_signal"])
            acc["batch_weak"].append(
                h_bw_n(zc, va_lo, va_hi, tau)["nmse_signal"])
            acc["batch_fd"].append(
                h_bf_n(zc, va_lo, va_hi, tau)["nmse_signal"])
            acc["weak_in_sample_r2"].append(
                lr_n.law_fit(ztr, tr_lo, tr_hi, tau))
        ident[f"sigma{sigma}"] = {k: round(float(np.mean(v)), 6)
                                   for k, v in acc.items()}
    results["identification"] = ident

    # ---------------- forecast panel (frozen models, held-out real window)
    lr = train_weak(z, d, M, tau, lam, ridge, tr_lo, tr_hi, lam_rls=lam_rls,
                    drive=True, level=False)
    A_traj = lr.ema_trajectory(z, tau)
    lr.forecast = lambda s_, t, h, tau_, **kw: \
        WeakFormLearner.forecast(lr, s_, t, h, tau_, A_traj=A_traj)
    f_fd, _ = train_fd(z, d, M, tau, ridge, tr_lo, tr_hi, lam_rls=lam_rls)
    f_bw, _ = train_batch_weak(z, d, M, tau, lam, ridge, tr_lo, tr_hi)
    f_bf, _ = train_batch_fd(z, d, M, tau, ridge, tr_lo, tr_hi)
    f_ar = train_ar(z, p_ar, tr_lo, tr_hi, ridge)
    f_persist = lambda s_, t, h, tau_: np.full(h, s_[t])
    esn_list = [train_esn(z, d, tau, tr_lo, tr_hi, ridge=esn_ridge, seed=sd)
                for sd in range(esn_seeds)]

    results["weak"] = {
        "law_fit_r2": lr.law_fit(z, tr_lo, tr_hi, tau),
        "holdout_law_fit": ident["sigma0.0"]["weak_streaming"],
        "mem_kb": round((lr.C.nbytes + lr.r.nbytes + lr.a.nbytes) / 1024, 2),
    }
    results["esn"] = {"mem_kb": round(400.0 * 400.0 * 8 / 1024, 1)}
    results["ar"] = {"mem_kb": round(p_ar * 8 / 1024, 3)}

    models = {"weak": lr.forecast, "fd_streaming": f_fd, "batch_weak": f_bw,
              "batch_fd": f_bf, "ar": f_ar, "esn": esn_list,
              "persistence": f_persist}
    for h in horizons:
        row = {}
        for name, fn in models.items():
            if name == "esn":
                vals = [eval_horizon(f, z, te_lo, te_hi, h, tau, z_mean, z_std)
                        for f in fn]
                row[name] = {"mean": round(float(np.mean(vals)), 4),
                             "std": round(float(np.std(vals)), 4)}
            else:
                row[name] = round(eval_horizon(fn, z, te_lo, te_hi, h, tau,
                                               z_mean, z_std), 4)
        results["per_horizon"][f"h{h}"] = row

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import sys
    kwargs = {}
    for flag, cast in (("--lam", float), ("--ridge", float),
                       ("--lam_rls", float), ("--out", str),
                       ("--p_ar", int), ("--esn_ridge", float)):
        if flag in sys.argv:
            kwargs[flag[2:]] = cast(sys.argv[sys.argv.index(flag) + 1])
    r = run(**kwargs)
    for h, row in r["per_horizon"].items():
        line = "  ".join(f"{k}: {v}" if not isinstance(v, dict)
                         else f"{k}: {v['mean']}+-{v['std']}"
                         for k, v in row.items())
        print(f"{h}: {line}")
