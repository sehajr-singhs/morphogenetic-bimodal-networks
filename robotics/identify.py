"""
identify.py -- streaming law identification for robot logs.

Given the noisy observable streams (q, q_dot, tau) a robot controller
records, recover the inverse-dynamics law  tau = Theta * Phi(q, qd, qdd).

MOMENTUM FORM, per joint, minimal regressors (the Coriolis terms cancel
into exact total derivatives; each joint uses only the columns its law
needs -- this keeps the regressor well-conditioned, cond ~ 10^2):

    tau1 = A0 a1 + B0 a2 + B2 c2 (2 a1 + a2)
           + G1 cos q1 + G2 cos(q1+q2) + b1 v1 + fc1 sign(v1)
    tau2 = B0 a1 + B0 a2 + B2 c2 a1 + B2 s2 v1^2
           + G2 cos(q1+q2) + b2 v2 + fc2 sign(v2)

    Phi1_raw = [1, cos q1, cos(q1+q2), a1, a2, c2 (2a1+a2), v1, sign v1]
    Phi2_raw = [1, cos(q1+q2), a1, a2, c2 a1, s2 v1^2, v2, sign v2]

    Theta1* = [0, G1, G2, A0, B0, B2, b1, fc1]
    Theta2* = [0, G2, B0, B0, B2, B2, b2, fc2]

THE WEAK FORM is the WINDOW of each identity.  The one-pole window W
(EMA) is linear, so  W[tau] = Theta * W[Phi_raw]  holds pointwise, and
integration by parts turns every acceleration into a weak derivative
with no differentiation of any signal:

    W[a1]           = y_{v1} = lam (v1 - W[v1])
    W[c2 (2a1+a2)]  = y_{c2(2v1+v2)} + W[s2 v2 (2v1+v2)]
    W[c2 a1]        = y_{c2 v1} + W[s2 v1 v2]

so the weak-form regressors are all windowed signals and weak
derivatives, and the target is W[tau].  The fit is per-joint streaming
RLS on COLUMN-NORMALIZED features (the weak columns span 1..10^2 in
scale), with exponential forgetting at the law-drift timescale (~1 s).

Estimators of Theta*:
  * WeakForm : per-joint streaming RLS on the weak-form regressors.
  * FD-RLS   : identical streaming RLS on the raw regressors with
               central-difference accelerations (classic rigid-body ID).
  * BatchWeak: ridge LS over the whole training window on the weak-form
               regressors (the global-batch oracle).

Metrics (holdout window the estimator never trained on):
  * windowed-domain law-fit NMSE (weak form)
  * torque-domain reconstruction NMSE -- the field metric: predict the
    torque the law says the robot needs, with the estimator's own
    acceleration estimate (y_v for weak, central difference for FD)
  * mechanism angle error: angle between the identified law vector and
    Theta*  (the 'shape is the mechanism' readout).
"""

from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from biomaterial_net import IIRWindow            # noqa: E402  (verified noise armor)
from arm import Arm2DOF                         # noqa: E402

D1, D2 = 8, 8                                    # per-joint basis sizes


# ---------------------------------------------------------------- raw bases
def basis1_raw(q, v, a):
    """Joint-1 momentum regressor: 8 columns."""
    q1, q2 = q
    v1, v2 = v
    s2, c2 = np.sin(q2), np.cos(q2)
    return np.array([1.0, np.cos(q1), np.cos(q1 + q2),
                     a[0], a[1], c2 * (2 * a[0] + a[1]),
                     v1, np.sign(v1)])


def basis2_raw(q, v, a):
    """Joint-2 momentum regressor: 8 columns."""
    q1, q2 = q
    v1, v2 = v
    s2, c2 = np.sin(q2), np.cos(q2)
    return np.array([1.0, np.cos(q1 + q2),
                     a[0], a[1], c2 * a[0], s2 * v1 ** 2,
                     v2, np.sign(v2)])


RAW_BASES = (basis1_raw, basis2_raw)


def theta_star(arm: Arm2DOF):
    """(2, 8) exact law coefficients, per-joint minimal bases."""
    c = arm.law_constants()
    t1 = np.array([0.0, c["G1"], c["G2"], c["A0"], c["B0"], c["B2"],
                   arm.b1, arm.fc1])
    t2 = np.array([0.0, c["G2"], c["B0"], c["B0"], c["B2"], c["B2"],
                   arm.b2, arm.fc2])
    return np.stack([t1, t2])


# ---------------------------------------------------------------- weak form
class _WindowBank:
    """IIR windows for a per-joint weak-form regressor."""

    def __init__(self, lam, dt, names):
        self.win = {k: IIRWindow(lam, dt, ()) for k in names}
        self.lam = lam

    def push(self, name, f):
        return self.win[name].push(f)

    def wd(self, name, f):
        """Weak derivative of f using window `name` (push + return)."""
        A = self.win[name].push(f)
        return self.lam * (f - A)


class WeakFormID:
    """Per-joint streaming RLS on column-normalized weak-form regressors."""

    def __init__(self, lam: float, ridge: float = 1e-3, lam_rls: float = 0.2,
                 dt: float = 2e-3, warmup: int = 150):
        self.lam, self.dt = lam, dt
        self.ridge = ridge
        self.alpha_r = np.exp(-lam_rls * dt)
        self.warmup = warmup
        self._n = 0
        # joint-1 windows
        self.w1 = _WindowBank(lam, dt, ("c1", "c12", "v1", "v2", "t1",
                                        "f2", "f6", "sg1"))
        # joint-2 windows
        self.w2 = _WindowBank(lam, dt, ("c12", "v1", "v2", "t2",
                                        "f3", "f4", "f5", "sg2"))
        self.M = [np.ones(D1), np.ones(D2)]          # running column scales
        self.C = [np.zeros((D1, D1)), np.zeros((D2, D2))]
        self.r = [np.zeros(D1), np.zeros(D2)]
        self.th = [np.zeros(D1), np.zeros(D2)]

    # ------------------------------------------------------- regressors
    def _phi1_weak(self, q, v):
        q1, q2 = q
        v1, v2 = v
        s2, c2 = np.sin(q2), np.cos(q2)
        w = self.w1
        Wc1 = w.push("c1", np.cos(q1))
        Wc12 = w.push("c12", np.cos(q1 + q2))
        Wv1 = w.push("v1", v1)
        yv1 = self.lam * (v1 - Wv1)
        Wv2 = w.push("v2", v2)
        yv2 = self.lam * (v2 - Wv2)
        f2 = c2 * (2 * v1 + v2)
        y2 = w.wd("f2", f2)
        W6 = w.push("f6", s2 * v2 * (2 * v1 + v2))
        Wsg1 = w.push("sg1", np.sign(v1))
        return np.array([1.0, Wc1, Wc12, yv1, yv2, y2 + W6, Wv1, Wsg1])

    def _phi2_weak(self, q, v):
        q1, q2 = q
        v1, v2 = v
        s2, c2 = np.sin(q2), np.cos(q2)
        w = self.w2
        Wc12 = w.push("c12", np.cos(q1 + q2))
        Wv1 = w.push("v1", v1)
        yv1 = self.lam * (v1 - Wv1)
        Wv2 = w.push("v2", v2)
        yv2 = self.lam * (v2 - Wv2)
        f3 = c2 * v1
        y3 = w.wd("f3", f3)
        W4 = w.push("f4", s2 * v1 * v2)
        W5 = w.push("f5", s2 * v1 ** 2)
        Wsg2 = w.push("sg2", np.sign(v2))
        return np.array([1.0, Wc12, yv1, yv2, y3 + W4, W5, Wv2, Wsg2])

    # ------------------------------------------------------- RLS
    def _step(self, j, z, y):
        m = self.M[j]
        m[:] = self.alpha_r * m + (1 - self.alpha_r) * z ** 2
        s = np.sqrt(m) + 1e-9
        zn = z / s
        C, r = self.C[j], self.r[j]
        C[:] = self.alpha_r * C + (1 - self.alpha_r) * np.outer(zn, zn)
        r[:] = self.alpha_r * r + (1 - self.alpha_r) * zn * y
        tr = np.trace(C) / len(zn)
        A = C + (self.ridge * tr + 1e-9) * np.eye(len(zn))
        self.th[j] = np.linalg.solve(A, r) / s

    def law(self) -> np.ndarray:
        return np.stack(self.th)

    # ------------------------------------------------------- streaming fit
    def fit(self, q, v, tau, t_lo=0, t_hi=None, store=True, store_law=False):
        self._n = 0
        T = len(q)
        t_hi = T if t_hi is None else t_hi
        Z = np.zeros((T, D1 + D2)); WT = np.zeros((T, 2)); A = np.zeros((T, 2))
        LH = np.zeros((T, D1 + D2)) if store_law else None
        for t in range(T):
            z1 = self._phi1_weak(q[t], v[t])
            z2 = self._phi2_weak(q[t], v[t])
            Wt1 = self.w1.push("t1", tau[t, 0])
            Wt2 = self.w2.push("t2", tau[t, 1])
            if t_lo <= t < t_hi and self._n >= self.warmup:
                self._step(0, z1, Wt1)
                self._step(1, z2, Wt2)
            self._n += 1
            if store_law:
                LH[t] = np.concatenate([self.th[0], self.th[1]])
            if store:
                Z[t] = np.concatenate([z1, z2])
                WT[t] = [Wt1, Wt2]
                A[t] = [self.lam * (v[t, 0] - self.w1.win["v1"].A),
                        self.lam * (v[t, 1] - self.w2.win["v2"].A)]
        res = dict(law=self.law())
        if store:
            res.update(Z=Z, WT=WT, A=A)
        if store_law:
            res["law_hist"] = LH
        return res

    # ------------------------------------------------------- evaluation
    def eval_domains(self, res, q, v, tau, t0, t1):
        th = res["law"]
        Z, WT = res["Z"], res["WT"]
        pred = np.stack([Z[t0:t1, :8] @ th[0], Z[t0:t1, 8:] @ th[1]], axis=1)
        e_w = np.mean((WT[t0:t1] - pred) ** 2, axis=0) / \
            (WT[t0:t1].var(axis=0) + 1e-12)
        # torque-domain reconstruction with the estimator's own acceleration
        tauhat = np.stack([self._reconstruct(th, q[t], v[t], res["A"][t])
                           for t in range(t0, t1)])
        e_t = np.mean((tau[t0:t1] - tauhat) ** 2, axis=0) / \
            (tau[t0:t1].var(axis=0) + 1e-12)
        return dict(nmse_weak=float(e_w.mean()), nmse_tau=float(e_t.mean()))

    def _reconstruct(self, th, q, v, a):
        return np.array([th[0] @ basis1_raw(q, v, a),
                         th[1] @ basis2_raw(q, v, a)])

    def angle_errors(self, res, theta_star: np.ndarray) -> np.ndarray:
        th = res["law"]
        ang = np.zeros(2)
        for j in range(2):
            a_, b = th[j], theta_star[j]
            ang[j] = np.degrees(np.arccos(np.clip(
                (a_ @ b) / (np.linalg.norm(a_) * np.linalg.norm(b) + 1e-12), -1, 1)))
        return ang


# ---------------------------------------------------------------- FD baseline
class FDID:
    """Identical streaming RLS on raw regressors with finite-difference
    accelerations -- the classic rigid-body identification pipeline."""

    def __init__(self, ridge: float = 1e-3, lam_rls: float = 0.2,
                 dt: float = 2e-3, warmup: int = 20):
        self.ridge, self.dt, self.warmup = ridge, dt, warmup
        self.alpha_r = np.exp(-lam_rls * dt)
        self.M = [np.ones(D1), np.ones(D2)]
        self.C = [np.zeros((D1, D1)), np.zeros((D2, D2))]
        self.r = [np.zeros(D1), np.zeros(D2)]
        self.th = [np.zeros(D1), np.zeros(D2)]
        self._n = 0

    def fit(self, q, v, tau, t_lo=0, t_hi=None, store=True, store_law=False):
        T = len(q)
        t_hi = T if t_hi is None else t_hi
        qdd = np.zeros((T, 2))
        qdd[1:-1] = (v[2:] - v[:-2]) / (2 * self.dt)
        Z = np.zeros((T, D1 + D2))
        LH = np.zeros((T, D1 + D2)) if store_law else None
        for t in range(T):
            z1 = basis1_raw(q[t], v[t], qdd[t])
            z2 = basis2_raw(q[t], v[t], qdd[t])
            if t_lo <= t < t_hi and t >= self.warmup and t < T - 1:
                self._step(0, z1, tau[t, 0])
                self._step(1, z2, tau[t, 1])
            self._n += 1
            if store_law:
                LH[t] = np.concatenate([self.th[0], self.th[1]])
            if store:
                Z[t] = np.concatenate([z1, z2])
        res = dict(law=np.stack(self.th))
        if store:
            res["Z"] = Z
        if store_law:
            res["law_hist"] = LH
        return res

    def _step(self, j, z, y):
        m = self.M[j]
        m[:] = self.alpha_r * m + (1 - self.alpha_r) * z ** 2
        s = np.sqrt(m) + 1e-9
        zn = z / s
        C, r = self.C[j], self.r[j]
        C[:] = self.alpha_r * C + (1 - self.alpha_r) * np.outer(zn, zn)
        r[:] = self.alpha_r * r + (1 - self.alpha_r) * zn * y
        tr = np.trace(C) / len(zn)
        A = C + (self.ridge * tr + 1e-9) * np.eye(len(zn))
        self.th[j] = np.linalg.solve(A, r) / s

    def eval_domains(self, res, q, v, tau, t0, t1):
        th = res["law"]
        Z = res["Z"]
        e_t = np.zeros(2)
        for j in range(2):
            pred = Z[t0:t1, j * 8:(j + 1) * 8] @ th[j]
            e_t[j] = np.mean((tau[t0:t1, j] - pred) ** 2) / \
                (tau[t0:t1, j].var() + 1e-12)
        return dict(nmse_weak=float(e_t.mean()), nmse_tau=float(e_t.mean()))

    def angle_errors(self, res, theta_star):
        th = res["law"]
        ang = np.zeros(2)
        for j in range(2):
            a_, b = th[j], theta_star[j]
            ang[j] = np.degrees(np.arccos(np.clip(
                (a_ @ b) / (np.linalg.norm(a_) * np.linalg.norm(b) + 1e-12), -1, 1)))
        return ang


# ---------------------------------------------------------------- batch oracle
def batch_weak(q, v, tau, lam: float, theta_star: np.ndarray, ridge: float = 1e-3,
               dt: float = 2e-3, t_lo: int = 0, t_hi: int | None = None,
               warmup: int = 150):
    """Global-batch ridge LS on the weak-form regressors (the oracle)."""
    T = len(q)
    t_hi = T if t_hi is None else t_hi
    idf = WeakFormID(lam, ridge, dt=dt, warmup=warmup)
    res = idf.fit(q, v, tau, t_lo=0, t_hi=t_hi)
    Z, WT = res["Z"], res["WT"]
    th = np.zeros((2, 8))
    for j in range(2):
        X = Z[t_lo:t_hi, j * 8:(j + 1) * 8]
        s = np.sqrt(np.mean(X ** 2, axis=0) + 1e-9)
        Xn = X / s
        y = WT[t_lo:t_hi, j]
        A = Xn.T @ Xn + ridge * np.eye(8) + 1e-9 * np.eye(8)
        th[j] = np.linalg.solve(A, Xn.T @ y) / s
    res2 = dict(law=th, Z=Z, WT=WT, A=res["A"])
    return (res2, idf.eval_domains(res2, q, v, tau, t_hi, T),
            idf.angle_errors(res2, theta_star))


def tune_lam(q, v, tau, theta_star, dt=2e-3, lams=(20.0, 40.0, 80.0, 160.0),
             ridge=1e-3, seed=0):
    """Pick lam on a validation slice inside the training window (honest)."""
    T = len(q)
    t_val_lo, t_val_hi = int(0.45 * T), int(0.55 * T)
    best, best_lam = None, lams[0]
    for lam in lams:
        idf = WeakFormID(lam, ridge, dt=dt)
        res = idf.fit(q, v, tau, t_lo=0, t_hi=t_val_lo)
        m = idf.eval_domains(res, q, v, tau, t_val_lo, t_val_hi)
        if best is None or m["nmse_weak"] < best:
            best, best_lam = m["nmse_weak"], lam
    return best_lam
