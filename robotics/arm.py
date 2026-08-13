"""
arm.py -- 2-DOF planar industrial arm.

The plant a factory logs every day: two revolute joints under gravity,
viscous + Coulomb friction, an optional tip payload, and an optional
contact/insertion socket.  The simulator is exact rigid-body dynamics
(semi-implicit Euler at dt = 2 ms); the OBSERVABLES are noisy streams
(q, q_dot, tau) -- encoder + tachometer + torque-sensor noise -- exactly
what a robot controller records.

Inverse dynamics in MOMENTUM form (where the Coriolis terms cancel),
per joint with MINIMAL regressors (well-conditioned, cond ~ 10^2):

    tau1 = A0 a1 + B0 a2 + B2 c2 (2 a1 + a2)
           + G1 cos q1 + G2 cos(q1+q2) + b1 v1 + fc1 sign(v1)
    tau2 = B0 a1 + B0 a2 + B2 c2 a1 + B2 s2 v1^2
           + G2 cos(q1+q2) + b2 v2 + fc2 sign(v2)

    Phi1 = [1, cos q1, cos(q1+q2), a1, a2, c2 (2a1+a2), v1, sign v1]
    Phi2 = [1, cos(q1+q2), a1, a2, c2 a1, s2 v1^2, v2, sign v2]

The law is EXACTLY linear in the parameter vectors (given below), so
identification has a ground-truth target to converge to.

    theta1* = [0, G1, G2, A0, B0, B2, b1, fc1]
    theta2* = [0, G2, B0, B0, B2, B2, b2, fc2]

with the payload-dependent constants
    m2e = m2 + mp,  lc2e = (m2 lc2 + mp l2)/m2e,  I2e = I2 + mp l2^2
    G1 = m1 g lc1 + m2e g l1      G2 = m2e g lc2e
    A0 = I1 + I2e + m1 lc1^2 + m2e (l1^2 + lc2e^2)
    B0 = I2e + m2e lc2e^2         B2 = m2e l1 lc2e
"""

from __future__ import annotations

import numpy as np

G = 9.81


class Arm2DOF:
    """2-DOF planar arm.  All parameters SI (kg, m, Nm, rad)."""

    def __init__(self, m1=4.0, m2=3.0, l1=0.5, l2=0.45, lc1=0.25, lc2=0.22,
                 I1=0.12, I2=0.09, b1=0.35, b2=0.25, fc1=0.8, fc2=0.6,
                 mp=0.0, contact=None, dt=2e-3):
        self.dt = dt
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.lc1, self.lc2 = lc1, lc2
        self.I1, self.I2 = I1, I2
        self.b1, self.b2 = b1, b2
        self.fc1, self.fc2 = fc1, fc2
        self.contact = contact          # dict(sx, sy, rs, kc, cc) or None
        self.set_payload(mp)

    # ------------------------------------------------------------- physics
    def set_payload(self, mp: float) -> None:
        """Attach mass mp (kg) at the tip; updates the effective law."""
        self.mp = mp
        m2e = self.m2 + mp
        self.m2e = m2e
        self.lc2e = (self.m2 * self.lc2 + mp * self.l2) / m2e if m2e > 0 else self.lc2
        self.I2e = self.I2 + mp * self.l2 ** 2

    def law_constants(self):
        m2e, lc2e, I2e = self.m2e, self.lc2e, self.I2e
        m1, l1, lc1, I1 = self.m1, self.l1, self.lc1, self.I1
        G1 = (m1 * G * lc1 + m2e * G * l1)
        G2 = m2e * G * lc2e
        A0 = I1 + I2e + m1 * lc1 ** 2 + m2e * (l1 ** 2 + lc2e ** 2)
        B0 = I2e + m2e * lc2e ** 2
        B2 = m2e * l1 * lc2e
        return dict(G1=G1, G2=G2, A0=A0, B0=B0, B2=B2)

    def theta_star(self) -> np.ndarray:
        """(2, 8) exact law coefficients, per-joint minimal momentum bases."""
        c = self.law_constants()
        t1 = np.array([0.0, c["G1"], c["G2"], c["A0"], c["B0"], c["B2"],
                       self.b1, self.fc1])
        t2 = np.array([0.0, c["G2"], c["B0"], c["B0"], c["B2"], c["B2"],
                       self.b2, self.fc2])
        return np.stack([t1, t2])

    def basis1(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Joint-1 momentum regressor (8 cols)."""
        q1, q2 = q
        v1, v2 = v
        s2, c2 = np.sin(q2), np.cos(q2)
        return np.array([1.0, np.cos(q1), np.cos(q1 + q2),
                         a[0], a[1], c2 * (2 * a[0] + a[1]),
                         v1, np.sign(v1)])

    def basis2(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Joint-2 momentum regressor (8 cols)."""
        q1, q2 = q
        v1, v2 = v
        s2, c2 = np.sin(q2), np.cos(q2)
        return np.array([1.0, np.cos(q1 + q2),
                         a[0], a[1], c2 * a[0], s2 * v1 ** 2,
                         v2, np.sign(v2)])

    def inv_dynamics(self, q: np.ndarray, v: np.ndarray, a: np.ndarray,
                     add_contact: bool = True) -> np.ndarray:
        """Plant torque for state (q, v) accelerating at a (continuous law)."""
        m2e, lc2e, I2e = self.m2e, self.lc2e, self.I2e
        m1, l1, lc1, I1 = self.m1, self.l1, self.lc1, self.I1
        q1, q2 = q
        v1, v2 = v
        s2, c2 = np.sin(q2), np.cos(q2)
        c12 = np.cos(q1 + q2)
        A0 = I1 + I2e + m1 * lc1 ** 2 + m2e * (l1 ** 2 + lc2e ** 2)
        B0 = I2e + m2e * lc2e ** 2
        B2 = m2e * l1 * lc2e
        M11 = A0 + 2 * B2 * c2
        M12 = B0 + B2 * c2
        M22 = B0
        C1 = -B2 * s2 * (2 * v1 * v2 + v2 ** 2)
        C2 = B2 * s2 * v1 ** 2
        g1 = (m1 * G * lc1 + m2e * G * l1) * np.cos(q1) + m2e * G * lc2e * c12
        g2 = m2e * G * lc2e * c12
        tau = np.array([M11 * a[0] + M12 * a[1] + C1 + g1 + self.b1 * v1
                        + self.fc1 * np.sign(v1),
                        M12 * a[0] + M22 * a[1] + C2 + g2 + self.b2 * v2
                        + self.fc2 * np.sign(v2)])
        if add_contact:
            tau = tau + self.contact_torque(q, v)
        return tau

    def inertia(self, q: np.ndarray) -> np.ndarray:
        q2 = q[1]
        c = self.law_constants()
        B2 = c["B2"]
        c2 = np.cos(q2)
        return np.array([[c["A0"] + 2 * B2 * c2, c["B0"] + B2 * c2],
                         [c["B0"] + B2 * c2, c["B0"]]])

    def _bias(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """C + g + friction (everything except inertia)."""
        q1, q2 = q
        v1, v2 = v
        c = self.law_constants()
        s2, c2 = np.sin(q2), np.cos(q2)
        c12 = np.cos(q1 + q2)
        B2 = c["B2"]
        C1 = -B2 * s2 * (2 * v1 * v2 + v2 ** 2)
        C2 = B2 * s2 * v1 ** 2
        g1 = c["G1"] * np.cos(q1) + c["G2"] * c12
        g2 = c["G2"] * c12
        return np.array([C1 + g1 + self.b1 * v1 + self.fc1 * np.sign(v1),
                         C2 + g2 + self.b2 * v2 + self.fc2 * np.sign(v2)])

    def forward(self, q: np.ndarray, v: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Acceleration from the plant:  a = M^-1 (tau - bias - tau_c)."""
        tau = tau - self._bias(q, v) - self.contact_torque(q, v)
        return np.linalg.solve(self.inertia(q), tau)

    # ------------------------------------------------------------- contact
    def tip(self, q: np.ndarray) -> np.ndarray:
        q1, q2 = q
        return np.array([self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2),
                         self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q1, q2 = q
        s1, s12 = np.sin(q1), np.sin(q1 + q2)
        c1, c12 = np.cos(q1), np.cos(q1 + q2)
        return np.array([[-self.l1 * s1 - self.l2 * s12, -self.l2 * s12],
                         [self.l1 * c1 + self.l2 * c12,  self.l2 * c12]])

    def contact_torque(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.contact is None:
            return np.zeros(2)
        sx, sy, rs, kc, cc = (self.contact["sx"], self.contact["sy"],
                              self.contact["rs"], self.contact["kc"],
                              self.contact["cc"])
        tip = self.tip(q)
        dx, dy = tip[0] - sx, tip[1] - sy
        r = np.hypot(dx, dy)
        if r >= rs or r < 1e-9:
            return np.zeros(2)
        vt = self.jacobian(q) @ v
        F = np.array([kc * (rs - r) * dx / r - cc * vt[0],
                      kc * (rs - r) * dy / r - cc * vt[1]])
        return self.jacobian(q).T @ F

    # ------------------------------------------------------------- logging
    def run(self, ctrl, T: int, noise_frac: float = 0.05, seed: int = 0,
            q0: np.ndarray | None = None, v0: np.ndarray | None = None,
            contact_segs: list | None = None):
        """Close the loop with controller `ctrl(q, v, t) -> tau` and log the
        noisy observable streams (q, q_dot, tau), n = T steps at self.dt.

        ctrl may be a list of (t_until, fn) segments; contact_segs is a
        list of (t_until, contact_cfg | None) segments that switches the
        socket mid-trajectory (e.g. insertion after identification)."""
        rng = np.random.default_rng(seed)
        dt = self.dt
        q = np.array(q0 if q0 is not None else [0.6, -0.8])
        v = np.array(v0 if v0 is not None else [0.0, 0.0])
        segs = ctrl if isinstance(ctrl, list) else [(np.inf, ctrl)]
        csegs = contact_segs or [(np.inf, self.contact)]
        seg_idx = c_idx = 0
        Q = np.zeros((T, 2)); V = np.zeros((T, 2)); TAU = np.zeros((T, 2))
        for t in range(T):
            tt = t * dt
            while seg_idx < len(segs) - 1 and tt >= segs[seg_idx][0]:
                seg_idx += 1
            while c_idx < len(csegs) - 1 and tt >= csegs[c_idx][0]:
                c_idx += 1
                self.contact = csegs[c_idx][1]
            fn = segs[seg_idx][1]
            tau = fn(q, v, tt)
            tau = np.clip(tau, -60.0, 60.0)
            a = self.forward(q, v, tau)
            v = v + a * dt
            q = q + v * dt
            Q[t] = q; V[t] = v; TAU[t] = tau
        # observational noise: fraction of each channel's RMS
        noise = {}
        for name, arr in (("q", Q), ("v", V), ("tau", TAU)):
            sig = noise_frac * arr.std(axis=0) + 1e-9
            noise[name] = rng.normal(0.0, 1.0, arr.shape) * sig
        return dict(q=Q + noise["q"], v=V + noise["v"], tau=TAU + noise["tau"],
                    q_clean=Q, v_clean=V, tau_clean=TAU, dt=dt)


# ---------------------------------------------------------------- references
def make_reference(freqs=((1.7, 0.53), (2.3, 0.71)), amps=((0.9, 0.25), (1.0, 0.3)),
                   phases=((0.0, 1.2), (0.8, 0.4))):
    """Analytic reference trajectory (position, velocity, acceleration)."""
    def ref(t):
        q = np.array([amps[j][0] * np.sin(freqs[j][0] * t + phases[j][0])
                      + amps[j][1] * np.sin(freqs[j][1] * t + phases[j][1])
                      for j in range(2)])
        v = np.array([amps[j][0] * freqs[j][0] * np.cos(freqs[j][0] * t + phases[j][0])
                      + amps[j][1] * freqs[j][1] * np.cos(freqs[j][1] * t + phases[j][1])
                      for j in range(2)])
        a = np.array([-amps[j][0] * freqs[j][0] ** 2 * np.sin(freqs[j][0] * t + phases[j][0])
                      - amps[j][1] * freqs[j][1] ** 2 * np.sin(freqs[j][1] * t + phases[j][1])
                      for j in range(2)])
        return q, v, a
    return ref


def pd_ctrl(arm: Arm2DOF, kp=150.0, kd=20.0, ff_model: Arm2DOF | None = None,
            ref=None):
    """PD + computed-torque feedforward from `ff_model` (None = pure PD)."""
    if ref is None:
        ref = make_reference()
    def ctrl(q, v, t):
        qd, vd, ad = ref(t)
        tau_ff = np.zeros(2)
        if ff_model is not None:
            tau_ff = ff_model.inv_dynamics(qd, vd, ad, add_contact=False)
        return tau_ff + kp * (qd - q) + kd * (vd - v)
    return ctrl
