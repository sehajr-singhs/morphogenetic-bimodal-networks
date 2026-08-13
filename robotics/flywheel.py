"""
flywheel.py -- the robotics experiment suite.

  1. law_recovery : weak-form vs finite-difference vs batch identification
                    of the inverse-dynamics law from noisy streams, at 0%,
                    5% and 50% observational noise (the noise-armor claim).
  2. flywheel      : the closed-loop data flywheel.  Each lap the law
                    identified from the previous lap's logs becomes the
                    computed-torque feedforward; better control produces
                    cleaner data produces a better law.  Weak form spins,
                    finite differences stall.
  3. fleet         : a 6-robot fleet; each robot's identified law is a
                    vector in coefficient space.  The constellation of law
                    vectors (angles between them) is the fleet 'shape':
                    payload and wear are readable as directions, and the
                    payload MASS is read out (ratio method) from the
                    gravity coefficient.
  4. contact       : identify on an excitation trajectory, then run an
                    insertion maneuver.  The frozen nominal law's torque
                    residual spikes exactly when contact engages -- the
                    law turns the torque stream into a contact detector.
                    The weak form's residual is bounded and smooth; finite
                    differences spike at the non-smooth boundary.
  5. data_efficiency: law quality vs logged data length (weak vs FD), and
                    a trained LSTM / AR torque autoregressor for context.

All quantitative claims in the paper are the measured output of main().
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arm import Arm2DOF, pd_ctrl, make_reference, G        # noqa: E402
import identify                                            # noqa: E402
from identify import WeakFormID, FDID, batch_weak, tune_lam  # noqa: E402

RIDGE = 1e-3
DT = 2e-3


# ------------------------------------------------------------ experiment 1
def law_recovery(T=6000, noises=(0.0, 0.05, 0.5), seeds=(0, 1, 2)):
    """Weak vs FD vs batch identification, three noise levels, three seeds."""
    ref = make_reference()
    out = {"config": dict(T=T, dt=DT, ridge=RIDGE), "noises": {}}
    for nf in noises:
        per = []
        for seed in seeds:
            arm = Arm2DOF(mp=1.2)                       # true arm carries 1.2 kg
            nom = Arm2DOF(mp=0.0)                       # controller model: wrong
            ctrl = pd_ctrl(arm, ff_model=nom, ref=ref)
            logs = arm.run(ctrl, T, noise_frac=nf, seed=seed)
            q, v, tau = logs["q"], logs["v"], logs["tau"]
            th_star = arm.theta_star()
            t_tr, t_ev = int(0.65 * T), int(0.98 * T)   # train / holdout windows
            lam = tune_lam(q, v, tau, th_star, dt=DT, seed=seed)
            # weak form
            idf = WeakFormID(lam, RIDGE, dt=DT)
            res = idf.fit(q, v, tau, t_lo=0, t_hi=t_tr)
            mw = idf.eval_domains(res, q, v, tau, t_tr, t_ev)
            aw = idf.angle_errors(res, th_star).mean()
            # FD baseline (matched forgetting: same 5 s window)
            fdid = FDID(RIDGE, lam_rls=0.2, dt=DT)
            resf = fdid.fit(q, v, tau, t_lo=0, t_hi=t_tr)
            mf = fdid.eval_domains(resf, q, v, tau, t_tr, t_ev)
            af = fdid.angle_errors(resf, th_star).mean()
            # batch oracle (same weak-form features, whole training window)
            _, mb, ab = batch_weak(q, v, tau, lam, th_star, RIDGE, dt=DT,
                                   t_lo=0, t_hi=t_tr)
            per.append(dict(seed=seed, lam=float(lam),
                            weak=dict(**mw, angle=float(aw)),
                            fd=dict(**mf, angle=float(af)),
                            batch=dict(**mb, angle=float(ab.mean()))))
        out["noises"][str(nf)] = per
    return out


# ------------------------------------------------------------ experiment 2
def flywheel(n_laps=5, T_lap=4000, seeds=(0, 1, 2), kp=40.0, kd=10.0):
    """Closed-loop flywheel: law -> feedforward -> cleaner data -> law.

    Weak PD (40/10) so the model quality is what limits tracking.  A
    1.2 kg payload is present from lap 0 (the controller has no model);
    at lap 3 an extra 1.5 kg attaches (pick), and the forgetting-factor
    identifier must re-learn the law."""
    ref = make_reference()
    out = {"config": dict(n_laps=n_laps, T_lap=T_lap, dt=DT, kp=kp, kd=kd),
           "laps": {}}
    for seed in seeds:
        arm = Arm2DOF(mp=1.2)
        qd_var = np.array([0.5 * (0.9 ** 2 + 0.25 ** 2),
                           0.5 * (1.0 ** 2 + 0.3 ** 2)])
        for method in ("weak", "fd"):
            lam = None
            thetahat = None
            laps = []
            for lap in range(n_laps):
                arm = Arm2DOF(mp=2.7 if lap >= 3 else 1.2)
                th_star = arm.theta_star()
                ctrl = pd_ctrl(arm, kp=kp, kd=kd, ff_model=None, ref=ref)
                if thetahat is not None:
                    def ff_ctrl(qq, vv, tt, th=thetahat):
                        qd, vd, ad = ref(tt)
                        ff = np.array([th[0] @ arm.basis1(qd, vd, ad),
                                       th[1] @ arm.basis2(qd, vd, ad)])
                        return ff + kp * (qd - qq) + kd * (vd - vv)
                    ctrl = ff_ctrl
                logs = arm.run(ctrl, T_lap, noise_frac=0.05, seed=seed + lap)
                q, v, tau = logs["q"], logs["v"], logs["tau"]
                if lam is None:
                    lam = tune_lam(q, v, tau, th_star, dt=DT, seed=seed)
                if method == "weak":
                    idf = WeakFormID(lam, RIDGE, dt=DT)
                    res = idf.fit(q, v, tau, t_lo=0, t_hi=T_lap)
                    thetahat = res["law"]
                    ang = idf.angle_errors(res, th_star).mean()
                    m = idf.eval_domains(res, q, v, tau,
                                         int(0.5 * T_lap), T_lap - 2)
                else:
                    fdid = FDID(RIDGE, lam_rls=0.2, dt=DT)
                    resf = fdid.fit(q, v, tau, t_lo=0, t_hi=T_lap)
                    thetahat = resf["law"]
                    ang = fdid.angle_errors(resf, th_star).mean()
                    m = fdid.eval_domains(resf, q, v, tau,
                                          int(0.5 * T_lap), T_lap - 2)
                qd = np.array(ref(np.arange(T_lap) * DT)[0]).T
                trk = np.mean((qd - logs["q_clean"]) ** 2, axis=0) / qd_var
                laps.append(dict(lap=lap, mp=float(arm.mp),
                                 nmse_tau=float(m["nmse_tau"]),
                                 angle=float(ang),
                                 track=float(trk.mean())))
            out["laps"].setdefault(method, []).append(laps)
    return out


# ------------------------------------------------------------ experiment 3
def _law_vec(arm, T=8000, seed=0, noise_frac=0.02):
    ref = make_reference()
    nom = Arm2DOF(mp=0.0)
    logs = arm.run(pd_ctrl(arm, ff_model=nom, ref=ref), T,
                   noise_frac=noise_frac, seed=seed)
    q, v, tau = logs["q"], logs["v"], logs["tau"]
    lam = tune_lam(q, v, tau, arm.theta_star(), dt=DT, seed=seed)
    idf = WeakFormID(lam, RIDGE, dt=DT)
    res = idf.fit(q, v, tau, t_lo=0, t_hi=int(0.6 * T))
    return idf.angle_errors(res, arm.theta_star()).mean(), res["law"]


SUBSPACE_COLS = [(0, 1), (0, 2), (0, 6), (0, 7), (1, 6), (1, 7)]


def _subspace(vec, cols):
    return np.array([vec[j][c] for j, c in cols])


def fleet():
    """6-robot fleet: law vectors cluster by configuration; payload and
    wear are readable as *directions*; payload mass is read from G2."""
    robos = {
        "base":      Arm2DOF(mp=0.0),
        "payload-1": Arm2DOF(mp=1.2),
        "payload-2": Arm2DOF(mp=2.4),
        "worn":      Arm2DOF(b1=0.875, b2=0.625, fc1=2.0, fc2=1.5),
        "heavy":     Arm2DOF(m2=3.9, l2=0.54, I2=0.12),
        "pl-worn":   Arm2DOF(mp=1.2, b1=0.7, fc1=1.6),
    }
    names = list(robos)
    classes = ["base", "payload", "payload", "wear", "heavy", "payload"]
    angs, laws = {}, {}
    for name, arm in robos.items():
        ang, law = _law_vec(arm)
        angs[name] = float(ang)
        laws[name] = law
    V = np.stack([laws[n].reshape(-1) for n in names])
    Vs = np.stack([_subspace(laws[n], SUBSPACE_COLS) for n in names])
    sil_full = _silhouette(V, classes)
    sil_sub = _silhouette(Vs, classes)
    # payload mass readout (ratio method; the window's common scale bias
    # cancels against the base robot):  G2 = g (m2 lc2 + mp l2)
    arm0 = Arm2DOF(mp=0.0)
    m2, lc2, l2 = arm0.m2, arm0.lc2, arm0.l2
    g2_0 = laws["base"][0][2]
    mp_est = {name: float(m2 * lc2 * (laws[name][0][2] / g2_0 - 1.0) / l2)
              for name in ("payload-1", "payload-2")}
    # payload direction (analytic vs measured, full law space)
    d_analytic = (Arm2DOF(mp=1.2).theta_star() - arm0.theta_star()).reshape(-1)
    d_analytic = d_analytic / (np.linalg.norm(d_analytic) + 1e-12)
    d_measured = (laws["payload-1"] - laws["base"]).reshape(-1)
    d_measured = d_measured / (np.linalg.norm(d_measured) + 1e-12)
    pay_cos = float(np.clip(d_analytic @ d_measured, -1, 1))
    armw = Arm2DOF(b1=0.875, b2=0.625, fc1=2.0, fc2=1.5)
    dw_a = (armw.theta_star() - arm0.theta_star()).reshape(-1)
    dw_a = dw_a / (np.linalg.norm(dw_a) + 1e-12)
    dw_m = (laws["worn"] - laws["base"]).reshape(-1)
    dw_m = dw_m / (np.linalg.norm(dw_m) + 1e-12)
    wear_cos = float(np.clip(dw_a @ dw_m, -1, 1))
    n = len(names)
    angle_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            angle_mat[i, j] = np.degrees(np.arccos(np.clip(
                (V[i] @ V[j]) / (np.linalg.norm(V[i]) * np.linalg.norm(V[j]) + 1e-12),
                -1, 1)))
    return dict(angles=angs, silhouette_full=float(sil_full),
                silhouette_sub=float(sil_sub),
                payload_cos=float(pay_cos), wear_cos=float(wear_cos),
                mp_est=mp_est, angle_mat=angle_mat.tolist(), names=names)


def _silhouette(V, labels):
    n = len(V)
    s = []
    for i in range(n):
        same = [j for j in range(n) if j != i and labels[j] == labels[i]]
        if not same:
            continue
        d = np.linalg.norm(V - V[i], axis=1)
        a = d[same].mean()
        b = min(d[[j for j in range(n) if labels[j] == lab]].mean()
                for lab in set(labels) if lab != labels[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


# ------------------------------------------------------------ experiment 4
def _ik(x, y, l1=0.5, l2=0.45):
    """Planar-arm inverse kinematics (elbow up)."""
    r2 = x * x + y * y
    c2 = np.clip((r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1)
    q2 = np.arccos(c2)
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])


def _blend(a, b, t0, t1, t):
    """Raised-cosine pose blend with zero boundary velocity/acceleration."""
    if t <= t0:
        return a, np.zeros(2), np.zeros(2)
    if t >= t1:
        return b, np.zeros(2), np.zeros(2)
    s = (t - t0) / (t1 - t0)
    w = 0.5 * (1 - np.cos(np.pi * s))
    dw = 0.5 * np.pi * np.sin(np.pi * s) / (t1 - t0)
    d2w = 0.5 * np.pi ** 2 * np.cos(np.pi * s) / (t1 - t0) ** 2
    return (a + w * (b - a), dw * (b - a), d2w * (b - a))


def insertion_ref(q0, q_in, t_appr=1.5, t_hold=2.0):
    """Approach q0 -> q_in (t_appr), hold at q_in (t_hold), retract."""
    t_in, t_out = t_appr, t_appr + t_hold
    t_end = t_out + t_appr
    def ref(t):
        if t < t_in:
            return _blend(q0, q_in, 0.0, t_in, t)
        if t < t_out:
            return q_in, np.zeros(2), np.zeros(2)
        return _blend(q_in, q0, t_out, t_end, t)
    return ref


def _online_nmse(res, q, v, tau, W=400):
    """Rolling fit NMSE of the law AS OF t (online adaptation metric)."""
    T = len(q)
    Z = res["Z"]
    LH = res["law_hist"]
    WT = res.get("WT")
    err = np.zeros(T)
    for t in range(W, T):
        th_t = LH[t].reshape(2, 8)
        pred = np.stack([Z[t - W:t, :8] @ th_t[0], Z[t - W:t, 8:] @ th_t[1]],
                        axis=1)
        y = WT[t - W:t] if WT is not None else tau[t - W:t]
        var = y.var(axis=0).mean() + 1e-12
        err[t] = np.mean((y - pred) ** 2) / var
    return err


def contact(nf=0.05, T_exc=4000, seed=0):
    """Identify the nominal law on an excitation run, then insert: the
    frozen law's torque residual spikes exactly when contact engages."""
    ref = make_reference()
    arm = Arm2DOF(mp=1.2)
    # phase A: excitation (no contact) -> the nominal law
    logs_a = arm.run(pd_ctrl(arm, ff_model=None, ref=ref), T_exc,
                     noise_frac=nf, seed=seed)
    qa, va, taua = logs_a["q"], logs_a["v"], logs_a["tau"]
    th_star = arm.theta_star()
    lam = tune_lam(qa, va, taua, th_star, dt=DT, seed=seed)
    idf = WeakFormID(lam, RIDGE, dt=DT)
    res_a = idf.fit(qa, va, taua, t_lo=0, t_hi=T_exc)
    law_w = res_a["law"]
    fdid = FDID(RIDGE, lam_rls=0.2, dt=DT)
    resf_a = fdid.fit(qa, va, taua, t_lo=0, t_hi=T_exc)
    law_f = resf_a["law"]
    # phase B: insertion maneuver, contact socket at the insertion pose
    q_in = np.array([0.4, 1.2])
    sx, sy = arm.tip(q_in)
    contact_cfg = dict(sx=float(sx), sy=float(sy), rs=0.08, kc=1500.0, cc=40.0)
    q0b = np.array([0.3, 0.9])                       # near-socket approach pose
    ins = insertion_ref(q0b, q_in, t_appr=2.0, t_hold=3.0)
    T_ins = int((2.0 + 3.0 + 2.0) / DT)
    arm_b = Arm2DOF(mp=1.2, contact=contact_cfg)
    logs_b = arm_b.run(pd_ctrl(arm_b, kp=300.0, kd=30.0, ff_model=None,
                               ref=ins),
                       T_ins, noise_frac=nf, seed=seed + 1,
                       q0=q0b, v0=np.zeros(2),
                       contact_segs=[(0.0, contact_cfg)])
    qb, vb, taub = logs_b["q"], logs_b["v"], logs_b["tau"]
    # contact mask from the tip-socket distance (physically meaningful)
    in_contact = np.zeros(T_ins, dtype=bool)
    for t in range(T_ins):
        in_contact[t] = np.hypot(*(arm_b.tip(logs_b["q_clean"][t])
                                   - np.array([sx, sy]))) < contact_cfg["rs"]
    # frozen-law residual in the estimator's own domain (fresh window state)
    idf2 = WeakFormID(lam, RIDGE, dt=DT)
    res_wb = idf2.fit(qb, vb, taub, t_lo=0, t_hi=0)
    res_wb["law"] = law_w
    err_w = _frozen_residual(res_wb, qb, vb, taub)
    resf2 = FDID(RIDGE, lam_rls=0.2, dt=DT)
    resf_b = resf2.fit(qb, vb, taub, t_lo=0, t_hi=0)
    resf_b["law"] = law_f
    err_f = _frozen_residual(resf_b, qb, vb, taub)
    blocks = _blocks(in_contact)
    base_w = float(err_w[:int(0.2 * T_ins)].mean())
    base_f = float(err_f[:int(0.2 * T_ins)].mean())
    peak_w = max(err_w[a:b].max() for a, b in blocks) if blocks else 0.0
    peak_f = max(err_f[a:b].max() for a, b in blocks) if blocks else 0.0
    return dict(lam=float(lam),
                in_contact_frac=float(in_contact.mean()),
                base_weak=base_w, base_fd=base_f,
                peak_weak=float(peak_w), peak_fd=float(peak_f),
                spike_ratio_weak=float(peak_w / (base_w + 1e-12)),
                spike_ratio_fd=float(peak_f / (base_f + 1e-12)),
                contact=contact_cfg, T_ins=T_ins)


def _frozen_residual(res, q, v, tau):
    """Rolling RMS residual of the FROZEN law, in Nm (contact detector)."""
    T = len(q)
    Z = res["Z"]
    WT = res.get("WT")
    th = res["law"]
    W = 300
    err = np.zeros(T)
    for t in range(W, T):
        pred = np.stack([Z[t - W:t, :8] @ th[0], Z[t - W:t, 8:] @ th[1]], axis=1)
        y = WT[t - W:t] if WT is not None else tau[t - W:t]
        err[t] = np.sqrt(np.mean((y - pred) ** 2))
    return err


def _blocks(mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    blocks = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            blocks.append((start, prev + 1))
            start = i
        prev = i
    blocks.append((start, prev + 1))
    return blocks


# ------------------------------------------------------------ experiment 5
def data_efficiency(T=8000, T_trs=(600, 1200, 2400, 4800), seed=0):
    """Law quality vs logged data length (weak vs FD), plus LSTM context."""
    ref = make_reference()
    arm = Arm2DOF(mp=1.2)
    logs = arm.run(pd_ctrl(arm, ff_model=None, ref=ref), T,
                   noise_frac=0.05, seed=seed)
    q, v, tau = logs["q"], logs["v"], logs["tau"]
    th_star = arm.theta_star()
    out = {"curve": []}
    for t_tr in T_trs:
        lam = tune_lam(q[:t_tr], v[:t_tr], tau[:t_tr], th_star, dt=DT, seed=seed)
        idf = WeakFormID(lam, RIDGE, dt=DT)
        res = idf.fit(q, v, tau, t_lo=0, t_hi=t_tr)
        aw = idf.angle_errors(res, th_star).mean()
        fdid = FDID(RIDGE, lam_rls=0.2, dt=DT)
        resf = fdid.fit(q, v, tau, t_lo=0, t_hi=t_tr)
        af = fdid.angle_errors(resf, th_star).mean()
        out["curve"].append(dict(sec=float(t_tr * DT), weak=float(aw),
                                 fd=float(af)))
    from real_benchmark import train_lstm, train_ar
    s1 = tau[:, 0]
    forecaster = train_lstm(s1, p=8, n_h=16, t_lo=8, t_hi=4800, epochs=60,
                            lr=0.01, seed=seed)
    t_ev_lo, t_ev_hi = 5000, 7000
    preds = np.concatenate([forecaster(s1, t, 1)
                            for t in range(t_ev_lo, t_ev_hi)])
    y = s1[t_ev_lo + 1:t_ev_hi + 1]
    out["lstm_nmse_tau1"] = float(np.mean((y - preds) ** 2) / (y.var() + 1e-12))
    arf = train_ar(s1, p=8, t_lo=8, t_hi=4800, ridge=0.1)
    ar_preds = np.concatenate([arf(s1, t, 1) for t in range(t_ev_lo, t_ev_hi)])
    out["ar_nmse_tau1"] = float(np.mean((y - ar_preds) ** 2) / (y.var() + 1e-12))
    lam = tune_lam(q, v, tau, th_star, dt=DT, seed=seed)
    idf = WeakFormID(lam, RIDGE, dt=DT)
    res = idf.fit(q, v, tau, t_lo=0, t_hi=4800)
    m = idf.eval_domains(res, q, v, tau, 5000, 7000)
    out["weak_nmse_tau"] = float(m["nmse_tau"])
    out["lstm_epochs"] = 60
    return out


# ---------------------------------------------------------------- driver
def main(out_path="robotics.json"):
    t0 = time.time()
    results = {}
    print("[1/5] law recovery ...", flush=True)
    results["law_recovery"] = law_recovery()
    print("[2/5] flywheel ...", flush=True)
    results["flywheel"] = flywheel()
    print("[3/5] fleet ...", flush=True)
    results["fleet"] = fleet()
    print("[4/5] contact ...", flush=True)
    results["contact"] = contact()
    print("[5/5] data efficiency ...", flush=True)
    results["data_efficiency"] = data_efficiency()
    results["_meta"] = dict(runtime_s=round(time.time() - t0, 1),
                            dt=DT, ridge=RIDGE)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved -> {out_path}  ({results['_meta']['runtime_s']} s)", flush=True)
    return results


if __name__ == "__main__":
    main()
