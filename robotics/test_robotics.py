"""
test_robotics.py -- sanity tests for the operations-data-flywheel project.
Runnable directly (`python test_robotics.py`) or via pytest.

Covers:
  * the momentum weak identity holds (integration by parts, no derivatives)
  * weak-form law recovery beats finite differences under noise
  * no NaN stability over a noisy run
  * the flywheel loop converges (tracking error drops)
  * payload mass is readable from the gravity-coefficient ratio
"""

import numpy as np

from arm import Arm2DOF, pd_ctrl, make_reference
from identify import WeakFormID, FDID, theta_star

T = 4000          # 8 s at 500 Hz -- fast enough for CI, long enough to identify
TRAIN = 2600      # fit window; the rest is the holdout


def _logs(noise_frac=0.05, mp=1.2, seed=0, T=T):
    arm = Arm2DOF(mp=mp)
    return arm, arm.run(pd_ctrl(arm, ff_model=None, ref=make_reference()),
                        T, noise_frac=noise_frac, seed=seed)


def test_weak_identity_holds_clean():
    """Windowed both sides, the momentum law must reproduce the windowed
    torque without ever differentiating a signal (O(lambda dt) residual)."""
    arm, logs = _logs(noise_frac=0.0)
    th = theta_star(arm)
    idf = WeakFormID(80.0, dt=2e-3)
    res = idf.fit(logs["q"], logs["v"], logs["tau"], t_lo=0, t_hi=TRAIN)
    Z, WT = res["Z"], res["WT"]
    pred = np.stack([Z[TRAIN:, :8] @ th[0], Z[TRAIN:, 8:] @ th[1]], axis=1)
    nmse = np.mean((WT[TRAIN:] - pred) ** 2) / (WT[TRAIN:].var() + 1e-12)
    assert nmse < 0.05, f"weak identity failed (NMSE {nmse:.3f})"


def test_weak_beats_fd_under_noise():
    """At 5% noise the weak form must recover the law direction far better
    than finite differences (the 2/dt^2 noise amplification kill)."""
    _, logs = _logs(noise_frac=0.05, seed=1)
    arm = Arm2DOF(mp=1.2)
    th = theta_star(arm)
    idf = WeakFormID(80.0, dt=2e-3)
    res = idf.fit(logs["q"], logs["v"], logs["tau"], t_lo=0, t_hi=TRAIN)
    ang_w = idf.angle_errors(res, th).mean()
    fd = FDID(dt=2e-3)
    resf = fd.fit(logs["q"], logs["v"], logs["tau"], t_lo=0, t_hi=TRAIN)
    ang_f = fd.angle_errors(resf, th).mean()
    assert ang_w < ang_f, f"weak {ang_w:.1f}deg not better than FD {ang_f:.1f}deg"
    assert ang_w < 15.0, f"weak law too far off ({ang_w:.1f}deg)"


def test_no_nan_over_noisy_run():
    _, logs = _logs(noise_frac=0.5, seed=2)
    idf = WeakFormID(160.0, dt=2e-3)
    res = idf.fit(logs["q"], logs["v"], logs["tau"], t_lo=0, t_hi=TRAIN)
    assert np.all(np.isfinite(res["law"])), "law went non-finite"
    assert np.all(np.isfinite(logs["q"])), "logs went non-finite"


def test_flywheel_converges():
    """Two laps of identify -> feedforward must drop tracking error below the
    blind baseline (the loop spins)."""
    ref = make_reference()
    kp, kd = 40.0, 10.0
    qd_var = np.array([0.5 * (0.9 ** 2 + 0.25 ** 2),
                       0.5 * (1.0 ** 2 + 0.3 ** 2)])
    def nmse(logs):
        qd = np.array([ref(t * 2e-3)[0] for t in range(T)])
        return float(np.mean((logs["q"] - qd) ** 2 / qd_var))
    # lap 0: blind (no feedforward)
    arm0 = Arm2DOF(mp=1.2)
    logs0 = arm0.run(pd_ctrl(arm0, kp=kp, kd=kd, ref=ref), T,
                     noise_frac=0.05, seed=3)
    track0 = nmse(logs0)
    # laps 1-2: identify on the last lap, feed the law back as feedforward
    track = track0
    law = None
    for lap in (1, 2):
        arm = Arm2DOF(mp=1.2)
        if law is not None:
            def ff_ctrl(qq, vv, tt, th=law):
                qd, vd, ad = ref(tt)
                ff = np.array([th[0] @ arm.basis1(qd, vd, ad),
                               th[1] @ arm.basis2(qd, vd, ad)])
                return ff + kp * (qd - qq) + kd * (vd - vv)
            ctrl = ff_ctrl
        else:
            ctrl = pd_ctrl(arm, kp=kp, kd=kd, ref=ref)
        logs = arm.run(ctrl, T, noise_frac=0.05, seed=3)
        idf = WeakFormID(80.0, dt=2e-3)
        res = idf.fit(logs["q"], logs["v"], logs["tau"], t_lo=0, t_hi=T)
        law = res["law"]
        track = nmse(logs)
    assert track < 0.45 * track0, \
        f"flywheel did not spin (track {track0:.4f} -> {track:.4f})"


def test_payload_readable_from_law():
    """Payload mass must be readable from the gravity-coefficient ratio to a
    base robot, even when the windowed coefficients carry a common bias:
    G2 = g (m2 lc2 + mp l2), so mp = m2 lc2 (G2/G2_base - 1) / l2."""
    def law_vec(mp, seed):
        arm = Arm2DOF(mp=mp)
        nom = Arm2DOF(mp=0.0)
        ref = make_reference()
        logs = arm.run(pd_ctrl(arm, ff_model=nom, ref=ref), 8000,
                       noise_frac=0.02, seed=seed)
        idf = WeakFormID(80.0, dt=2e-3)
        res = idf.fit(logs["q"], logs["v"], logs["tau"], t_lo=0, t_hi=8000)
        return res["law"]
    base = law_vec(0.0, 4)
    heavy = law_vec(2.4, 4)
    arm0 = Arm2DOF(mp=0.0)
    m2, lc2, l2 = arm0.m2, arm0.lc2, arm0.l2
    mp_est = m2 * lc2 * (heavy[0, 2] / base[0, 2] - 1.0) / l2
    assert 1.6 < mp_est < 3.3, f"payload readout off (est {mp_est:.2f} kg)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
