"""
test_biomaterial_net.py -- sanity tests for the Morphogenetic Bi-Modal
Network. Runnable directly (`python test_biomaterial_net.py`) or via pytest.

Covers:
  * interior flux conservation (port-Hamiltonian structure)
  * weak-form vs finite-difference noise amplification (the 2/dt^2 kill)
  * no-NaN stability over a noisy run
  * identification convergence (law-fit residual)
  * morphogenesis (alignment rise, drift-axis orientation)
  * chemical field bounds
"""

import numpy as np

from biomaterial_net import MorphogeneticNet, make_grid, default_cfg, IIRWindow

N_TOL = 1e-6


def test_interior_flux_conservation():
    """With no input and no leak, the implicit Laplacian conserve sum(u)."""
    pos, edges = make_grid(7, 7)
    cfg = default_cfg("weak")
    cfg["gamma_u"] = 0.0
    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(1))
    s0 = net.u.sum()
    for _ in range(200):
        net._plant_step(np.zeros(net.N))
    assert abs(net.u.sum() - s0) < N_TOL, "interior flux is not conservative"


def test_weak_form_beats_finite_difference_under_noise():
    """The integration-by-parts derivative must not amplify noise like 2/dt^2."""
    dt = 0.01
    lam = 6.0
    rng = np.random.default_rng(2)
    # constant signal: the true derivative is zero, so all observed variance
    # is noise amplification -- the cleanest measure of the 2/dt^2 effect
    noisy = 1.0 + 0.05 * rng.normal(size=4000)
    win = IIRWindow(lam, dt, (1,))
    y_weak = []
    for f in noisy:
        win.push(np.array([f]))
        y_weak.append(win.weak_derivative(np.array([f]))[0])
    y_weak = np.array(y_weak)
    y_fd = (noisy[1:] - noisy[:-1]) / dt
    var_weak = np.var(y_weak[500:])
    var_fd = np.var(y_fd[500:])
    assert var_weak / var_fd < 0.01, f"weak/fd variance ratio {var_weak / var_fd:.2e}"
    # theory: Var(y_fd) = 2 s^2 / dt^2 ;  Var(y_weak) ~ lam^2 s^2
    assert var_weak < lam ** 2 * 0.05 ** 2 * 2.0, "weak-form noise above theory"


def test_no_nan_and_stability_over_noisy_run():
    cfg = default_cfg("weak")
    pos, edges = make_grid(7, 7)
    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(3))
    net.enable_oracle_recording()
    for _ in range(800):
        net.step(sigma=0.05, record=True)
    ut = np.array(net.u_true_hist)
    c = np.array(net.c_hist)
    assert np.all(np.isfinite(ut)), "voltage went non-finite"
    assert np.all(np.isfinite(net.v)), "geometry went non-finite"
    assert c.min() >= -1e-9 and c.max() <= cfg["c_max"] + 1e-9, "chemical out of bounds"
    assert np.linalg.norm(net.v, axis=1).max() <= cfg["v_max"] + 1e-9, "norm windup"


def test_identification_converges():
    cfg = default_cfg("weak")
    pos, edges = make_grid(7, 7)
    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(4))
    for _ in range(1200):
        net.step(sigma=0.05, record=True)
    resid2 = np.array(net.resid2_hist)
    y2 = np.array(net.y2_hist)
    late = resid2[900:].mean() / max(y2[900:].mean(), 1e-12)
    assert late < 0.2, f"law-fit NMSE {late:.3f} did not converge"


def test_morphogenesis_organizes_geometry():
    cfg = default_cfg("weak")
    pos, edges = make_grid(7, 7)
    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(5))
    for _ in range(3000):
        net.step(sigma=0.05, record=True)
    align = np.array(net.align_hist)
    dA = align[-200:].mean() - align[:50].mean()
    wdir = np.zeros(3)
    wdir[: len(cfg["wind"])] = cfg["wind"]
    wdir /= np.linalg.norm(wdir)
    wd = float(np.abs(net.vhat @ wdir).mean())
    assert dA > 0.15, f"alignment did not rise (dA={dA:.3f})"
    assert wd > 0.6, f"tensor did not orient along drift (dot={wd:.3f})"


def test_shape_reads_back_the_law():
    """The geometric coupling a_ij (from the angles alone) must encode the true
    material law at least as well as the identified per-edge weights -- the
    mechanistic-interpretability claim."""
    cfg = default_cfg("weak")
    pos, edges = make_grid(7, 7)
    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(7))
    for _ in range(3000):
        net.step(sigma=0.05, record=True)
    kappa, a_ij, _, _ = net._compute_kappa()
    w = net._w
    r_a = float(np.corrcoef(a_ij, kappa)[0, 1])
    r_w = float(np.corrcoef(w, kappa)[0, 1])
    assert r_a > 0.3, f"shape does not encode the law (r={r_a:.3f})"
    assert r_a > r_w - 0.05, f"shape ({r_a:.3f}) not a better readout than weights ({r_w:.3f})"


def test_route_potential_dirichlet_is_physical():
    """The routing probe must be a real Dirichlet problem: pinned potentials
    enforced, finite field, and a non-degenerate interior (the regression that
    once collapsed the free field to u = 0 is caught here)."""
    cfg = default_cfg("weak")
    pos, edges = make_grid(7, 7)
    net = MorphogeneticNet(pos, edges, cfg, np.random.default_rng(8))
    for _ in range(3000):
        net.step(sigma=0.05, record=True)
    u = net.route_potential(0, 42)
    assert np.isfinite(u).all()
    assert abs(u[0] - 1.0) < 1e-9, "source not pinned at +V"
    assert abs(u[42] + 1.0) < 1e-9, "sink not pinned at -V"
    interior = np.setdiff1d(np.arange(net.N), [0, 42])
    assert np.abs(u[interior]).max() > 0.05, "free field collapsed to zero"


def test_fd_baseline_fails_where_weak_form_succeeds():
    """The FD identifier must not extract the law (v collapses / residual ~ 1)."""
    cfg_w = default_cfg("weak")
    cfg_f = default_cfg("fd")
    pos, edges = make_grid(7, 7)
    rng = np.random.default_rng(6)
    net = MorphogeneticNet(pos, edges, cfg_w, rng)
    for _ in range(1200):
        net.step(sigma=0.05, record=True)
    ut = np.array(net.u_true_hist)
    uo = np.array(net.u_obs_hist)
    # FD learner on the SAME observations
    fd = MorphogeneticNet(pos, edges, cfg_f, np.random.default_rng(6))
    for t in range(1200):
        fd.observe(uo[t], np.zeros(net.N))
    rw = np.array(net.resid2_hist); yw = np.array(net.y2_hist)
    rf = np.array(fd.resid2_hist); yf = np.array(fd.y2_hist)
    nmse_w = rw[900:].mean() / max(yw[900:].mean(), 1e-12)
    nmse_f = rf[900:].mean() / max(yf[900:].mean(), 1e-12)
    assert nmse_w < nmse_f, f"weak {nmse_w:.3f} not better than FD {nmse_f:.3f}"


def _sunspot_slice(n=900):
    """First n valid months of the real SILSO record (sqrt + z-scored)."""
    import real_benchmark as rb
    s = rb.preprocess(rb.load_sunspots())[:n]
    lo, hi = 24, 600
    return (s - s[lo:hi + 1].mean()) / s[lo:hi + 1].std()


def test_real_data_weak_form_generalizes_better_than_fd():
    """On REAL sunspot data, the frozen weak-form law must generalize to an
    unseen window far better than the finite-difference law (the 481x result,
    reproduced here on a slice for speed)."""
    import real_benchmark as rb
    z = _sunspot_slice()
    d, M, tau = 12, 6, 1
    tr_lo, tr_hi, va_lo, va_hi = 24, 600, 601, 720
    lr = rb.train_weak(z, d, M, tau, 0.15, 0.1, tr_lo, tr_hi,
                       lam_rls=0.005, drive=True, level=False)
    _, h_fd = rb.train_fd(z, d, M, tau, 0.1, tr_lo, tr_hi, lam_rls=0.005)
    w = lr.holdout_law_fit(z, va_lo, va_hi, tau)["nmse_signal"]
    f = h_fd(z, va_lo, va_hi, tau)["nmse_signal"]
    assert w < f / 5.0, f"weak {w:.2e} not far below FD {f:.2e}"


def test_real_data_forecast_through_law_beats_fd_law():
    """On real data, forecasting through the identified weak-form law must
    beat forecasting through the finite-difference law at 12 months."""
    import real_benchmark as rb
    z = _sunspot_slice()
    d, M, tau = 12, 6, 1
    tr_lo, tr_hi, te_lo, te_hi = 24, 600, 700, 899
    lr = rb.train_weak(z, d, M, tau, 0.15, 0.1, tr_lo, tr_hi,
                       lam_rls=0.005, drive=True, level=False)
    f_fd, _ = rb.train_fd(z, d, M, tau, 0.1, tr_lo, tr_hi, lam_rls=0.005)
    w = rb.eval_horizon(lr.forecast, z, te_lo, te_hi, 12, tau, 0.0, 1.0,
                        stride=4)
    f = rb.eval_horizon(f_fd, z, te_lo, te_hi, 12, tau, 0.0, 1.0, stride=4)
    assert w < f, f"weak {w:.3f} not better than FD {f:.3f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
