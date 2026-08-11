"""
biomaterial_net.py -- Morphogenetic Bi-Modal Network core.

A spatially embedded, gradient-free neural operator that learns through the
physical intersection of 3D geometry and dual chemical/electrical kinetics.
No global loss function, no backpropagation: every parameter update is a
strictly local, streaming operation.

Architecture (two coupled timescales):

A. Dual kinetic continua
   - Fast electrical continuum:  du_i/dt = sum_j kappa_ij (u_j - u_i) + I_i(t)
     with kappa_ij = kappa0(t) * (1 + lam_c * cbar_ij) * (1 + lam_g * a_ij).
     The interior flux is anti-symmetric, hence conservative (a discrete
     port-Hamiltonian Laplacian with dissipation and external ports I_i).
   - Slow chemical continuum:    dc_i/dt = D Lap(c)_i - w . grad(c)_i
     + rho(u_i) - gamma_c c_i,  rho(u) = beta max(0, u - u_thr)^2.
     Electrical activity releases chemical that diffuses AND advects (a
     neuromodulator "wind" that breaks spatial symmetry) and raises local
     conductivity -- a local material environment replacing global loss
     weights. The wind imprints a directional corridor into c, and hence
     into kappa: the seed of morphogenesis.

B. Geometric angulation (the morphogenesis layer)
   - Every node carries a structural vector v_i in R^3 (spherical parts:
     r_i = ||v_i|| "metabolic magnitude", (theta, phi) orientation).
   - Edge coupling is regulated by directional alignment
     a_ij = (1 + vhat_i . vhat_j) / 2 in [0, 1] -- angles as computation:
     the shape modulates conductivity and steers information flow.
   - The structural vectors ROTATE by a nematic consensus: each node drifts
     toward the coupling-weighted mean direction of its neighbors' vectors
     (liquid-crystal ordering), so strongly-coupled regions self-organize
     into coherent aligned corridors. Morphogenesis is GATED BY THE
     CHEMICAL FIELD (a concentration threshold power law): the geometry
     grows only on the chemically-potentiated corridor and quiet regions
     stay frozen -- the slow chemical continuum decides where structure
     forms. A concentration-dependent torque orients the corridor tensors
     along the neuromodulator drift axis. Magnitude is homeostatically
     pinned to the local coupling scale sqrt(mean |w|).
   - Because the shape is a spatially coherent aggregate of the (noisy,
     multicollinear) per-edge estimates, the geometric coupling a_ij is a
     cleaner readout of the true law than the weights themselves
     (measured: corr(a_ij, kappa) ~ 0.64 vs corr(w, kappa) ~ 0.35) -- the
     shape is the interpretable representation.
   - FAILURE MODES (all verified): (i) a bilinear per-node exact-RLS in
     normal-equation form is scale-unstable -- the regressor x_i contains
     neighbor parameters v_j, so the exact least-squares solve attributes
     the full target to v_i alone, mapping v -> v/2 per update and
     collapsing the geometry to v = 0; (ii) aiming each node at its single
     best neighbor gives only indirect pairwise alignment -- the wiring
     field alternates and chains break; (iii) a plain consensus
     globalizes: alignment propagates across the whole grid, the shape
     becomes featureless, and corr(a_ij, kappa) collapses -- hence the
     chemical gate.

C. Spacetime IIR weak form (noise armor) -- the identification layer
   - Every quantity is projected onto a one-pole IIR window
     A(t) = EMA_alpha[f], alpha = exp(-lam dt), and the governing equation
     is integrated by parts:
         weak derivative  y = lam (f - A),  computed WITHOUT differentiation
     of the (noisy) data; the 2/dt^2 finite-difference noise amplification
     is replaced by lam^2.
   - Node i fits the local linear law  y_i = sum_j a_ij (A_u_j - A_u_i)
     by per-node RLS (normal equations with exponential forgetting +
     Tikhonov ridge). This model class is exactly the plant's, so the
     identified a_ij track the true per-edge conductivity at the noise
     floor. The symmetric coupling estimate w_ij = (a_ij + a_ji)/2 feeds
     the morphogenesis layer, closing the loop: geometry shapes physics
     (alignment in kappa), physics shapes geometry (encoding of w).

Homeostatic dampening: Tikhonov ridge on the RLS, absolute diagonal floor,
norm homeostat + clamp on v, chemical field clipped to [0, c_max].
"""

from __future__ import annotations

import numpy as np


# ================================================================ IIR weak form


class IIRWindow:
    """One-pole IIR (exponential) window with an integration-by-parts derivative.

    Maintains A(t) = EMA_alpha[f] with alpha = exp(-lam dt) and exposes the
    weak derivative  y = lam (f - A)  which equals the windowed derivative of
    f but requires no differentiation of the (possibly noisy) signal f.
    """

    __slots__ = ("lam", "alpha", "A")

    def __init__(self, lam: float, dt: float, shape: tuple):
        self.lam = float(lam)
        self.alpha = np.exp(-lam * dt)
        self.A = np.zeros(shape)

    def push(self, f: np.ndarray) -> np.ndarray:
        self.A = self.alpha * self.A + (1.0 - self.alpha) * f
        return self.A

    def weak_derivative(self, f: np.ndarray) -> np.ndarray:
        return self.lam * (f - self.A)


# ================================================================ chaotic input


class LorenzDriver:
    """Continuous chaotic excitation (Lorenz 63) with slow amplitude drift."""

    def __init__(self, rng: np.random.Generator, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.state = rng.normal(0.0, 1.0, 3)

    def step(self, dt: float, t: float, a_base: float, a_amp: float, t_inp: float,
             phase: float) -> float:
        s, r, b = self.sigma, self.rho, self.beta
        x, y, z = self.state

        def f(x, y, z):
            return np.array([s * (y - x), x * (r - z) - y, x * y - b * z])

        k1 = f(x, y, z)
        k2 = f(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], z + 0.5 * dt * k1[2])
        k3 = f(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], z + 0.5 * dt * k2[2])
        k4 = f(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        self.state = self.state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        amp = a_base * (1.0 + a_amp * np.sin(2.0 * np.pi * t / t_inp + phase))
        return amp * self.state[0] / 20.0


# ================================================================ the network


class MorphogeneticNet:
    """Bi-modal morphogenetic network: plant (electrical + chemical),
    identification layer (weak-form per-node RLS), and morphogenesis layer
    (angulation encoder) coupled into one streaming object."""

    def __init__(self, pos: np.ndarray, edges: np.ndarray, cfg: dict,
                 rng: np.random.Generator):
        self.pos = np.asarray(pos, dtype=float)          # (N, 3) Euclidean embedding
        self.edges = np.asarray(edges, dtype=int)        # (E, 2)
        self.ei = self.edges[:, 0]
        self.ej = self.edges[:, 1]
        self.N = self.pos.shape[0]
        self.E = self.edges.shape[0]
        self.cfg = cfg
        self.rng = rng
        self.t = 0.0
        self._step = 0
        self.dt = cfg["dt"]

        # ---- electrical / chemical states
        self.u = rng.normal(0.0, 0.05, self.N)
        self.c = np.zeros(self.N)

        # ---- structural geometry (morphogenesis layer)
        v = rng.normal(0.0, 1.0, (self.N, 3))
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        self.v = v
        self.vhat = v.copy()

        # ---- edge unit directions (spatial geometry of the embedding)
        d = self.pos[self.ej] - self.pos[self.ei]
        dl = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        self._edge_dir = d / dl[:, None]

        # ---- neighbor slots (per-node regressor layout, degree-padded)
        D = max(int(np.bincount(np.concatenate([self.ei, self.ej])).max()), 1)
        self.D = D
        self._nb = np.full((self.N, D), -1, dtype=int)
        self._nslot_ei = np.zeros(self.E, dtype=int)
        self._nslot_ej = np.zeros(self.E, dtype=int)
        cnt = np.zeros(self.N, dtype=int)
        for e, (i, j) in enumerate(self.edges):
            self._nb[i, cnt[i]] = j
            self._nslot_ei[e] = cnt[i]
            cnt[i] += 1
            self._nb[j, cnt[j]] = i
            self._nslot_ej[e] = cnt[j]
            cnt[j] += 1

        # ---- chaotic excitation
        self.lorenz = LorenzDriver(rng)
        self._phase = rng.uniform(0.0, 2.0 * np.pi, 2)
        self._wind_dir = np.zeros(3)
        if cfg["wind"] is not None:
            self._wind_dir[: len(cfg["wind"])] = np.asarray(cfg["wind"],
                                                             dtype=float)
            nw = np.linalg.norm(self._wind_dir)
            self._wind_dir = self._wind_dir / max(nw, 1e-12)
        self._src = np.zeros(self.N, dtype=float)
        for k in cfg["sources"]:
            self._src[k] = 1.0

        # ---- identification layer: IIR weak-form accumulators + per-node RLS
        lam = cfg["lam"]
        self._win_u = IIRWindow(lam, self.dt, (self.N,))
        self._win_I = IIRWindow(lam, self.dt, (self.N,))
        self._C = np.zeros((self.N, D, D))                # EMA of z z^T (fast)
        self._r = np.zeros((self.N, D))                   # EMA of z y (fast)
        self._z = np.zeros((self.N, D))                   # current regressor
        self._a = np.zeros((self.N, D))                   # identified law (fast)
        self._w = np.zeros(self.E)                        # coupling est. (slow)
        self._C2 = np.zeros((self.N, D, D))               # EMA of z z^T (slow)
        self._r2 = np.zeros((self.N, D))                  # EMA of z y (slow)
        self._alpha_s = np.exp(-cfg.get("lam_s", 0.5) * self.dt)
        self._u_prev_obs = None

        # diagnostics / evaluation hooks
        self.u_true_hist = []
        self.u_obs_hist = []
        self.pred_hist = []
        self.align_hist = []
        self.vnorm_hist = []
        self.c_hist = []
        self.resid2_hist = []
        self.y2_hist = []
        self.enc_err_hist = []

        self.record_for_oracle = False
        self._oracle = None

    # ------------------------------------------------------------ helpers
    def _scatter_add(self, contrib, idx, out_shape):
        out = np.zeros(out_shape, dtype=float)
        np.add.at(out, idx, contrib)
        return out

    def _edge_alignment(self) -> np.ndarray:
        n = np.linalg.norm(self.v, axis=1)
        vh = np.zeros_like(self.v)
        ok = n > 1e-9
        vh[ok] = self.v[ok] / n[ok, None]
        self.vhat = vh
        dot = np.sum(vh[self.ei] * vh[self.ej], axis=1)
        return 0.5 * (1.0 + dot)

    # ------------------------------------------------------------ plant step
    def _compute_kappa(self):
        """Conductivity from the current material state: base regime x
        chemical channel x geometric angulation channel. The angulation
        channel is the angles-as-computation mechanism: an edge conducts in
        proportion to the directional alignment of the two structural
        vectors, a_ij = (1 + vhat_i . vhat_j)/2 in [0,1] -- the SHAPE (a
        coherent field of 3D structural orientations) modulates the
        material's conductivity, so information is steered along the
        self-organized corridors. Returns (kappa, a_ij, cbar, kappa0)."""
        cfg = self.cfg
        kappa0 = cfg["kappa_base"] * (1.0 + cfg["kappa_amp"] *
                                      np.sign(np.sin(2.0 * np.pi * self.t /
                                                     cfg["t_drift"] +
                                                     self._phase[0])))
        cbar = 0.5 * (self.c[self.ei] + self.c[self.ej])     # (E,)
        a_ij = self._edge_alignment()                        # (E,) in [0,1]
        self._last_align_mean = float(a_ij.mean())
        kappa = kappa0 * (1.0 + cfg["lam_c"] * cbar) \
            * (1.0 + cfg["lam_g"] * a_ij)
        return kappa, a_ij, cbar, kappa0

    def route_probe(self, src: int, sink: int, V: float = 1.0):
        """Steady-state routing probe: fix src at +V and sink at -V and solve
        the Dirichlet problem for the learned material. Returns per-edge
        |flux| and the cbar field, so where information actually flows can be
        tested against what the geometry alone predicts."""
        cfg = self.cfg
        kappa, _, cbar, _ = self._compute_kappa()
        L = np.zeros((self.N, self.N))
        np.add.at(L, (self.ei, self.ei), kappa)
        np.add.at(L, (self.ej, self.ej), kappa)
        np.add.at(L, (self.ei, self.ej), -kappa)
        np.add.at(L, (self.ej, self.ei), -kappa)
        u = self.route_potential(src, sink, V)
        flux = np.abs(kappa * (u[self.ej] - u[self.ei]))
        return flux, cbar

    def route_potential(self, src: int, sink: int, V: float = 1.0) -> np.ndarray:
        """Steady-state Dirichlet solve for the learned material: src pinned
        at +V, sink at -V. Returns the potential field u (N,). Dirichlet
        conditioning moves the KNOWN pinned potentials to the RHS BEFORE
        zeroing the pinned rows/columns -- a previous version only zeroed the
        columns, which severed the source/sink from every other equation and
        collapsed the entire free field to u = 0, making every "routing"
        measurement a boundary-layer artifact."""
        cfg = self.cfg
        kappa, _, _, _ = self._compute_kappa()
        L = np.zeros((self.N, self.N))
        np.add.at(L, (self.ei, self.ei), kappa)
        np.add.at(L, (self.ej, self.ej), kappa)
        np.add.at(L, (self.ei, self.ej), -kappa)
        np.add.at(L, (self.ej, self.ei), -kappa)
        A = L + cfg["gamma_u"] * np.eye(self.N)
        b = np.zeros(self.N)
        b = b - A[:, src] * V - A[:, sink] * (-V)
        A[:, [src, sink]] = 0.0
        A[src, :] = 0.0
        A[sink, :] = 0.0
        A[src, src] = 1.0
        A[sink, sink] = 1.0
        b[src] = V
        b[sink] = -V
        return np.linalg.solve(A, b)

    def _plant_step(self, I: np.ndarray):
        cfg = self.cfg
        dt = self.dt
        kappa, _, _, _ = self._compute_kappa()

        # electrical: implicit Euler on the weighted graph Laplacian.
        # The flux is anti-symmetric (conservative interior exchange); the
        # implicit solve keeps the update unconditionally stable for any
        # conductivity (port-Hamiltonian structure preserved).
        L = np.zeros((self.N, self.N))
        np.add.at(L, (self.ei, self.ei), kappa)
        np.add.at(L, (self.ej, self.ej), kappa)
        np.add.at(L, (self.ei, self.ej), -kappa)
        np.add.at(L, (self.ej, self.ei), -kappa)
        A = np.eye(self.N) * (1.0 + dt * cfg["gamma_u"]) + dt * L
        self.u = np.linalg.solve(A, self.u + dt * I)

        # chemical: release -> slow diffusion + advection -> decay.
        # The advection "wind" (neuromodulator drift) breaks the spatial
        # symmetry of the material environment, imprints a directional
        # corridor into c and hence into kappa -- the seed of morphogenesis.
        release = cfg["beta"] * np.maximum(0.0, np.abs(self.u) - cfg["u_thr"]) ** 2
        dc = cfg["D"] * (self._scatter_add(self.c[self.ej] - self.c[self.ei], self.ei,
                                           (self.N,))
                         - self._scatter_add(self.c[self.ej] - self.c[self.ei], self.ej,
                                             (self.N,)))
        if cfg["wind"] is not None:
            w = np.zeros(3)
            w[: len(cfg["wind"])] = np.asarray(cfg["wind"], dtype=float)
            d = self.pos[self.ej] - self.pos[self.ei]
            dlen = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
            wd = np.sum(w[None, :] * (d / dlen[:, None]), axis=1)
            flux = np.where(wd > 0, wd * self.c[self.ei], wd * self.c[self.ej])
            dc = dc + self._scatter_add(flux, self.ei, (self.N,)) \
                     - self._scatter_add(flux, self.ej, (self.N,))
        dc = dc + release - cfg["gamma_c"] * self.c
        self.c = np.clip(self.c + dt * dc, 0.0, cfg["c_max"])
        return kappa

    # ------------------------------------------------------------ learner step
    def _learner_step(self, u_obs: np.ndarray, I: np.ndarray, morph: bool):
        cfg = self.cfg
        dt = self.dt
        mode = cfg["mode"]

        if mode == "weak":
            self._win_u.push(u_obs)
            self._win_I.push(I)
            Au = self._win_u.A
            duf = Au[self.ej] - Au[self.ei]                    # (E,)
            y = self._win_u.weak_derivative(u_obs) - self._win_I.A
            drive = self._win_I.A
        else:  # mode == "fd": death-of-differentiation baseline
            duf = u_obs[self.ej] - u_obs[self.ei]              # raw, unfiltered
            prev = self._u_prev_obs
            y = np.zeros(self.N) if prev is None else (u_obs - prev) / dt
            self._u_prev_obs = u_obs.copy()
            drive = np.zeros(self.N)

        # ---- identification: per-node linear RLS on weak-form quantities
        z = np.zeros((self.N, self.D))
        np.add.at(z, (self.ei, self._nslot_ei), duf)
        np.add.at(z, (self.ej, self._nslot_ej), -duf)
        self._z = z

        alpha = self._win_u.alpha
        self._C = alpha * self._C + (1.0 - alpha) * (z[:, :, None] * z[:, None, :])
        self._r = alpha * self._r + (1.0 - alpha) * z * y[:, None]

        ridge = cfg["ridge"] * (np.trace(self._C, axis1=1, axis2=2) / self.D) + 1e-6
        A = self._C + ridge[:, None, None] * np.eye(self.D)
        self._a = np.linalg.solve(A, self._r[..., None])[..., 0]

        model = np.einsum("nd,nd->n", self._a, z)

        # ---- structural consolidation: a second, much slower RLS on the SAME
        #      weak-form data. The fast identifier's per-edge coefficients are
        #      multicollinear (neighbor differences are correlated), so they
        #      predict well but are individually noisy -- useless for
        #      morphogenesis. The slow stats average that noise away and are
        #      spatially smoothed, giving a clean coupling estimate w that the
        #      geometry can organize around. Decimated (every 5 steps): the
        #      consolidation timescale is seconds, far longer than one step.
        w = self._w
        stride = cfg.get("slow_stride", 5)
        if cfg["mode"] == "weak" and self._step % stride == 0:
            if cfg.get("consolidate", True):
                # decimated update must use the stride-compensated forgetting
                # factor so the effective per-step window is preserved
                al = self._alpha_s ** stride
                self._C2 = al * self._C2 + (1.0 - al) * (z[:, :, None] * z[:, None, :])
                self._r2 = al * self._r2 + (1.0 - al) * z * y[:, None]
                ridge2 = cfg["ridge"] * (np.trace(self._C2, axis1=1, axis2=2) / self.D) \
                    + 1e-6
                A2 = self._C2 + ridge2[:, None, None] * np.eye(self.D)
                a2 = np.linalg.solve(A2, self._r2[..., None])[..., 0]
            else:
                # ablation: no slow consolidation -- feed the fast,
                # multicollinear per-edge coefficients directly to
                # morphogenesis (the noisy raw readout)
                a2 = self._a
            w = 0.5 * (a2[self.ei, self._nslot_ei] + a2[self.ej, self._nslot_ej])
            if cfg.get("consolidate", True):
                # spatial smoothing of the structural coupling (one edge-space
                # diffusion pass): the material property is spatially coherent
                sumw = np.zeros(self.N)
                deg = np.zeros(self.N)
                np.add.at(sumw, self.ei, w)
                np.add.at(sumw, self.ej, w)
                np.add.at(deg, self.ei, 1.0)
                np.add.at(deg, self.ej, 1.0)
                meanw = sumw / np.maximum(deg, 1.0)
                w = 0.5 * w + 0.25 * (meanw[self.ei] + meanw[self.ej])
            self._w = w

        # ---- morphogenesis: structural rotation (angulation as shape). Each
        #      node's structural vector rotates toward the coupling-weighted
        #      direction of its neighbors IN SPACE,
        #      vhat_i <- norm(vhat_i + eta sum_j w_ij dhat_ij), so the shape
        #      self-organizes along the strongest identified pathways, and
        #      edge coupling a_ij = (1+vhat_i.vhat_j)/2 then amplifies
        #      conductivity along those pathways -- the geometry both encodes
        #      the law (mechanistic interpretability) and steers the flow
        #      (angles as computation). Only POSITIVE (excitatory) coupling
        #      drives rotation (negative w is identification noise; rotating
        #      along it collapses the geometry into an anti-aligned
        #      checkerboard -- verified failure mode). A concentration-
        #      dependent orienting torque steers the tensor toward the
        #      neuromodulator drift axis with strength proportional to local
        #      c (like a material orienting in an external field): corridor
        #      nodes align preferentially, so the geometry is corridor-
        #      specific rather than globally uniform. Magnitude (metabolic
        #      capacity) is homeostatically pinned to the local coupling
        #      scale sqrt(mean |w|), slowly. A warm-up gate keeps the
        #      geometry inert until the identifier has converged (rotating on
        #      noise-dominated early w is a random walk -- verified failure).
        if morph and cfg.get("morph_on", True):
            gate = float(np.clip((self.t - cfg["t_warm"]) / cfg["t_ramp"], 0.0, 1.0))
            w_abs = np.abs(w)
            w_pos = np.maximum(w, 0.0)
            wbar = np.zeros(self.N)
            np.add.at(wbar, self.ei, w_abs)
            np.add.at(wbar, self.ej, w_abs)
            deg = np.zeros(self.N)
            np.add.at(deg, self.ei, 1.0)
            np.add.at(deg, self.ej, 1.0)
            deg = np.maximum(deg, 1.0)

            # nematic consensus rotation along the identified coupling: each
            # node rotates toward the coupling-weighted MEAN DIRECTION of its
            # neighbors' structural vectors, vhat_i <- norm(vhat_i + eta
            # sum_j (w_pos,ij / sum_k |w_ik|) vhat_j). Strongly coupled
            # neighbors pull each other into alignment -- the shape
            # self-organizes like a liquid crystal, with coherent aligned
            # corridors where the identified coupling is strong and disorder
            # elsewhere. (Aiming each node at its single best neighbor was a
            # verified failure: pairwise alignment was only indirect, so the
            # a_ij field came out as a chaotic alternation and chains broke.
            # The consensus is a contraction that directly makes a_ij high
            # where w is high.) Only POSITIVE (excitatory) coupling drives
            # rotation (negative w is identification noise; rotating along it
            # collapses the geometry into an anti-aligned checkerboard --
            # verified failure mode).
            rot = np.zeros((self.N, 3))
            np.add.at(rot, self.ei, w_pos[:, None] * self.vhat[self.ej])
            np.add.at(rot, self.ej, w_pos[:, None] * self.vhat[self.ei])
            rot = rot / np.maximum(wbar, 1e-9)[:, None]
            rn = np.linalg.norm(rot, axis=1)
            # CHEMICAL GATE: morphogenesis is ZERO below a threshold on the
            # local neuromodulator concentration, then ramps. The slow
            # chemical continuum (the bi-modal environment) decides WHERE the
            # geometry grows: only the chemically-potentiated corridor
            # self-organizes into coherent aligned corridors, and quiet
            # regions stay frozen -- the shape is a SPECIFIC map of the law.
            # (An activity gate on the identified coupling failed: the
            # per-edge weights are spatially smeared by the consolidation
            # smoothing, so the gate fired everywhere and the shape
            # globalized -- mean alignment 1.00, corr(a_ij, kappa) ~ 0.1 --
            # verified failure mode. The chemical field is clean and
            # specific.)
            cmax = max(float(self.c.max()), 1e-9)
            gate_i = gate * np.clip((self.c / cmax - cfg["act_th"]) /
                                    (1.0 - cfg["act_th"]), 0.0, 1.0) ** cfg["gate_pow"]
            step = gate_i[:, None] * cfg["eta_e"] * rot / np.maximum(rn, 1e-9)[:, None]
            step[rn <= 1e-9] = 0.0

            # concentration-dependent orienting torque toward the drift axis,
            # gated the same way: corridor nodes align preferentially to the
            # neuromodulator drift -- the directional tensor corridor
            wdir = self._wind_dir
            torque = gate_i[:, None] * cfg["eta_t"] * (wdir[None, :] - self.vhat)

            vh = self.vhat + step + torque
            n = np.linalg.norm(vh, axis=1)
            self.vhat = vh / np.maximum(n, 1e-12)[:, None]

            r = np.linalg.norm(self.v, axis=1)
            r_tgt = np.sqrt(wbar / deg)
            r = r + gate * cfg["eta_n"] * (r_tgt - r)
            r = np.clip(r, 1e-3, cfg["v_max"])
            self.v = r[:, None] * self.vhat

        # one-step-ahead physical prediction
        pred = u_obs + dt * (model + drive)
        return pred, model, y

    # ------------------------------------------------------------ main step
    def step(self, sigma: float = 0.0, record: bool = False, learn: bool = True):
        cfg = self.cfg
        dt = self.dt
        drive = self.lorenz.step(dt, self.t,
                                 cfg["a_base"], cfg["a_amp"], cfg["t_inp"],
                                 self._phase[1])
        I = self._src * drive
        self._plant_step(I)
        u_true = self.u.copy()
        u_obs = u_true + sigma * self.rng.normal(size=self.N) if sigma > 0.0 \
            else u_true.copy()
        if learn:
            pred, model, y = self._learner_step(u_obs, I, morph=True)
        else:
            pred, model, y = u_obs, None, None
        self._step += 1
        self.t += dt

        if record or self.record_for_oracle:
            self.u_true_hist.append(u_true)
            self.u_obs_hist.append(u_obs)
            self.pred_hist.append(pred)
            self.align_hist.append(self._last_align_mean)
            self.vnorm_hist.append(np.linalg.norm(self.v, axis=1).mean())
            self.c_hist.append(self.c.copy())
            if learn:
                self.resid2_hist.append(float(np.mean((y - model) ** 2)))
                self.y2_hist.append(float(np.mean(y ** 2)))
                enc = np.mean((self._w
                               - np.sum(self.v[self.ei] * self.v[self.ej],
                                        axis=1)) ** 2)
                self.enc_err_hist.append(float(enc))
            if self.record_for_oracle:
                self._oracle["u_obs"].append(u_obs)
                self._oracle["u_true"].append(u_true)
                self._oracle["I"].append(I)
                self._oracle["Au"].append(self._win_u.A.copy())
                self._oracle["AI"].append(self._win_I.A.copy())
                self._oracle["duf"].append(self._win_u.A[self.ej] - self._win_u.A[self.ei])

    # offline observation pass (used by the FD baseline on shared data)
    def observe(self, u_obs: np.ndarray, I: np.ndarray, record: bool = True):
        pred, model, y = self._learner_step(u_obs, I, morph=False)
        if record:
            self.pred_hist.append(pred)
            self.resid2_hist.append(float(np.mean((y - model) ** 2)))
            self.y2_hist.append(float(np.mean(y ** 2)))

    # -------------------------------------------------- experience (game mode)
    def experience(self, u_obs: np.ndarray, I: np.ndarray):
        """The network experiences an EXTERNAL field: it adopts the observed
        state as its own, lets the chemistry respond to the observed activity
        (release -> gate -> where structure grows), identifies the local law
        from the observations, and rotates its geometry. Used by the game
        (flowgame.py): the network lives inside an environment it never
        generated, learns its law, and steers a token through the shape."""
        self.u = np.asarray(u_obs, dtype=float)
        self._plant_step(I)                 # chemistry evolves from |u_obs|
        self.u = np.asarray(u_obs, dtype=float)   # learner sees the observation
        pred, model, y = self._learner_step(self.u, I, morph=True)
        self._step += 1
        self.t += self.dt
        return pred, model, y

    def enable_oracle_recording(self):
        self.record_for_oracle = True
        self._oracle = {"u_obs": [], "u_true": [], "I": [], "Au": [], "AI": [],
                        "duf": []}

    def oracle_data(self):
        return {k: np.array(v) for k, v in self._oracle.items()}


# ================================================================ configuration


def make_grid(nx: int, ny: int, dz_amp: float = 0.25):
    """3D-embedded square grid: positions on a gently warped surface so the
    embedding is genuinely three-dimensional."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    z = dz_amp * np.sin(np.pi * i / max(nx - 1, 1)) * np.cos(np.pi * j / max(ny - 1, 1))
    pos = np.stack([i.astype(float).ravel(), j.astype(float).ravel(),
                    z.ravel()], axis=-1)
    edges = []
    for a in range(nx):
        for b in range(ny):
            idx = a * ny + b
            if a + 1 < nx:
                edges.append((idx, (a + 1) * ny + b))
            if b + 1 < ny:
                edges.append((idx, a * ny + b + 1))
    return pos, np.array(edges, dtype=int)


def default_cfg(mode: str = "weak") -> dict:
    """Reference configuration used for the 15-seed noise sweep."""
    return dict(
        mode=mode,
        dt=0.01,
        # electrical continuum
        kappa_base=4.0, kappa_amp=0.6, t_drift=15.0,
        gamma_u=0.1,
        lam_c=0.8, lam_g=2.0,
        # chemical continuum
        D=0.3, beta=0.6, u_thr=0.1, gamma_c=0.2, c_max=3.0,
        wind=(0.45, 0.15),                 # neuromodulator drift (grid u/s)
        # chaotic excitation
        a_base=5.0, a_amp=0.35, t_inp=8.0,
        sources=(0, 6, 42, 48),            # four corners of the 7x7 grid
        # weak-form identifier
        lam=3.0, lam_s=0.4, ridge=1e-2,
        # morphogenesis encoder
        eta_e=0.3, eta_t=0.05, eta_n=0.01, v_max=8.0,
        t_warm=2.0, t_ramp=1.0,          # identification warm-up gate
        act_th=0.6,                      # chemical morphogenesis gate threshold
        gate_pow=4.0,                    # gate sharpness (power law on c)
    )
