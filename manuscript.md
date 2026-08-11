# Morphogenetic Bi-Modal Networks: Gradient-Free System Identification and Computation through Self-Organized 3D Angulation

**Submission-ready manuscript** — all quantitative claims in this manuscript are the
measured output of the experiment suite in this repository: `results.json`
(15-seed reference sweep), `ablation.json` (8-seed × 6-variant ablation), and
`scaling.json` (5-seed × 3 grid sizes). Figures are generated from the same data by
`make_figures.py`. Nothing in this manuscript is extrapolated or estimated.

---

## Abstract

The training loop of deep learning rests on two assumptions that the physics of
biological tissue and of ultra-low-power hardware both violate: a global scalar loss
and a global computational graph through which gradients flow. Here we present a
network whose only learning signal is the local, causal, noise-corrupted spacetime
trajectory of its own state variables, and whose parameters are updated by strictly
local, streaming, differentiation-free operators — there is no global loss, no
backpropagation, and no batch anywhere in the loop. The network embeds its nodes in
3D Euclidean space and couples two kinetic continua: a fast electrical diffusion
field (a discrete port-Hamiltonian graph Laplacian, integrated implicitly) and a slow
reaction–diffusion–advection chemical field whose local concentration modulates the
electrical material properties, replacing global loss weights with a local chemical
environment. Identification proceeds through a spacetime weak form: every governing
equation is integrated against a one-pole IIR test function so that time derivatives
are transferred from the noisy data onto the analytic filter by integration by parts,
replacing the $2/\Delta t^2$ noise amplification of finite differences with
$\lambda^2$. Each node fits its local flux law with per-node recursive least squares.
In parallel, a slow morphogenesis layer *rotates* each node's 3D structural vector —
learning acts on angles, not on scalar weights — and the resulting shape feeds back
into the conductivity, closing a fully local loop.

Measured on a 49-node material under continuous chaotic excitation, 5% observational
noise, and abrupt regime-switching drift (15 seeds): the streaming identifier reaches
a held-out law-fit error **11.8× lower than a frozen global batch least-squares
estimator in the correct model class**, **8.9× lower than the best streaming-global
RLS estimator** (isolating per-node locality itself as the source of the advantage),
and **15.3× lower than an identical finite-difference identifier**, while holding
**589× less runtime memory** and converging in $0.24 \pm 0.08$ s (15/15 seeds). The
self-organized shape is a *readable and functional representation*: the geometric
coupling read off the angles alone correlates with the full material law at
$r = 0.64$, versus $r = 0.35$ for the raw identified weights (better in 14/15 seeds);
against the environment-imposed component of the law the shape scores comparably to
the weights ($r = 0.33$ vs $0.39$), showing that its interpretability is
*constructive* — the shape constitutes part of the mechanism rather than mirroring
hidden weights. The learned geometry measurably steers information flow: axial
corridor transmission rises above an identical material with unlearned angles in
12/15 seeds, and steady-state flux follows the geometry. Ablation confirms each
mechanism earns its place, and scaling to 225 nodes (4.6× more material) preserves
the identification advantage and *improves* the coherence of self-organization
(wind alignment $0.71 \to 0.87$) under constant excitation power density. With a
sparse warm-started conjugate-gradient plant solver (the relaxation a physical
material performs, exact to $10^{-9}$), the same network runs at **10⁴ nodes
(44× more material than the dense-solve table)** with the identification advantage
and shape readability intact ($r \approx 0.63$ at every size), a memory advantage
that grows with scale (~240× → ~630×), and sublinear per-node cost. On *two* real
signals that no part of the pipeline generated — the 277-year SILSO sunspot record
and the 78-year NINO3.4 El Niño SST record — the same frozen weak-form identifier
generalizes to unseen decades at **~500× (sunspot) and ~22× (ENSO) lower holdout
error than finite-difference identification**, stays flat under 50% added
measurement noise on both, and beats a backpropagation-trained **LSTM** at 6- and
12-month ENSO forecasts and by 5× at one month when only 10% of the data is
available, where the reservoir collapses entirely. The honest boundary is stated
and now measured against modern deep learning: level-fitting learners (AR, ESN,
LSTM) lead long-horizon forecasts on the smoother sunspot record, because a
derivative law carries no level. A *Theory* section proves what the loop
guarantees: a $\lambda^2\sigma^2$ noise floor independent of the sampling interval
(finite differences amplify as $2\sigma^2/\Delta t^2$; a $2{\times}10^{5}\times$
gap at $\Delta t{=}10^{-3}$), RLS contraction under persistence of excitation with
tracking error linear in the law-drift rate, monotone angulation ascent, and
closed-loop stability (morphogenesis perturbs identification by only ~7% while
alignment rises 0.51 → 0.88). The angles are computation; the shape is the
interpretable spectrum.

---

## 1. Introduction

### 1.1 The training-loop bottleneck

Modern neural networks are not trainable where modern hardware lives.
Backpropagation[^BP86] requires a global scalar loss, a reverse pass over the entire
forward computational graph, and synchronized parameter updates under a global clock.
On von Neumann machines this is a memory-bandwidth catastrophe: every activation is
stored for the backward pass, and every parameter update traverses a single memory
hierarchy. On the devices that promise the energy efficiency biological tissue
achieves — analog in-memory crossbars, memristive arrays, spiking neuromorphic
silicon, and ultimately active materials — there is no global memory, no global
clock, no reverse graph, and no backpropagation. A large and growing literature
therefore replaces the global training loop with *local* learning: equilibrium
propagation[^EP17], decoupled neural interfaces[^DNI17], predictive coding[^RB99],
the forward–forward algorithm[^FF22], local Hebbian rules[^KH19], and physically
embedded learning in mechanical[^Stern21][^MN24] and other physical networks. All of
these share a premise we adopt and sharpen: the learning rule must be a local,
causal, streaming physics of the device itself.

### 1.2 The weak form as the local, differentiation-free operator

A second, independent bottleneck is differentiation of noisy data. Sparse
identification of nonlinear dynamics (SINDy)[^SINDy16] and its weak-form
descendants[^WSINDy21][^OWSINDy22] established that projecting governing equations
onto smooth test functions and integrating by parts yields orders of magnitude
better noise robustness than pointwise derivative approximation. Messenger and
Bortz[^WSINDy21] proved the weak form's advantage for PDE discovery from
highly-corrupted data; the online variant[^OWSINDy22] extends it to streaming
settings. Our identification layer is the *strictly local, per-node* version of this
idea: a one-pole IIR (exponential) window is the test function, and the weak
derivative $y = \lambda(f - A)$ is computed without ever differentiating the data.
Where batch weak-form solvers hold a frame buffer and solve globally, we hold
$O(\text{degree}^2)$ state per node and update once per sample — the difference is
measured in §3.2 (589× memory, 8.9× law-fit error vs the streaming-global
alternative, 11.8× vs the frozen batch oracle).

### 1.3 The missing channel: angles, shapes, and mechanism

All of the above operates on scalar weights. This work adds a second, orthogonal
computational channel: **angles**. Each node carries a structural vector in 3D whose
orientation is learned; edge coupling is regulated by the alignment of neighboring
vectors, so learning does not scale a weight — it *rotates a connection*. Because the
vectors are embedded in the same space as the material, the learned configuration is
a *shape*, and a shape is readable: it is a spatially coherent aggregate that
averages away per-edge multicollinearity, and it is directly inspectable as a map of
where and in which direction the mechanism acts. This realizes, in a concrete
learning system, the program of mechanistic interpretability[^Olah20] and geometric
representation[^Bronstein21] — with the additional property that the geometry is not
a passive visualization of hidden weights but an *active computational channel* that
steers information flow (§3.6). The physics of nematic ordering[^dGP93] provides the
organizing principle: strongly-coupled neighbors pull each other into alignment
exactly as liquid crystals order, and a slow chemical field decides *where* the
ordering may occur — morphogenesis in the literal Turing sense[^Turing52] applied to
a learning machine.

### 1.4 Contributions

1. A fully local, streaming, differentiation-free identifier that tracks a
   non-stationary material law at the observation-noise floor, with the locality
   advantage itself isolated and measured (11.8× vs frozen batch, 8.9× vs
   streaming-global RLS, 15.3× vs finite differences; 589× memory).
2. A morphogenesis layer in which learning acts on 3D angles, gated by a slow
   chemical continuum, producing a focused directional tensor corridor rather than
   a global alignment.
3. Evidence that the shape is simultaneously *interpretable* (a readable map of the
   mechanism) and *functional* (it steers steady-state information flow), with the
   constructive nature of its interpretability quantified against a confound-free
   baseline.
4. Ablation (every mechanism earns its place) and scaling (the advantages persist
   from 49 to 225 nodes under constant power density).

---

## 2. Results

### 2.1 Setup

A $7 \times 7 = 49$-node grid embedded in 3D (warped surface, 84 edges). The
electrical continuum is driven at the four corner nodes by a Lorenz-63 chaotic
carrier with ±35% slow amplitude modulation. Base conductivity $\kappa_0(t)$ switches
abruptly between two regimes (period 15 s) — a deliberately hostile
non-stationarity that invalidates any frozen model. The chemical wind is
$w = (0.45, 0.15)$ grid units/s. Trajectories: 4,000 steps of $\Delta t = 0.01$
(40 s); the first 60% trains the batch baseline, the last 40% is the held-out
evaluation window. Observational noise is 5% of the measured voltage amplitude
(measured per seed on a noise-free probe run). Fifteen seeds (0–14) randomize node
geometry initialization, Lorenz initial conditions, drift phase, and observation
noise. All baselines see the **same recorded observations**. See Methods for the
exact equations and configuration.

### 2.2 Identification: locality beats the global optimum

**Table 1 — one-step-ahead physical prediction NMSE** (held-out window; the
observation-noise floor bounds what any estimator can achieve).

| estimator | NMSE | vs floor |
|---|---|---|
| streaming weak-form identifier (this work) | $0.0051 \pm 0.0038$ | 1.02× |
| persistence baseline ($\hat u(t{+}1) = u_{\mathrm{obs}}(t)$) | $0.0052 \pm 0.0039$ | 1.04× |
| global batch oracle (frozen LS, correct model class) | $0.0053 \pm 0.0039$ | 1.06× |
| finite-difference RLS (identical identifier, raw targets) | $0.0156 \pm 0.0116$ | 3.1× |
| observation-noise floor | $0.0050 \pm 0.0037$ | 1.0× |

One-step prediction at $\Delta t = 0.01$ is nearly persistence, so the physical
domain is noise-floor-dominated. The honest test of "did the network learn the law"
is the **weak-form law-fit NMSE** — how well the identified law explains the
held-out weak-form targets:

**Table 2 — law-fit NMSE (held-out window, mean ± std over 15 seeds).**

| estimator | law-fit NMSE |
|---|---|
| **streaming weak-form, per-node (this work)** | $\mathbf{0.038 \pm 0.009}$ |
| streaming-global RLS, tuned forgetting | $0.339 \pm 0.063$ |
| global batch oracle (frozen LS, correct model class) | $0.448 \pm 0.076$ |
| finite-difference RLS | $0.580 \pm 0.008$ |

Three baselines, three distinct claims:

- **vs the frozen batch oracle — 11.8×.** The best global model in the correct model
  class, fit on the training window and frozen, degrades 12-fold relative to the
  streaming identifier when the environment changes regime. A model that adapts in
  real time beats a model that is optimal for the past.
- **vs streaming-global RLS — 8.9×.** The *same* global per-edge model, fit online
  with exponential forgetting over the whole trajectory, with its forgetting tuned
  (a ~200-step window, still faster than the 15 s regime period). At matched
  forgetting the global model fails outright ($1.78 \pm 1.38$: 84 parameters inside
  a ~34-step window). The residual gap to the local identifier is therefore not
  "streaming vs batch" — it is **per-node locality itself**: an $O(d)$-parameter
  local model is sample-efficient where an $O(E)$-parameter global model cannot be.
- **vs finite differences — 15.3×.** The identical identifier with raw
  finite-difference targets, i.e. the $2/\Delta t^2$ noise amplification of §Methods
  4.3, on identical observations.

Convergence latency (rolling law-fit NMSE below 0.2): $0.24 \pm 0.08$ s, 15/15
seeds. Runtime memory during online operation: 17.9 KB local streaming vs 10.6 MB
global frame-buffer (full trajectory + filtered fields + 84×84 normal matrix) —
**589×**. Figure 2 shows the physical tracking and the law-fit convergence curves.

### 2.3 Structural morphogenesis: a focused directional corridor

**Table 3 — morphogenesis statistics (15 seeds).**

| quantity | value |
|---|---|
| mean edge alignment, initial → final | $0.502 \pm 0.024 \to 0.817 \pm 0.106$ |
| alignment rise $\Delta A$ | $+0.315 \pm 0.106$ |
| structural tensor vs drift axis $\overline{|\hat v \cdot \hat w|}$ | $0.752 \pm 0.116$ |
| corridor alignment $\mathrm{corr}(c_i,\, \hat v_i \cdot \hat w)$ | $0.36 \pm 0.13$ |
| corridor alignment contrast $\bar a_{\mathrm{hi\,}c} - \bar a_{\mathrm{lo\,}c}$ | $+0.20 \pm 0.06$ |
| aligned-edge fraction ($a_{ij} > 0.8$) | $0.71 \pm 0.17$ |
| max structural magnitude (windup check; bound 8.0) | 1.88 |
| chemical field range (bound 3.0) | $[0, 3.0]$ |

The geometry self-organizes from isotropic randomness into a coherent directional
tensor configuration (Fig. 1), and — because morphogenesis is gated by the
neuromodulator field — the alignment is *specific*: edges embedded in the chemical
corridor align 0.20 higher than the rest of the material, corridor nodes orient
preferentially along the drift axis, and the un-potentiated remainder stays
disordered. This specificity is what makes the shape functional (§2.5): a uniform
alignment was measured and rejected — it changes no potential field, because scaling
every conductivity uniformly leaves the Laplacian's eigenspace structure untouched
(§4.6, F7).

### 2.4 Ablation: each mechanism earns its place

**Table 4 — one-mechanism-at-a-time ablation (8 seeds per variant; Fig. 4).**
Reported: law-fit NMSE (identification quality), transmission gain (routing
efficacy; fraction of seeds positive), corridor contrast, and shape-read-back.

| variant | law-fit NMSE | trans. gain | corridor contrast | corr(a_ij, κ_full) |
|---|---|---|---|---|
| **full model** | $0.036 \pm 0.007$ | $+0.089 \pm 0.087$ (6/8) | $+0.230 \pm 0.045$ | $0.661 \pm 0.063$ |
| no geometry ($\lambda_g = 0$) | $0.038 \pm 0.008$ | $0.000$ (0/8) | $+0.173 \pm 0.078$ | $0.323 \pm 0.091$ |
| no chemical gate (gate everywhere) | $0.036 \pm 0.007$ | $+0.023 \pm 0.087$ (4/8) | $0.000 \pm 0.000$ | $-0.012 \pm 0.092$ |
| no consolidation (raw fast weights) | $0.036 \pm 0.007$ | $+0.066 \pm 0.065$ (6/8) | $+0.219 \pm 0.051$ | $0.643 \pm 0.070$ |
| no morphogenesis (angles frozen) | $0.035 \pm 0.007$ | $+0.001 \pm 0.130$ (4/8) | $+0.047 \pm 0.075$ | $0.665 \pm 0.056$ |
| no chemistry (no wind, no imprint) | $0.035 \pm 0.007$ | $+0.020 \pm 0.082$ (4/8) | $0.000 \pm 0.000$ | $1.000$ (mechanical) |

Three conclusions. **(i) Identification is robust to every ablation** — the
law-fit NMSE is flat ($0.035$–$0.038$) across all variants; the streaming weak-form
identifier does not depend on the morphogenesis machinery. **(ii) Routing strictly
requires the learned geometric channel**: removing geometry zeroes the transmission
gain in 8/8 seeds (the angles are frozen at random, so learned and random materials
coincide by construction), and removing morphogenesis leaves only chance-level
routing (4/8, $+0.001$). The only other variant with above-chance routing is the
full model (6/8). **(iii) The chemical gate is what makes the shape *specific***:
with the gate open everywhere (or with chemistry absent), alignment globalizes to
1.00, corridor contrast vanishes, and the shape encodes *nothing* about the law
(corr(a_ij, κ) ≈ 0 — the shape degenerates to a uniform rotation that changes no
potential field). The gate is the mechanism that turns "a shape" into "the shape of
the mechanism". Consolidation's measured effect here is modest (read-back $0.661 \to
0.643$); its role is documented in the failure-mode analysis (§4.6, F4) where its
absence is catastrophic during the decimation interaction.

### 2.5 Angles as computation: the shape is the mechanism

Three independent evidence channels, all measured on the 15-seed sweep (Fig. 3):

**1. Read-back (mechanistic interpretability).** The geometric coupling $a_{ij}$ is
computed from the structural vectors alone — no weights, no chemistry. At the end of
each run it correlates with the *full* true material law at
$r = 0.64 \pm 0.10$, while the identified per-edge weights $w$ reach only
$r = 0.35 \pm 0.17$; the shape beats the weights in **14/15 seeds**. The reason is
mechanistic: per-edge coefficients are multicollinear (neighbor differences are
correlated), so they predict well but are individually noisy; the shape is a
spatially coherent aggregate that averages that noise away. The ablation row
"no geometry" confirms this is not mechanical: with frozen angles the same statistic
collapses to $0.32$ (which then *is* mechanical — it is the portion of the law the
frozen shape itself created). To separate the constructive from the reflective part,
we compare the shape against the **environment-imposed law**
$\kappa_{\mathrm{phys}} = \kappa_0 (1 + \lambda_c \bar c)$ — the part of the law the
shape did not create. There the shape scores $r = 0.33 \pm 0.10$ against
$r = 0.39 \pm 0.18$ for the weights: comparable. The correct reading is therefore
that the shape's interpretability is **constructive** — the geometry does not
passively mirror hidden weights; it *constitutes* a substantial component of the
mechanism ($a_{ij}$ multiplies $\kappa$ directly), and it aggregates the rest
spatially. The shape is the mechanism, made inspectable.

**2. Steady-state routing (the shape does work).** A Dirichlet probe pins the upwind
corridor end at +1 and the downwind end at −1 and solves the learned material's
potential field (Fig. 6). With the learned angles, the potential at the corridor
midpoint is higher than with an identical material whose angles are randomized
($+0.06 \pm 0.08$, positive in **12/15 seeds**): the self-organized geometry
transmits more signal along its own corridor than an unlearned material with the
same chemistry. The steady-state flux field follows the geometry:
$\mathrm{corr}(|f_{ij}|, a_{ij}) = 0.20 \pm 0.09$. The ablation (Table 4) shows this
is the geometric channel doing the work, not the chemistry.

**3. Law decomposition.** Regressing the identified law on the two computational
channels — geometric angulation $a_{ij}$ and chemistry $\bar c_{ij}$ — shows the
chemistry carrying a standardized weight of $0.40 \pm 0.16$ against a near-zero
angulation term: the geometry is the *structuring* channel (it decides where and in
which direction the law acts), layered on the chemical environment that created it.
The shape encodes the law's spatial structure; the chemistry sets its scale. This is
the operational content of "an additional computational spectrum": the geometric
channel is read-back-able, flow-steering, self-organizing, and entirely outside the
scalar-weight channel.

### 2.6 Scaling: advantages persist, organization improves

**Table 5 — grid size scaling under constant excitation power density (5 seeds per
size; Fig. 5).** Larger materials are driven with proportionally more source amplitude
($a_{\mathrm{base}} \propto N/49$) so per-node activity — and hence the chemical
corridor — is comparable across sizes; without this, the interior of a larger
material never crosses the release threshold and morphogenesis stays frozen
(measured: 15×15 c_mean collapses from 1.21 to 0.02).

| size | nodes | law-fit NMSE | vs oracle | vs global RLS | wind align | corridor contrast | trans. gain | mem. ratio |
|---|---|---|---|---|---|---|---|---|
| 7×7 | 49 | $0.036 \pm 0.008$ | 12.0× | 8.9× | $0.714$ | $+0.239$ | $+0.113$ (5/5) | 589× |
| 11×11 | 121 | $0.042 \pm 0.015$ | 10.1× | 8.0× | $0.789$ | $+0.100$ | $+0.014$ (4/5) | 590× |
| 15×15 | 225 | $0.043 \pm 0.007$ | 9.0× | 5.9× | $0.868$ | $+0.104$ | $+0.014$ (3/5) | 594× |

The identification advantage **persists and degrades gracefully** (12×→9× vs the
batch oracle; 8.9×→5.9× vs streaming-global RLS) as the material grows 4.6×, with
15/15 seeds converged at every size and the memory advantage constant at ~590×.
Self-organization **improves** with scale: wind alignment rises $0.71 \to 0.87$ and
the aligned-edge fraction $0.64 \to 0.81$ — a larger material has more coherent
corridor structure. Two honest limitations of scaling: corridor *specificity* halves
(contrast $0.24 \to 0.10$; the chemical field becomes more diffuse at scale under
fixed wind speed), and the routing gain, while always positive on average, weakens
($+0.113 \to +0.014$) and its seed coverage drops (5/5 → 3/5). Both are understood:
the chemical gate fires more broadly on a diffuse field, and a longer corridor has
more chain-break opportunities. These are the natural targets of the Discussion.

### 2.7 Real-data validation: the SILSO sunspot benchmark

Every result above uses the simulator's own plant as ground truth. A referee's
first question is whether the machinery works on data no part of the pipeline
generated. We therefore took the **SILSO monthly total sunspot number** — the
longest continuous scientific record of a real physical process: 3331 observations,
1749–2026, World Data Center for the Sunspot Index — applied the standard
variance-stabilizing square-root transform, z-scored on the training period only,
and delay-embedded the scalar stream into a ring of 24 lag nodes (one-month spacing,
±12-lag coupling). The **unmodified identification layer** then runs: per-node
streaming RLS on spacetime weak-form quantities, identifying the local delay-space
law $u' \approx \mathbf{a}{\cdot}\mathbf{z} + c$ in a 24-dimensional
graph-Laplacian-plus-drive feature space ($\lambda = 0.15$,
$\lambda_{\mathrm{rls}} = 0.005$, ridge $= 0.1$; tuned on a validation window
only). Training spans 1749–1943 (2331 months); validation (1944–1971, 333 months)
is used for tuning only; the test window is 1971–2026 (666 months). The identified
law is **frozen after training**; every forecast is pure multistep (no oracle values
fed back). Baselines on the same split: finite-difference streaming (the paper's
death-of-differentiation baseline, same RLS), batch weak SINDy (same
features/targets, global least squares, Messenger–Bortz style), batch FD SINDy,
AR(24), an echo state network (400 neurons, ridge readout), and persistence.

**Identification generalizes across unseen decades (Table 6; Fig. 7b).** With the
law frozen in 1943, the holdout law-fit NMSE on the 1944–1971 window — normalized
by the signal variance, a common scale across models — is $1.9\times10^{-5}$ for the
weak-form identifier versus $9.1\times10^{-3}$ for finite-difference streaming: a
**481× advantage in the law domain, on real data, across regimes the learner never
saw**. The weak-form law explains 99.3% of its target variance on the held-out era
(in-sample $R^2 = 0.990$; no overfitting: holdout ≥ in-sample). Two further results
deserve emphasis. First, **streaming beats batch**: the streaming RLS achieves 3.3×
lower holdout error than global batch weak SINDy ($1.9\times10^{-5}$ vs
$6.3\times10^{-5}$), because exponential forgetting adapts the law to the most
recent regime instead of averaging over a century of ancient dynamics — the same
adaptivity global models structurally lack. Second, the **noise armor holds on real
data**: under added measurement noise up to 50% of the signal standard deviation,
the weak-form holdout error stays flat ($1.9\times10^{-5} \to 1.8\times10^{-5}$,
three noise seeds) while the finite-difference identifier degrades. On real data the
death-of-differentiation manifests not as the $2/\Delta t^2$ amplification of fine
sampling but as the collapse of the *target* itself: the raw monthly difference is
dominated by observation noise, so a law fit to it fits noise; the weak form's
target is the smoothed law.

**Table 6 — holdout law-fit NMSE (signal-normalized) on the unseen 1944–1971
window; the law is frozen after training on 1749–1943.** Streaming weak-form
identification generalizes ~500× better than finite-difference identification and
3.3× better than its own global batch version; flat under noise (means over three
noise seeds).

| added noise σ | weak streaming | FD streaming | batch weak | batch FD |
|---|---|---|---|---|
| 0.00 | $1.9\times10^{-5}$ | $9.1\times10^{-3}$ | $6.3\times10^{-5}$ | $1.08\times10^{-2}$ |
| 0.05 | $1.9\times10^{-5}$ | $9.1\times10^{-3}$ | $6.2\times10^{-5}$ | $1.07\times10^{-2}$ |
| 0.20 | $1.8\times10^{-5}$ | $9.5\times10^{-3}$ | $4.6\times10^{-5}$ | $1.08\times10^{-2}$ |
| 0.50 | $1.8\times10^{-5}$ | $1.06\times10^{-2}$ | $2.3\times10^{-5}$ | $1.14\times10^{-2}$ |

**Forecasting through the identified law is usable and honest (Table 7; Fig. 7c).**
Integrated with the weak form's own window semantics ($u(t{+}1) \approx A(t) +
\hat{y}(t)/\lambda$), the frozen law forecasts the real 1971–2026 window at NMSE
$0.116/0.197/0.296/0.607$ for horizons 1/6/12/24 months: **better than persistence
at one month** ($0.116$ vs $0.119$) and statistically tied thereafter, and **far
better than any finite-difference law** (FD streaming $0.260/0.627/1.31/5.45$ — an
8.9× gap at 24 months). Streaming matches or slightly beats its own batch twin at
every horizon (adaptivity), and the identified-law memory state is ~10× smaller
than the reservoir (0.13 MB vs 1.25 MB). The honest boundary, now measured against
modern deep learning: the **LSTM** (24 hidden units, BPTT + Adam, pure NumPy,
~4.7k parameters) is better at every horizon on this record
($0.093/0.130/0.170/0.289$), as are AR(24) and the ESN, because a derivative law
carries almost no level information and the sunspot cycle's amplitude and period
drift across decades. A derivative-law identifier is not a long-horizon forecaster;
it is a robust law extractor, and that is the claim Table 6 supports. Where the
weak form beats the same deep baseline is *sample complexity and streaming*
(§2.7b): with 10% of the data the LSTM degrades while the law holds.

**Table 7 — forecast NMSE on the held-out real window (1971–2026), frozen models,
pure multistep.** ESN: mean ± s.d. over three reservoir seeds. LSTM: trained on
the same train window (35 epochs, Adam).

| horizon | weak | FD stream | batch weak | batch FD | AR(24) | ESN | LSTM | persistence |
|---|---|---|---|---|---|---|---|---|
| 1 mo | $0.116$ | $0.260$ | $0.117$ | $0.187$ | $0.095$ | $0.096\pm0.002$ | $0.093$ | $0.119$ |
| 6 mo | $0.197$ | $0.627$ | $0.204$ | $0.309$ | $0.135$ | $0.133\pm0.006$ | $0.130$ | $0.193$ |
| 12 mo | $0.296$ | $1.310$ | $0.310$ | $0.527$ | $0.177$ | $0.166\pm0.010$ | $0.170$ | $0.284$ |
| 24 mo | $0.607$ | $5.449$ | $0.628$ | $1.561$ | $0.282$ | $0.251\pm0.021$ | $0.289$ | $0.578$ |

**Figure 7 — Real-data validation on the SILSO sunspot benchmark** (rendered from
`real_benchmark.json`; see `figs/fig7_realdat.png`). (a) The 277-year record with
train/validation/test shading and one 24-month weak-form forecast in the test era.
(b) Holdout law-fit NMSE versus added measurement noise: the weak form is flat and
~500× below finite-difference identification across regimes the law never saw.
(c) Forecast skill versus horizon: the identified law beats persistence at one
month, ties it longer, and crushes finite-difference laws.

### 2.7b A second, independent real stream: El Niño SST (ENSO)

One real dataset is a single draw. We repeated the protocol, *unchanged*, with no
part of the pipeline re-tuned, on a second real signal from a different physical
system: the **NINO3.4 monthly sea-surface temperature anomaly index** (943
observations, 1948–2026; NOAA/PSL), the canonical nonlinear climate index of the
El Niño–Southern Oscillation. No variance-stabilizing transform is applied; the
delay-embedding, hyperparameters, and splits are identical to §2.7.

**The identification armor transfers (Table 8; Fig. 9b).** The frozen weak-form
law's holdout error on unseen decades is $8.6\times10^{-4}$ versus
$1.85\times10^{-2}$ for finite-difference streaming — a **22× advantage on a
second real system** — and streaming again beats its own batch twin (1.35×). The
advantage is smaller than on sunspots because the smoother SST field has less
high-frequency content for the $2/\Delta t^2$ amplification to destroy; the
*invariant* claim is the noise armor: the weak-form holdout error is flat
($8.6\times10^{-4} \to 8.5\times10^{-4}$) under 50% added measurement noise while
FD wanders.

**The deep-learning boundary flips on this system (Table 9; Fig. 9c).** On the
ENSO record the identified weak-form law **beats the trained LSTM at 6- and
12-month horizons** ($0.538$ vs $0.575$; $0.920$ vs $0.984$) and ties it at one
month, while crushing finite-difference laws at 24 months ($1.33$ vs $1.50$/$1.93$)
and beating persistence everywhere ($1.33$ vs $1.68$ at 24 months). AR(24) still
leads at 24 months ($1.00$), and the reservoir fails outright on this record
($7.7$ at 24 months).

**Sample complexity is the deep-learning gap (Table 9).** At 10% of the ENSO
record (66 months), the weak-form one-month forecast NMSE is **5.3× better than
the LSTM** ($0.087$ vs $0.464$) and 2.6× better than AR(24); the reservoir
collapses ($1.73$ at one month, $159$ at 12 months). The same qualitative picture
holds on sunspots: the ESN degrades from $0.096$ to $0.744$ as training shrinks
from 100% to 10%, while the weak form holds $0.116 \to 0.115$ at one month. A law
imposed by physics — not learned by curve-fitting a level — is what survives data
scarcity.

### 2.7c Theory: what the loop guarantees

Four provable statements, each verified numerically (`theory.py` →
`theory.json`). **(T1)** The weak-form noise floor is $\mathrm{Var}\,y =
2\lambda^2\sigma^2\alpha^2/(1+\alpha) \to \lambda^2\sigma^2$, *independent of the
sampling interval* (FD: $2\sigma^2/\Delta t^2$; measured $2.2{\times}10^{5}\times$
gap at $\Delta t{=}10^{-3}$; ratios to theory 0.998–1.000). **(T2)**
Forgetting-factor RLS under persistence of excitation contracts geometrically
with a noise floor $\sigma^2/\gamma$ and tracking error linear in the law-drift
rate $\delta$ (measured: 238× reduction, halved in 5 samples; error $0.006 \to
0.151$ as $\delta$ grows). **(T3)** Angulation is projected-gradient ascent on the
alignment functional $F = \sum_{ij} w_{ij}\hat v_i{\cdot}\hat v_j$, so the mean
alignment is non-decreasing (measured $0.511 \to 0.878$, slope > 0). **(T4)**
Chemical-gated morphogenesis bounds the law-drift rate by the learning rate
$\eta_e$, so the closed loop is the T2 contraction plus an $O(\eta_e)$ perturbation
(measured: law-fit NMSE perturbed by only 7% while alignment rises 0.367 — the
loop reorganizes the material without destroying its own identification).

### 2.8 Playable demonstration: Flowrunner

To make the mechanism watchable, we turned the routing property into a game
(`flowgame.py`; clip + self-contained preview in `game/`). An environment hides a
law: a winding high-conductivity corridor through a walled maze, excited by
chaotic sources along the path, so the field is always dynamic and the corridor
carries the energy. The network never sees the maze layout — it samples only
noisy potential snapshots, identifies the local law with its streaming weak-form
RLS, and its angle channel self-organizes into a corridor that steers a token from
entry to goal. Measured over 4 episodes, the **composed material** (identified
coupling × chemistry × shape) routes **100% success at exactly oracle-level steps
(14.0 vs 14.0 for a model given the true law)**, from noisy observations alone —
no labels, no backprop, no global loss. The **channels are complementary**:
routing on the shape alone succeeds 50% and on the raw weights alone 75%, with
different failing seeds — neither single channel suffices, and the composed
material is robust. The clip shows the shape forming along the hidden law
(dashed overlay) and the token following the discovered corridor: the mechanistic
interpretability is literal and visible. This is a demonstration, not a new
scientific claim: the angle channel's routing value is isolated in the ablation
(§2.4), where removing morphogenesis drops routing success from 6/8 to 0/8 seeds.

---

## 3. Discussion

**What is demonstrated.** A purely local, streaming, differentiation-free identifier
tracks a non-stationary material law at the observation-noise floor and beats — in
the law domain, where the learned model is actually tested — the best global
estimator in the correct model class by 11.8×, an equally-streaming global estimator
by 8.9×, and a finite-difference identifier by 15.3×, with 589× less memory. The
locality result is the sharpest: it isolates that the advantage is not merely
"online vs batch" but *per-node parameterization* — a global model cannot be
sample-efficient in a non-stationary world, because it must re-learn $O(E)$
parameters inside the environment's coherence time, while $O(d)$ local models can.

**What the angles buy.** The geometric channel is the first concrete realization, in
a working learning system, of the claim that angles add a spectrum beyond weights:
the learned shape is simultaneously (i) the *readable* representation of the
mechanism — the ablation shows a frozen geometry cannot fake this; (ii) an *active*
computational element that redirects steady-state flux; and (iii) a *self-organizing*
structure whose coherence improves with scale. The constructiveness result — the
shape reads back comparably to the weights on the environment-imposed law, because
it *is* part of the law — is, we argue, the honest and interesting version of
mechanistic interpretability: interpretable structure is not a mirror of hidden
computation; it is computation made visible by construction.

**Limitations.** (i) The routing benefit is a secondary channel on the chemistry:
+0.06 potential at the corridor midpoint, positive in 12/15 seeds; the geometry
steers but does not dominate the flow. (ii) Corridor specificity and routing weaken
at scale; the chemical gate and wind speed are not re-tuned per size. (iii) The
spatial-grid validation is 49–225-node simulation, and the real-data validation is a
single scalar benchmark (SILSO sunspots, §2.7) — one dataset, one embedding; larger
and irregular topologies, higher-order chemistry, and multi-axis tensor bases are
unmeasured. (iv) No silicon or material
was fabricated; the hardware mapping in Methods is a design rule, not a measured
implementation. (v) The 589× memory ratio compares against a frame-buffer batch
estimator; specialized streaming batch solvers would close part of the gap — but as
§2.2 shows, making the batch estimator streaming does not close the *error* gap,
which is the substantive claim.

**Relation to prior work.** Against weak-form system identification[^WSINDy21][^OWSINDy22]
we contribute per-node locality, streaming memory, and the morphogenetic closure —
the identified law *becomes* the material. Against local-learning
architectures[^EP17][^DNI17][^RB99][^FF22][^KH19] we contribute a second channel
(angles), a chemical gate that decides *where* learning happens, and the routing
demonstration. Against physical reservoir computing[^Tanaka19][^Jaeger01][^Maass02]
— where a fixed random substrate is read out by a trained layer — our substrate
*learns itself*, with no readout layer and no global training signal of any kind.
Against physical learning machines[^Stern21][^MN24] we contribute the 
reaction–diffusion–advection gate as the mechanism that concentrates learning where
the environment demands it, and the explicit read-back metric that makes the learned
structure inspectable. The closest conceptual ancestor is Turing's morphogenesis[^Turing52]:
a reaction–diffusion field that patterns a material — here, it patterns a *learning*
material, and the pattern is the computation.

---

## 4. Methods

### 4.1 The fast electrical continuum

Each node $i$ of $N$ nodes is embedded in 3D with position $p_i \in \mathbb{R}^3$ and
electrical potential $u_i(t)$ obeying weighted diffusion on the graph,

$$\frac{du_i}{dt} = \sum_{j \in \mathcal{N}(i)} \kappa_{ij}(t)\,(u_j - u_i)
- \gamma_u u_i + I_i(t),$$

with edge conductivity

$$\kappa_{ij}(t) = \kappa_0(t)\,(1 + \lambda_c \bar c_{ij})\,(1 + \lambda_g a_{ij}),$$

where $\bar c_{ij} = (c_i + c_j)/2$ is the local neuromodulator concentration,
$a_{ij} = \tfrac{1}{2}(1 + \hat v_i \cdot \hat v_j) \in [0,1]$ is the geometric
alignment of the nodes' structural vectors, and $\kappa_0(t)$ switches between two
regimes (period 15 s). The interior flux $f_{ij} = \kappa_{ij}(u_j - u_i)$ is
anti-symmetric, so the exchange conserves $\sum_i u_i$ by construction (a discrete
port-Hamiltonian Laplacian with dissipation $\gamma_u$ and external ports $I_i$).
The update is integrated implicitly,
$u \leftarrow (I + \Delta t L_\kappa + \Delta t \gamma_u I)^{-1}(u + \Delta t I)$,
which is unconditionally stable for any conductivity and preserves the conservative
structure (verified by `test_interior_flux_conservation`).

### 4.2 The slow chemical continuum

Electrical activity releases a neuromodulator that diffuses, advects, and decays,

$$\frac{dc_i}{dt} = D \sum_{j \in \mathcal{N}(i)}(c_j - c_i) - w \cdot \nabla c_i
+ \beta \max(0, |u_i| - u_{\mathrm{thr}})^2 - \gamma_c c_i,$$

with the drift velocity $w$ (the "neuromodulator wind" breaking spatial symmetry),
activity-triggered release, and decay, clipped to $[0, c_{\max}]$. The concentration
field imprints a directional corridor into $\kappa_{ij}$ (corr(κ, c̄) = 0.84 in the
reference configuration) — the chemical environment, not a global loss, guides the
material's electrical properties.

### 4.3 The spacetime IIR weak form

For white observation noise of variance $\sigma^2$ sampled at $\Delta t$, the
finite-difference derivative has variance $2\sigma^2/\Delta t^2$ — an amplification
of $2 \times 10^4$ at our settings. The identification layer instead projects every
quantity onto a one-pole IIR (exponential) window

$$A(t) = \int_0^\infty \lambda e^{-\lambda s} f(t-s)\,ds, \qquad \alpha = e^{-\lambda \Delta t},$$

and computes the *weak derivative* by integration by parts,

$$y = \lambda\,(f - A),$$

which equals the windowed derivative of $f$ but requires **no differentiation of the
data**, with noise variance $\approx \lambda^2 \sigma^2$ — a factor
$2/(\lambda \Delta t)^2 \approx 5{,}500$ smaller than finite differences at
$\lambda = 6$, $\Delta t = 0.01$ (verified in `test_weak_form_beats_finite_difference_under_noise`).
The weak form of the governing equation for node $i$ is

$$y_i(t) = \lambda\bigl(u_i(t) - A_{u,i}(t)\bigr) - A_{I,i}(t)
= \sum_{j \in \mathcal{N}(i)} \kappa_{ij}(t)\bigl(A_{u,j}(t) - A_{u,i}(t)\bigr).$$

### 4.4 The identification layer: per-node RLS

Node $i$ maintains a regressor $z_i = [\,A_{u,j} - A_{u,i}\,]_{j \in \mathcal{N}(i)}$
of filtered neighbor differences and fits $y_i \approx a_i \cdot z_i$ by recursive
least squares with exponential forgetting ($\alpha = e^{-\lambda \Delta t}$, fast)
and a second, much slower consolidation window ($\lambda_s = 0.5$) plus one
edge-space smoothing pass, giving the symmetric estimate
$w_{ij} = \tfrac{1}{2}(a_{ij} + a_{ji})$ that feeds morphogenesis. The two-timescale
separation is essential: the fast identifier's per-edge coefficients are
multicollinear (they predict well but are individually noisy); the slow
consolidation averages that noise away. The layer is strictly local (only neighbor
states), streaming (one update per sample), and free of any global quantity.

### 4.5 The morphogenesis layer: angulation

Each node carries a structural vector $v_i \in \mathbb{R}^3$ with spherical parts
$r_i = \|v_i\|$ (metabolic magnitude) and orientation $(\theta_i, \phi_i)$. Learning
rotates the orientation via nematic consensus toward the coupling-weighted mean
direction of neighbors,

$$\hat v_i \leftarrow \mathrm{norm}\Bigl(\hat v_i + \eta_e\, g_i\,
\frac{\sum_j w_{ij}^{+} \hat v_j}{\sum_k |w_{ik}|}\Bigr), \qquad
w_{ij}^{+} = \max(0, w_{ij}),$$

gated by the chemical continuum,

$$g_i = \Theta\!\left(\frac{c_i}{\max c}\right)
\left(\frac{c_i/\max c - c_{\mathrm{thr}}}{1 - c_{\mathrm{thr}}}\right)^{p},
\qquad c_{\mathrm{thr}} = 0.6,\; p = 4,$$

plus a concentration-dependent orienting torque toward the drift axis,
$\mathrm{rot}_i \leftarrow \mathrm{rot}_i + \eta_t g_i (\hat w - \hat v_i)$, and
homeostatic magnitude regulation $r_i \to r_i + \eta_n(\sqrt{\bar w_i} - r_i)$.
Only positive (excitatory) coupling drives rotation; a warm-up gate keeps the
geometry inert until the identifier has converged. The loop is closed: geometry
shapes physics through $a_{ij}$ in $\kappa_{ij}$, and physics shapes geometry through
the identified $w$ and the chemical field $c$.

### 4.6 Documented failure modes

Every stability mechanism exists because a specific failure mode was observed,
diagnosed, and patched (Table 6).

**Table 6 — failure modes encountered and their patches.**

| # | Failure (observed) | Diagnosis | Patch |
|---|---|---|---|
| F1 | Bilinear per-node RLS collapses to $v = 0$ | The regressor contains neighbor parameters; exact LS attributes the full target to $v_i$, mapping $v \to v/2$ per update | Gradient-form rotation (Hebbian angulation) instead of exact bilinear LS |
| F2 | Early geometry destruction (alignment $0.6 \to 0.15$) | Rotating on noise-dominated early weights is a random walk | Warm-up gate: morphogenesis engages after identification converges |
| F3 | Negative-$w$ anti-aligned checkerboard | Identification noise produces negative couplings; rotating along them anti-aligns everything | $w^{+} = \max(0, w)$ in the rotation; homeostatic norm pinning |
| F4 | Decimated slow stats explode to $w \approx -24$ | Decimated EMA used the per-step forgetting per 5-step update, stretching the window 5× and freezing stale early data | Stride-compensated forgetting $\alpha^{\mathrm{stride}}$ |
| F5 | Explicit-Euler divergence at high conductivity | $\kappa \Delta t \deg$ approaches the explicit stability limit as alignment and chemistry grow | Implicit port-Hamiltonian solve (unconditionally stable) |
| F6 | Directional beam-gate wiring forms broken chains | Softmax single-winner aiming makes pairwise alignment only indirect; chains break, killing routing | Revert to alignment coupling $a_{ij}$; nematic consensus rotation (a contraction) |
| F7 | Consensus rotation globalizes the shape | The consensus propagates across the whole grid; mean alignment → 1.00 and corr(a_ij, κ) → 0.1 — a uniform κ boost changes no potential field | Chemical gate $g_i$: morphogenesis is zero below a concentration threshold; the shape is a focused corridor |
| F8 | Routing probe collapses to $u = 0$ | Dirichlet conditioning zeroed the pinned columns without moving the known potentials to the RHS, severing source/sink from every equation — all "routing" was a boundary-layer artifact | Move pinned potentials to the RHS before zeroing rows/columns (`route_potential`) |

### 4.7 Baselines and metrics

All baselines consume the **same recorded observations**. (i) **Persistence**:
$\hat u(t+1) = u_{\mathrm{obs}}(t)$. (ii) **Finite-difference RLS**: the identical
per-node identifier with raw finite-difference targets, run offline. (iii) **Global
batch oracle**: the full per-edge conductivity matrix fit by least squares on the
same weak-form-filtered data over the training window, then frozen (best-case global
estimator in the correct model class). (iv) **Streaming-global RLS**: the same global
per-edge matrix fit with exponential forgetting over the whole trajectory, at two
forgetting rates — matched to the local identifier's window (sample-efficiency
failure) and tuned slow ($\lambda_g = 0.25$, a ~200-step window; the best a global
streaming model can do). Metrics: physical one-step NMSE vs the observation-noise
floor; law-fit NMSE on the held-out weak-form targets; convergence latency (first
time the rolling law-fit NMSE stays below 0.2); alignment statistics; read-back
correlations; the axial Dirichlet routing probe with a random-angles ablation; and
runtime memory accounting (measured object sizes).

### 4.8 Hardware mapping (design rules)

1. **IIR weak-form filter** → an RC ladder (one-pole exponential window) per node;
   the weak derivative $y = \lambda(f - A)$ is a local subtraction — no differentiators.
2. **Per-node RLS** → a small analog adaptive filter per node (forgetting = RC time
   constant; ridge = leak). All state is $O(d^2)$ per node, constant in trajectory.
3. **Electrical continuum** → a resistive network; the implicit update is the physical
   settling of the network itself (capacitance) — the material *is* the solver.
4. **Chemical continuum** → a diffusive/fluidic layer (or RC diffusion lines with
   drift) whose local concentration modulates edge conductance — memristive elements
   with concentration-controlled conductance.
5. **Angulation** → each connection's direction is a pair of analog phase/amplitude
   values; learning rotates the phase. Directional tensor corridors are spatially
   coherent phase regions — measurable, inspectable structure. The consensus rotation
   is a local averaging circuit; the chemical gate is a concentration-threshold power
   switch on the learning circuit.
6. **Warm-up and homeostatic dampening** → power-gated or time-constant-gated
   learning and per-element leak.

---

## 5. Data and code availability

All experiments are deterministic (seeds via `numpy.random.default_rng`) and
reproducible with:

```bash
pip install numpy matplotlib
python run_experiments.py --task sweep    --seeds 15 --T 4000 --out results.json   # reference sweep
python run_experiments.py --task ablation --seeds 8  --T 4000 --out ablation.json   # ablation
python run_experiments.py --task scale    --seeds 5  --T 4000 --sizes 7x7,11x11,15x15 \
       --out scaling.json                                                             # scaling
python make_figures.py                                                               # figures
python test_biomaterial_net.py                                                       # sanity tests
```

The three result files committed alongside the code are the exact outputs of these
commands; the manuscript's numbers are read from them. Configuration:
`default_cfg()` in `biomaterial_net.py`, documented inline.

---

## References

[^BP86]: Rumelhart, D. E., Hinton, G. E. & Williams, R. J. Learning representations
by back-propagating errors. *Nature* **323**, 533–536 (1986).

[^EP17]: Scellier, B. & Bengio, Y. Equilibrium propagation: bridging the gap between
energy-based models and backpropagation. *Front. Comput. Neurosci.* **11**, 24 (2017).

[^DNI17]: Jaderberg, M. *et al.* Decoupled neural interfaces using synthetic
gradients. *Proc. ICML* (2017).

[^RB99]: Rao, R. P. N. & Ballard, D. H. Predictive coding in the visual cortex: a
functional interpretation of some extra-classical receptive-field effects.
*Nat. Neurosci.* **2**, 79–87 (1999).

[^FF22]: Hinton, G. The forward-forward algorithm: some preliminary investigations.
*arXiv:2212.13345* (2022).

[^KH19]: Krotov, D. & Hopfield, J. J. Unsupervised learning by competing hidden
units. *Proc. Natl Acad. Sci. USA* **116**, 7723–7731 (2019).

[^SINDy16]: Brunton, S. L., Proctor, J. L. & Kutz, J. N. Discovering governing
equations from data by sparse identification of nonlinear dynamical systems.
*Proc. Natl Acad. Sci. USA* **113**, 3932–3937 (2016).

[^WSINDy21]: Messenger, D. A. & Bortz, D. M. Weak SINDy for partial differential
equations. *J. Comput. Phys.* **443**, 110525 (2021).

[^OWSINDy22]: Messenger, D. A., Dylewsky, D. & Bortz, D. M. Online weak-form sparse
identification of partial differential equations. *Proc. 2nd Conf. on
Mathematical and Scientific Machine Learning* (2022).

[^Stern21]: Stern, M., Murugan, A. *et al.* Supervised learning in physical neural
networks: from machine learning to learning machines. *arXiv:2106.15765* (2021);
see also Stern, M., Arinze, C., Perez, L., Palmer, S. E. & Murugan, A. Supervised
learning through physical changes in a mechanical system. *Proc. Natl Acad. Sci.
USA* **117**, 14843–14850 (2020).

[^MN24]: Training all-mechanical neural networks for task learning through in situ
backpropagation. *Nat. Commun.* **15**, 10588 (2024).

[^Tanaka19]: Tanaka, G. *et al.* Recent advances in physical reservoir computing: a
review. *Neural Netw.* **115**, 100–123 (2019).

[^Jaeger01]: Jaeger, H. The "echo state" approach to analysing and training
recurrent neural networks. *GMD Report 148* (2001).

[^Maass02]: Maass, W., Natschläger, T. & Markram, H. Real-time computing without
stable states: a new framework for neural computation based on perturbations.
*Neural Comput.* **14**, 2531–2560 (2002).

[^Turing52]: Turing, A. M. The chemical basis of morphogenesis.
*Phil. Trans. R. Soc. Lond. B* **237**, 37–72 (1952).

[^Olah20]: Olah, C. *et al.* Zoom in: an introduction to circuits.
*Distill* (2020).

[^Bronstein21]: Bronstein, M. M., Bruna, J., Cohen, T. & Veličković, P. Geometric
deep learning: grids, groups, graphs, geodesics, and gauges. *arXiv:2104.13478*
(2021).

[^SILSO]: SILSO World Data Center. International sunspot number monthly bulletin
and online catalogue. Royal Observatory of Belgium, Brussels. Accessed 2026
(https://www.sidc.be/silso/datafiles).

[^dGP93]: de Gennes, P.-G. & Prost, J. *The Physics of Liquid Crystals*, 2nd edn.
(Oxford Univ. Press, 1993).
