# Morphogenetic Bi-Modal Networks

> **Angles are the computation; the shape is the interpretable spectrum.**

A spatially embedded, gradient-free neural operator that learns through the physical
intersection of 3D geometry and dual chemical/electrical kinetics. No global loss, no
backpropagation: every update is a strictly local, streaming operation. Every neuron
carries a 3D angle channel that multiplies the weight channel — **weights say how
strongly, angles say where** — and the emergent shape is a readable, mechanistic map
of what the network learned.

**Website:** https://sehajr-singhs.github.io/morphogenetic-bimodal-networks/ ·
[play the Flowrunner demo](docs/game.html) · download the [Nature-style](docs/papers/nmi_paper.pdf)
and [IEEE-style](docs/papers/ieee_paper.pdf) papers.

Submission materials: `manuscript.md` (full paper, figures, tables, references),
`figs/` (seven figures rendered from the measured data), and the result files the
paper's numbers are read from.

## What it does

- **Fast electrical continuum** — weighted graph diffusion (discrete port-Hamiltonian
  Laplacian, implicit integration, conservative interior flux).
- **Slow chemical continuum** — reaction–diffusion–advection neuromodulator field whose
  local concentration modulates electrical conductivity (a local "material environment"
  replacing global loss weights; the advection wind breaks symmetry and imprints a
  corridor). Electrical activity releases the chemical; the chemical gates where
  structure grows — bi-modal morphogenesis.
- **IIR spacetime weak form** — all derivatives computed by integration by parts against
  a one-pole exponential window (`y = λ(f − A)`), killing the `2/Δt²` finite-difference
  noise amplification (≈5,500× variance reduction at the reference settings).
- **Identification layer** — per-node recursive least squares over weak-form quantities
  with a slow consolidation window; tracks the local conductivity law at the
  observation-noise floor.
- **Morphogenesis layer** — slow nematic-consensus angulation: 3D structural vectors
  rotate toward the coupling-weighted mean direction of their neighbors
  (liquid-crystal ordering), gated by the chemical field so the geometry grows only on
  the chemically-potentiated corridor, with a drift-axis orienting torque and
  homeostatic magnitude regulation. The closed loop: geometry shapes physics through
  `a_ij` in κ, physics shapes geometry through the identified law and the chemical
  field.

## Measured results (nothing extrapolated)

**Identification** (15 seeds, 5% noise, regime-switching drift): held-out law-fit NMSE
**0.038 ± 0.009** vs **0.339** for the best streaming-global RLS (8.9× — per-node
*locality* itself earns most of the advantage), **0.448** for the frozen global batch
oracle (11.8×), and **0.580** for finite-difference RLS (15.3×). Tracking sits at the
observation-noise floor; convergence in 0.24 ± 0.08 s (15/15 seeds); **589× less**
runtime memory than the frame-buffer batch estimator.

**Morphogenesis** (15 seeds): alignment self-organizes 0.50 → 0.82 into a focused,
corridor-specific directional tensor corridor (contrast +0.20, drift-axis alignment
0.75, corridor correlation 0.36); no windup; chemistry bounded.

**Angles as computation**: the geometric coupling `a_ij` (from the angles alone)
correlates with the full law at r = 0.64 vs r = 0.35 for the raw weights (better in
14/15 seeds) — its interpretability is *constructive* (the shape constitutes part of
the law; on the environment-imposed law it scores 0.33 vs 0.39 for the weights);
learned geometry raises axial corridor transmission in 12/15 seeds; steady-state flux
follows the geometry (r = 0.20).

**Ablation** (8 seeds × 6 variants): identification survives every ablation (NMSE flat
at ~0.036); routing strictly requires the learned geometric channel (6/8 → 0/8
positive); the chemical gate is what makes the shape specific (without it the shape
globalizes and encodes nothing).

**Scaling** (5 seeds × 3 sizes, constant power density): the identification advantage
persists from 49 → 225 nodes (12× → 9× vs oracle; 8.9× → 5.9× vs global RLS), memory
ratio constant at ~590×, and self-organization *improves* with scale (wind alignment
0.71 → 0.87). Corridor specificity and routing weaken at scale — stated as limitations
in the paper.

**The material at 10⁴ nodes** (`scale_large.py`, sparse warm-started CG plant
solver, exact to 1e-9 and verified identical to the dense solve; 3 seeds × 4 sizes,
484 → 10,000 nodes): identification *improves* with scale (law-fit 0.032 → 0.020,
physical NMSE 0.007 → 0.0016), the shape stays readable at every size (r = 0.60–0.65
vs 0.34–0.42 for the raw weights), the memory advantage grows (~240× → ~630×), and
per-node cost is sublinear (4.5 → 158 ms/step for 20.7× more nodes). Honest
limitation: the corridor-routing contrast saturates at 10³–10⁴ nodes while
identification and readability persist.

**Real-data validation — two independent real streams, including modern deep
learning.** (1) SILSO monthly sunspot numbers (3331 real observations, 1749–2026;
delay-embedded as a ring of 24 lag nodes; law frozen after training): holdout
law-fit NMSE on an unseen 27-year window is **1.9e-5 for the streaming weak form vs
9.1e-3 for finite-difference streaming (481×)** and 3.3× better than its own global
batch version; flat under up to 50% added measurement noise. Forecasts through the
frozen law beat persistence at one month (0.116 vs 0.119) and FD laws by up to ~9×
at 24 months; AR/ESN/LSTM (level-fitters) lead at long horizons — the boundary is
stated. (2) NINO3.4 El Niño SST anomalies (943 real observations, 1948–2026),
protocol unchanged: **22× below FD identification** (8.6e-4 vs 1.85e-2), noise armor
flat, and the weak-form law **beats the trained LSTM at 6- and 12-month forecasts**
and by **5.3× at one month with 10% of the data** (0.087 vs 0.464), where the
reservoir collapses (1.73). A pure-NumPy LSTM (BPTT + Adam, no external deps) is
now part of the baseline suite.

**Theory** (`theory.py`): four theorems with numerical verification — a Δt-
independent λ²σ² weak-form noise floor (vs 2σ²/Δt² for finite differences, a
2.2e5× gap at dt=1e-3), RLS contraction under persistence of excitation (238×
reduction, halved in 5 samples; tracking error linear in the law-drift rate),
monotone angulation ascent (alignment 0.51 → 0.88), and closed-loop stability
(morphogenesis perturbs identification by only ~7%).

**Playable demonstration — Flowrunner** (`flowgame.py`; watch `game/preview.html`):
the network plays a maze it never saw. An environment hides a winding high-
conductivity corridor through walls, excited by chaotic sources along the path;
the network samples only noisy potential fields, identifies the local law, and its
angle channel self-organizes into a corridor that steers a token to the goal. Measured
over 4 episodes: the composed material (weights × chemistry × shape) routes **100%
success at exactly oracle-level steps (14.0 vs 14.0)** from noisy observations alone,
while each channel alone fails — shape-only 50%, weights-only 75% — they are
complementary. The clip (`game/flowrunner_seed0.gif`, 26 frames) shows the shape
growing along the hidden law (dashed) and the token following it.

## Files

| file | purpose |
|---|---|
| `biomaterial_net.py` | core module (plant + identifier + morphogenesis) |
| `run_experiments.py` | sweeps, ablation, scaling; baselines; metrics |
| `real_benchmark.py` | real-data benchmarks (SILSO sunspots + NINO3.4 ENSO; weak/FD streaming, batch SINDy, AR, ESN, LSTM, persistence; low-data + streaming panels) |
| `scale_large.py` | scaling to 10⁴ nodes (sparse CG plant solver) |
| `theory.py` | numerical verification of the four theorems |
| `flowgame.py` | Flowrunner: the network plays a hidden-corridor maze (clips for mechanistic interpretability) |
| `make_figures.py` | renders `figs/fig1..fig9` from the measured data |
| `make_paradigm_fig.py` | renders the CROWN-style 3-panel angle-paradigm hero figure |
| `test_biomaterial_net.py` | 12 sanity tests (conservation, noise robustness, stability, shape read-back, Dirichlet solve, real data, noise floor) |
| `manuscript.md` | full submission-form paper (abstract, results, tables, figures, methods, references) |
| `results.json` | 15-seed reference sweep |
| `ablation.json` | 8-seed × 6-variant ablation |
| `scaling.json` | 5-seed × 3 grid sizes |
| `scaling_large.json` | 3-seed × 4 sizes to 10,000 nodes |
| `real_benchmark.json` | sunspot benchmark results (identification + forecast + low-data + online) |
| `real_benchmark_enso.json` | ENSO benchmark results (same panels) |
| `theory.json` | theorem verifications |
| `data_sunspots.csv` | raw SILSO monthly sunspot numbers (public domain) |
| `data_nino34.csv` | NINO3.4 SST anomaly index (NOAA/PSL, public domain) |
| `game/` | Flowrunner results + animated clip + `preview.html` (self-contained, open in a browser) |
| `figs/` | nine paper figures |
| `docs/` | GitHub Pages site (landing page, game demo, papers, figures, robotics page) |
| `robotics/` | The Operations Data Flywheel — streaming weak-form laws for industrial robotics: arm simulator, online identifier, closed-loop flywheel, fleet + contact + data-efficiency experiments, NMI-style paper, and a cover email to Mind Robotics |

## License

MIT — see `LICENSE`. The SILSO sunspot data (`data_sunspots.csv`) is public domain
(SILSO, Royal Observatory of Belgium, https://www.sidc.be/silso/).

## Run

```bash
pip install numpy matplotlib
python run_experiments.py --task sweep    --seeds 15 --T 4000 --out results.json
python run_experiments.py --task ablation --seeds 8  --T 4000 --out ablation.json
python run_experiments.py --task scale    --seeds 5  --T 4000 --sizes 7x7,11x11,15x15 --out scaling.json
python real_benchmark.py --dataset sunspot --out real_benchmark.json
python real_benchmark.py --dataset enso    --out real_benchmark_enso.json
python scale_large.py --sizes 22x22,32x32,64x64,100x100 --seeds 3 --T 1500 \
       --out scaling_large.json
python theory.py
python flowgame.py --episodes 4 --out game            # the maze game + clip
python make_figures.py
python make_paradigm_fig.py                            # the angle-paradigm hero figure
python test_biomaterial_net.py

# the robotics flywheel project
cd robotics && python flywheel.py && python make_figs.py && python test_robotics.py
```

Documented failure modes and their patches (bilinear-RLS collapse, noise-driven rotation
destruction, decimated-EMA forgetting error, explicit-Euler instability, beam-gate chain
breaks, globalizing consensus, Dirichlet-probe collapse) are in `manuscript.md` §4.6.
