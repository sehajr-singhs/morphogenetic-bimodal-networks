# The Operations Data Flywheel

**Streaming weak-form laws for industrial robotics — no backpropagation, no
batches, no re-fitting.**

A 2-DOF arm identifies its own rigid-body law from the noisy operations
stream, feeds it back as feedforward computation in a closed flywheel, reads
payload and wear from the resulting law vectors across a fleet, and detects
contact with a frozen-law residual — all measured, all reproducible in ~2
minutes.

Built on the weak-form operator from the companion framework paper
([Morphogenetic Bi-Modal Networks](../nmi_paper.tex)): governing laws are
identified by integrating them against a one-pole IIR window, so derivatives
move off the noisy data and onto the analytic filter (integration by parts).
Finite differences amplify noise by `2/Δt²`; the weak form amplifies by `λ²`
— a >1,000× reduction at control rates.

## Why this is the flywheel engine

| Measured result | Number |
|---|---|
| Law direction at 5% noise (weak vs FD vs batch) | 3.4° / 9.8° / 4.3° |
| Torque prediction at 5% noise (weak vs FD) | 3.7× more accurate |
| Flywheel: tracking NMSE drop in 2 laps | 0.61 → 0.086 (7.1×) |
| Mid-run payload change: law re-adaptation | 1 lap (FD degrades to 50.8°) |
| Fleet separation (payload / wear direction cosine) | 0.99 |
| Payload mass read from the law | within 17% |
| Contact detection SNR (frozen-law residual) | 31× |
| Data needed for a usable law at 5% noise | 4.8 s of stream |

## Quickstart

```bash
pip install numpy matplotlib

# run all five experiments (writes robotics.json, ~2 min)
python flywheel.py

# render the paper figures from robotics.json
python make_figs.py

# compile the paper
pdflatex robotics_nmi.tex
```

## Files

| File | What it is |
|---|---|
| `arm.py` | 2-DOF arm plant: gravity, viscous+Coulomb friction, payload, socket contact; PD control; persistently-exciting references |
| `identify.py` | Streaming weak-form identifier (per-joint minimal momentum bases, column-normalized RLS), FD and batch baselines |
| `flywheel.py` | The five experiments → `robotics.json`: law recovery, the closed flywheel, fleet law-vector constellation, contact detection, data efficiency |
| `make_figs.py` | Paper figures drawn entirely from `robotics.json` |
| `robotics_nmi.tex` | The NMI-style paper (7 pages, all figures from measured data) |
| `email_mind_robotics.md` | A cover email draft connecting this to Mind Robotics' data-flywheel thesis |
| `robotics.json` | Raw measured output of the experiment suite |

## The five experiments

1. **Law recovery under noise** — noise fractions 0 / 5% / 50%, 3 seeds each;
   weak form vs finite differences vs batch oracle, on law direction and
   torque-domain NMSE.
2. **The flywheel** — 5 laps; identified laws drive computed-torque
   feedforward; payload +1.5 kg at lap 3. Weak form converges and re-adapts;
   finite differences stall.
3. **The fleet** — 6 robots (base, payload ×2, worn, heavy, payload+wear);
   law vectors as points in coefficient space; payload mass read from the
   gravity-coefficient ratio.
4. **Contact detection** — identify once, freeze, insert; the residual
   `τ − θ̂·Φ` spikes at contact (31× SNR).
5. **Data efficiency** — law error vs logged seconds; weak form needs 4.8 s
   at 5% noise; the LSTM baseline (60 epochs, no mechanism) is shown for
   contrast.

## Results are honest

Every figure in the paper is drawn from `robotics.json`, the direct output of
`flywheel.py`. Weaknesses are reported as measured: full-vector fleet
clustering is weak (silhouette 0.25 — identification noise dominates small
directions), the clean-data floor is set by the O(λΔt) discretization error,
and at 50% noise torque-domain NMSE converges across methods while law
*direction* remains the differentiator.
