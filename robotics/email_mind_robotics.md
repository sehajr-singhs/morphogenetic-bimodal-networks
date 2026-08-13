# Cover email: Mind Robotics

A draft you can send to Mind Robotics (the Rivian industrial-AI spinout).
The paper and code live in this repository under `robotics/`; the full paper
is `robotics_nmi.pdf`, the measured data is `robotics.json`, and everything is
reproducible with `python flywheel.py`.

---

**Subject: The engine for your data flywheel — streaming laws from the operations stream, no batches, no backprop**

Hi Mind Robotics team,

I've been following the thesis behind the company — that Rivian's manufacturing
operations data should become the foundation of a robotics data flywheel —
and I built something that addresses the bottleneck I believe sits at the
center of it: *re-identification*.

The classical loop works like this: log data, stop, re-fit a dynamics model in
batch, deploy. The loop ratchets; it doesn't spin. Every payload change, tool
swap, or bit of wear invalidates the deployed model, and re-fitting is too
expensive to do often. Online alternatives (finite differences) amplify
encoder noise by ~2/Δt² — at a 2 ms control period, a factor of ~500,000 —
which is why online identification in industry is limited to slow, curated
excitation trajectories instead of the operations stream itself.

The paper attached here shows a differentiation-free alternative that closes
the loop. We identify each joint's rigid-body law *from the noisy operations
stream itself* by integrating it against a one-pole IIR window — integration
by parts moves derivatives off the data and onto the analytic filter, cutting
noise gain by more than 1,000×. The identified coefficients *are* the
physics (payload shows up in the gravity term, wear in the friction term),
so the law is a per-robot health readout, not a black box.

Measured results (2-DOF arm, 500 Hz logging, all in the paper):

- **The flywheel spins**: closed-loop identification feeding computed-torque
  feedforward cuts tracking error **7.1×** in two laps; the same loop with
  finite differences stalls. When we add a 1.5 kg payload mid-experiment, the
  weak-form law re-adapts within **one lap**; the FD law degrades to a 50.8°
  direction error (weak form: 24°).
- **Mechanism under noise**: at 5% sensor noise, law direction is recovered to
  **3.4°** (finite differences: 9.8°, batch oracle: 4.3°) and torque
  prediction is **3.7×** more accurate than finite differences.
- **The fleet is readable**: across six robots, identified law vectors separate
  payload and wear variants (direction cosine 0.99) and read payload mass to
  within 17% — from the operations stream alone.
- **Contact detection is free**: the frozen nominal law doubles as a contact
  detector with **31×** residual SNR during a socket insertion, per-robot and
  threshold-free.

The whole thing runs causally, at control frequency, in O(1) memory per law
coefficient, with no labels and no batches — exactly what a flywheel that
runs continuously across a fleet needs. The experiment suite runs in ~2
minutes and is fully open in this repository.

I'd love to talk about whether this maps onto the Rivian manufacturing
streams, and about what a pilot on a real arm would look like. Happy to share
the full framework paper (the biomorphic angle-paradigm work this descends
from), or to set up a short call.

Best,
[Your name]
[Your affiliation / contact]

---

*Repository:* https://sehajr-singhs.github.io/morphogenetic-bimodal-networks
(robotics section) — `robotics/` contains the arm simulator, identifier,
experiment suite, figures, and this paper's LaTeX source.
