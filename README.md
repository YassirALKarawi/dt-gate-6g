# AI-Enabled Digital Twin Gate for Secure & Resilient 6G (Python Simulation)

<p align="center">
  <a href="https://github.com/YassirALKarawi/dt-gate-6g/actions">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/YassirALKarawi/dt-gate-6g/ci.yml?branch=main">
  </a>
  <a href="https://colab.research.google.com/github/YassirALKarawi/dt-gate-6g/blob/main/notebooks/00_colab_minimal.ipynb">
    <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
  <a href="https://github.com/YassirALKarawi/dt-gate-6g/releases">
    <img alt="Release" src="https://img.shields.io/github/v/release/YassirALKarawi/dt-gate-6g">
  </a>
  <a href="https://github.com/YassirALKarawi/dt-gate-6g/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-blue">
</p>

Python-only simulation of the paper’s control loop: EKF/NIS trust (τ), Kingman + Cantelli reliability, tube-based tightening, risk scoring with isotonic calibration + randomized smoothing (CVaR), and a projected optimizer.  
All figures/tables are generated into `data/outputs/`. **No ns-3/srsRAN** dependencies.

---

## Quickstart

```bash
# optional venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run simulation (700 epochs, 5 seeds)
python -m src.dt_gate.simulate --epochs 700 --seeds 1 2 3 4 5 --variant Proposed

# generate figures + tables (high-DPI, consistent palette)
python scripts/make_figs.py --inputs data/outputs/*.csv --outdir data/outputs
