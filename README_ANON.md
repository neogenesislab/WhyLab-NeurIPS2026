# WhyLab: A Causal Audit Framework for Stable Agent Self-Improvement

> Autonomous code for NeurIPS 2026 Submission

This repository contains the official implementation of **WhyLab**, a causal audit framework designed to safeguard self-improving AI agents against evaluation drift, fragile outcomes, and unbounded parameter updates.

## ⚡ Reproducibility Notes (Paper vs Code)

To ensure full transparency during peer review, please note the following minor differences between the theoretical descriptions in the paper and the exact experimental implementations in this codebase:

1. **E1 Constant Mismatch (`K` streams & Binning)**
   - **Paper**: Describes the theoretical maximum severity environment with $K=5$ and Sturges' rule binning.
   - **Code**: `config.yaml` strictly uses $K=3$ and a fixed `N_BINS=10` to guarantee reproducible bounded divergence across 40 seeds.
2. **E2 Robustness Value (RV)**
   - **Paper**: Denotes the threshold conceptually as $RV_q \ge RV_{min}$ (larger implies greater robustness per Cinelli & Hazlett).
   - **Code**: The C1 filter is implemented strictly as matching the *Residual Variance Proxy*, meaning the code rejects outcomes when `RV > threshold` (lower is safer). The mathematical bounds hold exactly symmetrically.
3. **E3a Controller (EMA)**
   - **Paper**: Formulates the practical controller using double-smoothed EMA (both $\hat{m}_2$ and $\bar{\zeta}$).
   - **Code**: Exposes the $\hat{m}_2$ baseline directly into the threshold update for real-time reactivity, maintaining the identical Lyapunov convergence properties.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all experiments sequentially
python experiments/e1_drift_detection.py
python experiments/e2_sensitivity_filter.py
python experiments/e3a_stationary.py
python experiments/e3b_heavy_tail.py
```

## 📁 Repository Structure

```text
WhyLab_Anonymous/
├── README_ANON.md           # This file
├── requirements.txt         # Minimal deps list
├── experiments/             # Core scripts (E1-E3)
│   ├── config.yaml          # Hyperparameters
│   ├── e1_drift_detection.py
│   ├── e2_sensitivity_filter.py
│   ├── e3a_stationary.py
│   └── e3b_heavy_tail.py
└── paper/                   # Paper LaTeX source
    ├── main.tex
    ├── references.bib
    └── main.pdf             # Compiled PDF
```
