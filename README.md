# WhyLab: Causal Safety Monitoring for Stable Agent Self-Improvement

> NeurIPS 2026 Submission

## Overview

WhyLab is a causal safety monitoring framework that prevents cognitive policy oscillation in self-improving AI agents. It provides three contributions:

- **C1**: Information-theoretic drift detection
- **C2**: Sensitivity-aware effect filtering (E-values + partial R^2)
- **C3**: Lyapunov-bounded adaptive damping

## Repository Structure

```
paper/          # LaTeX source (main.tex, references.bib)
experiments/    # All experiment code and results
  results/      # CSV/Parquet output files
  prompts/      # LLM prompt templates for E4/E5
  cache/        # Cached API responses (gitignored)
  data/         # Downloaded datasets (gitignored)
submission/     # Packaged ZIP files for OpenReview
```

## Reproducing Experiments

### Prerequisites

```bash
pip install numpy pandas scipy pyyaml
```

### Synthetic Experiments (No API key required)

These experiments are fully reproducible with no external dependencies:

```bash
# E1: Drift Detection (CUSUM, Page-Hinkley, WhyLab)
python -m experiments.e1_drift_detection

# E3a: Stability Validation (PID, SGD, Adam, Lyapunov)
python -m experiments.e3a_stability

# Proxy Correlation Analysis (Theorem 1 validation)
python -m experiments.proxy_correlation_analysis
```

### LLM Agent Experiments (Gemini API key required)

These experiments require a `GEMINI_API_KEY` in `.env`:

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# E4: HumanEval benchmark (164 problems x 5 seeds)
python -m experiments.e4_humaneval_benchmark

# E5: SWE-bench Lite (300 problems x 5 seeds)
python -m experiments.e5_swebench_benchmark
```

> **Note**: E4/E5 results are non-deterministic due to LLM sampling.
> Pre-computed results are available in `experiments/results/`.

### Analysis Scripts (Use cached results)

```bash
# E5 subset analysis (oscillating vs non-oscillating)
python -m experiments.e5_subset_analysis

# Safety baseline comparison (Best-of-N, Rollback, etc.)
python -m experiments.e5_safety_baselines
```

## Building the Paper

Requires a TeX distribution (MiKTeX or TeX Live):

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Important Notes

- **Best-of-N baselines** in Table 6 are estimated via `1-(1-p_1)^N` under an
  independence assumption (not actual parallel runs). This is stated in the paper.
- **SWE-bench evaluation** uses lightweight (string-match) test execution.
  Docker-sandboxed full evaluation may yield different absolute pass rates.
- All experiments use **Gemini Flash** (`gemini-2.0-flash`). Generalization to
  other LLM families is not yet validated.

## License

This code is provided for academic review purposes.
