# Changelog

All notable changes to WhyLab are documented in this file.

## [0.1.0] - 2026-02-14

### Added
- **3-Line Python API** (`whylab.analyze()`) for effortless causal analysis
- **7 Meta-Learners**: S, T, X, DR, R-Learner + LinearDML + Oracle Ensemble
- **3 Academic Benchmarks**: IHDP, ACIC, Jobs with automated evaluation
- **Multi-Agent Debate System**: Advocate + Critic + Judge for auto-verdict
- **11-Cell Pipeline**: DataCell → CausalCell → ... → DebateCell
- **Interactive Dashboard**: Next.js with real-time causal visualization
- **Conformal Prediction**: Distribution-free individual confidence intervals
- **Refutation Tests**: Placebo, Bootstrap CI, Random Cause
- **Sensitivity Analysis**: E-value, Overlap diagnostics, GATES/CLAN
- **SHAP Explanations**: Feature importance + counterfactual simulation
- **DuckDB Integration**: Zero-copy data processing engine
- **GPU Acceleration**: LightGBM GPU support (auto-detection)
- **LaTeX Export**: Auto-generated benchmark comparison tables

### Documentation
- Comprehensive README with benchmark results and competitive comparison
- Getting Started guide (5-minute quickstart)
- Benchmark reproduction guide
- Architecture documentation with Living Ledger mapping

### Infrastructure
- `pyproject.toml` packaging (`pip install whylab`)
- CI workflow: Python 3.9/3.10/3.11, pytest-cov, benchmark smoke test
- 22+ test cases covering benchmarks, meta-learners, debate system
