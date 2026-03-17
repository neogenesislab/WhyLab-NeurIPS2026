# WhyLab Pipeline & Architecture

## Overview

WhyLab wraps any base agent/estimator with a **three-layer causal audit**
that intercepts each proposed update and decides whether to accept or reject it.

```
┌─────────────────────────────────────────────────────────┐
│                   Agent Loop (Reflexion)                 │
│  solve → test → reflect → propose update                │
└──────────────────────┬──────────────────────────────────┘
                       │ proposed update
                       ▼
┌─────────────────────────────────────────────────────────┐
│              WhyLab Audit Layer                         │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ C1: Drift│──▸│ C2: Sensitiv.│──▸│ C3: Lyapunov │    │
│  │ Monitor  │   │ Gate (E/RV)  │   │ Damping      │    │
│  └──────────┘   └──────────────┘   └──────────────┘    │
│       │                │                  │             │
│       ▼                ▼                  ▼             │
│  drift_flag       gate_decision      damped_step        │
│                                                         │
│  Final: ACCEPT (damped) or REJECT                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
              Apply / Skip update
```

---

## Component Details

### C1: Drift Monitor (ADWIN)

**Paper ref:** Bifet & Gavaldà (2007), SDM  
**Code:** `experiments/e1_drift_detection.py`, `experiments/audit_layer.py::DriftMonitor`

Monitors the evaluation score stream for distributional shift using an
information-theoretic divergence measure. When drift is detected, downstream
components are alerted to increase caution.

| Parameter | Default | Description |
|---|---|---|
| `c1_window` | 5 | Sliding window size (aligned to `max_attempts`) |
| `agreement_threshold` | 0.7 | Minimum agreement ratio to pass |

### C2: Sensitivity Gate (E-value × Robustness Value)

**Paper ref:** VanderWeele & Ding (2017); Cinelli & Hazlett (2020)  
**Code:** `experiments/e2_sensitivity_filter.py`, `experiments/audit_layer.py::SensitivityGate`

Evaluates each proposed update's causal robustness using dual thresholds:
- **E-value**: minimum confounder strength (as risk ratio) to nullify the effect
- **RV (Robustness Value)**: fraction of residual variance explained by unobserved confounders

Updates with weak causal evidence (low E, high RV) are rejected.

| Parameter | Default | Calibrated | Description |
|---|---|---|---|
| `c2_e_thresh` | 2.0 | 1.5 | E-value threshold (strong → moderate) |
| `c2_rv_thresh` | 0.10 | 0.05 | RV threshold |

> **Calibration rationale:** In short-window agent loops (5 attempts),
> observational-study thresholds (E≥2.0) cause excessive rejection (~85%).
> Relaxing to E≥1.5 corresponds to "moderate" robustness per WhyLab's
> own E-value scale. Both operating points are reported as a Pareto frontier.

### C3: Lyapunov Damping Controller

**Paper ref:** Berkenkamp et al. (2017); Chow et al. (2018)  
**Code:** `experiments/e3a_stationary.py`, `experiments/audit_layer.py::DampingController`

Bounds the step size of accepted updates using an observable energy proxy
with EMA-smoothed adaptive control:

1. Compute energy proxy `V(t)` from score trajectory
2. If `V(t) > V(t-1)` (energy increasing), shrink step size
3. Clamp step within `[epsilon_floor, ceiling]`

| Parameter | Default | Description |
|---|---|---|
| `c3_epsilon_floor` | 0.01 | Minimum damping factor |
| `c3_ceiling` | 0.80 | Maximum damping factor |

---

## Experiment Pipeline

### E1–E3: Synthetic Validation

```
config.yaml → e1_drift_detection.py → results/e1_metrics.csv → e1_figures.py → figures/
           → e2_sensitivity_filter.py → results/e2_metrics.csv → e2_figures.py → figures/
           → e3a_stationary.py → results/e3a_*.csv → e3a_figures.py → figures/
           → e3b_heavy_tail.py → results/e3b_*.csv
```

### E4: Agent Benchmark (HumanEval + Reflexion)

```
config.yaml (splits + ablations)
    │
    ▼
e4_agent_benchmark.py
    │  --split pilot|main|full
    │  --holdout_exclude pilot
    │
    ├── For each (seed, ablation, problem):
    │     reflexion_loop.py::run_reflexion_episode()
    │       ├── LLM solve → test → reflect
    │       ├── audit_layer.py::AgentAuditLayer.evaluate()
    │       │     C1 → C2 → C3 → accept/reject
    │       └── Record metrics (pass, safe_pass, regression, oscillation)
    │
    ├── Output: results/e4_metrics.csv
    │
    └── e4_analyze.py
          ├── Cluster bootstrap CI (problem-level)
          ├── Paired Δ tests (ablation vs none)
          ├── Pareto data (acceptance vs regression)
          └── LaTeX table → paper/tables/e4_main.tex
```

### Ablation Configurations

| Name | C1 | C2 | C3 | C2 Thresholds | Notes |
|---|---|---|---|---|---|
| `none` | ✗ | ✗ | ✗ | — | Unaudited baseline |
| `C1_only` | ✓ | ✗ | ✗ | — | Drift detection only |
| `C2_default` | ✗ | ✓ | ✗ | E≥2.0, RV≥0.1 | Observational thresholds |
| `C2_calibrated` | ✗ | ✓ | ✗ | E≥1.5, RV≥0.05 | Env-adapted thresholds |
| `C3_only` | ✗ | ✗ | ✓ | — | Damping only |
| `full_default` | ✓ | ✓ | ✓ | E≥2.0, RV≥0.1 | All layers, strict |
| `full_calibrated` | ✓ | ✓ | ✓ | E≥1.5, RV≥0.05 | All layers, adapted |
| `none_stress` | ✗ | ✗ | ✗ | — | 10 attempts, no audit |
| `full_stress` | ✓ | ✓ | ✓ | E≥1.5, RV≥0.05 | 10 attempts + audit |

---

## Key Metrics

| Metric | Definition | Direction |
|---|---|---|
| `pass_rate` (Pass@1) | Final attempt solved the problem | ↑ higher is better |
| `safe_pass` | Passed without any audit rejection | ↑ |
| `first_pass_attempt` | Attempt number of first success | ↓ lower is better |
| `acceptance_rate` | Proposed updates accepted / total | ↑ |
| `regression_count` | cheap_score decreased between attempts | ↓ |
| `oscillation_index` | Sign changes in score delta | ↓ |

---

## Running Experiments

```bash
# E1–E3: Synthetic (no API key needed)
python experiments/e1_drift_detection.py
python experiments/e2_sensitivity_filter.py
python experiments/e3a_stationary.py

# E4: Agent benchmark (requires GEMINI_API_KEY in .env)
python -m experiments.e4_agent_benchmark --split pilot
python -m experiments.e4_agent_benchmark --split main --holdout_exclude pilot

# Analysis
python -m experiments.e4_analyze \
  --input experiments/results/e4_metrics.csv \
  --emit_latex paper/tables/e4_main.tex
```
