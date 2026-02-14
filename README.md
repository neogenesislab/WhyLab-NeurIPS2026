# WhyLab: AI-Driven Causal Inference Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Causal inference meets AI agents.**
> WhyLab combines state-of-the-art causal inference algorithms with a multi-agent debate system
> that automatically validates causal claims — no other tool does this.

## What Makes WhyLab Different?

| | DoWhy | EconML | CausalML | **WhyLab** |
|---|:---:|:---:|:---:|:---:|
| Causal Graph Modeling | O | - | - | **O** |
| Meta-Learners (S/T/X/DR/R) | - | O | O | **O** |
| Double Machine Learning | - | O | - | **O** |
| Refutation Tests | O | - | - | **O** |
| **AI Agent Auto-Debate** | - | - | - | **O** |
| **Auto Verdict (CAUSAL/NOT)** | - | - | - | **O** |
| **Interactive Dashboard** | - | - | - | **O** |

**WhyLab's killer feature**: While existing tools help you *write code* for causal analysis,
WhyLab deploys AI agents that *independently discover, debate, and validate* causal relationships.

---

## Benchmark Results

Evaluated on 3 standard causal inference benchmarks (10 replications each):

### IHDP (Hill 2011, n=747, p=25)

| Method | sqrt(PEHE) | ATE Bias |
|---|:---:|:---:|
| **T-Learner** | **1.164 +/- 0.024** | **0.039 +/- 0.031** |
| DR-Learner | 1.194 +/- 0.034 | 0.038 +/- 0.029 |
| Ensemble | 1.214 +/- 0.025 | 0.046 +/- 0.034 |
| X-Learner | 1.324 +/- 0.029 | 0.035 +/- 0.024 |
| S-Learner | 1.383 +/- 0.033 | 0.064 +/- 0.040 |
| LinearDML | 1.465 +/- 0.024 | 0.066 +/- 0.061 |
| R-Learner | 1.635 +/- 0.046 | 0.135 +/- 0.107 |

> **Ref**: BART ~1.0 (Hill 2011), GANITE ~1.9 (Yoon 2018), CEVAE ~2.7 (Louizos 2017)

### ACIC (Dorie 2019, n=4802, p=58)

| Method | sqrt(PEHE) | ATE Bias |
|---|:---:|:---:|
| **S-Learner** | **0.491 +/- 0.017** | **0.018 +/- 0.013** |
| X-Learner | 0.569 +/- 0.009 | 0.020 +/- 0.011 |
| Ensemble | 0.612 +/- 0.013 | 0.013 +/- 0.007 |
| LinearDML | 0.614 +/- 0.010 | 0.071 +/- 0.025 |
| DR-Learner | 0.799 +/- 0.017 | 0.040 +/- 0.018 |
| T-Learner | 0.835 +/- 0.013 | 0.041 +/- 0.018 |
| R-Learner | 1.206 +/- 0.035 | 0.111 +/- 0.060 |

### Jobs (LaLonde 1986, n=722, p=8)

| Method | sqrt(PEHE) | ATE Bias |
|---|:---:|:---:|
| **LinearDML** | **170.5 +/- 32.3** | 39.2 +/- 36.6 |
| S-Learner | 288.4 +/- 11.3 | 79.2 +/- 36.8 |
| X-Learner | 377.2 +/- 22.4 | 38.6 +/- 16.3 |
| Ensemble | 381.8 +/- 18.4 | 39.8 +/- 33.8 |
| T-Learner | 482.7 +/- 23.2 | **35.2 +/- 21.7** |
| DR-Learner | 535.0 +/- 29.3 | 34.9 +/- 25.2 |
| R-Learner | 703.4 +/- 36.6 | 81.7 +/- 73.8 |

---

## Architecture: Cellular Agents

Inspired by biological cells, WhyLab's engine consists of modular, autonomous "cells":

```
Data -> Causal -> MetaLearner -> Conformal -> Explain -> Refutation
  |                                                         |
  v                                                         v
  Sensitivity -> Viz -> Export -> Report -> Debate -> VERDICT
```

| Cell | Role | Lines |
|---|---|:---:|
| `DataCell` | SCM-based synthetic data + external CSV loading | 343 |
| `CausalCell` | DML estimation (Linear/Forest/Auto) | 420 |
| `MetaLearnerCell` | 5 meta-learners + Oracle ensemble | 420 |
| `ConformalCell` | Distribution-free confidence intervals | 330 |
| `RefutationCell` | Placebo, Bootstrap, Random Cause tests | 400 |
| `SensitivityCell` | E-value, Overlap, GATES analysis | 400 |
| `DebateCell` | Multi-agent causal verdict | 562 |

### Multi-Agent Debate System

Three AI agents evaluate causal claims:

1. **Advocate** (10 evidence types): Defends the causal relationship
2. **Critic** (8 attack vectors): Challenges the causal claim
3. **Judge**: Weighs evidence and delivers verdict (`CAUSAL` / `NOT_CAUSAL` / `UNCERTAIN`)

---

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+ (Dashboard)

### Installation
```bash
# Clone
git clone https://github.com/your-org/whylab.git
cd whylab

# Python environment
conda create -n whylab python=3.10
conda activate whylab
pip install -r engine/requirements.txt

# Dashboard
cd dashboard && npm install
```

### Usage

#### 1. Run Causal Pipeline (Synthetic Data)
```bash
python -m engine.main --scenario A   # Credit limit -> Default
python -m engine.main --scenario B   # Marketing coupon -> Signup
```

#### 2. Run with Your Own CSV
```bash
python -m engine.main \
  --data "your_data.csv" \
  --treatment "treatment_col" \
  --outcome "outcome_col" \
  --features "age,income,score"
```

#### 3. Run Benchmarks
```bash
python -m engine.pipeline --benchmark ihdp acic jobs \
  --replications 10 --output results/ --latex
```

#### 4. Launch Dashboard
```bash
cd dashboard && npm run dev
# Open http://localhost:3004
```

---

## Project Structure

```
whylab/
  engine/
    cells/          # 11 modular analysis cells
    agents/         # AI debate & discovery agents
    data/           # Benchmark data loaders (IHDP/ACIC/Jobs)
    rag/            # RAG-based Q&A agent
    server/         # FastAPI backend
    config.py       # Central configuration (no magic numbers)
    orchestrator.py # Cell pipeline orchestrator
    pipeline.py     # CLI entry point
  dashboard/        # Next.js interactive dashboard
  paper/            # Research vision & figures
  tests/            # Unit & integration tests
  results/          # Benchmark output (JSON + LaTeX)
```

## Citation

If you use WhyLab in your research, please cite:

```bibtex
@software{whylab2026,
  title={WhyLab: AI-Driven Causal Inference Engine with Multi-Agent Debate},
  author={WhyLab Contributors},
  year={2026},
  url={https://github.com/your-org/whylab}
}
```

## License

MIT License
