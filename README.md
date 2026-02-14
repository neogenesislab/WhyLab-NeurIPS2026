# WhyLab: Causal Decision Intelligence Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Decision Intelligence](https://img.shields.io/badge/Decision-Intelligence-purple)](https://en.wikipedia.org/wiki/Decision_intelligence)

> **"Don't just predict the future. Cause it."**

WhyLab is the world's first **Decision Intelligence Engine** powered by **Multi-Agent Debate**.
It bridges the gap between **Causal Inference** (Science) and **Business Decision** (Art).

### 🎯 Why WhyLab?

- **For POs**: "Rollout or Not?" — Get actionable verdicts (e.g., "ROI +12%, Risk Low → **Rollout**").
- **For Data Scientists**: SOTA accuracy (R-Learner error -25% vs benchmark).
- **For Devs**: 3 lines of code to integrate causal AI into your pipeline.

```python
import whylab as wl

# 1. Analyze (Science)
result = wl.analyze(data, treatment='coupon', outcome='purchase')

# 2. Debate (Business Logic)
verdict = result.debate(
    growth_hacker="Maximize Revenue",
    risk_manager="Minimize Churn"
)

# 3. Decision
print(verdict.action_item)
# "🚀 [Approved] Rollout 100%. Expected Profit: +$1.2M"
```

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

### Multi-Agent Debate System (Decision Intelligence)

Three AI agents simulate real organizational decision-making:

1. **Growth Hacker** (10 evidence types): Finds revenue opportunities from causal signals
2. **Risk Manager** (8 attack vectors): Warns about potential losses and model vulnerabilities
3. **Product Owner (Judge)**: Synthesizes Growth vs Risk → delivers actionable verdict
   - `🚀 Rollout 100%` | `⚖️ A/B Test 5%` | `🛑 Reject`

---

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+ (Dashboard)

### Installation
```bash
# Clone
git clone https://github.com/Yesol-Pilot/WhyLab.git
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

#### 2. Connect Your Data (CSV / SQL / BigQuery)
```bash
# CSV
python -m engine.cli --data "sales.csv" --treatment coupon --outcome purchase

# PostgreSQL
python -m engine.cli --data "postgresql://user:pass@host/db" \
  --db-query "SELECT * FROM users" --treatment coupon --outcome purchase

# BigQuery
python -m engine.cli --data "my-gcp-project" --source-type bigquery \
  --db-query "SELECT * FROM dataset.table" --treatment treatment --outcome outcome
```

#### 3. Ask Questions (RAG Agent)
```bash
python -m engine.cli --query "쿠폰 효과가 있어?" --persona growth_hacker
python -m engine.cli --query "리스크는 없어?" --persona risk_manager
```

#### 4. Monitor Causal Drift
```bash
# 1회 드리프트 체크
python -m engine.cli --monitor

# 30분 간격 연속 모니터링 + Slack 알림
python -m engine.cli --monitor --interval 30 --slack-webhook $SLACK_URL
```

#### 5. Run Benchmarks
```bash
python -m engine.pipeline --benchmark ihdp acic jobs \
  --replications 10 --output results/ --latex
```

#### 6. Launch Dashboard
```bash
cd dashboard && npm run dev
# Open http://localhost:4000
```

---

## Project Structure

```
whylab/
  engine/
    cells/          # 11 modular analysis cells
    agents/         # AI debate & discovery agents
    connectors/     # Multi-source data (CSV/SQL/BigQuery)
    monitoring/     # Causal drift detection & alerting
    data/           # Benchmark data loaders (IHDP/ACIC/Jobs)
    rag/            # RAG-based Q&A agent (multi-turn, persona)
    server/         # MCP Protocol server (7 tools, 3 resources)
    config.py       # Central configuration (no magic numbers)
    orchestrator.py # Cell pipeline orchestrator
    cli.py          # CLI entry point (v3)
  dashboard/        # Next.js interactive dashboard
  paper/            # Research vision & figures
  tests/            # Unit & integration tests (55+)
  results/          # Benchmark output (JSON + LaTeX)
```

## Citation

If you use WhyLab in your research, please cite:

```bibtex
@software{whylab2026,
  title={WhyLab: Causal Decision Intelligence Engine with Multi-Agent Debate},
  author={WhyLab Contributors},
  year={2026},
  url={https://github.com/Yesol-Pilot/WhyLab}
}
```

## License

MIT License
