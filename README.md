# WhyLab: Causal Decision Intelligence Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![v0.2.0](https://img.shields.io/badge/version-0.2.0-green.svg)](https://pypi.org/project/whylab/)
[![Pipeline Cells](https://img.shields.io/badge/Pipeline-16_Cells-blueviolet)](#architecture-16-cell-pipeline)
[![Live Demo](https://img.shields.io/badge/Live_Demo-whylab.vercel.app-00d4aa)](https://whylab.vercel.app/dashboard)

> **"Don't just predict the future. Cause it."**

<p align="center">
  <a href="https://whylab.vercel.app/dashboard">
    <img src="docs/dashboard_hero.png" alt="WhyLab Dashboard" width="800" />
  </a>
  <br />
  <em>▲ 인터랙티브 대시보드 — ROI 시뮬레이터, AI 토론, CATE 탐색기, 인과 그래프</em>
</p>

WhyLab is the world's first **Decision Intelligence Engine** powered by **Multi-Agent Debate**.
It bridges the gap between **Causal Inference** (Science) and **Business Decision** (Art).

### 🎯 Why WhyLab?

- **For POs**: "Rollout or Not?" — Get actionable verdicts (e.g., "ROI +12%, Risk Low → **Rollout**").
- **For Data Scientists**: SOTA accuracy (T-Learner PEHE 1.164 on IHDP) + IV/DiD/RDD/Granger out-of-the-box.
- **For Devs**: 3 lines of code to integrate causal AI into your pipeline.

```python
import whylab

result = whylab.analyze(data, treatment='coupon', outcome='purchase')
print(result.verdict)   # "CAUSAL"
result.summary()        # ATE, CI, Meta-learners, Sensitivity, Debate verdict
```

## What Makes WhyLab Different?

| | DoWhy | EconML | CausalML | **WhyLab** |
|---|:---:|:---:|:---:|:---:|
| Causal Graph Modeling | O | - | - | **O** |
| Meta-Learners (S/T/X/DR/R) | - | O | O | **O** |
| Double Machine Learning | - | O | - | **O** |
| Refutation Tests | O | - | - | **O** |
| **IV / DiD / RDD** | - | △ | - | **O** |
| **Granger / CausalImpact** | - | - | - | **O** |
| **Structural Counterfactual** | - | - | - | **O** |
| **AI Agent Auto-Debate** | - | - | - | **O** |
| **Auto Verdict (CAUSAL/NOT)** | - | - | - | **O** |
| **Auto Discovery (PC + LLM)** | - | - | - | **O** |
| **Interactive Dashboard** | - | - | - | **O** |
| **REST API Server** | - | - | - | **O** |

---

## Architecture: 16-Cell Pipeline

```
Data → Discovery → AutoCausal → Causal → MetaLearner → Conformal →
  Explain → Refutation → Sensitivity →
    QuasiExperimental → TemporalCausal → Counterfactual →
      Viz → Debate → Export → Report
```

| # | Cell | Role |
|:---:|---|---|
| 1 | `DataCell` | SCM-based synthetic data + external CSV/SQL/BigQuery |
| 2 | `DiscoveryCell` | Auto causal graph discovery (PC + LLM hybrid) |
| 3 | `AutoCausalCell` | Data profiling → methodology auto-recommendation |
| 4 | `CausalCell` | DML estimation (Linear/Forest/Sparse) |
| 5 | `MetaLearnerCell` | 5 meta-learners (S/T/X/DR/R) + Oracle ensemble |
| 6 | `ConformalCell` | Distribution-free confidence intervals |
| 7 | `ExplainCell` | SHAP-based feature importance & explanations |
| 8 | `RefutationCell` | Placebo, Bootstrap, Random Cause tests |
| 9 | `SensitivityCell` | E-value, Overlap, GATES analysis |
| 10 | `QuasiExperimentalCell` | **IV (2SLS)**, **DiD** (parallel trend), **Sharp RDD** |
| 11 | `TemporalCausalCell` | **Granger causality**, **CausalImpact**, lag correlation |
| 12 | `CounterfactualCell` | Structural counterfactuals, Manski bounds, ITE ranking |
| 13 | `VizCell` | Publication-ready figures |
| 14 | `DebateCell` | 3-agent LLM debate (Growth Hacker / Risk Manager / PO) |
| 15 | `ExportCell` | JSON serialization + LLM debate results |
| 16 | `ReportCell` | Automated analysis reports |

### Multi-Agent Debate System

Three AI agents simulate real organizational decision-making:

1. **Growth Hacker** (10 evidence types): Finds revenue opportunities from causal signals
2. **Risk Manager** (8 attack vectors): Warns about potential losses and model vulnerabilities
3. **Product Owner (Judge)**: Synthesizes Growth vs Risk → delivers actionable verdict
   - `🚀 Rollout 100%` | `⚖️ A/B Test 5%` | `🛑 Reject`

Supports **LLM-enhanced debate** (Gemini API) with automatic rule-based fallback.

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

### Jobs (LaLonde 1986, n=722, p=8)

| Method | sqrt(PEHE) | ATE Bias |
|---|:---:|:---:|
| **LinearDML** | **170.5 +/- 32.3** | 39.2 +/- 36.6 |
| S-Learner | 288.4 +/- 11.3 | 79.2 +/- 36.8 |
| X-Learner | 377.2 +/- 22.4 | 38.6 +/- 16.3 |
| Ensemble | 381.8 +/- 18.4 | 39.8 +/- 33.8 |
| T-Learner | 482.7 +/- 23.2 | **35.2 +/- 21.7** |
| DR-Learner | 535.0 +/- 29.3 | 34.9 +/- 25.2 |

---

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+ (Dashboard)

### Installation

```bash
# Clone
git clone https://github.com/Yesol-Pilot/WhyLab.git
cd WhyLab

# Python
pip install -e ".[all]"

# Dashboard
cd dashboard; npm install
```

### Usage

#### 1. Python SDK (3 Lines)
```python
import whylab

result = whylab.analyze("data.csv", treatment="T", outcome="Y")
result.summary()
```

#### 2. CLI — Causal Pipeline
```bash
python -m engine.pipeline --scenario A   # Credit limit -> Default
python -m engine.pipeline --scenario B   # Marketing coupon -> Signup
```

#### 3. REST API Server
```bash
# Start
uvicorn whylab.server:app --reload --port 8000

# Analyze
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"treatment": "T", "outcome": "Y", "data_path": "data.csv"}'

# Available methods
curl http://localhost:8000/api/v1/methods
```

#### 4. Connect Your Data (CSV / SQL / BigQuery)
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

#### 5. Ask Questions (RAG Agent)
```bash
python -m engine.cli --query "쿠폰 효과가 있어?" --persona growth_hacker
python -m engine.cli --query "리스크는 없어?" --persona risk_manager
```

#### 6. Run Benchmarks
```bash
python -m engine.pipeline --benchmark ihdp acic jobs \
  --replications 10 --output results/ --latex
```

#### 7. Launch Dashboard
```bash
cd dashboard; npm run dev
# Open http://localhost:4000
```

#### 8. Docker (GPU)
```bash
docker compose up whylab       # Default pipeline
docker compose up benchmark    # Benchmark mode
docker compose up pipeline     # Full pipeline + Debate
```

---

## Project Structure

```
WhyLab/
  engine/
    cells/            # 16 modular analysis cells
    agents/           # AI debate & discovery agents (LLM hybrid)
    connectors/       # Multi-source data (CSV/SQL/BigQuery)
    monitoring/       # Causal drift detection & alerting
    data/             # Benchmark data loaders (IHDP/ACIC/Jobs)
    rag/              # RAG-based Q&A agent (multi-turn, persona)
    server/           # MCP Protocol server (7 tools, 3 resources)
    audit.py          # Governance: analysis audit trail (JSONL)
    config.py         # Central configuration (no magic numbers)
    orchestrator.py   # 16-cell pipeline orchestrator
    cli.py            # CLI entry point
  whylab/
    api.py            # 3-line SDK (analyze → CausalResult)
    server.py         # FastAPI REST API server
  dashboard/          # Next.js interactive dashboard
  tests/              # Unit & integration tests (58+)
  results/            # Benchmark output (JSON + LaTeX)
  .github/workflows/  # CI + Deploy + PyPI Release (OIDC)
```

## Tests

```bash
# All tests
python -m pytest tests/ -v

# Phase-specific
python -m pytest tests/test_phase10.py -v   # IV/DiD/RDD + Temporal + Counterfactual
python -m pytest tests/test_phase11.py -v   # Server + Audit + Version
```

> **58+ tests** | Phase 10: 14 passed | Phase 11: 4 passed, 3 skipped (FastAPI optional)

---

## Citation

If you use WhyLab in your research, please cite:

```bibtex
@software{whylab2026,
  title={WhyLab: Causal Decision Intelligence Engine with Multi-Agent Debate},
  author={Yesol Heo},
  year={2026},
  url={https://github.com/Yesol-Pilot/WhyLab}
}
```

## License

MIT License
