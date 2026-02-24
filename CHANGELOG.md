# Changelog

All notable changes to WhyLab are documented in this file.

## [1.0.0] - 2026-02-24

### Added
- **MCP Server v2**: 7 tools + 3 resources for external agent integration (Claude Desktop, etc.)
- **Policy Simulator**: `simulate_intervention()` — treatment intensity & target ratio based ROI prediction
- **Agent Evolution System**: Sprint 15 generational agent evolution with strategy memory
- **Research Cycle Dashboard**: Sprint 16 autonomous research history aggregation
- **Auto Research Report**: Sprint 17 automated report generation
- **Autopilot Mode**: Server auto-starts autonomous research cycle on boot
- **Dose Response Cell**: Continuous treatment effect estimation
- **Deep CATE Cell**: Deep learning-based CATE estimation
- **Fairness Audit Cell**: Algorithmic fairness assessment across protected groups
- **Log Rotation**: Automatic log management and rotation

### Changed
- Pipeline expanded from 16 to **22 cells**
- Agent system expanded to **11 modules** (debate, discovery, architect, director, etc.)
- Dashboard Backend (`api/main.py`) upgraded with agent management, evolution, and knowledge graph endpoints
- Docker Compose now includes Next.js dashboard service for full-stack local development

### Infrastructure
- CI enforces **80% minimum coverage gate** (`--cov-fail-under=80`)
- Version badge aligned across README and `pyproject.toml`

## [0.2.0] - 2026-02-14

### Added
- **Multi-Source Data Connectors**: CSV, Parquet, TSV, Excel, SQL (MySQL/PostgreSQL/SQLite), BigQuery
- **Real-time Causal Monitoring**: DriftDetector (ATE/KL-Div/Sign-flip), Alerter (Console + Slack), Scheduler
- **RAG Agent v2**: Multi-turn conversations, persona support (Growth Hacker / Risk Manager / Product Owner), auto-analysis trigger
- **MCP Server v2**: 7 tools + 3 resources for external agent integration
- **CATE Explorer**: Interactive segment-level treatment effect visualization (Dashboard)
- **CLI v3**: `--monitor`, `--source-type`, `--db-query`, `--persona` flags

### Changed
- `DataCell` now auto-detects data source type from URI patterns (SQL/BigQuery/Parquet/etc.)
- README updated with DB connector, RAG, and monitoring usage examples
- Project structure expanded: `engine/connectors/`, `engine/monitoring/`

### Tests
- Added 33 new tests for connectors and monitoring packages (55+ total)

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
