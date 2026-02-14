# Benchmark Reproduction Guide

WhyLab의 벤치마크 결과를 재현하는 방법을 설명합니다.

## 지원 벤치마크

| Dataset | Reference | n | p | 특징 |
|---|---|:---:|:---:|---|
| **IHDP** | Hill 2011 | 747 | 25 | 비선형 Response Surface, 불균형 처치 |
| **ACIC** | Dorie 2019 | 4,802 | 58 | 고차원, 비선형 HTE |
| **Jobs** | LaLonde 1986 | 722 | 8 | 강한 Selection Bias, 소표본 |

## 실행 방법

### 전체 벤치마크 (3 datasets x 10 replications)
```bash
python -m engine.pipeline --benchmark ihdp acic jobs \
  --replications 10 --output results/ --latex
```

### 단일 데이터셋
```bash
python -m engine.pipeline --benchmark ihdp --replications 10
```

### 출력물
- `results/benchmark_results.json` — 전체 수치 데이터
- `results/tables/benchmark_table.tex` — 논문용 LaTeX 비교표
- 콘솔: 마크다운 비교표

## 평가 지표

### sqrt(PEHE) — Precision in Estimation of HTE
```
sqrt(PEHE) = sqrt( (1/n) * sum( (tau_hat(x_i) - tau_true(x_i))^2 ) )
```
CATE 추정치와 Ground Truth 간의 RMSE입니다. **낮을수록 좋습니다.**

### ATE Bias
```
ATE Bias = | mean(tau_hat) - mean(tau_true) |
```
평균 처치 효과의 편향입니다. **0에 가까울수록 좋습니다.**

## 평가 대상 메타러너

| Method | Description |
|---|---|
| S-Learner | Single model with treatment as feature |
| T-Learner | Separate models for treated/control |
| X-Learner | Pseudo-residual + propensity weighting (Kunzel 2019) |
| DR-Learner | Doubly Robust CATE estimation (Kennedy 2023) |
| R-Learner | Robinson Decomposition (Nie & Wager 2021) |
| LinearDML | EconML Double Machine Learning |
| Ensemble | Oracle-weighted ensemble of all learners |

## 기준선 비교 (참고)

| Method | IHDP sqrt(PEHE) | Source |
|---|:---:|---|
| BART | ~1.0 | Hill 2011 |
| GANITE | ~1.9 | Yoon et al. 2018 |
| CEVAE | ~2.7 | Louizos et al. 2017 |
| **WhyLab T-Learner** | **1.16** | This work |
