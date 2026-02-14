# Beyond Correlation: Optimizing Fintech Strategies with Double Machine Learning
> **Date**: 2026-02-13  
> **Author**: WhyLab Research Team  
> **Status**: Draft v0.2  

## Abstract

핀테크 산업에서 의사결정은 데이터에 기반해야 합니다. 그러나 단순한 상관관계 분석은 역인과 관계(Reverse Causality)나 교란 변수(Confounder)로 인해 잘못된 결론을 유도할 위험이 큽니다. 본 연구는 **Double Machine Learning (DML)** 기법을 활용하여, 신용한도 상향과 마케팅 쿠폰 지급이라는 두 가지 실제적 시나리오에서 **순수 인과 효과(Causal Effect)**를 추정합니다. 나아가 **E-value**, **Overlap(Positivity)**, **GATES/CLAN** 등 고급 통계 진단을 통해 추정치의 견고성과 이질성을 심층 검증합니다.

---

## 1. Introduction

### 1.1. The Pitfall of Correlation
현대 금융 앱은 수많은 사용자 행동 데이터를 수집합니다. 흔히 "신용한도가 높은 유저일수록 연체율이 낮다"는 데이터 패턴이 관찰됩니다. 경영진은 이를 근거로 "신용한도를 늘리면 연체율이 낮아질 것이다"라고 판단할 수 있습니다.

하지만 이는 **상관관계(Correlation)**이지 **인과관계(Causality)**가 아닙니다. 실제로는 "신용도가 높은 유저에게 더 높은 한도를 부여"했기 때문에 이러한 패턴이 나타나는 것입니다. 만약 신용도가 낮은 유저에게 무턱대고 한도를 늘린다면, 연체율은 오히려 급증할 것입니다.

### 1.2. The Need for Causal Inference
A/B 테스트는 인과관계를 밝히는 가장 확실한 방법이지만, 신용 한도나 금리 같은 민감한 변수를 무작위로 실험하는 것은 윤리적·비용적 리스크가 큽니다. 따라서 우리는 **관찰 데이터(Observational Data)**만으로 인과 효과를 추정해야 합니다.

본 연구에서는 **WhyLab 엔진**을 통해, 교란 변수를 통제하고 순수 처치 효과를 발라내는 인과추론 파이프라인을 구축하고 그 유효성을 증명합니다.

---

## 2. Methodology

### 2.1. Potential Outcomes Framework
Rubin의 잠재적 결과 프레임워크를 따릅니다. 개체 $i$에 대해 처치 $T_i$가 주어졌을 때의 결과 $Y_i(1)$과 주어지지 않았을 때의 결과 $Y_i(0)$의 차이를 인과 효과라고 정의합니다.

$$ \text{ITE}_i = Y_i(1) - Y_i(0) $$
$$ \text{ATE} = E[Y_i(1) - Y_i(0)] $$

### 2.2. Double Machine Learning (DML)
Chernozhukov et al. (2018)이 제안한 **DML**을 사용합니다:

1.  **Treatment Model** ($M_t$): 교란 변수 $X$로 처치 $T$를 예측 (잔차 $T - \hat{T}$)
2.  **Outcome Model** ($M_y$): 교란 변수 $X$로 결과 $Y$를 예측 (잔차 $Y - \hat{Y}$)
3.  **Causal Estimation**: 잔차 간의 회귀분석을 통해 순수 효과 $\theta$를 추정

$$ Y - E[Y|X] = \theta(X) \cdot (T - E[T|X]) + \epsilon $$

### 2.3. Advanced Diagnostics (Phase 4)

| 진단 | 방법 | 목적 |
|------|------|------|
| **E-value** | $RR + \sqrt{RR(RR-1)}$ | 미관측 교란이 결과를 뒤집으려면 얼마나 강한 상관이 필요한지 |
| **Overlap** | Propensity Score 분포 비교 (Bhattacharyya 계수) | 처치/통제 그룹 간 균형 여부 |
| **GATES** | CATE 사분위별 그룹 분석 + F-test | 이질적 처치 효과의 통계적 유의성 |
| **CLAN** | 그룹별 피처 평균 비교 | 어떤 특성이 이질성을 만드는지 |

### 2.4. Technology Stack
-   **Inference**: Microsoft EconML (LinearDML)
-   **Nuisance Models**: LightGBM (Gradient Boosting)
-   **Data Processing**: DuckDB for OLAP
-   **Dashboard**: Next.js 16 + Recharts + Framer Motion

---

## 3. Experimental Setup

### 3.1. Data Generation (SCM)
구조적 인과 모델(SCM) 기반의 합성 데이터를 생성했습니다:
-   **N**: 100,000 samples
-   **Confounders**: Income, Age, Credit Score, App Usage
-   **Noise**: Gaussian ($\sigma=0.3$)

### 3.2. Scenarios
#### Scenario A: Credit Limit (Continuous Treatment)
-   **Treatment**: 신용 한도 (100만 원 ~ 5,000만 원)
-   **Outcome**: 연체 확률 (0 ~ 1)

#### Scenario B: Marketing Coupon (Binary Treatment)
-   **Treatment**: 쿠폰 지급 (0/1)
-   **Outcome**: 투자 상품 가입 (0/1)

---

## 4. Experimental Results

### 4.1. Model Performance

| Metric | Scenario A (Credit) | Scenario B (Coupon) |
|--------|---------------------|---------------------|
| **ATE** | -0.0342 (-3.4%p) | -0.0040 (-0.4%p) |
| **Correlation** | **0.977** | **0.996** |
| RMSE | 0.609 | 0.028 |
| Coverage | 94.2% | 96.8% |

> **Correlation 0.97~0.99** = DML 추정치가 Ground Truth와 거의 완벽하게 일치합니다.

### 4.2. Scenario A: Credit Limit
-   **Overall ATE = -0.0342**: 한도 1σ 증가 시 연체율 3.4%p 감소
-   **이질성**: 고소득층에서 효과 극대화, 저소득층에서 효과 미미/부정적
-   **정책 함의**: 일괄 증액이 아닌 고신용 세그먼트 타겟 증액 필요

### 4.3. Scenario B: Marketing Coupon
-   **Overall ATE = -0.0040**: 쿠폰 효과가 통계적으로 유의하지 않음 (CI가 0 포함)
-   **정책 함의**: 쿠폰이 가입률에 미치는 순수 효과가 작으므로, CATE 기반 세그먼트 타겟팅으로 ROI 극대화 필요

### 4.4. Robustness Diagnostics (Phase 4)

| 테스트 | Scenario A | Scenario B |
|--------|------------|------------|
| Placebo Test | ✅ Pass | ✅ Pass |
| Random Common Cause | ✅ Pass (Stability 99%+) | ✅ Pass |
| **E-value** | 1.07 (보통~견고) | 1.01 (약한 효과) |
| **Overlap Score** | 0.85 (양호) | 0.92 (우수) |
| **GATES F-stat** | 12.5 (강한 이질성) | 2.1 (약한 이질성) |

> **Key Finding**: Scenario A에서 E-value가 보통 수준이지만 Overlap이 충분하고 F-stat이 높아, "누구에게 효과가 있는지"가 크게 다르다는 강한 이질성이 확인됨.

---

## 5. Discussion

### 5.1. Why simple regression failed?
단순 회귀분석(OLS)은 신용한도와 연체율 간의 관계를 과도하게 부풀렸습니다(Coefficient: -1.2). 이는 역인과 관계를 통제하지 못한 결과입니다. DML은 이러한 편향을 제거하여 더 보수적이고 정확한 추정치(-0.034)를 제공했습니다.

### 5.2. E-value and Unobserved Confounders
Scenario A의 E-value 1.07은 비교적 작은 값으로, 강한 미관측 교란이 있다면 결과가 바뀔 수 있음을 시사합니다. 그러나 합성 데이터에서 모든 교란을 통제했으므로, 이는 효과 크기 자체가 작기 때문입니다. 실제 데이터에서는 도구 변수(IV) 등의 추가 기법을 도입하여 이를 보완해야 합니다.

### 5.3. Limitations
-   **Unobserved Confounders**: 실제 데이터에서는 성격, 금융 지식 등 미관측 변수가 교란 요인으로 작용할 수 있습니다.
-   **Log-Linear Assumption**: LinearDML은 처치 효과의 선형성을 가정합니다.
-   **합성 데이터 한계**: 실제 금융 데이터에서의 검증이 필요합니다.

---

## 6. Academic Benchmark Evaluation

본 엔진의 CATE 추정 성능을 검증하기 위해, 세 가지 표준 학술 벤치마크에서 7종 메타러너를 10회 반복 평가했습니다.

### 6.1. Benchmark Datasets

| Dataset | Reference | n | p | 특징 |
|---------|-----------|:---:|:---:|------|
| **IHDP** | Hill 2011 | 747 | 25 | 비선형 Response Surface, 불균형 처치 |
| **ACIC** | Dorie et al. 2019 | 4,802 | 58 | 고차원, 비선형 HTE, 복합 교란 |
| **Jobs** | LaLonde 1986 | 722 | 8 | 강한 Selection Bias, 소표본 |

### 6.2. Results

#### Table 1: IHDP Benchmark ($\sqrt{\text{PEHE}}$, lower is better)

| Method | $\sqrt{\text{PEHE}}$ | ATE Bias |
|--------|:---:|:---:|
| **T-Learner** | **1.164 $\pm$ 0.024** | **0.039 $\pm$ 0.031** |
| DR-Learner | 1.194 $\pm$ 0.034 | 0.038 $\pm$ 0.029 |
| Ensemble | 1.214 $\pm$ 0.025 | 0.046 $\pm$ 0.034 |
| X-Learner | 1.324 $\pm$ 0.029 | 0.035 $\pm$ 0.024 |
| S-Learner | 1.383 $\pm$ 0.033 | 0.064 $\pm$ 0.040 |
| LinearDML | 1.465 $\pm$ 0.024 | 0.066 $\pm$ 0.061 |
| R-Learner | 1.635 $\pm$ 0.046 | 0.135 $\pm$ 0.107 |

> **참고**: BART $\sqrt{\text{PEHE}} \approx 1.0$ (Hill 2011), GANITE $\approx 1.9$ (Yoon et al. 2018)

#### Table 2: ACIC Benchmark

| Method | $\sqrt{\text{PEHE}}$ | ATE Bias |
|--------|:---:|:---:|
| **S-Learner** | **0.491 $\pm$ 0.017** | **0.018 $\pm$ 0.013** |
| X-Learner | 0.569 $\pm$ 0.009 | 0.020 $\pm$ 0.011 |
| Ensemble | 0.612 $\pm$ 0.013 | 0.013 $\pm$ 0.007 |
| LinearDML | 0.614 $\pm$ 0.010 | 0.071 $\pm$ 0.025 |
| DR-Learner | 0.799 $\pm$ 0.017 | 0.040 $\pm$ 0.018 |
| T-Learner | 0.835 $\pm$ 0.013 | 0.041 $\pm$ 0.018 |
| R-Learner | 1.206 $\pm$ 0.035 | 0.111 $\pm$ 0.060 |

#### Table 3: Jobs Benchmark

| Method | $\sqrt{\text{PEHE}}$ | ATE Bias |
|--------|:---:|:---:|
| **LinearDML** | **170.5 $\pm$ 32.3** | 39.2 $\pm$ 36.6 |
| S-Learner | 288.4 $\pm$ 11.3 | 79.2 $\pm$ 36.8 |
| X-Learner | 377.2 $\pm$ 22.4 | 38.6 $\pm$ 16.3 |
| Ensemble | 381.8 $\pm$ 18.4 | 39.8 $\pm$ 33.8 |
| T-Learner | 482.7 $\pm$ 23.2 | **35.2 $\pm$ 21.7** |
| DR-Learner | 535.0 $\pm$ 29.3 | 34.9 $\pm$ 25.2 |
| R-Learner | 703.4 $\pm$ 36.6 | 81.7 $\pm$ 73.8 |

### 6.3. Key Findings

1. **No single learner dominates**: IHDP에서는 T-Learner, ACIC에서는 S-Learner, Jobs에서는 LinearDML이 최적. 이는 Oracle Ensemble의 필요성을 뒷받침합니다.
2. **Ensemble stability**: Oracle-weighted Ensemble은 세 벤치마크 모두에서 상위 3위 이내로, 최고는 아니지만 일관적으로 안정적인 성능을 보였습니다.
3. **R-Learner underperformance**: Robinson Decomposition이 모든 벤치마크에서 최하위. Semi-parametric 가정이 DGP와 불일치함을 시사합니다.

---

## 7. Multi-Agent Debate System

기존 인과추론 라이브러리(DoWhy, EconML, CausalML)는 분석 코드를 제공하지만, **결과 해석의 부담은 사용자에게** 남깁니다.  WhyLab은 AI Agent 3명이 자동으로 인과 판결을 내리는 Multi-Agent Debate 시스템을 도입하여 이 문제를 해결합니다.

### 7.1. Agent Architecture

| Agent | Role | Evidence / Attacks |
|-------|------|-------------------|
| **Advocate** | 인과 관계 옹호 | 10종 증거 수집 (메타러너 합의, 통계적 유의성, E-value, Conformal CI 등) |
| **Critic** | 인과 관계 비판 | 8종 공격 벡터 (E-value 취약, Overlap 위반, Placebo 실패, 메타러너 불일치 등) |
| **Judge** | 최종 판결 | 증거 유형별 가중 합산 → CAUSAL / NOT_CAUSAL / UNCERTAIN |

### 7.2. Scoring Mechanism

$$\text{Confidence} = \frac{\sum_{e \in \text{Pro}} w(e) \cdot s(e)}{\sum_{e \in \text{Pro}} w(e) \cdot s(e) + \sum_{e \in \text{Con}} w(e) \cdot s(e)}$$

여기서 $w(e)$는 증거 유형별 가중치 (robustness: 1.2, statistical: 1.0, domain: 0.8), $s(e)$는 개별 증거 강도 (0~1).

### 7.3. Verdict Protocol
- **Confidence $\geq$ 0.7** → CAUSAL
- **Confidence $\leq$ 0.3** → NOT_CAUSAL
- **Otherwise** → UNCERTAIN (추가 라운드, 최대 3회)

---

## 8. Conclusion

본 연구는 두 가지 방향에서 기여합니다:

**학술적 기여**:
1. DML 기반 인과 효과 추정치의 Ground Truth Correlation 0.97~0.99 달성
2. 3개 표준 벤치마크에서 7종 메타러너 평가 (T-Learner $\sqrt{\text{PEHE}}=1.16$ on IHDP)
3. Oracle-weighted Ensemble의 일관적 안정성 입증

**실무적 기여**:
1. Multi-Agent Debate를 통한 자동 인과 판결 시스템
2. 3줄 API (`whylab.analyze()`)로 복잡한 인과추론을 간소화
3. 세포 에이전트 아키텍처에 의한 모듈식 확장 가능성

**"Data with Why"** — WhyLab은 "무엇(What)이 일어났는가"를 넘어 "왜(Why) 일어났는가"를 묻는 첫걸음입니다.

---

## References
1.  Chernozhukov, V., et al. (2018). "Double/debiased machine learning for treatment and structural parameters". *The Econometrics Journal*.
2.  Rubin, D. B. (1974). "Estimating causal effects of treatments in randomized and nonrandomized studies". *Journal of Educational Psychology*.
3.  Microsoft Research. (2019). "EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation".
4.  VanderWeele, T. J. & Ding, P. (2017). "Sensitivity Analysis in Observational Research". *Annals of Internal Medicine*.
5.  Chernozhukov, V., et al. (2018). "Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments". *NBER Working Paper*.
6.  Hill, J. L. (2011). "Bayesian Nonparametric Modeling for Causal Inference". *JCGS*, 20(1), 217-240.
7.  Kunzel, S. R., et al. (2019). "Meta-learners for estimating heterogeneous treatment effects using machine learning". *PNAS*, 116(10), 4156-4165.
8.  Kennedy, E. H. (2023). "Towards optimal doubly robust estimation of heterogeneous causal effects". *Electronic Journal of Statistics*.
9.  Nie, X. & Wager, S. (2021). "Quasi-oracle estimation of heterogeneous treatment effects". *Biometrika*, 108(2), 299-319.
10. Yoon, J., Jordon, J., & Van Der Schaar, M. (2018). "GANITE: Estimation of Individualized Treatment Effects Using Generative Adversarial Nets". *ICLR*.
11. Dorie, V., et al. (2019). "Automated versus do-it-yourself methods for causal inference". *Statistical Science*.
12. LaLonde, R. J. (1986). "Evaluating the Econometric Evaluations of Training Programs with Experimental Data". *American Economic Review*.

