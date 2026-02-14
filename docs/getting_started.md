# Getting Started

WhyLab을 5분 안에 시작하는 가이드입니다.

## 설치

```bash
git clone https://github.com/your-org/whylab.git
cd whylab
pip install -e .
```

## 1. Quick Analysis (3줄 코드)

```python
import whylab

result = whylab.analyze("your_data.csv", treatment="T", outcome="Y")
result.summary()
```

출력 예시:
```
============================================================
  WhyLab Causal Analysis Result
============================================================
  ATE (Average Treatment Effect): -0.0234
  95% CI: [-0.0456, -0.0012]
  AI Verdict: CAUSAL (confidence: 82.3%)
------------------------------------------------------------
  Meta-Learner Results:
    S-Learner: ATE = -0.0198
    T-Learner: ATE = -0.0267
    X-Learner: ATE = -0.0241
    DR-Learner: ATE = -0.0225
    R-Learner: ATE = -0.0219
------------------------------------------------------------
  E-value: 2.34
  Placebo Test: PASS
------------------------------------------------------------
============================================================
```

## 2. CLI 사용법

### 합성 데이터로 분석
```bash
# 시나리오 A: 신용한도 상향 → 연체율
python -m engine.main --scenario A

# 시나리오 B: 마케팅 쿠폰 → 가입률
python -m engine.main --scenario B
```

### 외부 CSV 데이터 분석
```bash
python -m engine.main \
  --data "your_data.csv" \
  --treatment "treatment_col" \
  --outcome "outcome_col" \
  --features "age,income,score"
```

### 벤치마크 실행
```bash
python -m engine.pipeline --benchmark ihdp acic jobs \
  --replications 10 --output results/ --latex
```

## 3. 대시보드

```bash
cd dashboard
npm install
npm run dev
# http://localhost:3004 에서 확인
```

## 다음 단계

- [벤치마크 재현 가이드](benchmark.md)
- [아키텍처 설명](architecture.md)
