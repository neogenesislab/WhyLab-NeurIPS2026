# Architecture: Cellular Agents

WhyLab은 생물학적 세포에서 영감받은 **"Cellular Agent"** 아키텍처를 사용합니다.

## 설계 철학

기존 인과추론 라이브러리(DoWhy, EconML, CausalML)는 **"코드를 짜서 분석하는 도구"**입니다.

WhyLab은 다릅니다:
- AI Agent가 **스스로 인과 구조를 발견**하고
- 찬성/반대 에이전트가 **자동으로 토론**하여
- 최종 **인과 판결(CAUSAL/NOT_CAUSAL)**을 내립니다

## 파이프라인 구조

```
DataCell → CausalCell → MetaLearnerCell → ConformalCell → ExplainCell
    ↓                                                         ↓
RefutationCell → SensitivityCell → VizCell → ExportCell → DebateCell
                                                              ↓
                                                       VERDICT (AI 판결)
```

## 셀 (Cell) 설명

각 셀은 독립적인 분석 단위입니다. 하나의 입력을 받아 출력을 생성하며,
`Orchestrator`가 DAG 순서에 따라 조율합니다.

### 데이터 계층
| Cell | 역할 |
|---|---|
| `DataCell` | SCM 기반 합성 데이터 생성 + 외부 CSV 로드 + DuckDB 전처리 |

### 추정 계층
| Cell | 역할 |
|---|---|
| `CausalCell` | DML (Linear/Forest/Auto) ATE 추정 |
| `MetaLearnerCell` | S/T/X/DR/R 5종 메타러너 + Oracle 앙상블 CATE 추정 |
| `ConformalCell` | 분포무가정 신뢰구간 (Conformal Prediction) |

### 검증 계층 (면역계)
| Cell | 역할 |
|---|---|
| `RefutationCell` | Placebo Test, Bootstrap CI, Random Cause 반증 |
| `SensitivityCell` | E-value, Overlap 진단, GATES/CLAN 분석 |

### 설명 계층
| Cell | 역할 |
|---|---|
| `ExplainCell` | SHAP 기반 변수 중요도 + 반사실 시뮬레이션 |

### 출력 계층
| Cell | 역할 |
|---|---|
| `VizCell` | 시각화 (matplotlib) |
| `ExportCell` | Dashboard JSON 내보내기 |
| `ReportCell` | 마크다운/LaTeX 보고서 자동 생성 |

### 판결 계층 (세포핵)
| Cell | 역할 |
|---|---|
| `DebateCell` | 3-Agent Debate → 자동 인과 판결 |

## Multi-Agent Debate 시스템 (Decision Intelligence)

### 에이전트 구성

```
Growth Hacker (성장)        Risk Manager (위험)
  10가지 증거 수집              8가지 공격 벡터
  - 메타러너 합의율             - E-value 취약
  - Bootstrap 유의성            - Overlap 위반
  - ATE CI 비포함 0             - CI 과대
  - E-value 강도                - Placebo 실패
  - Conformal CI                - LOO 부호 반전
  - LOO 안정성                  - 메타러너 불일치
  - Subset 안정성               - Subset 불안정
  - Overlap 양호                - 소표본 경고
  - GATES 이질성
  - SHAP-CATE 정합성
         ↓                         ↓
    비즈니스 기회 해석          비즈니스 리스크 해석
    "매출 +5% 기회"             "예산 낭비 위험"
         ↓                         ↓
         └──── Product Owner ──────┘
                      ↓
             비즈니스 액션 아이템 도출
         🚀 Rollout 100% (확신도 > 90%)
         📈 단계적 배포 (확신도 70~90%)
         ⚖️ A/B Test 5% (UNCERTAIN)
         🛑 기각 (NOT_CAUSAL)
```

### 판결 기준
- **CAUSAL**: 확신도 >= 0.7 (가중 옹호 점수 비율)
- **NOT_CAUSAL**: 확신도 <= 0.3
- **UNCERTAIN**: 그 사이 → 추가 라운드 (최대 3회)

### 비즈니스 영향 번역
| 에이전트 | 역할 | 출력 예시 |
|---|---|---|
| Growth Hacker | 인과 신호 → 매출 기회 | "타겟팅 효율화 기회 발견" |
| Risk Manager | 모델 취약점 → 비용 리스크 | "일반화 시 성과 하락 우려" |
| Product Owner | 종합 판단 → 실행 가능 액션 | "🚀 전면 배포. 예상 수익 +$1.2M" |

## Living Ledger 비전과의 매핑

| Living Ledger 개념 | 코드 구현체 | 설명 |
|---|---|---|
| 세포막 (Membrane) | `BaseCell` + MCP Server | 표준화된 입출력 + 외부 에이전트 연동 |
| 세포핵 (Nucleus) | `DebateCell` + `discovery.py` | LLM + 규칙 기반 하이브리드 두뇌 |
| 미토콘드리아 | DuckDB in `DataCell` | 제로카피 고속 데이터 처리 |
| 면역계 | `RefutationCell` + `SensitivityCell` | 자동 반증 + 견고성 검증 |
| 항상성 | Debate 자동 기각 | UNCERTAIN → 추가 라운드 |
| 신경계 | `discovery.py` | DAG 자동 생성 (LLM + 데이터) |
| 근육계 | `CausalCell` + `MetaLearnerCell` | DML + 5종 메타러너 추정 |
