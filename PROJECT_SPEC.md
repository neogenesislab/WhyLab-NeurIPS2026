# 🔬 WhyLab — PROJECT SPEC

> **최종 업데이트**: 2026-03-11 | **유형**: NeurIPS 2026 제출 논문
> **상위 문서**: [PAPER PROJECT_SPEC](../PROJECT_SPEC.md) | [마스터 바이블](file:///d:/00.test/FOLDER_BIBLE.md)

---

## 개요

| 항목 | 값 |
|------|------|
| **GitHub (private)** | Yesol-Pilot/WhyLab |
| **GitHub (public)** | neogenesislab/WhyLab-NeurIPS2026 |
| **Zenodo DOI** | 10.5281/zenodo.18948929 |
| **제출 대상** | NeurIPS 2026 (Main Track) |
| **논문 제목** | Causal Audit Framework for Stable Agent Self-Improvement |
| **브랜치** | main |
| **제출 상태** | ✅ 제출 준비 완료 |

---

## 논문 기여 (Contributions)

| ID | 기여 | 핵심 실험 | 선행연구 접점 |
|:---|:---|:---|:---|
| C1 | 정보이론 기반 드리프트 탐지 | E1: 40 seeds, KM curves | ADWIN (Bifet & Gavaldà 2007) |
| C2 | E-value × RV 민감도 필터링 | E2: 40 seeds, Pareto frontier | E-value (VanderWeele & Ding 2017), OVB (Cinelli & Hazlett 2020) |
| C3 | Lyapunov 기반 적응형 댐핑 | E3a: 20 seeds × 4 step sizes | Safe RL (Chow et al. 2018, Berkenkamp et al. 2017) |

---

## 제출본 상태

| 지표 | 값 | 판정 |
|:---|:---|:---:|
| Content pages | ~7 | ✅ ≤ 9 |
| Undefined refs | 0 | ✅ |
| TODO 마커 | 0 | ✅ |
| 체크리스트 | 9항목 완성 | ✅ |
| PDF | 11p / 477KB / US Letter / Type1 | ✅ |
| Double-blind | main.tex 식별정보 0건 | ✅ |
| 익명 리뷰팩 | WhyLab_NeurIPS2026_anonymous.zip (19.53 MB) | ✅ |

---

## 논문-코드 정합성 알려진 차이 (Reproducibility Notes)

> 리뷰어 방어용으로 `README_ANON.md`에도 투명하게 명시되어 있음.

| 항목 | 논문 | 코드 | 방어 논거 |
|:---|:---|:---|:---|
| E1 K (스트림 수) | K=5 (이론 최대) | K=3 (`config.yaml`) | 안정적 재현을 위한 고정 세팅 |
| E1 Binning | Sturges rule | N_BINS=10 고정 | 결과 수치에 실질적 영향 없음 |
| E2 RV 부호 | RV_q ≥ RV_min | RV ≤ threshold (residual variance proxy) | 수학적 동치, 부호 반전 |
| E3a EMA | 2중 EMA (m̂₂ + ζ̄) | 단일 m̂₂ EMA | 동일 Lyapunov 수렴 보장 |

---

## 구조

```text
WhyLab/
├── paper/                      # LaTeX 소스 + 컴파일된 PDF
│   ├── main.tex               ← 메인 논문
│   ├── main.pdf               ← 제출용 PDF
│   ├── references.bib
│   └── neurips_2025.sty
├── experiments/                # 실험 스크립트
│   ├── e1_drift_detection.py  # E1: 드리프트 탐지 (C1)
│   ├── e2_sensitivity_filter.py # E2: 민감도 필터 (C2)
│   ├── e3a_stationary.py      # E3a: 정상 안정성 (C3)
│   ├── e3b_heavy_tail.py      # E3b: 헤비테일 스트레스
│   ├── config.yaml            # 공유 하이퍼파라미터
│   ├── figures/               # 생성된 그림 (PDF + PNG)
│   └── results/               # 실험 결과 (CSV, 커밋됨)
├── scripts/
│   ├── zenodo_upload.py       # Zenodo DOI 발급 스크립트
│   └── package_anonymous.py   # 리뷰용 익명 ZIP 패키징
├── README.md                  # 공개 리포 README (DOI 뱃지 포함)
├── README_ANON.md             # 리뷰용 익명 README (식별정보 제거)
├── LICENSE                    # MIT
├── CITATION.cff               # 인용 메타데이터
└── .gitignore
```

---

## 보안 주의사항

- `.env` (Gemini API 키 포함) → .gitignore + `git-filter-repo`로 history 완전 제거 완료
- 공개 리포에 저자 식별 정보 없음 (double-blind 준수)
- 제출 PDF에 GitHub 링크 없음
- ⚠️ Gemini API 키 (`AIzaSyC3_...`) rotation 권장 (history에서 제거됨)

---

## 배포

| 채널 | URL | 용도 |
|:---|:---|:---|
| GitHub (public) | neogenesislab/WhyLab-NeurIPS2026 | 코드 공개 |
| GitHub (private) | Yesol-Pilot/WhyLab | 본진 (clean history) |
| Zenodo | doi.org/10.5281/zenodo.18948929 | DOI 발급 + PDF 아카이브 |
| NeurIPS submission | OpenReview (예정) | 논문 제출 |

---

## 리포지토리 품질 평가 (2026-03-11 정밀 감사)

### 평점 요약

| 항목 | 논문 아티팩트 트랙 | 플랫폼 트랙 |
|:---|:---:|:---:|
| 재현성 | 3.5/5 | 2.5/5 |
| 의존성/환경 명세 | 2.5/5 | 3/5 |
| 코드 일관성 | 3/5 | 1.5/5 |
| 과학적 검증 스펙 | 3/5 | 2.5/5 |
| 리뷰 친화성 | 3/5 | 2/5 |

### 핵심 강점

- 원시 결과 CSV 커밋 → 표/그림 근거 즉시 확인 가능
- 문제 분해(드리프트/취약 수용/발산 업데이트) 3중 방어 구조 명확
- 단일 스크립트 재현 형태

### 개선 우선순위

| 우선 | 작업 | 상태 |
|:---|:---|:---:|
| P0 | 익명 zip에서 식별 파일 제거 | ✅ 완료 |
| P0 | E2 RV 부호/정의 문서화 | ✅ 완료 |
| P1 | E1 K/binning 차이 문서화 | ✅ 완료 |
| P1 | E3a EMA 구조 차이 문서화 | ✅ 완료 |
| P2 | 실제 에이전트 벤치마크 검증 (ReAct류) | ⬜ 미착수 |
| P2 | 플랫폼 트랙 스키마/버전 정합성 복구 | ⬜ 미착수 |
