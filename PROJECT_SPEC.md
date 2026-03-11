# 🔬 WhyLab — PROJECT SPEC

> **최종 업데이트**: 2026-03-11 | **유형**: NeurIPS 2026 제출 논문
> **상위 문서**: [PAPER PROJECT_SPEC](../PROJECT_SPEC.md) | [마스터 바이블](file:///d:/00.test/FOLDER_BIBLE.md)

---

## 개요

| 항목 | 값 |
|------|------|
| **GitHub (private)** | Yesol-Pilot/WhyLab |
| **GitHub (public)** | neogenesislab/WhyLab-NeurIPS2026 |
| **제출 대상** | NeurIPS 2026 (Main Track) |
| **논문 제목** | Causal Audit Framework for Stable Agent Self-Improvement |
| **브랜치** | main |
| **제출 상태** | ✅ 제출 준비 완료 |

---

## 논문 기여 (Contributions)

| ID | 기여 | 핵심 실험 |
|:---|:---|:---|
| C1 | 정보이론 기반 드리프트 탐지 | E1: 40 seeds, KM curves |
| C2 | E-value × RV 민감도 필터링 | E2: 40 seeds, Pareto frontier |
| C3 | Lyapunov 기반 적응형 댐핑 | E3a: 20 seeds × 4 step sizes |

---

## 제출본 상태

| 지표 | 값 | 판정 |
|:---|:---|:---:|
| Content pages | ~7 | ✅ ≤ 9 |
| Undefined refs | 0 | ✅ |
| TODO 마커 | 0 | ✅ |
| 체크리스트 | 9항목 완성 | ✅ |
| PDF | 11p / 477KB / US Letter / Type1 | ✅ |

---

## 구조

```
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
│   └── results/               # 실험 결과 (CSV)
├── README.md                  # 공개 리포 README
├── LICENSE                    # MIT
└── .gitignore
```

---

## 보안 주의사항

- `.env` (Gemini API 키 포함) → .gitignore + git-filter-repo로 history 완전 제거
- 공개 리포에 저자 식별 정보 없음 (double-blind 준수)
- 제출 PDF에 GitHub 링크 없음

---

## 배포

| 채널 | URL | 용도 |
|:---|:---|:---|
| GitHub (public) | neogenesislab/WhyLab-NeurIPS2026 | 코드 공개 |
| Zenodo | (예정) | DOI 발급 + 익명 아카이브 |
| NeurIPS submission | OpenReview (예정) | 논문 제출 |
