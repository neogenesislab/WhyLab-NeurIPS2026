# -*- coding: utf-8 -*-
"""CausalLoopAgent — 반복 자기교정 인과 발견 에이전트.

가설 → 검증 → 반증 → 수정 순환 워크플로.
LLM이 가설을 생성하고, 통계적 방법이 검증하고,
불일치 시 에이전트가 가설을 수정하여 반복합니다.

Sprint 2 확장: LLM 실연동 + Reflexion 메모리 패턴.
- LLM 연동: GeminiClient를 통해 인과 가설을 지능적으로 생성
- Reflexion: 이전 반복의 성공/실패를 명시적 메모리로 축적

학술 참조:
  - Shinn et al. (2023). "Reflexion: Language Agents with Verbal
    Reinforcement Learning." NeurIPS.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CausalHypothesis:
    """인과 가설."""
    edges: List[Tuple[str, str]]  # (원인, 결과) 쌍 리스트
    confidence: float = 0.0
    rationale: str = ""
    iteration: int = 0


@dataclass
class ReflexionEntry:
    """Reflexion 메모리 항목."""
    iteration: int
    hypothesis_summary: str
    validated_edges: List[Tuple[str, str]]
    rejected_edges: List[Tuple[str, str]]
    lesson: str  # LLM이 생성한 교훈 또는 규칙 기반 교훈


@dataclass
class LoopState:
    """반복 상태."""
    hypotheses: List[CausalHypothesis] = field(default_factory=list)
    validations: List[Dict[str, Any]] = field(default_factory=list)
    refutations: List[Dict[str, Any]] = field(default_factory=list)
    reflexion_memory: List[ReflexionEntry] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    final_dag: List[Tuple[str, str]] = field(default_factory=list)
    llm_used: bool = False


class CausalLoopAgent:
    """반복 자기교정 인과 발견 에이전트.

    CausalLoop 프로세스:
    1. **가설 생성** (Hypothesize): LLM 또는 PC 알고리즘으로 초기 DAG 가설 생성
    2. **검증** (Validate): 조건부 독립 검정 + 상관 분석으로 가설 검증
    3. **반증** (Refute): 반증 증거(역방향 인과, 숨겨진 교란) 탐색
    4. **수정** (Revise): 불일치를 기반으로 가설 수정
    5. **수렴 판단**: 수정 없으면 수렴, 아니면 1로 복귀

    Args:
        max_iterations: 최대 반복 횟수.
        convergence_threshold: 수렴 판단 임계값 (변경된 엣지 비율).
        significance_level: 통계 검정 유의 수준.
        use_llm: LLM 가설 생성 활성화 여부.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.05,
        significance_level: float = 0.05,
        use_llm: bool = True,
    ) -> None:
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.significance_level = significance_level
        self.use_llm = use_llm
        self._llm_client = None

    def _get_llm(self):
        """LLM 클라이언트를 지연 로드합니다."""
        if self._llm_client is None and self.use_llm:
            try:
                from engine.agents.llm_adapter import GeminiClient
                self._llm_client = GeminiClient()
                if not self._llm_client.is_available():
                    logger.info("LLM API 미사용 (키 없음). 규칙 기반 모드.")
                    self._llm_client = None
            except Exception as e:
                logger.warning("LLM 로드 실패: %s. 규칙 기반 모드.", e)
                self._llm_client = None
        return self._llm_client

    def run(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        features: Optional[List[str]] = None,
    ) -> LoopState:
        """CausalLoop를 실행합니다.

        Args:
            df: 분석 대상 데이터프레임.
            treatment: 처치 변수명.
            outcome: 결과 변수명.
            features: 공변량 리스트 (None이면 자동 감지).

        Returns:
            LoopState: 반복 결과.
        """
        if features is None:
            features = [c for c in df.columns if c not in [treatment, outcome]]

        all_vars = features + [treatment, outcome]
        state = LoopState()

        logger.info("🔄 CausalLoop 시작 — 변수 %d개, 최대 %d회 반복", len(all_vars), self.max_iterations)

        for iteration in range(1, self.max_iterations + 1):
            state.iterations = iteration

            # ──── 1. 가설 생성 (LLM 또는 규칙 기반) ────
            hypothesis = self._hypothesize(df, all_vars, treatment, outcome, state)
            state.hypotheses.append(hypothesis)

            # ──── 2. 검증 ────
            validation = self._validate(df, hypothesis, all_vars)
            state.validations.append(validation)

            # ──── 3. 반증 ────
            refutation = self._refute(df, hypothesis, treatment, outcome, features)
            state.refutations.append(refutation)

            # ──── 4. Reflexion 메모리 축적 ────
            reflexion = self._reflect(
                hypothesis, validation, refutation, iteration,
            )
            state.reflexion_memory.append(reflexion)

            # ──── 5. 수렴 판단 ────
            if self._check_convergence(state, hypothesis, validation, refutation):
                state.converged = True
                state.final_dag = hypothesis.edges
                logger.info(
                    "✅ CausalLoop 수렴 (반복 %d회) — 엣지 %d개, LLM=%s",
                    iteration, len(hypothesis.edges),
                    "ON" if state.llm_used else "OFF",
                )
                break

            # ──── 6. 수정 (다음 반복에서 가설 생성 시 반영) ────
            logger.info(
                "🔁 반복 %d: 수정 필요 — 기각 %d개, 신규 후보 %d개, 교훈: %s",
                iteration,
                refutation.get("rejected_count", 0),
                refutation.get("new_candidate_count", 0),
                reflexion.lesson[:80],
            )

        if not state.converged:
            # 최대 반복 도달 — 마지막 가설 사용
            state.final_dag = state.hypotheses[-1].edges
            logger.warning("⚠️ CausalLoop 최대 반복 도달. 마지막 가설 사용.")

        return state

    def _hypothesize(
        self,
        df: pd.DataFrame,
        all_vars: List[str],
        treatment: str,
        outcome: str,
        state: LoopState,
    ) -> CausalHypothesis:
        """인과 가설을 생성합니다.

        LLM 사용 가능 시: LLM에게 통계적 요약 + Reflexion 메모리를 전달.
        LLM 미사용 시: 기존 상관관계 기반 규칙.
        """
        iteration = state.iterations

        # LLM 가설 생성 시도
        llm = self._get_llm()
        if llm is not None:
            try:
                llm_hypothesis = self._hypothesize_with_llm(
                    df, all_vars, treatment, outcome, state, llm,
                )
                if llm_hypothesis is not None:
                    state.llm_used = True
                    return llm_hypothesis
            except Exception as e:
                logger.warning("LLM 가설 생성 실패: %s. 규칙 기반 fallback.", e)

        # Fallback: 규칙 기반 가설 생성
        return self._hypothesize_rule_based(df, all_vars, treatment, outcome, state)

    def _hypothesize_with_llm(
        self,
        df: pd.DataFrame,
        all_vars: List[str],
        treatment: str,
        outcome: str,
        state: LoopState,
        llm_client,
    ) -> Optional[CausalHypothesis]:
        """LLM 기반 인과 가설 생성.

        프롬프트 구성:
        1. 변수 목록 + 기술통계 요약
        2. 상관행렬 (상위 관계만)
        3. Reflexion 메모리 (이전 교훈들)
        4. 가설 출력 형식 지정 (JSON)
        """
        iteration = state.iterations
        numeric_cols = [c for c in all_vars if df[c].dtype in [np.float64, np.int64, float, int]]

        # 통계 요약 생성
        stats_summary = ""
        if numeric_cols:
            desc = df[numeric_cols].describe().round(3).to_string()
            corr = df[numeric_cols].corr().round(3)
            # 상관계수 상위 10개만
            corr_pairs = []
            for i, v1 in enumerate(numeric_cols):
                for v2 in numeric_cols[i + 1:]:
                    r = corr.loc[v1, v2]
                    corr_pairs.append((v1, v2, r))
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_corr = corr_pairs[:10]
            corr_str = "\n".join(f"  {v1} ↔ {v2}: r={r:.3f}" for v1, v2, r in top_corr)
            stats_summary = f"기술통계:\n{desc}\n\n상위 상관관계:\n{corr_str}"

        # Reflexion 메모리 구성
        reflexion_str = ""
        if state.reflexion_memory:
            entries = []
            for entry in state.reflexion_memory[-3:]:  # 최근 3개만
                entries.append(
                    f"  반복 {entry.iteration}: "
                    f"검증됨={entry.validated_edges}, "
                    f"기각됨={entry.rejected_edges}, "
                    f"교훈={entry.lesson}"
                )
            reflexion_str = "\n과거 교훈 (Reflexion Memory):\n" + "\n".join(entries)

        prompt = f"""당신은 인과추론 전문가입니다. 아래 데이터를 분석하여 인과 그래프(DAG) 가설을 제안하세요.

## 설정
- 처치 변수: {treatment}
- 결과 변수: {outcome}
- 전체 변수: {all_vars}
- 반복: {iteration}/{self.max_iterations}

## 데이터 통계
{stats_summary}
{reflexion_str}

## 요구 사항
1. 인과적으로 타당한 방향의 엣지만 제안하세요 (상관 ≠ 인과).
2. 교란 변수(Confounder)가 있다면 적절한 방향으로 연결하세요.
3. Reflexion 메모리의 교훈을 반드시 반영하세요.

## 출력 형식 (JSON)
```json
{{
  "edges": [["원인1", "결과1"], ["원인2", "결과2"]],
  "confidence": 0.7,
  "rationale": "이 가설을 제안한 이유"
}}
```

JSON만 출력하세요."""

        response = llm_client.generate(prompt, max_tokens=1024)
        if response is None:
            return None

        # JSON 파싱
        try:
            # 코드 블록 내 JSON 추출
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(response)

            edges = [tuple(e) for e in parsed.get("edges", [])]
            confidence = float(parsed.get("confidence", 0.5))
            rationale = parsed.get("rationale", "LLM 생성 가설")

            logger.info("🤖 LLM 가설: 엣지 %d개, 확신도 %.2f", len(edges), confidence)
            return CausalHypothesis(
                edges=edges,
                confidence=confidence,
                rationale=f"LLM 반복 {iteration}: {rationale}",
                iteration=iteration,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("LLM 응답 파싱 실패: %s", e)
            return None

    def _hypothesize_rule_based(
        self,
        df: pd.DataFrame,
        all_vars: List[str],
        treatment: str,
        outcome: str,
        state: LoopState,
    ) -> CausalHypothesis:
        """규칙 기반 인과 가설 생성 (기존 로직)."""
        iteration = state.iterations
        numeric_cols = [c for c in all_vars if df[c].dtype in [np.float64, np.int64, float, int]]

        if not numeric_cols:
            return CausalHypothesis(edges=[(treatment, outcome)], iteration=iteration)

        corr_matrix = df[numeric_cols].corr().abs()
        edges = []

        # Treatment → Outcome (핵심 엣지)
        if treatment in numeric_cols and outcome in numeric_cols:
            edges.append((treatment, outcome))

        # 강한 상관관계 기반 엣지 후보
        for i, v1 in enumerate(numeric_cols):
            for v2 in numeric_cols[i + 1:]:
                if v1 == v2:
                    continue
                r = corr_matrix.loc[v1, v2] if v1 in corr_matrix.index and v2 in corr_matrix.columns else 0
                if r > 0.3:
                    if v2 == outcome:
                        edges.append((v1, v2))
                    elif v1 == outcome:
                        edges.append((v2, v1))
                    elif v1 == treatment:
                        edges.append((v1, v2))
                    elif v2 == treatment:
                        edges.append((v2, v1))
                    else:
                        edges.append((v1, v2))

        # Reflexion 메모리 반영: 이전에 기각된 엣지 제거
        if state.reflexion_memory:
            all_rejected = set()
            for entry in state.reflexion_memory:
                all_rejected.update(tuple(e) for e in entry.rejected_edges)
            edges = [e for e in edges if e not in all_rejected]

        # 이전 반증에서 기각된 엣지도 제거 (하위 호환)
        if state.refutations:
            last_refutation = state.refutations[-1]
            rejected = set(tuple(e) for e in last_refutation.get("rejected_edges", []))
            edges = [e for e in edges if e not in rejected]

        edges = list(set(edges))

        return CausalHypothesis(
            edges=edges,
            confidence=0.5 + 0.1 * iteration,
            rationale=f"반복 {iteration}: 상관 기반 + Reflexion 반영",
            iteration=iteration,
        )

    def _reflect(
        self,
        hypothesis: CausalHypothesis,
        validation: Dict[str, Any],
        refutation: Dict[str, Any],
        iteration: int,
    ) -> ReflexionEntry:
        """Reflexion 메모리 항목을 생성합니다.

        이전 반복의 성공/실패를 구조화된 교훈으로 변환합니다.
        """
        validated = [v["edge"] for v in validation.get("validated", [])]
        rejected = [tuple(e) for e in refutation.get("rejected_edges", [])]

        # 교훈 생성
        lessons = []
        if rejected:
            lessons.append(f"엣지 {rejected}은 부분상관 검정에서 기각됨 — 교란에 의한 허위 상관일 가능성.")
        if validation.get("validation_rate", 0) < 0.5:
            lessons.append("검증률이 50% 미만 — 가설이 너무 공격적. 보수적 접근 필요.")
        if validation.get("validation_rate", 0) > 0.9 and not rejected:
            lessons.append("높은 검증률 + 기각 없음 — 현재 방향 유지.")
        new_cands = refutation.get("new_candidates", [])
        if new_cands:
            lessons.append(f"누락된 엣지 후보 발견: {new_cands[:3]}")

        lesson = " | ".join(lessons) if lessons else "특기사항 없음."

        return ReflexionEntry(
            iteration=iteration,
            hypothesis_summary=f"엣지 {len(hypothesis.edges)}개",
            validated_edges=validated,
            rejected_edges=rejected,
            lesson=lesson,
        )

    def _validate(
        self,
        df: pd.DataFrame,
        hypothesis: CausalHypothesis,
        all_vars: List[str],
    ) -> Dict[str, Any]:
        """가설의 각 엣지를 조건부 독립 검정으로 검증합니다."""
        from scipy import stats

        validated = []
        failed = []

        for cause, effect in hypothesis.edges:
            if cause not in df.columns or effect not in df.columns:
                failed.append((cause, effect, "변수 없음"))
                continue

            try:
                # 단순 상관 검정
                if df[cause].dtype in [np.float64, np.int64, float, int] and \
                   df[effect].dtype in [np.float64, np.int64, float, int]:
                    r, p_val = stats.pearsonr(df[cause].dropna(), df[effect].dropna())

                    if p_val < self.significance_level:
                        validated.append({
                            "edge": (cause, effect),
                            "correlation": round(float(r), 4),
                            "p_value": round(float(p_val), 6),
                            "status": "validated",
                        })
                    else:
                        failed.append((cause, effect, f"p={p_val:.4f}"))
                else:
                    validated.append({
                        "edge": (cause, effect),
                        "status": "skipped_non_numeric",
                    })
            except Exception as e:
                failed.append((cause, effect, str(e)))

        return {
            "validated": validated,
            "failed": failed,
            "validation_rate": len(validated) / max(len(hypothesis.edges), 1),
        }

    def _refute(
        self,
        df: pd.DataFrame,
        hypothesis: CausalHypothesis,
        treatment: str,
        outcome: str,
        features: List[str],
    ) -> Dict[str, Any]:
        """가설에 대한 반증 증거를 탐색합니다."""
        from scipy import stats

        rejected_edges = []
        new_candidates = []

        for cause, effect in hypothesis.edges:
            if cause not in df.columns or effect not in df.columns:
                continue

            # 반증 1: 역방향이 더 강한가?
            try:
                if df[cause].dtype in [np.float64, np.int64, float, int] and \
                   df[effect].dtype in [np.float64, np.int64, float, int]:
                    # 부분 상관 — 다른 변수 통제 후에도 유지되는지
                    other_vars = [v for v in features if v != cause and v != effect and v in df.columns]
                    if other_vars:
                        # 잔차 기반 부분 상관
                        from sklearn.linear_model import LinearRegression
                        valid_others = [v for v in other_vars[:5] if df[v].dtype in [np.float64, np.int64, float, int]]
                        if valid_others:
                            mask = df[[cause, effect] + valid_others].dropna().index
                            if len(mask) > 10:
                                X_ctrl = df.loc[mask, valid_others].values
                                res_cause = cause
                                res_effect = effect

                                lr1 = LinearRegression().fit(X_ctrl, df.loc[mask, cause])
                                lr2 = LinearRegression().fit(X_ctrl, df.loc[mask, effect])

                                resid1 = df.loc[mask, cause] - lr1.predict(X_ctrl)
                                resid2 = df.loc[mask, effect] - lr2.predict(X_ctrl)

                                partial_r, partial_p = stats.pearsonr(resid1, resid2)

                                if partial_p > self.significance_level:
                                    rejected_edges.append((cause, effect))
            except Exception:
                pass

        # 반증 2: 누락된 엣지 후보 탐색
        existing = set(hypothesis.edges)
        numeric_features = [f for f in features if f in df.columns and df[f].dtype in [np.float64, np.int64, float, int]]

        for feat in numeric_features[:10]:
            if outcome in df.columns and df[outcome].dtype in [np.float64, np.int64, float, int]:
                try:
                    r, p = stats.pearsonr(df[feat].dropna(), df[outcome].dropna())
                    if abs(r) > 0.2 and p < 0.01 and (feat, outcome) not in existing:
                        new_candidates.append((feat, outcome))
                except Exception:
                    pass

        return {
            "rejected_edges": rejected_edges,
            "rejected_count": len(rejected_edges),
            "new_candidates": new_candidates,
            "new_candidate_count": len(new_candidates),
        }

    def _check_convergence(
        self,
        state: LoopState,
        hypothesis: CausalHypothesis,
        validation: Dict[str, Any],
        refutation: Dict[str, Any],
    ) -> bool:
        """수렴 여부를 판단합니다."""
        # 기각된 엣지가 없고, 새 후보도 없으면 수렴
        if refutation["rejected_count"] == 0 and refutation["new_candidate_count"] == 0:
            return True

        # 변경 비율이 임계값 이하면 수렴
        total_edges = max(len(hypothesis.edges), 1)
        change_rate = (refutation["rejected_count"] + refutation["new_candidate_count"]) / total_edges

        return change_rate <= self.convergence_threshold
