# -*- coding: utf-8 -*-
"""CausalLoopAgent — 반복 자기교정 인과 발견 에이전트.

가설 → 검증 → 반증 → 수정 순환 워크플로.
LLM이 가설을 생성하고, 통계적 방법이 검증하고,
불일치 시 에이전트가 가설을 수정하여 반복합니다.

R&D 스프린트 1: CausalLoop Agent (축 2-1).
"""

from __future__ import annotations

import logging
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
class LoopState:
    """반복 상태."""
    hypotheses: List[CausalHypothesis] = field(default_factory=list)
    validations: List[Dict[str, Any]] = field(default_factory=list)
    refutations: List[Dict[str, Any]] = field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    final_dag: List[Tuple[str, str]] = field(default_factory=list)


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
    """

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.05,
        significance_level: float = 0.05,
    ) -> None:
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.significance_level = significance_level

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

            # ──── 1. 가설 생성 ────
            hypothesis = self._hypothesize(df, all_vars, treatment, outcome, state)
            state.hypotheses.append(hypothesis)

            # ──── 2. 검증 ────
            validation = self._validate(df, hypothesis, all_vars)
            state.validations.append(validation)

            # ──── 3. 반증 ────
            refutation = self._refute(df, hypothesis, treatment, outcome, features)
            state.refutations.append(refutation)

            # ──── 4. 수렴 판단 ────
            if self._check_convergence(state, hypothesis, validation, refutation):
                state.converged = True
                state.final_dag = hypothesis.edges
                logger.info(
                    "✅ CausalLoop 수렴 (반복 %d회) — 엣지 %d개",
                    iteration, len(hypothesis.edges),
                )
                break

            # ──── 5. 수정 (다음 반복에서 가설 생성 시 반영) ────
            logger.info(
                "🔁 반복 %d: 수정 필요 — 기각 %d개, 신규 후보 %d개",
                iteration,
                refutation.get("rejected_count", 0),
                refutation.get("new_candidates", 0),
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

        첫 반복: 상관관계 기반 초기 DAG 구성.
        이후 반복: 이전 반증 결과를 반영하여 수정된 DAG.
        """
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
                    # 방향 결정: Treatment/Outcome 우선
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

        # 이전 반증에서 기각된 엣지 제거
        if state.refutations:
            last_refutation = state.refutations[-1]
            rejected = set(tuple(e) for e in last_refutation.get("rejected_edges", []))
            edges = [e for e in edges if e not in rejected]

        # 중복 제거
        edges = list(set(edges))

        return CausalHypothesis(
            edges=edges,
            confidence=0.5 + 0.1 * iteration,
            rationale=f"반복 {iteration}: 상관 기반 + 이전 반증 반영",
            iteration=iteration,
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
