# -*- coding: utf-8 -*-
"""MACDiscoveryAgent — 다중 에이전트 인과 구조 발견(MAC).

여러 인과 발견 알고리즘(PC, GES, LiNGAM)을 독립 Specialist 에이전트로
실행하고, Aggregator가 투표 기반 앙상블로 최종 DAG를 결정합니다.

아키텍처:
  ┌──────────────┐
  │  Coordinator │
  └──────┬───────┘
    ┌────┼────┐
    │    │    │
  ┌─┴─┐┌─┴─┐┌─┴───┐
  │ PC ││GES││LiNGAM│  ← Specialist 에이전트
  └─┬─┘└─┬─┘└──┬───┘
    └────┼─────┘
    ┌────┴────┐
    │Aggregator│  ← 투표 + 안정성 분석
    └─────────┘

학술 참조:
  - Spirtes et al. (2000). Causation, Prediction, and Search.
  - Chickering (2002). Optimal structure identification with GES.
  - Shimizu et al. (2006). LiNGAM.
  - Claassen et al. (2022). Multi-Agent Causal Discovery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class Edge:
    """인과 엣지."""
    source: str
    target: str
    weight: float = 1.0
    direction: str = "directed"    # "directed" | "undirected"
    discovered_by: List[str] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """개별 Specialist 발견 결과."""
    algorithm: str
    edges: List[Edge]
    adjacency: Optional[np.ndarray] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedDAG:
    """투표 앙상블 최종 DAG."""
    edges: List[Edge]
    adjacency: np.ndarray
    stability_scores: Dict[str, float]
    agreement_matrix: np.ndarray
    consensus_level: float
    variable_names: List[str]


# ──────────────────────────────────────────────
# Specialist 에이전트
# ──────────────────────────────────────────────

class PCSpecialist:
    """PC 알고리즘 Specialist (조건부 독립성 기반)."""

    name = "PC"

    def discover(
        self,
        data: np.ndarray,
        variable_names: List[str],
        alpha: float = 0.05,
    ) -> DiscoveryResult:
        """PC 알고리즘으로 인과 구조를 발견합니다.

        간소화된 PC: 조건부 독립성 검정 기반 스켈레톤 추정.
        """
        n, p = data.shape
        adj = np.ones((p, p)) - np.eye(p)  # 완전 그래프에서 시작

        # Phase 1: 스켈레톤 추정 — 조건부 독립 엣지 제거
        for depth in range(p):
            for i in range(p):
                for j in range(i + 1, p):
                    if adj[i, j] == 0:
                        continue
                    # 가능한 조건화 변수 집합
                    neighbors = [k for k in range(p) if k != i and k != j and adj[i, k] == 1]
                    if depth > len(neighbors):
                        continue

                    # 깊이별 조건 집합 샘플링
                    from itertools import combinations
                    for cond_set in combinations(neighbors, min(depth, len(neighbors))):
                        if not cond_set:
                            # 무조건 상관
                            r, pval = sp_stats.pearsonr(data[:, i], data[:, j])
                        else:
                            # 편상관 (조건화)
                            pval = self._partial_corr_test(data, i, j, list(cond_set))

                        if pval > alpha:
                            adj[i, j] = 0
                            adj[j, i] = 0
                            break

        # Phase 2: 방향 결정 (V-구조 탐지)
        directed_adj = self._orient_edges(adj, data)

        edges = self._adj_to_edges(directed_adj, variable_names, self.name)

        return DiscoveryResult(
            algorithm=self.name,
            edges=edges,
            adjacency=directed_adj,
            score=self._bic_score(data, directed_adj),
        )

    def _partial_corr_test(
        self, data: np.ndarray, i: int, j: int, cond: List[int],
    ) -> float:
        """편상관 기반 조건부 독립성 검정 p-value."""
        n = len(data)
        try:
            from numpy.linalg import lstsq
            Z = data[:, cond]
            # i와 j에서 조건 변수 효과 제거
            ri = data[:, i] - Z @ lstsq(Z, data[:, i], rcond=None)[0]
            rj = data[:, j] - Z @ lstsq(Z, data[:, j], rcond=None)[0]
            r, pval = sp_stats.pearsonr(ri, rj)
            return pval
        except Exception:
            return 1.0  # 검정 실패 시 독립으로 간주

    def _orient_edges(self, adj: np.ndarray, data: np.ndarray) -> np.ndarray:
        """V-구조를 탐지하여 엣지 방향을 결정합니다."""
        p = adj.shape[0]
        directed = adj.copy()

        for j in range(p):
            parents = [i for i in range(p) if adj[i, j] == 1]
            for idx_a in range(len(parents)):
                for idx_b in range(idx_a + 1, len(parents)):
                    a, b = parents[idx_a], parents[idx_b]
                    # a - j - b에서 a와 b가 인접하지 않으면 V-구조
                    if adj[a, b] == 0 and adj[b, a] == 0:
                        directed[j, a] = 0  # a → j
                        directed[j, b] = 0  # b → j

        return directed

    @staticmethod
    def _bic_score(data: np.ndarray, adj: np.ndarray) -> float:
        """BIC 점수 (낮을수록 좋음)."""
        n, p = data.shape
        score = 0.0
        for j in range(p):
            parents = np.where(adj[:, j] > 0)[0]
            if len(parents) == 0:
                var = np.var(data[:, j])
            else:
                from numpy.linalg import lstsq
                X_pa = data[:, parents]
                coef = lstsq(X_pa, data[:, j], rcond=None)[0]
                residuals = data[:, j] - X_pa @ coef
                var = np.var(residuals)
            score += n * np.log(var + 1e-10) + len(parents) * np.log(n)
        return score

    @staticmethod
    def _adj_to_edges(
        adj: np.ndarray, names: List[str], algorithm: str,
    ) -> List[Edge]:
        """인접 행렬을 엣지 리스트로 변환."""
        edges = []
        p = adj.shape[0]
        for i in range(p):
            for j in range(p):
                if adj[i, j] > 0 and i != j:
                    if adj[j, i] > 0:
                        if i < j:
                            edges.append(Edge(
                                source=names[i], target=names[j],
                                direction="undirected",
                                discovered_by=[algorithm],
                            ))
                    else:
                        edges.append(Edge(
                            source=names[i], target=names[j],
                            direction="directed",
                            discovered_by=[algorithm],
                        ))
        return edges


class GESSpecialist:
    """GES(Greedy Equivalence Search) Specialist."""

    name = "GES"

    def discover(
        self,
        data: np.ndarray,
        variable_names: List[str],
        **kwargs,
    ) -> DiscoveryResult:
        """탐욕적 등가 탐색으로 DAG를 발견합니다."""
        n, p = data.shape
        adj = np.zeros((p, p))

        # Forward phase: 엣지 추가
        adj = self._forward_phase(data, adj)

        # Backward phase: 불필요 엣지 제거
        adj = self._backward_phase(data, adj)

        edges = PCSpecialist._adj_to_edges(adj, variable_names, self.name)

        return DiscoveryResult(
            algorithm=self.name,
            edges=edges,
            adjacency=adj,
            score=self._compute_bic(data, adj),
        )

    def _forward_phase(self, data: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """엣지를 탐욕적으로 추가합니다."""
        n, p = data.shape
        improved = True
        current_score = self._compute_bic(data, adj)

        while improved:
            improved = False
            best_delta = 0
            best_edge = None

            for i in range(p):
                for j in range(p):
                    if i == j or adj[i, j] == 1:
                        continue
                    # 엣지 추가 시도
                    adj_new = adj.copy()
                    adj_new[i, j] = 1
                    new_score = self._compute_bic(data, adj_new)
                    delta = current_score - new_score  # BIC 감소가 좋음

                    if delta > best_delta:
                        best_delta = delta
                        best_edge = (i, j)

            if best_edge is not None and best_delta > 0:
                adj[best_edge[0], best_edge[1]] = 1
                current_score -= best_delta
                improved = True

        return adj

    def _backward_phase(self, data: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """불필요한 엣지를 제거합니다."""
        n, p = data.shape
        improved = True
        current_score = self._compute_bic(data, adj)

        while improved:
            improved = False
            best_delta = 0
            best_edge = None

            for i in range(p):
                for j in range(p):
                    if adj[i, j] == 0:
                        continue
                    adj_new = adj.copy()
                    adj_new[i, j] = 0
                    new_score = self._compute_bic(data, adj_new)
                    delta = current_score - new_score

                    if delta > best_delta:
                        best_delta = delta
                        best_edge = (i, j)

            if best_edge is not None and best_delta > 0:
                adj[best_edge[0], best_edge[1]] = 0
                current_score -= best_delta
                improved = True

        return adj

    @staticmethod
    def _compute_bic(data: np.ndarray, adj: np.ndarray) -> float:
        """BIC 점수."""
        return PCSpecialist._bic_score(data, adj)


class LiNGAMSpecialist:
    """LiNGAM(Linear Non-Gaussian Acyclic Model) Specialist."""

    name = "LiNGAM"

    def discover(
        self,
        data: np.ndarray,
        variable_names: List[str],
        **kwargs,
    ) -> DiscoveryResult:
        """비가우시안 독립 성분 기반 인과 순서 발견."""
        n, p = data.shape

        # 간소화된 DirectLiNGAM: 독립 성분 회귀 기반
        order = self._estimate_causal_order(data)
        adj = self._estimate_adjacency(data, order)

        edges = PCSpecialist._adj_to_edges(adj, variable_names, self.name)

        return DiscoveryResult(
            algorithm=self.name,
            edges=edges,
            adjacency=adj,
            score=PCSpecialist._bic_score(data, adj),
        )

    def _estimate_causal_order(self, data: np.ndarray) -> List[int]:
        """비가우시안 독립성 기반 인과 순서 추정."""
        n, p = data.shape
        remaining = list(range(p))
        order = []

        for _ in range(p):
            # 가장 외생적인(exogenous) 변수를 찾음
            # 기준: 잔차의 비가우시안 정도 (절대 쿨토시스 기반)
            best_var = None
            best_score = -np.inf

            for var in remaining:
                others = [v for v in remaining if v != var]
                if not others:
                    residual = data[:, var]
                else:
                    from numpy.linalg import lstsq
                    X = data[:, others]
                    coef = lstsq(X, data[:, var], rcond=None)[0]
                    residual = data[:, var] - X @ coef

                # 비가우시안 정도: |kurtosis|가 클수록 비가우시안
                kurtosis = sp_stats.kurtosis(residual)
                score = abs(kurtosis)

                if score > best_score:
                    best_score = score
                    best_var = var

            if best_var is not None:
                order.append(best_var)
                remaining.remove(best_var)

        return order

    def _estimate_adjacency(
        self, data: np.ndarray, order: List[int],
    ) -> np.ndarray:
        """인과 순서를 기반으로 인접 행렬 추정."""
        n, p = data.shape
        adj = np.zeros((p, p))
        threshold = 0.1  # 약한 연결 제거

        for idx, target in enumerate(order):
            predecessors = order[:idx]
            if not predecessors:
                continue

            from numpy.linalg import lstsq
            X = data[:, predecessors]
            coef = lstsq(X, data[:, target], rcond=None)[0]

            for k, pred in enumerate(predecessors):
                if abs(coef[k]) > threshold:
                    adj[pred, target] = 1

        return adj


# ──────────────────────────────────────────────
# Aggregator
# ──────────────────────────────────────────────

class VoteAggregator:
    """투표 기반 DAG 앙상블 Aggregator."""

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: 엣지 포함을 위한 최소 투표 비율 (기본 50%).
        """
        self.threshold = threshold

    def aggregate(
        self,
        results: List[DiscoveryResult],
        variable_names: List[str],
    ) -> AggregatedDAG:
        """Specialist 결과를 투표로 앙상블합니다."""
        p = len(variable_names)
        n_agents = len(results)

        # 투표 행렬: 각 셀 = 해당 엣지에 투표한 에이전트 수
        vote_matrix = np.zeros((p, p))
        name_to_idx = {name: i for i, name in enumerate(variable_names)}

        for result in results:
            if result.adjacency is not None:
                vote_matrix += (result.adjacency > 0).astype(float)
            else:
                for edge in result.edges:
                    src = name_to_idx.get(edge.source)
                    tgt = name_to_idx.get(edge.target)
                    if src is not None and tgt is not None:
                        vote_matrix[src, tgt] += 1
                        if edge.direction == "undirected":
                            vote_matrix[tgt, src] += 1

        # 합의 행렬: 투표 비율
        agreement_matrix = vote_matrix / max(n_agents, 1)

        # 임계값 기반 최종 DAG
        final_adj = (agreement_matrix >= self.threshold).astype(float)

        # 안정성 점수: 각 엣지의 합의율
        stability_scores = {}
        edges = []
        for i in range(p):
            for j in range(p):
                if final_adj[i, j] > 0 and i != j:
                    edge_key = f"{variable_names[i]}->{variable_names[j]}"
                    stability_scores[edge_key] = float(agreement_matrix[i, j])

                    discovered_by = []
                    for r in results:
                        if r.adjacency is not None and r.adjacency[i, j] > 0:
                            discovered_by.append(r.algorithm)

                    edges.append(Edge(
                        source=variable_names[i],
                        target=variable_names[j],
                        weight=float(agreement_matrix[i, j]),
                        direction="directed" if final_adj[j, i] == 0 else "undirected",
                        discovered_by=discovered_by,
                    ))

        consensus = float(np.mean(agreement_matrix[final_adj > 0])) if np.any(final_adj > 0) else 0.0

        return AggregatedDAG(
            edges=edges,
            adjacency=final_adj,
            stability_scores=stability_scores,
            agreement_matrix=agreement_matrix,
            consensus_level=consensus,
            variable_names=variable_names,
        )


# ──────────────────────────────────────────────
# MACDiscoveryAgent
# ──────────────────────────────────────────────

class MACDiscoveryAgent:
    """다중 에이전트 인과 구조 발견 에이전트.

    사용법:
        agent = MACDiscoveryAgent()
        result = agent.discover(data, variable_names)

    파이프라인 인터페이스:
        result = agent.execute(inputs)
    """

    name = "MACDiscovery"

    SPECIALIST_REGISTRY = {
        "PC": PCSpecialist,
        "GES": GESSpecialist,
        "LiNGAM": LiNGAMSpecialist,
    }

    def __init__(
        self,
        specialists: Optional[List[str]] = None,
        vote_threshold: float = 0.5,
        alpha: float = 0.05,
    ):
        """
        Args:
            specialists: 사용할 알고리즘 이름 목록. None이면 전부 사용.
            vote_threshold: 엣지 포함 최소 투표 비율.
            alpha: PC 알고리즘 독립성 검정 유의수준.
        """
        specialist_names = specialists or list(self.SPECIALIST_REGISTRY.keys())
        self.specialists = []
        for name in specialist_names:
            if name in self.SPECIALIST_REGISTRY:
                self.specialists.append(self.SPECIALIST_REGISTRY[name]())
            else:
                logger.warning("알 수 없는 specialist: %s", name)

        self.aggregator = VoteAggregator(threshold=vote_threshold)
        self.alpha = alpha
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def discover(
        self,
        data: np.ndarray,
        variable_names: List[str],
    ) -> AggregatedDAG:
        """다중 에이전트 인과 발견을 수행합니다.

        Args:
            data: (n, p) 관측 데이터.
            variable_names: 변수 이름 리스트.

        Returns:
            AggregatedDAG: 앙상블 DAG.
        """
        n, p = data.shape
        assert p == len(variable_names), "변수 이름 수와 데이터 차원 불일치"

        self.logger.info(
            "MAC 발견 시작: n=%d, p=%d, specialists=%s",
            n, p, [s.name for s in self.specialists],
        )

        # 각 Specialist 독립 실행
        results = []
        for specialist in self.specialists:
            try:
                result = specialist.discover(
                    data, variable_names, alpha=self.alpha,
                )
                results.append(result)
                self.logger.info(
                    "  %s: %d개 엣지, BIC=%.1f",
                    specialist.name, len(result.edges), result.score,
                )
            except Exception as e:
                self.logger.warning("  %s 실패: %s", specialist.name, e)

        if not results:
            self.logger.error("모든 Specialist가 실패했습니다.")
            return AggregatedDAG(
                edges=[], adjacency=np.zeros((p, p)),
                stability_scores={}, agreement_matrix=np.zeros((p, p)),
                consensus_level=0.0, variable_names=variable_names,
            )

        # Aggregator 투표 앙상블
        dag = self.aggregator.aggregate(results, variable_names)

        self.logger.info(
            "MAC 완료: %d개 합의 엣지, 합의율=%.1f%%",
            len(dag.edges), dag.consensus_level * 100,
        )

        return dag

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """파이프라인 셀 인터페이스."""
        df = inputs.get("dataframe")
        feature_names = inputs.get("feature_names")
        treatment_col = inputs.get("treatment_col", "treatment")
        outcome_col = inputs.get("outcome_col", "outcome")

        if df is None:
            self.logger.warning("데이터프레임 없음.")
            return {**inputs, "mac_discovery": None}

        # 분석 대상 컬럼 결정
        if feature_names:
            cols = list(feature_names) + [treatment_col, outcome_col]
            cols = [c for c in cols if c in df.columns]
        else:
            cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64, float, int]]

        if len(cols) < 2:
            return {**inputs, "mac_discovery": {"skipped": True, "reason": "insufficient_columns"}}

        data = df[cols].values.astype(np.float64)
        dag = self.discover(data, cols)

        # 직렬화
        mac_output = {
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "direction": e.direction,
                    "discovered_by": e.discovered_by,
                }
                for e in dag.edges
            ],
            "consensus_level": dag.consensus_level,
            "stability_scores": dag.stability_scores,
            "n_edges": len(dag.edges),
            "variable_names": dag.variable_names,
        }

        # dag_edges 형식도 파이프라인 호환용으로 추가
        dag_edges = [(e.source, e.target) for e in dag.edges if e.direction == "directed"]
        return {**inputs, "mac_discovery": mac_output, "dag_edges": dag_edges}
