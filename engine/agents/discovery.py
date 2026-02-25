# -*- coding: utf-8 -*-
"""Discovery Agent — 인과 구조 발견을 위한 Nucleus Module.

데이터와 메타데이터를 분석하여 변수 간의 인과 관계(DAG)를 스스로 수립합니다.
LLM의 상식적 추론(Prior Knowledge)과 통계적 알고리즘(PC Algorithm)을 결합하는
하이브리드 발견 전략을 사용합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import networkx as nx

from engine.config import WhyLabConfig

class DiscoveryAgent:
    """인과 구조(DAG)를 자율적으로 발견하는 에이전트 (Nucleus)."""

    def __init__(self, config: WhyLabConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("whylab.agents.discovery")
        self._llm_client = None  # 추후 LLM 클라이언트 연동 (MCP 등)

    def auto_discover(
        self, df: pd.DataFrame, description: str = "",
    ) -> Dict[str, Any]:
        """CSV만 넣으면 treatment/outcome/confounder를 자동 탐색합니다.

        Args:
            df: 분석 대상 데이터프레임.
            description: (선택) 데이터 설명 텍스트.

        Returns:
            roles dict: treatment, outcome, confounders, dag.
        """
        self.logger.info("🔍 Auto-Discovery 시작: %d 컬럼 분석", len(df.columns))

        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        sample = df.head(3).to_dict(orient="records")

        # LLM으로 역할 탐색
        roles = self._discover_roles_with_llm(columns, dtypes, sample, description)

        if roles:
            self.logger.info("🤖 LLM 역할 탐색 완료: T=%s, Y=%s",
                             roles.get("treatment"), roles.get("outcome"))
        else:
            # 폴백: 도메인 휴리스틱
            roles = self._discover_roles_heuristic(df)
            self.logger.info("📏 휴리스틱 역할 탐색: T=%s, Y=%s",
                             roles.get("treatment"), roles.get("outcome"))

        # DAG 발견
        metadata = {
            "feature_names": roles.get("confounders", []),
            "treatment_col": roles.get("treatment"),
            "outcome_col": roles.get("outcome"),
        }
        dag = self.discover(df, metadata)
        roles["dag"] = list(dag.edges())

        return roles

    def discover(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> nx.DiGraph:
        """데이터로부터 인과 그래프를 발견합니다.

        Args:
            df: 분석 대상 데이터프레임.
            metadata: 컬럼 설명 등 메타데이터.

        Returns:
            NetworkX DiGraph 객체 (발견된 DAG).
        """
        self.logger.info("🧠 Nucleus(Discovery) 활성화: 데이터 분석 시작 (Rows: %d)", len(df))

        # 분석 대상 칼럼만 선택 (파생 칼럼 제외 → singular matrix 방지)
        analysis_cols = list(metadata.get("feature_names", []))
        for col_key in ("treatment_col", "outcome_col"):
            col = metadata.get(col_key)
            if col and col not in analysis_cols:
                analysis_cols.append(col)
        analysis_df = df[analysis_cols] if analysis_cols else df

        # 1. LLM 기반 사전 지식(Prior Knowledge) 수립
        prior_dag = self._reason_with_llm(metadata)

        # 2. 통계적 인과 발견 (PC Algorithm)
        stat_dag = self._discover_statistically(analysis_df)

        # 3. 하이브리드 병합 (Ensemble)
        final_dag = self._merge_graphs(prior_dag, stat_dag)

        self.logger.info("✨ 인과 구조 발견 완료 (Nodes: %d, Edges: %d)",
                         final_dag.number_of_nodes(), final_dag.number_of_edges())
        return final_dag

    def _discover_roles_with_llm(
        self, columns, dtypes, sample, description,
    ) -> Optional[Dict[str, Any]]:
        """LLM으로 변수 역할을 탐색합니다."""
        import os
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None

        try:
            import google.generativeai as genai
            import json
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")

            prompt = (
                "당신은 인과추론 전문가입니다. 아래 데이터셋의 컬럼을 분석하여 "
                "treatment(처치), outcome(결과), confounders(교란변수)를 식별해주세요.\n\n"
                f"컬럼: {columns}\n"
                f"타입: {dtypes}\n"
                f"샘플: {sample[:2]}\n"
                f"설명: {description or '없음'}\n\n"
                "다음 JSON 형식으로만 응답하세요:\n"
                '{"treatment": "컬럼명", "outcome": "컬럼명", '
                '"confounders": ["컬럼1", "컬럼2", ...], '
                '"reasoning": "한 줄 설명"}'
            )

            response = model.generate_content(prompt)
            text = response.text.strip()
            # JSON 추출
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            return json.loads(text)
        except Exception as e:
            self.logger.warning("LLM 역할 탐색 실패: %s", e)
            return None

    def _discover_roles_heuristic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """휴리스틱 기반 변수 역할 탐색 (폴백)."""
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # 이진 변수 → outcome 후보
        binary_cols = [c for c in columns if df[c].nunique() <= 2]
        # 연속 변수 → treatment 후보
        continuous_cols = [c for c in columns if df[c].nunique() > 10]

        outcome = binary_cols[0] if binary_cols else columns[-1]
        treatment = continuous_cols[0] if continuous_cols else columns[0]
        confounders = [c for c in columns if c not in (treatment, outcome)]

        return {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "reasoning": "휴리스틱: 이진변수→outcome, 연속변수→treatment",
        }

    def _reason_with_llm(self, metadata: Dict[str, Any]) -> nx.DiGraph:
        """LLM을 사용하여 변수 간의 인과관계를 추론합니다."""
        self.logger.info("   [1] LLM Reasoning: 변수 의미론적 분석 중...")

        dag = nx.DiGraph()
        nodes = metadata.get("feature_names", []) + [
            metadata.get("treatment_col"), metadata.get("outcome_col")
        ]
        for node in nodes:
            if node:
                dag.add_node(node)

        # Gemini LLM 호출 시도
        import os
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                import json
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash")

                prompt = (
                    "당신은 인과추론 전문가입니다. 아래 변수들 간의 인과관계(DAG)를 "
                    "도메인 지식으로 추론하세요.\n\n"
                    f"변수: {[n for n in nodes if n]}\n"
                    f"Treatment: {metadata.get('treatment_col')}\n"
                    f"Outcome: {metadata.get('outcome_col')}\n\n"
                    "JSON 배열로 엣지를 반환하세요: "
                    '[["원인", "결과"], ["원인2", "결과2"], ...]'
                )

                response = model.generate_content(prompt)
                text = response.text.strip()
                if "```" in text:
                    text = text.split("```")[1].replace("json", "").strip()
                edges = json.loads(text)
                for u, v in edges:
                    if u in dag.nodes and v in dag.nodes:
                        dag.add_edge(u, v)
                self.logger.info("       🤖 LLM 가설 수립 완료 (엣지 %d개)", dag.number_of_edges())
                return dag
            except Exception as e:
                self.logger.warning("       LLM 실패 → 규칙 기반 폴백: %s", e)

        # 폴백: 도메인 규칙 기반
        if "age" in nodes:
            for target in ("credit_limit", "is_default", "income", "credit_score"):
                if target in nodes:
                    dag.add_edge("age", target)
        if "income" in nodes:
            for target in ("credit_limit", "is_default", "credit_score"):
                if target in nodes:
                    dag.add_edge("income", target)
        if "credit_score" in nodes:
            for target in ("credit_limit", "is_default"):
                if target in nodes:
                    dag.add_edge("credit_score", target)

        treatment = metadata.get("treatment_col")
        outcome = metadata.get("outcome_col")
        if treatment and outcome and treatment in nodes and outcome in nodes:
            dag.add_edge(treatment, outcome)

        self.logger.info("       규칙 기반 가설 수립 완료 (엣지 %d개)", dag.number_of_edges())
        return dag

    def _discover_statistically(self, df: pd.DataFrame) -> nx.DiGraph:
        """PC 알고리즘으로 조건부 독립성 기반 인과관계를 발견합니다."""
        self.logger.info("   [2] Statistical Discovery: PC Algorithm 실행 중...")

        numeric_df = df.select_dtypes(include=[np.number])
        columns = numeric_df.columns.tolist()
        data = numeric_df.values

        dag = nx.DiGraph()
        dag.add_nodes_from(columns)

        try:
            from causallearn.search.ConstraintBased.PC import pc

            cg = pc(data, alpha=0.05, indep_test='fisherz', show_progress=False)
            adj = cg.G.graph  # numpy adjacency matrix

            for i in range(len(columns)):
                for j in range(len(columns)):
                    if adj[i, j] == -1 and adj[j, i] == 1:
                        # i → j (방향 확정)
                        dag.add_edge(columns[i], columns[j])
                    elif adj[i, j] == -1 and adj[j, i] == -1:
                        # i — j (무방향) → 도메인 heuristic으로 방향 결정
                        if columns[i] in ("age", "income", "credit_score"):
                            dag.add_edge(columns[i], columns[j])
                        else:
                            dag.add_edge(columns[j], columns[i])

            self.logger.info("       PC Algorithm 완료 (엣지 %d개 발견)", dag.number_of_edges())

        except ImportError:
            self.logger.warning("       causal-learn 미설치 — 상관 heuristic fallback 사용")
        except ValueError as e:
            # Singular correlation matrix: 합성 데이터 공선성 또는 분산 0 컬럼
            self.logger.warning(
                "       PC Algorithm ValueError: %s — 상관 heuristic fallback", e
            )

        # PC 알고리즘 실패 시 (ImportError/ValueError) 상관 heuristic
        if dag.number_of_edges() == 0:
            corr_matrix = numeric_df.corr().abs()
            threshold = 0.3

            for i, col_a in enumerate(columns):
                for j, col_b in enumerate(columns):
                    if i >= j:
                        continue
                    if corr_matrix.iloc[i, j] > threshold:
                        if col_a == "age":
                            dag.add_edge(col_a, col_b)
                        elif col_b == "age":
                            dag.add_edge(col_b, col_a)
                        else:
                            dag.add_edge(col_a, col_b)

        return dag

    def _merge_graphs(self, prior: nx.DiGraph, stat: nx.DiGraph) -> nx.DiGraph:
        """LLM의 가설(Prior)과 통계적 발견(Data)을 통합합니다."""
        self.logger.info("   [3] Hybrid Fusion: 가설과 데이터의 통합")
        
        # 기본 전략: 통계적 발견을 존중하되, LLM의 상식으로 방향을 교정
        merged = stat.copy()
        
        # LLM의 강력한 제약조건(Hard Constraints) 적용
        # 예: Prior에 있는 엣지는 반드시 포함하거나 방향을 강제
        for u, v in prior.edges():
            if not merged.has_edge(u, v):
                # 데이터에선 약했지만 상식적으로 확실하면 추가
                if not merged.has_edge(v, u): # 역방향이 없다면
                    merged.add_edge(u, v)
            elif merged.has_edge(v, u):
                # 데이터가 역방향을 가리키면, 상식(LLM)을 우선하여 뒤집음
                merged.remove_edge(v, u)
                merged.add_edge(u, v)
                
        return merged
