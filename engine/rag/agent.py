# -*- coding: utf-8 -*-
"""RAG Agent Implementation (v2 — 고도화).

변경점 (v1 대비):
- 멀티턴 대화 히스토리 (최근 5턴)
- 비즈니스 페르소나별 톤 전환
- 자동 분석 트리거 (왜? → 파이프라인 자동 실행)
- 프롬프트 모듈 분리 (prompts.py)
"""

import os
import logging
from typing import Optional, List, Dict

from engine.config import WhyLabConfig
from engine.rag.store import VectorStore
from engine.rag.loader import KnowledgeLoader
from engine.rag.prompts import (
    SYSTEM_PROMPT,
    build_query_prompt,
    should_trigger_analysis,
)

# 대화 히스토리 최대 보관 수
MAX_HISTORY_TURNS = 5


class RAGAgent:
    """Retrieval-Augmented Generation Agent (v2).

    주요 기능:
    - 벡터 검색 기반 컨텍스트 제공
    - 멀티턴 대화 히스토리
    - 비즈니스 페르소나별 답변 생성
    - 자동 파이프라인 실행 트리거
    """

    def __init__(self, config: WhyLabConfig):
        self.config = config
        self.logger = logging.getLogger("whylab.rag.agent")
        self.history: List[Dict[str, str]] = []

        # 컴포넌트 초기화
        self.store = VectorStore(
            persist_directory=str(config.paths.data_dir / "knowledge_db")
        )
        self.loader = KnowledgeLoader()

        # LLM 클라이언트 설정
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if self.api_key:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        else:
            self.logger.warning("LLM API 키가 설정되지 않음. RAG 기능 제한됨.")
            self.model = None

    # ──────────────────────────────────────────
    # 지식 인덱싱
    # ──────────────────────────────────────────
    def index_knowledge(self):
        """최신 리포트와 데이터를 벡터 스토어에 인덱싱합니다."""
        self.logger.info("Knowledge Indexing 시작...")

        # 1. Markdown 리포트 로드
        report_dir = self.config.paths.reports_dir
        reports = list(report_dir.glob("whylab_report_*.md"))
        if reports:
            latest_report = sorted(reports)[-1]
            self.logger.info(f"Report 로드: {latest_report}")
            docs, metas = self.loader.load_markdown_report(str(latest_report))
            if docs:
                ids = [
                    f"report_{latest_report.stem}_{i}" for i in range(len(docs))
                ]
                self.store.add_documents(docs, metas, ids)

        # 2. JSON 데이터 로드
        json_path = self.config.paths.dashboard_data_dir / "latest.json"
        if json_path.exists():
            self.logger.info(f"Metric 로드: {json_path}")
            docs, metas = self.loader.load_json_metric(str(json_path))
            if docs:
                ids = [f"metric_{i}" for i in range(len(docs))]
                self.store.add_documents(docs, metas, ids)

        self.logger.info("Indexing 완료")

    # ──────────────────────────────────────────
    # 대화 히스토리 관리
    # ──────────────────────────────────────────
    def _add_to_history(self, role: str, content: str):
        """대화 히스토리에 추가하고, 최대 턴수를 초과하면 오래된 것부터 제거."""
        self.history.append({"role": role, "content": content})
        # 최근 N턴만 유지 (user + assistant = 2 * N 메시지)
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def clear_history(self):
        """대화 히스토리를 초기화합니다."""
        self.history.clear()

    # ──────────────────────────────────────────
    # 질의 응답
    # ──────────────────────────────────────────
    def ask(
        self,
        query: str,
        persona: str = "product_owner",
        auto_analyze: bool = True,
    ) -> str:
        """질문에 대한 답변을 생성합니다.

        Args:
            query: 사용자 질문.
            persona: 답변 페르소나 ("growth_hacker"|"risk_manager"|"product_owner").
            auto_analyze: True이면, 인과 질문 감지 시 자동 파이프라인 실행.

        Returns:
            에이전트 답변 문자열.
        """
        self._add_to_history("user", query)

        # 1. 자동 분석 트리거 확인
        if auto_analyze and should_trigger_analysis(query):
            self.logger.info("🔬 자동 분석 트리거 감지: '%s'", query)
            self._run_auto_analysis()

        # 2. Retrieve
        try:
            results = self.store.query(query, n_results=5)
            context_docs = results.get("documents", [[]])[0]
        except Exception as e:
            self.logger.error(f"검색 실패: {e}")
            context_docs = []

        context_text = (
            "\n\n".join(context_docs)
            if context_docs
            else "관련된 분석 결과를 찾을 수 없습니다. 파이프라인을 먼저 실행해주세요."
        )

        # 3. LLM 생성
        if not self.model:
            answer = self._mock_response(context_text)
        else:
            answer = self._generate_with_llm(context_text, query, persona)

        self._add_to_history("assistant", answer)
        return answer

    def _generate_with_llm(
        self, context: str, query: str, persona: str
    ) -> str:
        """LLM으로 답변을 생성합니다."""
        prompt = build_query_prompt(
            context=context,
            query=query,
            history=self.history[:-1],  # 현재 질문 제외
            persona=persona,
        )

        try:
            response = self.model.generate_content(
                [SYSTEM_PROMPT, prompt]
            )
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"LLM 생성 실패: {e}")
            return f"[Error] 답변 생성 중 오류 발생: {e}"

    def _mock_response(self, context: str) -> str:
        """API Key 없을 때 검색된 Context를 반환합니다."""
        return (
            "[Mock Mode] API Key가 설정되지 않았습니다.\n\n"
            f"검색된 문맥:\n{context}\n\n"
            "(실제 답변을 위해서는 GEMINI_API_KEY 설정이 필요합니다.)"
        )

    def _run_auto_analysis(self):
        """파이프라인을 자동 실행하고 결과를 인덱싱합니다."""
        try:
            from engine.orchestrator import Orchestrator

            orchestrator = Orchestrator(config=self.config)
            orchestrator.run_pipeline(scenario="A")
            self.logger.info("✅ 자동 분석 완료. 결과 인덱싱 중...")
            self.index_knowledge()
        except Exception as e:
            self.logger.warning(f"자동 분석 실패: {e}")
