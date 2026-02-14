# -*- coding: utf-8 -*-
"""RAG Agent Implementation."""

import os
import logging
from typing import Optional

from engine.config import WhyLabConfig
from engine.rag.store import VectorStore
from engine.rag.loader import KnowledgeLoader

class RAGAgent:
    """Retrieval-Augmented Generation Agent."""
    
    def __init__(self, config: WhyLabConfig):
        self.config = config
        self.logger = logging.getLogger("whylab.rag.agent")
        
        # 컴포넌트 초기화
        self.store = VectorStore(persist_directory=str(config.paths.data_dir / "knowledge_db"))
        self.loader = KnowledgeLoader()
        
        # LLM 클라이언트 설정
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
        else:
            self.logger.warning("LLM API 키가 설정되지 않음. RAG 기능 제한됨.")
            self.model = None

    def index_knowledge(self):
        """최신 리포트와 데이터를 벡터 스토어에 인덱싱합니다."""
        self.logger.info("Knowledge Indexing 시작...")
        
        # 1. Markdown 리포트 로드
        # 가장 최근 리포트 찾기
        report_dir = self.config.paths.reports_dir
        reports = list(report_dir.glob("whylab_report_*.md"))
        if reports:
            latest_report = sorted(reports)[-1]
            self.logger.info(f"Report 로드: {latest_report}")
            docs, metas = self.loader.load_markdown_report(str(latest_report))
            if docs:
                # 기존 데이터 삭제 후 갱신 전략 (간단하게 reset)
                # self.store.reset() # 주의: 영구 저장소라면 reset은 신중해야 함. 일단 추가만.
                ids = [f"report_{latest_report.stem}_{i}" for i in range(len(docs))]
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

    def ask(self, query: str) -> str:
        """질문에 대한 답변을 생성합니다."""
        # 1. Retrieve
        try:
            results = self.store.query(query, n_results=3)
            context_docs = results.get("documents", [[]])[0]
        except Exception as e:
            self.logger.error(f"검색 실패: {e}")
            context_docs = []

        if not context_docs:
            context_text = "관련된 정보를 찾을 수 없습니다."
        else:
            context_text = "\n\n".join(context_docs)

        # 2. LLM Generation
        if not self.model:
            # Mock Mode: API Key가 없을 때 검색된 Context만이라도 보여줌
            return f"""[Mock Mode] API Key가 설정되지 않았습니다. 검색된 문맥은 다음과 같습니다:
            
{context_text}
            
(실제 답변을 위해서는 GEMINI_API_KEY 설정이 필요합니다.)"""
            
        # 3. Generate with LLM
        prompt = f"""
        당신은 WhyLab의 인과추론 분석 전문가 에이전트입니다.
        아래의 Context(실험 리포트 및 분석 데이터)를 바탕으로 사용자의 질문에 답하세요.
        
        [Context]
        {context_text}
        
        [Question]
        {query}
        
        [Instructions]
        - Context에 있는 정보만 사용하여 사실에 기반해 답변하세요.
        - 수치(ATE, Confidence 등)를 정확하게 인용하세요.
        - Context에 정보가 없으면 "분석 결과에 해당 내용이 없습니다"라고 말하세요.
        - 답변은 한국어로 명확하고 전문적으로 작성하세요.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"LLM 생성 실패: {e}")
            return f"[Error] 답변 생성 중 오류 발생: {e}"
