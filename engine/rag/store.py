# -*- coding: utf-8 -*-
"""Vector Store implementation using ChromaDB."""

import os
import logging
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    """문서 임베딩 및 검색을 위한 벡터 저장소."""

    def __init__(self, persist_directory: str = "whylab_chroma_db"):
        self.logger = logging.getLogger("whylab.rag.store")
        
        # 절대 경로 사용
        if not os.path.isabs(persist_directory):
            persist_directory = os.path.join(os.getcwd(), persist_directory)
            
        self.persist_directory = persist_directory
        self.logger.info(f"Vector Store 초기화: {persist_directory}")
        
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # 임베딩 함수 (all-MiniLM-L6-v2: 작고 빠름)
            # sentence-transformers가 설치되어 있어야 함
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # 컬렉션 로드 또는 생성
            self.collection = self.client.get_or_create_collection(
                name="whylab_knowledge",
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            self.logger.error(f"ChromaDB 초기화 실패: {e}")
            raise e

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """문서 추가."""
        if not documents:
            return
            
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            self.logger.info(f"문서 {len(documents)}건 추가 완료")
        except Exception as e:
            self.logger.error(f"문서 추가 실패: {e}")
            raise e

    def query(self, query_text: str, n_results: int = 3) -> dict:
        """유사 문서 검색."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            self.logger.error(f"검색 실패: {e}")
            return {}

    def count(self) -> int:
        """저장된 문서 수 반환."""
        return self.collection.count()

    def reset(self):
        """컬렉션 초기화 (삭제 후 재생성)."""
        try:
            self.client.delete_collection("whylab_knowledge")
            self.collection = self.client.create_collection(
                name="whylab_knowledge",
                embedding_function=self.embedding_fn
            )
            self.logger.info("Vector Store 초기화 완료")
        except Exception as e:
            self.logger.error(f"초기화 실패: {e}")
