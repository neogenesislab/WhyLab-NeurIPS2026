# -*- coding: utf-8 -*-
"""Knowledge Loader for RAG."""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple

class KnowledgeLoader:
    """다양한 형식의 문서를 로드하여 RAG용 청크로 변환합니다."""
    
    def __init__(self):
        self.logger = logging.getLogger("whylab.rag.loader")

    def load_markdown_report(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """마크다운 리포트를 섹션별로 분할하여 로드합니다.
        
        Returns:
            (documents, metadatas) 튜플
        """
        path = Path(file_path)
        if not path.exists():
            self.logger.warning(f"파일을 찾을 수 없음: {file_path}")
            return [], []

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # 헤더(#) 기준으로 청크 분할 (간이 구현)
            # ## Header -> Content
            chunks = []
            metadatas = []
            
            # 정규식으로 ## 헤더 탐색
            # (## Title\nContent...)
            sections = re.split(r'(^|\n)(?=#+ )', content)
            
            current_header = "Intro"
            for section in sections:
                if not section.strip():
                    continue
                
                # 헤더 확인
                header_match = re.match(r'(#+) (.*)', section)
                if header_match:
                    level = len(header_match.group(1))
                    title = header_match.group(2).strip()
                    current_header = title
                    # 헤더 줄 제거 후 내용만 남김
                    text = section[header_match.end():].strip()
                else:
                    text = section.strip()
                
                if text:
                    chunks.append(f"[{current_header}] {text}")
                    metadatas.append({
                        "source": path.name,
                        "section": current_header,
                        "type": "report"
                    })
            
            self.logger.info(f"Markdown 로드 완료: {len(chunks)} 청크")
            return chunks, metadatas

        except Exception as e:
            self.logger.error(f"Markdown 로드 실패: {e}")
            return [], []

    def load_json_metric(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """JSON 데이터(latest.json)를 문장 형태로 변환하여 로드합니다."""
        path = Path(file_path)
        if not path.exists():
            return [], []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            docs = []
            metas = []

            # 1. ATE
            # 1. ATE
            if "ate" in data:
                ate_data = data["ate"]
                if isinstance(ate_data, dict):
                    ate_val = ate_data.get("value", 0)
                else:
                    ate_val = ate_data
                
                if isinstance(ate_val, (int, float)):
                    docs.append(f"전체 평균 처치 효과(ATE)는 {ate_val:.4f}입니다.")
                else:
                    docs.append(f"전체 평균 처치 효과(ATE)는 {ate_val}입니다.")
                    
                metas.append({"source": path.name, "key": "ate", "type": "metric"})

            # 2. CATE Stats
            if "cate_stats" in data:
                stats = data["cate_stats"]
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                # 안전한 변환
                if isinstance(mean_val, (int, float)):
                    mean_str = f"{mean_val:.4f}"
                else:
                    mean_str = str(mean_val)
                    
                if isinstance(std_val, (int, float)):
                    std_str = f"{std_val:.4f}"
                else:
                    std_str = str(std_val)
                    
                docs.append(f"개별 처치 효과(CATE)의 평균은 {mean_str}, 표준편차는 {std_str}입니다.")
                metas.append({"source": path.name, "key": "cate_stats", "type": "metric"})
                
            # 3. Debate
            if "debate" in data:
                debate = data["debate"]
                verdict = debate.get('verdict', 'Unknown')
                conf = debate.get('confidence', 0)
                docs.append(f"AI 토론 결과, 판결은 '{verdict}'이며 신뢰도는 {conf:.2f}입니다.")
                docs.append(f"AI 토론의 권고 사항: {debate.get('recommendation', 'N/A')}")
                metas.append({"source": path.name, "key": "debate_verdict", "type": "insight"})
                metas.append({"source": path.name, "key": "debate_recommendation", "type": "insight"})

            self.logger.info(f"JSON 로드 완료: {len(docs)} 청크")
            return docs, metas

        except Exception as e:
            self.logger.error(f"JSON 로드 실패: {e}")
            return [], []
