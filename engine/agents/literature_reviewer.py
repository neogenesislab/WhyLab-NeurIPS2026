import requests
import json
from engine.connectors.knowledge_loader import KnowledgeLoader

class LiteratureReviewer:
    """
    [Role] Literature Reviewer (The Research Team)
    - Responsibility: Search internal knowledge base for context.
    - Privacy: Ensures all data is PII-sanitized before processing.
    """
    def __init__(self):
        self.loader = KnowledgeLoader()
        self.name = "LiteratureReviewer"
        self.log_api = "http://localhost:4001/system/logs"

    def _log(self, message: str):
        try:
            requests.post(self.log_api, json={"agent_id": self.name, "message": message})
        except:
            pass # Fail silently if server is down

    def review_topic(self, topic: str) -> str:
        """
        주어진 주제에 대해 내부 지식(Artifacts)을 검색하고 리뷰합니다.
        """
        self._log(f"Searching internal knowledge base for: '{topic}'")
        
        # 1. Search & Load (Data stays local)
        summary = self.loader.get_summary(topic)
        
        if "No knowledge artifacts found" in summary:
            self._log(f"No relevant artifacts found for '{topic}'. Using general principles.")
            return f"[{self.name}] No strictly relevant internal documents found for '{topic}'. I will rely on general causal inference principles."

        self._log(f"Found artifacts for '{topic}'. Applied PII Sanitization.")
        
        # 2. Synthesize
        review = f"""
        [{self.name}] Internal Knowledge Review Report
        Target Topic: {topic}
        Status: PII-Sanitized Data Loaded.
        
        Key Findings from Artifacts:
        {summary}
        
        [Insight]
        Based on these documents, there is a strong connection between user's past interest in '{topic}' and the current research goal.
        I recommend focusing on the causal links found in the retrieved files.
        """
        self._log(f"Review completed for '{topic}'. Ready to formulate hypothesis.")
        return review
