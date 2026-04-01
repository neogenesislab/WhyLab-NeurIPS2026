import random

class KnowledgeLoader:
    def __init__(self, base_path: str = None):
        # SECURITY UPDATE: Removed local file access completely based on user request.
        # This module now strictly generates synthetic data for simulation purposes.
        self.base_path = "SYNTHETIC_MODE"

    def find_artifacts(self, keyword: str) -> list:
        return []

    def load_content(self, file_path: str) -> str:
        return ""
    
    def _sanitize_pii(self, text: str) -> str:
        return text

    def get_summary(self, keyword: str) -> str:
        """
        Returns synthetic research summaries instead of reading local files.
        """
        if keyword.lower() in ["career", "salary", "finance"]:
            return self._generate_synthetic_career_summary()
        elif keyword.lower() in ["health", "medical"]:
             return self._generate_synthetic_health_summary()
        else:
            return f"Synthetic research summary for '{keyword}': No specific pattern found, assuming random walk."

    def _generate_synthetic_career_summary(self):
        return """
        [Synthetic Research Output]
        Title: Impact of Continuous Learning on Tech Salary (Simulation)
        Summary: 
        - Analysis of 50,000 synthetic developer profiles suggests a 15% income boost per new skill acquired.
        - 'Project Completion' is highly correlated (0.85) with 'Market Value'.
        - This data is computer-generated and does not reflect any real individual.
        """

    def _generate_synthetic_health_summary(self):
        return """
        [Synthetic Research Output]
        Title: Circadian Rhythms and Productivity
        Summary: Simulated data shows 20% efficiency drop after 10 hours of work.
        """
