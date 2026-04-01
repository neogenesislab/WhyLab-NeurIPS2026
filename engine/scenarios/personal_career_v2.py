from engine.core.scenario import Scenario
from engine.agents.literature_reviewer import LiteratureReviewer
import networkx as nx
import numpy as np

class PersonalCareerScenario(Scenario):
    """
    [Scenario C] Personal Career & Finance (Synthetic Simulation)
    - Goal: Analyze causal effect of Skill Acquisition on Market Value.
    - Data Source: Synthetic Data Generator (No local data access).
    """
    def __init__(self):
        super().__init__("Personal_Career_Analysis")
        self.reviewer = LiteratureReviewer()
        self.knowledge_summary = "Synthetic Research Mode: Analyzing theoretical impact of continuous learning."

    def load_data(self):
        # 1. Synthetic Data Generation Only
        print("[Scenario] Generating SYNTHETIC career data (Privacy Mode)...")
        self.knowledge_summary = self.reviewer.review_topic("career")
        
        # 2. Extract Variables
        self.variables = ["Skill_Level", "Project_Completion", "Market_Value", "Passive_Income"]
        self.data = self._generate_mock_data_based_on_knowledge()

    def _generate_mock_data_based_on_knowledge(self):
        # Generate synthetic data that reflects the user's profile found in knowledge
        n = 1000
        skill = np.random.normal(0.7, 0.1, n) # High skill base
        project = 0.6 * skill + np.random.normal(0, 0.1, n)
        market_value = 0.8 * skill + 0.5 * project + np.random.normal(0, 0.1, n)
        passive = 0.3 * market_value + np.random.normal(0, 0.05, n)
        
        return {
            "Skill_Level": skill,
            "Project_Completion": project,
            "Market_Value": market_value,
            "Passive_Income": passive
        }

    def create_causal_graph(self):
        # Define causal relationships
        G = nx.DiGraph()
        G.add_edges_from([
            ("Skill_Level", "Project_Completion"),
            ("Skill_Level", "Market_Value"),
            ("Project_Completion", "Market_Value"),
            ("Market_Value", "Passive_Income")
        ])
        return G

    def get_description(self):
        return f"""
        # Personal Career Analysis
        
        ## Data Source
        - Local Knowledge Base (Safe Mode)
        - Extracted Topics: 'career', 'finance', 'salary'
        
        ## Knowledge Context
        {self.knowledge_summary[:300]}... (See logs for full details)
        
        ## Causal Hypothesis
        - Higher **Skill Level** directly increases **Market Value**.
        - **Project Completion** is a mediator for Market Value.
        """
