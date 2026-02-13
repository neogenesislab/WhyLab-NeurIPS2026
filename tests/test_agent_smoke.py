"""에이전트 단독 실행 스모크 테스트."""
import logging
logging.basicConfig(level=logging.INFO)

from engine.config import WhyLabConfig
from engine.cells.data_cell import DataCell
from engine.agents.discovery import DiscoveryAgent

config = WhyLabConfig()
data_out = DataCell(config).execute({"scenario": "A"})
agent = DiscoveryAgent(config)
dag = agent.discover(data_out["dataframe"], data_out)

print("\n=== Discovered DAG ===")
for u, v in dag.edges():
    print(f"  {u} -> {v}")
print(f"Total: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
