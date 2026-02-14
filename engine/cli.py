# -*- coding: utf-8 -*-
"""WhyLab Engine Entrypoint (CLI Alternative)."""

import argparse
import sys
import io
from pathlib import Path

# Windows 콘솔 인코딩 호환성 확보
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from engine.config import DEFAULT_CONFIG
from engine.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="WhyLab Causal Inference Engine")
    
    # 데이터 관련 인자
    parser.add_argument("--data", type=str, help="External CSV Data Path")
    parser.add_argument("--treatment", type=str, default="treatment", help="Treatment Column")
    parser.add_argument("--outcome", type=str, default="outcome", help="Outcome Column")
    parser.add_argument("--features", type=str, help="Feature Columns (comma-separated)")
    
    # 실행 옵션
    parser.add_argument("--scenario", type=str, default="Scenario A", help="Scenario Name")
    parser.add_argument("--query", type=str, help="RAG: Ask a question about the analysis.")
    
    args = parser.parse_args()
    
    config = DEFAULT_CONFIG
    
    print("WhyLab Engine CLI (v2) Initializing...")

    # 1. RAG Query 모드
    if args.query:
        print(f"Query: {args.query}")
        try:
            from engine.rag.agent import RAGAgent
            agent = RAGAgent(config)
            
            # 지식 인덱싱 (최신 리포트 반영)
            print("indexing knowledge...")
            agent.index_knowledge()
            
            print("generating answer...")
            answer = agent.ask(args.query)
            print(f"\nAnswer:\n{answer}\n")
            return
        except ImportError:
            print("RAG Module Error: pip install chromadb sentence-transformers")
            sys.exit(1)
        except Exception as e:
            print(f"RAG Error: {e}")
            sys.exit(1)
    
    # 2. 외부 데이터 설정 적용
    if args.data:
        print(f"External Data Mode: {args.data}")
        config.data.input_path = args.data
        config.data.treatment_col = args.treatment
        config.data.outcome_col = args.outcome
        
        if args.features:
            config.data.feature_cols = [f.strip() for f in args.features.split(",")]
            print(f"Selected Features: {config.data.feature_cols}")
    else:
        print(f"Synthetic Data Mode: {args.scenario}")

    # 3. 파이프라인 실행
    orchestrator = Orchestrator(config)
    
    try:
        orchestrator.run_pipeline(scenario=args.scenario)
        print("Pipeline Completed Successfully via CLI.")
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
