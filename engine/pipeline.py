# -*- coding: utf-8 -*-
"""WhyLab 파이프라인 진입점.

커맨드라인에서 실행 가능한 스크립트입니다.
사용법:
    python -m engine.pipeline --scenario A
    python -m engine.pipeline --scenario B
"""

import argparse
import logging
import sys
import warnings

# EconML/sklearn이 반복적으로 발생시키는 수치 경고 억제
# (Co-variance underdetermined, feature name mismatch 등)
warnings.filterwarnings("ignore", category=UserWarning, module="econml")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from engine.orchestrator import Orchestrator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="WhyLab Causal Inference Pipeline")
    parser.add_argument(
        "--scenario", 
        choices=["A", "B"], 
        default="A", 
        help="실행할 시나리오 (A: 신용한도, B: 마케팅)"
    )
    args = parser.parse_args()

    orchestrator = Orchestrator()
    try:
        orchestrator.run_pipeline(scenario=args.scenario)
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
