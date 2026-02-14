# -*- coding: utf-8 -*-
"""E2E Pipeline Test.

Phase 2-3 기능(Debate, MetaLearner 등)을 포함한
전체 파이프라인의 통합 실행을 검증합니다.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from engine.orchestrator import Orchestrator


class TestE2EPipeline(unittest.TestCase):
    """E2E 파이프라인 실행 테스트."""

    def setUp(self):
        """테스트 환경 설정 (임시 디렉토리)."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Orchestrator 초기화
        self.orchestrator = Orchestrator()
        
        # 설정 수정 (Fast Mode for Testing)
        self.orchestrator.config.paths.project_root = Path(self.temp_dir)
        (Path(self.temp_dir) / "dashboard" / "public" / "data").mkdir(parents=True)
        (Path(self.temp_dir) / "paper" / "data").mkdir(parents=True)
        (Path(self.temp_dir) / "paper" / "reports").mkdir(parents=True)

        # 테스트 속도를 위해 반복 횟수 최소화
        # 1. Debate
        self.orchestrator.config.debate.max_rounds = 1
        # self.orchestrator.config.debate.evidence_per_agent = 1  # (옵션이 있다면)

        # 2. Sensitivity & Refutation (Config 경로 주의)
        self.orchestrator.config.sensitivity.n_simulations = 2
        self.orchestrator.config.sensitivity.n_bootstrap = 2
        self.orchestrator.config.sensitivity.n_refutation_iter = 2
        
        # 3. MetaLearner (DML Config)
        self.orchestrator.config.dml.cv_folds = 2
        
        # LLM 사용 비활성화 및 환경변수 체크
        if "OPENAI_API_KEY" not in os.environ:
             print("Warning: OPENAI_API_KEY not found. Skipping LLM-dependent tests might be needed.")

    def tearDown(self):
        """임시 디렉토리 삭제."""
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline_scenario_A(self):
        """Scenario A 전체 파이프라인 실행 검증."""
        try:
            results = self.orchestrator.run_pipeline(scenario="A")
            
            # 1. 주요 결과 키 확인
            required_keys = [
                "ate", "ate_ci_lower", "ate_ci_upper",
                "meta_learner_results",
                "conformal_results",
                "debate_verdict", "debate_summary",
                "json_path"
            ]
            for key in required_keys:
                self.assertIn(key, results, f"Missing key: {key}")
                
            # 2. 값 유효성 검증
            self.assertIsInstance(results["ate"], float)
            self.assertNotEqual(results["ate"], 0.0)
            
            # 3. 파일 생성 확인
            self.assertTrue(os.path.exists(results["json_path"]))
            
            # 4. Debate 결과 구조 확인
            debate = results["debate_summary"]
            self.assertIn("verdict", debate)
            self.assertIn("pro_evidence", debate)
            self.assertIn("con_evidence", debate)
            
            # 5. Conformal 결과 확인
            conformal = results["conformal_results"]
            self.assertIn("ci_lower_mean", conformal)
            self.assertIn("ci_upper_mean", conformal)
            
            # 6. 리포트 생성 확인
            report_path = results.get("report_path")
            self.assertTrue(report_path and os.path.exists(report_path))
            
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Debate Verdict", content)
                self.assertIn("Conformal Prediction", content)
                self.assertIn("Statistical Diagnostics", content)
            
        except Exception as e:
            self.fail(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    unittest.main()
