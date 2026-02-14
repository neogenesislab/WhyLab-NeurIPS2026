# -*- coding: utf-8 -*-
"""ReportCell — 실험 결과 자동 리포팅 + LLM 자연어 해석.

분석 결과를 종합하여 Markdown 형식의 리포트를 자동으로 생성합니다.
Phase 3 기능(Debate, Conformal, Benchmark) 결과를 포함하여
심도 있는 인과 추론 리포트를 제공합니다.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional

import numpy as np

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


class ReportCell(BaseCell):
    """분석 결과를 Markdown 리포트 + AI 인사이트로 변환하는 셀."""

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="report_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """분석 결과를 바탕으로 종합 리포트를 생성합니다."""
        self.logger.info("리포트 생성 시작")

        # 1. 데이터 추출 (Phase 2 & 3)
        ate = inputs.get("ate", 0.0)
        ate_ci_lower = inputs.get("ate_ci_lower", 0.0)
        ate_ci_upper = inputs.get("ate_ci_upper", 0.0)
        cate_preds = inputs.get("cate_predictions", np.array([]))
        feature_names = inputs.get("feature_names", [])
        scenario = inputs.get("scenario_name", "Unknown Scenario")
        
        # Phase 3 Data
        debate = inputs.get("debate_summary", {})
        conformal = inputs.get("conformal_results", {})
        benchmark = inputs.get("benchmark_results", {})
        sensitivity = inputs.get("sensitivity_results", {})
        est_acc = inputs.get("estimation_accuracy", {})
        feat_imp = inputs.get("feature_importance", {})

        # Stats
        cate_mean = float(np.mean(cate_preds)) if len(cate_preds) > 0 else 0.0
        cate_std = float(np.std(cate_preds)) if len(cate_preds) > 0 else 0.0
        n_samples = len(inputs.get("dataframe", []))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 2. AI 인사이트 생성
        ai_insights = self._generate_insights(
            scenario=scenario, ate=ate, ci=(ate_ci_lower, ate_ci_upper),
            debate=debate, est_acc=est_acc
        )

        # 3. Markdown 리포트 생성
        report_content = self._generate_markdown(
            timestamp=timestamp, scenario=scenario,
            ate=ate, ci=(ate_ci_lower, ate_ci_upper),
            cate_stats={"mean": cate_mean, "std": cate_std},
            features=feature_names, n_samples=n_samples,
            debate=debate, conformal=conformal, benchmark=benchmark,
            sensitivity=sensitivity, est_acc=est_acc, ai_insights=ai_insights
        )

        # 4. 파일 저장
        output_dir = self.config.paths.reports_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"whylab_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        file_path = output_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        self.logger.info("리포트 저장 완료: %s", file_path)

        return {
            "report_path": str(file_path),
            "report_content": report_content,
            "ai_insights": ai_insights,
        }

    def _generate_insights(self, scenario, ate, ci, debate, est_acc) -> Dict[str, Any]:
        """간단한 규칙 기반 인사이트 생성 (LLM 연동은 필요시 복구 가능)."""
        is_sig = not (ci[0] <= 0 <= ci[1])
        verdict = debate.get("verdict", "UNKNOWN")
        
        summary = (
            f"**{scenario}** 분석 결과, 처치 효과(ATE)는 {ate:.4f}로 추정되었으며, "
            f"통계적으로 {'유의합니다' if is_sig else '유의하지 않습니다'}. "
            f"AI Debate 시스템은 이를 **{verdict}**로 판정했습니다."
        )
        
        return {
            "summary": summary,
            "headline": f"ATE {ate:.4f} ({verdict})",
            "generated_by": "rule_based"
        }

    def _generate_markdown(
        self, timestamp, scenario, ate, ci, cate_stats, features, n_samples,
        debate, conformal, benchmark, sensitivity, est_acc, ai_insights
    ) -> str:
        """Markdown 리포트 본문 생성."""
        
        # Helper for significance
        is_sig = not (ci[0] <= 0 <= ci[1])
        sig_icon = "✅" if is_sig else "⚠️"
        
        # Debate Section
        debate_section = ""
        if debate:
            d_verdict = debate.get("verdict", "UNKNOWN")
            d_conf = debate.get("confidence", 0) * 100
            d_rec = debate.get("recommendation", "-")
            d_rounds = debate.get("rounds", 0)
            
            debate_section = f"""
## 2. 🏛️ Causal Debate Verdict
> **Verdict**: **{d_verdict}** (Confidence: {d_conf:.1f}%) in {d_rounds} rounds.

- **Recommendation**: {d_rec}
- **Score**: Pro {debate.get('pro_score', 0):.1f} vs Con {debate.get('con_score', 0):.1f}
"""

        # Diagnostics Section
        diag_section = ""
        if sensitivity:
            s_status = sensitivity.get("status", "Unknown")
            e_val = sensitivity.get("e_value", {}).get("point", 0) if isinstance(sensitivity.get("e_value"), dict) else 0
            ov_score = sensitivity.get("overlap", {}).get("overlap_score", 0) if isinstance(sensitivity.get("overlap"), dict) else 0
            
            diag_section = f"""
## 3. 🛡️ Statistical Diagnostics (Robustness)
- **Overall Status**: {s_status}
- **E-value**: {e_val:.2f} (Target > 1.5)
- **Overlap Score**: {ov_score:.2f} (Target > 0.5)
- **Placebo Test**: {sensitivity.get('placebo_test', {}).get('status', 'N/A')}
- **Refutation**: {sensitivity.get('random_common_cause', {}).get('status', 'N/A')}
"""

        # Conformal Section
        conf_section = ""
        if conformal:
            cov = conformal.get("coverage", 0) * 100
            width = conformal.get("ci_upper_mean", 0) - conformal.get("ci_lower_mean", 0)
            conf_section = f"""
## 4. 📏 Conformal Prediction (Reliability)
- **Target Coverage**: 95%
- **Actual Coverage**: **{cov:.1f}%** (Valid if near 95%)
- **Avg CI Width**: {width:.4f}
"""
        
        # Benchmark Section
        bench_section = ""
        if benchmark:
            # 첫 번째 데이터셋 결과만 예시로
            ds_name = list(benchmark.keys())[0] if benchmark else "N/A"
            methods = benchmark.get(ds_name, {})
            rows = []
            for m_name, metrics in methods.items():
                rows.append(f"| {m_name} | {metrics['pehe_mean']:.4f} | {metrics['ate_bias_mean']:.4f} |")
            
            table_body = "\n".join(rows)
            bench_section = f"""
## 5. 🏆 Benchmark Performance ({ds_name})
| Method | √PEHE (Lower is better) | ATE Bias |
|--------|--------------------------|----------|
{table_body}
"""

        return f"""# 🧪 WhyLab Comprehensive Analysis Report

**Date**: {timestamp}  
**Scenario**: {scenario}  
**Samples**: {n_samples:,}  

---

## 1. 📋 Executive Summary
{sig_icon} **Treatment Effect (ATE)**: {ate:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])  
{ai_insights['summary']}

> **Model Quality**: Correlation with Ground Truth = **{est_acc.get('correlation', 0):.3f}**

{debate_section}
{diag_section}
{conf_section}

## 6. Heterogeneity Analysis (CATE)
- **Mean CATE**: {cate_stats['mean']:.4f}
- **Std Dev**: {cate_stats['std']:.4f}

{bench_section}

---
*Generated by: WhyLab Causal Engine (Phase 3)*
"""
