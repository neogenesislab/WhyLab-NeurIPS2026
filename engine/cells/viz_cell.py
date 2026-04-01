# -*- coding: utf-8 -*-
"""VizCell — 정적 시각화 이미지 생성.

matplotlib과 seaborn을 사용하여 보고서(Report)용 고품질 이미지를 생성합니다.
대시보드는 JSON 데이터를 받아 동적으로 렌더링하므로,
이 셀은 주로 "White Paper"나 "README"에 들어갈 정적 이미지를 담당합니다.

출력: paper/figures/*.png

기능:
    1. CATE 분포 히스토그램
    2. 주요 변수 vs CATE 산점도 (Trendline 포함)
    3. SHAP Summary Plot (이미지 저장)
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


class VizCell(BaseCell):
    """정적 시각화 및 이미지 저장 셀.

    Args:
        config: WhyLab 전역 설정 객체.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="viz_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """시각화 생성 및 이미지 저장.

        Args:
            inputs: CausalCell/ExplainCell 출력.

        Returns:
            저장된 이미지 파일 경로 목록을 포함한 딕셔너리.
        """
        self.validate_inputs(inputs, ["dataframe", "cate_predictions"])
        
        df = inputs["dataframe"]
        cate = inputs["cate_predictions"]
        scenario = inputs.get("scenario", "A")
        
        output_dir = self.config.paths.figures_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_figures = []

        # 한글 폰트 설정 (Windows)
        plt.rcParams['font.family'] = self.config.viz.font_family
        plt.rcParams['axes.unicode_minus'] = False

        self.logger.info("VizCell 실행 시작 (시나리오 %s)", scenario)

        # ──────────────────────────────────────────
        # 1. CATE 분포 히스토그램 (공통)
        # ──────────────────────────────────────────
        plt.figure(figsize=(10, 6))
        sns.histplot(cate, kde=True, bins=50, color='skyblue')
        plt.title(f"CATE(조건부 처치 효과) 분포 - 시나리오 {scenario}")
        plt.xlabel("CATE")
        plt.ylabel("Frequency")
        
        path = output_dir / f"cate_distribution_{scenario}.png"
        plt.savefig(path, dpi=self.config.viz.figure_dpi, bbox_inches='tight')
        plt.close()
        saved_figures.append(str(path))

        # ──────────────────────────────────────────
        # 2. 시나리오별 특화 그래프
        # ──────────────────────────────────────────
        if scenario == "A":
            # 신용점수 vs CATE 산점도
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=df.sample(min(1000, len(df))), 
                x="credit_score", y="estimated_cate", 
                alpha=0.5, hue="income"
            )
            plt.title("신용점수에 따른 처치 효과(CATE) 변화")
            path = output_dir / "cate_by_credit_score.png"
            plt.savefig(path, dpi=self.config.viz.figure_dpi, bbox_inches='tight')
            plt.close()
            saved_figures.append(str(path))

        elif scenario == "B":
            # 연령대별 CATE 막대 그래프
            # (DataCell에서 age 컬럼 보장됨)
            if "age" in df.columns:
                df["age_group"] = pd.cut(df["age"], bins=[20, 30, 40, 50, 60, 70], labels=["20대", "30대", "40대", "50대", "60대"])
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x="age_group", y="estimated_cate", errorbar="sd")
                plt.title("연령대별 투자 쿠폰 효과(CATE)")
                path = output_dir / "cate_by_age_group.png"
                plt.savefig(path, dpi=self.config.viz.figure_dpi, bbox_inches='tight')
                plt.close()
                saved_figures.append(str(path))

        self.logger.info("이미지 저장 완료: %d장", len(saved_figures))

        return {
            **inputs,
            "saved_figures": saved_figures
        }
