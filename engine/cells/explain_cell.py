# -*- coding: utf-8 -*-
"""ExplainCell — SHAP 설명 및 반사실 시뮬레이션.

머신러닝 모델의 "블랙박스" 속성을 해결하고,
인과추론 결과의 신뢰도를 높이기 위한 설명 가능한 AI(XAI) 셀입니다.

기능:
    1. SHAP (SHapley Additive exPlanations):
       - 모델이 왜 그런 CATE를 예측했는지 변수별 기여도 분석.
    2. 반사실(Counterfactual) 시뮬레이션:
       - "만약 이 유저의 소득이 10% 올랐다면, CATE는 어떻게 변했을까?"
       - "만약 투자를 안 했다면(T=0), 결과는 어땠을까?" (Outcome 모델 활용)

입력 키 (CausalCell 출력):
    - "model": 학습된 DML 모델
    - "dataframe": 전체 데이터
    - "feature_names": 교란 변수 목록
    - "treatment_col": 처치 변수명
    - "outcome_col": 결과 변수명

출력 키:
    - "shap_values": SHAP 값 배열
    - "feature_importance": 변수 중요도 딕셔너리
    - "counterfactuals": 시뮬레이션 결과 리스트
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


class ExplainCell(BaseCell):
    """모델 설명 및 반사실 시뮬레이션 셀.

    Args:
        config: WhyLab 전역 설정 객체.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="explain_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """SHAP 분석 및 반사실 시뮬레이션 실행.

        Args:
            inputs: CausalCell 출력.

        Returns:
            설명 및 시뮬레이션 결과.
        """
        self.validate_inputs(inputs, ["model", "dataframe", "feature_names"])

        model = inputs["model"]
        df = inputs["dataframe"]
        feature_names = inputs["feature_names"]
        cfg = self.config.explain

        self.logger.info("ExplainCell 실행 시작")

        # ──────────────────────────────────────────
        # 1. SHAP 피처 중요도 (CATE 모델 설명)
        # ──────────────────────────────────────────
        # EconML 모델의 최종 CATE 예측 모델(cate_model)을 설명
        # 대부분의 CATE 모델은 내부적으로 sklearn/LGBM 등을 사용
        
        shap_values = None
        feature_importance = {}

        try:
            # EconML 모델의 effect() 함수를 SHAP 함수 설명자로 래핑
            # 이 방식은 LinearDML, CausalForestDML 등 모든 EconML 모델에 범용 적용
            X_shap = df[feature_names].sample(
                n=min(cfg.shap_sample_size, len(df)),
                random_state=self.config.data.random_seed
            ).values.astype(np.float64)

            # model.effect()를 SHAP가 이해하는 함수로 래핑
            def predict_cate(X_input):
                return model.effect(X_input).flatten()

            # 배경 데이터(background) 샘플링 — KernelExplainer 사용
            background = shap.sample(X_shap, min(100, len(X_shap)))
            explainer = shap.KernelExplainer(predict_cate, background)
            shap_values_raw = explainer.shap_values(X_shap[:min(200, len(X_shap))])

            # 평균 절대 SHAP 값으로 중요도 산출
            vals = np.abs(shap_values_raw).mean(0)
            feature_importance = dict(zip(feature_names, vals.tolist()))

            self.logger.info("SHAP 분석 완료: 중요도=%s", feature_importance)

        except Exception as e:
            self.logger.warning("SHAP 분석 실패 (무시됨): %s", e)

        # ──────────────────────────────────────────
        # 2. 반사실적(Counterfactual) 시뮬레이션
        # "소득이 상위 10%인 유저들이 만약 소득이 평균이었다면?"
        # ──────────────────────────────────────────
        counterfactuals = []
        
        if "batch_simulation" in inputs: # 미래 확장성
            pass
        else:
            # 데모용: 상위 5명 유저에 대해 소득 -50% 시나리오
            target_users = df.nlargest(5, "income").copy()
            original_cate = model.effect(target_users[feature_names])
            
            # 반사실 조작: 소득 절반 감소
            target_users_cf = target_users.copy()
            target_users_cf["income"] = target_users_cf["income"] * 0.5
            cf_cate = model.effect(target_users_cf[feature_names])
            
            for i in range(len(target_users)):
                uid = target_users.iloc[i].get("user_id", i)
                diff = cf_cate[i] - original_cate[i]
                counterfactuals.append({
                    "user_id": int(uid),
                    "original_cate": float(original_cate[i]),
                    "counterfactual_cate": float(cf_cate[i]),
                    "diff": float(diff),
                    "description": "소득 50% 감소 시 CATE 변화"
                })
        
        self.logger.info("반사실 시뮬레이션 완료 (%d건)", len(counterfactuals))

        # 기존 입력값 통과 (Pipeline Flow)
        return {
            **inputs,
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "counterfactuals": counterfactuals,
        }
