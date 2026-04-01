# -*- coding: utf-8 -*-
"""CausalCell — DML 코어 엔진 (EconML).

Double Machine Learning을 통해 CATE(Conditional Average Treatment Effect)를
추정하고, 평균 처치 효과(ATE) 및 95% 신뢰구간을 산출합니다.

DML 2단계 추정 과정:
    1) Y = q(W) + ε_Y  →  잔차 Ỹ = Y - q(W)
    2) T = p(W) + ε_T  →  잔차 T̃ = T - p(W)
    3) Ỹ = θ(X)·T̃ + ε  →  인과 효과 θ(X) 추정

입력 키 (DataCell 출력):
    - "dataframe": pd.DataFrame
    - "feature_names": list[str]
    - "treatment_col": str
    - "outcome_col": str

출력 키:
    - "ate": float (평균 처치 효과)
    - "ate_ci_lower": float (ATE 95% CI 하한)
    - "ate_ci_upper": float (ATE 95% CI 상한)
    - "cate_predictions": np.ndarray (개별 CATE 추정값)
    - "cate_ci_lower": np.ndarray (CATE 95% CI 하한)
    - "cate_ci_upper": np.ndarray (CATE 95% CI 상한)
    - "model": 학습된 DML 모델 객체
    - "model_type": str ("linear" | "forest")
    - "dataframe": pd.DataFrame (CATE 컬럼 추가된 원본)
    - "feature_names": list[str] (통과)
    - "treatment_col": str (통과)
    - "outcome_col": str (통과)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


class CausalCell(BaseCell):
    """DML 기반 인과 효과 추정 셀.

    LinearDML 또는 CausalForestDML을 사용하여
    교란 변수를 통제한 순수 처치 효과를 추정합니다.

    Args:
        config: WhyLab 전역 설정 객체.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="causal_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """DML 모델을 학습하고 CATE를 추정합니다.

        Args:
            inputs: DataCell의 출력 딕셔너리.
                필수 키: "dataframe", "feature_names", "treatment_col", "outcome_col"

        Returns:
            ATE, CATE, 신뢰구간, 학습된 모델을 담은 딕셔너리.
        """
        self.validate_inputs(
            inputs,
            ["dataframe", "feature_names", "treatment_col", "outcome_col"],
        )

        df: pd.DataFrame = inputs["dataframe"]
        feature_names: list[str] = inputs["feature_names"]
        treatment_col: str = inputs["treatment_col"]
        outcome_col: str = inputs["outcome_col"]
        cfg = self.config.dml

        scenario = inputs.get("scenario", "A")
        is_discrete = (scenario == "B")
        
        self.logger.info(
            "DML 모델 학습 시작 (시나리오=%s, 이산처치=%s, 타입=%s, CV=%d)",
            scenario, is_discrete, cfg.model_type, cfg.cv_folds,
        )

        # ──────────────────────────────────────────
        # 1. 데이터 준비
        # ──────────────────────────────────────────
        Y = df[outcome_col].values.astype(np.float64)
        T_raw = df[treatment_col].values.astype(np.float64)
        X = df[feature_names].values.astype(np.float64)

        # Treatment 정규화: 연속형(Scenario A)만 z-score 표준화.
        # 이산형(Scenario B, coupon_sent 0/1)은 정규화하면
        # DML discrete_treatment 모드에서 카테고리 에러 발생.
        if is_discrete:
            T = T_raw
            t_mean, t_std = 0.0, 1.0  # 역변환 불필요
        else:
            t_mean, t_std = float(T_raw.mean()), float(T_raw.std())
            T = (T_raw - t_mean) / t_std if t_std > 0 else T_raw

        # ──────────────────────────────────────────
        # 2. DML 모델 생성 및 학습 (AutoML 지원)
        # ──────────────────────────────────────────
        best_model = None
        best_rmse = float("inf")
        best_type = cfg.model_type

        # AutoML 모드: 후보 모델 경쟁
        if cfg.model_type == "auto" or cfg.auto_ml:
            self.logger.info("🤖 AutoML 시작: 후보 모델 %s", cfg.candidate_models)
            
            for model_name in cfg.candidate_models:
                self.logger.info("   >> 모델 평가 중: %s", model_name)
                # 임시 Config 생성하여 모델 생성 (model_type만 변경)
                temp_cfg = self.config.dml # 참조 copy (주의: dataclass는 mutable하므로 replace 활용 권장)
                # 여기서는 _create_model 내부 로직을 위해 직접 model string 전달하도록 수정하거나
                # 내부에서 분기 처리. _create_model 메서드 시그니처 변경 필요 없이,
                # cfg 객체의 model_type을 일시적으로 변경하는 방식 사용 (비권장이지만 간단함)
                
                # 안전한 방법: _create_model_by_name 메서드 분리
                model = self._create_model_by_name(model_name, cfg, is_discrete)
                
                # 학습 (Fit)
                model.fit(Y=Y, T=T, X=X)
                
                # 평가 (Self-Validation RMSE for CATE)
                # Ground Truth가 없으므로 R-Score 등을 써야 하지만,
                # 여기서는 학습된 모델의 잔차(Residual) 기반 점수 또는
                # 단순히 합성 데이터이므로 Ground Truth (만약 있다면)와 비교해야 함.
                # *Project Context*: 합성 데이터 생성 시 True CATE를 어딘가에 저장했다면 비교 가능.
                # DataCell에서 'true_cate' 컬럼을 만들어뒀는지 확인 필요. 없는 경우 OOB Score 사용.
                
                # 여기서는 간단히 모델의 score 메서드(R-Score 의미) 사용
                try:
                    score = model.score(Y=Y, T=T, X=X) # R^2 와 유사, 높을수록 좋음
                    rmse = -score # 편의상 RMSE처럼 낮을수록 좋은 지표로 변환 (R-Score는 최대화)
                except:
                    rmse = 0.0 # 에러 시 제외
                    
                self.logger.info("      Score(R-Risk): %.4f", rmse)
                
                if rmse < best_rmse: # R-Score는 높을수록 좋으므로, -score는 낮을수록 좋음
                    best_rmse = rmse
                    best_model = model
                    best_type = model_name
            
            self.logger.info("🏆 Best Model Selected: %s (Score=%.4f)", best_type, -best_rmse)
            model = best_model
            
        else:
            # 단일 모델 모드
            model = self._create_model_by_name(cfg.model_type, cfg, is_discrete)
            self.logger.info("DML fit 시작 (Cross-Fitting %d-fold)", cfg.cv_folds)
            model.fit(Y=Y, T=T, X=X)
            self.logger.info("DML fit completed")

        # ──────────────────────────────────────────
        # 4. ATE Estimation + Confidence Interval
        # ──────────────────────────────────────────
        ate = float(model.ate(X=X))
        ate_ci = model.ate_interval(X=X, alpha=cfg.alpha)
        ate_ci_lower = float(ate_ci[0])
        ate_ci_upper = float(ate_ci[1])

        self.logger.info(
            "ATE estimated: %.6f [%.6f, %.6f] (%.0f%% CI)",
            ate, ate_ci_lower, ate_ci_upper, (1 - cfg.alpha) * 100,
        )

        # ──────────────────────────────────────────
        # 5. CATE Estimation
        # ──────────────────────────────────────────
        cate_predictions = model.effect(X=X).flatten()
        cate_interval = model.effect_interval(X=X, alpha=cfg.alpha)
        cate_ci_lower = cate_interval[0].flatten()
        cate_ci_upper = cate_interval[1].flatten()

        self.logger.info(
            "CATE estimated: mean=%.6f, std=%.6f, range=[%.6f, %.6f]",
            cate_predictions.mean(),
            cate_predictions.std(),
            cate_predictions.min(),
            cate_predictions.max(),
        )

        # ──────────────────────────────────────────
        # 6. Append CATE to DataFrame
        # ──────────────────────────────────────────
        df = df.copy()
        df["estimated_cate"] = cate_predictions
        df["cate_ci_lower"] = cate_ci_lower
        df["cate_ci_upper"] = cate_ci_upper

        # ──────────────────────────────────────────
        # 7. Ground Truth 검증 (합성 데이터 전용)
        #    논문 수준 포트폴리오의 핵심: 추정(Estimated) vs 실제(True) 비교
        # ──────────────────────────────────────────
        estimation_accuracy = {}
        if "true_cate" in df.columns:
            true = df["true_cate"].values
            pred = cate_predictions

            rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
            mae = float(np.mean(np.abs(pred - true)))
            bias = float(np.mean(pred - true))
            # Coverage Rate: true_cate가 [CI_lower, CI_upper] 안에 있는 비율
            coverage = float(np.mean(
                (true >= cate_ci_lower) & (true <= cate_ci_upper)
            ))
            # 상관계수 (방향성 일치도)
            corr = float(np.corrcoef(pred, true)[0, 1]) if np.std(true) > 0 else 0.0

            estimation_accuracy = {
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "coverage_rate": coverage,
                "correlation": corr,
                "n_samples": len(true),
            }

            self.logger.info(
                "📊 Ground Truth 검증: RMSE=%.4f, MAE=%.4f, Bias=%.4f, "
                "Coverage=%.1f%%, Corr=%.3f",
                rmse, mae, bias, coverage * 100, corr,
            )

        return {
            "ate": ate,
            "ate_ci_lower": ate_ci_lower,
            "ate_ci_upper": ate_ci_upper,
            "cate_predictions": cate_predictions,
            "cate_ci_lower": cate_ci_lower,
            "cate_ci_upper": cate_ci_upper,
            "model": model,
            "model_type": best_type if (cfg.model_type == "auto" or cfg.auto_ml) else cfg.model_type,
            "dataframe": df,
            "feature_names": feature_names,
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "estimation_accuracy": estimation_accuracy,
        }

    def _create_model_by_name(self, model_type: str, cfg: Any, discrete_treatment: bool) -> Any:
        """Create a DML model by name (for AutoML support)."""
        from lightgbm import LGBMRegressor

        model_y = LGBMRegressor(
            n_estimators=cfg.lgbm_n_estimators,
            learning_rate=cfg.lgbm_learning_rate,
            verbose=-1,
        )
        model_t = LGBMRegressor(
            n_estimators=cfg.lgbm_n_estimators,
            learning_rate=cfg.lgbm_learning_rate,
            verbose=-1,
        )

        if model_type == "linear":
            from econml.dml import LinearDML
            return LinearDML(
                model_y=model_y, model_t=model_t,
                discrete_treatment=discrete_treatment,
                cv=cfg.cv_folds, random_state=42,
            )
        elif model_type == "forest":
            from econml.dml import CausalForestDML
            return CausalForestDML(
                model_y=model_y, model_t=model_t,
                discrete_treatment=discrete_treatment,
                cv=cfg.cv_folds, n_estimators=100, random_state=42,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _create_model(self, cfg: Any, discrete_treatment: bool = False) -> Any:
        """설정에 따라 DML 모델 객체를 생성합니다.

        Args:
            cfg: DMLConfig 설정 객체.
            discrete_treatment: 처치 변수가 이산형인지 여부.

        Returns:
            EconML DML 모델 인스턴스.

        Raises:
            ValueError: 지원하지 않는 model_type인 경우.
        """
        from lightgbm import LGBMClassifier, LGBMRegressor
        from engine.gpu_factory import create_lgbm_regressor

        # Nuisance 모델 정의 (GPU 가속)
        model_y = create_lgbm_regressor(self.config)
        model_t = create_lgbm_regressor(self.config)

        if cfg.model_type == "linear":
            from econml.dml import LinearDML

            self.logger.info("LinearDML 모델 생성")
            return LinearDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=discrete_treatment,
                cv=cfg.cv_folds,
                random_state=self.config.data.random_seed,
            )

        elif cfg.model_type == "forest":
            from econml.dml import CausalForestDML

            self.logger.info("CausalForestDML 모델 생성")
            return CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=discrete_treatment,
                cv=cfg.cv_folds,
                n_estimators=200,
                random_state=self.config.data.random_seed,
            )

        else:
            raise ValueError(
                f"지원하지 않는 model_type: '{cfg.model_type}'. "
                "'linear' 또는 'forest'만 지원합니다."
            )
