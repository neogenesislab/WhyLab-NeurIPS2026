# -*- coding: utf-8 -*-
"""DataCell — SCM 기반 합성 데이터 생성 + DuckDB SQL 탐색.

WhyLab 파이프라인의 첫 번째 셀.
구조적 인과 모델(SCM)에 기반한 현실적인 핀테크 합성 데이터를 생성합니다.

시나리오 설명:
    A. 신용 한도(T) → 연체 여부(Y) (Existing)
    B. 투자 쿠폰(T) → 투자 가입 여부(Y) (New - HTE 분석용)

DuckDB 활용:
    - 대용량 데이터 로드 및 Window Function 기반 시계열 피처 엔지니어링 수행.
    - AVG(spend) OVER ... 등의 쿼리 실행.

출력 키:
    - "dataframe": pd.DataFrame (통합 데이터)
    - "feature_names": list[str]
    - "treatment_col": str
    - "outcome_col": str
    - "true_cate_col": str
    - "dag_edges": list[tuple]
    - "scenario": str ("A" or "B")
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import os
import glob

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig


def _normalize(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    if std < 1e-8:
        return arr - arr.mean()
    return (arr - arr.mean()) / std


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class DataCell(BaseCell):
    """SCM 기반 합성 핀테크 데이터를 생성하고 DuckDB로 전처리하는 셀."""

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="data_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """데이터 생성 및 전처리 실행.

        Args:
            inputs:
                - scenario (str, optional): "A"(신용) 또는 "B"(마케팅). 기본값 "A".

        Returns:
            생성된 데이터셋 및 메타데이터.
        """
        scenario = inputs.get("scenario", "A")
        
        # 0.1. CSV 파일 자동 감지 (engine/data/*.csv)
        # 주의: WHYLAB_AUTO_CSV=1 일 때만 활성화. CI/테스트 환경에서
        # 의도하지 않은 CSV 로드로 시나리오 분기가 우회되는 것을 방지.
        auto_csv_enabled = os.environ.get("WHYLAB_AUTO_CSV", "0") == "1"
        if not self.config.data.input_path and auto_csv_enabled:
            local_csv = self._detect_local_csv()
            if local_csv:
                self.config.data.input_path = local_csv
                self.logger.info("📂 로컬 CSV 자동 감지: %s", local_csv)
        
        # 0. 외부 데이터 로드 (CLI 인자 우선)
        if self.config.data.input_path:
            self.logger.info("📂 외부 데이터 로드: %s", self.config.data.input_path)
            df_final, meta = self._load_external_data()
            # 외부 데이터는 시나리오 이름 오버라이드
            scenario = "External Data"
        else:
            self.logger.info("시나리오 %s 데이터 생성 시작", scenario)

            # 1. 기본 외생 변수 생성
            df_base = self._generate_base_features()
    
            # 2. 시나리오별 처리
            if scenario == "A":
                df_final, meta = self._generate_scenario_A(df_base)
            elif scenario == "B":
                df_final, meta = self._generate_scenario_B(df_base)
            else:
                raise ValueError(f"지원하지 않는 시나리오: {scenario}")

        # 3. DuckDB 전처리 (Window Function 등)
        # 외부 데이터라도 DuckDB 전처리를 통과시켜 일관성 유지 (선택적)
        try:
            df_processed = self._apply_duckdb_preprocessing(df_final)
        except Exception as e:
            self.logger.warning("DuckDB 전처리 실패 (외부 데이터 구조 불일치?): %s", e)
            df_processed = df_final

        # 메타데이터 업데이트
        meta["dataframe"] = df_processed
        
        # DuckDB로 추가된 컬럼이 있다면 feature_names에 추가
        if "feature_names" in meta:
            new_cols = set(df_processed.columns) - set(df_final.columns)
            if new_cols:
                meta["feature_names"] = list(set(meta["feature_names"]) | new_cols)
        
        meta["scenario"] = scenario

        self.logger.info(
            "최종 데이터: shape=%s, 컬럼=%s",
            df_processed.shape, list(df_processed.columns)
        )

        return meta

    def _load_external_data(self) -> Tuple[pd.DataFrame, Dict]:
        """설정된 경로/URI에서 외부 데이터를 로드합니다.
        
        커넥터 팩토리를 통해 CSV/Parquet/SQL/BigQuery 등 다양한 소스를 지원합니다.
        """
        cfg = self.config.data
        input_path = cfg.input_path
        
        # URI 패턴으로 소스 타입 자동 감지
        source_type = self._detect_source_type(input_path)
        
        try:
            from engine.connectors import create_connector, ConnectorConfig
            
            connector_config = ConnectorConfig(
                source_type=source_type,
                uri=input_path,
                query=getattr(cfg, 'query', None),
                table=getattr(cfg, 'table', None),
                treatment_col=cfg.treatment_col,
                outcome_col=cfg.outcome_col,
                feature_cols=cfg.feature_cols or [],
            )
            
            connector = create_connector(connector_config)
            with connector:
                result = connector.fetch_with_meta()
            
            df = result.pop("dataframe")
            self.logger.info(
                "%s 커넥터 로드 성공: %d rows", source_type.upper(), len(df)
            )
            return df, result
            
        except ImportError:
            # 커넥터 모듈 없으면 기존 로직 폴백
            self.logger.warning("커넥터 모듈 로드 실패. 기본 CSV 로더로 폴백.")
            return self._load_csv_fallback()
    
    @staticmethod
    def _detect_source_type(path: str) -> str:
        """경로/URI로부터 데이터 소스 타입을 자동 감지합니다."""
        path_lower = path.lower()
        if path_lower.startswith(("postgresql://", "postgres://")):
            return "postgresql"
        elif path_lower.startswith("mysql"):
            return "mysql"
        elif path_lower.startswith("sqlite"):
            return "sqlite"
        elif path_lower.endswith(".parquet"):
            return "parquet"
        elif path_lower.endswith((".xlsx", ".xls")):
            return "excel"
        elif path_lower.endswith(".tsv"):
            return "tsv"
        elif "bigquery" in path_lower or path_lower.startswith("bq://"):
            return "bigquery"
        return "csv"
    
    def _load_csv_fallback(self) -> Tuple[pd.DataFrame, Dict]:
        """커넥터 없이 기존 CSV 로드 로직 (폴백)."""
        cfg = self.config.data
        try:
            df = pd.read_csv(cfg.input_path)
        except Exception as e:
            raise RuntimeError(f"데이터 로드 실패: {cfg.input_path}") from e
        
        self.logger.info("CSV 로드 성공: %d rows", len(df))
        
        if cfg.treatment_col not in df.columns:
            raise ValueError(f"처치 변수 '{cfg.treatment_col}'이(가) 데이터에 없습니다.")
        if cfg.outcome_col not in df.columns:
            raise ValueError(f"결과 변수 '{cfg.outcome_col}'이(가) 데이터에 없습니다.")
        
        features = cfg.feature_cols
        if not features:
            exclude = {cfg.treatment_col, cfg.outcome_col, "user_id", "id", "index", "Unnamed: 0"}
            features = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
            self.logger.info("피처 자동 추론: %s", features)
        
        return df, {
            "treatment_col": cfg.treatment_col,
            "outcome_col": cfg.outcome_col,
            "feature_names": features,
            "dag_edges": [],
            "true_cate_col": None,
        }
    def _generate_base_features(self) -> pd.DataFrame:
        cfg = self.config.data
        rng = np.random.default_rng(cfg.random_seed)
        n = cfg.n_samples

        income = rng.lognormal(
            mean=cfg.income_log_mean, sigma=cfg.income_log_sigma, size=n
        )
        age = rng.normal(loc=cfg.age_mean, scale=cfg.age_std, size=n).clip(
            cfg.age_min, cfg.age_max
        )
        credit_score = rng.normal(
            loc=cfg.credit_score_mean, scale=cfg.credit_score_std, size=n
        ).clip(cfg.credit_score_min, cfg.credit_score_max)
        app_usage = rng.exponential(scale=cfg.app_usage_scale, size=n).clip(
            cfg.app_usage_min, cfg.app_usage_max
        )
        # 소비는 소득에 비례 + 노이즈
        consumption = (
            cfg.consumption_income_coef * income + rng.normal(0, 500, n)
        ).clip(0, None)

        # 가상의 시계열 생성을 위한 'month' 컬럼 (DuckDB 실습용)
        # 각 유저별로 단일 레코드지만, 과거 이력을 가정하여 생성
        # 실제로는 유저별 다건이 필요하지만 여기서는 간소화
        months = np.random.randint(1, 13, size=n)

        return pd.DataFrame({
            "user_id": np.arange(n),
            "month": months,
            "income": income,
            "age": age,
            "credit_score": credit_score,
            "app_usage_time": app_usage,
            "consumption": consumption,
        })

    def _detect_local_csv(self) -> str | None:
        """engine/data 디렉토리에서 가장 최근 수정된 CSV 파일을 찾습니다."""
        try:
            # 현재 파일 기준 engine/data 경로 추정
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = os.path.join(base_dir, "engine", "data")
            
            if not os.path.exists(data_dir):
                return None
                
            csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
            if not csv_files:
                return None
                
            # 가장 최근 수정된 파일 반환
            latest_file = max(csv_files, key=os.path.getmtime)
            return latest_file
        except Exception as e:
            self.logger.warning("로컬 CSV 검색 중 오류 발생: %s", e)
            return None

    def _generate_scenario_A(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """시나리오 A: 신용 한도(Continuous) → 연체 여부(Binary)."""
        cfg = self.config.data
        rng = np.random.default_rng(cfg.random_seed)
        n = len(df)

        income_z = _normalize(df["income"].values)
        age_z = _normalize(df["age"].values)
        credit_z = _normalize(df["credit_score"].values)
        consumption_z = _normalize(df["consumption"].values)

        # Treatment: 신용 한도 (교란: 소득/신용 높으면 한도 높음)
        treat_logit = (
            cfg.treat_income_coef * income_z
            + cfg.treat_age_coef * age_z
            + cfg.treat_credit_coef * credit_z
            + rng.normal(0, cfg.treat_noise_std, n)
        )
        credit_limit = (
            cfg.treat_min
            + (cfg.treat_max - cfg.treat_min) * _sigmoid(treat_logit)
        )
        credit_limit_z = _normalize(credit_limit)

        # CATE (Ground Truth)
        true_cate = (
            cfg.cate_income_coef * income_z
            + cfg.cate_age_coef * age_z
            + cfg.cate_credit_coef * credit_z
        )

        # Outcome: 연체 여부
        # Y = T * CATE + g(X) + noise
        outcome_logit = (
            true_cate * credit_limit_z
            + cfg.outcome_income_coef * income_z
            + cfg.outcome_consumption_coef * consumption_z
            + cfg.outcome_credit_coef * credit_z
            + rng.normal(0, cfg.outcome_noise_std, n)
        )
        default_prob = _sigmoid(outcome_logit)
        is_default = default_prob  # DML은 연속형 outcome에서 최적 동작 → 확률 유지

        df["credit_limit"] = credit_limit
        df["is_default"] = is_default            # 연체 확률 (0~1 연속)
        df["is_default_binary"] = (default_prob > 0.5).astype(int)  # 이진 참조용
        df["true_cate"] = true_cate
        df["default_prob"] = default_prob

        feature_names = ["income", "age", "credit_score", "app_usage_time", "consumption"]
        dag_edges = [
            ("income", "credit_limit"), ("income", "is_default"),
            ("age", "credit_limit"), ("age", "is_default"),
            ("credit_score", "credit_limit"), ("credit_score", "is_default"),
            ("credit_limit", "is_default"),
        ]

        return df, {
            "treatment_col": "credit_limit",
            "outcome_col": "is_default",
            "true_cate_col": "true_cate",
            "feature_names": feature_names,
            "dag_edges": dag_edges,
        }

    def _generate_scenario_B(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """시나리오 B: 투자 쿠폰(Binary) → 가입 여부(Binary)."""
        cfg = self.config.data
        rng = np.random.default_rng(cfg.random_seed)
        n = len(df)

        income_z = _normalize(df["income"].values)
        age_z = _normalize(df["age"].values)
        app_z = _normalize(df["app_usage_time"].values)

        # Treatment: 쿠폰 지급 여부 (무작위가 아니라 타겟팅 가정)
        # 활동성(app_usage) 높은 유저에게 더 많이 지급
        treat_prob = _sigmoid(0.5 * app_z - 0.2 * age_z)
        coupon_sent = rng.binomial(1, treat_prob)

        # CATE (Ground Truth) - HTE
        # 20대(age ↓), 저소득(income ↓)일수록 효과 큼
        true_cate_logit = (
            cfg.cate_b_base
            + cfg.cate_b_income_coef * income_z
            + cfg.cate_b_age_coef * age_z
        )
        true_cate_prob = _sigmoid(true_cate_logit) - 0.5 # 확률 증가분 (약식)
        # 실제 효과는 확률 스케일에서 작동
        
        # Outcome: 가입 여부
        # Base Prob + T * CATE + Noise
        base_logit = -2.0 + 0.5 * income_z + 0.3 * app_z  # 기본적으로 돈 많고 활동성 높으면 가입
        outcome_prob = _sigmoid(base_logit + coupon_sent * true_cate_logit)
        is_joined = rng.binomial(1, outcome_prob)

        df["coupon_sent"] = coupon_sent
        df["is_joined"] = is_joined
        df["true_cate"] = true_cate_prob # 근사치

        feature_names = ["income", "age", "credit_score", "app_usage_time"]
        dag_edges = [
            ("app_usage_time", "coupon_sent"),
            ("age", "coupon_sent"),
            ("coupon_sent", "is_joined"),
            ("income", "is_joined"),
            ("app_usage_time", "is_joined"),
        ]

        return df, {
            "treatment_col": "coupon_sent",
            "outcome_col": "is_joined",
            "true_cate_col": "true_cate",
            "feature_names": feature_names,
            "dag_edges": dag_edges,
        }

    def _apply_duckdb_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """DuckDB Window Function을 활용한 전처리.

        보고서 가이드에 따라 SQL을 사용하여 파생 변수를 생성합니다.
        (실제로는 1인당 1행이지만, 윈도우 함수 문법 데모를 위해 가상 적용)
        """
        self.logger.info("DuckDB 전처리 시작")
        
        con = duckdb.connect()
        con.register("raw_data", df)

        # 윈도우 함수 데모 (여기서는 id순 정렬을 시계열로 가정)
        query = """
        SELECT 
            *,
            -- 3개월 평균 소비 (가상: 현재 행 포함 전후 맥락)
            AVG(consumption) OVER (
                ORDER BY income -- 임의 정렬
                ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
            ) AS avg_spend_3m,
            
            -- 최근 신용점수 최대값
            MAX(credit_score) OVER (
                PARTITION BY age_group_10
            ) AS max_credit_score_6m
        FROM (
             SELECT *, FLOOR(age / 10) * 10 AS age_group_10 FROM raw_data
        )
        """
        
        df_processed = con.execute(query).df()
        con.close()
        
        # 임시 컬럼 제거
        if "age_group_10" in df_processed.columns:
            df_processed = df_processed.drop(columns=["age_group_10"])

        self.logger.info("DuckDB 전처리 완료")
        return df_processed
