# -*- coding: utf-8 -*-
"""데이터 커넥터 추상 기반 클래스.

모든 커넥터는 이 클래스를 상속하여 `connect()`, `fetch()`, `close()`를 구현합니다.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ConnectorConfig:
    """데이터 커넥터 공통 설정.

    Attributes:
        source_type: 데이터 소스 타입 ("csv", "parquet", "sql", "bigquery").
        uri: 연결 문자열 또는 파일 경로.
        query: SQL 쿼리 (SQL/BigQuery 전용).
        table: 테이블 이름 (SQL/BigQuery 전용).
        treatment_col: 처치 변수 컬럼명.
        outcome_col: 결과 변수 컬럼명.
        feature_cols: 공변량 컬럼 리스트 (빈 리스트면 자동 추론).
        options: 추가 옵션 딕셔너리.
    """

    source_type: str = "csv"
    uri: str = ""
    query: Optional[str] = None
    table: Optional[str] = None
    treatment_col: str = "treatment"
    outcome_col: str = "outcome"
    feature_cols: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    """데이터 커넥터 추상 기반 클래스.

    사용법:
        connector = CSVConnector(config)
        connector.connect()
        df = connector.fetch()
        connector.close()

    또는 Context Manager:
        with CSVConnector(config) as conn:
            df = conn.fetch()
    """

    def __init__(self, config: ConnectorConfig):
        self.config = config
        self.logger = logging.getLogger(f"whylab.connector.{config.source_type}")
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @abstractmethod
    def connect(self) -> None:
        """데이터 소스에 연결합니다."""
        ...

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """데이터를 DataFrame으로 가져옵니다."""
        ...

    @abstractmethod
    def close(self) -> None:
        """연결을 정리합니다."""
        ...

    def fetch_with_meta(self) -> Dict[str, Any]:
        """데이터와 메타데이터를 함께 반환합니다.

        Returns:
            {"dataframe": pd.DataFrame, "treatment_col": str, ...}
        """
        df = self.fetch()
        cfg = self.config

        # 피처 자동 추론
        features = cfg.feature_cols
        if not features:
            exclude = {
                cfg.treatment_col, cfg.outcome_col,
                "user_id", "id", "index", "Unnamed: 0",
            }
            features = [
                c for c in df.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
            ]
            self.logger.info("피처 자동 추론: %d개 — %s", len(features), features[:5])

        return {
            "dataframe": df,
            "treatment_col": cfg.treatment_col,
            "outcome_col": cfg.outcome_col,
            "feature_names": features,
            "dag_edges": [],
            "true_cate_col": None,
            "source_type": cfg.source_type,
        }

    def validate(self, df: pd.DataFrame) -> None:
        """필수 컬럼 존재 여부를 검증합니다."""
        cfg = self.config
        if cfg.treatment_col not in df.columns:
            raise ValueError(
                f"처치 변수 '{cfg.treatment_col}'이(가) 데이터에 없습니다. "
                f"사용 가능한 컬럼: {list(df.columns)}"
            )
        if cfg.outcome_col not in df.columns:
            raise ValueError(
                f"결과 변수 '{cfg.outcome_col}'이(가) 데이터에 없습니다. "
                f"사용 가능한 컬럼: {list(df.columns)}"
            )

    def schema(self) -> Dict[str, str]:
        """데이터 스키마(컬럼명→타입)를 반환합니다. 서브클래스에서 오버라이드 가능."""
        df = self.fetch()
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
