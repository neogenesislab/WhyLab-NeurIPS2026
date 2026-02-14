# -*- coding: utf-8 -*-
"""SQL 데이터베이스 커넥터 (MySQL / PostgreSQL / SQLite)."""

import pandas as pd

from engine.connectors.base import BaseConnector, ConnectorConfig


class SQLConnector(BaseConnector):
    """SQL 데이터베이스에서 데이터를 로드합니다.

    config.uri: SQLAlchemy 연결 문자열.
      - SQLite:     "sqlite:///data.db"
      - PostgreSQL: "postgresql://user:pass@host:5432/dbname"
      - MySQL:      "mysql+pymysql://user:pass@host:3306/dbname"

    config.query: SELECT 쿼리 문자열 (우선).
    config.table: 테이블 이름 (query가 없을 때 전체 로드).

    사용법:
        config = ConnectorConfig(
            source_type="sql",
            uri="postgresql://...",
            query="SELECT * FROM users WHERE created_at > '2025-01-01'",
            treatment_col="coupon",
            outcome_col="purchase",
        )
        with SQLConnector(config) as conn:
            df = conn.fetch()
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self._engine = None

    def connect(self) -> None:
        """SQLAlchemy 엔진을 생성합니다."""
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError(
                "SQLConnector를 사용하려면 sqlalchemy를 설치하세요: "
                "pip install sqlalchemy pymysql psycopg2-binary"
            )

        self._engine = create_engine(
            self.config.uri,
            pool_pre_ping=True,
            **self.config.options,
        )
        self._connected = True
        self.logger.info("🔗 SQL 연결: %s", self.config.uri.split("@")[-1])

    def fetch(self) -> pd.DataFrame:
        """SQL 쿼리 또는 테이블에서 DataFrame을 로드합니다."""
        if not self._connected or self._engine is None:
            self.connect()

        if self.config.query:
            self.logger.info("SQL 쿼리 실행...")
            df = pd.read_sql(self.config.query, self._engine)
        elif self.config.table:
            self.logger.info("테이블 로드: %s", self.config.table)
            df = pd.read_sql_table(self.config.table, self._engine)
        else:
            raise ValueError("query 또는 table 중 하나를 지정해야 합니다.")

        self.logger.info("SQL 로드 완료: %d행 × %d열", len(df), len(df.columns))
        self.validate(df)
        return df

    def close(self) -> None:
        """엔진 연결을 해제합니다."""
        if self._engine:
            self._engine.dispose()
            self.logger.info("SQL 연결 해제")
        self._connected = False

    def schema(self):
        """테이블 스키마를 반환합니다."""
        if not self._connected:
            self.connect()
        if self.config.table:
            from sqlalchemy import inspect
            inspector = inspect(self._engine)
            columns = inspector.get_columns(self.config.table)
            return {col["name"]: str(col["type"]) for col in columns}
        return super().schema()
