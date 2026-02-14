# -*- coding: utf-8 -*-
"""Google BigQuery 커넥터."""

import pandas as pd

from engine.connectors.base import BaseConnector, ConnectorConfig


class BigQueryConnector(BaseConnector):
    """Google BigQuery에서 데이터를 로드합니다.

    config.uri: GCP 프로젝트 ID.
    config.query: BigQuery SQL 쿼리.
    config.table: 완전한 테이블 참조 (project.dataset.table).
    config.options: {"credentials_path": "/path/to/key.json"} (선택).

    사용법:
        config = ConnectorConfig(
            source_type="bigquery",
            uri="my-gcp-project",
            query="SELECT * FROM `project.dataset.table` LIMIT 10000",
            treatment_col="coupon_sent",
            outcome_col="converted",
        )
        with BigQueryConnector(config) as conn:
            df = conn.fetch()
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self._client = None

    def connect(self) -> None:
        """BigQuery 클라이언트를 초기화합니다."""
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "BigQueryConnector를 사용하려면 google-cloud-bigquery를 설치하세요: "
                "pip install google-cloud-bigquery pandas-gbq"
            )

        credentials_path = self.config.options.get("credentials_path")
        if credentials_path:
            import os
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        self._client = bigquery.Client(project=self.config.uri)
        self._connected = True
        self.logger.info("☁️ BigQuery 연결: 프로젝트=%s", self.config.uri)

    def fetch(self) -> pd.DataFrame:
        """BigQuery에서 DataFrame을 로드합니다."""
        if not self._connected or self._client is None:
            self.connect()

        if self.config.query:
            self.logger.info("BigQuery 쿼리 실행...")
            df = self._client.query(self.config.query).to_dataframe()
        elif self.config.table:
            self.logger.info("BigQuery 테이블 로드: %s", self.config.table)
            df = self._client.list_rows(self.config.table).to_dataframe()
        else:
            raise ValueError("query 또는 table 중 하나를 지정해야 합니다.")

        self.logger.info("BigQuery 로드 완료: %d행 × %d열", len(df), len(df.columns))
        self.validate(df)
        return df

    def close(self) -> None:
        """BigQuery 클라이언트를 정리합니다."""
        if self._client:
            self._client.close()
            self.logger.info("BigQuery 연결 해제")
        self._connected = False
