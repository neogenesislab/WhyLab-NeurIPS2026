# -*- coding: utf-8 -*-
"""CSV / Parquet 파일 커넥터."""

from pathlib import Path

import pandas as pd

from engine.connectors.base import BaseConnector, ConnectorConfig


class CSVConnector(BaseConnector):
    """CSV 및 Parquet 파일에서 데이터를 로드합니다.

    config.uri에 파일 경로를 지정합니다.
    config.options으로 추가 pandas 옵션을 전달할 수 있습니다.

    사용법:
        config = ConnectorConfig(source_type="csv", uri="data.csv", ...)
        with CSVConnector(config) as conn:
            df = conn.fetch()
    """

    def __init__(self, config: ConnectorConfig):
        super().__init__(config)
        self._path: Path | None = None

    def connect(self) -> None:
        """파일 존재 여부를 확인합니다."""
        path = Path(self.config.uri)
        if not path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없음: {path}")
        self._path = path
        self._connected = True
        self.logger.info("📂 파일 연결: %s", path)

    def fetch(self) -> pd.DataFrame:
        """파일을 DataFrame으로 로드합니다."""
        if not self._connected or self._path is None:
            self.connect()

        suffix = self._path.suffix.lower()
        options = self.config.options

        if suffix == ".parquet":
            df = pd.read_parquet(self._path, **options)
        elif suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(self._path, sep=sep, **options)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(self._path, **options)
        else:
            # 기본: CSV로 시도
            df = pd.read_csv(self._path, **options)

        self.logger.info("로드 완료: %d행 × %d열", len(df), len(df.columns))
        self.validate(df)
        return df

    def close(self) -> None:
        """파일 커넥터는 정리할 리소스가 없습니다."""
        self._connected = False
