# -*- coding: utf-8 -*-
"""WhyLab 데이터 커넥터 패키지.

다양한 데이터 소스(CSV, SQL, BigQuery 등)에서 데이터를 로드하여
WhyLab 파이프라인에 공급하는 통합 인터페이스를 제공합니다.
"""

from engine.connectors.base import BaseConnector, ConnectorConfig
from engine.connectors.csv_connector import CSVConnector
from engine.connectors.sql_connector import SQLConnector
from engine.connectors.bigquery_connector import BigQueryConnector
from engine.connectors.factory import create_connector

__all__ = [
    "BaseConnector",
    "ConnectorConfig",
    "CSVConnector",
    "SQLConnector",
    "BigQueryConnector",
    "create_connector",
]
