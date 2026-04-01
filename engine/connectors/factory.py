# -*- coding: utf-8 -*-
"""커넥터 팩토리 — source_type에 따라 적절한 커넥터를 생성합니다."""

from engine.connectors.base import BaseConnector, ConnectorConfig


# 지원하는 커넥터 레지스트리
_CONNECTOR_REGISTRY = {
    "csv": "engine.connectors.csv_connector.CSVConnector",
    "parquet": "engine.connectors.csv_connector.CSVConnector",
    "tsv": "engine.connectors.csv_connector.CSVConnector",
    "excel": "engine.connectors.csv_connector.CSVConnector",
    "sql": "engine.connectors.sql_connector.SQLConnector",
    "postgresql": "engine.connectors.sql_connector.SQLConnector",
    "mysql": "engine.connectors.sql_connector.SQLConnector",
    "sqlite": "engine.connectors.sql_connector.SQLConnector",
    "bigquery": "engine.connectors.bigquery_connector.BigQueryConnector",
    "bq": "engine.connectors.bigquery_connector.BigQueryConnector",
}


def create_connector(config: ConnectorConfig) -> BaseConnector:
    """ConnectorConfig에 따라 적절한 커넥터 인스턴스를 생성합니다.

    Args:
        config: 커넥터 설정.

    Returns:
        BaseConnector 인스턴스.

    Raises:
        ValueError: 지원하지 않는 source_type.

    사용법:
        from engine.connectors import create_connector, ConnectorConfig

        config = ConnectorConfig(
            source_type="sql",
            uri="postgresql://user:pass@host/db",
            query="SELECT * FROM users",
            treatment_col="coupon",
            outcome_col="purchase",
        )
        connector = create_connector(config)
        with connector:
            result = connector.fetch_with_meta()
    """
    source_type = config.source_type.lower()

    if source_type not in _CONNECTOR_REGISTRY:
        available = ", ".join(sorted(_CONNECTOR_REGISTRY.keys()))
        raise ValueError(
            f"지원하지 않는 데이터 소스 타입: '{source_type}'. "
            f"사용 가능한 타입: {available}"
        )

    # Lazy import + instantiation
    class_path = _CONNECTOR_REGISTRY[source_type]
    module_path, class_name = class_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    connector_class = getattr(module, class_name)

    return connector_class(config)
