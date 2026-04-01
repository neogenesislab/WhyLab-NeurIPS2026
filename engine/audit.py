# -*- coding: utf-8 -*-
"""AuditLogger — 인과 분석 감사 추적 시스템.

모든 인과 분석 실행과 판결을 추적 가능한 감사 로그로 기록합니다.
"왜 이 정책을 승인했는가?"에 대한 증거 체인을 제공합니다.

Phase 11-4: 거버넌스.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 감사 로그 저장 경로 (환경 변수로 오버라이드 가능)
AUDIT_LOG_DIR = os.environ.get("WHYLAB_AUDIT_DIR", "audit_logs")


@dataclass
class AuditEntry:
    """단일 감사 로그 항목."""
    audit_id: str = ""
    timestamp: str = ""
    action: str = ""           # "analyze" | "discover" | "debate" | "export"
    user: str = "anonymous"
    treatment: str = ""
    outcome: str = ""
    dataset_hash: str = ""
    n_samples: int = 0
    result_summary: Dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    confidence: float = 0.0
    methods_used: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """감사 로그 관리자.

    모든 인과 분석의 입력, 방법론, 결과, 판결을
    JSON Lines 형식으로 기록합니다.
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self.log_dir = Path(log_dir or AUDIT_LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_session = str(uuid.uuid4())[:8]
        self._entries: List[AuditEntry] = []
        logger.info("📋 감사 로그 초기화 (디렉토리: %s)", self.log_dir)

    def log_analysis(
        self,
        context: Dict[str, Any],
        execution_time_ms: int = 0,
        user: str = "anonymous",
    ) -> str:
        """파이프라인 실행 결과를 감사 로그에 기록합니다.

        Args:
            context: 파이프라인 최종 컨텍스트.
            execution_time_ms: 실행 시간 (밀리초).
            user: 실행 사용자.

        Returns:
            감사 ID.
        """
        import hashlib

        audit_id = f"AUD-{self._current_session}-{len(self._entries):04d}"

        # 데이터셋 해시 (재현성)
        df = context.get("dataframe")
        if df is not None:
            try:
                dataset_hash = hashlib.sha256(
                    str(df.shape).encode() + str(df.columns.tolist()).encode()
                ).hexdigest()[:12]
            except Exception:
                dataset_hash = "unknown"
        else:
            dataset_hash = "synthetic"

        # 사용된 방법론 수집
        methods = []
        if context.get("ate"):
            methods.append("DML")
        if context.get("meta_learners"):
            methods.extend(list(context["meta_learners"].keys()))
        if context.get("quasi_experimental"):
            methods.extend([f"QE:{k}" for k in context["quasi_experimental"].keys()])
        if context.get("temporal_causal"):
            methods.extend([f"TC:{k}" for k in context["temporal_causal"].keys()])
        if context.get("counterfactual"):
            methods.append("Counterfactual")
        if context.get("dag_edges"):
            methods.append("Discovery")

        # Debate 결과
        debate = context.get("debate", {})
        verdict = debate.get("verdict", "N/A") if isinstance(debate, dict) else "N/A"
        confidence = debate.get("confidence", 0.0) if isinstance(debate, dict) else 0.0

        # 경고 수집
        warnings = []
        profile = context.get("data_profile", {})
        if isinstance(profile, dict):
            warnings.extend(profile.get("warnings", []))
        recommendation = context.get("auto_recommendation", {})
        if isinstance(recommendation, dict):
            warnings.extend(recommendation.get("warnings", []))

        # ATE 요약
        ate_raw = context.get("ate", {})
        if isinstance(ate_raw, dict):
            ate_val = ate_raw.get("point_estimate", ate_raw.get("value", 0))
        elif isinstance(ate_raw, (int, float)):
            ate_val = ate_raw
        else:
            ate_val = 0

        entry = AuditEntry(
            audit_id=audit_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action="analyze",
            user=user,
            treatment=context.get("treatment_col", ""),
            outcome=context.get("outcome_col", ""),
            dataset_hash=dataset_hash,
            n_samples=len(df) if df is not None else 0,
            result_summary={"ate": ate_val, "verdict": verdict, "confidence": confidence},
            verdict=verdict,
            confidence=confidence,
            methods_used=methods,
            warnings=warnings,
            execution_time_ms=execution_time_ms,
            metadata={
                "discovery_mode": context.get("discovery_mode", ""),
                "recommended_method": context.get("auto_recommendation", {}).get(
                    "primary_method", ""
                ) if isinstance(context.get("auto_recommendation"), dict) else "",
                "pipeline_cells": 16,
            },
        )

        self._entries.append(entry)
        self._write_entry(entry)

        logger.info(
            "📋 감사 로그 기록 [%s]: %s → %s (확신도=%.1f%%)",
            audit_id, entry.treatment, entry.verdict, entry.confidence * 100,
        )

        return audit_id

    def get_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """최근 감사 로그 항목을 반환합니다."""
        return [asdict(e) for e in self._entries[-limit:]]

    def search(self, treatment: Optional[str] = None, verdict: Optional[str] = None) -> List[Dict[str, Any]]:
        """감사 로그를 검색합니다."""
        results = []
        for entry in self._entries:
            if treatment and entry.treatment != treatment:
                continue
            if verdict and entry.verdict != verdict:
                continue
            results.append(asdict(entry))
        return results

    def _write_entry(self, entry: AuditEntry) -> None:
        """감사 로그를 JSON Lines 파일에 기록합니다."""
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{today}.jsonl"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("감사 로그 기록 실패: %s", e)
