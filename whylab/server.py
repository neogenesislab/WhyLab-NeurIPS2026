# -*- coding: utf-8 -*-
"""WhyLab SDK REST API 서버 — 외부 사용자용 인과추론 서비스.

역할: SDK 사용자 및 외부 시스템이 인과추론 파이프라인을 호출하는 공개 API.
      경량 엔드포인트 (analyze, jobs, methods, health) 제공.

참고: Dashboard 백엔드(에이전트 관리, 진화, 자율 연구 등)는 별도로
      api/main.py (포트 4001)에서 담당합니다.

사용법:
    # 개발 서버 실행
    uvicorn whylab.server:app --reload --port 8000

    # 분석 요청
    curl -X POST http://localhost:8000/api/v1/analyze \
      -H "Content-Type: application/json" \
      -d '{"treatment": "T", "outcome": "Y", "data_path": "data.csv"}'
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("whylab.server")

# ──────────────────────────────────────────────
# FastAPI 앱
# ──────────────────────────────────────────────

app = FastAPI(
    title="WhyLab API",
    description="인과추론 Decision Intelligence 엔진",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# 요청/응답 스키마
# ──────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """인과 분석 요청."""
    data_path: Optional[str] = Field(None, description="CSV 파일 경로")
    data_inline: Optional[List[Dict[str, Any]]] = Field(None, description="인라인 데이터 (JSON 레코드)")
    treatment: str = Field("treatment", description="처치 변수 컬럼명")
    outcome: str = Field("outcome", description="결과 변수 컬럼명")
    features: Optional[List[str]] = Field(None, description="공변량 리스트")
    scenario: Optional[str] = Field(None, description="합성 데이터 시나리오")
    enable_debate: bool = Field(True, description="AI Debate 활성화")
    enable_discovery: bool = Field(True, description="인과 구조 자동 발견 활성화")


class AnalyzeResponse(BaseModel):
    """인과 분석 응답."""
    job_id: str
    status: str  # "completed" | "failed" | "running"
    ate: float = 0.0
    ate_ci: List[float] = [0.0, 0.0]
    verdict: str = "UNCERTAIN"
    confidence: float = 0.0
    meta_learners: Dict[str, Any] = {}
    sensitivity: Dict[str, Any] = {}
    quasi_experimental: Dict[str, Any] = {}
    temporal_causal: Dict[str, Any] = {}
    counterfactual: Dict[str, Any] = {}
    data_profile: Dict[str, Any] = {}
    execution_time_ms: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """헬스체크 응답."""
    status: str = "healthy"
    version: str = "1.0.0"
    cells: int = 22
    uptime_seconds: float = 0.0


# ──────────────────────────────────────────────
# 글로벌 상태
# ──────────────────────────────────────────────

_start_time = time.time()
_jobs: Dict[str, AnalyzeResponse] = {}


# ──────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["시스템"])
def health_check():
    """서버 헬스체크."""
    return HealthResponse(
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["분석"])
def analyze(req: AnalyzeRequest):
    """동기 인과 분석 실행.

    데이터를 받아 16셀 파이프라인을 실행하고 결과를 반환합니다.
    """
    import pandas as pd
    from whylab.api import analyze as whylab_analyze

    job_id = str(uuid.uuid4())[:8]
    start = time.time()

    try:
        # 데이터 로드
        if req.data_inline:
            df = pd.DataFrame(req.data_inline)
        elif req.data_path:
            df = pd.read_csv(req.data_path)
        else:
            df = None  # 합성 데이터 사용

        # 분석 실행
        data_arg = df if df is not None else req.data_path
        result = whylab_analyze(
            data=data_arg,
            treatment=req.treatment,
            outcome=req.outcome,
            features=req.features,
            scenario=req.scenario,
            debate=req.enable_debate,
        )

        elapsed = int((time.time() - start) * 1000)

        response = AnalyzeResponse(
            job_id=job_id,
            status="completed",
            ate=result.ate,
            ate_ci=list(result.ate_ci),
            verdict=result.verdict,
            confidence=result.confidence,
            meta_learners=result.meta_learners,
            sensitivity=result.sensitivity,
            quasi_experimental=result.raw.get("quasi_experimental", {}),
            temporal_causal=result.raw.get("temporal_causal", {}),
            counterfactual=result.raw.get("counterfactual", {}),
            data_profile=result.raw.get("data_profile", {}),
            execution_time_ms=elapsed,
        )

        _jobs[job_id] = response
        logger.info("✅ 분석 완료 [%s] %dms — %s", job_id, elapsed, result.verdict)
        return response

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        logger.error("❌ 분석 실패 [%s]: %s", job_id, e)
        response = AnalyzeResponse(
            job_id=job_id,
            status="failed",
            error=str(e),
            execution_time_ms=elapsed,
        )
        _jobs[job_id] = response
        return response


@app.get("/api/v1/jobs/{job_id}", response_model=AnalyzeResponse, tags=["분석"])
def get_job(job_id: str):
    """이전 분석 결과를 조회합니다."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"작업 {job_id}을(를) 찾을 수 없습니다")
    return _jobs[job_id]


@app.get("/api/v1/methods", tags=["정보"])
def list_methods():
    """지원 방법론 목록."""
    return {
        "estimators": ["LinearDML", "CausalForest", "SparseLinearDML"],
        "meta_learners": ["S-Learner", "T-Learner", "X-Learner", "DR-Learner", "R-Learner"],
        "quasi_experimental": ["IV (2SLS)", "DiD", "Sharp RDD"],
        "temporal": ["Granger Causality", "CausalImpact", "Lag Correlation"],
        "robustness": ["E-value", "Placebo", "Random Cause", "Bootstrap", "GATES"],
        "discovery": ["PC Algorithm", "LLM Hybrid"],
        "debate": ["Advocate/Critic/Judge + LLM"],
        "pipeline_cells": 22,
    }
