"""
SaaS Blueprint — WhyLab SaaS 전환 설계문서 (Sprint 34)
======================================================
현재 로컬 연구 도구 → 클라우드 SaaS 플랫폼으로의 전환 계획.

[핵심 변환]
로컬 SQLite → PostgreSQL + Redis
로컬 파일 → S3/GCS 오브젝트 스토리지
단일 프로세스 → Celery 워커 풀
단일 사용자 → 멀티 테넌트 (테넌트별 격리)

[사용 예시]
    blueprint = SaaSBlueprint()
    plan = blueprint.get_migration_plan()
    readiness = blueprint.assess_readiness()
"""
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("whylab.saas")


@dataclass
class MigrationItem:
    """마이그레이션 항목."""
    category: str
    current: str
    target: str
    effort: str         # "Low" | "Medium" | "High"
    priority: int       # 1 (최우선) ~ 5
    notes: str = ""


class SaaSBlueprint:
    """
    WhyLab SaaS 전환 청사진.
    
    현재 로컬 아키텍처에서 클라우드 SaaS로의 전환을 위한
    마이그레이션 계획, 준비도 평가, 아키텍처 설계를 제공합니다.
    """

    # ── 마이그레이션 매트릭스 ──
    MIGRATIONS: list[MigrationItem] = [
        MigrationItem(
            category="Database",
            current="SQLite + WAL",
            target="PostgreSQL (RDS/Cloud SQL)",
            effort="Medium",
            priority=1,
            notes="SQLAlchemy 기반이므로 engine 교체만으로 가능. 마이그레이션 스크립트 필요.",
        ),
        MigrationItem(
            category="Cache",
            current="In-memory (dict)",
            target="Redis / Valkey",
            effort="Medium",
            priority=2,
            notes="Method Registry, KG 캐시를 Redis로 전환. TTL 기반 만료 정책.",
        ),
        MigrationItem(
            category="File Storage",
            current="로컬 파일시스템 (data/)",
            target="S3 / GCS Object Storage",
            effort="Medium",
            priority=2,
            notes="업로드/아카이브/백업 경로를 S3 프리셋으로 전환.",
        ),
        MigrationItem(
            category="Task Queue",
            current="동기 실행 (FastAPI)",
            target="Celery + Redis Broker",
            effort="High",
            priority=1,
            notes="Autopilot/실험 실행을 비동기 워커로 전환. SandboxExecutor를 Celery task로 래핑.",
        ),
        MigrationItem(
            category="Authentication",
            current="없음 (로컬)",
            target="OAuth2 + JWT (Auth0/Supabase)",
            effort="High",
            priority=1,
            notes="멀티 테넌트 전제. 테넌트별 데이터 격리 필수.",
        ),
        MigrationItem(
            category="Multi-tenancy",
            current="단일 사용자",
            target="Row-Level Security (RLS)",
            effort="High",
            priority=1,
            notes="PostgreSQL RLS + tenant_id 컬럼 추가. 모든 모델에 tenant_id FK.",
        ),
        MigrationItem(
            category="Deployment",
            current="로컬 uvicorn",
            target="Docker + K8s (EKS/GKE)",
            effort="Medium",
            priority=3,
            notes="Dockerfile + Helm chart. Health check 엔드포인트 /health 이미 존재.",
        ),
        MigrationItem(
            category="Monitoring",
            current="Python logging",
            target="OpenTelemetry + Grafana",
            effort="Low",
            priority=4,
            notes="구조화 로깅 → OTLP 익스포터 추가. Architect 진단 데이터를 Grafana 대시보드로.",
        ),
        MigrationItem(
            category="API Gateway",
            current="직접 노출",
            target="Kong / AWS API Gateway",
            effort="Low",
            priority=4,
            notes="Rate limiting, API key 관리, 요금 계량(metering).",
        ),
        MigrationItem(
            category="Billing",
            current="없음",
            target="Stripe Metered Billing",
            effort="High",
            priority=5,
            notes="실험 실행 횟수 기반 종량제. Grand Challenge별 과금 단위.",
        ),
    ]

    def get_migration_plan(self) -> dict:
        """마이그레이션 계획을 반환합니다."""
        by_priority = {}
        for item in self.MIGRATIONS:
            phase = f"Phase {item.priority}"
            if phase not in by_priority:
                by_priority[phase] = []
            by_priority[phase].append({
                "category": item.category,
                "current": item.current,
                "target": item.target,
                "effort": item.effort,
                "notes": item.notes,
            })

        return {
            "plan_version": "v1.0",
            "total_items": len(self.MIGRATIONS),
            "phases": by_priority,
            "estimated_timeline": {
                "Phase 1": "4주 (DB + Auth + Queue + Multi-tenancy)",
                "Phase 2": "2주 (Cache + Storage)",
                "Phase 3": "2주 (Deployment)",
                "Phase 4": "1주 (Monitoring + Gateway)",
                "Phase 5": "2주 (Billing)",
            },
        }

    def assess_readiness(self) -> dict:
        """SaaS 전환 준비도를 평가합니다."""
        scores = {
            "api_layer": 90,        # FastAPI 기반, REST 완비
            "db_abstraction": 85,   # SQLAlchemy ORM → DB 교체 용이
            "auth": 0,              # 인증 미구현
            "multi_tenancy": 0,     # 테넌트 격리 없음
            "async_execution": 40,  # SandboxExecutor 있으나 동기
            "monitoring": 50,       # 기본 로깅 + Architect 진단
            "deployment": 30,       # 로컬 실행, Docker 미구성
            "billing": 0,           # 과금 없음
        }

        total = sum(scores.values())
        max_total = len(scores) * 100
        readiness_pct = round(total / max_total * 100, 1)

        return {
            "readiness_score": readiness_pct,
            "category_scores": scores,
            "strengths": [
                "FastAPI 기반 API 레이어 완비 (90%)",
                "SQLAlchemy ORM으로 DB 교체 용이 (85%)",
                "Architect 자기 진단 시스템 가동 중",
            ],
            "gaps": [
                "인증/인가 미구현 → OAuth2 + JWT 필수",
                "멀티 테넌트 지원 없음 → RLS 설계 필요",
                "비동기 실행 미완 → Celery 도입 필요",
                "과금 시스템 없음 → Stripe 연동 필요",
            ],
            "recommendation": (
                "Phase 1(DB+Auth+Queue+Multi-tenancy)부터 시작하면 "
                "4주 내 MVP SaaS 배포가 가능합니다."
            ),
        }


# 모듈 레벨 싱글턴
saas_blueprint = SaaSBlueprint()
