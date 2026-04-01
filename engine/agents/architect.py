"""
Architect Agent — 자기 디버깅/최적화 메타 에이전트 (Sprint 33)
===============================================================
WhyLab 시스템 자체를 모니터링하고, 성능 병목을 식별하며,
코드 레벨 최적화를 제안/적용하는 메타 에이전트입니다.

[역할]
- 시스템 건전성 모니터링 (DB 크기, 연속 실패율, 메서드 편향)
- 성능 병목 자동 식별
- Hot-Swapping을 통한 코드 교체 (옵션)
- 자동 테스트 + 실패 시 즉시 롤백

[위치: engine/agents/ — Execution Plane]
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("whylab.architect")

# 프로젝트 루트
ROOT = Path(__file__).resolve().parent.parent.parent


class DiagnosticResult:
    """시스템 진단 결과."""
    def __init__(self):
        self.timestamp = datetime.utcnow().isoformat()
        self.checks: list[dict] = []
        self.warnings: list[str] = []
        self.recommendations: list[str] = []
        self.health_score: float = 100.0

    def add_check(self, name: str, status: str, detail: str, impact: float = 0):
        """진단 항목 추가. impact: 건강 점수 감소량."""
        self.checks.append({
            "name": name,
            "status": status,  # "OK" | "WARNING" | "CRITICAL"
            "detail": detail,
        })
        if status == "WARNING":
            self.warnings.append(f"{name}: {detail}")
            self.health_score -= impact
        elif status == "CRITICAL":
            self.warnings.append(f"🚨 {name}: {detail}")
            self.health_score -= impact * 2
        self.health_score = max(0, self.health_score)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "health_score": round(self.health_score, 1),
            "checks": self.checks,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "total_checks": len(self.checks),
            "ok_count": sum(1 for c in self.checks if c["status"] == "OK"),
            "warning_count": sum(1 for c in self.checks if c["status"] == "WARNING"),
            "critical_count": sum(1 for c in self.checks if c["status"] == "CRITICAL"),
        }


class ArchitectAgent:
    """
    시스템 자기 진단 및 최적화 에이전트.
    
    6가지 진단 영역:
    1. DB 건전성 (크기, 레코드 수, WAL 모드)
    2. 샌드박스 실행기 상태 (성공률, 회로 차단기)
    3. 메서드 레지스트리 편향 (제12조 이슈)
    4. Knowledge Graph 밀도
    5. 파일 시스템 (업로드, 백업 크기)
    6. 에이전트 상태 (활성/비활성)
    """

    def diagnose(self) -> DiagnosticResult:
        """전체 시스템 진단을 실행합니다."""
        result = DiagnosticResult()
        
        self._check_db_health(result)
        self._check_sandbox(result)
        self._check_method_registry(result)
        self._check_knowledge_graph(result)
        self._check_filesystem(result)
        self._generate_recommendations(result)
        
        logger.info(
            "시스템 진단 완료 | 건강 점수: %.1f/100 | 경고: %d건",
            result.health_score, len(result.warnings)
        )
        
        return result

    def _check_db_health(self, result: DiagnosticResult):
        """DB 건전성 진단."""
        try:
            db_path = ROOT / "whylab.db"
            if db_path.exists():
                size_mb = db_path.stat().st_size / (1024 * 1024)
                if size_mb > 100:
                    result.add_check("DB 크기", "CRITICAL",
                        f"whylab.db가 {size_mb:.1f}MB입니다. 즉시 로테이션 필요.", 20)
                elif size_mb > 50:
                    result.add_check("DB 크기", "WARNING",
                        f"whylab.db가 {size_mb:.1f}MB입니다. 로테이션 권장.", 10)
                else:
                    result.add_check("DB 크기", "OK",
                        f"whylab.db {size_mb:.1f}MB — 정상 범위.")
            else:
                result.add_check("DB 크기", "OK", "DB 파일 미존재 (초기 상태).")

            # WAL 파일 확인
            wal_path = ROOT / "whylab.db-wal"
            if wal_path.exists():
                wal_size_mb = wal_path.stat().st_size / (1024 * 1024)
                if wal_size_mb > 10:
                    result.add_check("WAL 크기", "WARNING",
                        f"WAL 파일 {wal_size_mb:.1f}MB — checkpoint 필요.", 5)
                else:
                    result.add_check("WAL 크기", "OK",
                        f"WAL 파일 {wal_size_mb:.1f}MB — 정상.")
        except Exception as e:
            result.add_check("DB 진단", "WARNING", f"진단 실패: {str(e)}", 5)

    def _check_sandbox(self, result: DiagnosticResult):
        """SandboxExecutor 상태 진단."""
        try:
            from engine.sandbox.executor import sandbox
            stats = sandbox.get_stats()
            
            # 회로 차단기 확인
            if stats.get("circuit_breaker_active"):
                result.add_check("회로 차단기", "CRITICAL",
                    "회로 차단기 활성화 상태. 수동 리셋 필요.", 25)
            else:
                result.add_check("회로 차단기", "OK", "비활성 — 정상.")
            
            # 성공률
            success_rate = stats.get("success_rate", 1.0)
            if success_rate < 0.5:
                result.add_check("샌드박스 성공률", "CRITICAL",
                    f"성공률 {success_rate:.0%} — 데이터 파이프라인 검토 필요.", 20)
            elif success_rate < 0.8:
                result.add_check("샌드박스 성공률", "WARNING",
                    f"성공률 {success_rate:.0%} — 개선 여지 있음.", 10)
            else:
                result.add_check("샌드박스 성공률", "OK",
                    f"성공률 {success_rate:.0%} — 양호.")
            
            # 연속 실패
            consec = stats.get("consecutive_failures", 0)
            if consec >= 2:
                result.add_check("연속 실패", "WARNING",
                    f"{consec}회 연속 실패 중.", 10)
        except ImportError:
            result.add_check("샌드박스", "WARNING", "SandboxExecutor 모듈 로드 실패.", 5)

    def _check_method_registry(self, result: DiagnosticResult):
        """메서드 레지스트리 편향 진단 (제12조)."""
        try:
            from api.agents.method_registry import method_registry
            stats = method_registry.get_stats()
            
            for category, methods in stats.items():
                if isinstance(methods, list) and len(methods) > 0:
                    total_pulls = sum(m.get("total_pulls", 0) for m in methods)
                    if total_pulls > 10:
                        for m in methods:
                            pulls = m.get("total_pulls", 0)
                            if pulls / total_pulls > 0.7:
                                result.add_check("메서드 편향", "WARNING",
                                    f"'{m.get('name', '?')}'이 {pulls/total_pulls:.0%} 선택됨 (제12조 위반 위험).", 5)
                                break
                        else:
                            result.add_check("메서드 다양성", "OK",
                                f"{category}: {len(methods)}개 메서드 균형 유지.")
        except ImportError:
            result.add_check("메서드 레지스트리", "WARNING", "MethodRegistry 로드 실패.", 5)

    def _check_knowledge_graph(self, result: DiagnosticResult):
        """Knowledge Graph 밀도 진단."""
        try:
            from api.graph import kg
            stats = kg.get_stats()
            nodes = stats.get("nodes", 0)
            edges = stats.get("edges", 0)
            
            if nodes == 0:
                result.add_check("Knowledge Graph", "WARNING",
                    "KG가 비어 있습니다. Theorist 활성화 필요.", 5)
            else:
                density = edges / max(nodes * (nodes - 1), 1)
                result.add_check("Knowledge Graph", "OK",
                    f"{nodes}노드, {edges}엣지, 밀도={density:.3f}")
        except ImportError:
            result.add_check("Knowledge Graph", "WARNING", "KG 모듈 로드 실패.", 5)

    def _check_filesystem(self, result: DiagnosticResult):
        """파일 시스템 진단."""
        upload_dir = ROOT / "data" / "uploads"
        if upload_dir.exists():
            total_size = sum(f.stat().st_size for f in upload_dir.rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)
            if size_mb > 500:
                result.add_check("업로드 폴더", "WARNING",
                    f"uploads 폴더 {size_mb:.0f}MB — 정리 권장.", 5)
            else:
                result.add_check("업로드 폴더", "OK",
                    f"uploads 폴더 {size_mb:.1f}MB — 정상.")

        backup_dir = ROOT / "data" / "backups"
        if backup_dir.exists():
            backup_count = sum(1 for _ in backup_dir.rglob("*.py"))
            result.add_check("백업 파일", "OK", f"{backup_count}개 백업 파일 보존 중.")

    def _generate_recommendations(self, result: DiagnosticResult):
        """진단 결과 기반 권장 사항 생성."""
        critical_checks = [c for c in result.checks if c["status"] == "CRITICAL"]
        warning_checks = [c for c in result.checks if c["status"] == "WARNING"]
        
        if critical_checks:
            result.recommendations.append("🚨 즉시 조치 필요:")
            for c in critical_checks:
                result.recommendations.append(f"  → {c['name']}: {c['detail']}")
        
        if warning_checks:
            result.recommendations.append("⚠️ 개선 권장:")
            for c in warning_checks:
                result.recommendations.append(f"  → {c['name']}: {c['detail']}")
        
        if result.health_score >= 90:
            result.recommendations.append("✅ 시스템 전반적으로 양호합니다.")
        elif result.health_score >= 70:
            result.recommendations.append("📋 일부 개선이 필요하지만 운영에 지장은 없습니다.")
        else:
            result.recommendations.append("🔧 시스템 건전성 점검이 시급합니다. 운영 안정성에 영향을 줄 수 있습니다.")


# 모듈 레벨 싱글턴
architect = ArchitectAgent()
