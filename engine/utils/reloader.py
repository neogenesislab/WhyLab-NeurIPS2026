"""
Hot-Swapping Reloader (Sprint 33)
==================================
에이전트 코드를 런타임에 안전하게 교체(Hot-Swap)합니다.

[설계 원칙]
- 교체 전 자동 백업 → 실패 시 즉시 롤백
- importlib.reload()로 모듈 갱신
- Architect 에이전트가 코드 개선 후 이 모듈을 통해 적용

사용 예시:
    from engine.utils.reloader import reloader
    result = reloader.hot_swap("api.agents.engineer")
    if not result["success"]:
        # → 자동 롤백 완료
"""
import importlib
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("whylab.reloader")

# 프로젝트 루트
ROOT = Path(__file__).resolve().parent.parent.parent
BACKUP_DIR = ROOT / "data" / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


class HotSwapResult:
    """핫 스왑 결과."""
    def __init__(self, module_name: str, success: bool, message: str, rollback: bool = False):
        self.module_name = module_name
        self.success = success
        self.message = message
        self.rollback = rollback
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "module": self.module_name,
            "success": self.success,
            "message": self.message,
            "rollback": self.rollback,
            "timestamp": self.timestamp,
        }


class Reloader:
    """
    안전한 Hot-Swapping 엔진.
    
    교체 전 자동 백업, 교체 후 기본 검증, 실패 시 즉시 롤백.
    """
    
    MAX_BACKUPS = 10  # 모듈당 최대 백업 수

    def __init__(self):
        self._swap_history: list[dict] = []

    def _get_module_path(self, module_name: str) -> Optional[Path]:
        """모듈 이름에서 파일 경로를 추출합니다."""
        parts = module_name.split(".")
        # 프로젝트 루트 기준으로 경로 구성
        candidate = ROOT / "/".join(parts)
        if candidate.with_suffix(".py").exists():
            return candidate.with_suffix(".py")
        # __init__.py 확인
        init_path = candidate / "__init__.py"
        if init_path.exists():
            return init_path
        return None

    def _backup_module(self, module_path: Path) -> Optional[Path]:
        """모듈 파일을 백업합니다."""
        if not module_path.exists():
            return None
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{module_path.stem}_{timestamp}{module_path.suffix}"
        backup_subdir = BACKUP_DIR / module_path.parent.name
        backup_subdir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_subdir / backup_name
        
        shutil.copy2(module_path, backup_path)
        logger.info("백업 생성: %s → %s", module_path.name, backup_path)
        
        # 오래된 백업 정리
        self._cleanup_old_backups(backup_subdir, module_path.stem)
        
        return backup_path

    def _cleanup_old_backups(self, backup_dir: Path, stem: str):
        """오래된 백업 파일을 정리합니다."""
        backups = sorted(
            backup_dir.glob(f"{stem}_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_backup in backups[self.MAX_BACKUPS:]:
            old_backup.unlink()
            logger.debug("오래된 백업 삭제: %s", old_backup.name)

    def _restore_from_backup(self, backup_path: Path, module_path: Path) -> bool:
        """백업에서 모듈을 복원합니다."""
        try:
            shutil.copy2(backup_path, module_path)
            logger.warning("롤백 완료: %s → %s", backup_path.name, module_path.name)
            return True
        except Exception as e:
            logger.error("롤백 실패: %s", str(e))
            return False

    def hot_swap(self, module_name: str) -> HotSwapResult:
        """
        모듈을 안전하게 핫 스왑(런타임 교체)합니다.
        
        1. 현재 모듈 백업
        2. importlib.reload() 실행
        3. 기본 검증 (import 성공 여부)
        4. 실패 시 자동 롤백
        """
        module_path = self._get_module_path(module_name)
        if module_path is None:
            return HotSwapResult(module_name, False, f"모듈 파일을 찾을 수 없습니다: {module_name}")
        
        # Step 1: 백업
        backup_path = self._backup_module(module_path)
        
        # Step 2: reload
        try:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                importlib.reload(module)
                logger.info("핫 스왑 성공: %s (reload)", module_name)
            else:
                importlib.import_module(module_name)
                logger.info("핫 스왑 성공: %s (fresh import)", module_name)
            
            result = HotSwapResult(module_name, True, "핫 스왑 완료")
            
        except Exception as e:
            # Step 3: 롤백
            logger.error("핫 스왑 실패: %s | 에러: %s", module_name, str(e))
            
            rolled_back = False
            if backup_path:
                rolled_back = self._restore_from_backup(backup_path, module_path)
                # 롤백 후 다시 reload
                if rolled_back and module_name in sys.modules:
                    try:
                        importlib.reload(sys.modules[module_name])
                    except Exception:
                        pass
            
            result = HotSwapResult(
                module_name, False,
                f"핫 스왑 실패: {str(e)}" + (" (롤백 완료)" if rolled_back else " (롤백 실패)"),
                rollback=rolled_back,
            )
        
        self._swap_history.append(result.to_dict())
        return result

    def get_history(self) -> list[dict]:
        """핫 스왑 이력 조회."""
        return list(reversed(self._swap_history[-20:]))

    def list_backups(self) -> dict:
        """백업 현황 조회."""
        backups = {}
        for subdir in BACKUP_DIR.iterdir():
            if subdir.is_dir():
                files = sorted(subdir.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
                backups[subdir.name] = [
                    {"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
                    for f in files[:5]
                ]
        return backups


# 모듈 레벨 싱글턴
reloader = Reloader()
