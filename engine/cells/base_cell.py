from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseCell(ABC):
    """
    모든 셀 에이전트의 추상 베이스 클래스.
    각 셀은 독립적인 실행 단위이며, 입력을 받아 처리를 수행하고 출력을 반환합니다.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        초기화
        
        Args:
            name: 셀의 고유 이름
            config: 셀 별 설정 딕셔너리
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(self.name)
        self.state = "IDLE"  # IDLE, RUNNING, DONE, ERROR
        self._execution_time = 0.0

    def validate_inputs(self, inputs: Dict[str, Any], required_keys: list) -> None:
        """입력 딕셔너리에 필수 키가 존재하는지 검증합니다.

        Args:
            inputs: 검증할 입력 딕셔너리.
            required_keys: 반드시 존재해야 하는 키 목록.

        Raises:
            ValueError: 필수 키가 누락된 경우.
        """
        missing = [k for k in required_keys if k not in inputs]
        if missing:
            raise ValueError(
                f"[{self.name}] Missing required input keys: {missing}"
            )

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        셀의 메인 로직을 실행하는 추상 메서드.
        하위 클래스에서 반드시 구현해야 합니다.
        
        Args:
            inputs: 이전 셀이나 Orchestrator로부터 전달받은 데이터
            
        Returns:
            Dict[str, Any]: 다음 단계로 넘길 결과 데이터
        """
        pass

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        실행 래퍼 메서드. 상태 관리와 시간 측정을 담당합니다.
        """
        self.state = "RUNNING"
        self.logger.info(f"[START] Execution started")
        start_time = time.time()
        
        try:
            results = self.execute(inputs)
            self.state = "DONE"
            self._execution_time = time.time() - start_time
            self.logger.info(f"[DONE] Execution completed ({self._execution_time:.2f}s)")
            return results
        except Exception as e:
            self.state = "ERROR"
            import traceback
            self.logger.error(f"[ERROR] Execution failed: {str(e)}\n{traceback.format_exc()}")
            raise e
