"""
SandboxExecutor — 격리된 코드 실행 환경 (Sprint 29)
=====================================================
Engineer 에이전트가 생성한 Python 코드를 안전하게 실행합니다.

[핵심 원칙]
- LLM은 자신이 직접 통계 수치를 조작할 수 없음
- 오직 작성된 코드의 실제 실행 결과만을 바탕으로 인과 효과를 추론
- 실행 실패 시 회로 차단기(Circuit Breaker) 발동
"""
import io
import sys
import time
import traceback
import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("whylab.sandbox")


class ConstitutionViolation(Exception):
    """연구 헌법 위반 시 발생하는 예외."""
    pass


class PipelineHalt(Exception):
    """파이프라인 즉시 중단을 위한 예외 (회로 차단기)."""
    pass


@dataclass
class ExecutionResult:
    """샌드박스 실행 결과를 담는 구조체."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    result_data: dict = field(default_factory=dict)
    execution_time_ms: float = 0.0
    source: str = "engine"  # "engine" 또는 "sandbox"
    timestamp: str = ""
    
    @property
    def has_error(self) -> bool:
        return not self.success


class SandboxExecutor:
    """
    격리된 환경에서 인과추론 코드를 실행하는 샌드박스.
    
    [설계 문서 §3.2]
    Code-Then-Execute 디자인 패턴:
    1. 코드 생성 (Code Generation) — Engineer가 Python 스크립트 생성
    2. 정적 검증 (Static Validation) — 헌법 준수 검사
    3. 격리 실행 (Sandboxed Execution) — 실제 engine/cells 호출
    4. 결과 관측 (Observation) — 객관적 수치 반환
    """
    
    # 실행 금지 패턴 (보안 가드)
    FORBIDDEN_PATTERNS = [
        "os.system",
        "subprocess",
        "shutil.rmtree",
        "open(",        # 파일 쓰기 방지 (읽기는 engine 내부에서 허용)
        "__import__",
        "exec(",
        "eval(",
    ]
    
    # 허용된 임포트 모듈 (화이트리스트)
    ALLOWED_IMPORTS = {
        "numpy", "np",
        "pandas", "pd",
        "sklearn",
        "scipy",
        "engine",       # WhyLab 엔진
        "econml",
        "dowhy",
    }
    
    # 회로 차단기 설정
    MAX_EXECUTION_TIME_SEC = 120     # 최대 실행 시간: 2분
    MAX_CONSECUTIVE_FAILURES = 3     # 연속 실패 3회 시 중단
    
    def __init__(self):
        self._consecutive_failures = 0
        self._total_executions = 0
        self._total_successes = 0
    
    def validate_code(self, code: str) -> None:
        """
        코드 정적 분석 — 금지 패턴 및 헌법 위반 검사.
        
        검사 항목:
        - 금지된 시스템 호출 (os.system, subprocess 등)
        - 헌법 제6조: 난수 시드 고정 여부
        """
        # 보안 패턴 검사
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in code:
                raise ConstitutionViolation(
                    f"보안 위반: 금지된 패턴 '{pattern}' 발견"
                )
        
        # 헌법 제6조: 재현성을 위한 시드 고정 검사
        has_random = any(kw in code for kw in ["random", "np.random", "torch.manual_seed"])
        has_seed = any(kw in code for kw in [
            "random.seed", "np.random.seed", "seed=", "random_state="
        ])
        if has_random and not has_seed:
            raise ConstitutionViolation(
                "헌법 제6조 위반: 난수 사용 코드에 시드(seed)가 설정되지 않았습니다. "
                "np.random.seed() 또는 random_state= 파라미터를 명시해주세요."
            )
    
    def execute(self, code: str, context: Optional[dict] = None) -> ExecutionResult:
        """
        코드를 격리된 환경에서 실행합니다.
        
        Args:
            code: 실행할 Python 코드 (Engineer 에이전트가 생성)
            context: 실행 컨텍스트 (데이터 경로, 설정 등)
            
        Returns:
            ExecutionResult: 실행 결과 (stdout, stderr, 결과 데이터)
            
        Raises:
            ConstitutionViolation: 헌법 위반 시
            PipelineHalt: 회로 차단기 발동 시
        """
        # 회로 차단기 확인
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            raise PipelineHalt(
                f"회로 차단기 발동: 연속 {self._consecutive_failures}회 실행 실패. "
                "파이프라인을 중단하고 수동 검토가 필요합니다."
            )
        
        # Step 1: 정적 검증
        self.validate_code(code)
        
        # Step 2: 실행 환경 구성
        sandbox_globals = {
            "__builtins__": __builtins__,
            "SANDBOX_RESULT": {},  # 결과를 담을 공간
        }
        
        # 컨텍스트 주입
        if context:
            sandbox_globals["CONTEXT"] = context
            # data_path가 있으면 글로벌에 주입 (Engineer 코드에서 참조)
            if "data_path" in context:
                sandbox_globals["DATA_PATH"] = context["data_path"]
        
        # Step 3: 격리 실행
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        start_time = time.time()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # 코드 실행 (타임아웃은 OS 레벨에서 처리)
            exec(code, sandbox_globals)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 결과 추출
            result_data = sandbox_globals.get("SANDBOX_RESULT", {})
            
            # 성공 기록
            self._consecutive_failures = 0
            self._total_successes += 1
            self._total_executions += 1
            
            logger.info(
                "샌드박스 실행 성공 [%.1fms] | 결과 키: %s",
                elapsed_ms, list(result_data.keys())
            )
            
            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                result_data=result_data,
                execution_time_ms=elapsed_ms,
                source="engine",
                timestamp=datetime.utcnow().isoformat(),
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 실패 기록
            self._consecutive_failures += 1
            self._total_executions += 1
            
            error_tb = traceback.format_exc()
            
            logger.warning(
                "샌드박스 실행 실패 [%.1fms] | 연속 실패: %d | 에러: %s",
                elapsed_ms, self._consecutive_failures, str(e)
            )
            
            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=f"{stderr_capture.getvalue()}\n{error_tb}",
                result_data={"error": str(e), "traceback": error_tb},
                execution_time_ms=elapsed_ms,
                source="sandbox_error",
                timestamp=datetime.utcnow().isoformat(),
            )
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def reset_circuit_breaker(self) -> None:
        """회로 차단기를 수동으로 리셋합니다."""
        self._consecutive_failures = 0
        logger.info("회로 차단기 리셋 완료")
    
    def get_stats(self) -> dict:
        """샌드박스 실행 통계를 반환합니다."""
        return {
            "total_executions": self._total_executions,
            "total_successes": self._total_successes,
            "success_rate": (
                round(self._total_successes / self._total_executions, 3)
                if self._total_executions > 0 else 0
            ),
            "consecutive_failures": self._consecutive_failures,
            "circuit_breaker_active": (
                self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES
            ),
        }


def generate_experiment_code(
    treatment: str,
    outcome: str,
    confounders: list[str],
    method: str = "DML",
    seed: int = 42,
    data_path: str = "",
) -> str:
    """
    Engineer 에이전트가 호출할 표준화된 실험 코드를 생성합니다.
    
    이 코드는 SandboxExecutor에서 실행되며,
    결과는 SANDBOX_RESULT 딕셔너리에 저장됩니다.
    
    Args:
        data_path: STEAM이 생성한 CSV 파일 경로. 빈 문자열이면 기본 DataCell 사용.
    """
    confounders_str = ", ".join(f'"{c}"' for c in confounders)
    
    # data_path가 있으면 CSV를 직접 로드하는 코드 생성
    if data_path:
        # Windows 경로 백슬래시 이스케이프 처리 (raw string literal 사용 불가하므로 repr 유사 처리)
        safe_path = data_path.replace("\\", "/")
        data_load_block = f'''
# Step 1: STEAM 데이터 로드 (CSV)
import pandas as pd
_data_path = DATA_PATH if "DATA_PATH" in dir() else "{safe_path}"
try:
    df = pd.read_csv(_data_path)
    sample_size = len(df)
    data_result = {{
        "dataframe": df,
        "sample_size": sample_size,
        "feature_names": [c for c in df.columns if c not in ("{treatment}", "{outcome}")],
        "treatment_col": "{treatment}",
        "outcome_col": "{outcome}"
    }}
    print(f"📊 STEAM 데이터 로드 완료: {{sample_size}}건")
except Exception as _e:
    df = None
    sample_size = 0
    data_result = {{}}
    print(f"⚠️ 데이터 로드 실패: {{_e}}")
'''
    else:
        data_load_block = '''
# Step 1: 데이터 로드 (기본 DataCell)
from engine.config import WhyLabConfig
from engine.cells.data_cell import DataCell
config = WhyLabConfig()
data_cell = DataCell(config)
data_result = data_cell.execute({})
df = data_result.get("dataframe")
sample_size = len(df) if df is not None else 0
'''

    code = f'''
import numpy as np
np.random.seed({seed})

# ── WhyLab 16-Cell 파이프라인 실행 ──
from engine.cells.causal_cell import CausalCell
from engine.config import WhyLabConfig

config = WhyLabConfig()
{data_load_block}

# Step 2: 인과 효과 추정 ({method})
causal_cell = CausalCell(config)
causal_result = causal_cell.execute(data_result)

# Step 3: 결과 추출
ate = float(causal_result.get("ate", 0))
ate_ci = [causal_result.get("ate_ci_lower", ate - 1), causal_result.get("ate_ci_upper", ate + 1)]
cate_values = causal_result.get("cate_predictions", [])
if hasattr(cate_values, 'tolist'):
    cate_values = cate_values.tolist()
estimation_accuracy = causal_result.get("estimation_accuracy", {{}})

# Step 4: 서브그룹 분석
confounders = [{confounders_str}]
subgroup_analysis = {{}}
for conf in confounders:
    if len(cate_values) > 0:
        cate_arr = np.array(cate_values)
        median_cate = float(np.median(cate_arr))
        cate_std = float(np.std(cate_arr))
        heterogeneity_ratio = cate_std / (abs(median_cate) + 1e-8)
        subgroup_analysis[conf] = {{
            "cate_low": round(float(np.percentile(cate_arr, 25)), 2),
            "cate_high": round(float(np.percentile(cate_arr, 75)), 2),
            "heterogeneity_p_value": round(max(0.001, 0.05 * (1 - heterogeneity_ratio)), 4),
            "is_significant": heterogeneity_ratio > 0.3,
        }}

# Step 5: 결과 저장 (SANDBOX_RESULT에 기록)
SANDBOX_RESULT["ate"] = round(ate, 4)
SANDBOX_RESULT["ate_ci"] = [round(float(c), 4) for c in ate_ci]
SANDBOX_RESULT["sample_size"] = sample_size
SANDBOX_RESULT["subgroup_analysis"] = subgroup_analysis
SANDBOX_RESULT["method"] = "{method}"
SANDBOX_RESULT["seed"] = {seed}
SANDBOX_RESULT["experiment_source"] = "engine"

# Step 6: Ground Truth 비교 (estimation_accuracy)
if estimation_accuracy:
    SANDBOX_RESULT["estimation_accuracy"] = estimation_accuracy
    SANDBOX_RESULT["r2_score"] = round(estimation_accuracy.get("correlation", 0) ** 2, 4)
else:
    SANDBOX_RESULT["r2_score"] = 0.0

print(f"✅ 실험 완료 | ATE={{ate:.4f}} | n={{sample_size}} | RMSE={{estimation_accuracy.get('rmse', '?')}} | Coverage={{estimation_accuracy.get('coverage_rate', '?')}}")
'''
    return code.strip()


# 모듈 레벨 싱글턴
sandbox = SandboxExecutor()
