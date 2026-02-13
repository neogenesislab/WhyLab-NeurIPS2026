# -*- coding: utf-8 -*-
"""WhyLab 중앙 설정 모듈.

모든 하이퍼파라미터, 경로, 시드를 한 곳에서 관리합니다.
매직 넘버를 철저히 금지하고, 재현 가능성을 보장합니다.

사용법:
    from engine.config import DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG
    print(cfg.data.n_samples)  # 100_000
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ──────────────────────────────────────────────
# 데이터 생성 설정
# ──────────────────────────────────────────────

@dataclass
class DataConfig:
    """합성 데이터 생성 하이퍼파라미터.

    시나리오 A: 신용 한도(연속형 Treatment) → 연체 여부(이진 Outcome)
    시나리오 B: 투자 쿠폰(이진형 Treatment) → 가입 여부(이진 Outcome)
    """

    n_samples: int = 100_000
    random_seed: int = 42

    # 외생 변수 분포 파라미터
    income_log_mean: float = 8.0         # 소득 로그정규분포 평균 (≈ 연 3,000만 원)
    income_log_sigma: float = 0.5        # 소득 로그정규분포 표준편차
    age_mean: float = 35.0               # 나이 정규분포 평균
    age_std: float = 10.0                # 나이 정규분포 표준편차
    credit_score_mean: float = 650.0     # 신용점수 정규분포 평균 (KCB 기준)
    credit_score_std: float = 80.0       # 신용점수 정규분포 표준편차
    app_usage_scale: float = 30.0        # 앱 사용시간 지수분포 스케일 (분/일)
    consumption_income_coef: float = 0.3 # 소비 = coef * 소득 + 노이즈

    # 시나리오 A: 신용 한도 (Continuous Treatment)
    treat_income_coef: float = 0.5
    treat_age_coef: float = -0.3
    treat_credit_coef: float = 0.4
    treat_noise_std: float = 0.5
    treat_min: float = 100.0             # 최소 한도 (만 원)
    treat_max: float = 5000.0            # 최대 한도 (만 원)

    # 시나리오 A Ground Truth CATE (세그먼트별 극적 차이를 위해 강한 계수)
    cate_income_coef: float = -0.50      # 소득 ↑ → 한도의 연체 감소 효과 ↑↑
    cate_age_coef: float = 0.25          # 나이 ↑ → 한도의 연체 감소 효과 ↓
    cate_credit_coef: float = -0.40      # 신용 ↑ → 한도의 연체 감소 효과 ↑↑

    # 시나리오 A Outcome
    outcome_income_coef: float = -0.6    # 소득 ↑ → 연체 ↓
    outcome_consumption_coef: float = 0.2  # 소비 ↑ → 연체 ↓
    outcome_credit_coef: float = -0.4    # 신용 ↑ → 연체 ↓
    outcome_noise_std: float = 0.3

    # 시나리오 B: 크로스셀링 쿠폰 (Binary Treatment)
    cate_b_base: float = 0.05
    cate_b_income_coef: float = -0.2     # 소득 낮을수록 효과 큼
    cate_b_age_coef: float = -0.15       # 나이 어릴수록 효과 큼

    # 범위 클리핑
    age_min: float = 20.0
    age_max: float = 70.0
    credit_score_min: float = 300.0
    credit_score_max: float = 900.0
    app_usage_min: float = 1.0
    app_usage_max: float = 200.0


# ──────────────────────────────────────────────
# DML 모델링 설정
# ──────────────────────────────────────────────

@dataclass
class DMLConfig:
    """Double Machine Learning 모델 설정."""

    model_type: str = "linear"           # "linear" | "forest" | "auto" (AutoML)
    cv_folds: int = 5                    # Cross-Fitting 폴드 수
    alpha: float = 0.05                  # 신뢰구간 유의수준 (1 - α = 95%)
    
    # AutoML 설정
    auto_ml: bool = False                # AutoML 활성화 여부
    candidate_models: list[str] = field( # 경쟁 모델 리스트
        default_factory=lambda: ["linear", "forest"]
    )
    
    # LightGBM Nuisance 모델 하이퍼파라미터
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.05
    lgbm_max_depth: int = 5
    lgbm_num_leaves: int = 31
    lgbm_verbose: int = -1               # 로그 억제

    # GPU 가속 설정 (NVIDIA RTX 4070 SUPER 12GB)
    use_gpu: bool = True                 # LightGBM GPU 모드 활성화
    gpu_device_id: int = 0               # GPU 디바이스 ID

# ──────────────────────────────────────────────
# PyTorch Nuisance 모델 설정
# ──────────────────────────────────────────────

@dataclass
class NuisanceConfig:
    """PyTorch 기반 심층 Nuisance 모델 설정."""

    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 1e-3
    dropout: float = 0.3
    use_fp16: bool = True                # Mixed Precision (torch.cuda.amp)
    use_gpu: bool = True                 # CUDA 자동 감지
    weight_decay: float = 1e-4           # AdamW 가중치 감쇠


# ──────────────────────────────────────────────
# 민감도 분석 (Sensitivity Analysis) 설정
# ──────────────────────────────────────────────

@dataclass
class SensitivityConfig:
    """인과 추론 결과의 견고성(Robustness) 검증 설정."""

    enabled: bool = True                 # 민감도 분석 수행 여부
    n_simulations: int = 10              # 시뮬레이션 반복 횟수
    placebo_treatment: bool = True       # 처치 변수 랜덤 셔플링 검증
    random_common_cause: bool = True     # 무작위 교란 변수 추가 검증
    significance_threshold: float = 0.05 # p-value 임계값
    e_value: bool = True                 # E-value (미관측 교란 견고성)
    overlap: bool = True                 # Overlap(Positivity) 진단
    gates: bool = True                   # GATES/CLAN 이질성 심화 분석
    n_gates_groups: int = 4              # GATES 그룹 수 (사분위)

    # Refutation (진짜 반증) 설정
    n_refutation_iter: int = 20          # Placebo Test 반복 횟수
    n_bootstrap: int = 100               # Bootstrap CI 반복 횟수


# ──────────────────────────────────────────────
# 설명(Explainability) 설정
# ──────────────────────────────────────────────

@dataclass
class ExplainConfig:
    """SHAP 설명 및 반사실 시뮬레이션 설정."""

    shap_sample_size: int = 1000         # SHAP 계산용 샘플 수
    top_k_features: int = 10             # 상위 K개 피처 표시
    n_counterfactual: int = 5            # 반사실 시나리오 수


# ──────────────────────────────────────────────
# 시각화 설정
# ──────────────────────────────────────────────

@dataclass
class VizConfig:
    """시각화 출력 설정."""

    figure_dpi: int = 150
    figure_format: str = "png"
    color_palette: str = "viridis"
    max_scatter_points: int = 5000       # 산점도 최대 점 수 (성능)
    font_family: str = "Malgun Gothic"   # 한글 폰트 (Windows)


# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────

@dataclass
class PathConfig:
    """프로젝트 디렉토리 경로 관리."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    @property
    def engine_dir(self) -> Path:
        """엔진 루트 디렉토리."""
        return self.project_root / "engine"

    @property
    def data_dir(self) -> Path:
        """합성 데이터 저장 디렉토리."""
        return self.project_root / "paper" / "data"

    @property
    def figures_dir(self) -> Path:
        """Figure 이미지 저장 디렉토리."""
        return self.project_root / "paper" / "figures"

    @property
    def dashboard_data_dir(self) -> Path:
        """대시보드 JSON 데이터 디렉토리."""
        return self.project_root / "dashboard" / "public" / "data"

    @property
    def reports_dir(self) -> Path:
        """White Paper 보고서 디렉토리."""
        return self.project_root / "paper" / "reports"

    def ensure_dirs(self) -> None:
        """필요한 모든 출력 디렉토리를 생성합니다."""
        for directory in [
            self.data_dir,
            self.figures_dir,
            self.dashboard_data_dir,
            self.reports_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 전역 설정 (Single Source of Truth)
# ──────────────────────────────────────────────

@dataclass
class WhyLabConfig:
    """WhyLab 전역 설정."""

    data: DataConfig = field(default_factory=DataConfig)
    dml: DMLConfig = field(default_factory=DMLConfig)
    nuisance: NuisanceConfig = field(default_factory=NuisanceConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# 기본 설정 인스턴스 — import 시 즉시 사용 가능
DEFAULT_CONFIG = WhyLabConfig()
