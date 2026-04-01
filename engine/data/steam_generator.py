"""
STEAM Synthetic Data Generator (Sprint 31)
============================================
Synthetic Treatment Effect and Attribution for Machine-learning

Grand Challenge별 인과적으로 일관된 합성 데이터를 생성합니다.
실제 데이터가 없는 환경에서도 WhyLab 엔진을 완전하게 테스트할 수 있도록
Ground Truth τ(x)가 포함된 합성 데이터를 만듭니다.

[핵심 원리]
- Structural Causal Model (SCM) 기반 DGP: 인과 구조가 보존된 데이터
- 교란 변수(Confounders) + 조절 변수(Moderators) 명시적 분리
- UPEHE, JSD_π 등 합성 데이터 품질 메트릭 자동 산출

사용 예시:
    generator = STEAMGenerator()
    data = generator.generate("labor_market", n=5000, seed=42)
    metrics = generator.evaluate_quality(data)
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("whylab.steam")


@dataclass
class SyntheticData:
    """STEAM 합성 데이터 컨테이너."""
    df: pd.DataFrame                        # 전체 데이터프레임
    treatment_col: str                      # 처치 변수명
    outcome_col: str                        # 결과 변수명
    confounder_cols: list[str]              # 교란 변수 목록
    moderator_cols: list[str]               # 조절 변수 목록
    tau_true: np.ndarray                    # Ground Truth CATE
    y0: np.ndarray                          # 반사실 결과 (미처치)
    y1: np.ndarray                          # 반사실 결과 (처치)
    dgp_name: str                           # DGP 이름
    grand_challenge_id: Optional[str] = None
    seed: int = 42
    metadata: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.df)

    @property
    def ate_true(self) -> float:
        return float(np.mean(self.tau_true))


@dataclass
class DGPTemplate:
    """Data-Generating Process 템플릿."""
    name: str
    grand_challenge_id: str
    category: str
    description: str
    treatment_name: str
    outcome_name: str
    confounders: list[str]
    moderators: list[str]
    n_default: int = 3000
    effect_type: str = "heterogeneous"  # "constant" | "heterogeneous" | "nonlinear"


# ── Grand Challenge별 DGP 템플릿 ──
DGP_TEMPLATES: dict[str, DGPTemplate] = {
    "ai_alignment": DGPTemplate(
        name="AI Alignment & Human Values",
        grand_challenge_id="GC_001",
        category="AI Ethics",
        description="AI의 인간 가치 정렬 정도가 의사결정 품질에 미치는 효과",
        treatment_name="alignment_training",
        outcome_name="decision_quality",
        confounders=["model_complexity", "data_diversity", "training_budget"],
        moderators=["domain_specificity", "human_oversight_level"],
        n_default=3000,
        effect_type="nonlinear",
    ),
    "labor_market": DGPTemplate(
        name="AI Impact on Labor Market",
        grand_challenge_id="GC_002",
        category="Economy",
        description="AI 도입이 고용 안정성에 미치는 인과적 영향",
        treatment_name="ai_adoption_level",
        outcome_name="employment_stability",
        confounders=["industry_sector", "education_level", "experience_years", "company_size"],
        moderators=["skill_adaptability", "job_complexity"],
        n_default=5000,
        effect_type="heterogeneous",
    ),
    "carbon_tax": DGPTemplate(
        name="Carbon Tax Effectiveness",
        grand_challenge_id="GC_003",
        category="Climate",
        description="탄소세가 기업 탄소 배출량 감축에 미치는 효과",
        treatment_name="carbon_tax_rate",
        outcome_name="emission_reduction",
        confounders=["industry_type", "revenue", "baseline_emission", "energy_source"],
        moderators=["green_tech_adoption", "regulation_stringency"],
        n_default=4000,
        effect_type="heterogeneous",
    ),
    "social_media_mental_health": DGPTemplate(
        name="Social Media & Mental Health",
        grand_challenge_id="GC_004",
        category="Society",
        description="숏폼 콘텐츠 소비 시간이 청소년 우울증에 미치는 인과적 영향",
        treatment_name="daily_screen_time_hours",
        outcome_name="depression_score",
        confounders=["age", "family_income", "peer_social_support", "prior_mental_health"],
        moderators=["content_type_ratio", "offline_activity_hours"],
        n_default=3000,
        effect_type="nonlinear",
    ),
    "telemedicine": DGPTemplate(
        name="Telemedicine & Accessibility",
        grand_challenge_id="GC_005",
        category="Healthcare",
        description="원격 진료 허용이 의료 소외 지역 수명 연장에 미치는 효과",
        treatment_name="telemedicine_access",
        outcome_name="health_outcome_score",
        confounders=["distance_to_hospital", "income_level", "chronic_conditions", "age"],
        moderators=["digital_literacy", "insurance_coverage"],
        n_default=4000,
        effect_type="heterogeneous",
    ),
}


class STEAMGenerator:
    """
    STEAM 합성 데이터 생성기.
    
    SCM 기반으로 인과적 일관성이 보장된 합성 데이터를 생성합니다.
    각 데이터 포인트에 대해 Ground Truth CATE(τ(x))가 포함됩니다.
    """

    def __init__(self):
        self._templates = DGP_TEMPLATES.copy()

    @property
    def available_dgps(self) -> list[str]:
        """사용 가능한 DGP 이름 목록."""
        return list(self._templates.keys())

    def generate(
        self,
        dgp_name: str,
        n: Optional[int] = None,
        seed: int = 42,
        noise_scale: float = 1.0,
    ) -> SyntheticData:
        """
        지정된 DGP 템플릿으로 합성 데이터를 생성합니다.
        
        Args:
            dgp_name: DGP 템플릿 이름 (예: "labor_market")
            n: 표본 크기 (None이면 기본값 사용)
            seed: 랜덤 시드 (헌법 제6조 준수)
            noise_scale: 노이즈 스케일 (1.0 = 기본)
            
        Returns:
            SyntheticData: Ground Truth τ(x) 포함 합성 데이터
        """
        if dgp_name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise ValueError(f"알 수 없는 DGP: '{dgp_name}'. 사용 가능: {available}")

        template = self._templates[dgp_name]
        n = n or template.n_default
        rng = np.random.RandomState(seed)

        logger.info(
            "STEAM 데이터 생성 시작 | DGP=%s | n=%d | seed=%d",
            dgp_name, n, seed
        )

        # ── SCM 기반 데이터 생성 ──
        data = {}

        # Step 1: 교란 변수 (U → X)
        for i, conf in enumerate(template.confounders):
            data[conf] = rng.normal(0, 1, n)

        # Step 2: 조절 변수 (독립)
        for mod in template.moderators:
            data[mod] = rng.normal(0, 1, n)

        # Step 3: 처치 배정 (교란에 의존 — selection bias 생성)
        confounders_matrix = np.column_stack(
            [data[c] for c in template.confounders]
        )
        propensity_logit = 0.3 * confounders_matrix.sum(axis=1) + rng.normal(0, 0.5, n)
        propensity = 1 / (1 + np.exp(-propensity_logit))
        treatment = (rng.uniform(0, 1, n) < propensity).astype(float)
        data[template.treatment_name] = treatment

        # Step 4: CATE 생성 (effect_type에 따라 분기)
        moderators_matrix = np.column_stack(
            [data[m] for m in template.moderators]
        ) if template.moderators else np.zeros((n, 1))

        tau_true = self._generate_cate(
            template.effect_type, confounders_matrix, moderators_matrix, rng
        )

        # Step 5: 잠재적 결과 생성 (Rubin의 반사실 프레임워크)
        y0 = (
            2.0 * confounders_matrix[:, 0]
            + 1.5 * np.sin(confounders_matrix[:, 1] if confounders_matrix.shape[1] > 1 else 0)
            + rng.normal(0, noise_scale, n)
        )
        y1 = y0 + tau_true

        # Step 6: 관측된 결과 (SUTVA)
        observed_outcome = treatment * y1 + (1 - treatment) * y0
        data[template.outcome_name] = observed_outcome

        df = pd.DataFrame(data)
        # Ground Truth CATE를 DataFrame에 포함 → CausalCell이 자동 검증
        df["true_cate"] = tau_true

        result = SyntheticData(
            df=df,
            treatment_col=template.treatment_name,
            outcome_col=template.outcome_name,
            confounder_cols=template.confounders,
            moderator_cols=template.moderators,
            tau_true=tau_true,
            y0=y0,
            y1=y1,
            dgp_name=dgp_name,
            grand_challenge_id=template.grand_challenge_id,
            seed=seed,
            metadata={
                "ate_true": float(np.mean(tau_true)),
                "cate_std": float(np.std(tau_true)),
                "propensity_mean": float(np.mean(propensity)),
                "treatment_rate": float(np.mean(treatment)),
                "noise_scale": noise_scale,
            },
        )

        logger.info(
            "STEAM 생성 완료 | n=%d | ATE(true)=%.3f | CATE_std=%.3f | 처치율=%.1f%%",
            n, result.ate_true, np.std(tau_true), np.mean(treatment) * 100
        )

        return result

    @staticmethod
    def _generate_cate(
        effect_type: str,
        X_conf: np.ndarray,
        X_mod: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """CATE 생성 (DGP 유형별 분기)."""
        n = X_conf.shape[0]

        if effect_type == "constant":
            # 동질적 처리 효과
            return np.full(n, 3.0)

        elif effect_type == "heterogeneous":
            # 선형 이질적 효과: τ(x) = β₀ + β₁·x₁ + β₂·m₁
            base_effect = 2.0
            confounder_effect = 1.5 * X_conf[:, 0] if X_conf.shape[1] > 0 else 0
            moderator_effect = 1.0 * X_mod[:, 0] if X_mod.shape[1] > 0 else 0
            return base_effect + confounder_effect + moderator_effect

        elif effect_type == "nonlinear":
            # 비선형 이질적 효과: τ(x) = 2 + 3·sigmoid(x₁) + x₂² + m₁·x₁
            x1 = X_conf[:, 0] if X_conf.shape[1] > 0 else np.zeros(n)
            x2 = X_conf[:, 1] if X_conf.shape[1] > 1 else np.zeros(n)
            m1 = X_mod[:, 0] if X_mod.shape[1] > 0 else np.zeros(n)

            sigmoid = 1 / (1 + np.exp(-x1))
            return 2.0 + 3.0 * sigmoid + 0.5 * x2**2 + 0.8 * m1 * x1

        else:
            raise ValueError(f"알 수 없는 effect_type: {effect_type}")

    def evaluate_quality(self, data: SyntheticData) -> dict:
        """
        합성 데이터 품질 메트릭을 산출합니다.
        
        Metrics:
        - UPEHE (Uncertainty-weighted PEHE): 처치 효과 예측 정확도
        - JSD_π (Jensen-Shannon Divergence of propensity): 공변피 균형
        - Overlap coefficient: 처치/대조 분포 겹침 정도
        """
        tau = data.tau_true
        df = data.df
        t = df[data.treatment_col].values
        y = df[data.outcome_col].values

        # ── PEHE (Oracle): √E[(τ_true - τ_est)²] → 0에 가까울수록 좋음 ──
        # Oracle PEHE는 tau_true를 모를 때 추정치와의 차이이므로,
        # 여기서는 naive ATE와의 차이를 산출
        naive_ate = y[t == 1].mean() - y[t == 0].mean() if (t == 1).any() and (t == 0).any() else 0
        pehe_naive = float(np.sqrt(np.mean((tau - naive_ate) ** 2)))

        # ── JSD_π: 처치/대조 그룹의 공변량 분포 균형 ──
        jsd_values = []
        for col in data.confounder_cols:
            treated = df.loc[t == 1, col].values
            control = df.loc[t == 0, col].values
            if len(treated) > 0 and len(control) > 0:
                jsd = self._compute_jsd(treated, control)
                jsd_values.append(jsd)

        jsd_mean = float(np.mean(jsd_values)) if jsd_values else 0.0

        # ── Overlap coefficient ──
        propensity_logit = np.column_stack(
            [df[c].values for c in data.confounder_cols]
        ).sum(axis=1) * 0.3
        propensity = 1 / (1 + np.exp(-propensity_logit))
        overlap = float(np.mean(np.minimum(propensity, 1 - propensity) * 2))

        # ── CATE 이질성 비율 ──
        heterogeneity = float(np.std(tau) / (np.abs(np.mean(tau)) + 1e-8))

        metrics = {
            "pehe_naive": round(pehe_naive, 4),
            "jsd_pi_mean": round(jsd_mean, 4),
            "overlap_coefficient": round(overlap, 4),
            "heterogeneity_ratio": round(heterogeneity, 4),
            "ate_true": round(float(np.mean(tau)), 4),
            "cate_std": round(float(np.std(tau)), 4),
            "sample_size": data.n,
            "treatment_rate": round(float(np.mean(t)), 4),
            "quality_grade": self._grade_quality(pehe_naive, jsd_mean, overlap),
        }

        logger.info(
            "STEAM 품질 평가 | PEHE=%.4f | JSD=%.4f | Overlap=%.3f | 등급=%s",
            pehe_naive, jsd_mean, overlap, metrics["quality_grade"]
        )

        return metrics

    @staticmethod
    def _compute_jsd(p_samples: np.ndarray, q_samples: np.ndarray, bins: int = 50) -> float:
        """Jensen-Shannon Divergence (히스토그램 기반)."""
        min_val = min(p_samples.min(), q_samples.min())
        max_val = max(p_samples.max(), q_samples.max())
        edges = np.linspace(min_val, max_val, bins + 1)

        p_hist, _ = np.histogram(p_samples, bins=edges, density=True)
        q_hist, _ = np.histogram(q_samples, bins=edges, density=True)

        # 정규화
        p_hist = p_hist / (p_hist.sum() + 1e-10)
        q_hist = q_hist / (q_hist.sum() + 1e-10)

        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
        m = 0.5 * (p_hist + q_hist)
        kl_pm = np.sum(p_hist * np.log((p_hist + 1e-10) / (m + 1e-10)))
        kl_qm = np.sum(q_hist * np.log((q_hist + 1e-10) / (m + 1e-10)))

        return float(0.5 * kl_pm + 0.5 * kl_qm)

    @staticmethod
    def _grade_quality(pehe: float, jsd: float, overlap: float) -> str:
        """합성 데이터 품질 등급 산출."""
        score = 0
        # PEHE: 낮을수록 좋음
        if pehe < 1.0:
            score += 3
        elif pehe < 2.0:
            score += 2
        elif pehe < 5.0:
            score += 1

        # JSD: 낮을수록 좋음 (균형 잡힌 처치/대조)
        if jsd < 0.05:
            score += 3
        elif jsd < 0.1:
            score += 2
        elif jsd < 0.2:
            score += 1

        # Overlap: 높을수록 좋음
        if overlap > 0.7:
            score += 3
        elif overlap > 0.5:
            score += 2
        elif overlap > 0.3:
            score += 1

        if score >= 8:
            return "S"
        elif score >= 6:
            return "A"
        elif score >= 4:
            return "B"
        elif score >= 2:
            return "C"
        return "F"


# 모듈 레벨 싱글턴
steam = STEAMGenerator()
