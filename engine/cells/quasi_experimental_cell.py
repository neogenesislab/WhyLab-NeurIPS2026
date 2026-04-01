# -*- coding: utf-8 -*-
"""QuasiExperimentalCell — IV/DiD/RDD 준실험 방법론.

관찰 데이터에서 미관측 교란 변수를 처리하기 위한
세 가지 준실험(Quasi-Experimental) 방법론을 제공합니다.

- **IV (Instrumental Variable)**: 2SLS 추정, 약한 도구 검정
- **DiD (Difference-in-Differences)**: 병렬 트렌드 검정
- **RDD (Regression Discontinuity)**: Sharp RDD + 대역폭 최적화

Phase 10-1: 방법론 확장.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.cells.base_cell import BaseCell
from engine.config import WhyLabConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 결과 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class IVResult:
    """도구 변수(IV) 추정 결과."""
    method: str = "2SLS"
    ate: float = 0.0
    se: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    f_stat: float = 0.0  # 1단계 F-통계량 (> 10이면 강한 도구)
    weak_instrument: bool = True
    instrument_col: str = ""
    interpretation: str = ""


@dataclass
class DiDResult:
    """이중차분법(DiD) 추정 결과."""
    method: str = "DiD"
    ate: float = 0.0
    se: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    parallel_trend_pvalue: float = 0.0
    parallel_trend_holds: bool = False
    n_treated: int = 0
    n_control: int = 0
    interpretation: str = ""


@dataclass
class RDDResult:
    """회귀 단절 설계(RDD) 추정 결과."""
    method: str = "Sharp_RDD"
    ate: float = 0.0
    se: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    cutoff: float = 0.0
    bandwidth: float = 0.0
    n_left: int = 0
    n_right: int = 0
    interpretation: str = ""


class QuasiExperimentalCell(BaseCell):
    """IV/DiD/RDD 준실험 방법론 셀.

    데이터 특성에 따라 적용 가능한 준실험 방법론을 자동으로
    실행하고, 결과를 비교합니다.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="quasi_experimental_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """준실험 방법론 실행.

        Args:
            inputs: 파이프라인 컨텍스트.

        Returns:
            준실험 분석 결과 추가된 inputs.
        """
        df = inputs.get("dataframe")
        if df is None:
            self.logger.warning("데이터프레임 없음 → QuasiExperimental 건너뜀")
            return inputs

        treatment_col = inputs.get("treatment_col", "treatment")
        outcome_col = inputs.get("outcome_col", "outcome")
        feature_names = inputs.get("feature_names", [])

        results = {}

        # IV 추정 시도
        instrument_col = inputs.get("instrument_col")
        if instrument_col and instrument_col in df.columns:
            self.logger.info("🔧 IV(2SLS) 추정 시작 (도구: %s)", instrument_col)
            results["iv"] = self._estimate_iv(
                df, treatment_col, outcome_col, instrument_col, feature_names
            )
        else:
            # 잠재 도구 변수 자동 탐색
            candidate = self._find_instrument_candidate(
                df, treatment_col, outcome_col, feature_names
            )
            if candidate:
                self.logger.info("🔍 잠재 도구 변수 발견: %s → IV 추정 시도", candidate)
                results["iv"] = self._estimate_iv(
                    df, treatment_col, outcome_col, candidate, feature_names
                )

        # DiD 추정 시도
        time_col = inputs.get("time_col")
        group_col = inputs.get("group_col")
        if time_col and group_col:
            self.logger.info("📊 DiD 추정 시작 (시간: %s, 그룹: %s)", time_col, group_col)
            results["did"] = self._estimate_did(
                df, outcome_col, time_col, group_col, feature_names
            )
        elif treatment_col in df.columns and df[treatment_col].nunique() == 2:
            # 이진 처치 → 간이 DiD 시뮬레이션
            results["did"] = self._simulate_did(
                df, treatment_col, outcome_col, feature_names
            )

        # RDD 추정 시도
        running_col = inputs.get("running_col")
        cutoff = inputs.get("rdd_cutoff")
        if running_col and cutoff is not None:
            self.logger.info("📐 RDD 추정 시작 (절단: %s=%.2f)", running_col, cutoff)
            results["rdd"] = self._estimate_rdd(
                df, outcome_col, running_col, cutoff
            )
        elif len(feature_names) > 0:
            # 연속 변수에서 RDD 후보 자동 탐색
            rdd_candidate = self._find_rdd_candidate(
                df, treatment_col, feature_names
            )
            if rdd_candidate:
                col, cut = rdd_candidate
                self.logger.info("🔍 RDD 후보 발견: %s (절단=%.2f)", col, cut)
                results["rdd"] = self._estimate_rdd(df, outcome_col, col, cut)

        # 결과 종합
        if results:
            methods_used = list(results.keys())
            self.logger.info(
                "✅ 준실험 분석 완료: %s", ", ".join(m.upper() for m in methods_used)
            )
        else:
            self.logger.info("ℹ️ 적용 가능한 준실험 방법론 없음 (건너뜀)")

        return {
            **inputs,
            "quasi_experimental": self._serialize_results(results),
        }

    # ── IV (Instrumental Variable) ──

    def _estimate_iv(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        instrument_col: str,
        covariates: List[str],
    ) -> IVResult:
        """2SLS(Two-Stage Least Squares) IV 추정."""
        from sklearn.linear_model import LinearRegression

        # 유효 컬럼 필터
        cov_cols = [c for c in covariates if c in df.columns and c != instrument_col]
        clean_df = df[[treatment_col, outcome_col, instrument_col] + cov_cols].dropna()

        Z = clean_df[instrument_col].values.reshape(-1, 1)
        Y = clean_df[outcome_col].values
        T = clean_df[treatment_col].values
        X = clean_df[cov_cols].values if cov_cols else np.ones((len(clean_df), 1))

        # 1단계: T ~ Z + X
        ZX = np.hstack([Z, X])
        stage1 = LinearRegression().fit(ZX, T)
        T_hat = stage1.predict(ZX)

        # 1단계 F-통계량 (약한 도구 검정)
        ss_res = np.sum((T - T_hat) ** 2)
        ss_tot = np.sum((T - T.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        n, k = ZX.shape
        f_stat = (r2 / max(1, k)) / ((1 - r2) / max(1, n - k - 1) + 1e-10)

        # 2단계: Y ~ T_hat + X
        TX = np.hstack([T_hat.reshape(-1, 1), X])
        stage2 = LinearRegression().fit(TX, Y)
        ate = float(stage2.coef_[0])

        # 표준 오차 (근사)
        residuals = Y - stage2.predict(TX)
        n_obs = len(Y)
        se = float(np.std(residuals) / np.sqrt(n_obs)) if n_obs > 0 else 0.0
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        weak = f_stat < 10
        interp = (
            f"IV(2SLS) 추정 ATE = {ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]. "
            f"1단계 F = {f_stat:.1f} ({'⚠️ 약한 도구' if weak else '✅ 강한 도구'})."
        )

        self.logger.info("   IV: ATE=%.4f, F=%.1f, 약한 도구=%s", ate, f_stat, weak)

        return IVResult(
            ate=ate, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
            f_stat=float(f_stat), weak_instrument=weak,
            instrument_col=instrument_col, interpretation=interp,
        )

    def _find_instrument_candidate(
        self, df: pd.DataFrame, treatment_col: str,
        outcome_col: str, features: List[str],
    ) -> Optional[str]:
        """잠재 도구 변수를 자동 탐색합니다.

        도구 변수 조건: Treatment과 상관 높고, Outcome과 직접 상관 낮음.
        """
        best_col = None
        best_score = 0
        for col in features:
            if col == treatment_col or col == outcome_col:
                continue
            if col not in df.columns:
                continue
            try:
                corr_t = abs(df[col].corr(df[treatment_col]))
                corr_y = abs(df[col].corr(df[outcome_col]))
                # 도구 조건: T와 상관 높고 (>0.3), Y와 직접 상관 낮음 (<0.15)
                if corr_t > 0.3 and corr_y < 0.15:
                    score = corr_t - corr_y
                    if score > best_score:
                        best_score = score
                        best_col = col
            except Exception:
                continue
        return best_col

    # ── DiD (Difference-in-Differences) ──

    def _estimate_did(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        time_col: str,
        group_col: str,
        covariates: List[str],
    ) -> DiDResult:
        """이중차분법(DiD) 추정."""
        # 사전/사후 기간 구분
        times = sorted(df[time_col].unique())
        if len(times) < 2:
            return DiDResult(interpretation="시간 기간이 2개 미만")

        pre_time = times[0]
        post_time = times[-1]

        # 그룹별 평균 계산
        groups = df[group_col].unique()
        if len(groups) < 2:
            return DiDResult(interpretation="그룹이 2개 미만")

        treated_group = groups[0]
        control_group = groups[1]

        pre_treated = df[(df[time_col] == pre_time) & (df[group_col] == treated_group)][outcome_col]
        post_treated = df[(df[time_col] == post_time) & (df[group_col] == treated_group)][outcome_col]
        pre_control = df[(df[time_col] == pre_time) & (df[group_col] == control_group)][outcome_col]
        post_control = df[(df[time_col] == post_time) & (df[group_col] == control_group)][outcome_col]

        # DiD 추정
        diff_treated = post_treated.mean() - pre_treated.mean()
        diff_control = post_control.mean() - pre_control.mean()
        ate = float(diff_treated - diff_control)

        # 표준 오차 (풀링된 분산)
        n_t = len(post_treated) + len(pre_treated)
        n_c = len(post_control) + len(pre_control)
        pooled_var = (post_treated.var() + pre_treated.var()) / n_t + \
                     (post_control.var() + pre_control.var()) / n_c
        se = float(np.sqrt(pooled_var + 1e-10))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        # 병렬 트렌드 검정 (간이: 사전 기간 차이가 유의하지 않으면 병렬)
        pre_diff = abs(pre_treated.mean() - pre_control.mean())
        pre_se = float(np.sqrt(pre_treated.var() / len(pre_treated) + pre_control.var() / len(pre_control) + 1e-10))
        parallel_z = pre_diff / (pre_se + 1e-10)
        from scipy import stats as scipy_stats
        parallel_pvalue = float(2 * (1 - scipy_stats.norm.cdf(parallel_z)))
        parallel_holds = parallel_pvalue > 0.05

        interp = (
            f"DiD ATE = {ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]. "
            f"병렬 트렌드 {'✅ 성립' if parallel_holds else '⚠️ 미성립'} (p={parallel_pvalue:.3f})."
        )

        self.logger.info("   DiD: ATE=%.4f, 병렬트렌드=%s", ate, parallel_holds)

        return DiDResult(
            ate=ate, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
            parallel_trend_pvalue=parallel_pvalue,
            parallel_trend_holds=parallel_holds,
            n_treated=n_t, n_control=n_c, interpretation=interp,
        )

    def _simulate_did(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariates: List[str],
    ) -> DiDResult:
        """이진 처치에서 간이 DiD 시뮬레이션.

        시간 변수 없이, 처치/통제 그룹 간 차이를 DiD처럼 추정합니다.
        """
        treated = df[df[treatment_col] == 1][outcome_col]
        control = df[df[treatment_col] == 0][outcome_col]

        if len(treated) == 0 or len(control) == 0:
            return DiDResult(interpretation="처치/통제 그룹 크기 부족")

        ate = float(treated.mean() - control.mean())
        se = float(np.sqrt(treated.var() / len(treated) + control.var() / len(control) + 1e-10))

        return DiDResult(
            ate=ate, se=se,
            ci_lower=ate - 1.96 * se, ci_upper=ate + 1.96 * se,
            parallel_trend_pvalue=1.0,  # 시간 없으므로 검정 불가
            parallel_trend_holds=True,
            n_treated=len(treated), n_control=len(control),
            interpretation=f"간이 DiD: ATE={ate:.4f} (시간 변수 없이 그룹 비교)",
        )

    # ── RDD (Regression Discontinuity Design) ──

    def _estimate_rdd(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        running_col: str,
        cutoff: float,
        bandwidth: Optional[float] = None,
    ) -> RDDResult:
        """Sharp RDD 추정 (국소 선형 회귀)."""
        from sklearn.linear_model import LinearRegression

        running = df[running_col].values
        outcome = df[outcome_col].values

        # 대역폭 자동 결정 (IK rule of thumb)
        if bandwidth is None:
            bandwidth = float(1.06 * np.std(running) * len(running) ** (-0.2))

        # 대역폭 내 관측치 선택
        mask = np.abs(running - cutoff) <= bandwidth
        local_df = df[mask].copy()

        if len(local_df) < 10:
            return RDDResult(
                cutoff=cutoff, bandwidth=bandwidth,
                interpretation="대역폭 내 관측치 부족 (n < 10)",
            )

        local_running = local_df[running_col].values - cutoff  # 중심화
        local_outcome = local_df[outcome_col].values
        local_treatment = (local_running >= 0).astype(float)

        # 국소 선형 회귀: Y ~ D + (X-c) + D*(X-c)
        X_rdd = np.column_stack([
            local_treatment,
            local_running,
            local_treatment * local_running,
        ])
        reg = LinearRegression().fit(X_rdd, local_outcome)
        ate = float(reg.coef_[0])

        # 표준 오차
        residuals = local_outcome - reg.predict(X_rdd)
        se = float(np.std(residuals) / np.sqrt(len(local_outcome)))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        n_left = int(np.sum(local_treatment == 0))
        n_right = int(np.sum(local_treatment == 1))

        interp = (
            f"RDD ATE = {ate:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] "
            f"(절단={cutoff:.2f}, 대역폭={bandwidth:.2f}, "
            f"좌={n_left}, 우={n_right})."
        )

        self.logger.info("   RDD: ATE=%.4f, cutoff=%.2f, bw=%.2f", ate, cutoff, bandwidth)

        return RDDResult(
            ate=ate, se=se, ci_lower=ci_lower, ci_upper=ci_upper,
            cutoff=cutoff, bandwidth=bandwidth,
            n_left=n_left, n_right=n_right, interpretation=interp,
        )

    def _find_rdd_candidate(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        features: List[str],
    ) -> Optional[Tuple[str, float]]:
        """RDD 적용 가능한 변수와 절단점을 자동 탐색합니다.

        처치 확률이 급변하는 지점이 있으면 RDD 후보로 판단.
        """
        if treatment_col not in df.columns or df[treatment_col].nunique() != 2:
            return None

        for col in features:
            if col not in df.columns or df[col].nunique() < 10:
                continue
            try:
                # 10분위별 처치 비율 계산
                quantiles = pd.qcut(df[col], 10, duplicates='drop')
                group_means = df.groupby(quantiles, observed=True)[treatment_col].mean()

                # 인접 분위 간 최대 차이 찾기
                diffs = group_means.diff().abs()
                max_diff = diffs.max()

                if max_diff > 0.3:  # 처치율 30%p 이상 급변
                    max_idx = diffs.idxmax()
                    # 절단점 = 해당 분위 경계
                    cutoff = float(df[col][quantiles == max_idx].mean())
                    return (col, cutoff)
            except Exception:
                continue
        return None

    # ── 직렬화 ──

    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """결과를 JSON 직렬화 가능 딕셔너리로 변환."""
        serialized = {}
        for method, result in results.items():
            if hasattr(result, '__dict__'):
                serialized[method] = {
                    k: v for k, v in result.__dict__.items()
                    if not k.startswith('_')
                }
            else:
                serialized[method] = result
        return serialized
