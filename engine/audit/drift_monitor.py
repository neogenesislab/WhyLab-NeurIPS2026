# -*- coding: utf-8 -*-
"""인과 드리프트 모니터 — 정보이론 기반 DI(Drift Index) 계산.

에이전트 결정의 인과적 영향력이 시간에 따라 변화하는지 감시합니다.
DI 임계값 초과 시 DampingController에 경고를 보내
보수적 업데이트 모드로 전환합니다.

수학적 정당성 (R1 Reviewer 방어):
- 각 성분(verdict, ATE, confidence)에 고정 가중치를 부여하는
  heuristic 대신, 정보이론 기반 동적 가중치를 사용합니다.
- 가중치 w_i ∝ 1/H(S_i): 각 성분의 엔트로피가 낮을수록(정보량 높을수록)
  해당 성분의 드리프트가 더 의미 있으므로 높은 가중치를 부여합니다.
  이는 Maximum Entropy Principle의 역(inverse)을 적용한 것으로,
  '기대 밖의 변화'에 더 큰 경보를 울리게 합니다.
- 참조: Jaynes (1957), "Information Theory and Statistical Mechanics"
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple

from engine.audit.schemas import AuditResult, AuditVerdict

logger = logging.getLogger("whylab.audit.drift_monitor")

# 엔트로피 계산 시 log(0) 방지 상수
_EPS = 1e-10


def _shannon_entropy(distribution: List[float]) -> float:
    """이산 분포의 Shannon 엔트로피 H(X) = -Σ p_i log₂ p_i.

    정규화된 확률 분포를 입력으로 받습니다.
    """
    total = sum(distribution)
    if total < _EPS:
        return 0.0
    probs = [p / total for p in distribution if p > 0]
    return -sum(p * math.log2(p + _EPS) for p in probs)


def _empirical_verdict_entropy(verdicts: List[AuditVerdict]) -> float:
    """판결 시퀀스의 경험적 엔트로피."""
    if not verdicts:
        return 0.0
    counts = {}
    for v in verdicts:
        counts[v] = counts.get(v, 0) + 1
    return _shannon_entropy(list(counts.values()))


def _wasserstein_1d(p: List[float], q: List[float]) -> float:
    """1차원 Wasserstein 거리 (Earth Mover's Distance).

    두 경험적 분포 사이의 최소 이동 비용.
    Sorted CDF 차이의 적분으로 O(n log n) 계산.
    """
    if not p or not q:
        return 0.0
    ps, qs = sorted(p), sorted(q)
    n = max(len(ps), len(qs))

    # CDF 보간
    def _quantile(arr: List[float], t: float) -> float:
        idx = t * (len(arr) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(arr) - 1)
        frac = idx - lo
        return arr[lo] * (1 - frac) + arr[hi] * frac

    steps = min(n, 50)
    distance = sum(
        abs(_quantile(ps, i / steps) - _quantile(qs, i / steps))
        for i in range(steps + 1)
    ) / (steps + 1)

    return distance


class CausalDriftMonitor:
    """정보이론 기반 인과적 드리프트 지수(DI) 모니터링.

    수학적 정당성:
        DI = Σ_i w_i · d_i  where w_i ∝ 1/H(S_i)

    성분:
        d₁: 판결 변동률 (verdict transition rate)
        d₂: ATE Wasserstein 거리 (전반부 vs 후반부 분포)
        d₃: 신뢰도 분포 이동 (Wasserstein distance)

    가중치: 각 성분의 엔트로피 역수에 비례하여 동적 결정.
    """

    def __init__(
        self,
        drift_threshold: float = 0.3,
        window_size: int = 10,
        break_sensitivity: float = 2.0,
    ) -> None:
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.break_sensitivity = break_sensitivity
        self._audit_history: List[AuditResult] = []

    def record(self, result: AuditResult) -> float:
        """감사 결과를 기록하고 현재 DI를 반환합니다."""
        self._audit_history.append(result)
        di = self.compute_drift_index()
        if di > self.drift_threshold:
            logger.warning(
                "🚨 Drift Alert: DI=%.3f > threshold=%.3f (history=%d)",
                di, self.drift_threshold, len(self._audit_history),
            )
        return di

    def compute_drift_index(
        self,
        recent_audits: Optional[List[AuditResult]] = None,
        window_days: int = 30,
    ) -> float:
        """정보이론 기반 드리프트 지수를 계산합니다.

        DI = Σ_i w_i · d_i  where w_i = (1/H_i) / Σ_j(1/H_j)

        각 성분:
        - d₁: 판결 전환율 (transition rate)
        - d₂: ATE Wasserstein 거리 (전반부 vs 후반부)
        - d₃: 신뢰도 Wasserstein 거리 (전반부 vs 후반부)

        Returns:
            드리프트 지수 (0~1, 높을수록 불안정)
        """
        audits = recent_audits or self._audit_history
        if len(audits) < 3:
            return 0.0

        recent = audits[-self.window_size:]
        mid = len(recent) // 2

        # ── 성분 계산 ──

        # d₁: 판결 변동률
        verdict_changes = sum(
            1 for i in range(1, len(recent))
            if recent[i].verdict != recent[i - 1].verdict
        )
        d1 = verdict_changes / max(len(recent) - 1, 1)

        # d₂: ATE Wasserstein 거리 (전반부 vs 후반부 분포)
        ates_first = [r.ate for r in recent[:mid]]
        ates_second = [r.ate for r in recent[mid:]]
        if ates_first and ates_second:
            raw_w = _wasserstein_1d(ates_first, ates_second)
            ate_scale = max(
                abs(statistics.mean(ates_first + ates_second)), _EPS
            )
            d2 = min(raw_w / ate_scale, 1.0)
        else:
            d2 = 0.0

        # d₃: 신뢰도 Wasserstein 거리
        conf_first = [r.confidence for r in recent[:mid]]
        conf_second = [r.confidence for r in recent[mid:]]
        if conf_first and conf_second:
            d3 = min(_wasserstein_1d(conf_first, conf_second), 1.0)
        else:
            d3 = 0.0

        # ── 정보이론 기반 동적 가중치 계산 ──
        # H_i = 각 성분의 히스토리 엔트로피. 낮을수록 → 더 예측 가능 →
        # 드리프트 시 더 놀라움(surprised) → 높은 가중치
        h1 = _empirical_verdict_entropy([r.verdict for r in recent])
        h2 = _signal_entropy([r.ate for r in recent])
        h3 = _signal_entropy([r.confidence for r in recent])

        weights = _entropy_inverse_weights([h1, h2, h3])

        di = weights[0] * d1 + weights[1] * d2 + weights[2] * d3

        return round(min(di, 1.0), 4)

    def compute_drift_components(self) -> Dict[str, float]:
        """각 DI 성분과 가중치를 분해하여 반환합니다 (디버깅/논문용)."""
        audits = self._audit_history
        if len(audits) < 3:
            return {"d1": 0, "d2": 0, "d3": 0, "w1": 0, "w2": 0, "w3": 0}

        recent = audits[-self.window_size:]
        mid = len(recent) // 2

        d1 = sum(
            1 for i in range(1, len(recent))
            if recent[i].verdict != recent[i - 1].verdict
        ) / max(len(recent) - 1, 1)

        ates_f = [r.ate for r in recent[:mid]]
        ates_s = [r.ate for r in recent[mid:]]
        if ates_f and ates_s:
            raw_w = _wasserstein_1d(ates_f, ates_s)
            d2 = min(raw_w / max(abs(statistics.mean(ates_f + ates_s)), _EPS), 1.0)
        else:
            d2 = 0.0

        conf_f = [r.confidence for r in recent[:mid]]
        conf_s = [r.confidence for r in recent[mid:]]
        d3 = min(_wasserstein_1d(conf_f, conf_s), 1.0) if conf_f and conf_s else 0.0

        h1 = _empirical_verdict_entropy([r.verdict for r in recent])
        h2 = _signal_entropy([r.ate for r in recent])
        h3 = _signal_entropy([r.confidence for r in recent])
        weights = _entropy_inverse_weights([h1, h2, h3])

        return {
            "d1_verdict": round(d1, 4),
            "d2_ate_wasserstein": round(d2, 4),
            "d3_conf_wasserstein": round(d3, 4),
            "w1": round(weights[0], 4),
            "w2": round(weights[1], 4),
            "w3": round(weights[2], 4),
            "h1_verdict_entropy": round(h1, 4),
            "h2_ate_entropy": round(h2, 4),
            "h3_conf_entropy": round(h3, 4),
        }

    def detect_structural_break(self) -> bool:
        """환경의 구조적 변화를 감지합니다.

        최근 ATE가 이전 평균에서 break_sensitivity × σ 이상
        벗어나면 구조적 변화로 판단합니다.
        """
        if len(self._audit_history) < 6:
            return False

        mid = len(self._audit_history) // 2
        old_ates = [r.ate for r in self._audit_history[:mid]]
        new_ates = [r.ate for r in self._audit_history[mid:]]

        if not old_ates or not new_ates:
            return False

        old_mean = statistics.mean(old_ates)
        old_std = statistics.stdev(old_ates) if len(old_ates) > 1 else 1.0
        new_mean = statistics.mean(new_ates)

        deviation = abs(new_mean - old_mean) / max(old_std, _EPS)
        is_break = deviation > self.break_sensitivity

        if is_break:
            logger.warning(
                "🔴 Structural Break 감지: old_mean=%.4f, new_mean=%.4f, "
                "deviation=%.2fσ",
                old_mean, new_mean, deviation,
            )

        return is_break

    def get_status(self) -> Dict[str, Any]:
        """현재 모니터링 상태를 반환합니다."""
        di = self.compute_drift_index()
        return {
            "drift_index": di,
            "is_drifting": di > self.drift_threshold,
            "structural_break": self.detect_structural_break(),
            "history_size": len(self._audit_history),
            "recent_verdicts": [
                r.verdict.value for r in self._audit_history[-5:]
            ],
            "components": self.compute_drift_components(),
        }


# ── 모듈 수준 유틸 ──

def _signal_entropy(values: List[float], n_bins: int = 5) -> float:
    """연속 신호의 경험적 엔트로피 (히스토그램 기반).

    연속 값을 n_bins개의 등간격 구간으로 이산화한 후
    Shannon 엔트로피를 계산합니다.
    """
    if len(values) < 2:
        return 0.0
    vmin, vmax = min(values), max(values)
    if abs(vmax - vmin) < _EPS:
        return 0.0  # 모든 값이 동일 → 엔트로피 0
    bin_width = (vmax - vmin) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - vmin) / bin_width), n_bins - 1)
        counts[idx] += 1
    return _shannon_entropy(counts)


def _entropy_inverse_weights(entropies: List[float]) -> List[float]:
    """엔트로피 역수 기반 가중치 계산.

    w_i = (1 / (H_i + ε)) / Σ_j (1 / (H_j + ε))

    논문 정당성:
    - H_i가 낮을수록(예측 가능할수록) 해당 성분에서
      드리프트 발생 시 '놀라움(surprisal)'이 크므로
      더 높은 가중치를 부여합니다.
    - 모든 성분이 균일하면 (1/3, 1/3, 1/3)이 되어
      기존 균등 가중치의 상위 호환입니다.
    """
    inv = [1.0 / (h + _EPS) for h in entropies]
    total = sum(inv)
    if total < _EPS:
        n = len(entropies)
        return [1.0 / n] * n
    return [w / total for w in inv]
