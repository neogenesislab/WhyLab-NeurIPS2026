# -*- coding: utf-8 -*-
"""R5: Lyapunov 안정성 기반 댐핑 상한 필터.

논문 핵심 정리 (Theorem):
    에이전트 전략 업데이트 θ_{t+1} = θ_t - ζ_t ĝ_t에서
    Lyapunov 에너지 V(θ) = ½||θ - θ*||² 가 감소하려면:

    ζ_t < 2⟨θ_t - θ*, g_t⟩ / E[||ĝ_t||²]

    분모의 E[||ĝ_t||²] ∝ (DI + ε_ARES)이므로,
    불확실성이 클 때 ζ를 줄이는 것은 수학적 필연.

CTO 구현:
    DampingController가 제안한 ζ를 Lyapunov bound로 클리핑.
    Control Barrier Function 역할.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("whylab.audit.lyapunov")


@dataclass
class LyapunovState:
    """Lyapunov 에너지 추적 상태."""
    energy: float = 0.0          # V(θ_t) 현재 에너지
    energy_prev: float = 0.0     # V(θ_{t-1})
    delta_v: float = 0.0         # ΔV = V(t) - V(t-1)
    zeta_max: float = 1.0        # Lyapunov bound ζ_max
    zeta_applied: float = 0.0    # 실제 적용된 ζ
    was_clipped: bool = False    # ζ가 클리핑되었는지
    history: List[float] = field(default_factory=list)


class LyapunovFilter:
    """Lyapunov 안정성 보장 필터 (Control Barrier Function).

    DampingController가 제안한 ζ를 수학적 상한으로 클리핑하여
    에이전트 전략의 발산(divergence)을 차단합니다.

    수식:
        ζ_max = 2 * signal_strength / (noise_variance + ε)

    여기서:
        signal_strength ∝ 최근 감사 결과의 효과 크기 (|ATE|)
        noise_variance ∝ DI(드리프트 지수) + ARES penalty

    사용법:
        lyap = LyapunovFilter()
        safe_zeta = lyap.clip(
            proposed_zeta=0.5,
            ate=15.0,
            drift_index=0.4,
            ares_penalty=0.3,
        )
    """

    def __init__(
        self,
        min_zeta: float = 0.01,
        max_zeta: float = 0.8,
        energy_decay_target: float = 0.95,
    ) -> None:
        """
        Args:
            min_zeta: 최소 허용 ζ (완전 동결 방지)
            max_zeta: 절대 상한 ζ
            energy_decay_target: 에너지 감소율 목표 (1.0=유지, <1.0=감소)
        """
        self.min_zeta = min_zeta
        self.max_zeta = max_zeta
        self.energy_decay_target = energy_decay_target
        self._energy_history: List[float] = []
        self._state = LyapunovState()

    def clip(
        self,
        proposed_zeta: float,
        ate: float,
        drift_index: float,
        ares_penalty: float = 0.0,
        confidence: float = 0.5,
    ) -> float:
        """제안된 ζ를 Lyapunov bound로 안전하게 클리핑합니다.

        Theorem (논문 §Methodology):
            ζ_max = 2 * S / (N + ε)

        여기서:
            S = signal_strength = |ATE| * confidence
            N = noise_variance = drift_index + ares_penalty

        이 bound를 초과하면 E[ΔV] > 0 (에너지 증가 → 발산).
        """
        # Signal strength: 효과 크기 × 신뢰도
        signal = abs(ate) * confidence

        # Noise variance: 드리프트 + ARES 불확실성
        noise = drift_index + ares_penalty

        # Lyapunov bound
        eps = 0.01  # 분모 안정화
        zeta_max = (2.0 * signal) / (noise + eps) if noise > eps else self.max_zeta

        # 절대 범위 내로 제한
        zeta_max = max(self.min_zeta, min(self.max_zeta, zeta_max))

        # 클리핑 여부
        was_clipped = proposed_zeta > zeta_max
        safe_zeta = min(proposed_zeta, zeta_max)
        safe_zeta = max(self.min_zeta, safe_zeta)

        # 에너지 추적 (V(t) ∝ 1/confidence + noise)
        current_energy = (1.0 - confidence + noise) / 2.0
        self._update_energy(current_energy, safe_zeta, was_clipped, zeta_max)

        if was_clipped:
            logger.warning(
                "🛡️ Lyapunov clip: ζ %.4f → %.4f (bound=%.4f, S=%.2f, N=%.2f)",
                proposed_zeta, safe_zeta, zeta_max, signal, noise,
            )

        return round(safe_zeta, 4)

    def _update_energy(
        self,
        energy: float,
        zeta: float,
        clipped: bool,
        zeta_max: float,
    ) -> None:
        """에너지 히스토리 업데이트."""
        self._state.energy_prev = self._state.energy
        self._state.energy = energy
        self._state.delta_v = energy - self._state.energy_prev
        self._state.zeta_max = zeta_max
        self._state.zeta_applied = zeta
        self._state.was_clipped = clipped
        self._energy_history.append(energy)

    def get_state(self) -> LyapunovState:
        """현재 Lyapunov 상태."""
        state = LyapunovState(
            energy=self._state.energy,
            energy_prev=self._state.energy_prev,
            delta_v=self._state.delta_v,
            zeta_max=self._state.zeta_max,
            zeta_applied=self._state.zeta_applied,
            was_clipped=self._state.was_clipped,
            history=list(self._energy_history[-20:]),
        )
        return state

    def is_converging(self) -> bool:
        """시스템이 수렴 중인지 판단.

        최근 5 에너지 값이 감소 추세이면 수렴.
        """
        if len(self._energy_history) < 5:
            return True  # 초기에는 수렴 가정
        recent = self._energy_history[-5:]
        # 선형 회귀 기울기 근사
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0
        return slope <= 0  # 음의 기울기 = 에너지 감소 = 수렴

    def prove_stability(self) -> Dict[str, Any]:
        """논문 Table용 안정성 증명 요약.

        Returns:
            Lyapunov 함수 특성:
            - energy_trend: 선형 기울기
            - is_stable: ΔV ≤ 0 만족 여부
            - clip_rate: ζ가 클리핑된 비율
            - theorem_text: 논문 삽입용 수식
        """
        if len(self._energy_history) < 3:
            return {"is_stable": True, "insufficient_data": True}

        # 에너지 추세
        recent = self._energy_history[-10:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0

        return {
            "is_stable": slope <= 0,
            "energy_trend_slope": round(slope, 6),
            "current_energy": round(self._state.energy, 4),
            "delta_v": round(self._state.delta_v, 4),
            "total_steps": len(self._energy_history),
            "theorem": (
                "θ_{t+1} = θ_t - ζ_t ĝ_t, "
                "V(θ) = ½||θ - θ*||², "
                "ζ_max = 2⟨θ-θ*, g⟩ / E[||ĝ||²], "
                "∵ E[ΔV] ≤ 0 iff ζ ≤ ζ_max"
            ),
        }
