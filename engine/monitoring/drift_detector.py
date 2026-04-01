# -*- coding: utf-8 -*-
"""인과 드리프트 탐지기 (Causal Drift Detector).

ATE/CATE 분포의 시간적 변화를 감지합니다.
KL-Divergence, PSI(Population Stability Index), 부호 반전 등으로 드리프트를 판별합니다.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("whylab.monitoring.drift")


@dataclass
class DriftResult:
    """드리프트 탐지 결과.

    Attributes:
        drifted: 드리프트 발생 여부.
        metric: 사용된 메트릭 이름 (예: "kl_divergence").
        score: 드리프트 점수 (높을수록 심각).
        threshold: 판단 기준 임계값.
        details: 추가 상세 정보.
    """

    drifted: bool = False
    metric: str = ""
    score: float = 0.0
    threshold: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class DriftDetector:
    """인과 효과의 시계열 드리프트를 탐지합니다.

    사용법:
        detector = DriftDetector()
        detector.add_snapshot(ate=0.05, cate_distribution=[...])
        detector.add_snapshot(ate=0.12, cate_distribution=[...])
        result = detector.check_drift()
        if result.drifted:
            print(f"⚠️ 드리프트 감지! 점수: {result.score}")
    """

    def __init__(
        self,
        kl_threshold: float = 0.1,
        ate_change_threshold: float = 0.5,
        min_snapshots: int = 2,
        window_size: int = 10,
    ):
        """
        Args:
            kl_threshold: KL-Divergence 드리프트 임계값.
            ate_change_threshold: ATE 변화율 임계값 (예: 0.5 = 50%).
            min_snapshots: 드리프트 판단에 필요한 최소 스냅샷 수.
            window_size: 비교 윈도우 크기.
        """
        self.kl_threshold = kl_threshold
        self.ate_change_threshold = ate_change_threshold
        self.min_snapshots = min_snapshots
        self.window_size = window_size

        self._snapshots: List[Dict[str, Any]] = []

    def add_snapshot(
        self,
        ate: float,
        cate_distribution: Optional[List[float]] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """새로운 분석 스냅샷을 추가합니다."""
        snapshot = {
            "ate": ate,
            "cate_distribution": cate_distribution or [],
            "timestamp": timestamp,
            "metadata": metadata or {},
        }
        self._snapshots.append(snapshot)
        logger.info(
            "스냅샷 추가: ATE=%.4f (총 %d개)", ate, len(self._snapshots)
        )

    def check_drift(self) -> DriftResult:
        """현재 상태에서 드리프트를 판단합니다.

        Returns:
            DriftResult: 종합 드리프트 결과.
        """
        if len(self._snapshots) < self.min_snapshots:
            return DriftResult(
                drifted=False,
                metric="insufficient_data",
                details={"snapshots": len(self._snapshots)},
            )

        results = []

        # 1. ATE 변화율 체크
        ate_result = self._check_ate_drift()
        results.append(ate_result)

        # 2. CATE 분포 KL-Divergence 체크
        kl_result = self._check_kl_divergence()
        if kl_result:
            results.append(kl_result)

        # 3. ATE 부호 반전 체크
        sign_result = self._check_sign_flip()
        results.append(sign_result)

        # 종합 판단: 하나라도 드리프트면 전체 드리프트
        drifted_results = [r for r in results if r.drifted]
        if drifted_results:
            worst = max(drifted_results, key=lambda r: r.score)
            worst.details["all_checks"] = [
                {"metric": r.metric, "drifted": r.drifted, "score": r.score}
                for r in results
            ]
            return worst

        return DriftResult(
            drifted=False,
            metric="all_checks_passed",
            details={
                "checks": [
                    {"metric": r.metric, "score": r.score} for r in results
                ]
            },
        )

    def _check_ate_drift(self) -> DriftResult:
        """ATE 변화율이 임계값을 초과하는지 확인합니다."""
        recent = self._snapshots[-1]["ate"]
        baseline = np.mean([s["ate"] for s in self._snapshots[:-1]])

        if abs(baseline) < 1e-10:
            change_rate = abs(recent) * 100
        else:
            change_rate = abs(recent - baseline) / abs(baseline)

        drifted = change_rate > self.ate_change_threshold
        return DriftResult(
            drifted=drifted,
            metric="ate_change_rate",
            score=change_rate,
            threshold=self.ate_change_threshold,
            details={"recent_ate": recent, "baseline_ate": baseline},
        )

    def _check_kl_divergence(self) -> Optional[DriftResult]:
        """CATE 분포의 KL-Divergence를 계산합니다."""
        recent_dist = self._snapshots[-1].get("cate_distribution", [])
        prev_dist = self._snapshots[-2].get("cate_distribution", [])

        if not recent_dist or not prev_dist:
            return None

        # 히스토그램으로 변환 후 KL-Div 계산
        try:
            bins = 20
            all_values = list(recent_dist) + list(prev_dist)
            bin_edges = np.linspace(min(all_values), max(all_values), bins + 1)

            p, _ = np.histogram(recent_dist, bins=bin_edges, density=True)
            q, _ = np.histogram(prev_dist, bins=bin_edges, density=True)

            # 스무딩 (0 방지)
            epsilon = 1e-10
            p = p + epsilon
            q = q + epsilon
            p = p / p.sum()
            q = q / q.sum()

            kl_div = float(np.sum(p * np.log(p / q)))

            return DriftResult(
                drifted=kl_div > self.kl_threshold,
                metric="kl_divergence",
                score=kl_div,
                threshold=self.kl_threshold,
            )
        except Exception as e:
            logger.warning("KL-Divergence 계산 실패: %s", e)
            return None

    def _check_sign_flip(self) -> DriftResult:
        """ATE 부호가 반전되었는지 확인합니다."""
        recent_sign = np.sign(self._snapshots[-1]["ate"])
        prev_signs = [np.sign(s["ate"]) for s in self._snapshots[:-1]]
        majority_sign = np.sign(np.mean(prev_signs))

        flipped = recent_sign != majority_sign and recent_sign != 0
        return DriftResult(
            drifted=flipped,
            metric="sign_flip",
            score=1.0 if flipped else 0.0,
            threshold=0.5,
            details={"recent_sign": int(recent_sign), "majority_sign": int(majority_sign)},
        )

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def reset(self) -> None:
        """스냅샷 히스토리를 초기화합니다."""
        self._snapshots.clear()
