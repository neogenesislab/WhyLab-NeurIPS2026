# -*- coding: utf-8 -*-
"""DebateCell — Multi-Agent Debate 파이프라인 셀.

Orchestrator 파이프라인의 최종 단계로,
Advocate/Critic/Judge 3-에이전트 Debate를 실행합니다.

UNCERTAIN 판결 시 최대 MAX_ROUNDS까지 반복하며,
각 라운드에서 증거를 누적합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from engine.cells.base_cell import BaseCell
from engine.agents.debate import AdvocateAgent, CriticAgent, JudgeAgent, Verdict
from engine.config import WhyLabConfig

logger = logging.getLogger(__name__)


class DebateCell(BaseCell):
    """Multi-Agent Debate 셀.

    파이프라인 결과를 3-에이전트 Debate로 판결합니다.
    UNCERTAIN 판결 시 추가 라운드를 진행합니다.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="debate_cell", config=config)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Debate 실행.

        Args:
            inputs: 전체 파이프라인 결과.

        Returns:
            판결 결과 포함 dict.
        """
        cfg = self.config.debate
        advocate = AdvocateAgent()
        critic = CriticAgent()
        judge = JudgeAgent(weights=cfg.evidence_weights)

        self.logger.info("🎙️ Multi-Agent Debate 시작 (최대 %d 라운드)", cfg.max_rounds)

        verdict = None
        all_pro = []
        all_con = []

        for round_num in range(1, cfg.max_rounds + 1):
            self.logger.info("── Round %d ──", round_num)

            # 증거 수집
            pro = advocate.gather_evidence(inputs)
            con = critic.challenge(inputs)

            # 누적
            all_pro.extend(pro)
            all_con.extend(con)

            # 판결
            verdict = judge.deliberate(
                all_pro, all_con, threshold=cfg.confidence_threshold,
            )
            verdict.rounds = round_num

            if verdict.verdict != "UNCERTAIN":
                self.logger.info(
                    "🏛️ 판결 확정 (Round %d): %s (확신도=%.2f)",
                    round_num, verdict.verdict, verdict.confidence,
                )
                break

            self.logger.info(
                "⚠️ Round %d: UNCERTAIN (확신도=%.2f) → 추가 라운드",
                round_num, verdict.confidence,
            )

        # 최종 결과
        debate_summary = {
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
            "pro_score": verdict.pro_score,
            "con_score": verdict.con_score,
            "rounds": verdict.rounds,
            "recommendation": verdict.recommendation,
            "pro_evidence": [
                {"claim": e.claim, "type": e.evidence_type,
                 "strength": e.strength, "source": e.source}
                for e in verdict.pro_evidence
            ],
            "con_evidence": [
                {"claim": e.claim, "type": e.evidence_type,
                 "strength": e.strength, "source": e.source}
                for e in verdict.con_evidence
            ],
        }

        self.logger.info(
            "📋 Debate 완료: verdict=%s, rounds=%d, "
            "pro_evidence=%d, con_evidence=%d",
            verdict.verdict, verdict.rounds,
            len(verdict.pro_evidence), len(verdict.con_evidence),
        )

        return {
            **inputs,
            "debate_verdict": verdict,
            "debate_summary": debate_summary,
        }
