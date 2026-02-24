# -*- coding: utf-8 -*-
"""DebateCell — Multi-Agent Debate 파이프라인 셀.

Orchestrator 파이프라인의 최종 단계로,
Advocate/Critic/Judge 3-에이전트 Debate를 실행합니다.

Phase 9: LLM 자연어 토론 레이어 통합.
규칙 기반 증거 수집 → LLM 자연어 토론 생성 → 가중 스코어링 판결.
LLM 비활성/장애 시 기존 규칙 기반으로 자동 Fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from engine.cells.base_cell import BaseCell
from engine.agents.debate import AdvocateAgent, CriticAgent, JudgeAgent, Verdict
from engine.agents.llm_adapter import LLMDebateAdapter
from engine.config import WhyLabConfig
from engine.tracing import trace_cell

logger = logging.getLogger(__name__)


class DebateCell(BaseCell):
    """Multi-Agent Debate 셀.

    파이프라인 결과를 3-에이전트 Debate로 판결합니다.
    LLM이 활성화되면 자연어 토론문을 추가 생성합니다.
    """

    def __init__(self, config: WhyLabConfig) -> None:
        super().__init__(name="debate_cell", config=config)

    @trace_cell
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

        # LLM 어댑터 초기화
        llm_adapter = None
        if cfg.use_llm:
            llm_adapter = LLMDebateAdapter(model_name=cfg.llm_model)
            if llm_adapter.is_llm_active:
                self.logger.info("🤖 LLM 토론 모드 활성화 (모델: %s)", cfg.llm_model)
            else:
                self.logger.info("📝 LLM 미설정 → 규칙 기반 Fallback 토론 모드")

        self.logger.info("🎙️ Multi-Agent Debate 시작 (최대 %d 라운드)", cfg.max_rounds)

        verdict = None
        all_pro = []
        all_con = []

        for round_num in range(1, cfg.max_rounds + 1):
            self.logger.info("── Round %d ──", round_num)

            # 1단계: 규칙 기반 증거 수집
            pro = advocate.gather_evidence(inputs)
            con = critic.challenge(inputs)

            # 누적
            all_pro.extend(pro)
            all_con.extend(con)

            # 2단계: 가중 스코어링 판결
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

        # 3단계: LLM 자연어 토론 생성 (최종 판결 후)
        llm_debate = {}
        if llm_adapter is not None:
            # ATE 값 추출 (float 또는 dict)
            ate_raw = inputs.get("ate")
            if isinstance(ate_raw, dict):
                ate_value = ate_raw.get("value", "N/A")
            elif isinstance(ate_raw, (int, float)):
                ate_value = f"{ate_raw:.4f}"
            else:
                ate_value = "N/A"

            llm_context = {
                "treatment_col": inputs.get("treatment_col", "Treatment"),
                "outcome_col": inputs.get("outcome_col", "Outcome"),
                "ate_value": ate_value,
            }

            self.logger.info("🤖 LLM 토론문 생성 중...")

            advocate_arg = llm_adapter.generate_advocate_argument(
                all_pro, llm_context,
            )
            critic_arg = llm_adapter.generate_critic_argument(
                all_con, llm_context,
            )
            verdict_arg = llm_adapter.generate_verdict(
                advocate_arg, critic_arg,
                {
                    "verdict": verdict.verdict,
                    "confidence": verdict.confidence,
                    "pro_score": verdict.pro_score,
                    "con_score": verdict.con_score,
                    "recommendation": verdict.recommendation,
                },
                llm_context,
            )

            llm_debate = {
                "advocate_argument": advocate_arg,
                "critic_argument": critic_arg,
                "judge_verdict": verdict_arg,
                **llm_adapter.get_debate_summary(),
            }

            self.logger.info(
                "✅ LLM 토론 완료 (활성: %s, 응답 %d건)",
                llm_adapter.is_llm_active,
                len(llm_adapter.responses),
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
                 "strength": e.strength, "source": e.source,
                 "business_impact": e.business_impact}
                for e in verdict.pro_evidence
            ],
            "con_evidence": [
                {"claim": e.claim, "type": e.evidence_type,
                 "strength": e.strength, "source": e.source,
                 "business_impact": e.business_impact}
                for e in verdict.con_evidence
            ],
            # LLM 토론 결과 포함
            "llm_debate": llm_debate,
        }

        self.logger.info(
            "📋 Debate 완료: verdict=%s, rounds=%d, "
            "pro_evidence=%d, con_evidence=%d, llm=%s",
            verdict.verdict, verdict.rounds,
            len(verdict.pro_evidence), len(verdict.con_evidence),
            "active" if llm_debate.get("llm_active") else "fallback",
        )

        return {
            **inputs,
            "debate_verdict": verdict,
            "debate_summary": debate_summary,
        }
