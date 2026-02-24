# -*- coding: utf-8 -*-
"""Gemini LLM 어댑터 — 멀티 에이전트 토론 시스템용.

WhyLab의 규칙 기반 증거 수집 위에 LLM 자연어 추론 레이어를 추가합니다.
기존 AdvocateAgent/CriticAgent의 구조화된 Evidence를 LLM이 해석하여
더 풍부하고 맥락적인 토론을 생성합니다.

환경 변수:
    GEMINI_API_KEY 또는 GOOGLE_API_KEY: Gemini API 인증 키

사용법:
    from engine.agents.llm_adapter import LLMDebateAdapter
    adapter = LLMDebateAdapter()
    result = adapter.run_debate(pipeline_results)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

# LLM 모델 식별자 (환경 변수로 오버라이드 가능)
DEFAULT_MODEL = os.environ.get("WHYLAB_LLM_MODEL", "gemini-2.0-flash")

# 최대 토론 라운드
MAX_DEBATE_ROUNDS = int(os.environ.get("WHYLAB_DEBATE_ROUNDS", "2"))

# Tracing
from engine.tracing import trace_llm


@dataclass
class LLMResponse:
    """LLM 응답 구조체."""
    role: str  # "advocate" | "critic" | "judge"
    content: str  # 자연어 응답
    reasoning: str  # 추론 과정
    raw_evidence_count: int  # 기반 증거 수


# ──────────────────────────────────────────────
# Gemini 클라이언트 래퍼
# ──────────────────────────────────────────────

class GeminiClient:
    """Gemini API 최소 래퍼. 장애 시 graceful fallback."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        self._initialized = False

    def _ensure_init(self) -> bool:
        """지연 초기화. API 키 없으면 False 반환."""
        if self._initialized:
            return self.model is not None
        self._initialized = True

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("[LLM] GEMINI_API_KEY 미설정 → 규칙 기반 Fallback 모드")
            return False

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("[LLM] Gemini 초기화 성공 (모델: %s)", self.model_name)
            return True
        except Exception as e:
            logger.warning("[LLM] Gemini 초기화 실패: %s → Fallback 모드", e)
            return False

    @trace_llm
    def generate(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        """프롬프트를 Gemini에 전송하고 텍스트 응답을 반환합니다.

        Args:
            prompt: 입력 프롬프트.
            max_tokens: 최대 출력 토큰 수.

        Returns:
            응답 텍스트. 실패 시 None.
        """
        if not self._ensure_init():
            return None

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": 0.3,  # 분석적 태도 유지
                },
            )
            return response.text
        except Exception as e:
            logger.warning("[LLM] Gemini 호출 실패: %s", e)
            return None

    @property
    def is_available(self) -> bool:
        """LLM API 사용 가능 여부."""
        return self._ensure_init()


# ──────────────────────────────────────────────
# 프롬프트 템플릿
# ──────────────────────────────────────────────

ADVOCATE_PROMPT = """당신은 WhyLab 인과추론 시스템의 **Growth Hacker(옹호자)**입니다.
아래 증거를 검토하고, 인과 관계가 존재한다고 주장하는 논변을 작성하세요.

## 분석 맥락
- 처치(Treatment): {treatment}
- 결과(Outcome): {outcome}
- ATE: {ate}

## 수집된 긍정 증거
{evidence_summary}

## 지시사항
1. 증거를 종합하여 인과 관계를 옹호하는 **핵심 논변 3가지**를 작성하세요.
2. 각 논변에 대해 비즈니스 기회(Revenue, Growth, ROI)를 연결하세요.
3. 마지막에 **추천 액션**(배포/확장/타겟팅)을 한 줄로 제시하세요.
4. 한국어로 작성하되, 전문적이지만 명확하게 쓰세요.
5. 총 300자 이내로 간결하게 작성하세요."""

CRITIC_PROMPT = """당신은 WhyLab 인과추론 시스템의 **Risk Manager(비판자)**입니다.
아래 증거를 검토하고, 인과 관계의 약점과 리스크를 지적하세요.

## 분석 맥락
- 처치(Treatment): {treatment}
- 결과(Outcome): {outcome}
- ATE: {ate}

## 수집된 부정 증거 (공격 벡터)
{evidence_summary}

## 지시사항
1. 증거를 분석하여 인과 판단의 **핵심 리스크 3가지**를 지적하세요.
2. 각 리스크에 대해 비즈니스 위험(Loss, Churn, Compliance)을 연결하세요.
3. 마지막에 **요구 사항**(추가 검증/표본 확대/대안 방법)을 한 줄로 제시하세요.
4. 한국어로 작성하되, 전문적이지만 명확하게 쓰세요.
5. 총 300자 이내로 간결하게 작성하세요."""

JUDGE_PROMPT = """당신은 WhyLab 인과추론 시스템의 **Product Owner(판사)**입니다.
양측 에이전트의 공방을 검토하고 최종 판결을 내리세요.

## 분석 맥락
- 처치(Treatment): {treatment}
- 결과(Outcome): {outcome}
- ATE: {ate}
- 스코어: 옹호측 {pro_score:.2f} vs 비판측 {con_score:.2f}
- 확신도: {confidence:.1%}

## Growth Hacker의 주장
{advocate_argument}

## Risk Manager의 반론
{critic_argument}

## 판결 가이드
- 확신도 ≥ 70%: CAUSAL (인과 관계 인정)
- 확신도 ≤ 30%: NOT_CAUSAL (인과 관계 기각)
- 그 외: UNCERTAIN (추가 검증 필요)

## 지시사항
1. 양측 주장의 핵심을 한 문장씩 요약하세요.
2. 최종 판결(CAUSAL/NOT_CAUSAL/UNCERTAIN)과 근거를 밝히세요.
3. **비즈니스 액션 아이템**을 구체적으로 한 줄 작성하세요.
   - CAUSAL이면: 배포 전략 (Rollout %, 타겟 세그먼트)
   - NOT_CAUSAL이면: 리소스 회수 또는 대안 실험 제안
   - UNCERTAIN이면: A/B 테스트 설계 제안
4. 한국어로 작성하세요.
5. 총 400자 이내로 간결하게 작성하세요."""


# ──────────────────────────────────────────────
# LLM Debate 어댑터
# ──────────────────────────────────────────────

class LLMDebateAdapter:
    """규칙 기반 증거 + LLM 자연어 토론 하이브리드 시스템.

    기존 Advocate/Critic/Judge의 구조화된 증거 수집은 유지하면서,
    LLM이 증거를 해석하여 자연어 토론문을 생성합니다.
    LLM 장애 시 기존 규칙 기반으로 자동 Fallback합니다.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.client = GeminiClient(model_name)
        self.responses: List[LLMResponse] = []

    def format_evidence(self, evidence_list: list) -> str:
        """Evidence 리스트를 프롬프트용 텍스트로 변환."""
        lines = []
        for i, e in enumerate(evidence_list, 1):
            impact = f" → {e.business_impact}" if e.business_impact else ""
            lines.append(
                f"{i}. [{e.evidence_type}] {e.claim} "
                f"(강도: {e.strength:.2f}){impact}"
            )
        return "\n".join(lines) if lines else "(수집된 증거 없음)"

    def generate_advocate_argument(
        self,
        evidence: list,
        context: Dict[str, Any],
    ) -> str:
        """Growth Hacker의 옹호 논변을 LLM으로 생성합니다."""
        prompt = ADVOCATE_PROMPT.format(
            treatment=context.get("treatment_col", "T"),
            outcome=context.get("outcome_col", "Y"),
            ate=context.get("ate_value", "N/A"),
            evidence_summary=self.format_evidence(evidence),
        )

        response = self.client.generate(prompt)
        if response:
            self.responses.append(LLMResponse(
                role="advocate",
                content=response,
                reasoning="Gemini LLM 기반 논변 생성",
                raw_evidence_count=len(evidence),
            ))
            return response

        # Fallback: 규칙 기반 요약
        return self._fallback_advocate(evidence)

    def generate_critic_argument(
        self,
        evidence: list,
        context: Dict[str, Any],
    ) -> str:
        """Risk Manager의 비판 논변을 LLM으로 생성합니다."""
        prompt = CRITIC_PROMPT.format(
            treatment=context.get("treatment_col", "T"),
            outcome=context.get("outcome_col", "Y"),
            ate=context.get("ate_value", "N/A"),
            evidence_summary=self.format_evidence(evidence),
        )

        response = self.client.generate(prompt)
        if response:
            self.responses.append(LLMResponse(
                role="critic",
                content=response,
                reasoning="Gemini LLM 기반 반론 생성",
                raw_evidence_count=len(evidence),
            ))
            return response

        return self._fallback_critic(evidence)

    def generate_verdict(
        self,
        advocate_arg: str,
        critic_arg: str,
        verdict_data: dict,
        context: Dict[str, Any],
    ) -> str:
        """Product Owner의 최종 판결문을 LLM으로 생성합니다."""
        prompt = JUDGE_PROMPT.format(
            treatment=context.get("treatment_col", "T"),
            outcome=context.get("outcome_col", "Y"),
            ate=context.get("ate_value", "N/A"),
            pro_score=verdict_data.get("pro_score", 0),
            con_score=verdict_data.get("con_score", 0),
            confidence=verdict_data.get("confidence", 0),
            advocate_argument=advocate_arg,
            critic_argument=critic_arg,
        )

        response = self.client.generate(prompt, max_tokens=1024)
        if response:
            self.responses.append(LLMResponse(
                role="judge",
                content=response,
                reasoning="Gemini LLM 기반 최종 판결",
                raw_evidence_count=0,
            ))
            return response

        return self._fallback_verdict(verdict_data)

    # ── Fallback 메서드 ──

    def _fallback_advocate(self, evidence: list) -> str:
        """LLM 장애 시 규칙 기반 옹호 요약."""
        if not evidence:
            return "수집된 옹호 증거가 없습니다."
        top = sorted(evidence, key=lambda e: e.strength, reverse=True)[:3]
        lines = [f"📗 **Growth Hacker 핵심 주장:**"]
        for e in top:
            lines.append(f"  • {e.claim}")
            if e.business_impact:
                lines.append(f"    → {e.business_impact}")
        return "\n".join(lines)

    def _fallback_critic(self, evidence: list) -> str:
        """LLM 장애 시 규칙 기반 비판 요약."""
        if not evidence:
            return "수집된 비판 증거가 없습니다."
        top = sorted(evidence, key=lambda e: e.strength, reverse=True)[:3]
        lines = [f"📕 **Risk Manager 핵심 리스크:**"]
        for e in top:
            lines.append(f"  • {e.claim}")
            if e.business_impact:
                lines.append(f"    ⚠️ {e.business_impact}")
        return "\n".join(lines)

    def _fallback_verdict(self, verdict_data: dict) -> str:
        """LLM 장애 시 규칙 기반 판결 요약."""
        verdict = verdict_data.get("verdict", "UNCERTAIN")
        confidence = verdict_data.get("confidence", 0)
        recommendation = verdict_data.get("recommendation", "")
        return (
            f"⚖️ **판결: {verdict}** (확신도: {confidence:.1%})\n"
            f"{recommendation}"
        )

    @property
    def is_llm_active(self) -> bool:
        """LLM이 실제로 활성화되어 있는지."""
        return self.client.is_available

    def get_debate_summary(self) -> Dict[str, Any]:
        """토론 결과 요약 (대시보드 JSON 내보내기용)."""
        return {
            "llm_active": self.is_llm_active,
            "model": self.client.model_name if self.is_llm_active else "rule_based",
            "rounds": len([r for r in self.responses if r.role == "judge"]),
            "responses": [
                {
                    "role": r.role,
                    "content": r.content,
                    "evidence_count": r.raw_evidence_count,
                }
                for r in self.responses
            ],
        }
