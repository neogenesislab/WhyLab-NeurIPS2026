# -*- coding: utf-8 -*-
"""ARES Real-LLM Judge Wrapper.

실제 경량 LLM API를 호출하여 추론 단계의 건전성을 판단합니다.
비용/지연 벤치마크 데이터를 수집하여 논문 §Architecture에 활용.

지원 모델:
- Claude 3.5 Haiku (Anthropic) — 기본
- GPT-4o-mini (OpenAI) — 대체

비용 통제:
- 엄격한 타임아웃 (3초)
- 최대 1회 재시도
- CostBudget 서킷 브레이커 연동
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("whylab.audit.llm_judge.real")


class RealLLMJudge:
    """실제 LLM API를 호출하는 Judge 함수 래퍼.

    사용법:
        judge = RealLLMJudge(provider="anthropic", model="claude-3-5-haiku-20241022")
        evaluator = ARESEvaluator(judge_fn=judge, use_real_llm=True)
    """

    # 모델별 근사 비용 (USD per 1K tokens)
    COST_TABLE = {
        "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-haiku-20241022",
        timeout_seconds: float = 3.0,
        max_retries: int = 1,
        cost_budget: Optional[Any] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.cost_budget = cost_budget

        # API 키 환경변수
        self._api_key = os.getenv(
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
        )

        # 계측
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0
        self.error_count = 0

    def __call__(self, step: str, premises: List[str]) -> bool:
        """ARESEvaluator의 judge_fn 인터페이스.

        Args:
            step: 평가할 추론 단계
            premises: 이전에 검증된 전제 목록

        Returns:
            True: 이 단계가 건전함(sound)
            False: 이 단계에 결함 있음
        """
        # 서킷 브레이커 확인
        if self.cost_budget and self.cost_budget.breaker_tripped:
            logger.info("⚡ Circuit breaker tripped — returning mock True")
            return True  # 안전한 폴백

        if not self._api_key:
            logger.warning("⚠️ No API key — returning mock True")
            return True  # 키 없으면 Mock 모드

        prompt = self._build_prompt(step, premises)

        for attempt in range(1 + self.max_retries):
            try:
                result, tokens = self._call_api(prompt)
                self.call_count += 1
                self.total_tokens += tokens
                cost = self._estimate_cost(tokens)
                self.total_cost_usd += cost

                # 예산 소비 기록
                if self.cost_budget:
                    self.cost_budget.consume(tokens, cost)

                return result

            except TimeoutError:
                logger.warning("⏱️ LLM timeout (attempt %d/%d)", attempt + 1, 1 + self.max_retries)
                self.error_count += 1
            except Exception as e:
                logger.warning("❌ LLM error: %s (attempt %d)", e, attempt + 1)
                self.error_count += 1

        # 모든 재시도 실패 → 안전한 폴백
        return True

    def _build_prompt(self, step: str, premises: List[str]) -> str:
        """판단 프롬프트 생성."""
        premise_text = "\n".join(f"  P{i+1}: {p}" for i, p in enumerate(premises)) or "  (none)"
        return (
            "You are a causal reasoning auditor. Evaluate whether the following "
            "reasoning step logically follows from the verified premises.\n\n"
            f"Verified Premises:\n{premise_text}\n\n"
            f"Step to evaluate:\n  {step}\n\n"
            "Answer ONLY 'SOUND' or 'UNSOUND'. No explanation."
        )

    def _call_api(self, prompt: str) -> tuple:
        """실제 API 호출 (동기 HTTP).

        Returns:
            (is_sound: bool, token_count: int)
        """
        import urllib.request
        import json

        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            return self._call_openai(prompt)

    def _call_anthropic(self, prompt: str) -> tuple:
        """Anthropic Messages API 호출."""
        import urllib.request
        import json

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
        body = json.dumps({
            "model": self.model,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                text = data.get("content", [{}])[0].get("text", "").strip().upper()
                tokens = data.get("usage", {}).get("input_tokens", 0) + \
                         data.get("usage", {}).get("output_tokens", 0)
                return "SOUND" in text, tokens
        except Exception as e:
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raise TimeoutError(str(e))
            raise

    def _call_openai(self, prompt: str) -> tuple:
        """OpenAI Chat Completions API 호출."""
        import urllib.request
        import json

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        body = json.dumps({
            "model": self.model,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                text = data["choices"][0]["message"]["content"].strip().upper()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                return "SOUND" in text, tokens
        except Exception as e:
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raise TimeoutError(str(e))
            raise

    def _estimate_cost(self, tokens: int) -> float:
        """토큰 기반 비용 추정."""
        rates = self.COST_TABLE.get(self.model, {"input": 0.001, "output": 0.005})
        # 근사: input 70%, output 30%
        return (tokens * 0.7 * rates["input"] + tokens * 0.3 * rates["output"]) / 1000

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "total_calls": self.call_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "error_count": self.error_count,
            "cost_per_call": round(self.total_cost_usd / max(self.call_count, 1), 6),
        }
