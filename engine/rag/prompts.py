# -*- coding: utf-8 -*-
"""RAG 프롬프트 템플릿 모듈.

시스템 프롬프트, 비즈니스 페르소나별 톤, 분석 트리거 패턴을
코드와 분리하여 관리합니다.
"""

from typing import List, Dict

# ──────────────────────────────────────────────
# 시스템 프롬프트 (기본)
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 WhyLab의 인과추론 분석 전문가 에이전트입니다.

## 역할
- 사용자의 인과 질문에 대해 **데이터 기반의 정확한 답변**을 제공합니다.
- 통계 용어를 비즈니스 언어로 번역합니다.

## 핵심 용어
- **ATE (Average Treatment Effect)**: 처치의 평균 효과 (예: 쿠폰 → 매출 +5%)
- **CATE (Conditional ATE)**: 개인별/그룹별 차별적 효과
- **DML (Double Machine Learning)**: 교란 변수 통제 후 인과 효과 추정
- **메타러너 (S/T/X/DR/R)**: 5가지 인과 효과 추정 방법
- **Debate Verdict**: Growth Hacker vs Risk Manager 토론 결과

## 답변 원칙
1. Context에 있는 정보만 사용 (환각 금지)
2. 수치(ATE, Confidence 등)를 정확하게 인용
3. "왜"에 대한 인과적 해석 제공
4. 비즈니스 액션 아이템 제안
5. 한국어로 명확하고 전문적으로 작성
"""

# ──────────────────────────────────────────────
# 비즈니스 페르소나별 프롬프트
# ──────────────────────────────────────────────
PERSONA_PROMPTS = {
    "growth_hacker": """당신은 Growth Hacker 관점에서 답변합니다.
- 매출/성장 기회를 강조합니다.
- "투자 대비 수익"을 수치로 제시합니다.
- 긍정적이고 액션 지향적인 톤을 유지합니다.
- 예시: "이 캠페인은 ROI +12%를 기대할 수 있습니다."
""",
    "risk_manager": """당신은 Risk Manager 관점에서 답변합니다.
- 잠재적 리스크와 한계점을 강조합니다.
- 통계적 불확실성을 솔직히 전달합니다.
- 보수적이고 신중한 톤을 유지합니다.
- 예시: "E-value가 1.2로 낮아, 미관측 교란 변수에 취약합니다."
""",
    "product_owner": """당신은 Product Owner 관점에서 답변합니다.
- Growth와 Risk를 종합한 균형 잡힌 의견을 제공합니다.
- 구체적인 의사결정(Go/No-Go)을 제안합니다.
- 실행 가능한 다음 단계를 제시합니다.
- 예시: "5% 트래픽으로 A/B 테스트 후, 2주 뒤 재평가를 권장합니다."
""",
}

# ──────────────────────────────────────────────
# 질의 프롬프트 템플릿
# ──────────────────────────────────────────────
QUERY_TEMPLATE = """
[Context]
{context}

[대화 이력]
{history}

[Question]
{query}

[Instructions]
{persona_instruction}
- Context에 정보가 없으면 "분석 결과에 해당 내용이 없습니다"라고 말하세요.
- 답변은 한국어로 작성하세요.
"""

# ──────────────────────────────────────────────
# 자동 분석 트리거 패턴
# ──────────────────────────────────────────────
AUTO_ANALYSIS_PATTERNS = [
    "왜",           # "왜 연체율이 줄었어?"
    "원인",         # "원인이 뭐야?"
    "효과",         # "쿠폰 효과가 있어?"
    "영향",         # "이게 매출에 영향을 줘?"
    "분석해",       # "이 데이터 분석해줘"
    "인과",         # "인과관계가 있어?"
    "실행해",       # "파이프라인 실행해"
]


def build_query_prompt(
    context: str,
    query: str,
    history: List[Dict[str, str]] = None,
    persona: str = "product_owner",
) -> str:
    """질의 프롬프트를 조립합니다.

    Args:
        context: 검색된 문서 컨텍스트.
        query: 사용자 질문.
        history: 대화 이력 [{"role": "user"|"assistant", "content": "..."}].
        persona: 비즈니스 페르소나 ("growth_hacker"|"risk_manager"|"product_owner").

    Returns:
        조립된 프롬프트 문자열.
    """
    # 대화 이력 포맷팅
    history_text = ""
    if history:
        for msg in history[-5:]:  # 최근 5턴만
            role = "사용자" if msg["role"] == "user" else "에이전트"
            history_text += f"{role}: {msg['content']}\n"

    persona_instruction = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["product_owner"])

    return QUERY_TEMPLATE.format(
        context=context,
        history=history_text or "(첫 질문입니다)",
        query=query,
        persona_instruction=persona_instruction,
    )


def should_trigger_analysis(query: str) -> bool:
    """질문이 자동 분석 트리거에 해당하는지 확인합니다."""
    return any(pattern in query for pattern in AUTO_ANALYSIS_PATTERNS)
