"""
Paper Draft Generator — 자동 논문 초안 생성기 (Sprint 34)
=========================================================
WhyLab 연구 파이프라인의 결과물을 학술 논문 형태로 자동 정리합니다.

[목표]
- KG에서 검증된 인과 관계를 추출
- 실험 이력 + Critic 리뷰를 Method/Results로 구조화
- LaTeX/Markdown 형태의 논문 초안 생성

[사용 예시]
    generator = PaperDraftGenerator()
    draft = generator.generate_draft("GC_002")
"""
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("whylab.paper")


class PaperDraftGenerator:
    """
    학술 논문 초안 자동 생성기.
    
    Knowledge Graph + 실험 이력 + 헌법 가드 판정을 종합하여
    IMRAD (Introduction-Methods-Results-And-Discussion) 구조로 생성합니다.
    """

    TEMPLATE_SECTIONS = [
        "title", "abstract", "introduction",
        "related_work", "methodology", "results",
        "discussion", "conclusion", "references",
    ]

    def generate_draft(
        self,
        grand_challenge_id: Optional[str] = None,
        include_latex: bool = False,
    ) -> dict:
        """
        논문 초안를 생성합니다.
        
        Args:
            grand_challenge_id: 특정 Grand Challenge 기반 초안 (None이면 전체)
            include_latex: LaTeX 형식 포함 여부
        """
        # KG에서 데이터 수집
        kg_data = self._collect_from_kg(grand_challenge_id)
        
        # IMRAD 구조 생성
        draft = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "grand_challenge_id": grand_challenge_id,
                "version": "draft-v0.1",
                "whylab_version": "v4.0",
            },
            "title": self._generate_title(kg_data),
            "abstract": self._generate_abstract(kg_data),
            "sections": {
                "introduction": self._generate_introduction(kg_data),
                "methodology": self._generate_methodology(kg_data),
                "results": self._generate_results(kg_data),
                "discussion": self._generate_discussion(kg_data),
                "conclusion": self._generate_conclusion(kg_data),
            },
            "statistics": {
                "total_experiments": kg_data.get("experiment_count", 0),
                "validated_hypotheses": kg_data.get("validated_count", 0),
                "rejected_hypotheses": kg_data.get("rejected_count", 0),
                "methods_used": kg_data.get("methods", []),
                "total_nodes": kg_data.get("node_count", 0),
                "total_edges": kg_data.get("edge_count", 0),
            },
        }

        if include_latex:
            draft["latex"] = self._to_latex(draft)

        logger.info(
            "논문 초안 생성 | GC=%s | 실험=%d건 | 검증 가설=%d건",
            grand_challenge_id,
            kg_data.get("experiment_count", 0),
            kg_data.get("validated_count", 0),
        )

        return draft

    def _collect_from_kg(self, gc_id: Optional[str]) -> dict:
        """Knowledge Graph에서 논문에 필요한 데이터를 수집합니다."""
        try:
            from api.graph import kg
            stats = kg.get_stats()
            
            # KG에서 가설, 실험, 리뷰 데이터 수집
            hypotheses = []
            experiments = []
            reviews = []
            methods = set()
            
            for node, data in kg.graph.nodes(data=True):
                node_type = data.get("type", "")
                if node_type == "hypothesis":
                    hypotheses.append(data)
                elif node_type == "experiment":
                    experiments.append(data)
                    method = data.get("method", "")
                    if method:
                        methods.add(method)
                elif node_type == "review":
                    reviews.append(data)
            
            validated = [h for h in hypotheses if h.get("status") == "validated"]
            rejected = [h for h in hypotheses if h.get("status") == "rejected"]
            
            return {
                "node_count": stats.get("nodes", 0),
                "edge_count": stats.get("edges", 0),
                "hypotheses": hypotheses,
                "experiments": experiments,
                "reviews": reviews,
                "methods": sorted(methods),
                "experiment_count": len(experiments),
                "validated_count": len(validated),
                "rejected_count": len(rejected),
            }
        except Exception as e:
            logger.warning("KG 데이터 수집 실패: %s", str(e))
            return {
                "node_count": 0, "edge_count": 0,
                "hypotheses": [], "experiments": [],
                "reviews": [], "methods": [],
                "experiment_count": 0, "validated_count": 0,
                "rejected_count": 0,
            }

    def _generate_title(self, data: dict) -> str:
        methods = ", ".join(data.get("methods", ["Causal Inference"])[:3])
        return f"Autonomous Causal Discovery via Multi-Agent Collaboration: A {methods} Approach"

    def _generate_abstract(self, data: dict) -> str:
        return (
            f"본 연구는 다중 에이전트 시스템(MAS)을 활용한 자율적 인과 발견 프레임워크를 제시합니다. "
            f"WhyLab 엔진은 가설 생성(Theorist), 실험 실행(Engineer), 비판적 검토(Critic)의 "
            f"3자 상호작용을 통해 인과적 주장을 자율적으로 검증합니다. "
            f"총 {data.get('experiment_count', 0)}건의 실험에서 "
            f"{data.get('validated_count', 0)}건의 가설이 검증되었으며, "
            f"{len(data.get('methods', []))}가지 추정 알고리즘이 사용되었습니다. "
            f"연구 헌법(Research Constitution)을 통한 자동화된 방법론적 가드레일이 "
            f"재현성과 엄밀성을 보장합니다."
        )

    def _generate_introduction(self, data: dict) -> str:
        return (
            "## 1. Introduction\n\n"
            "인과 추론(Causal Inference)은 관찰 데이터에서 처치 효과를 추정하는 "
            "통계학적 프레임워크입니다. 기존 연구는 개별 추정량(estimator)의 성능 비교에 "
            "집중하였으나, 가설 생성부터 검증까지의 전체 연구 프로세스를 자동화하는 시도는 "
            "제한적이었습니다.\n\n"
            "본 연구는 WhyLab — 다중 에이전트 기반 인과 추론 엔진을 통해 "
            "연구 프로세스 전체를 자율화하는 새로운 접근법을 제안합니다."
        )

    def _generate_methodology(self, data: dict) -> str:
        methods = data.get("methods", [])
        method_list = "\n".join([f"- {m}" for m in methods]) if methods else "- (실험 미실행)"
        return (
            "## 3. Methodology\n\n"
            "### 3.1 Multi-Agent Architecture\n"
            "WhyLab은 계층적 오케스트레이터-워커(Orchestrator-Worker) 패턴을 사용합니다:\n"
            "- **Theorist**: Knowledge Graph Gap 분석 기반 가설 생성\n"
            "- **Engineer**: Code-Then-Execute 패턴의 SandboxExecutor 실험\n"
            "- **Critic**: LLM-as-a-Judge + ConstitutionGuard 이중 검증\n\n"
            "### 3.2 Estimation Methods\n"
            f"다음 추정 알고리즘이 UCB1 밴디트에 의해 자율 선택되었습니다:\n{method_list}\n\n"
            "### 3.3 Research Constitution\n"
            "모든 실험은 연구 헌법(12개 조항)에 의해 자동 가드레일됩니다. "
            "제1조(반증 테스트), 제4조(다원적 방법론), 제5조(표본 크기), "
            "제6조(시드 고정), 제12조(방법론 다양성) 등이 런타임에 강제됩니다."
        )

    def _generate_results(self, data: dict) -> str:
        return (
            "## 4. Results\n\n"
            f"총 {data.get('experiment_count', 0)}건의 실험이 SandboxExecutor를 통해 실행되었습니다.\n\n"
            f"| 지표 | 값 |\n|---|---|\n"
            f"| 총 실험 | {data.get('experiment_count', 0)}건 |\n"
            f"| 검증된 가설 | {data.get('validated_count', 0)}건 |\n"
            f"| 기각된 가설 | {data.get('rejected_count', 0)}건 |\n"
            f"| 사용 알고리즘 | {len(data.get('methods', []))}종 |\n"
            f"| Knowledge Graph | {data.get('node_count', 0)}노드, {data.get('edge_count', 0)}엣지 |\n"
        )

    def _generate_discussion(self, data: dict) -> str:
        return (
            "## 5. Discussion\n\n"
            "본 연구의 주요 기여는 다음과 같습니다:\n\n"
            "1. **자율적 연구 프로세스**: 가설 생성→실험→비판→진화의 전체 사이클을 "
            "인간 개입 없이 자율적으로 수행하는 최초의 인과 추론 프레임워크\n"
            "2. **방법론적 엄밀성 자동화**: Research Constitution을 통한 런타임 "
            "가드레일로 재현성과 타당성을 자동 보장\n"
            "3. **적응적 방법론 선택**: UCB1 밴디트 기반의 최적 추정기 자동 선택으로 "
            "연구자 편향 제거\n\n"
            "### 5.1 Limitations\n"
            "- 합성 데이터(STEAM) 기반 검증으로, 실세계 데이터에서의 성능은 추가 검증 필요\n"
            "- LLM 기반 평가의 일관성에 대한 추가 연구 필요\n"
            "- 대규모 DAG에서의 계산 비용 최적화 미완"
        )

    def _generate_conclusion(self, data: dict) -> str:
        return (
            "## 6. Conclusion\n\n"
            "WhyLab은 다중 에이전트 협업과 연구 헌법을 결합하여 "
            "인과 추론 연구의 전체 프로세스를 자율화하는 새로운 패러다임을 제시합니다. "
            "향후 실세계 데이터셋 적용과 에이전트 진화 메커니즘의 고도화를 통해 "
            "자율적 과학 발견(Autonomous Scientific Discovery)에 기여할 것으로 기대됩니다."
        )

    def _to_latex(self, draft: dict) -> str:
        """Markdown 초안을 LaTeX 형태로 변환합니다."""
        return (
            "\\documentclass{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            f"\\title{{{draft['title']}}}\n"
            "\\author{WhyLab Autonomous Research Engine}\n"
            f"\\date{{{draft['metadata']['generated_at'][:10]}}}\n\n"
            "\\begin{document}\n"
            "\\maketitle\n\n"
            "\\begin{abstract}\n"
            f"{draft['abstract']}\n"
            "\\end{abstract}\n\n"
            "% 전체 섹션은 Markdown 초안에서 수동 변환 필요\n"
            "\\end{document}\n"
        )


# 모듈 레벨 싱글턴
paper_generator = PaperDraftGenerator()
