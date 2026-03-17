# WhyLab — Roadmap & Future Work

## Current Status (2026-03-17)

- **Paper**: NeurIPS 2026 submission draft, 17p, 6 experiments (E1-E6)
- **Validated on**: Gemini 2.0 Flash only
- **Main result**: Zero regression across 10,500 episodes; E6 shows all 3 components contribute independently
- **Known weakness**: Single-LLM validation, oscillation effect not statistically significant (p=0.26)

---

## Phase 2: Multi-LLM Empirical Study (March–May 2026)

### Goal
Establish whether oscillation in self-improving agents is a universal phenomenon across model families.

### Target Models

| Family | Mid-tier (main) | Flagship (subset) | Justification |
|--------|----------------|-------------------|---------------|
| OpenAI | GPT-5 mini ($0.25/$2.00) | GPT-5.4 ($1.75/$14) | Most widely used |
| Anthropic | Claude Sonnet 4.6 ($3/$15) | Opus 4.6 ($5/$25) | Different architecture |
| Google | Gemini 3 Flash ($0.50/$3) | Gemini 3.1 Pro ($2/$12) | Already validated on 2.0 |

**Why mid-tier for main experiments, not flagship?**
1. Cost-controlled: 100 problems × 3 seeds × 7 attempts = 2,100 calls/model
2. Fair comparison: same capability tier across families
3. Flagship subset (50 problems) validates that results hold for stronger models
4. Reviewers care about generalization across families, not single-model performance

**Why not DeepSeek, Llama, etc.?**
- Focus on closed-source models where self-improvement is most relevant
- Open-source models can be added as supplementary material

### Experiment Design
- Reuse existing SWE-bench Reflexion loop (swap LLM client only)
- Use Batch API (OpenAI/Anthropic) to avoid 429 rate limits
- Checkpoint after each problem for crash recovery
- Budget: ~$50 total

### Deliverables
1. Cross-model oscillation frequency comparison
2. Component ablation (C1/C2/C3) per model
3. Statistical significance tests per model
4. Paper reframing: "empirical study of failure modes → WhyLab as solution"

### Timeline
| Week | Task |
|:----:|------|
| W1 (3/17) | Current paper finalized, this roadmap |
| W2 (3/24) | LLM client adapters (OpenAI, Anthropic) |
| W3 (3/31) | Run GPT-5 mini experiments |
| W4 (4/7) | Run Claude Sonnet 4.6 experiments |
| W5 (4/14) | Run Gemini 3 Flash + flagship subset |
| W6 (4/21) | Analysis + paper rewrite |
| W7 (4/28) | Final verification + NeurIPS submission update |

---

## Phase 3: Long-term Directions

- Non-stationary real-world deployment (C1/C3 in production agents)
- Multi-agent orchestration safety
- Integration with existing agent frameworks (LangChain, CrewAI)
