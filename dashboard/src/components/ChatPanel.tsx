"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { MessageCircle, X, Send, Bot, User, Sparkles } from "lucide-react";
import { CausalAnalysisResult } from "@/types";

interface Props {
    data: CausalAnalysisResult;
}

interface ChatMessage {
    role: "user" | "assistant";
    content: string;
    timestamp: Date;
}

/* ────────────────────────────────────────────
 * 규칙 기반 Q&A 엔진
 * 데이터 컨텍스트를 기반으로 자연어 질문에 응답
 * ──────────────────────────────────────────── */
function answerFromData(question: string, data: CausalAnalysisResult): string {
    const q = question.toLowerCase();
    const m = data.metadata;
    const ate = data.ate;
    const s = data.sensitivity;
    const ai = data.ai_insights;
    const ea = data.estimation_accuracy;

    // ATE 관련
    if (q.includes("ate") || q.includes("효과") || q.includes("인과") || q.includes("영향")) {
        const dir = ate.value < 0 ? "감소" : "증가";
        return `📊 **ATE = ${ate.value.toFixed(4)}**\n\n` +
            `${m.treatment_col}이(가) ${m.outcome_col}에 미치는 평균 처치 효과는 ` +
            `**${Math.abs(ate.value * 100).toFixed(2)}%p ${dir}**입니다.\n\n` +
            `95% 신뢰구간: [${ate.ci_lower.toFixed(4)}, ${ate.ci_upper.toFixed(4)}]\n\n` +
            (ai ? `> ${ai.summary}` : "");
    }

    // 유의성
    if (q.includes("유의") || q.includes("신뢰") || q.includes("p-value") || q.includes("significant")) {
        const sig = !(ate.ci_lower <= 0 && ate.ci_upper >= 0);
        return sig
            ? `✅ **통계적으로 유의합니다.** 95% 신뢰구간 [${ate.ci_lower.toFixed(4)}, ${ate.ci_upper.toFixed(4)}]이 0을 포함하지 않습니다.`
            : `⚠️ **통계적으로 유의하지 않습니다.** 신뢰구간이 0을 포함하므로, 효과가 우연일 가능성을 배제할 수 없습니다.`;
    }

    // 견고성 / 민감도
    if (q.includes("견고") || q.includes("robust") || q.includes("민감") || q.includes("sensitivity")) {
        let r = `🛡️ **견고성 검증 결과**: ${s.status}\n\n`;
        r += `- Placebo Test: ${s.placebo_test.status}\n`;
        r += `- Random Common Cause: ${s.random_common_cause.status}\n`;
        if (s.e_value && s.e_value.status !== "Not Run") {
            r += `- E-value: ${s.e_value.point.toFixed(2)} (${s.e_value.interpretation})\n`;
        }
        if (s.overlap && s.overlap.status !== "Not Run") {
            r += `- Overlap: ${s.overlap.overlap_score} (${s.overlap.interpretation})\n`;
        }
        return r;
    }

    // E-value
    if (q.includes("e-value") || q.includes("교란") || q.includes("confounder")) {
        if (s.e_value && s.e_value.status !== "Not Run") {
            return `🔬 **E-value = ${s.e_value.point.toFixed(2)}** (CI bound: ${s.e_value.ci_bound.toFixed(2)})\n\n` +
                `${s.e_value.interpretation}\n\n` +
                `E-value가 높을수록 미관측 교란에 대해 견고합니다. 일반적으로 ≥2.0이면 양호합니다.`;
        }
        return "E-value 데이터가 아직 계산되지 않았습니다.";
    }

    // GATES / 이질성
    if (q.includes("gates") || q.includes("이질") || q.includes("세그먼트") || q.includes("그룹")) {
        if (s.gates && s.gates.groups.length > 0) {
            let r = `📊 **GATES 분석** (F-stat: ${s.gates.f_statistic})\n\n`;
            r += `${s.gates.heterogeneity}\n\n`;
            s.gates.groups.forEach(g => {
                r += `- **${g.label}** (n=${g.n}): CATE = ${g.mean_cate.toFixed(4)} [${g.ci_lower.toFixed(4)}, ${g.ci_upper.toFixed(4)}]\n`;
            });
            return r;
        }
        return "GATES 분석 데이터가 아직 없습니다.";
    }

    // Overlap
    if (q.includes("overlap") || q.includes("positivity") || q.includes("propensity")) {
        if (s.overlap && s.overlap.status !== "Not Run") {
            return `🔄 **Overlap Score = ${s.overlap.overlap_score}**\n\n` +
                `${s.overlap.interpretation}\n\n` +
                (s.overlap.ps_stats
                    ? `Propensity Score 평균:\n- 처치그룹: ${s.overlap.ps_stats.treated_mean}\n- 통제그룹: ${s.overlap.ps_stats.control_mean}`
                    : "");
        }
        return "Overlap 진단 데이터가 아직 없습니다.";
    }

    // 피처 / SHAP
    if (q.includes("피처") || q.includes("feature") || q.includes("shap") || q.includes("중요")) {
        if (data.explainability?.feature_importance) {
            const top = data.explainability.feature_importance.slice(0, 5);
            let r = "🎯 **Top 5 피처 중요도:**\n\n";
            top.forEach((f, i) => {
                r += `${i + 1}. **${f.feature}**: ${f.importance.toFixed(4)}\n`;
            });
            return r;
        }
        return "피처 중요도 데이터가 없습니다.";
    }

    // 추천 / 권고
    if (q.includes("추천") || q.includes("권고") || q.includes("recommendation") || q.includes("전략")) {
        return ai?.recommendation || "AI 인사이트가 아직 생성되지 않았습니다.";
    }

    // 정확도
    if (q.includes("정확") || q.includes("accuracy") || q.includes("rmse") || q.includes("상관")) {
        if (ea) {
            return `📈 **모델 정확도:**\n\n` +
                `- Correlation: **${ea.correlation.toFixed(3)}**\n` +
                `- RMSE: ${ea.rmse.toFixed(4)}\n` +
                `- MAE: ${ea.mae.toFixed(4)}\n` +
                `- Coverage Rate: ${(ea.coverage_rate * 100).toFixed(1)}%\n` +
                `- Bias: ${ea.bias.toFixed(4)}`;
        }
        return "추정 정확도 데이터가 없습니다.";
    }

    // 시나리오 정보
    if (q.includes("시나리오") || q.includes("scenario") || q.includes("데이터")) {
        return `📋 **시나리오 정보:**\n\n` +
            `- Treatment: **${m.treatment_col}**\n` +
            `- Outcome: **${m.outcome_col}**\n` +
            `- 샘플 수: ${m.n_samples.toLocaleString()}\n` +
            `- 피처: ${m.feature_names.join(", ")}`;
    }

    // 기본 응답
    return `제가 답할 수 있는 질문 유형:\n\n` +
        `- "ATE가 뭐야?" / "인과 효과 알려줘"\n` +
        `- "유의한가?" / "통계적으로 유의해?"\n` +
        `- "견고성 검증 결과는?"\n` +
        `- "E-value 알려줘"\n` +
        `- "GATES 분석 결과"\n` +
        `- "Overlap 진단"\n` +
        `- "피처 중요도" / "SHAP"\n` +
        `- "추천 전략"\n` +
        `- "모델 정확도"\n` +
        `- "시나리오 정보"\n\n` +
        `위 키워드를 포함하여 질문해 주세요! 🤖`;
}

/* ────────────────────────────────────────────
 * 추천 질문 목록
 * ──────────────────────────────────────────── */
const SUGGESTIONS = [
    "인과 효과가 유의한가요?",
    "E-value는 얼마인가요?",
    "세그먼트별 효과 차이는?",
    "어떤 전략을 추천하나요?",
];

/* ────────────────────────────────────────────
 * ChatPanel 컴포넌트
 * ──────────────────────────────────────────── */
export default function ChatPanel({ data }: Props) {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<ChatMessage[]>([
        {
            role: "assistant",
            content: `안녕하세요! WhyLab 인과추론 결과에 대해 물어보세요. 🧠\n\n현재 분석: **${data.metadata.treatment_col} → ${data.metadata.outcome_col}**`,
            timestamp: new Date(),
        },
    ]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSend = async (text?: string) => {
        const question = text || input.trim();
        if (!question) return;

        // 사용자 메시지 추가
        const userMsg: ChatMessage = { role: "user", content: question, timestamp: new Date() };
        setMessages(prev => [...prev, userMsg]);
        setInput("");
        setIsTyping(true);

        // 약간의 딜레이 (타이핑 효과)
        await new Promise(r => setTimeout(r, 500 + Math.random() * 500));

        // 응답 생성
        const answer = answerFromData(question, data);
        const assistantMsg: ChatMessage = { role: "assistant", content: answer, timestamp: new Date() };
        setMessages(prev => [...prev, assistantMsg]);
        setIsTyping(false);
    };

    return (
        <>
            {/* FAB 버튼 */}
            <motion.button
                onClick={() => setIsOpen(!isOpen)}
                className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-brand-500 hover:bg-brand-400 text-white shadow-lg shadow-brand-500/30 flex items-center justify-center transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
            >
                {isOpen ? <X className="w-6 h-6" /> : <MessageCircle className="w-6 h-6" />}
            </motion.button>

            {/* 채팅 패널 */}
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: 20, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 20, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className="fixed bottom-24 right-6 z-50 w-[380px] h-[520px] bg-dark-800/95 backdrop-blur-xl border border-white/10 rounded-2xl flex flex-col overflow-hidden shadow-2xl"
                    >
                        {/* 헤더 */}
                        <div className="px-4 py-3 border-b border-white/10 flex items-center gap-3">
                            <div className="p-1.5 rounded-lg bg-brand-500/20">
                                <Sparkles className="w-4 h-4 text-brand-400" />
                            </div>
                            <div className="flex-1">
                                <h3 className="text-sm font-bold text-white">WhyLab Chat</h3>
                                <p className="text-[10px] text-slate-500">인과추론 결과에 대해 물어보세요</p>
                            </div>
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 border border-green-500/20">
                                온라인
                            </span>
                        </div>

                        {/* 메시지 영역 */}
                        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
                            {messages.map((msg, i) => (
                                <div
                                    key={i}
                                    className={`flex gap-2 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                                >
                                    {msg.role === "assistant" && (
                                        <div className="w-6 h-6 rounded-full bg-brand-500/20 flex items-center justify-center flex-shrink-0 mt-1">
                                            <Bot className="w-3.5 h-3.5 text-brand-400" />
                                        </div>
                                    )}
                                    <div
                                        className={`max-w-[85%] px-3 py-2 rounded-xl text-xs leading-relaxed whitespace-pre-wrap ${msg.role === "user"
                                                ? "bg-brand-500/20 text-white rounded-br-sm"
                                                : "bg-white/5 text-slate-300 rounded-bl-sm"
                                            }`}
                                    >
                                        {msg.content}
                                    </div>
                                    {msg.role === "user" && (
                                        <div className="w-6 h-6 rounded-full bg-slate-600/50 flex items-center justify-center flex-shrink-0 mt-1">
                                            <User className="w-3.5 h-3.5 text-slate-300" />
                                        </div>
                                    )}
                                </div>
                            ))}

                            {isTyping && (
                                <div className="flex gap-2">
                                    <div className="w-6 h-6 rounded-full bg-brand-500/20 flex items-center justify-center flex-shrink-0">
                                        <Bot className="w-3.5 h-3.5 text-brand-400" />
                                    </div>
                                    <div className="px-3 py-2 rounded-xl bg-white/5 text-slate-400 text-xs">
                                        <span className="inline-flex gap-1">
                                            <span className="animate-bounce">·</span>
                                            <span className="animate-bounce" style={{ animationDelay: "0.1s" }}>·</span>
                                            <span className="animate-bounce" style={{ animationDelay: "0.2s" }}>·</span>
                                        </span>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* 추천 질문 */}
                        {messages.length <= 2 && (
                            <div className="px-4 py-2 flex flex-wrap gap-1.5">
                                {SUGGESTIONS.map((s) => (
                                    <button
                                        key={s}
                                        onClick={() => handleSend(s)}
                                        className="text-[10px] px-2.5 py-1 rounded-full bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors border border-white/5"
                                    >
                                        {s}
                                    </button>
                                ))}
                            </div>
                        )}

                        {/* 입력 영역 */}
                        <div className="px-3 py-2 border-t border-white/10">
                            <div className="flex items-center gap-2 bg-white/5 rounded-xl px-3 py-2">
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                                    placeholder="질문을 입력하세요..."
                                    className="flex-1 bg-transparent text-xs text-white placeholder-slate-500 outline-none"
                                    disabled={isTyping}
                                />
                                <button
                                    onClick={() => handleSend()}
                                    disabled={!input.trim() || isTyping}
                                    className="p-1.5 rounded-lg bg-brand-500/20 hover:bg-brand-500/40 text-brand-400 disabled:opacity-30 transition-colors"
                                >
                                    <Send className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}
