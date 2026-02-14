"use client";

import { useState } from "react";
import { Info, X, ChevronRight, Lightbulb } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function OnboardingGuide() {
    const [isVisible, setIsVisible] = useState(true);
    const [step, setStep] = useState(0);

    const steps = [
        {
            title: "WhyLab에 오신 것을 환영합니다",
            desc: "이곳은 데이터의 '인과관계'를 분석하여 더 나은 의사결정을 돕는 공간입니다. 단순히 현상을 관찰하는 것을 넘어, '왜' 그런 일이 일어났는지 탐구해보세요.",
            icon: <Lightbulb className="w-6 h-6 text-yellow-400" />
        },
        {
            title: "시나리오 선택 (Scenario)",
            desc: "상단의 'Scenario A'와 'Scenario B' 버튼을 통해 분석하고 싶은 주제를 전환할 수 있습니다. 현재 A는 '신용한도 상향', B는 '쿠폰 발송' 시나리오입니다.",
            icon: <ChevronRight className="w-6 h-6 text-brand-400" />
        },
        {
            title: "인과 효과 (ATE) 확인",
            desc: "가장 중요한 숫자는 상단의 'Total ATE'입니다. 이것은 우리가 시행한 정책(처치)이 결과에 미친 순수한 영향력을 의미합니다.",
            icon: <Info className="w-6 h-6 text-blue-400" />
        },
        {
            title: "What-If 시뮬레이션",
            desc: "우측의 시뮬레이터를 조작해보세요. '만약(If)' 우리가 투입량을 늘리거나 줄이면 결과가 어떻게 변할지 실시간으로 예측해줍니다.",
            icon: <Info className="w-6 h-6 text-green-400" />
        }
    ];

    if (!isVisible) return (
        <button
            onClick={() => setIsVisible(true)}
            className="fixed bottom-4 right-4 bg-brand-600 text-white p-3 rounded-full shadow-lg hover:bg-brand-500 transition-colors z-50"
        >
            <Info className="w-6 h-6" />
        </button>
    );

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="fixed bottom-6 right-6 w-80 glass-card border border-brand-500/30 shadow-2xl z-50 overflow-hidden"
            >
                <div className="bg-brand-500/10 p-4 border-b border-white/5 flex justify-between items-start">
                    <div className="flex gap-3">
                        <div className="mt-1">{steps[step].icon}</div>
                        <div>
                            <h3 className="font-bold text-white text-sm">{steps[step].title}</h3>
                            <div className="flex gap-1 mt-1">
                                {steps.map((_, i) => (
                                    <div key={i} className={`h-1 rounded-full transition-all ${i === step ? "w-4 bg-brand-400" : "w-1 bg-slate-600"}`} />
                                ))}
                            </div>
                        </div>
                    </div>
                    <button onClick={() => setIsVisible(false)} className="text-slate-400 hover:text-white">
                        <X className="w-4 h-4" />
                    </button>
                </div>

                <div className="p-4">
                    <p className="text-sm text-slate-300 leading-relaxed min-h-[60px]">
                        {steps[step].desc}
                    </p>
                </div>

                <div className="p-4 pt-0 flex justify-between items-center">
                    <button
                        onClick={() => setStep(Math.max(0, step - 1))}
                        disabled={step === 0}
                        className="text-xs text-slate-500 hover:text-white disabled:opacity-30"
                    >
                        Prev
                    </button>
                    <button
                        onClick={() => {
                            if (step < steps.length - 1) setStep(step + 1);
                            else setIsVisible(false);
                        }}
                        className="bg-brand-500 hover:bg-brand-400 text-white px-4 py-1.5 rounded-md text-xs font-medium transition-colors"
                    >
                        {step === steps.length - 1 ? "Start Exploring" : "Next"}
                    </button>
                </div>
            </motion.div>
        </AnimatePresence>
    );
}
