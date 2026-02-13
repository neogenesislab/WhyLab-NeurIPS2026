"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { getCausalData } from "@/lib/dataLoader";
import { CausalAnalysisResult } from "@/types";
import CausalGraph from "@/components/CausalGraph";
import WhatIfSimulator from "@/components/WhatIfSimulator";
import StatsCards from "@/components/StatsCards";
import CausalCharts from "@/components/CausalCharts";
import SensitivityReport from "@/components/SensitivityReport";
import ModelComparison from "@/components/ModelComparison";
import ExplainabilityPanel from "@/components/ExplainabilityPanel";
import EstimationAccuracy from "@/components/EstimationAccuracy";
import AIInsightPanel from "@/components/AIInsightPanel";
import DiagnosticsPanel from "@/components/DiagnosticsPanel";
import ChatPanel from "@/components/ChatPanel";

function DashboardContent() {
    const searchParams = useSearchParams();
    const scenario = searchParams.get("scenario") || "A";
    const [data, setData] = useState<CausalAnalysisResult | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        setLoading(true);
        getCausalData(scenario).then((result) => {
            setData(result);
            setLoading(false);
        });
    }, [scenario]);

    if (loading || !data) {
        return (
            <div className="flex items-center justify-center h-[60vh]">
                <div className="flex flex-col items-center gap-4">
                    <div className="w-12 h-12 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
                    <p className="text-slate-400 text-sm">분석 데이터 로딩 중...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Title & Scenario Selector */}
            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Analysis Overview</h1>
                    <p className="text-slate-400">
                        {data.ate.description ?? `ATE = ${data.ate.value.toFixed(4)} [${data.ate.ci_lower.toFixed(4)}, ${data.ate.ci_upper.toFixed(4)}]`}
                    </p>
                </div>

                <div className="flex bg-slate-800/50 p-1 rounded-lg border border-white/5">
                    <Link
                        href="/dashboard?scenario=A"
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${scenario === 'A' ? 'bg-brand-500 text-white shadow-lg' : 'text-slate-400 hover:text-white'}`}
                    >
                        Scenario A (Credit Limit)
                    </Link>
                    <Link
                        href="/dashboard?scenario=B"
                        className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${scenario === 'B' ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30 shadow-lg' : 'text-slate-400 hover:text-white'}`}
                    >
                        Scenario B (Coupon)
                    </Link>
                </div>
            </div>

            {/* KPI Cards */}
            <StatsCards data={data} />

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[500px]">
                <div className="lg:col-span-2 h-full">
                    <CausalGraph nodes={data.dag.nodes} edges={data.dag.edges} />
                </div>
                <div className="lg:col-span-1 h-full">
                    <WhatIfSimulator
                        baseValue={scenario === 'A' ? 200 : 0.5}
                        coefficient={scenario === 'A' ? -0.05 : 0.2}
                        intercept={scenario === 'A' ? 5 : 0.1}
                        treatmentName={scenario === 'A' ? "Credit Limit (만원)" : "Coupon Sent (Prob)"}
                        outcomeName={scenario === 'A' ? "Default Rate (%)" : "Join Prob (%)"}
                    />
                </div>
            </div>

            {/* Bottom: Detailed Charts */}
            <div className="h-[350px]">
                <CausalCharts data={data} />
            </div>

            {/* Estimation Accuracy (Ground Truth 검증) */}
            <EstimationAccuracy data={data} />

            {/* AI Interpretation */}
            <AIInsightPanel data={data} />

            {/* Explainability (SHAP) */}
            <ExplainabilityPanel data={data} />

            {/* Deep Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <SensitivityReport data={data} />
                <ModelComparison data={data} />
            </div>

            {/* Statistical Diagnostics (Phase 4) */}
            <DiagnosticsPanel data={data} />

            {/* Interactive Chat (Phase 6) */}
            <ChatPanel data={data} />
        </div>
    );
}

export default function DashboardPage() {
    return (
        <Suspense fallback={
            <div className="flex items-center justify-center h-[60vh]">
                <div className="w-12 h-12 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
            </div>
        }>
            <DashboardContent />
        </Suspense>
    );
}
