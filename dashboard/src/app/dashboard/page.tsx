"use client";

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { getCausalData } from "@/lib/dataLoader";
import { CausalAnalysisResult } from "@/types";
import CausalGraph from "@/components/CausalGraph";
import PolicySimulator from "@/components/PolicySimulator";
import StatsCards from "@/components/StatsCards";
import CausalCharts from "@/components/CausalCharts";
import SensitivityReport from "@/components/SensitivityReport";
import ModelComparison from "@/components/ModelComparison";
import ExplainabilityPanel from "@/components/ExplainabilityPanel";
import EstimationAccuracy from "@/components/EstimationAccuracy";
import AIInsightPanel from "@/components/AIInsightPanel";
import DiagnosticsPanel from "@/components/DiagnosticsPanel";
import ChatPanel from "@/components/ChatPanel";
import DebateVerdict from "@/components/DebateVerdict";
import BenchmarkTable from "@/components/BenchmarkTable";
import ConformalBand from "@/components/ConformalBand";
import OnboardingGuide from "@/components/OnboardingGuide";
import AnalysisSection from "@/components/AnalysisSection";
import CATEExplorer from "@/components/CATEExplorer";

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
        <div className="space-y-8 pb-20 relative">
            <OnboardingGuide />

            {/* Header Area */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 border-b border-white/5 pb-6">
                <div>
                    <div className="flex items-center gap-3 mb-2">
                        <h1 className="text-3xl font-bold text-white tracking-tight">Causal Analysis Report</h1>
                        {scenario === 'A' ? (
                            <span className="px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 text-xs font-bold border border-green-500/30">
                                REAL DATA
                            </span>
                        ) : (
                            <span className="px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-400 text-xs font-bold border border-yellow-500/30">
                                DEMO MOCKUP
                            </span>
                        )}
                    </div>
                    <p className="text-slate-400 max-w-2xl text-sm leading-relaxed">
                        {data.ate.description ?? `ATE = ${data.ate.value.toFixed(4)} (Confidence Interval: [${data.ate.ci_lower.toFixed(4)}, ${data.ate.ci_upper.toFixed(4)}])`}
                    </p>
                </div>

                <div className="flex bg-slate-800/80 p-1.5 rounded-xl border border-white/10 shadow-lg">
                    <Link
                        href="/dashboard?scenario=A"
                        className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${scenario === 'A' ? 'bg-brand-500 text-white shadow-md' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                    >
                        <span className="w-2 h-2 rounded-full bg-green-400" />
                        <div>
                            <div className="leading-none">Scenario A</div>
                            <div className="text-[10px] font-normal opacity-80 mt-0.5">Credit Limit (Real)</div>
                        </div>
                    </Link>
                    <Link
                        href="/dashboard?scenario=B"
                        className={`px-5 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${scenario === 'B' ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30 shadow-md' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
                    >
                        <span className="w-2 h-2 rounded-full bg-yellow-400" />
                        <div>
                            <div className="leading-none">Scenario B</div>
                            <div className="text-[10px] font-normal opacity-80 mt-0.5">Coupon (Demo)</div>
                        </div>
                    </Link>
                </div>
            </div>

            {/* 1. HERO: Policy Simulator (Business Impact First) */}
            <div className="h-[600px] w-full">
                <PolicySimulator
                    baseLimit={scenario === 'A' ? 1000 : 50}
                    baseDefaultRate={scenario === 'A' ? 0.02 : 0.4}
                />
            </div>

            {/* 1.5 CATE Explorer (누구에게 효과가 큰가?) */}
            <CATEExplorer
                distribution={data.cate_distribution}
                segments={data.segments}
            />

            {/* 2. Executive Summary (Supporting Metrics) */}
            <div className="space-y-6">
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                    <span className="w-2 h-6 bg-slate-600 rounded-full"></span>
                    Detailed Metrics & Diagnostics
                </h2>
                <StatsCards data={data} />
                <DebateVerdict data={data} />
            </div>

            {/* 3. Visual Insight (Causal Graph) */}
            <div className="h-[500px] w-full glass-card border-white/5 overflow-hidden flex flex-col">
                <div className="p-4 border-b border-white/5 bg-slate-900/30 flex justify-between items-center">
                    <h3 className="font-semibold text-white">Causal Structure (DAG)</h3>
                    <Link href={`/dashboard/causal-graph?scenario=${scenario}`} className="text-xs text-brand-400 hover:text-brand-300 flex items-center gap-1">
                        Full Screen ↗
                    </Link>
                </div>
                <div className="flex-1 relative">
                    <CausalGraph nodes={data.dag.nodes} edges={data.dag.edges} />
                </div>
            </div>

            {/* 3. Detailed Analysis (Collapsible Sections) */}
            <div className="space-y-4">
                <AnalysisSection title="Deep Causal Metrics" description="Distribution, Conformal Prediction, and Sensitivity">
                    <div className="space-y-8">
                        <div className="h-[350px]">
                            <CausalCharts data={data} />
                        </div>
                        <div className="h-[400px]">
                            <ConformalBand data={data} />
                        </div>
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <SensitivityReport data={data} />
                            <EstimationAccuracy data={data} />
                        </div>
                    </div>
                </AnalysisSection>

                <AnalysisSection title="AI Interpretation & Explainability" description="Natural Language Insights and SHAP Values">
                    <div className="space-y-6">
                        <AIInsightPanel data={data} />
                        <ExplainabilityPanel data={data} />
                    </div>
                </AnalysisSection>

                <AnalysisSection title="Model Diagnostics & Benchmarks" description="Technical Robustness Checks">
                    <div className="space-y-6">
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <ModelComparison data={data} />
                            <BenchmarkTable data={data} />
                        </div>
                        <DiagnosticsPanel data={data} />
                    </div>
                </AnalysisSection>

                <AnalysisSection title="Ask WhyLab Agent" description="Interactive Q&A Session" defaultOpen={true}>
                    <ChatPanel data={data} />
                </AnalysisSection>
            </div>
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
