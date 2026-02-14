"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { getCausalData } from "@/lib/dataLoader";
import { CausalAnalysisResult } from "@/types";
import CausalGraph from "@/components/CausalGraph";

function CausalGraphContent() {
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
            <div className="flex items-center justify-center h-[calc(100vh-100px)]">
                <div className="flex flex-col items-center gap-4">
                    <div className="w-12 h-12 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
                    <p className="text-slate-400 text-sm">인과 그래프 로딩 중...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6 h-[calc(100vh-100px)] flex flex-col">
            <div className="flex justify-between items-end shrink-0">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-2">Detailed Causal Graph</h1>
                    <p className="text-slate-400">
                        Visualizing Causal Structure for Scenario {scenario}
                    </p>
                </div>
            </div>

            <div className="flex-1 w-full glass-card p-4 overflow-hidden">
                <CausalGraph nodes={data.dag.nodes} edges={data.dag.edges} />
            </div>
        </div>
    );
}

export default function CausalGraphPage() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <CausalGraphContent />
        </Suspense>
    );
}
