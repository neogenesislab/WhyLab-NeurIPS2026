"use client";

import { motion } from "framer-motion";
import { Shield, ShieldCheck, ShieldAlert, BarChart3, Layers, AlertTriangle, Info, CheckCircle2, XCircle } from "lucide-react";
import { CausalAnalysisResult } from "@/types";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend
} from "recharts";

interface Props {
    data: CausalAnalysisResult;
}

function StatusBadge({ status }: { status: string }) {
    const color =
        status === "Pass" ? "bg-green-500/20 text-green-400 border-green-500/20" :
            status === "Fail" ? "bg-red-500/20 text-red-400 border-red-500/20" :
                status === "Info" ? "bg-blue-500/20 text-blue-400 border-blue-500/20" :
                    "bg-slate-500/20 text-slate-400 border-slate-500/20";
    return (
        <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${color}`}>
            {status}
        </span>
    );
}

export default function DiagnosticsPanel({ data }: Props) {
    const s = data.sensitivity;
    if (!s) return null;

    const overallIcon = s.status === "Pass"
        ? <ShieldCheck className="w-5 h-5 text-green-400" />
        : <ShieldAlert className="w-5 h-5 text-yellow-400" />;

    // Overlap Histogram Data Preparation
    const overlapData = s.overlap?.ps_histogram ? s.overlap.ps_histogram.bin_edges.slice(0, -1).map((edge, i) => ({
        range: `${edge.toFixed(1)}-${s.overlap!.ps_histogram!.bin_edges[i + 1].toFixed(1)}`,
        Treated: s.overlap!.ps_histogram!.treated_counts[i],
        Control: s.overlap!.ps_histogram!.control_counts[i],
    })) : [];

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="glass-card space-y-6"
        >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-white/5 pb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-teal-500/10 border border-teal-500/20">
                        <Shield className="w-5 h-5 text-teal-400" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-white">Statistical Diagnostics</h2>
                        <p className="text-xs text-slate-500">Rigorous Assumption Checks (Positivity, Unconfoundedness)</p>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    {overallIcon}
                    <div className="text-right">
                        <div className="text-sm font-semibold text-white">Overall Status</div>
                        <StatusBadge status={s.status} />
                    </div>
                </div>
            </div>

            {/* 2x2 Grid for Core Diagnostics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

                {/* 1. Placebo Test */}
                <DiagCard
                    title="Placebo Treatment Test"
                    desc="Randomly shuffled treatment should yield zero effect."
                    status={s.placebo_test.status}
                    icon={<CheckCircle2 className="w-4 h-4 text-blue-400" />}
                >
                    <div className="grid grid-cols-2 gap-4 mt-2">
                        <MetricBox label="Mean Effect (Target ≈ 0)" value={(s.placebo_test.mean_effect || s.placebo_test.null_mean || 0).toFixed(4)} />
                        <MetricBox label="P-value (> 0.05)" value={s.placebo_test.p_value.toFixed(3)} />
                    </div>
                </DiagCard>

                {/* 2. Random Common Cause */}
                <DiagCard
                    title="Random Common Cause"
                    desc="Adding random confounder should not change ATE significantly."
                    status={s.random_common_cause.status}
                    icon={<Layers className="w-4 h-4 text-purple-400" />}
                >
                    <div className="grid grid-cols-2 gap-4 mt-2">
                        <MetricBox label="Stability" value={`${(s.random_common_cause.stability * 100).toFixed(1)}%`} />
                        <MetricBox label="New Mean Effect" value={s.random_common_cause.mean_effect.toFixed(4)} />
                    </div>
                </DiagCard>

                {/* 3. E-value */}
                {s.e_value && (
                    <DiagCard
                        title="E-value (Robustness)"
                        desc={s.e_value.interpretation}
                        status={s.e_value.status}
                        icon={<AlertTriangle className="w-4 h-4 text-yellow-400" />}
                    >
                        <div className="grid grid-cols-2 gap-4 mt-2 mb-2">
                            <MetricBox label="Point Estimate" value={s.e_value.point.toFixed(2)} />
                            <MetricBox label="CI Lower Bound" value={s.e_value.ci_bound.toFixed(2)} />
                        </div>
                        {/* Gauge for E-value */}
                        <div className="w-full bg-slate-700 h-2 rounded-full mt-2 overflow-hidden">
                            <div
                                className="bg-yellow-400 h-full rounded-full"
                                style={{ width: `${Math.min((s.e_value.point / 3) * 100, 100)}%` }}
                            />
                        </div>
                        <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                            <span>1.0 (Weak)</span>
                            <span>2.0 (Moderate)</span>
                            <span>3.0+ (Strong)</span>
                        </div>
                    </DiagCard>
                )}

                {/* 4. Overlap (Positivity) with Histogram */}
                {s.overlap && (
                    <div className="col-span-1 lg:col-span-1 p-4 rounded-xl bg-white/5 border border-white/5 space-y-3">
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                                <BarChart3 className="w-4 h-4 text-indigo-400" />
                                <span className="text-sm font-semibold text-white">Overlap (Positivity)</span>
                            </div>
                            <StatusBadge status={s.overlap.status} />
                        </div>
                        <p className="text-xs text-slate-400">{s.overlap.interpretation}</p>

                        {/* Overlap Metircs */}
                        <div className="grid grid-cols-3 gap-2 mb-2">
                            <MetricBox label="Overlap Score" value={s.overlap.overlap_score.toFixed(2)} />
                            {s.overlap.pct_extreme_weights !== undefined && (
                                <MetricBox label="Extreme Wt%" value={`${s.overlap.pct_extreme_weights}%`} />
                            )}
                        </div>

                        {/* Propensity Score Histogram */}
                        {overlapData.length > 0 && (
                            <div className="h-32 w-full mt-2">
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={overlapData}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" opacity={0.2} />
                                        <XAxis dataKey="range" tick={{ fontSize: 8 }} interval={1} stroke="#9CA3AF" />
                                        <YAxis hide />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', fontSize: '12px' }}
                                            itemStyle={{ color: '#F3F4F6' }}
                                        />
                                        <Legend iconSize={8} wrapperStyle={{ fontSize: '10px' }} />
                                        <Bar dataKey="Treated" fill="#818CF8" radius={[4, 4, 0, 0]} opacity={0.6} />
                                        <Bar dataKey="Control" fill="#9CA3AF" radius={[4, 4, 0, 0]} opacity={0.6} />
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        )}
                    </div>
                )}

            </div>

            {/* GATES Heterogeneity (Wide) */}
            {s.gates && s.gates.groups.length > 0 && (
                <div className="p-4 rounded-xl bg-white/5 border border-white/5 space-y-3">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <BarChart3 className="w-4 h-4 text-pink-400" />
                            <span className="text-sm font-semibold text-white">GATES Heterogeneity Analysis</span>
                        </div>
                        <div className="flex gap-2">
                            <span className="text-xs text-slate-400 self-center">F-stat: {s.gates.f_statistic.toFixed(2)}</span>
                            <StatusBadge status={s.gates.status} />
                        </div>
                    </div>
                    <p className="text-xs text-slate-400">{s.gates.heterogeneity}</p>

                    {/* GATES Chart - Reusing previous logic or new visualization */}
                    <div className="grid grid-cols-4 gap-2 h-24 items-end">
                        {s.gates.groups.map(g => {
                            const maxVal = Math.max(...s.gates!.groups.map(x => Math.abs(x.mean_cate)));
                            const height = (Math.abs(g.mean_cate) / maxVal) * 100;
                            return (
                                <div key={g.group_id} className="flex flex-col items-center gap-1 group">
                                    <div className="text-[10px] font-mono text-white opacity-0 group-hover:opacity-100 transition-opacity">
                                        {g.mean_cate.toFixed(3)}
                                    </div>
                                    <div className="w-full bg-slate-800 rounded-t-md relative h-16 flex items-end justify-center overflow-hidden">
                                        <div
                                            className={`w-full transition-all duration-500 ${g.mean_cate > 0 ? 'bg-pink-500/60' : 'bg-slate-500/60'}`}
                                            style={{ height: `${height}%` }}
                                        />
                                    </div>
                                    <span className="text-[10px] text-slate-400">{g.label}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

        </motion.div>
    );
}

// Sub-components
function DiagCard({ title, desc, status, icon, children }: any) {
    return (
        <div className="p-4 rounded-xl bg-white/5 border border-white/5 space-y-2">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    {icon}
                    <span className="text-sm font-semibold text-white">{title}</span>
                </div>
                <StatusBadge status={status} />
            </div>
            <p className="text-xs text-slate-400 leading-tight">{desc}</p>
            {children}
        </div>
    );
}

function MetricBox({ label, value }: { label: string, value: string }) {
    return (
        <div className="bg-white/5 rounded p-2 text-center">
            <div className="text-[10px] text-slate-500 uppercase">{label}</div>
            <div className="text-sm font-bold text-white font-mono">{value}</div>
        </div>
    );
}
