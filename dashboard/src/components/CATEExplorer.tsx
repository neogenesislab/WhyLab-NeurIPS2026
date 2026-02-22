'use client';
import React, { useMemo } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts';
import { Target, TrendingUp, AlertTriangle, Users } from 'lucide-react';

/* ─── 타입 ─── */
interface SegmentData {
    name: string;
    dimension: string;
    n: number;
    cate_mean: number;
    cate_ci_lower: number;
    cate_ci_upper: number;
}

interface CATEDistribution {
    mean: number;
    std: number;
    min: number;
    max: number;
    histogram?: { bin_edges: number[]; counts: number[] };
}

interface CATEExplorerProps {
    distribution?: CATEDistribution;
    segments?: SegmentData[];
}

/* ─── 색상 ─── */
const getBarColor = (value: number): string => {
    if (value > 0.05) return '#10b981';    // 강한 긍정 효과 → 초록
    if (value > 0) return '#6ee7b7';       // 약한 긍정 → 연초록
    if (value > -0.05) return '#fbbf24';   // 약한 부정 → 노랑
    return '#ef4444';                       // 강한 부정 → 빨강
};

/* ─── 메인 컴포넌트 ─── */
export default function CATEExplorer({
    distribution,
    segments = [],
}: CATEExplorerProps) {
    // 세그먼트를 CATE 내림차순 정렬
    const sortedSegments = useMemo(
        () => [...segments].sort((a, b) => b.cate_mean - a.cate_mean),
        [segments],
    );

    // 최고 ROI 세그먼트
    const bestSegment = sortedSegments[0];
    // 최저 ROI 세그먼트
    const worstSegment = sortedSegments[sortedSegments.length - 1];

    // 타겟팅 추천 계산
    const topSegments = sortedSegments.filter(s => s.cate_mean > 0);
    const topUserCount = topSegments.reduce((sum, s) => sum + s.n, 0);
    const totalUserCount = segments.reduce((sum, s) => sum + s.n, 0);
    const targetingRatio = totalUserCount > 0
        ? ((topUserCount / totalUserCount) * 100).toFixed(0)
        : '0';

    return (
        <div className="bg-gray-900/70 backdrop-blur-sm rounded-2xl border border-purple-500/20 p-6">
            {/* 헤더 */}
            <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-purple-500/20 rounded-lg">
                    <Target className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                    <h3 className="text-lg font-bold text-white">
                        CATE Explorer — Who Benefits Most?
                    </h3>
                    <p className="text-sm text-gray-400">
                        Individual treatment effect (CATE) distribution and targeting recommendations
                    </p>
                </div>
            </div>

            {/* 요약 카드 */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
                <SummaryCard
                    icon={<TrendingUp className="w-4 h-4" />}
                    label="Avg. CATE"
                    value={distribution?.mean?.toFixed(4) ?? 'N/A'}
                    color="text-emerald-400"
                />
                <SummaryCard
                    icon={<AlertTriangle className="w-4 h-4" />}
                    label="Std. Dev."
                    value={distribution?.std?.toFixed(4) ?? 'N/A'}
                    color="text-amber-400"
                />
                <SummaryCard
                    icon={<Users className="w-4 h-4" />}
                    label="Effective Users"
                    value={`${targetingRatio}%`}
                    color="text-blue-400"
                />
                <SummaryCard
                    icon={<Target className="w-4 h-4" />}
                    label="Best Segment"
                    value={bestSegment?.name ?? 'N/A'}
                    color="text-purple-400"
                />
            </div>

            {/* 세그먼트별 CATE 차트 */}
            {sortedSegments.length > 0 && (
                <div className="mb-6">
                    <h4 className="text-sm font-semibold text-gray-300 mb-3">
                        Treatment Effect by Segment (CATE)
                    </h4>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                                data={sortedSegments.slice(0, 10)}
                                margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
                                layout="vertical"
                            >
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                                <YAxis
                                    type="category"
                                    dataKey="name"
                                    width={120}
                                    tick={{ fill: '#d1d5db', fontSize: 11 }}
                                />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: '#1f2937',
                                        border: '1px solid #6b21a8',
                                        borderRadius: '8px',
                                        color: '#fff',
                                    }}
                                    formatter={((value: number | string) => [
                                        typeof value === 'number' ? value.toFixed(4) : value,
                                        'CATE',
                                    ]) as any}
                                />
                                <ReferenceLine x={0} stroke="#6b7280" strokeDasharray="3 3" />
                                <Bar dataKey="cate_mean" radius={[0, 4, 4, 0]}>
                                    {sortedSegments.slice(0, 10).map((entry, index) => (
                                        <Cell key={index} fill={getBarColor(entry.cate_mean)} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* 타겟팅 추천 */}
            {bestSegment && worstSegment && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <TrendingUp className="w-4 h-4 text-emerald-400" />
                            <span className="text-sm font-semibold text-emerald-400">
                                Focus Target Recommendation
                            </span>
                        </div>
                        <p className="text-white font-bold">{bestSegment.name}</p>
                        <p className="text-sm text-gray-400">
                            CATE: {bestSegment.cate_mean.toFixed(4)} |{' '}
                            {bestSegment.n.toLocaleString()} users
                        </p>
                        <p className="text-xs text-emerald-300 mt-1">
                            Focusing on this group can maximize ROI
                        </p>
                    </div>

                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
                        <div className="flex items-center gap-2 mb-2">
                            <AlertTriangle className="w-4 h-4 text-red-400" />
                            <span className="text-sm font-semibold text-red-400">
                                Caution
                            </span>
                        </div>
                        <p className="text-white font-bold">{worstSegment.name}</p>
                        <p className="text-sm text-gray-400">
                            CATE: {worstSegment.cate_mean.toFixed(4)} |{' '}
                            {worstSegment.n.toLocaleString()} users
                        </p>
                        <p className="text-xs text-red-300 mt-1">
                            Minimal or negative effect — consider budget reduction
                        </p>
                    </div>
                </div>
            )}

            {/* 빈 상태 */}
            {sortedSegments.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                    <Target className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No segment data available.</p>
                    <p className="text-xs">Please run the pipeline first.</p>
                </div>
            )}
        </div>
    );
}

/* ─── 서브 컴포넌트 ─── */
function SummaryCard({
    icon,
    label,
    value,
    color,
}: {
    icon: React.ReactNode;
    label: string;
    value: string;
    color: string;
}) {
    return (
        <div className="bg-gray-800/50 rounded-xl p-3 border border-gray-700/50">
            <div className={`flex items-center gap-1.5 mb-1 ${color}`}>
                {icon}
                <span className="text-xs">{label}</span>
            </div>
            <p className="text-white font-bold text-lg truncate">{value}</p>
        </div>
    );
}
