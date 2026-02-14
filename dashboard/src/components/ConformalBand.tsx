'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
    ComposedChart,
    Line,
    Area,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine
} from 'recharts';

interface ConformalBandProps {
    data: any; // 추후 types/index.ts의 DashboardData 사용
}

export default function ConformalBand({ data }: ConformalBandProps) {
    const { conformal_results, scatter_data } = data || {};

    // 데이터 유효성 검사
    if (!conformal_results || !scatter_data || !scatter_data.estimated_cate) {
        return null; // 데이터가 없으면 렌더링하지 않음 (또는 비활성 상태 표시)
    }

    // 1. CATE 데이터 추출 및 정렬 (Sorted CATE Plot)
    const cateData = scatter_data.estimated_cate as number[];
    const meanWidth = (conformal_results.ci_upper_mean - conformal_results.ci_lower_mean) || 0.15; // 기본값 안전장치
    const coverage = conformal_results.coverage || 0.95;

    // 정렬된 데이터 생성
    const chartData = cateData
        .map((cate, i) => ({ cate })) // 객체 변환
        .sort((a, b) => a.cate - b.cate) // 오름차순 정렬
        .map((item, index) => {
            const percent = (index / (cateData.length - 1)) * 100;
            // Split Conformal Prediction은 적응형이 아니므로 평균 폭을 사용 (CQR이라면 개별 폭 사용 가능)
            // 여기서는 시각화를 위해 평균 폭을 중심으로 Band를 그림
            return {
                percent,
                cate: item.cate,
                // Band: [Lower, Upper]
                ci_lower: item.cate - (meanWidth / 2),
                ci_upper: item.cate + (meanWidth / 2),
                // Recharts Area용 range
                ci_range: [item.cate - (meanWidth / 2), item.cate + (meanWidth / 2)]
            };
        })
        // 성능을 위해 데이터 샘플링 (최대 200포인트)
        .filter((_, i, arr) => i % Math.ceil(arr.length / 200) === 0);

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 flex flex-col h-full"
        >
            <div className="flex items-center justify-between mb-2">
                <div>
                    <h2 className="text-lg font-bold text-gray-900 dark:text-white">
                        Conformal Prediction Intervals
                    </h2>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                        Sorted CATE & {(coverage * 100).toFixed(0)}% Confidence Band
                    </p>
                </div>
                <div className="text-right">
                    <div className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">
                        {meanWidth.toFixed(4)}
                    </div>
                    <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold">
                        Avg. CI Width
                    </div>
                </div>
            </div>

            <div className="flex-1 h-64 min-h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" opacity={0.5} />
                        <XAxis
                            dataKey="percent"
                            type="number"
                            unit="%"
                            tickLine={false}
                            axisLine={false}
                            tick={{ fontSize: 10 }}
                            label={{ value: 'Percentile', position: 'insideBottom', offset: -10, fontSize: 10 }}
                        />
                        <YAxis
                            tickLine={false}
                            axisLine={false}
                            tick={{ fontSize: 10 }}
                            domain={['auto', 'auto']}
                        />
                        <Tooltip
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                            itemStyle={{ fontSize: '12px' }}
                            labelFormatter={(v) => `Top ${v.toFixed(0)}%`}
                            formatter={(value: any, name: any) => [
                                typeof value === 'number' ? value.toFixed(4) : value,
                                name === 'cate' ? 'CATE Estimate' : name
                            ]}
                        />

                        {/* 0 기준선 */}
                        <ReferenceLine y={0} stroke="#9CA3AF" strokeDasharray="3 3" />

                        {/* 신뢰구간 (Band) */}
                        <Area
                            type="monotone"
                            dataKey="ci_range"
                            stroke="none"
                            fill="#818CF8"
                            fillOpacity={0.2}
                            name="Confidence Interval"
                        />

                        {/* CATE 추정값 (Line) */}
                        <Line
                            type="monotone"
                            dataKey="cate"
                            stroke="#4338CA"
                            strokeWidth={2}
                            dot={false}
                            name="CATE"
                            activeDot={{ r: 4 }}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-2 text-xs text-gray-400 text-center">
                * Sorted CATE Plot with Split Conformal Intervals (Average Width)
            </div>
        </motion.div>
    );
}
