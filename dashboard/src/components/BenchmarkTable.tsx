'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Trophy, BarChart3, AlertCircle, Database, HelpCircle } from 'lucide-react';

interface BenchmarkTableProps {
    data: any; // 추후 types/index.ts의 DashboardData 사용
}

export default function BenchmarkTable({ data }: BenchmarkTableProps) {
    const benchmarks = data?.benchmark_results || {};
    const datasetNames = Object.keys(benchmarks);
    const hasData = datasetNames.length > 0;

    const [activeTab, setActiveTab] = useState(hasData ? datasetNames[0] : 'guide');

    // 가이드 탭 렌더링
    const renderGuide = () => (
        <div className="p-6 bg-gray-50 dark:bg-gray-700/50 rounded-xl border border-dashed border-gray-300 dark:border-gray-600 text-center">
            <div className="w-12 h-12 bg-gray-200 dark:bg-gray-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <Database className="w-6 h-6 text-gray-500 dark:text-gray-400" />
            </div>
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">
                벤치마크 결과 없음
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-4 max-w-md mx-auto">
                학술 데이터셋(IHDP, ACIC, Jobs)에 대한 WhyLab 메타러너의 성능 평가 결과가 없습니다.
                터미널에서 아래 명령어를 실행하여 벤치마크를 수행하세요.
            </p>
            <div className="bg-gray-900 text-gray-100 p-3 rounded-lg font-mono text-sm inline-block text-left relative group">
                <code className="block mb-1 text-xs text-gray-400"># 약 5~10분 소요 (GPU 권장)</code>
                <code className="select-all">python -m engine --benchmark ihdp acic</code>
                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                    <span className="text-xs bg-gray-700 px-2 py-1 rounded">Copy</span>
                </div>
            </div>
        </div>
    );

    // 테이블 렌더링
    const renderTable = (dataset: string) => {
        const results = benchmarks[dataset];
        if (!results) return null;

        // pehe_mean 기준으로 정렬
        const sortedMethods = Object.keys(results).sort(
            (a, b) => results[a].pehe_mean - results[b].pehe_mean
        );

        const bestMethod = sortedMethods[0];

        return (
            <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="border-b border-gray-100 dark:border-gray-700">
                            <th className="py-3 px-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">Method</th>
                            <th className="py-3 px-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">√PEHE (Error)</th>
                            <th className="py-3 px-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">ATE Bias</th>
                            <th className="py-3 px-4 text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                        {sortedMethods.map((method) => {
                            const r = results[method];
                            const isBest = method === bestMethod;

                            return (
                                <tr key={method} className={`hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors ${isBest ? 'bg-green-50/50 dark:bg-green-900/10' : ''
                                    }`}>
                                    <td className="py-3 px-4 font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
                                        {method}
                                        {isBest && <Trophy className="w-4 h-4 text-yellow-500" />}
                                        {method === 'Ensemble' && <span className="text-xs bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded">WhyLab</span>}
                                    </td>
                                    <td className="py-3 px-4 text-gray-700 dark:text-gray-300 font-mono text-sm">
                                        {r.pehe_mean.toFixed(4)} <span className="text-gray-400 text-xs">±{r.pehe_std.toFixed(4)}</span>
                                    </td>
                                    <td className="py-3 px-4 text-gray-700 dark:text-gray-300 font-mono text-sm">
                                        {r.ate_bias_mean.toFixed(4)} <span className="text-gray-400 text-xs">±{r.ate_bias_std.toFixed(4)}</span>
                                    </td>
                                    <td className="py-3 px-4">
                                        {isBest ? (
                                            <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                                                Best Performance
                                            </span>
                                        ) : (
                                            <span className="text-gray-400 text-xs">-</span>
                                        )}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>

                {/* 설명 */}
                <div className="mt-4 flex items-start gap-2 text-xs text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50 p-3 rounded">
                    <HelpCircle className="w-4 h-4 mt-0.5 shrink-0" />
                    <p>
                        <strong>√PEHE</strong>: 개별 처치 효과(CATE) 추정 오차의 제곱근 평균 (낮을수록 좋음).<br />
                        <strong>ATE Bias</strong>: 평균 처치 효과 추정 편향 (0에 가까울수록 좋음).
                        결과는 {hasData ? '10' : 'N'}회 반복 실험의 평균입니다.
                    </p>
                </div>
            </div>
        );
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700 p-6 flex flex-col h-full"
        >
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-indigo-500" />
                    <h2 className="text-lg font-bold text-gray-900 dark:text-white">Benchmark Performance</h2>
                </div>

                {/* 탭 버튼 */}
                {hasData && (
                    <div className="flex bg-gray-100 dark:bg-gray-700 p-1 rounded-lg">
                        {datasetNames.map(ds => (
                            <button
                                key={ds}
                                onClick={() => setActiveTab(ds)}
                                className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${activeTab === ds
                                        ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                                        : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'
                                    }`}
                            >
                                {ds.toUpperCase()}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            <div className="flex-1">
                {hasData ? renderTable(activeTab) : renderGuide()}
            </div>
        </motion.div>
    );
}
