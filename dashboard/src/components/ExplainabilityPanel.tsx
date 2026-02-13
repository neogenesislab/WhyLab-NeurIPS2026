"use client";

import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { CausalAnalysisResult } from "@/types";
import { motion } from 'framer-motion';

const COLORS = ['#06b6d4', '#22d3ee', '#67e8f9', '#a5f3fc', '#cffafe', '#e0f2fe', '#bae6fd'];

export default function ExplainabilityPanel({ data }: { data: CausalAnalysisResult }) {
    const explainability = data.explainability;

    if (!explainability) {
        return (
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="card"
            >
                <h2 className="text-xl font-bold mb-4">🔍 Explainability (SHAP)</h2>
                <p className="text-gray-400 text-sm">
                    파이프라인에 ExplainCell(SHAP)을 포함하여 재실행하면 Feature Importance가 표시됩니다.
                </p>
            </motion.div>
        );
    }

    const fiData = explainability.feature_importance.map((item) => ({
        feature: item.feature,
        importance: Number((item.importance * 100).toFixed(2)),
    }));

    const cfData = explainability.counterfactuals;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card space-y-6"
        >
            <h2 className="text-xl font-bold">🔍 Explainability (SHAP)</h2>
            <p className="text-gray-400 text-sm">
                각 변수가 CATE(이질적 치료 효과)에 기여하는 정도를 SHAP으로 분해
            </p>

            {/* Feature Importance 수평 바 차트 */}
            <div>
                <h3 className="text-sm font-semibold text-cyan-400 mb-3">
                    Feature Importance (×10⁻²)
                </h3>
                <ResponsiveContainer width="100%" height={fiData.length * 45 + 20}>
                    <BarChart
                        data={fiData}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                        <YAxis
                            dataKey="feature"
                            type="category"
                            tick={{ fill: '#e2e8f0', fontSize: 12 }}
                            width={80}
                        />
                        <Tooltip
                            contentStyle={{
                                background: 'rgba(15, 23, 42, 0.95)',
                                border: '1px solid rgba(6, 182, 212, 0.3)',
                                borderRadius: '8px',
                                color: '#e2e8f0',
                                fontSize: '12px',
                            }}
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            formatter={(value: any) => [`${Number(value).toFixed(2)}`, 'SHAP 기여도']}
                        />
                        <Bar dataKey="importance" radius={[0, 6, 6, 0]}>
                            {fiData.map((_, index) => (
                                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Counterfactual 시뮬레이션 */}
            {cfData.length > 0 && (
                <div>
                    <h3 className="text-sm font-semibold text-cyan-400 mb-3">
                        반사실 시뮬레이션 (What-If)
                    </h3>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-gray-400 border-b border-gray-700">
                                    <th className="text-left py-2 px-3">User</th>
                                    <th className="text-right py-2 px-3">원본 CATE</th>
                                    <th className="text-right py-2 px-3">반사실 CATE</th>
                                    <th className="text-right py-2 px-3">변화량</th>
                                    <th className="text-left py-2 px-3">시나리오</th>
                                </tr>
                            </thead>
                            <tbody>
                                {cfData.slice(0, 5).map((cf, i) => (
                                    <tr key={i} className="border-b border-gray-800 hover:bg-white/5 transition-colors">
                                        <td className="py-2 px-3 text-gray-300">#{cf.user_id}</td>
                                        <td className="py-2 px-3 text-right font-mono text-cyan-300">
                                            {cf.original_cate.toFixed(4)}
                                        </td>
                                        <td className="py-2 px-3 text-right font-mono text-amber-300">
                                            {cf.counterfactual_cate.toFixed(4)}
                                        </td>
                                        <td className={`py-2 px-3 text-right font-mono font-bold ${cf.diff > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                                            {cf.diff > 0 ? '+' : ''}{cf.diff.toFixed(4)}
                                        </td>
                                        <td className="py-2 px-3 text-gray-400 text-xs">
                                            {cf.description}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </motion.div>
    );
}
