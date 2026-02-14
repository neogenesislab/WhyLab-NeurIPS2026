"use client";

import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface WhatIfSimulatorProps {
    baseValue: number;    // 현재 평균 Treatment 값
    coefficient: number;  // 회귀 계수 (Slope)
    intercept: number;    // 절편
    treatmentName: string;
    outcomeName: string;
}

export default function WhatIfSimulator({
    baseValue = 100,
    coefficient = -0.5,
    intercept = 50,
    treatmentName = "Credit Limit",
    outcomeName = "Default Rate"
}: WhatIfSimulatorProps) {
    const [treatmentDelta, setTreatmentDelta] = useState(0);

    // 선형 모델 시뮬레이션: Y = aX + b
    const currentTreatment = baseValue;
    const simulatedTreatment = baseValue + treatmentDelta;

    const currentOutcome = (currentTreatment * coefficient) + intercept;
    const simulatedOutcome = (simulatedTreatment * coefficient) + intercept;

    const data = [
        {
            name: 'Current',
            [outcomeName]: Math.max(0, currentOutcome), // 음수 방지
            fill: '#94a3b8'
        },
        {
            name: 'Simulated',
            [outcomeName]: Math.max(0, simulatedOutcome),
            fill: '#8b5cf6' // Brand Color
        }
    ];

    return (
        <div className="glass-card flex flex-col gap-6 h-full">
            <div>
                <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <span className="w-2 h-8 bg-accent-pink rounded-full"></span>
                    What-If Simulator
                </h3>
                <p className="text-slate-400 text-sm mt-1">
                    &quot;{treatmentName}&quot;를 변경하면 &quot;{outcomeName}&quot;이 어떻게 변할까요?
                </p>
            </div>

            {/* Controller */}
            <div className="space-y-4 p-4 bg-white/5 rounded-xl border border-white/10">
                <div className="flex justify-between text-sm text-slate-300">
                    <span>Change {treatmentName}</span>
                    <span className="font-mono text-brand-300">{treatmentDelta > 0 ? '+' : ''}{treatmentDelta}</span>
                </div>
                <input
                    type="range"
                    min="-50"
                    max="50"
                    step="5"
                    value={treatmentDelta}
                    onChange={(e) => setTreatmentDelta(Number(e.target.value))}
                    className="w-full accent-brand-500 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-slate-500">
                    <span>-50%</span>
                    <span>Current</span>
                    <span>+50%</span>
                </div>
            </div>

            {/* Visualization */}
            <div className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                        <XAxis type="number" stroke="#94a3b8" />
                        <YAxis dataKey="name" type="category" stroke="#94a3b8" width={80} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
                            cursor={{ fill: 'transparent' }}
                        />
                        <Bar dataKey={outcomeName} barSize={30} radius={[0, 4, 4, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Insight */}
            <div className="p-3 bg-brand-500/10 border border-brand-500/20 rounded-lg">
                <p className="text-sm text-brand-200">
                    💡 <strong>Insight:</strong> {treatmentName}를
                    <span className="font-bold mx-1 text-white">{treatmentDelta}</span>만큼 조정하면,
                    예상되는 {outcomeName}는
                    <span className="font-bold mx-1 text-white">{(simulatedOutcome - currentOutcome).toFixed(2)}</span>
                    만큼 변합니다.
                </p>
            </div>
        </div>
    );
}
