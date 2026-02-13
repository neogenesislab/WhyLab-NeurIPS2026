"use client";

import { motion } from "framer-motion";
import { ArrowRight, BarChart3, BrainCircuit, GitFork } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 md:p-24 relative overflow-hidden">

      {/* Background Glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-brand-500/20 rounded-full blur-3xl opacity-30 pointer-events-none" />

      {/* Hero Section */}
      <section className="z-10 w-full max-w-5xl text-center space-y-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="space-y-4"
        >
          <div className="inline-block px-4 py-1.5 rounded-full border border-brand-500/30 bg-brand-500/10 text-brand-300 text-sm font-medium mb-4 backdrop-blur-md">
            v0.1.0 Public Preview
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white drop-shadow-lg">
            AI가 알려주는 <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-accent-cyan text-glow">
              진짜 원인, 진짜 의사결정
            </span>
          </h1>
          <p className="text-lg md:text-xl text-slate-300 max-w-2xl mx-auto leading-relaxed">
            상관관계와 인과관계를 AI가 자동으로 분리합니다. <span className="text-brand-300 font-semibold">AutoML + Double Machine Learning</span>으로 데이터 뒤에 숨겨진 진짜 원인을 밝히고, 액션 가능한 인사이트를 제공합니다.
          </p>
        </motion.div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="flex flex-col sm:flex-row gap-4 justify-center items-center mt-8"
        >
          <Link
            href="/dashboard"
            className="group px-8 py-4 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-bold text-lg transition-all shadow-lg shadow-brand-500/20 flex items-center gap-2"
          >
            Start Analysis
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>
          <Link
            href="https://github.com"
            target="_blank"
            className="px-8 py-4 rounded-xl border border-white/10 hover:bg-white/5 text-slate-300 transition-all backdrop-blur-sm"
          >
            GitHub Repository
          </Link>
        </motion.div>
      </section>

      {/* Feature Cards */}
      <section className="z-10 grid grid-cols-1 md:grid-cols-3 gap-6 mt-24 w-full max-w-6xl">
        <FeatureCard
          icon={<BrainCircuit className="w-8 h-8 text-brand-400" />}
          title="AI AutoML 인과추론"
          desc="LinearDML, CausalForest, DRLearner 중 AI가 최적 모델을 자동 선택합니다. 결과는 자연어로 해석됩니다."
          delay={0.6}
        />
        <FeatureCard
          icon={<GitFork className="w-8 h-8 text-accent-cyan" />}
          title="인과 그래프(DAG)"
          desc="변수 간의 인과 관계를 시각화하고, 교란 변수와 처치/결과 변수를 직관적으로 파악합니다."
          delay={0.7}
        />
        <FeatureCard
          icon={<BarChart3 className="w-8 h-8 text-accent-pink" />}
          title="What-If 의사결정"
          desc="'신용 한도를 올리면 연체율이 줄어들까?' 같은 반사실적 시나리오를 시뮬레이션하여 데이터 기반 의사결정을 지원합니다."
          delay={0.8}
        />
      </section>

      {/* Footer */}
      <footer className="absolute bottom-6 text-slate-500 text-sm">
        © 2026 WhyLab Project. Built with Next.js & EconML.
      </footer>
    </main>
  );
}

function FeatureCard({ icon, title, desc, delay }: { icon: React.ReactNode; title: string; desc: string; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay, duration: 0.6 }}
      className="glass-card flex flex-col items-start gap-4 hover:translate-y-[-5px]"
    >
      <div className="p-3 rounded-lg bg-white/5 border border-white/10">
        {icon}
      </div>
      <h3 className="text-xl font-bold text-white">{title}</h3>
      <p className="text-slate-400 leading-relaxed text-sm">{desc}</p>
    </motion.div>
  );
}
