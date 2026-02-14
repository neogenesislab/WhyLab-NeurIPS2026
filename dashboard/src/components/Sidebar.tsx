"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { LayoutDashboard, GitFork, BarChart3, Settings, HelpCircle, LogOut, CreditCard, Megaphone, Menu, X } from "lucide-react";
import { clsx } from "clsx";

const menuItems = [
    { href: "/dashboard", icon: LayoutDashboard, label: "Overview" },
    { href: "/dashboard/causal-graph", icon: GitFork, label: "Causal Graph" },
    { href: "/dashboard/simulator", icon: BarChart3, label: "Simulation" },
    { href: "/dashboard/settings", icon: Settings, label: "Settings" },
];

export default function Sidebar() {
    const pathname = usePathname();
    const searchParams = useSearchParams();
    const currentScenario = searchParams.get("scenario") || "A";
    const [mobileOpen, setMobileOpen] = useState(false);

    // 모바일에서 경로 변경 시 자동 닫기
    // eslint-disable-next-line react-hooks/exhaustive-deps
    useEffect(() => { setMobileOpen(false); }, [pathname]);

    return (
        <>
            {/* 모바일 햄버거 버튼 */}
            <button
                onClick={() => setMobileOpen(true)}
                className="lg:hidden fixed top-4 left-4 z-50 p-2 rounded-lg bg-slate-800/80 backdrop-blur-md border border-white/10 text-white"
                aria-label="메뉴 열기"
            >
                <Menu className="w-5 h-5" />
            </button>

            {/* 모바일 오버레이 */}
            {mobileOpen && (
                <div className="lg:hidden fixed inset-0 bg-black/60 z-40" onClick={() => setMobileOpen(false)} />
            )}

            {/* 사이드바 */}
            <aside className={clsx(
                "fixed left-0 top-0 h-screen w-64 bg-slate-900/80 backdrop-blur-xl border-r border-white/5 flex flex-col items-center py-6 z-50 transition-transform duration-300",
                mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
            )}>
                {/* 모바일 닫기 버튼 */}
                <button
                    onClick={() => setMobileOpen(false)}
                    className="lg:hidden absolute top-4 right-4 p-1 text-slate-400 hover:text-white"
                    aria-label="메뉴 닫기"
                >
                    <X className="w-5 h-5" />
                </button>

                {/* Logo */}
                <div className="mb-8 w-full px-6">
                    <Link href="/" className="flex items-center gap-2 group">
                        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-accent-pink flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-brand-500/20 group-hover:scale-105 transition-transform">
                            W
                        </div>
                        <span className="text-xl font-bold text-white tracking-tight group-hover:text-glow transition-all">WhyLab</span>
                    </Link>
                </div>

                {/* 시나리오 토글 */}
                <div className="w-full px-4 mb-6">
                    <p className="text-xs text-slate-500 uppercase tracking-wider mb-2 px-2">Scenario</p>
                    <div className="flex rounded-lg bg-slate-800/60 p-1 border border-white/5">
                        <Link
                            href="/dashboard?scenario=A"
                            className={clsx(
                                "flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-xs font-medium transition-all",
                                currentScenario === "A"
                                    ? "bg-brand-500/20 text-brand-300 shadow-sm"
                                    : "text-slate-500 hover:text-slate-300"
                            )}
                        >
                            <CreditCard className="w-3.5 h-3.5" />
                            신용한도
                        </Link>
                        <Link
                            href="/dashboard?scenario=B"
                            className={clsx(
                                "flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-xs font-medium transition-all",
                                currentScenario === "B"
                                    ? "bg-accent-cyan/20 text-accent-cyan shadow-sm"
                                    : "text-slate-500 hover:text-slate-300"
                            )}
                        >
                            <Megaphone className="w-3.5 h-3.5" />
                            쿠폰
                        </Link>
                    </div>
                </div>

                {/* Menu */}
                <nav className="flex-1 w-full px-4 space-y-1">
                    {menuItems.map((item) => {
                        const isActive = pathname === item.href;
                        return (
                            <Link
                                key={item.href}
                                href={`${item.href}?scenario=${currentScenario}`}
                                className={clsx(
                                    "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200",
                                    isActive
                                        ? "bg-brand-500/20 text-brand-300 border border-brand-500/20 shadow-[0_0_15px_rgba(139,92,246,0.1)]"
                                        : "text-slate-400 hover:text-white hover:bg-white/5"
                                )}
                            >
                                <item.icon className={clsx("w-5 h-5", isActive ? "text-brand-400" : "text-slate-500")} />
                                <span className="font-medium text-sm">{item.label}</span>
                            </Link>
                        );
                    })}
                </nav>

                {/* Footer Actions */}
                <div className="w-full px-4 space-y-2 mt-auto">
                    <Link
                        href="https://github.com/Yesol-Pilot/WhyLab"
                        target="_blank"
                        className="w-full flex items-center gap-3 px-4 py-3 text-slate-400 hover:text-white hover:bg-white/5 rounded-xl transition-colors"
                    >
                        <HelpCircle className="w-5 h-5" />
                        <span className="font-medium text-sm">Documentation</span>
                    </Link>
                    <div className="pt-4 border-t border-white/5">
                        <Link href="/" className="flex items-center gap-3 px-4 py-3 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-xl transition-colors">
                            <LogOut className="w-5 h-5" />
                            <span className="font-medium text-sm">Exit Demo</span>
                        </Link>
                    </div>
                </div>
            </aside>
        </>
    );
}
