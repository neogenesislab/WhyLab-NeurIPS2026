import { Suspense } from "react";
import Sidebar from "@/components/Sidebar";

export default function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <div className="flex min-h-screen bg-dots-pattern">
            <Suspense fallback={null}>
                <Sidebar />
            </Suspense>
            <div className="flex-1 ml-0 lg:ml-64 p-4 lg:p-8 overflow-y-auto">
                {/* Top Bar */}
                <header className="flex justify-between items-center mb-8 pl-10 lg:pl-0">
                    <div className="text-sm breadcrumbs text-slate-400">
                        Dashboard <span className="mx-2">/</span> Overview
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs text-white">Guest</div>
                    </div>
                </header>
                <main>
                    {children}
                </main>
            </div>
        </div>
    );
}
