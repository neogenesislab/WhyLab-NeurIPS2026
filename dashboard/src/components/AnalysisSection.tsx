"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
    title: string;
    description?: string;
    defaultOpen?: boolean;
    children: React.ReactNode;
}

export default function AnalysisSection({ title, description, defaultOpen = false, children }: Props) {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div className="border border-white/5 rounded-xl bg-slate-900/50 overflow-hidden">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between p-4 hover:bg-white/5 transition-colors text-left"
            >
                <div>
                    <h3 className="text-lg font-semibold text-white">{title}</h3>
                    {description && <p className="text-xs text-slate-400 mt-1">{description}</p>}
                </div>
                {isOpen ? <ChevronUp className="text-slate-400" /> : <ChevronDown className="text-slate-400" />}
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="border-t border-white/5"
                    >
                        <div className="p-4">{children}</div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
