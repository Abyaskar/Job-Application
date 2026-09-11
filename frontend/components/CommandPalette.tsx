"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Search, ArrowRight } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import type { Recommendation } from "@/lib/types";

export function CommandPalette({
  open,
  onClose,
  recommendations,
  candidateId,
}: {
  open: boolean;
  onClose: () => void;
  recommendations: Recommendation[];
  candidateId: string;
}) {
  const [query, setQuery] = useState("");
  const router = useRouter();

  useEffect(() => {
    if (!open) setQuery("");
  }, [open]);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const filtered = useMemo(() => {
    const q = query.toLowerCase().trim();
    if (!q) return recommendations.slice(0, 8);
    return recommendations
      .filter(
        (r) =>
          r.job_title.toLowerCase().includes(q) ||
          r.company.toLowerCase().includes(q) ||
          r.job_location.toLowerCase().includes(q)
      )
      .slice(0, 8);
  }, [query, recommendations]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-start justify-center bg-base-950/70 backdrop-blur-sm pt-[15vh]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.97, y: -8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.97, y: -8 }}
            transition={{ duration: 0.15 }}
            onClick={(e) => e.stopPropagation()}
            className="card w-full max-w-xl overflow-hidden"
          >
            <div className="flex items-center gap-3 border-b border-base-700/60 px-4 py-3">
              <Search className="h-4 w-4 text-base-500" />
              <input
                autoFocus
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search jobs by title, company, or location…"
                className="w-full bg-transparent text-sm text-base-100 placeholder:text-base-500 focus:outline-none"
              />
              <span className="kbd">Esc</span>
            </div>
            <div className="max-h-80 overflow-y-auto scrollbar-thin p-2">
              {filtered.length === 0 && (
                <div className="px-3 py-8 text-center text-sm text-base-500">No matching jobs.</div>
              )}
              {filtered.map((r) => (
                <button
                  key={r.job_id}
                  onClick={() => {
                    onClose();
                    router.push(`/recommendation/${r.job_id}?candidate=${candidateId}`);
                  }}
                  className="flex w-full items-center justify-between rounded-xl px-3 py-2.5 text-left transition-colors hover:bg-base-800"
                >
                  <div>
                    <div className="text-sm font-medium text-base-100">{r.job_title}</div>
                    <div className="text-xs text-base-500">
                      {r.company} · {r.job_location}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-base-500">
                    <span className="tabular-nums">{Math.round(r.score.final_score * 100)}%</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
