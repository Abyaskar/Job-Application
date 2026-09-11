"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Navbar } from "@/components/Navbar";
import { ActionBadge } from "@/components/ActionBadge";
import { api } from "@/lib/api";
import type { EvalMetrics, Recommendation } from "@/lib/types";

const CANDIDATES = [
  { id: "cand_msc_ds_01", label: "MSc Data Science grad" },
  { id: "cand_senior_backend_01", label: "Senior backend engineer" },
  { id: "cand_nlp_researcher_01", label: "PhD NLP researcher" },
];

const MODE_LABELS: Record<string, string> = { keyword: "Keyword", vector: "Vector", hybrid: "Hybrid" };

export default function ComparePage() {
  const [candidateId, setCandidateId] = useState(CANDIDATES[0].id);
  const [results, setResults] = useState<Record<string, { recommendations: Recommendation[]; latency_ms: number }> | null>(null);
  const [evalMetrics, setEvalMetrics] = useState<Record<string, EvalMetrics> | null>(null);
  const [loading, setLoading] = useState(true);
  const [evalLoading, setEvalLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .compareSearchModes(candidateId, 5)
      .then(setResults)
      .finally(() => setLoading(false));
  }, [candidateId]);

  useEffect(() => {
    setEvalLoading(true);
    api
      .runEvaluation(5)
      .then(setEvalMetrics)
      .finally(() => setEvalLoading(false));
  }, []);

  return (
    <div className="min-h-screen">
      <Navbar />
      <main className="mx-auto max-w-6xl px-6 py-8">
        <div className="mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
          <div>
            <h1 className="text-2xl font-semibold text-base-100">Search strategy comparison</h1>
            <p className="mt-1 text-sm text-base-400">
              The same candidate ranked three ways: pure keyword/skill match, pure semantic
              vector similarity, and the hybrid model this product uses by default.
            </p>
          </div>
          <select
            value={candidateId}
            onChange={(e) => setCandidateId(e.target.value)}
            className="rounded-xl border border-base-700 bg-base-900 px-3 py-2.5 text-sm text-base-200 focus:border-accent focus:outline-none sm:w-64"
          >
            {CANDIDATES.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </div>

        {/* Live evaluation metrics */}
        <section className="mb-10">
          <h2 className="mb-4 text-sm font-semibold text-base-200">
            Evaluation on the labeled test set (live)
          </h2>
          {evalLoading ? (
            <div className="card h-32 animate-pulse" />
          ) : evalMetrics ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              {Object.entries(evalMetrics).map(([mode, m]) => (
                <div key={mode} className="card p-5">
                  <div className="mb-3 flex items-center justify-between">
                    <span className="text-sm font-semibold text-base-100">{MODE_LABELS[mode] ?? mode}</span>
                    <span className="pill border-base-600 bg-base-800 text-base-400">
                      {m.avg_latency_ms} ms
                    </span>
                  </div>
                  <MetricRow label="Precision@3" value={m.precision_at_k["3"]} />
                  <MetricRow label="Recall@3" value={m.recall_at_k["3"]} />
                  <MetricRow label="NDCG@3" value={m.ndcg_at_k["3"]} highlight />
                  <MetricRow label="NDCG@5" value={m.ndcg_at_k["5"]} />
                </div>
              ))}
            </div>
          ) : (
            <div className="card p-6 text-sm text-base-500">Evaluation metrics unavailable.</div>
          )}
        </section>

        {/* Side-by-side ranked results */}
        <section>
          <h2 className="mb-4 text-sm font-semibold text-base-200">Top 5 ranked jobs per strategy</h2>
          {loading ? (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="card h-96 animate-pulse" />
              ))}
            </div>
          ) : results ? (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
              {Object.entries(results).map(([mode, data]) => (
                <motion.div
                  key={mode}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35 }}
                  className="card p-5"
                >
                  <div className="mb-4 flex items-center justify-between">
                    <span className="text-sm font-semibold text-base-100">{MODE_LABELS[mode] ?? mode}</span>
                    <span className="text-xs text-base-500">{data.latency_ms} ms</span>
                  </div>
                  <ol className="space-y-3">
                    {data.recommendations.map((r, i) => (
                      <li key={r.job_id} className="rounded-xl border border-base-700/60 bg-base-900/50 p-3">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <div className="text-xs text-base-500">#{i + 1}</div>
                            <div className="text-sm font-medium text-base-100">{r.job_title}</div>
                            <div className="text-xs text-base-500">{r.company}</div>
                          </div>
                          <span className="shrink-0 text-sm font-semibold tabular-nums text-base-200">
                            {Math.round(r.score.final_score * 100)}%
                          </span>
                        </div>
                        <div className="mt-2">
                          <ActionBadge action={r.action} />
                        </div>
                      </li>
                    ))}
                  </ol>
                </motion.div>
              ))}
            </div>
          ) : (
            <div className="card p-6 text-sm text-base-500">No results.</div>
          )}
        </section>
      </main>
    </div>
  );
}

function MetricRow({ label, value, highlight }: { label: string; value: number; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-base-500">{label}</span>
      <span className={highlight ? "font-semibold text-accent-light" : "text-base-300"}>
        {(value ?? 0).toFixed(3)}
      </span>
    </div>
  );
}
