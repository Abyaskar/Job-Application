"use client";

import { useEffect, useMemo, useState } from "react";
import { AnimatePresence } from "framer-motion";
import { AlertTriangle, RefreshCcw, Zap } from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { Filters } from "@/components/Filters";
import { RankingCard } from "@/components/RankingCard";
import { CardSkeleton } from "@/components/CardSkeleton";
import { CommandPalette } from "@/components/CommandPalette";
import { IntentBar } from "@/components/IntentBar";
import { api } from "@/lib/api";
import type { IntentProfile, Recommendation, SearchMode } from "@/lib/types";

const SAMPLE_CANDIDATES = [
  { id: "cand_msc_ds_01", label: "MSc Data Science grad — RAG/vector search projects" },
  { id: "cand_frontend_01", label: "Frontend developer — React/Next.js" },
  { id: "cand_senior_backend_01", label: "Senior backend engineer — distributed systems" },
  { id: "cand_nlp_researcher_01", label: "PhD candidate — NLP research" },
  { id: "cand_data_analyst_01", label: "Data analyst — healthcare reporting" },
  { id: "cand_new_grad_01", label: "Recent CS grad — internship experience" },
];

export default function DashboardPage() {
  const [candidateId, setCandidateId] = useState<string>("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [location, setLocation] = useState("");
  const [domain, setDomain] = useState("");
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [intent, setIntent] = useState<IntentProfile | null>(null);
  const [useIntent, setUseIntent] = useState(true);
  const [candidateLocation, setCandidateLocation] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const uploadedCandidateId = params.get("candidateId");

    setCandidateId(uploadedCandidateId || SAMPLE_CANDIDATES[0].id);
  }, []);

  // Fetch candidate's resume-parsed location for LinkedIn URL builder (Bug 2 fix)
  useEffect(() => {
    if (!candidateId) return;
    
    api
      .getResume(candidateId)
      .then((resume) => {
        // Use current_location from resume parsing as the source of truth
        const loc = resume.current_location || resume.preferred_locations?.[0] || null;
        setCandidateLocation(loc);
      })
      .catch(() => setCandidateLocation(null));
  }, [candidateId]);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen(true);
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  async function loadRecommendations() {
  if (!candidateId) return;

  setLoading(true);
  setError(null);
  const start = performance.now();

  try {
    const recs = await api.rankJobs({
      candidate_id: candidateId,
        top_k: 12,
        mode,
        location_filter: location || null,
        domain_filter: domain || null,
        use_intent: useIntent,
      });
      setRecommendations(recs);
      setLatencyMs(Math.round(performance.now() - start));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load recommendations.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setIntent(null);
    if (candidateId) {
      api
        .getIntent(candidateId)
        .then(setIntent)
        .catch(() => setIntent(null));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidateId]);

  useEffect(() => {
    if (candidateId) {
      loadRecommendations();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidateId, mode, location, domain, useIntent, intent?.role_family]);

  const locations = useMemo(
    () => Array.from(new Set(recommendations.map((r) => r.job_location))).sort(),
    [recommendations]
  );
  const domains = useMemo(
    () => Array.from(new Set(recommendations.map((r) => r.job_domain).filter(Boolean) as string[])).sort(),
    [recommendations]
  );

  const summary = useMemo(() => {
    const applyNow = recommendations.filter((r) => r.action === "apply_now").length;
    const avgMatch =
      recommendations.length > 0
        ? recommendations.reduce((s, r) => s + r.score.final_score, 0) / recommendations.length
        : 0;
    return { applyNow, avgMatch };
  }, [recommendations]);

  return (
    <div className="min-h-screen">
      <Navbar onOpenPalette={() => setPaletteOpen(true)} />
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        recommendations={recommendations}
        candidateId={candidateId}
      />

      <main className="mx-auto max-w-6xl px-6 py-8">
        <div className="mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
          <div>
            <h1 className="text-2xl font-semibold text-base-100">Application priority</h1>
            <p className="mt-1 text-sm text-base-400">
              Ranked by hybrid score across semantic similarity, hard-skill match, experience,
              education, and location.
            </p>
          </div>
        </div>

        <div className="mb-6">
          <IntentBar
            candidateId={candidateId}
            activeIntent={intent}
            onIntentChange={setIntent}
            useIntent={useIntent}
            onToggleUseIntent={setUseIntent}
          />
        </div>

        <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="Jobs ranked" value={recommendations.length.toString()} />
          <StatCard label="Apply now" value={summary.applyNow.toString()} accent="text-signal-apply" />
          <StatCard label="Avg match" value={`${Math.round(summary.avgMatch * 100)}%`} />
          <StatCard label="Latency" value={latencyMs !== null ? `${latencyMs} ms` : "—"} icon={<Zap className="h-3.5 w-3.5" />} />
        </div>

        <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <Filters
            mode={mode}
            onModeChange={setMode}
            location={location}
            onLocationChange={setLocation}
            domain={domain}
            onDomainChange={setDomain}
            locations={locations}
            domains={domains}
          />
          <button onClick={loadRecommendations} className="btn-secondary text-xs">
            <RefreshCcw className="h-3.5 w-3.5" /> Refresh
          </button>
        </div>

        {error && (
          <div className="mb-6 flex items-center gap-2 rounded-xl border border-signal-danger/30 bg-signal-danger/10 px-4 py-3 text-sm text-signal-danger">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </div>
        )}

        <div className="space-y-3">
          {loading &&
            Array.from({ length: 5 }).map((_, i) => <CardSkeleton key={i} />)}

          {!loading && recommendations.length === 0 && !error && (
            <div className="card p-10 text-center text-sm text-base-400">
              No jobs matched the current filters. Try clearing a filter.
            </div>
          )}

          <AnimatePresence>
            {!loading &&
              recommendations.map((rec, i) => (
                <RankingCard 
                  key={rec.job_id} 
                  rec={rec} 
                  rank={i + 1} 
                  candidateId={candidateId}
                  activeIntent={intent}
                  candidateLocation={candidateLocation}
                />
              ))}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

function StatCard({
  label,
  value,
  accent,
  icon,
}: {
  label: string;
  value: string;
  accent?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="card p-4">
      <div className="flex items-center gap-1.5 text-xs text-base-500">
        {icon}
        {label}
      </div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${accent ?? "text-base-100"}`}>{value}</div>
    </div>
  );
}
