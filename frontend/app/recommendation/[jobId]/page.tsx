"use client";

import { useEffect, useState } from "react";
import { useParams, useSearchParams, useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  CheckCircle2,
  XCircle,
  ThumbsUp,
  ThumbsDown,
  MapPin,
  Building2,
  Target,
} from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { ActionBadge, UncertaintyBadge } from "@/components/ActionBadge";
import { ScoreRing } from "@/components/ScoreRing";
import { EvidencePanel } from "@/components/EvidencePanel";
import { api } from "@/lib/api";
import { skillLabel } from "@/lib/cn";
import type { Recommendation } from "@/lib/types";

export default function RecommendationDetailPage() {
  const params = useParams<{ jobId: string }>();
  const searchParams = useSearchParams();
  const router = useRouter();
  const candidateId = searchParams.get("candidate") ?? "cand_msc_ds_01";

  const [rec, setRec] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [feedbackSent, setFeedbackSent] = useState<"accepted" | "rejected" | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        let detail: Recommendation;
        try {
          detail = await api.getRecommendationDetail(candidateId, params.jobId);
        } catch {
          // Not yet computed/cached — trigger a rank call which persists
          // the recommendation, then fetch detail again.
          const ranked = await api.rankJobs({ candidate_id: candidateId, top_k: 50 });
          const found = ranked.find((r) => r.job_id === params.jobId);
          if (!found) throw new Error("This job did not appear in the candidate's ranked results.");
          detail = found;
        }
        if (!cancelled) setRec(detail);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load recommendation.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [candidateId, params.jobId]);

  async function sendFeedback(accepted: boolean) {
    if (!rec) return;
    try {
      await api.submitFeedback({ candidate_id: candidateId, job_id: rec.job_id, accepted });
      setFeedbackSent(accepted ? "accepted" : "rejected");
    } catch {
      // non-fatal — feedback capture failing shouldn't block the user
    }
  }

  const highlightTerms = rec
    ? [...rec.skill_gap.matched_required, ...rec.skill_gap.missing_required].map(skillLabel)
    : [];

  return (
    <div className="min-h-screen">
      <Navbar />
      <main className="mx-auto max-w-5xl px-6 py-8">
        <button
          onClick={() => router.back()}
          className="mb-6 flex items-center gap-1.5 text-sm text-base-400 hover:text-base-100"
        >
          <ArrowLeft className="h-3.5 w-3.5" /> Back to dashboard
        </button>

        {loading && <div className="card h-64 animate-pulse" />}
        {error && (
          <div className="card p-8 text-center text-sm text-signal-danger">{error}</div>
        )}

        {rec && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
            {/* Header */}
            <div className="card mb-6 p-6">
              <div className="flex flex-col justify-between gap-6 sm:flex-row sm:items-center">
                <div>
                  <h1 className="text-2xl font-semibold text-base-100">{rec.job_title}</h1>
                  <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-base-400">
                    <span className="flex items-center gap-1.5">
                      <Building2 className="h-3.5 w-3.5" /> {rec.company}
                    </span>
                    <span className="flex items-center gap-1.5">
                      <MapPin className="h-3.5 w-3.5" /> {rec.job_location}
                    </span>
                  </div>
                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    <ActionBadge action={rec.action} />
                    <UncertaintyBadge uncertainty={rec.uncertainty} />
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <ScoreRing value={rec.score.final_score} size={88} strokeWidth={7} label="Match" />
                </div>
              </div>
            </div>

            {/* Score breakdown */}
            <div className="card mb-6 p-6">
              <h2 className="mb-4 text-sm font-semibold text-base-200">Score breakdown</h2>
              {rec.score.intent_gated && (
                <div className="mb-4 flex items-center gap-2 rounded-lg border border-signal-danger/30 bg-signal-danger/10 px-3 py-2 text-xs text-signal-danger">
                  <Target className="h-3.5 w-3.5" />
                  Deprioritized: this role&apos;s title/domain doesn&apos;t align with your stated career
                  intent, independent of how well your resume otherwise matches.
                </div>
              )}
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-6">
                <ScoreStat label="Intent" value={rec.score.intent_alignment} weight={rec.score.weights.intent} />
                <ScoreStat label="Semantic" value={rec.score.semantic_similarity} weight={rec.score.weights.semantic} />
                <ScoreStat label="Skills" value={rec.score.hard_skill_match} weight={rec.score.weights.skill} />
                <ScoreStat label="Experience" value={rec.score.experience_match} weight={rec.score.weights.experience} />
                <ScoreStat label="Education" value={rec.score.education_match} weight={rec.score.weights.education} />
                <ScoreStat label="Location" value={rec.score.location_match} weight={rec.score.weights.location} />
              </div>
            </div>

            {/* Why apply / why not apply */}
            <div className="mb-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="card p-6">
                <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-signal-apply">
                  <CheckCircle2 className="h-4 w-4" /> Why should I apply?
                </div>
                {rec.explanation.why_apply.length === 0 ? (
                  <p className="text-sm text-base-500">No strong supporting evidence was found.</p>
                ) : (
                  <ul className="space-y-2">
                    {rec.explanation.why_apply.map((reason, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-base-300">
                        <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-signal-apply" />
                        {reason}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
              <div className="card p-6">
                <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-signal-danger">
                  <XCircle className="h-4 w-4" /> Why might I not apply?
                </div>
                {rec.explanation.why_not_apply.length === 0 ? (
                  <p className="text-sm text-base-500">No significant concerns were found.</p>
                ) : (
                  <ul className="space-y-2">
                    {rec.explanation.why_not_apply.map((reason, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-base-300">
                        <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-signal-danger" />
                        {reason}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* Explanation summary */}
            <div className="card mb-6 p-6">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-sm font-semibold text-base-200">Summary</h2>
                <span className="pill border-base-600 bg-base-800 text-base-400">
                  {rec.explanation.grounded ? "Grounded in evidence" : "Low-confidence explanation"}
                </span>
              </div>
              <p className="text-sm text-base-300">{rec.explanation.summary}</p>
            </div>

            {/* Skill gap */}
            <div className="card mb-6 p-6">
              <h2 className="mb-4 text-sm font-semibold text-base-200">Skill gap</h2>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-signal-apply">
                    <CheckCircle2 className="h-3.5 w-3.5" /> Matched required skills
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {rec.skill_gap.matched_required.length === 0 && (
                      <span className="text-xs text-base-500">None matched.</span>
                    )}
                    {rec.skill_gap.matched_required.map((s) => (
                      <span key={s} className="pill border-signal-apply/30 bg-signal-apply/10 text-signal-apply">
                        {skillLabel(s)}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-signal-danger">
                    <XCircle className="h-3.5 w-3.5" /> Missing required skills
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {rec.skill_gap.missing_required.length === 0 && (
                      <span className="text-xs text-base-500">No gaps — fully covered.</span>
                    )}
                    {rec.skill_gap.missing_required.map((s) => (
                      <span key={s} className="pill border-signal-danger/30 bg-signal-danger/10 text-signal-danger">
                        {skillLabel(s)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-base-800">
                <div
                  className="h-full rounded-full bg-signal-apply"
                  style={{ width: `${rec.skill_gap.coverage_ratio * 100}%` }}
                />
              </div>
              <div className="mt-1.5 text-xs text-base-500">
                {Math.round(rec.skill_gap.coverage_ratio * 100)}% required-skill coverage
              </div>
            </div>

            {/* Evidence view */}
            <div className="mb-6">
              <h2 className="mb-4 text-sm font-semibold text-base-200">Resume vs. job description evidence</h2>
              <EvidencePanel evidence={rec.explanation.evidence} highlightTerms={highlightTerms} />
            </div>

            {/* Feedback */}
            <div className="card flex flex-col items-center gap-3 p-6 text-center">
              <p className="text-sm text-base-400">Was this recommendation useful?</p>
              {feedbackSent ? (
                <p className="text-sm text-signal-apply">
                  Thanks — feedback recorded ({feedbackSent}).
                </p>
              ) : (
                <div className="flex gap-3">
                  <button onClick={() => sendFeedback(true)} className="btn-secondary text-xs">
                    <ThumbsUp className="h-3.5 w-3.5" /> Useful
                  </button>
                  <button onClick={() => sendFeedback(false)} className="btn-secondary text-xs">
                    <ThumbsDown className="h-3.5 w-3.5" /> Not useful
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
}

function ScoreStat({ label, value, weight }: { label: string; value: number; weight: number }) {
  return (
    <div>
      <div className="text-xs text-base-500">
        {label} <span className="text-base-600">· {Math.round(weight * 100)}% weight</span>
      </div>
      <div className="mt-1 text-lg font-semibold tabular-nums text-base-100">{Math.round(value * 100)}%</div>
    </div>
  );
}
