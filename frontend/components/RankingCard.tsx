"use client";

import { motion } from "framer-motion";
import {
  ArrowUpRight,
  TriangleAlert,
  Target,
  Linkedin,
} from "lucide-react";
import Link from "next/link";
import { ActionBadge, UncertaintyBadge } from "./ActionBadge";
import { ScoreRing } from "./ScoreRing";
import type { Recommendation, IntentProfile } from "@/lib/types";

export function RankingCard({
  rec,
  rank,
  candidateId,
  activeIntent,
  candidateLocation,
}: {
  rec: Recommendation;
  rank: number;
  candidateId: string;
  activeIntent: IntentProfile | null;
  candidateLocation: string | null;
}) {
  // The backend is the source of truth for the job destination.
  // Do not construct a new LinkedIn search URL here.
  const jobUrl = rec.external_url?.trim() || null;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.35,
        delay: Math.min(rank, 8) * 0.04,
        ease: "easeOut",
      }}
      whileHover={{ y: -2 }}
      className="card group relative overflow-hidden p-5 transition-colors hover:border-accent/40"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-base-800 text-xs font-semibold text-base-300">
            {rank}
          </div>

          <div>
            <h3 className="text-base font-semibold text-base-100 group-hover:text-white">
              {rec.job_title}
            </h3>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <ActionBadge action={rec.action} />
              <UncertaintyBadge uncertainty={rec.uncertainty} />

              {rec.score.intent_gated && (
                <span
                  className="pill border-signal-danger/30 bg-signal-danger/10 text-signal-danger"
                  title="Deprioritized for conflicting with your stated career intent, not your resume content."
                >
                  <Target className="h-3 w-3" />
                  Off intent
                </span>
              )}

              {rec.skill_gap.missing_required.length > 0 && (
                <span className="pill border-base-600 bg-base-800 text-base-300">
                  <TriangleAlert className="h-3 w-3" />
                  {rec.skill_gap.missing_required.length} skill gap
                  {rec.skill_gap.missing_required.length > 1 ? "s" : ""}
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <ScoreRing value={rec.score.final_score} label="Match" />

          <Link
            href={`/recommendation/${rec.job_id}?candidate=${candidateId}`}
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-base-700 text-base-400 opacity-0 transition-all group-hover:opacity-100 hover:border-accent hover:text-accent"
            title="View recommendation details"
          >
            <ArrowUpRight className="h-4 w-4" />
          </Link>

          {jobUrl ? (
            <a
              href={jobUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="flex h-9 items-center gap-2 rounded-lg border border-base-700 px-3 text-sm font-medium text-base-300 opacity-0 transition-all group-hover:opacity-100 hover:border-accent hover:text-accent"
              title={`Open LinkedIn search for ${rec.job_title}`}
            >
              <Linkedin className="h-4 w-4" />
              <span className="hidden xl:inline">LinkedIn</span>
            </a>
          ) : (
            <span
              className="flex h-9 items-center gap-2 rounded-lg border border-base-800 px-3 text-sm font-medium text-base-600 opacity-0 transition-all group-hover:opacity-100"
              title="Job destination unavailable"
              aria-label="Job destination unavailable"
            >
              <Linkedin className="h-4 w-4" />
              <span className="hidden xl:inline">LinkedIn</span>
            </span>
          )}
        </div>
      </div>

      <div className="mt-4 grid grid-cols-5 gap-2 border-t border-base-700/60 pt-4 text-[11px]">
        <MiniStat
          label="Intent"
          value={rec.score.intent_alignment}
        />

        <MiniStat
          label="Semantic"
          value={rec.score.semantic_similarity}
        />

        <MiniStat
          label="Skills"
          value={rec.score.hard_skill_match}
        />

        <MiniStat
          label="Experience"
          value={rec.score.experience_match}
        />

        <MiniStat
          label="Education"
          value={rec.score.education_match}
        />
      </div>
    </motion.div>
  );
}

function MiniStat({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-base-500">
        <span>{label}</span>

        <span className="tabular-nums text-base-300">
          {Math.round(value * 100)}
        </span>
      </div>

      <div className="h-1 overflow-hidden rounded-full bg-base-800">
        <motion.div
          className="h-full rounded-full bg-accent/70"
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}