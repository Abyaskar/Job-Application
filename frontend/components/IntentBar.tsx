"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Target, Check, Loader2, X } from "lucide-react";
import { api } from "@/lib/api";
import type { IntentProfile } from "@/lib/types";
import { skillLabel } from "@/lib/cn";

const SUGGESTIONS = ["GenAI Engineer", "Data Analyst", "Business Analyst", "ML Engineer", "Backend Engineer"];

export function IntentBar({
  candidateId,
  activeIntent,
  onIntentChange,
  useIntent,
  onToggleUseIntent,
}: {
  candidateId: string;
  activeIntent: IntentProfile | null;
  onIntentChange: (intent: IntentProfile | null) => void;
  useIntent: boolean;
  onToggleUseIntent: (v: boolean) => void;
}) {
  const [text, setText] = useState("");
  const [preview, setPreview] = useState<IntentProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (text.trim().length < 2) {
      setPreview(null);
      return;
    }
    setLoading(true);
    const timeout = setTimeout(() => {
      api
        .previewIntent(text.trim())
        .then(setPreview)
        .catch(() => setPreview(null))
        .finally(() => setLoading(false));
    }, 350);
    return () => clearTimeout(timeout);
  }, [text]);

  async function confirmIntent(freeText: string) {
    setSubmitting(true);
    try {
      const profile = await api.submitIntent(candidateId, freeText);
      onIntentChange(profile);
      setText("");
      setPreview(null);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="card p-5">
      <div className="mb-3 flex items-center gap-2 text-sm font-medium text-base-200">
        <Target className="h-4 w-4 text-accent" />
        Career intent
      </div>

      {activeIntent?.role_family ? (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="pill border-accent/30 bg-accent/10 text-accent-light">
              <Check className="h-3 w-3" />
              {activeIntent.canonical_title}
            </span>
            <span className="text-xs text-base-500">
              resolved via {activeIntent.match_method.replace(/_/g, " ")} · {Math.round(activeIntent.confidence * 100)}% confidence
            </span>
          </div>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-xs text-base-400">
              <input
                type="checkbox"
                checked={useIntent}
                onChange={(e) => onToggleUseIntent(e.target.checked)}
                className="h-3.5 w-3.5 accent-accent"
              />
              Apply to ranking
            </label>
            <button
              onClick={() => onIntentChange(null)}
              className="flex items-center gap-1 text-xs text-base-500 hover:text-base-200"
            >
              <X className="h-3 w-3" /> Clear
            </button>
          </div>
        </div>
      ) : (
        <>
          <p className="mb-3 text-xs text-base-500">
            Type the role you want — e.g. &quot;GenAI Engineer&quot; or &quot;I want to move into data
            analytics&quot; — so ranking doesn&apos;t recommend unrelated roles just because your resume
            happens to share vocabulary with them.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <div className="relative flex-1 min-w-[240px]">
              <input
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && preview?.role_family) confirmIntent(text.trim());
                }}
                placeholder="e.g. GenAI Engineer"
                className="w-full rounded-xl border border-base-700 bg-base-900 px-3 py-2.5 text-sm text-base-100 placeholder:text-base-500 focus:border-accent focus:outline-none"
              />
              {loading && (
                <Loader2 className="absolute right-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 animate-spin text-base-500" />
              )}
            </div>
            <button
              disabled={!preview?.role_family || submitting}
              onClick={() => confirmIntent(text.trim())}
              className="btn-primary text-xs"
            >
              {submitting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
              Set intent
            </button>
          </div>

          <AnimatePresence>
            {preview && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-3 overflow-hidden"
              >
                {preview.role_family ? (
                  <div className="rounded-xl border border-base-700/60 bg-base-900/60 p-3 text-xs">
                    <div className="mb-1.5 text-base-300">
                      Did you mean <span className="font-medium text-accent-light">{preview.canonical_title}</span>?
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {preview.intent_skills.slice(0, 8).map((s) => (
                        <span key={s} className="pill border-base-600 bg-base-800 text-base-400">
                          {skillLabel(s)}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="rounded-xl border border-base-700/60 bg-base-900/60 p-3 text-xs text-base-500">
                    Couldn&apos;t confidently resolve that to a known role family — ranking will fall back to
                    resume-based similarity only. Try one of the suggestions below.
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          <div className="mt-3 flex flex-wrap gap-1.5">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => setText(s)}
                className="pill border-base-700 bg-base-900 text-base-400 hover:border-accent/40 hover:text-accent-light"
              >
                {s}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
