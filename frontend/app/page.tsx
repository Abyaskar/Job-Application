"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Sparkles,
  Target,
  GitCompareArrows,
  ShieldCheck,
  Gauge,
  Layers,
  Upload,
  Loader2,
} from "lucide-react";

const FEATURES = [
  {
    icon: Target,
    title: "Ranked, not just scored",
    body: "Every job gets a final priority score built from semantic similarity, hard-skill coverage, experience, education, and location — not a single opaque number.",
  },
  {
    icon: Sparkles,
    title: "Grounded explanations",
    body: "A RAG pipeline retrieves evidence from your resume and the job description before generating any explanation, and drops any sentence it can't ground.",
  },
  {
    icon: GitCompareArrows,
    title: "Keyword vs vector vs hybrid",
    body: "Compare three retrieval strategies side-by-side with live Precision@K, Recall@K, and NDCG@K — not just a demo, an evaluated system.",
  },
  {
    icon: Gauge,
    title: "Built for scale",
    body: "Async FastAPI, Redis-cached recommendations, rate limiting, and a vector index designed to swap in a managed ANN store without touching the ranking layer.",
  },
  {
    icon: ShieldCheck,
    title: "Confidence-aware",
    body: "Thin resumes and sparse job descriptions surface as explicit uncertainty, instead of a falsely precise match percentage.",
  },
  {
    icon: Layers,
    title: "Action, not just analysis",
    body: "Every recommendation resolves to one of four next actions: apply now, tailor your resume, build missing evidence, or deprioritize.",
  },
];

export default function LandingPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  async function handleResumeUpload(file: File) {
    setUploadError(null);

    const allowedTypes = [
      "application/pdf",
      "text/plain",
    ];

    const isAllowed =
      allowedTypes.includes(file.type) ||
      file.name.toLowerCase().endsWith(".pdf") ||
      file.name.toLowerCase().endsWith(".txt");

    if (!isAllowed) {
      setUploadError("Please upload a PDF or TXT resume.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setUploadError("Resume must be smaller than 10 MB.");
      return;
    }

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/backend/candidates/resume/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(body || "Resume upload failed.");
      }

      const resume = await response.json();

      router.push(
        `/dashboard?candidateId=${encodeURIComponent(resume.candidate_id)}`
      );
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : "Something went wrong while uploading your resume."
      );
      setUploading(false);
    }
  }

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    if (file) {
      void handleResumeUpload(file);
    }

    event.target.value = "";
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[560px] bg-grid-fade" />

      <header className="relative mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/15 text-accent">
            <Sparkles className="h-4 w-4" />
          </div>
          <span className="text-sm font-semibold tracking-tight">
            Job Application Strategy AI
          </span>
        </div>

        <Link href="/dashboard" className="btn-secondary text-xs">
          Open dashboard <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </header>

      <main className="relative mx-auto max-w-4xl px-6 pb-24 pt-16 text-center sm:pt-24">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-base-700 bg-base-900/70 px-3 py-1 text-xs text-base-400"
        >
          <span className="h-1.5 w-1.5 rounded-full bg-signal-apply" />
          Hybrid ranking · Grounded RAG · Evaluated on Precision@K / NDCG@K
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.05 }}
          className="text-4xl font-semibold tracking-tight text-base-50 sm:text-6xl"
        >
          Stop applying randomly.
          <br />
          <span className="bg-gradient-to-r from-accent-light to-accent bg-clip-text text-transparent">
            Know exactly which jobs to apply to first.
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.12 }}
          className="mx-auto mt-6 max-w-2xl text-base text-base-400 sm:text-lg"
        >
          Upload your resume and a set of job descriptions. This system ranks every job by
          application priority, explains the ranking with evidence it can actually point to,
          identifies your specific skill gaps, and tells you the next action to take —
          apply now, tailor your resume, or build missing evidence first.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.18 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-3"
        >
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="btn-primary"
          >
            {uploading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Uploading resume...
              </>
            ) : (
              <>
                <Upload className="h-4 w-4" />
                Upload your resume
              </>
            )}
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,application/pdf,text/plain"
            onChange={handleFileChange}
            className="hidden"
          />

          <Link href="/compare" className="btn-secondary">
            Compare search strategies
          </Link>
        </motion.div>

        {uploadError && (
          <p className="mx-auto mt-4 max-w-lg text-sm text-signal-danger">
            {uploadError}
          </p>
        )}

        <p className="mt-4 text-xs text-base-500">
          PDF or TXT · Maximum 10 MB
        </p>
      </main>

      <section className="relative mx-auto max-w-6xl px-6 pb-28">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.45, delay: (i % 3) * 0.06 }}
              className="card p-6"
            >
              <div className="mb-4 flex h-9 w-9 items-center justify-center rounded-lg bg-accent/10 text-accent">
                <f.icon className="h-4 w-4" />
              </div>

              <h3 className="text-sm font-semibold text-base-100">
                {f.title}
              </h3>

              <p className="mt-2 text-sm leading-relaxed text-base-400">
                {f.body}
              </p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="relative mx-auto max-w-4xl px-6 pb-28">
        <div className="card flex flex-col items-center gap-4 p-10 text-center">
          <h2 className="text-2xl font-semibold text-base-100">
            See it rank a real candidate against real job descriptions
          </h2>

          <p className="max-w-lg text-sm text-base-400">
            The dashboard is pre-seeded with sample resumes and job descriptions so you can see
            the full pipeline — extraction, hybrid ranking, RAG explanation, and evaluation —
            without uploading anything.
          </p>

          <Link href="/dashboard" className="btn-primary mt-2">
            Open the dashboard <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>

      <footer className="relative border-t border-base-800 px-6 py-8 text-center text-xs text-base-500">
        Job Application Strategy AI — a portfolio system demonstrating RAG, vector search, hybrid
        recommendation ranking, and evaluated retrieval on an async FastAPI + MongoDB + Redis backend.
      </footer>
    </div>
  );
}