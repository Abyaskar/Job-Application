import type {
  EvalMetrics,
  IntentProfile,
  ParsedJob,
  ParsedResume,
  Recommendation,
  SearchMode,
} from "./types";

// Routed through Next.js rewrites (see next.config.js) so the browser only
// ever talks to same-origin /api/backend/*, and the actual backend host is
// a server-side env var (NEXT_PUBLIC_API_URL) — one less CORS surface to
// manage in production.
const BASE = "/api/backend";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  rankJobs: (params: {
    candidate_id: string;
    top_k?: number;
    mode?: SearchMode;
    location_filter?: string | null;
    domain_filter?: string | null;
    use_intent?: boolean;
  }) =>
    request<Recommendation[]>("/recommendations/rank", {
      method: "POST",
      body: JSON.stringify({ top_k: 10, mode: "hybrid", use_intent: true, ...params }),
    }),

  previewIntent: (text: string) =>
    request<IntentProfile>(`/intent/preview?text=${encodeURIComponent(text)}`),

  submitIntent: (candidate_id: string, free_text: string) =>
    request<IntentProfile>("/intent", { method: "POST", body: JSON.stringify({ candidate_id, free_text }) }),

  getIntent: (candidateId: string) => request<IntentProfile>(`/intent/${candidateId}`),

  intentImpact: () => request<Record<string, unknown>>("/evaluation/intent-impact"),

  compareSearchModes: (candidate_id: string, top_k = 5) =>
    request<Record<string, { recommendations: Recommendation[]; latency_ms: number }>>(
      "/recommendations/compare",
      { method: "POST", body: JSON.stringify({ candidate_id, top_k }) }
    ),

  getRecommendationDetail: (candidateId: string, jobId: string) =>
    request<Recommendation>(`/recommendations/${candidateId}/${jobId}`),

  getResume: (candidateId: string) => request<ParsedResume>(`/candidates/${candidateId}/resume`),

  ingestResume: (payload: {
    candidate_id: string;
    raw_text: string;
    preferred_locations?: string[];
    preferred_domains?: string[];
  }) => request<ParsedResume>("/candidates/resume", { method: "POST", body: JSON.stringify(payload) }),

  listJobs: (limit = 100) => request<ParsedJob[]>(`/jobs?limit=${limit}`),

  getJob: (jobId: string) => request<ParsedJob>(`/jobs/${jobId}`),

  submitFeedback: (payload: { candidate_id: string; job_id: string; accepted: boolean; applied?: boolean }) =>
    request<{ status: string }>("/feedback", { method: "POST", body: JSON.stringify(payload) }),

  runEvaluation: (topK = 5) => request<Record<string, EvalMetrics>>(`/evaluation/run?top_k=${topK}`),

  cacheStats: () => request<{ hits: number; misses: number; hit_rate: number }>("/evaluation/cache-stats"),
};
