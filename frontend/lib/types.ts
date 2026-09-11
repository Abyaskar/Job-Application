export type RecommendedAction =
  | "apply_now"
  | "tailor_resume_first"
  | "build_missing_evidence"
  | "low_priority";

export type SearchMode = "keyword" | "vector" | "hybrid";

export interface ScoreBreakdown {
  intent_alignment: number;
  semantic_similarity: number;
  hard_skill_match: number;
  experience_match: number;
  education_match: number;
  location_match: number;
  final_score: number;
  weights: Record<string, number>;
  intent_gated: boolean;
}

export interface SkillGap {
  missing_required: string[];
  missing_preferred: string[];
  matched_required: string[];
  coverage_ratio: number;
}

export interface EvidenceSnippet {
  source: "resume" | "job_description" | "skill_taxonomy";
  text: string;
  relevance: number;
}

export interface Explanation {
  summary: string;
  reasons: string[];
  why_apply: string[];
  why_not_apply: string[];
  evidence: EvidenceSnippet[];
  grounded: boolean;
  confidence: number;
}

export interface Recommendation {
  candidate_id: string;
  job_id: string;
  job_title: string;
  company: string;
  job_location: string;
  job_domain?: string;
  external_url?: string | null;
  score: ScoreBreakdown;
  skill_gap: SkillGap;
  action: RecommendedAction;
  explanation: Explanation;
  uncertainty: number;
}

export interface ParsedResume {
  candidate_id: string;
  raw_text: string;
  skills: string[];
  education: { degree: string; field?: string; institution?: string; level: number }[];
  experience: { title: string; company?: string; years: number; description: string }[];
  total_experience_years: number;
  preferred_locations: string[];
  preferred_domains: string[];
  // V2 fields for document intelligence
  validation_state?: string;
  document_type?: string;
  extraction_quality_score?: number;
  detected_name?: string | null;
  detected_email?: string | null;
  detected_phone?: string | null;
  current_location?: string | null;
  career_domains?: { domain_id: string; domain_name: string; confidence: number; related_domains?: string[] }[];
  primary_domain?: string | null;
  seniority_level?: string | null;
  preferred_countries?: string[];
  open_to_remote?: boolean;
  open_to_relocation?: boolean;
  processing_metadata?: Record<string, any>;
  model_version?: string;
}

export interface ExtractedRequirements {
  required_skills: string[];
  preferred_skills: string[];
  min_experience_years: number;
  education_level_required: number;
  location: string;
  domain?: string;
  seniority?: string;
}

export interface ParsedJob {
  job_id: string;
  title: string;
  company: string;
  location: string;
  domain?: string;
  raw_description: string;
  requirements: ExtractedRequirements;
}

export interface EvalMetrics {
  precision_at_k: Record<string, number>;
  recall_at_k: Record<string, number>;
  ndcg_at_k: Record<string, number>;
  avg_latency_ms: number;
  n_queries: number;
}

export type IntentMatchMethod = "exact_alias" | "fuzzy_alias" | "keyword_overlap" | "embedding" | "unresolved";

export interface IntentProfile {
  candidate_id: string;
  raw_text: string;
  role_family: string | null;
  canonical_title: string | null;
  related_titles: string[];
  intent_skills: string[];
  intent_keywords: string[];
  match_method: IntentMatchMethod;
  confidence: number;
}

export const ACTION_META: Record<
  RecommendedAction,
  { label: string; colorClass: string; dotClass: string }
> = {
  apply_now: {
    label: "Apply now",
    colorClass: "text-signal-apply border-signal-apply/30 bg-signal-apply/10",
    dotClass: "bg-signal-apply",
  },
  tailor_resume_first: {
    label: "Tailor resume first",
    colorClass: "text-signal-tailor border-signal-tailor/30 bg-signal-tailor/10",
    dotClass: "bg-signal-tailor",
  },
  build_missing_evidence: {
    label: "Build missing evidence",
    colorClass: "text-signal-build border-signal-build/30 bg-signal-build/10",
    dotClass: "bg-signal-build",
  },
  low_priority: {
    label: "Low priority",
    colorClass: "text-signal-low border-signal-low/30 bg-signal-low/10",
    dotClass: "bg-signal-low",
  },
};
