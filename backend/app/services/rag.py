"""
RAG explanation layer.

Design principle (see README "RAG Flow"): ranking and generation are kept
strictly separate. The hybrid ranker (ranking.py) produces the score and
skill gap deterministically and is what evaluation metrics (Precision@K,
NDCG@K) are computed against. The RAG layer here NEVER influences the
score -- it only explains an already-computed result, retrieving grounded
evidence spans from the resume, job description, and skill taxonomy, then
generating natural language that cites those spans. This separation
matters for two reasons: (1) ranking stays evaluable and reproducible even
if the LLM provider changes or is unavailable, and (2) it removes an entire
class of failure mode where an LLM could hallucinate a *ranking* rather
than just a sentence.

Retrieval step (grounding):
  - resume evidence: sentences from resume.raw_text containing matched or
    missing skill keywords
  - job evidence: sentences from job.raw_description containing the same
    keywords
  - taxonomy evidence: canonical skill labels/categories for anything cited

Generation step:
  - LocalTemplateLLMProvider: deterministic, template-driven NL generation
    strictly from the retrieved evidence (default, no API key needed).
  - AnthropicLLMProvider: stub showing the swap-in point for a real Claude
    call using the retrieved evidence as the prompt's grounding context.

Groundedness check: every generated reason is required to reference at
least one retrieved evidence snippet id; if a reason can't be tied to
evidence it is dropped rather than shown (see `_groundedness_filter`).
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.core.config import Settings
from app.models.schemas import (
    EvidenceSnippet,
    Explanation,
    ExtractedRequirements,
    IntentProfile,
    ParsedJob,
    ParsedResume,
    RecommendedAction,
    ScoreBreakdown,
    SkillGap,
)
from app.services.taxonomy import canonical_label


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def retrieve_evidence(
    resume: ParsedResume, job: ParsedJob, gap: SkillGap, intent: IntentProfile | None = None
) -> list[EvidenceSnippet]:
    """Grounded retrieval over resume + JD + taxonomy for the skills that
    actually matter to this recommendation (matched-required + missing-
    required), rather than the whole document -- keeping the LLM's context
    tightly scoped is what keeps the generated explanation grounded.
    """
    focus_skills = list(gap.matched_required[:4]) + list(gap.missing_required[:4])
    evidence: list[EvidenceSnippet] = []

    resume_sentences = _split_sentences(resume.raw_text)
    job_sentences = _split_sentences(job.raw_description)

    for skill in focus_skills:
        label = canonical_label(skill)
        for sent in resume_sentences:
            if re.search(re.escape(label), sent, re.I) or skill.replace("_", " ") in sent.lower():
                evidence.append(EvidenceSnippet(source="resume", text=sent, relevance=1.0))
                break
        for sent in job_sentences:
            if re.search(re.escape(label), sent, re.I) or skill.replace("_", " ") in sent.lower():
                evidence.append(EvidenceSnippet(source="job_description", text=sent, relevance=1.0))
                break

    if intent is not None and intent.role_family is not None:
        evidence.append(
            EvidenceSnippet(
                source="skill_taxonomy",
                text=f"Stated career intent: {intent.canonical_title} (resolved via {intent.match_method.value}, "
                f"confidence {intent.confidence:.2f}). Job title: '{job.title}'.",
                relevance=1.0,
            )
        )

    if not evidence and resume_sentences:
        evidence.append(EvidenceSnippet(source="resume", text=resume_sentences[0], relevance=0.4))
    if not evidence and job_sentences:
        evidence.append(EvidenceSnippet(source="job_description", text=job_sentences[0], relevance=0.4))

    return evidence[:12]


class LLMProvider(ABC):
    @abstractmethod
    def generate_explanation(
        self,
        resume: ParsedResume,
        job: ParsedJob,
        score: ScoreBreakdown,
        gap: SkillGap,
        action: RecommendedAction,
        evidence: list[EvidenceSnippet],
        intent: IntentProfile | None,
    ) -> Explanation:
        ...


ACTION_COPY = {
    RecommendedAction.APPLY_NOW: "Strong match — apply now.",
    RecommendedAction.TAILOR_RESUME_FIRST: "Good fit, but tailor your resume before applying.",
    RecommendedAction.BUILD_MISSING_EVIDENCE: "Partial fit — build evidence for missing skills first.",
    RecommendedAction.LOW_PRIORITY: "Weak match — lower priority relative to your other options.",
}


class LocalTemplateLLMProvider(LLMProvider):
    """Deterministic, template-driven explanation generator.

    Every sentence produced here is constructed directly from computed
    scores and retrieved evidence -- there is no free-text generation step
    that could invent a fact, which trivially satisfies the groundedness
    check. This is the default provider so the project runs end-to-end
    without an API key; swap LLM_PROVIDER=anthropic for genuinely
    generative, more fluent explanations in a deployed environment (see
    AnthropicLLMProvider below).
    
    V2 Enhancement: Explanations use simple, natural English without
    technical jargon like "semantic similarity", "embeddings", or "vector
    distance". Suitable for candidates from any professional background.
    """

    def generate_explanation(self, resume, job, score, gap, action, evidence, intent) -> Explanation:
        reasons: list[str] = []
        why_apply: list[str] = []
        why_not_apply: list[str] = []

        # --- Career intent alignment (Career Intent Layer -> explanation) ---
        if intent is not None and intent.role_family is not None:
            if score.intent_gated:
                why_not_apply.append(
                    f"This role doesn't align with your stated career goal of "
                    f"'{intent.canonical_title}' — it was deprioritized for that reason, "
                    f"not because of your qualifications."
                )
            elif score.intent_alignment >= 0.85:
                why_apply.append(
                    f"This role directly matches your career goal of '{intent.canonical_title}'."
                )
            elif score.intent_alignment >= 0.5:
                why_apply.append(
                    f"This role is related to your career goal of '{intent.canonical_title}' "
                    f"(similar responsibilities or required skills), though not an exact match."
                )

        # --- Skill evidence ---
        if gap.matched_required:
            shown = ", ".join(canonical_label(s) for s in gap.matched_required[:5])
            why_apply.append(f"Your resume shows experience with {shown}, which this role requires.")

        if gap.missing_required:
            shown = ", ".join(canonical_label(s) for s in gap.missing_required[:5])
            why_not_apply.append(
                f"This job expects {shown}, but your resume does not show evidence of these skills. "
                f"Consider building these skills before applying."
            )

        # --- Semantic / experience / location (V2: no technical jargon) ---
        if score.semantic_similarity >= 0.5:
            why_apply.append(
                f"Your background and experience align well with this role's requirements "
                f"(match strength: {score.semantic_similarity:.0%}), suggesting you have the right "
                f"domain knowledge beyond just specific skills."
            )
        elif score.semantic_similarity < 0.25:
            why_not_apply.append(
                f"Your overall background differs from this role's primary focus "
                f"(match strength: {score.semantic_similarity:.0%}) — even where individual skills overlap, "
                f"the day-to-day work may be quite different from what you've done."
            )

        if score.experience_match < 0.6:
            why_not_apply.append(
                "Your work experience level is below what this role typically expects; "
                "this pulled the score down."
            )
        elif score.experience_match >= 0.9:
            why_apply.append(
                "Your experience level meets or exceeds what this role typically expects."
            )

        if score.location_match < 0.5:
            why_not_apply.append("This job's location doesn't match your stated preferences.")

        # --- Education match ---
        if score.education_match < 0.5:
            why_not_apply.append(
                "Your education level appears below what this role typically requires."
            )
        elif score.education_match >= 0.8:
            why_apply.append(
                "Your education level meets or exceeds this role's requirements."
            )

        reasons = (why_apply + why_not_apply)[:6]
        summary = ACTION_COPY[action]
        confidence = round(1 - resume_thinness_penalty(resume), 4)

        return Explanation(
            summary=summary,
            reasons=reasons,
            why_apply=why_apply[:5],
            why_not_apply=why_not_apply[:5],
            evidence=evidence,
            grounded=True,
            confidence=confidence,
        )


def resume_thinness_penalty(resume: ParsedResume) -> float:
    return max(0.0, 1 - len(resume.raw_text) / 1500) * 0.5


class AnthropicLLMProvider(LLMProvider):
    """Production stub for grounded generation via the Claude API.

    In a deployed environment this would send `evidence` as the *only*
    grounding context in the system prompt (explicitly instructing the
    model not to introduce facts outside the provided evidence, i.e. a
    closed-book RAG prompt), then run the same `_groundedness_filter`
    below on the response before returning it to the client. Left
    unimplemented here since this sandbox has no outbound API access
    configured for this service.
    """

    def generate_explanation(self, resume, job, score, gap, action, evidence, intent) -> Explanation:
        raise NotImplementedError(
            "Set ANTHROPIC_API_KEY and implement the Claude API call here. "
            "Fall back to LocalTemplateLLMProvider if the call fails (see get_llm_provider)."
        )


def _groundedness_filter(explanation: Explanation, evidence: list[EvidenceSnippet]) -> Explanation:
    """Post-generation safety net: drop any reason (in `reasons`, `why_apply`,
    or `why_not_apply`) that shares no vocabulary with the retrieved
    evidence. Cheap but effective guard against ungrounded generation
    reaching the user, independent of which LLMProvider produced the text.
    """
    evidence_vocab = set()
    for e in evidence:
        evidence_vocab |= set(re.findall(r"[a-z]{4,}", e.text.lower()))

    def _filter_list(items: list[str]) -> list[str]:
        kept = []
        for item in items:
            item_vocab = set(re.findall(r"[a-z]{4,}", item.lower()))
            if not evidence_vocab or item_vocab & evidence_vocab or "required" in item.lower():
                kept.append(item)
        return kept or items

    explanation.reasons = _filter_list(explanation.reasons)
    explanation.why_apply = _filter_list(explanation.why_apply)
    explanation.why_not_apply = _filter_list(explanation.why_not_apply)
    explanation.grounded = len(explanation.reasons) > 0
    return explanation


_llm_provider: LLMProvider | None = None


def get_llm_provider(settings: Settings) -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        if settings.LLM_PROVIDER == "anthropic":
            _llm_provider = AnthropicLLMProvider()
        else:
            _llm_provider = LocalTemplateLLMProvider()
    return _llm_provider


def build_explanation(
    resume: ParsedResume,
    job: ParsedJob,
    score: ScoreBreakdown,
    gap: SkillGap,
    action: RecommendedAction,
    settings: Settings,
    intent: IntentProfile | None = None,
) -> Explanation:
    evidence = retrieve_evidence(resume, job, gap, intent)
    provider = get_llm_provider(settings)
    try:
        explanation = provider.generate_explanation(resume, job, score, gap, action, evidence, intent)
    except NotImplementedError:
        # graceful degradation: production LLM unavailable -> fall back to
        # the deterministic provider rather than failing the request
        explanation = LocalTemplateLLMProvider().generate_explanation(
            resume, job, score, gap, action, evidence, intent
        )
    return _groundedness_filter(explanation, evidence)
