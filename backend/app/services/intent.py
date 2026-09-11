"""
Career Intent Layer.

The problem this solves: without an explicit signal for "what role does this
candidate actually want," ranking falls back entirely to resume-vs-JD
similarity. A senior backend engineer's resume is *textually* similar to a
"Senior Backend Engineer" JD and also, weakly, to a "Data Engineer" JD (both
mention Python, distributed systems, cloud) — generic similarity alone can't
tell the ranker that the candidate typed "I want to move into GenAI
engineering" and would rather see AI Engineer roles ranked above either.

`normalize_intent()` takes the candidate's free text and resolves it to a
`IntentProfile` via a cascade of increasingly fuzzy matching strategies,
each cheaper and more precise than the next:

  1. Exact alias match against `role_taxonomy.json` (e.g. "genai engineer").
  2. Fuzzy alias match (rapidfuzz token-sort ratio) — catches typos and
     minor phrasing differences ("Gen-AI Engineer", "genai enginer").
  3. Keyword overlap — scores each role family by how many of its
     `keywords`/`core_skills` appear in the free text, for longer
     free-form input ("I want to build things with large language models
     and retrieval augmented generation").
  4. Embedding similarity fallback — embeds the free text and each role
     family's canonical title + keywords using the same embedding provider
     ranking uses, for intent text that doesn't share vocabulary with the
     taxonomy at all.

This is deliberately NOT an LLM call: intent normalization needs to be
fast (runs synchronously in the intent-submission request), deterministic
enough to unit test, and free of per-request LLM cost for what is
fundamentally a classification-into-a-known-taxonomy problem. An LLM-based
normalizer is a natural swap-in for open-ended free text the taxonomy
genuinely can't resolve (see `IntentMatchMethod.UNRESOLVED`) — documented
as a follow-up in the README rather than built here, to keep this layer
evaluable the same way ranking is.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from app.core.logging import get_logger
from app.models.schemas import IntentMatchMethod, IntentProfile
from app.services.taxonomy import load_role_taxonomy, load_role_title_skills

logger = get_logger("services.intent")

FUZZY_MATCH_THRESHOLD = 78  # rapidfuzz score (0-100) above which we trust a fuzzy alias match
KEYWORD_OVERLAP_MIN_HITS = 2  # minimum keyword/skill hits before trusting keyword-overlap resolution


def _all_aliases() -> list[tuple[str, str]]:
    """Returns (alias, role_family_id) pairs across the whole taxonomy."""
    taxonomy = load_role_taxonomy()
    pairs = []
    for family_id, meta in taxonomy.items():
        for alias in meta["aliases"]:
            pairs.append((alias, family_id))
    return pairs


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _resolve_exact(text: str) -> str | None:
    normalized = text.strip().lower()
    for alias, family_id in _all_aliases():
        if alias == normalized:
            return family_id
    return None


def _resolve_fuzzy(text: str) -> tuple[str | None, float]:
    aliases = _all_aliases()
    choices = [a for a, _ in aliases]
    match = process.extractOne(text.strip().lower(), choices, scorer=fuzz.token_sort_ratio)
    if not match:
        return None, 0.0
    matched_alias, score, idx = match
    if score >= FUZZY_MATCH_THRESHOLD:
        return aliases[idx][1], score / 100.0
    return None, score / 100.0


def _resolve_keyword_overlap(text: str) -> tuple[str | None, float, int]:
    taxonomy = load_role_taxonomy()
    tokens = _tokenize(text)
    best_family, best_hits = None, 0
    for family_id, meta in taxonomy.items():
        vocab = set()
        for kw in meta.get("keywords", []) + [s.replace("_", " ") for s in meta.get("core_skills", [])]:
            vocab |= _tokenize(kw)
        hits = len(tokens & vocab)
        if hits > best_hits:
            best_family, best_hits = family_id, hits
    if best_family and best_hits >= KEYWORD_OVERLAP_MIN_HITS:
        confidence = min(1.0, best_hits / 5)
        return best_family, confidence, best_hits
    return None, 0.0, best_hits


def _resolve_embedding(text: str) -> tuple[str | None, float]:
    """Last-resort resolution via the same embedding space ranking uses.
    Imported lazily to avoid a hard dependency between the intent layer and
    the embedding provider at module-load time (keeps unit tests for exact/
    fuzzy/keyword resolution independent of embedding fit state).
    """
    from app.services.embeddings import cosine_similarity, get_embedding_provider

    provider = get_embedding_provider()
    taxonomy = load_role_taxonomy()
    text_vec = provider.embed(text)

    best_family, best_sim = None, 0.0
    for family_id, meta in taxonomy.items():
        family_text = meta["canonical_title"] + " " + " ".join(meta.get("keywords", []))
        family_vec = provider.embed(family_text)
        sim = cosine_similarity(text_vec, family_vec)
        if sim > best_sim:
            best_family, best_sim = family_id, sim
    return (best_family, best_sim) if best_sim >= 0.2 else (None, best_sim)


def normalize_intent(candidate_id: str, free_text: str) -> IntentProfile:
    taxonomy = load_role_taxonomy()
    free_text_clean = free_text.strip()

    family_id = _resolve_exact(free_text_clean)
    if family_id:
        return _build_profile(candidate_id, free_text_clean, family_id, IntentMatchMethod.EXACT_ALIAS, 1.0)

    family_id, fuzzy_conf = _resolve_fuzzy(free_text_clean)
    if family_id:
        return _build_profile(candidate_id, free_text_clean, family_id, IntentMatchMethod.FUZZY_ALIAS, fuzzy_conf)

    family_id, kw_conf, hits = _resolve_keyword_overlap(free_text_clean)
    if family_id:
        return _build_profile(candidate_id, free_text_clean, family_id, IntentMatchMethod.KEYWORD_OVERLAP, kw_conf)

    family_id, emb_conf = _resolve_embedding(free_text_clean)
    if family_id:
        return _build_profile(candidate_id, free_text_clean, family_id, IntentMatchMethod.EMBEDDING, emb_conf)

    logger.info("intent.unresolved", candidate_id=candidate_id, free_text=free_text_clean)
    return IntentProfile(
        candidate_id=candidate_id,
        raw_text=free_text_clean,
        match_method=IntentMatchMethod.UNRESOLVED,
        confidence=0.0,
    )


def _build_profile(
    candidate_id: str, free_text: str, family_id: str, method: IntentMatchMethod, confidence: float
) -> IntentProfile:
    taxonomy = load_role_taxonomy()
    meta = taxonomy[family_id]

    # Enrich core_skills with Job-Skill-Set-derived skills for this
    # family's *canonical* title only (see scripts/build_taxonomy_from_job_skill_set.py).
    # Deliberately NOT expanded across related_titles too: several related
    # titles are legitimately shared across more than one role family in
    # role_taxonomy.json (e.g. "Full-Stack Engineer, AI Products" is a
    # related title for both genai_engineer and frontend_engineer), so
    # enriching from every related title would leak one family's skills
    # into another's intent profile via that shared title — a real
    # data-quality issue surfaced by actually running this against the
    # dataset loader, not a hypothetical one. Falls back to just
    # role_taxonomy.json's curated core_skills if the dataset-derived
    # lookup hasn't been built yet — never a hard dependency.
    title_skills = load_role_title_skills()
    enriched_skills = set(meta.get("core_skills", [])) | set(title_skills.get(meta["canonical_title"], []))

    profile = IntentProfile(
        candidate_id=candidate_id,
        raw_text=free_text,
        role_family=family_id,
        canonical_title=meta["canonical_title"],
        related_titles=meta.get("related_titles", []),
        intent_skills=sorted(enriched_skills),
        intent_keywords=meta.get("keywords", []),
        match_method=method,
        confidence=round(confidence, 4),
    )
    logger.info(
        "intent.resolved",
        candidate_id=candidate_id,
        role_family=family_id,
        method=method.value,
        confidence=profile.confidence,
    )
    return profile


def title_alignment_score(intent: IntentProfile | None, job_title: str, job_domain: str | None) -> float:
    """Deterministic alignment score in [0, 1] between a candidate's stated
    intent and a specific job's title. Used by the ranking layer as the
    `intent_alignment` score component — kept simple and inspectable
    (substring / token overlap against the resolved role family's
    canonical title + related titles + keywords) rather than another
    embedding call, since intent alignment is meant to be a sharp,
    interpretable gate, not a second fuzzy similarity signal layered on
    top of `semantic_similarity`.
    """
    if intent is None or intent.role_family is None:
        return 1.0  # no stated intent -> don't penalize (falls back to pure resume-based ranking)

    job_title_lower = job_title.lower()
    job_tokens = _tokenize(job_title) | (_tokenize(job_domain) if job_domain else set())

    titles_to_check = [intent.canonical_title] + intent.related_titles
    for title in titles_to_check:
        if title and title.lower() in job_title_lower:
            return 1.0

    intent_tokens = _tokenize(intent.canonical_title or "") | _tokenize(" ".join(intent.related_titles))
    intent_tokens |= _tokenize(" ".join(intent.intent_keywords))
    overlap = len(job_tokens & intent_tokens)
    if overlap >= 2:
        return 0.85
    if overlap == 1:
        return 0.55
    return 0.15  # job title shares nothing with the stated intent — strong signal of misalignment
