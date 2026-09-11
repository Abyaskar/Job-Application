"""
Loads the skill taxonomy once and exposes fast alias -> canonical-skill
lookups. This is the shared vocabulary that both resume extraction and job
extraction map free text onto, which is what makes "hard skill match" a
well-defined set intersection rather than a fuzzy string comparison.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger("services.taxonomy")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "skill_taxonomy.json"
ROLE_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "role_taxonomy.json"
ROLE_TITLE_SKILLS_PATH = Path(__file__).resolve().parents[2] / "data" / "role_title_skills.json"


@lru_cache
def load_taxonomy() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


@lru_cache
def alias_to_canonical() -> dict[str, str]:
    taxonomy = load_taxonomy()
    mapping: dict[str, str] = {}
    for canonical, meta in sorted(taxonomy.items()):
        for alias in meta["aliases"]:
            alias_lower = alias.lower()
            if alias_lower in mapping and mapping[alias_lower] != canonical:
                logger.warning(
                    "taxonomy.alias_collision",
                    alias=alias_lower,
                    existing=mapping[alias_lower],
                    new=canonical,
                    detail="Two skill entries share this alias; keeping the first "
                    "(alphabetically-first canonical id) to make resolution deterministic "
                    "regardless of JSON key order. Fix by removing the duplicate alias "
                    "from one entry in skill_taxonomy.json.",
                )
                continue
            mapping[alias_lower] = canonical
    return mapping


def extract_skills(text: str) -> list[str]:
    """Extract canonical skill ids present in free text via alias matching.

    Uses word-boundary regex per alias rather than naive substring search to
    avoid false positives (e.g. "r" inside "framework").
    """
    text_lower = text.lower()
    found = set()
    for alias, canonical in alias_to_canonical().items():
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(alias) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.add(canonical)
    return sorted(found)


def canonical_label(skill_id: str) -> str:
    taxonomy = load_taxonomy()
    aliases = taxonomy.get(skill_id, {}).get("aliases", [skill_id])
    return aliases[0]


@lru_cache
def load_role_taxonomy() -> dict:
    with open(ROLE_DATA_PATH) as f:
        return json.load(f)


@lru_cache
def load_role_title_skills() -> dict:
    """Job title -> list[canonical skill id], built by
    scripts/build_taxonomy_from_job_skill_set.py from the Job Skill Set
    dataset (or its synthetic fallback). Returns {} if the script hasn't
    been run yet — callers should treat this as an optional enrichment
    source, not a required one (the Career Intent Layer's core_skills from
    role_taxonomy.json work fine without it).
    """
    if not ROLE_TITLE_SKILLS_PATH.exists():
        return {}
    with open(ROLE_TITLE_SKILLS_PATH) as f:
        return json.load(f)
