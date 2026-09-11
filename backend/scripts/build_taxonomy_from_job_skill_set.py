"""
Build/extend `data/skill_taxonomy.json` and `data/role_title_skills.json`
from the "Job Skill Set" Kaggle dataset (or its synthetic fallback — see
`data/external/README.md`).

Usage:
    python scripts/build_taxonomy_from_job_skill_set.py

Idempotent and additive: existing taxonomy entries are never overwritten,
and a skill text is only added as a *new* taxonomy entry if it doesn't
already match an existing alias (via the same taxonomy alias-lookup the
rest of the app uses) — this is the "normalize schemas, don't blindly
merge" requirement in practice: dataset skill strings are resolved against
the existing controlled vocabulary first, and only genuinely new terms
become new entries.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.dataset_loaders import load_job_skill_set  # noqa: E402
from app.services.taxonomy import alias_to_canonical, load_taxonomy  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TAXONOMY_PATH = DATA_DIR / "skill_taxonomy.json"
ROLE_TITLE_SKILLS_PATH = DATA_DIR / "role_title_skills.json"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown_skill"


def main() -> None:
    pairs, is_synthetic = load_job_skill_set()
    print(f"Loaded {len(pairs)} (job_title, skill) pairs "
          f"[{'SYNTHETIC fallback' if is_synthetic else 'real dataset'}]")

    taxonomy = load_taxonomy().copy()
    alias_map = alias_to_canonical()

    new_skills = 0
    role_title_skills: dict[str, set[str]] = {}

    for pair in pairs:
        skill_lower = pair.skill_text.lower().strip()
        canonical = alias_map.get(skill_lower)

        if canonical is None:
            # Not in the existing controlled vocabulary — add it as a new
            # taxonomy entry rather than silently dropping it.
            canonical = _slugify(skill_lower)
            if canonical not in taxonomy:
                taxonomy[canonical] = {"aliases": [skill_lower], "category": "dataset_derived"}
                alias_map[skill_lower] = canonical
                new_skills += 1

        role_title_skills.setdefault(pair.job_title, set()).add(canonical)

    with open(TAXONOMY_PATH, "w") as f:
        json.dump(taxonomy, f, indent=2, sort_keys=True)
    print(f"skill_taxonomy.json: {len(taxonomy)} total skills ({new_skills} newly added)")

    role_title_skills_serializable = {
        title: sorted(skills) for title, skills in sorted(role_title_skills.items())
    }
    with open(ROLE_TITLE_SKILLS_PATH, "w") as f:
        json.dump(role_title_skills_serializable, f, indent=2, sort_keys=True)
    print(f"role_title_skills.json: {len(role_title_skills_serializable)} job titles mapped")

    if is_synthetic:
        print(
            "\nNOTE: ran against synthetic fallback data (no network access to kaggle.com in this "
            "sandbox). Place the real CSV at data/external/job_skill_set.csv and re-run for real "
            "dataset coverage — see data/external/README.md."
        )


if __name__ == "__main__":
    main()
