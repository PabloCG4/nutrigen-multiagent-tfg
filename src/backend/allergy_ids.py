"""EU-14 canonical allergen identifiers (English snake_case) for profile JSON and agents."""

from __future__ import annotations

# Must stay in sync with frontend `ALLERGY_PRESET_IDS`.
ALLOWED_ALLERGY_IDS: frozenset[str] = frozenset(
    {
        "gluten",
        "crustaceans",
        "eggs",
        "fish",
        "peanuts",
        "soy",
        "milk",
        "tree_nuts",
        "celery",
        "mustard",
        "sesame",
        "sulfites",
        "lupin",
        "molluscs",
    }
)

# Legacy UI/API tokens from the previous 8-preset list → canonical EU-14 ids.
_LEGACY_EXPAND: dict[str, tuple[str, ...]] = {
    "dairy": ("milk",),
    # Old single "shellfish" covered both crustaceans and molluscs in EU terms.
    "shellfish": ("crustaceans", "molluscs"),
}


def migrate_legacy_allergies(items: list[str] | None) -> list[str]:
    """
    Map stored profile allergies to canonical ids for API responses and agents.
    Expands legacy keys; drops unknown junk; deduplicates preserving first-seen order.
    """
    if not items:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, str):
            continue
        token = raw.strip()
        if not token:
            continue
        if token in _LEGACY_EXPAND:
            expanded = _LEGACY_EXPAND[token]
        elif token in ALLOWED_ALLERGY_IDS:
            expanded = (token,)
        else:
            continue
        for canon in expanded:
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
    return out


def validate_allergies_for_save(items: list[str]) -> list[str]:
    """
    Validates POST /api/profile allergies: only canonical ids, max one per allergen, max 14 entries.
    """
    if len(items) > len(ALLOWED_ALLERGY_IDS):
        raise ValueError(
            f"At most {len(ALLOWED_ALLERGY_IDS)} allergy entries allowed (EU-14 list)."
        )
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, str):
            raise ValueError("Each allergy must be a string.")
        token = raw.strip()
        if not token:
            continue
        if token not in ALLOWED_ALLERGY_IDS:
            raise ValueError(
                f"Invalid allergy identifier: {token!r}. Use EU-14 canonical ids (e.g. milk, peanuts)."
            )
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out
