from __future__ import annotations

import atexit
import heapq
import json
import math
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from src.agents.runtime import get_nutritionist_agent  # re-export for src.backend.main
from src.backend import config
from src.common.product_data import (
    get_product_macros_mapping as _get_product_macros_mapping,
    get_product_traduced_macros_mapping as _get_product_traduced_macros_mapping,
)
from src.common.nutrition_targets import calculate_dynamic_targets, round_half_up

_product_name_mapping_cache: Optional[dict[str, str]] = None
_product_macros_mapping_cache: Optional[dict[str, dict]] = None

# Cached merged catalog for autocomplete:
# (mapping, has_trad_overlay, legacy_mtime, trad_mtime, row_norms_by_barcode)
_autocomplete_merged_cache: Optional[
    tuple[dict[str, dict], bool, float, float, dict[str, "AutocompleteRowNorms"]]
] = None

# Minimum fuzzy score (0–100) for autocomplete when substring tier does not match.
AUTOCOMPLETE_FUZZ_THRESHOLD = 58.0

# Parallel chunks for scoring (only used when catalog is large enough).
_AUTOCOMPLETE_CHUNK_WORKERS = 8

# Catalog size at or above this may use SQLite FTS5 candidate prefilter (with full-scan fallback).
AUTOCOMPLETE_USE_FTS_MIN_ITEMS = 4000

_autocomplete_executor: Optional[ThreadPoolExecutor] = None


@dataclass(frozen=True)
class AutocompleteRowNorms:
    """Precomputed normalized strings for one catalog row (invalidates with autocomplete cache)."""

    has_bilingual: bool
    n_bi_left: str
    n_bi_right: str
    n_pn: str
    n_disp_es: str
    n_disp_en: str


def _get_autocomplete_executor() -> ThreadPoolExecutor:
    global _autocomplete_executor
    if _autocomplete_executor is None:
        _autocomplete_executor = ThreadPoolExecutor(max_workers=_AUTOCOMPLETE_CHUNK_WORKERS)
    return _autocomplete_executor


def _shutdown_autocomplete_executor() -> None:
    global _autocomplete_executor
    if _autocomplete_executor is not None:
        _autocomplete_executor.shutdown(wait=True)
        _autocomplete_executor = None


atexit.register(_shutdown_autocomplete_executor)


def _try_rebuild_autocomplete_fts(
    merged: dict[str, dict],
    norms: dict[str, AutocompleteRowNorms],
    mt_legacy: float,
    mt_trad: float,
) -> None:
    try:
        from src.backend import autocomplete_fts

        autocomplete_fts.rebuild_autocomplete_fts(merged, norms, mt_legacy, mt_trad)
    except Exception:
        pass


def _norms_for_record(rec: dict) -> AutocompleteRowNorms:
    es = rec.get("name_es") if isinstance(rec.get("name_es"), str) else ""
    en = rec.get("name_en") if isinstance(rec.get("name_en"), str) else ""
    es, en = es.strip(), en.strip()
    has_bilingual = bool(es or en)
    pn_raw = rec.get("product_name")
    pn_stripped = pn_raw.strip() if isinstance(pn_raw, str) else ""

    if has_bilingual:
        n_left = normalize_product_text(es or en)
        n_right = normalize_product_text(en or es)
        n_disp_es = normalize_product_text(es or en)
        n_disp_en = normalize_product_text(en or es)
        n_pn = normalize_product_text(pn_stripped) if pn_stripped else ""
        return AutocompleteRowNorms(True, n_left, n_right, n_pn, n_disp_es, n_disp_en)

    n_pn = normalize_product_text(pn_stripped) if pn_stripped else ""
    return AutocompleteRowNorms(False, "", "", n_pn, n_pn, n_pn)


def _build_autocomplete_row_norms(merged: dict[str, dict]) -> dict[str, AutocompleteRowNorms]:
    out: dict[str, AutocompleteRowNorms] = {}
    for barcode, rec in merged.items():
        if not isinstance(barcode, str) or not isinstance(rec, dict):
            continue
        out[barcode] = _norms_for_record(rec)
    return out


def get_autocomplete_product_mapping() -> tuple[dict[str, dict], bool]:
    """
    Merge legacy macros JSON with optional bilingual overlay (product_database_macros_traduced.json).
    Overlay wins per barcode for name_es/name_en. Full catalog remains searchable if overlay is partial.
    Result is cached until source JSON mtimes change (avoids O(N) merge per request).
    Returns (merged_mapping, has_any_traduced_overlay).
    """
    global _autocomplete_merged_cache
    data_dir = Path(config.DATA_DIR)
    legacy_path = data_dir / "product_test_database_macros.json"
    trad_path = data_dir / "product_database_macros_traduced.json"
    mt_legacy = legacy_path.stat().st_mtime if legacy_path.is_file() else 0.0
    mt_trad = trad_path.stat().st_mtime if trad_path.is_file() else 0.0

    if _autocomplete_merged_cache is not None:
        _m, _flag, m1, m2, _norms = _autocomplete_merged_cache
        if m1 == mt_legacy and m2 == mt_trad:
            return _m, _flag

    legacy = get_product_macros_mapping()
    trad = _get_product_traduced_macros_mapping(data_dir)
    if not trad:
        norms = _build_autocomplete_row_norms(legacy)
        _autocomplete_merged_cache = (legacy, False, mt_legacy, mt_trad, norms)
        _try_rebuild_autocomplete_fts(legacy, norms, mt_legacy, mt_trad)
        return legacy, False

    merged: dict[str, dict] = {}
    for barcode, rec in legacy.items():
        if not isinstance(barcode, str) or not isinstance(rec, dict):
            continue
        overlay = trad.get(barcode)
        if isinstance(overlay, dict):
            merged[barcode] = {**rec, **overlay}
        else:
            merged[barcode] = dict(rec)
    for barcode, rec in trad.items():
        if barcode in merged or not isinstance(barcode, str) or not isinstance(rec, dict):
            continue
        merged[barcode] = dict(rec)

    norms = _build_autocomplete_row_norms(merged)
    _autocomplete_merged_cache = (merged, True, mt_legacy, mt_trad, norms)
    _try_rebuild_autocomplete_fts(merged, norms, mt_legacy, mt_trad)
    return merged, True


def display_product_name_for_lang(record: dict, lang: str) -> str:
    """Picks name_es / name_en for API responses; falls back to product_name."""
    low = (lang or "es").strip().lower()
    if low not in ("en", "es"):
        low = "es"
    if isinstance(record.get("name_es"), str) or isinstance(record.get("name_en"), str):
        es = record.get("name_es") if isinstance(record.get("name_es"), str) else ""
        en = record.get("name_en") if isinstance(record.get("name_en"), str) else ""
        es = es.strip()
        en = en.strip()
        if low == "en":
            return en or es
        return es or en
    pn = record.get("product_name")
    if isinstance(pn, str) and pn.strip():
        return pn.strip()
    return ""


def _wratio_term_with_boosts(q_norm: str, n: str) -> float:
    """WRatio(q_norm, n) plus same boosts as autocomplete; empty n => 0."""
    if n == "":
        return 0.0
    wr = float(fuzz.WRatio(q_norm, n))
    if n == q_norm:
        wr = max(wr, 100.0)
    elif n.startswith(q_norm):
        wr = max(wr, min(100.0, wr + 10.0))
    elif q_norm in n:
        wr = max(wr, min(100.0, wr + 6.0))
    return wr


def autocomplete_fuzzy_score_bilingual_normalized(q_norm: str, n_left: str, n_right: str) -> float:
    """
    Same as autocomplete_fuzzy_score_bilingual but q_norm and name norms are precomputed.
    n_left / n_right correspond to the two strings iterated in the original API (e.g. es|en and en|es).
    """
    if q_norm == "":
        return 0.0
    if n_left == n_right:
        return _wratio_term_with_boosts(q_norm, n_left)
    return max(
        _wratio_term_with_boosts(q_norm, n_left),
        _wratio_term_with_boosts(q_norm, n_right),
    )


def autocomplete_fuzzy_score_bilingual(query: str, name_es: str, name_en: str) -> float:
    """
    Best fuzzy match (0–100, higher better) against normalized name_es and name_en.
    Boosts exact / prefix / substring matches.
    """
    q_norm = normalize_product_text(query)
    n_es = normalize_product_text(name_es)
    n_en = normalize_product_text(name_en)
    return autocomplete_fuzzy_score_bilingual_normalized(q_norm, n_es, n_en)


def parse_dietary_style(raw_style: Optional[str]) -> Optional["DietaryStyle"]:
    """Parses optional dietary style string into DietaryStyle enum."""
    from src.agents.nutritionist import DietaryStyle

    if raw_style is None:
        return None

    normalized = raw_style.strip().lower()
    if normalized == "":
        return None

    alias_map = {
        "vegan": DietaryStyle.VEGAN,
        "vegetarian": DietaryStyle.VEGETARIAN,
        "celiac": DietaryStyle.CELIAC,
        "mediterranean": DietaryStyle.MEDITERRANEAN,
        "asian": DietaryStyle.ASIAN,
        "arabic": DietaryStyle.ARABIC,
        "latin": DietaryStyle.LATIN,
    }
    return alias_map.get(normalized)


def get_product_name_mapping() -> dict[str, str]:
    """
    Loads barcode -> product name mapping used by the vision module.
    Cached in memory after first read.
    """
    global _product_name_mapping_cache
    if _product_name_mapping_cache is not None:
        return _product_name_mapping_cache

    mapping_path = Path(config.DATA_DIR) / "product_names_mapping.json"
    if not mapping_path.exists():
        _product_name_mapping_cache = {}
        return _product_name_mapping_cache

    with open(mapping_path, "r", encoding="utf-8") as file:
        loaded = json.load(file)
    _product_name_mapping_cache = loaded if isinstance(loaded, dict) else {}
    return _product_name_mapping_cache


def get_product_macros_mapping() -> dict[str, dict]:
    """
    Loads barcode -> { product_name, kcal_per_100g, protein_g_per_100g, ... } for autocomplete.
    Cached in memory after first read.
    """
    global _product_macros_mapping_cache
    if _product_macros_mapping_cache is not None:
        return _product_macros_mapping_cache

    _product_macros_mapping_cache = _get_product_macros_mapping(Path(config.DATA_DIR))
    return _product_macros_mapping_cache


def normalize_product_text(value: str) -> str:
    """
    Normalizes product text for ranking:
    - lowercase
    - accent-insensitive
    - collapse repeated whitespace
    """
    lowered = value.strip().lower()
    if lowered == "":
        return ""
    normalized = unicodedata.normalize("NFKD", lowered)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    compact_spaces = re.sub(r"\s+", " ", without_accents)
    return compact_spaces.strip()


def tokenize_product_text(value: str) -> list[str]:
    """Splits normalized text into non-empty tokens."""
    normalized = normalize_product_text(value)
    if normalized == "":
        return []
    return [token for token in normalized.split(" ") if token != ""]


def score_autocomplete_candidate(query: str, candidate_name: str) -> tuple[int, int, int, str]:
    """
    Returns a sortable score tuple for autocomplete ranking.
    Higher quality => lexicographically smaller tuple.

    Tuple:
      (tier, extra_tokens_penalty, char_overflow_penalty, normalized_name)

    Tiers:
      0: exact normalized full-name match
      1: starts-with query
      2: contains query
      3: non-match (caller should normally ignore)
    """
    normalized_query = normalize_product_text(query)
    normalized_name = normalize_product_text(candidate_name)
    if normalized_query == "" or normalized_name == "":
        return (3, 10**9, 10**9, normalized_name)

    if normalized_name == normalized_query:
        tier = 0
    elif normalized_name.startswith(normalized_query):
        tier = 1
    elif normalized_query in normalized_name:
        tier = 2
    else:
        tier = 3

    query_tokens = tokenize_product_text(normalized_query)
    name_tokens = tokenize_product_text(normalized_name)
    extra_tokens_penalty = max(0, len(name_tokens) - len(query_tokens))
    char_overflow_penalty = max(0, len(normalized_name) - len(normalized_query))
    return (tier, extra_tokens_penalty, char_overflow_penalty, normalized_name)


def score_autocomplete_candidate_normalized(
    normalized_query: str, normalized_name: str
) -> tuple[int, int, int, str]:
    """Like score_autocomplete_candidate when query and name are already normalize_product_text outputs."""
    if normalized_query == "" or normalized_name == "":
        return (3, 10**9, 10**9, normalized_name)

    if normalized_name == normalized_query:
        tier = 0
    elif normalized_name.startswith(normalized_query):
        tier = 1
    elif normalized_query in normalized_name:
        tier = 2
    else:
        tier = 3

    query_tokens = tokenize_product_text(normalized_query)
    name_tokens = tokenize_product_text(normalized_name)
    extra_tokens_penalty = max(0, len(name_tokens) - len(query_tokens))
    char_overflow_penalty = max(0, len(normalized_name) - len(normalized_query))
    return (tier, extra_tokens_penalty, char_overflow_penalty, normalized_name)


def autocomplete_row_fuzzy_score(record: dict, q: str) -> float:
    """
    Same scoring as /api/products/autocomplete loop; returns -1.0 if the row has no usable name.
    """
    es_raw = record.get("name_es") if isinstance(record.get("name_es"), str) else ""
    en_raw = record.get("name_en") if isinstance(record.get("name_en"), str) else ""
    es_raw, en_raw = es_raw.strip(), en_raw.strip()
    has_bilingual = bool(es_raw or en_raw)

    if has_bilingual:
        return autocomplete_fuzzy_score_bilingual(q, es_raw or en_raw, en_raw or es_raw)

    raw_name = record.get("product_name")
    if not isinstance(raw_name, str):
        return -1.0
    pn = raw_name.strip()
    if pn == "":
        return -1.0
    fuzzy = autocomplete_fuzzy_score_bilingual(q, pn, pn)
    legacy = score_autocomplete_candidate(q, pn)
    if legacy[0] < 3:
        fuzzy = max(fuzzy, 82.0)
    return fuzzy


def autocomplete_row_fuzzy_score_norms(q_norm: str, row: AutocompleteRowNorms) -> float:
    """Same numeric score as autocomplete_row_fuzzy_score using precomputed norms."""
    if q_norm == "":
        return 0.0
    if row.has_bilingual:
        return autocomplete_fuzzy_score_bilingual_normalized(q_norm, row.n_bi_left, row.n_bi_right)
    if row.n_pn == "":
        return -1.0
    fuzzy = autocomplete_fuzzy_score_bilingual_normalized(q_norm, row.n_pn, row.n_pn)
    legacy = score_autocomplete_candidate_normalized(q_norm, row.n_pn)
    if legacy[0] < 3:
        fuzzy = max(fuzzy, 82.0)
    return fuzzy


def _autocomplete_score_chunk(
    chunk: list[tuple[int, str, dict]],
    q_norm: str,
    lang_norm: str,
    norms_map: dict[str, AutocompleteRowNorms],
) -> list[tuple[int, float, int, str, dict]]:
    """
    Scores a slice of mapping items (with global iteration index for stable tie-break).
    Returns (exact, fuzzy, idx, barcode, record); exact is 1 when display name matches q normalized.
    """
    rows: list[tuple[int, float, int, str, dict]] = []
    use_en_display = lang_norm.startswith("en")
    for idx, barcode, record in chunk:
        if not isinstance(barcode, str) or not isinstance(record, dict):
            continue
        row = norms_map.get(barcode)
        if row is None:
            row = _norms_for_record(record)
        fuzzy = autocomplete_row_fuzzy_score_norms(q_norm, row)
        if fuzzy < AUTOCOMPLETE_FUZZ_THRESHOLD:
            continue
        n_disp = row.n_disp_en if use_en_display else row.n_disp_es
        if n_disp == "":
            continue
        exact = 1 if n_disp == q_norm else 0
        rows.append((exact, fuzzy, idx, barcode, record))
    return rows


def autocomplete_top_matches(
    mapping: dict[str, dict],
    q: str,
    lang_norm: str,
    limit: int,
) -> list[tuple[float, str, dict]]:
    """
    Returns up to `limit` best (fuzzy, barcode, record).
    `q` should already be normalize_product_text output (as from the API handler); if not, it is normalized once here.

    Ranking: exact normalized display name match first, then descending fuzzy,
    then stable mapping order (earlier item wins on ties).
    Uses parallel chunk scoring + heapq.nlargest.
    """
    get_autocomplete_product_mapping()
    if _autocomplete_merged_cache is None:
        return []
    _, _, mt_legacy, mt_trad, norms_map = _autocomplete_merged_cache

    q_norm = normalize_product_text(q)
    if q_norm == "":
        return []

    items = [
        (i, b, r)
        for i, (b, r) in enumerate(mapping.items())
        if isinstance(b, str) and isinstance(r, dict)
    ]
    n = len(items)
    if n == 0:
        return []

    def rank_key(t: tuple[int, float, int, str, dict]) -> tuple[int, float, int]:
        # nlargest: max exact, then max fuzzy, then min idx (via -idx).
        return (t[0], t[1], -t[2])

    if n < AUTOCOMPLETE_USE_FTS_MIN_ITEMS:
        rows = _autocomplete_score_chunk(items, q_norm, lang_norm, norms_map)
        top = heapq.nlargest(limit, rows, key=rank_key)
        return [(fuzzy, barcode, rec) for _ex, fuzzy, _idx, barcode, rec in top]

    from src.backend import autocomplete_fts

    max_cand = autocomplete_fts.AUTOCOMPLETE_FTS_MAX_CANDIDATES
    candidates = autocomplete_fts.lookup_candidate_rows(
        q_norm, mt_legacy, mt_trad, max_candidates=max_cand
    )
    min_needed = max(3 * limit, autocomplete_fts.AUTOCOMPLETE_FTS_MIN_CANDIDATES_FLOOR)

    use_fts = len(candidates) >= min_needed
    items_to_score = items
    if use_fts:
        items_cand: list[tuple[int, str, dict]] = []
        for barcode, iter_idx in candidates:
            rec = mapping.get(barcode)
            if isinstance(rec, dict):
                items_cand.append((iter_idx, barcode, rec))
        if len(items_cand) >= min_needed:
            items_to_score = items_cand
        else:
            use_fts = False

    n_score = len(items_to_score)
    chunk_size = max(1, (n_score + _AUTOCOMPLETE_CHUNK_WORKERS - 1) // _AUTOCOMPLETE_CHUNK_WORKERS)
    chunks: list[list[tuple[int, str, dict]]] = [
        items_to_score[i : i + chunk_size] for i in range(0, n_score, chunk_size)
    ]

    pool = _get_autocomplete_executor()
    futures = [pool.submit(_autocomplete_score_chunk, ch, q_norm, lang_norm, norms_map) for ch in chunks]
    parts: list[list[tuple[int, float, int, str, dict]]] = []
    for fut in as_completed(futures):
        parts.append(fut.result())

    merged_iter = (row for part in parts for row in part)
    top = heapq.nlargest(limit, merged_iter, key=rank_key)
    return [(fuzzy, barcode, rec) for _ex, fuzzy, _idx, barcode, rec in top]


def calculate_exercise_burned_calories(
    category: str,
    duration_minutes: int,
    weight_kg: float,
    cardio_type: Optional[str] = None,
    intensity: Optional[str] = None,
) -> float:
    """
    Computes calories burned using MET formula:
      kcal = MET × weight_kg × duration_hours
    """
    normalized_category = category.strip().lower()
    duration_hours = duration_minutes / 60.0
    if duration_hours <= 0:
        raise ValueError("duration_minutes must be greater than zero.")

    if normalized_category == "cardio":
        normalized_cardio_type = (cardio_type or "").strip().lower()
        cardio_mets = {
            "walk": 3.5,
            "bike": 7.5,
            "run": 9.8,
            "swim": 8.3,
        }
        if normalized_cardio_type not in cardio_mets:
            raise ValueError("cardio type must be one of: walk, bike, run, swim.")
        met = cardio_mets[normalized_cardio_type]
    elif normalized_category == "strength":
        normalized_intensity = (intensity or "").strip().lower()
        strength_mets = {
            "medium": 4.5,
            "high": 6.0,
            "very_high": 8.0,
        }
        if normalized_intensity not in strength_mets:
            raise ValueError("strength intensity must be one of: medium, high, very_high.")
        met = strength_mets[normalized_intensity]
    else:
        raise ValueError("category must be either 'cardio' or 'strength'.")

    burned = met * float(weight_kg) * duration_hours
    return round(burned, 2)

