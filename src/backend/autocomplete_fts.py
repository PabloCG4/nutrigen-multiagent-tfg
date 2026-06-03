"""
SQLite FTS5 index for autocomplete candidate prefiltering.
Rebuild stays in sync with merged catalog JSON mtimes (see autocomplete_fts_meta).
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from src.backend import config

if TYPE_CHECKING:
    from src.backend.utils import AutocompleteRowNorms

# Tunables (integration uses these from autocomplete_top_matches).
AUTOCOMPLETE_FTS_MAX_CANDIDATES = 12_000
# Minimum candidate count to trust FTS prefilter vs full scan (plan: max(3*limit, floor)).
AUTOCOMPLETE_FTS_MIN_CANDIDATES_FLOOR = 40

_FTS_TABLE = "product_autocomplete_fts"
_META_TABLE = "autocomplete_fts_meta"


def search_text_from_norms(row: AutocompleteRowNorms) -> str:
    """Concatenate normalized fields aligned with _norms_for_record / AutocompleteRowNorms."""
    parts: list[str] = []
    if row.has_bilingual:
        parts.extend([row.n_bi_left, row.n_bi_right, row.n_disp_es, row.n_disp_en])
    parts.append(row.n_pn)
    return " ".join(p for p in parts if p).strip()


def _escape_fts5_token(token: str) -> str:
    """Wrap a token for FTS5 prefix match; escape double quotes per FTS5 rules."""
    t = token.strip()
    if not t:
        return ""
    escaped = t.replace('"', '""')
    return f'"{escaped}"*'


def build_match_query(q_norm: str) -> str | None:
    """OR of prefix terms; None if there is no usable token."""
    tokens = [t for t in q_norm.split() if t.strip()]
    if not tokens:
        return None
    parts: list[str] = []
    for tok in tokens:
        esc = _escape_fts5_token(tok)
        if esc:
            parts.append(esc)
    if not parts:
        return None
    return " OR ".join(parts)


def _meta_matches(conn: sqlite3.Connection, mt_legacy: float, mt_trad: float) -> bool:
    cur = conn.execute(
        f"SELECT key, value FROM {_META_TABLE} WHERE key IN ('legacy_mtime', 'trad_mtime')"
    )
    rows = {k: v for k, v in cur.fetchall()}
    return rows.get("legacy_mtime") == str(mt_legacy) and rows.get("trad_mtime") == str(mt_trad)


def rebuild_autocomplete_fts(
    merged: dict[str, dict],
    norms_map: dict[str, AutocompleteRowNorms],
    mt_legacy: float,
    mt_trad: float,
) -> None:
    """Rebuild the FTS file from merged mapping and norms (same iter_idx as enumerate(merged.items()))."""
    path = config.AUTOCOMPLETE_FTS_DB_PATH
    batch: list[tuple[str, int, str]] = []
    for i, (barcode, rec) in enumerate(merged.items()):
        if not isinstance(barcode, str) or not isinstance(rec, dict):
            continue
        row = norms_map.get(barcode)
        if row is None:
            continue
        st = search_text_from_norms(row)
        if not st:
            continue
        batch.append((barcode, i, st))

    try:
        conn = sqlite3.connect(str(path), timeout=60.0)
    except OSError:
        return
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            f"""
            DROP TABLE IF EXISTS {_FTS_TABLE};
            DROP TABLE IF EXISTS {_META_TABLE};
            CREATE VIRTUAL TABLE {_FTS_TABLE} USING fts5(
              barcode UNINDEXED,
              iter_idx UNINDEXED,
              search_text,
              tokenize='unicode61'
            );
            CREATE TABLE {_META_TABLE} (
              key TEXT PRIMARY KEY NOT NULL,
              value TEXT NOT NULL
            );
            """
        )
        conn.executemany(
            f"INSERT INTO {_FTS_TABLE}(barcode, iter_idx, search_text) VALUES (?,?,?)",
            batch,
        )
        conn.executemany(
            f"INSERT INTO {_META_TABLE}(key, value) VALUES (?, ?)",
            [("legacy_mtime", str(mt_legacy)), ("trad_mtime", str(mt_trad))],
        )
        conn.commit()
    except (OSError, sqlite3.Error):
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def lookup_candidate_rows(
    q_norm: str,
    mt_legacy: float,
    mt_trad: float,
    *,
    max_candidates: int,
) -> list[tuple[str, int]]:
    """
    Return (barcode, iter_idx) from FTS ordered by rank, capped at max_candidates.
    Empty list if DB missing, stale meta, query not buildable, or errors.
    """
    path = config.AUTOCOMPLETE_FTS_DB_PATH
    if not path.is_file():
        return []
    match_q = build_match_query(q_norm)
    if match_q is None:
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    try:
        if not _meta_matches(conn, mt_legacy, mt_trad):
            return []
        sql = (
            f"SELECT barcode, iter_idx FROM {_FTS_TABLE} "
            f"WHERE {_FTS_TABLE} MATCH ? ORDER BY rank LIMIT ?"
        )
        cur = conn.execute(sql, (match_q, max_candidates))
        out: list[tuple[str, int]] = []
        for barcode, iter_idx in cur.fetchall():
            if isinstance(barcode, str) and isinstance(iter_idx, int):
                out.append((barcode, iter_idx))
        return out
    except sqlite3.Error:
        return []
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass
