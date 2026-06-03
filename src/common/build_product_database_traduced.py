"""
Offline: read-only import from data/product_test_database_macros.json and write
data/product_database_macros_traduced.json with name_es + name_en per barcode.

Speed: translates each *unique product name* once (dedup), reuses across all barcodes;
persists a sidecar translation cache; parallel API calls with bounded workers.

Does NOT modify the source JSON. Uses deep-translator (Google).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

# Project root: .../TFG
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
SRC = DATA / "product_test_database_macros.json"
OUT = DATA / "product_database_macros_traduced.json"
# Reusable translations: normalized product_name -> {name_es, name_en}
CACHE = DATA / "product_translation_cache.json"

_print_lock = Lock()


def _normalize_text(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    return unicodedata.normalize("NFC", t)


def _translate_pair(text: str) -> tuple[str, str]:
    """
    Returns (name_es, name_en). Pivot: auto->en, then en->es; optional auto->es if still equal.
    Same logic as before for accuracy; inner sleeps reduced (parallelism provides spacing).
    """
    t = _normalize_text(text)
    if not t:
        return "", ""
    try:
        from deep_translator import GoogleTranslator

        name_en = GoogleTranslator(source="auto", target="en").translate(t)
        name_en = _normalize_text(name_en or t)
        time.sleep(0.04)

        name_es = GoogleTranslator(source="en", target="es").translate(name_en)
        name_es = _normalize_text(name_es or name_en)

        if name_es == name_en and len(t) > 8:
            time.sleep(0.04)
            alt_es = GoogleTranslator(source="auto", target="es").translate(t)
            alt_es = _normalize_text(alt_es or "")
            if alt_es and alt_es != name_en:
                name_es = alt_es

        return name_es, name_en
    except Exception:
        return t, t


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bilingual product JSON (fast: dedup + cache + parallel).")
    parser.add_argument("--limit", type=int, default=None, help="Max *new barcodes* to add this run (default: all missing).")
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Parallel translation workers (default 10). Lower if Google rate-limits.",
    )
    parser.add_argument(
        "--checkpoint-barcodes",
        type=int,
        default=5000,
        help="Write OUT + cache to disk every N new barcodes appended.",
    )
    args = parser.parse_args()
    workers = max(1, min(32, args.workers))

    if not SRC.is_file():
        print(f"Source not found: {SRC}", file=sys.stderr)
        return 1

    print(f"Loading {SRC} ...", flush=True)
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print("Source JSON must be an object keyed by barcode.", file=sys.stderr)
        return 1

    existing_out = _load_json(OUT)
    print(f"Resume OUT: {len(existing_out)} barcodes.", flush=True)

    translation_cache: dict[str, dict] = _load_json(CACHE)
    # Normalize cache keys (older runs might differ slightly)
    normalized_cache: dict[str, dict] = {}
    for k, v in translation_cache.items():
        nk = _normalize_text(k) if isinstance(k, str) else ""
        if nk and isinstance(v, dict) and "name_es" in v and "name_en" in v:
            normalized_cache[nk] = {"name_es": str(v["name_es"]), "name_en": str(v["name_en"])}
    translation_cache = normalized_cache
    print(f"Translation cache: {len(translation_cache)} unique names.", flush=True)

    # Pending rows: (barcode, record, name_key)
    pending: list[tuple[str, dict, str]] = []
    for barcode, record in raw.items():
        if not isinstance(barcode, str) or not isinstance(record, dict):
            continue
        if barcode in existing_out:
            continue
        pname = record.get("product_name")
        if not isinstance(pname, str) or not pname.strip():
            continue
        key = _normalize_text(pname)
        if not key:
            continue
        pending.append((barcode, record, key))
        if args.limit is not None and len(pending) >= args.limit:
            break

    if not pending:
        print("Nothing to do (all barcodes already in output or no valid names).", flush=True)
        return 0

    print(f"Pending barcodes this run: {len(pending)}", flush=True)

    # Unique name keys that still need translation
    unique_keys: set[str] = set()
    for _, _, key in pending:
        unique_keys.add(key)

    missing = [k for k in unique_keys if k not in translation_cache]
    print(f"Unique product names in batch: {len(unique_keys)}; need API for: {len(missing)}", flush=True)

    def translate_one(k: str) -> tuple[str, str, str]:
        es, en = _translate_pair(k)
        return k, es, en

    if missing:
        done = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(translate_one, k): k for k in missing}
            for fut in as_completed(futures):
                try:
                    key, es, en = fut.result()
                    translation_cache[key] = {"name_es": es, "name_en": en}
                except Exception:
                    k = futures[fut]
                    translation_cache[k] = {"name_es": k, "name_en": k}
                done += 1
                if done % 100 == 0 or done == len(missing):
                    elapsed = time.perf_counter() - t0
                    with _print_lock:
                        print(
                            f"  Translated {done}/{len(missing)} unique names in {elapsed:.1f}s "
                            f"({done / elapsed:.1f}/s)",
                            flush=True,
                        )
                    _save_json(CACHE, translation_cache)

        _save_json(CACHE, translation_cache)
        print(f"Cache saved: {CACHE}", flush=True)

    # Build output rows (macros per barcode from source)
    out: dict[str, dict] = dict(existing_out)
    new_count = 0
    for barcode, record, key in pending:
        pair = translation_cache.get(key)
        if not pair:
            pair = {"name_es": key, "name_en": key}
        out[barcode] = {
            "name_es": pair["name_es"],
            "name_en": pair["name_en"],
            "kcal_per_100g": record.get("kcal_per_100g"),
            "protein_g_per_100g": record.get("protein_g_per_100g"),
            "fat_g_per_100g": record.get("fat_g_per_100g"),
            "carbs_g_per_100g": record.get("carbs_g_per_100g"),
        }
        new_count += 1
        if new_count % args.checkpoint_barcodes == 0:
            _save_json(OUT, out)
            print(f"Checkpoint OUT: {len(out)} barcodes.", flush=True)

    _save_json(OUT, out)
    print(f"Done. OUT barcodes: {len(out)} (new this run: {new_count}).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
