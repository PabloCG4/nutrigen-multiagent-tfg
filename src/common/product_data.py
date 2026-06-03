from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_product_macros_mapping_cache: Optional[dict[str, dict]] = None
_product_traduced_macros_mapping_cache: Optional[dict[str, dict]] = None


def get_product_macros_mapping(data_dir: Path) -> dict[str, dict]:
    """
    Loads barcode -> { product_name, kcal_per_100g, protein_g_per_100g, ... }.
    Cached in memory after first read.
    """
    global _product_macros_mapping_cache
    if _product_macros_mapping_cache is not None:
        return _product_macros_mapping_cache

    mapping_path = data_dir / "product_test_database_macros.json"
    if not mapping_path.exists():
        _product_macros_mapping_cache = {}
        return _product_macros_mapping_cache

    try:
        loaded = json.loads(mapping_path.read_text(encoding="utf-8"))
        _product_macros_mapping_cache = loaded if isinstance(loaded, dict) else {}
    except Exception:
        _product_macros_mapping_cache = {}
    return _product_macros_mapping_cache


def get_product_traduced_macros_mapping(data_dir: Path) -> dict[str, dict]:
    """
    Bilingual catalog for autocomplete: barcode -> { name_es, name_en, macros... }.
    File: product_database_macros_traduced.json (generated offline; optional).
    """
    global _product_traduced_macros_mapping_cache
    if _product_traduced_macros_mapping_cache is not None:
        return _product_traduced_macros_mapping_cache

    path = data_dir / "product_database_macros_traduced.json"
    if not path.exists():
        _product_traduced_macros_mapping_cache = {}
        return _product_traduced_macros_mapping_cache

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        _product_traduced_macros_mapping_cache = loaded if isinstance(loaded, dict) else {}
    except Exception:
        _product_traduced_macros_mapping_cache = {}
    return _product_traduced_macros_mapping_cache

