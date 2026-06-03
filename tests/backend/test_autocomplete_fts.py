"""FTS5 autocomplete rebuild/lookup and parity with full-scan scoring (small catalog)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.backend import autocomplete_fts, config
from src.backend.utils import (
    _build_autocomplete_row_norms,
    autocomplete_top_matches,
    normalize_product_text,
)


def _tiny_mapping() -> dict[str, dict]:
    # Shared token "xyzbase" so a short prefix query matches all rows via FTS.
    return {
        "111": {
            "name_es": "alpha xyzbase uno",
            "name_en": "alpha xyzbase one",
            "product_name": "pn1",
        },
        "222": {
            "name_es": "beta xyzbase dos",
            "name_en": "beta xyzbase two",
            "product_name": "pn2",
        },
        "333": {
            "name_es": "gamma xyzbase tres",
            "name_en": "gamma xyzbase three",
            "product_name": "pn3",
        },
        "444": {
            "name_es": "delta xyzbase cuatro",
            "name_en": "delta xyzbase four",
            "product_name": "pn4",
        },
        "555": {
            "name_es": "epsilon xyzbase cinco",
            "name_en": "epsilon xyzbase five",
            "product_name": "pn5",
        },
    }


class AutocompleteFtsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.fts_path = Path(self._td.name) / "autocomplete_fts_test.sqlite"

    def test_build_match_query_escapes_and_prefix(self) -> None:
        self.assertIsNone(autocomplete_fts.build_match_query(""))
        self.assertIsNone(autocomplete_fts.build_match_query("   "))
        q = autocomplete_fts.build_match_query('foo "bar')
        self.assertIn('OR', q or '')
        self.assertIn('""', q or '')

    def test_rebuild_lookup_iter_idx_and_meta(self) -> None:
        merged = _tiny_mapping()
        norms = _build_autocomplete_row_norms(merged)
        mt_l, mt_t = 100.0, 200.0
        with mock.patch.object(config, "AUTOCOMPLETE_FTS_DB_PATH", self.fts_path):
            autocomplete_fts.rebuild_autocomplete_fts(merged, norms, mt_l, mt_t)
            rows = autocomplete_fts.lookup_candidate_rows(
                normalize_product_text("xyzbase"),
                mt_l,
                mt_t,
                max_candidates=50,
            )
        barcodes = [b for b, _ in rows]
        self.assertEqual(set(barcodes), set(merged.keys()))
        by_bc = dict(rows)
        for i, (bc, _) in enumerate(merged.items()):
            if bc in by_bc:
                self.assertEqual(by_bc[bc], i, msg=f"iter_idx mismatch for {bc}")

    def test_lookup_stale_meta_returns_empty(self) -> None:
        merged = _tiny_mapping()
        norms = _build_autocomplete_row_norms(merged)
        with mock.patch.object(config, "AUTOCOMPLETE_FTS_DB_PATH", self.fts_path):
            autocomplete_fts.rebuild_autocomplete_fts(merged, norms, 1.0, 2.0)
            out = autocomplete_fts.lookup_candidate_rows(
                normalize_product_text("xyzbase"),
                999.0,
                2.0,
                max_candidates=50,
            )
        self.assertEqual(out, [])

    def test_top_matches_fts_parity_vs_full_scan(self) -> None:
        import src.backend.utils as u

        merged = _tiny_mapping()
        norms = _build_autocomplete_row_norms(merged)
        mt_l, mt_t = 10.0, 20.0
        q = normalize_product_text("xyzbase")
        lang = "es"

        with mock.patch.object(config, "AUTOCOMPLETE_FTS_DB_PATH", self.fts_path):
            autocomplete_fts.rebuild_autocomplete_fts(merged, norms, mt_l, mt_t)

            u._autocomplete_merged_cache = (merged, False, mt_l, mt_t, norms)
            # Small catalog: lower floor so 5 FTS hits >= min_needed (max(3*limit, floor)).
            with mock.patch.object(autocomplete_fts, "AUTOCOMPLETE_FTS_MIN_CANDIDATES_FLOOR", 1):
                with mock.patch.object(u, "AUTOCOMPLETE_USE_FTS_MIN_ITEMS", 1):
                    with mock.patch.object(
                        u,
                        "get_autocomplete_product_mapping",
                        lambda: (merged, False),
                    ):
                        fts_results = autocomplete_top_matches(merged, q, lang, 1)

            u._autocomplete_merged_cache = (merged, False, mt_l, mt_t, norms)
            with mock.patch.object(autocomplete_fts, "AUTOCOMPLETE_FTS_MIN_CANDIDATES_FLOOR", 1):
                with mock.patch.object(u, "AUTOCOMPLETE_USE_FTS_MIN_ITEMS", 1):
                    with mock.patch.object(
                        u,
                        "get_autocomplete_product_mapping",
                        lambda: (merged, False),
                    ):
                        with mock.patch.object(
                            autocomplete_fts,
                            "lookup_candidate_rows",
                            lambda *a, **k: [],
                        ):
                            full_results = autocomplete_top_matches(merged, q, lang, 1)

        self.assertEqual(
            [bc for _f, bc, _r in fts_results],
            [bc for _f, bc, _r in full_results],
        )
        self.assertEqual(
            [f for f, _bc, _r in fts_results],
            [f for f, _bc, _r in full_results],
        )

    def tearDown(self) -> None:
        import src.backend.utils as u

        u._autocomplete_merged_cache = None


if __name__ == "__main__":
    unittest.main()
