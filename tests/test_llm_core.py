from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_ake.cleaning import clean_fulltext, compose_document
from llm_ake.evaluate import _counts, normalise_phrase
from llm_ake.parser import ordered_unique_after_top_k, parse_numbered_list, source_presence
from llm_ake.prompt import load_prompt


class ParserTests(unittest.TestCase):
    def test_numbered_list_only(self):
        text = 'Intro\n1. "machine learning"\n2. “neural networks”\n- ignored\n11. ignored'
        self.assertEqual(parse_numbered_list(text), ["machine learning", "neural networks"])

    def test_topk_before_dedup(self):
        values = ["A", "a", "B", "C"]
        self.assertEqual(ordered_unique_after_top_k(values, 2), ["a"])

    def test_source_presence(self):
        result = source_presence("Neural Networks", "We study neural networks.")
        self.assertFalse(result["exact"])
        self.assertTrue(result["nfkc_casefold"])


class CleaningTests(unittest.TestCase):
    def test_keyword_and_reference_removal(self):
        text = "INTRODUCTION\n\nKeywords: secret term\n\nMETHOD\n\nbody\n\nREFERENCES\n\nleaked citation"
        cleaned, audit = clean_fulltext(text)
        self.assertNotIn("secret term", cleaned)
        self.assertNotIn("leaked citation", cleaned)
        self.assertTrue(audit["references_truncated"])

    def test_views_share_title_abstract(self):
        record = {"title": "T", "abstract": "A", "fulltext": "METHOD\n\nF"}
        ta, _ = compose_document(record, "TA", 10000)
        ft, _ = compose_document(record, "FT", 10000)
        self.assertTrue(ft.startswith(ta))


class EvaluationTests(unittest.TestCase):
    def test_token_normalization(self):
        self.assertEqual(normalise_phrase("Model-based AKE"), "model-based ake")

    def test_micro_counts_topk_then_set(self):
        self.assertEqual(_counts(["A", "a", "B"], ["a", "b"], 2), (1, 1, 2))


class PromptTests(unittest.TestCase):
    def test_prompt_split(self):
        system, user, full = load_prompt(ROOT / "prompts" / "ZEROSHOT_PROMPT_v1.txt")
        self.assertIn("academic keyphrase extraction", system)
        self.assertIn("{document}", user)
        self.assertTrue(full.endswith("\n"))


if __name__ == "__main__":
    unittest.main()

