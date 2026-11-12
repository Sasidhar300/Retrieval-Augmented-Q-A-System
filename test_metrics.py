"""Small, offline regression tests for our diagnostic QA metrics."""

import unittest
from evaluation_metrics import normalize_answer, exact_match, token_f1


class EvaluationMetricsTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_answer("The, Blue   Cars!"), "blue cars")

    def test_exact_match_uses_alternative_references(self):
        self.assertEqual(exact_match("A whale.", ["shark", "the whale"]), 1.0)
        self.assertEqual(exact_match("a whale", ["dolphin"]), 0.0)

    def test_token_f1_partial_overlap(self):
        self.assertAlmostEqual(
            token_f1("blue green", ["blue red"]), 0.5
        )

    def test_empty_answer_pair(self):
        self.assertEqual(token_f1("", [""]), 1.0)
        self.assertEqual(token_f1("", ["something"]), 0.0)

    def test_missing_reference_rejected(self):
        with self.assertRaises(ValueError):
            exact_match("answer", [])
        with self.assertRaises(ValueError):
            token_f1("answer", [])


if __name__ == "__main__":
    unittest.main()
