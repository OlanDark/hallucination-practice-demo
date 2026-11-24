"""Автономные проверки 6-го этапа: API Groq не вызывается."""

import json
import tempfile
import unittest
from pathlib import Path

from experiment import build_examples, load_dataset, metrics, parse_baseline, stable_split, read_results, write_record


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for n in range(100):
            self.rows.append({"knowledge": "Knowledge " + str(n),
                              "question": "Question unique " + str(n),
                              "right_answer": "Correct " + str(n),
                              "hallucinated_answer": "Incorrect " + str(n)})

    def test_stable_split(self):
        self.assertEqual(stable_split("Some question"), stable_split("Some question"))
        self.assertIn(stable_split("Some question"), ("dev", "test"))

    def test_pair_balance_and_determinism(self):
        a = build_examples(self.rows, split="dev", pairs=5, seed=42)
        b = build_examples(self.rows, split="dev", pairs=5, seed=42)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 10)
        self.assertEqual(sum(x["true_label"] for x in a), 5)
        counts = {}
        for x in a:
            counts[x["pair_id"]] = counts.get(x["pair_id"], 0) + 1
        self.assertEqual(set(counts.values()), {2})

    def test_no_dev_test_overlap(self):
        dev = {r["question"] for r in build_examples(self.rows, "dev", 5)}
        test = {r["question"] for r in build_examples(self.rows, "test", 5)}
        self.assertFalse(dev & test)

    def test_jsonl_loading(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "qa.json"
            path.write_text("\n".join(json.dumps(r) for r in self.rows[:3]), encoding="utf-8")
            self.assertEqual(len(load_dataset(path)), 3)

    def test_json_array_loading(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "qa.json"
            path.write_text(json.dumps(self.rows[:3]), encoding="utf-8")
            self.assertEqual(len(load_dataset(path)), 3)

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "results.jsonl"
            write_record(path, {"id": "x", "status": "ok"})
            self.assertEqual(read_results(path)[0]["id"], "x")

    def test_baseline_parser(self):
        self.assertTrue(parse_baseline('{"hallucination":true}'))
        self.assertFalse(parse_baseline('{"hallucination": false}'))
        self.assertIsNone(parse_baseline("it might be false"))
        self.assertIsNone(parse_baseline('{"hallucination": "false"}'))

    def test_metrics_with_abstain(self):
        rows = [
            {"status": "ok", "true_label": 1, "predicted": True},
            {"status": "ok", "true_label": 0, "predicted": False},
            {"status": "ok", "true_label": 1, "predicted": None},
            {"status": "error", "true_label": 1, "predicted": True},
        ]
        m = metrics(rows)
        self.assertEqual(m["classified"], 2)
        self.assertEqual(m["samples"], 3)
        self.assertEqual(m["accuracy_classified"], 1.0)
        self.assertEqual(m["accuracy_all_abstentions_wrong"], round(2/3, 4))


if __name__ == "__main__":
    unittest.main()
