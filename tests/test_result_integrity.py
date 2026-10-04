import csv
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
csv.field_size_limit(2**31 - 1)


class ResultIntegrityTests(unittest.TestCase):
    def test_all_formal_results_match_fixed_subset(self):
        subset = json.loads((ROOT / "data" / "benchmark_subset_500.json").read_text(encoding="utf-8"))
        subset_ids = [item["_id"] for item in subset]
        self.assertEqual(len(subset_ids), 500)
        self.assertEqual(len(set(subset_ids)), 500)
        for kind in ("raw", "judged"):
            for path in sorted((ROOT / "results" / kind).glob("*.csv")):
                with path.open("r", encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                ids = [row["question_id"] for row in rows]
                self.assertEqual(len(ids), 500, path.name)
                self.assertEqual(set(ids), set(subset_ids), path.name)
                if kind == "raw":
                    self.assertEqual(ids, subset_ids, path.name)


if __name__ == "__main__":
    unittest.main()
