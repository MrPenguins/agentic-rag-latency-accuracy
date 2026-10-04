import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_agent_rag.evaluation.metrics import evaluate_csv, exact_match_score, f1_score


class MetricTests(unittest.TestCase):
    def test_normalized_answer_metrics(self):
        self.assertEqual(exact_match_score("The Eiffel Tower.", "Eiffel Tower"), 1)
        self.assertAlmostEqual(f1_score("red blue", "red green"), 0.5)

    def test_llama_reported_values(self):
        metrics = evaluate_csv(ROOT / "results" / "judged" / "llama3.1_8b.csv")
        self.assertEqual(metrics["questions"], 500)
        self.assertAlmostEqual(metrics["pipelines"]["baseline"]["llm_accuracy"], 54.8)
        self.assertAlmostEqual(metrics["pipelines"]["corrective_v1"]["context_recall"], 70.8)
        self.assertEqual(round(metrics["pipelines"]["linear_v1"]["efficiency"], 2), 4.22)


if __name__ == "__main__":
    unittest.main()
