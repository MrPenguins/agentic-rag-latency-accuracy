import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_agent_rag.evaluation.significance import evaluate_mcnemar, mcnemar_counts


class SignificanceTests(unittest.TestCase):
    def test_known_counts(self):
        result = mcnemar_counts([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(result["discordant_pairs"], 2)
        self.assertEqual(result["pvalue"], 1.0)

    def test_qwen4_reported_pvalues(self):
        results = evaluate_mcnemar(ROOT / "results" / "judged" / "qwen3.5_4b.csv")
        self.assertEqual(f"{results['linear_v1']['pvalue']:.3f}", "0.051")
        self.assertEqual(f"{results['corrective_v2']['pvalue']:.3f}", "0.012")


if __name__ == "__main__":
    unittest.main()
