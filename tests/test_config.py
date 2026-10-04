import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local_agent_rag.config import load_config


class ConfigTests(unittest.TestCase):
    def test_all_reported_configs_preserve_protocol(self):
        for path in sorted((ROOT / "configs").glob("*.yaml")):
            config = load_config(path)
            self.assertEqual(config["database"]["k_retrieval"], 2)
            self.assertEqual(config["database"]["hybrid_alpha"], 0.5)
            self.assertEqual(config["models"]["max_retries"], 3)
            self.assertEqual(config["models"]["llm_temperature"], 0)
            self.assertFalse(config["models"]["reasoning"])
            self.assertTrue(Path(config["paths"]["vectorstore"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
