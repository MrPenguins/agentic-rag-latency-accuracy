import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import rank_bm25  # noqa: F401
except ModuleNotFoundError:
    stub = types.ModuleType("rank_bm25")
    stub.BM25Okapi = object
    sys.modules["rank_bm25"] = stub

from local_agent_rag import runtime
from local_agent_rag.pipelines.common import merge_stateful_context, parse_search_queries
from local_agent_rag.pipelines.corrective_v1 import should_continue as route_v1
from local_agent_rag.pipelines.corrective_v2 import should_continue as route_v2


class PipelineBehaviorTests(unittest.TestCase):
    def test_imports_do_not_initialize_models(self):
        self.assertIsNone(runtime._resources)

    def test_query_parser_and_fallback(self):
        self.assertEqual(parse_search_queries("first | second", "original"), ["first", "second"])
        self.assertEqual(parse_search_queries(" | ", "original"), ["original"])

    def test_corrective_retry_limit(self):
        for router in (route_v1, route_v2):
            self.assertEqual(router({"feedback": None, "loop_count": 0}), "end")
            self.assertEqual(router({"feedback": "missing fact", "loop_count": 0}), "retry")
            self.assertEqual(router({"feedback": "missing fact", "loop_count": 3}), "end")

    def test_stateful_context_leaves_one_new_slot(self):
        result = merge_stateful_context(["old-1", "old-2"], ["new"], ["fallback"], 2)
        self.assertEqual(result, ["old-1", "new"])


if __name__ == "__main__":
    unittest.main()
