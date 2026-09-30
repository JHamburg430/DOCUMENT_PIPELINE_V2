import unittest

from scripts.benchmark.agent_repair_regressions import evaluate_regression


class RepairRegressionContractTest(unittest.TestCase):
    def setUp(self):
        self.spec = {
            "expect_abstention": False,
            "required_answer_terms": ["24 VDC"],
            "required_citation_document_ids": ["manual-1"],
            "required_event_types": ["run_completed"],
        }
        self.events = [{"event": "run_completed", "result": {
            "answer": "Use 24 VDC.", "insufficient_evidence": False,
            "citations": [{"document_id": "manual-1"}],
        }}]

    def test_matching_fix_passes(self):
        passed, failures, _ = evaluate_regression(self.spec, self.events)
        self.assertTrue(passed)
        self.assertEqual(failures, [])

    def test_changed_answer_or_missing_citation_fails(self):
        self.events[0]["result"] = {"answer": "Use 12 VDC.", "insufficient_evidence": False, "citations": []}
        passed, failures, _ = evaluate_regression(self.spec, self.events)
        self.assertFalse(passed)
        self.assertTrue(any("answer term" in item for item in failures))
        self.assertTrue(any("cited document" in item for item in failures))

    def test_timeout_cannot_count_as_pass(self):
        passed, failures, _ = evaluate_regression(self.spec, [{"event": "run_timed_out"}])
        self.assertFalse(passed)
        self.assertIn("No completed Agent result", failures)


if __name__ == "__main__":
    unittest.main()
