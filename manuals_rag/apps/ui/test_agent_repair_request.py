import unittest
from unittest.mock import patch

from apps.ui import repair_store, server


class AgentRepairRequestTest(unittest.TestCase):
    def setUp(self):
        self.job_id = "agent-run-012345abcdef"
        self.job = {
            "id": self.job_id,
            "surface": "chat",
            "status": "failed",
            "query": "Why did this fail?",
            "runs": {"langgraph_agent": {"error": "test failure", "events": [{"event": "run_failed"}]}},
        }
        self.jobs_patch = patch.dict(server.AGENT_LIVE_JOBS, {self.job_id: self.job}, clear=True)
        self.jobs_patch.start()
        self.addCleanup(self.jobs_patch.stop)

    @patch.object(server, "_db_save_agent_repair_case")
    @patch.object(server, "_db_get_agent_repair_case", return_value=None)
    def test_saves_server_owned_run_in_database(self, _read, save):
        save.return_value = {"request_id": f"repair-{self.job_id}"}
        result = server._save_agent_repair_request(self.job_id)
        self.assertEqual(result["request_id"], f"repair-{self.job_id}")
        self.assertEqual(save.call_args.args[1]["runs"]["langgraph_agent"]["events"][0]["event"], "run_failed")

    @patch.object(server, "_db_enqueue_agent_repair_handoff", return_value={"request_id": "existing", "gateway_handoff_status": "sent"})
    @patch.object(server, "_db_get_agent_repair_case", return_value={"request_id": "existing"})
    def test_repeated_submission_is_idempotent(self, _read, enqueue):
        self.assertEqual(server._save_agent_repair_request(self.job_id)["gateway_handoff_status"], "sent")
        enqueue.assert_called_once_with(server.POSTGRES_DSN, self.job_id)

    @patch.object(server, "_db_get_agent_repair_case", return_value=None)
    def test_rejects_unrelated_jobs(self, _read):
        self.job["surface"] = "lab"
        with self.assertRaisesRegex(ValueError, "Agent page"):
            server._save_agent_repair_request(self.job_id)

    def test_rejects_invalid_id(self):
        with self.assertRaisesRegex(ValueError, "valid Agent run ID"):
            repair_store.validate_run_id("../../etc/passwd")

    def test_regression_contract_requires_measurable_assertion(self):
        with self.assertRaises(ValueError):
            repair_store.validate_regression_spec({})
        with self.assertRaises(ValueError):
            repair_store.validate_regression_spec({"required_answer_terms": []})
        self.assertEqual(
            repair_store.validate_regression_spec({"expect_abstention": False, "required_answer_terms": ["24 VDC"]}),
            {"expect_abstention": False, "required_answer_terms": ["24 VDC"]},
        )


if __name__ == "__main__":
    unittest.main()
