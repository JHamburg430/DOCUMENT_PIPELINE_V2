import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from apps.ui import server


class AgentRepairRequestTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.job_id = "agent-run-012345abcdef"
        self.job = {
            "id": self.job_id,
            "surface": "chat",
            "status": "failed",
            "query": "Why did this fail?",
            "runs": {"langgraph_agent": {"error": "test failure", "events": [{"event": "run_failed"}]}},
        }
        self.path_patch = patch.object(server, "AGENT_REPAIR_REQUESTS_DIR", Path(self.temp_dir.name))
        self.jobs_patch = patch.dict(server.AGENT_LIVE_JOBS, {self.job_id: self.job}, clear=True)
        self.path_patch.start()
        self.jobs_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.addCleanup(self.jobs_patch.stop)

    def test_saves_exact_run_and_repeated_submission_is_idempotent(self):
        first = server._save_agent_repair_request(self.job_id)
        saved = json.loads(server._agent_repair_request_path(self.job_id).read_text())
        self.assertEqual(saved["source_run_id"], self.job_id)
        self.assertEqual(saved["run"]["runs"]["langgraph_agent"]["events"][0]["event"], "run_failed")
        self.assertEqual(first, saved)
        self.job["query"] = "changed later"
        self.assertEqual(server._save_agent_repair_request(self.job_id), first)

    def test_rejects_running_or_unrelated_jobs(self):
        self.job["status"] = "running"
        with self.assertRaisesRegex(ValueError, "finish"):
            server._save_agent_repair_request(self.job_id)
        self.job["status"] = "failed"
        self.job["surface"] = "lab"
        with self.assertRaisesRegex(ValueError, "Agent page"):
            server._save_agent_repair_request(self.job_id)

    def test_rejects_invalid_id(self):
        with self.assertRaisesRegex(ValueError, "valid Agent run ID"):
            server._save_agent_repair_request("../../etc/passwd")


if __name__ == "__main__":
    unittest.main()
