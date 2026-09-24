import json
import unittest
from pathlib import Path


UI_DIR = Path(__file__).resolve().parent


class EvaluationWorkspaceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        cls.js = (UI_DIR / "app.js").read_text(encoding="utf-8")
        cls.css = (UI_DIR / "styles.css").read_text(encoding="utf-8")
        cls.fixture = json.loads(
            (UI_DIR / "fixtures" / "evaluation-realtime.json").read_text(encoding="utf-8")
        )

    def test_unified_workspace_retains_all_agent_lab_controls(self):
        self.assertIn('data-tab="evaluation"', self.html)
        self.assertNotIn('data-tab="agent-lab"', self.html)
        for element_id in (
            "question-matrix-workspace",
            "agent-lab",
            "run-agent-test",
            "agent-results",
            "agent-matrix-workspace",
            "run-agent-matrix",
            "refresh-agent-matrix",
            "replay-agent-matrix-fixture",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("function setupEvaluationWorkspace()", self.js)
        self.assertIn("workspace.appendChild(agentLab)", self.js)

    def test_top_matrix_progress_bars_are_preserved(self):
        self.assertIn("function renderMatrixSummary(totals = {}, totalRows = 0)", self.js)
        self.assertIn('class="matrix-bar"', self.js)
        self.assertIn("renderMatrixSummary(totals, rows.length)", self.js)

    def test_fixture_drives_every_backend_layer_from_provisional_to_final(self):
        layers = {
            "tool_selection",
            "candidate_recall",
            "document_retention",
            "hop_dependencies",
            "evidence_sufficiency",
            "grounded_answer",
            "latency_token_cost",
        }
        events = self.fixture["events"]
        for backend in ("langgraph", "llamaindex"):
            for layer in layers:
                statuses = [
                    event["cell"]["status"]
                    for event in events
                    if event["backend"] == backend and event["layer"] == layer
                ]
                self.assertEqual(["provisional", "pass"], statuses)
        self.assertIn("applyAgentMatrixLiveCell(event)", self.js)
        self.assertIn(".matrix-cell.provisional", self.css)

    def test_live_job_snapshots_reuse_the_same_provisional_cell_contract(self):
        self.assertIn("mergeAgentMatrixJobSnapshot(job)", self.js)
        self.assertIn('status: "provisional"', self.js)
        self.assertIn('label: "LIVE"', self.js)

    def test_expanded_agent_stage_details_have_isolated_responsive_layout(self):
        self.assertIn('class="agent-matrix-stage-details"', self.js)
        self.assertIn(".agent-matrix-stage-details > div", self.css)
        self.assertIn("grid-template-columns: minmax(150px, 190px) minmax(0, 1fr)", self.css)
        self.assertIn("overflow-wrap: normal", self.css)
        self.assertIn("word-break: normal", self.css)


if __name__ == "__main__":
    unittest.main()
