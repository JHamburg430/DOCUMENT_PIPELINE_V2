# Ubuntu Codex RAG handoff - 2026-10-09

Goal: develop RAG independently in the Ubuntu Codex app, outside both OpenClaw gateways. This PC is also a compute worker for DigitalOcean; that does not place RAG development inside OpenClaw. Employee integration with DigitalOcean is a later step.

Checkout: /home/john/Desktop/Programming/Document_Pipeline_V2_evidence_gate_20261002
Repository: https://github.com/JHamburg430/DOCUMENT_PIPELINE_V2
Branch: codex/manuals-agent-evidence-gate-20261002
HEAD: a03c001b3a0abcf184cb6a7ccc67b57ece4cd075 (Harden Manuals RAG evidence-gate evaluation)
Tracked files were clean before this handoff. Seven existing untracked SQLite/lock/launch/exit report artifacts were preserved. Reuse this worktree; do not clone, reset, delete, or commit transient artifacts.
Read immutable .AGENT.md and manuals_rag/README.md first. Preserve Technical_Documents and existing work.

Codex setup:
Existing Ubuntu session 01a12200-afdb-7af1-91c9-2f2461cdd0fc, "Push Technical Documents RAG changes", is idle. Configured cwd: /home/john/Documents/ChatGPT/OpenClaw. Its completed commands committed and pushed this checkout.
Supported project inventory currently has no saved Ubuntu RAG project. Open/add this existing checkout folder as a project in the Ubuntu Codex app. No available tool registers an arbitrary remote project path or rebinds session cwd; do not edit session databases.
Obtain user authorization before messaging that independently operated session. A continuation message does not itself change its configured cwd. Confirm one owner before starting development.

Tests:
No tests were run during migration preparation; existing committed reports are not a fresh pass.
manuals_rag/pytest.ini defines package source paths and excludes live tests.
README interpreter /home/john/Desktop/Programming/Document_Pipeline/.venv/bin/python was not executable.
Existing /home/john/Desktop/Programming/Document_Pipeline_V2/.venv/bin/python is executable but lacks pytest (python -B -m pytest --version failed). No installation was performed.
After locating a suitable existing environment, run from manuals_rag:
python -B -m pytest -p no:cacheprovider tests/unit/test_agentic_retrieval.py tests/unit/test_evidence_gate.py tests/unit/test_onsite_agent_eval.py tests/unit/test_onsite_agent_eval_audit.py
Record actual interpreter and outcomes.

Next:
1. Establish Ubuntu Codex project at this checkout and coordinate a single owner.
2. Verify branch/HEAD/status again and read repository instructions before edits.
3. Locate an existing suitable environment; run focused non-live tests.
4. Inspect committed evidence-gate changes and benchmark audit tool; review Keyence diagnostics locally to identify reproducible remaining failures. Do not copy sensitive documents into handoffs.
5. Inventory gateway coupling and plan independent RAG development/runtime configuration. Do not invoke gateway repair/handoff scripts.
6. Plan employee integration separately with tenant/corpus permissions, grounded citations, evaluation gates, and rollback.

Boundaries: no gateway/worker service changes, credentials, duplicate clones, deletions, dependency installs, pushes, production ingestion or live evaluation in this setup step.
