# Agent repair history and regression gate

The Agent page's **Send for repair** button stores the complete server-owned run snapshot in PostgreSQL `agent_repair_cases`. Repeated clicks reuse the same case. It does not automatically change code or launch a repair agent.

The gateway operator can inspect cases from the running UI container:

```bash
docker compose -f infra/compose/docker-compose.yml exec -T ui python scripts/benchmark/agent_repair_regressions.py list
```

After diagnosing and fixing a case, record the root cause, code commit, verification evidence, and a measurable replay contract. Both JSON files must be available inside the UI container. The contract requires `expect_abstention` plus at least one of `required_answer_terms`, `required_citation_document_ids`, or `required_event_types`; it can optionally provide `replay_query` for a self-contained query when the original relied on session context.

```json
{"expect_abstention": false, "required_answer_terms": ["24 VDC"], "required_citation_document_ids": ["manual-id"]}
```

```bash
docker compose -f infra/compose/docker-compose.yml exec -T ui python scripts/benchmark/agent_repair_regressions.py record-fix \
  --request-id repair-agent-run-012345abcdef --root-cause '...' --summary '...' \
  --commit FULL_COMMIT_SHA --regression-spec /workspace/manuals_rag/path/to/spec.json \
  --verification /workspace/manuals_rag/path/to/verification.json
```

Before each release, commit source changes, then run `make repair-regressions` from `manuals_rag/`. It replays the latest fix for every open case against `/agent/stream`, stores the result in `agent_repair_regression_runs`, and fails if a case fails, is blocked, or no fixed cases exist. Never treat a timeout or missing completion as a pass. The case status becomes `verified` only after a passing replay. Use `--allow-empty` directly on the script only when deliberately accepting that there are no recorded fixes yet.

Existing deployments apply `infra/migrations/20260929_agent_repair_history.sql` once with `psql -v ON_ERROR_STOP=1`. Fresh PostgreSQL volumes receive the same schema through `infra/migrations/init.sql`.
If an older deployment has JSON repair packets, run `agent_repair_regressions.py import-legacy` in the UI container and verify the reported count before archiving those packets. Import is idempotent and does not delete the originals.
