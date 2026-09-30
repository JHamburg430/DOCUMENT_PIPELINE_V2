# Agent repair history and regression gate

The Agent page's **Send for repair** button stores the complete server-owned run snapshot in PostgreSQL `agent_repair_cases` and queues one visible Gateway diagnosis session per case. The host-side `openclaw-manuals-repair-handoff.service` creates it using the Gateway's built-in `sessions.create` RPC; the UI container never receives Gateway credentials. Repeated clicks reuse the same case and session. The UI shows pending, sent, or failed delivery, offers retry after failure, and links to the session after delivery. Session creation does not itself mean a fix was made or verified.

On upgraded deployments, apply `infra/migrations/20260930_gateway_repair_handoff.sql`, then install and enable `infra/systemd/openclaw-manuals-repair-handoff.service` as a user service. Its Python environment needs `psycopg[binary]==3.3.3`; the service uses the Gateway operator identity already configured for the host's `openclaw` CLI. New cases are queued by PostgreSQL state, so an interrupted dispatcher resumes them. Existing pre-upgrade cases remain unqueued until **Send for repair** is clicked again. Check `systemctl --user status openclaw-manuals-repair-handoff.service` and the case's `gateway_handoff_status` when delivery is not visible.

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
