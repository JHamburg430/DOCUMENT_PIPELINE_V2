# Contribution and release flow

Read [AGENTS.md](AGENTS.md), [.AGENT.md](.AGENT.md), and the
[inference CI/CD runbook](manuals_rag/docs/runbooks/provis_inference_cicd.md).

Start from the verified published integration branch, fetch deliberately and use
a clean `codex/` branch/worktree. Check `git status`, `git remote -v`, and the
base SHA before changes. Inspect nested instructions. Stage exact paths only;
never absorb a dirty employee-assistant/node/gateway working tree.

The inference implementation currently lives on
`codex/provis-production-updates-20261017`; `main` does not yet contain it. Target
that integration branch for this scoped PR. Moving the complete feature to
`main`, enabling Actions and configuring required reviews are separate maintainer
decisions. Instructions on a feature branch are not automatically installed on
the default branch or every PC.

Run the supported local check from repository root:

```sh
python manuals_rag/scripts/provis_release_pipeline.py ci
```

Install only already-approved test/build dependencies in an isolated environment
when authorized. A release also requires matching-OS packaging evidence using
`ci --packaging`. These builds use fixture configuration/keys, not employee
enrollment or the production publisher. Never install/run a candidate on a real
employee profile as a CI shortcut. Retrieval/application work retains the
additional real evaluation requirements in `.AGENT.md`.

PRs should explain the concrete behavior, exact validation, source SHA and any
unrun platform or live checks. Include a rollback/reconciliation plan for release
changes. The PR template makes these fields discoverable; it is not enforcement.
Keep releases blocked on failed, skipped or missing required evidence.

While CI budget and security bootstrap remain unapproved, keep the hard-disabled
workflow gates. Use `[skip ci]` in this contribution's commit so existing push/PR
triggers do not consume CI minutes. This is a temporary cost hold, not a way to
satisfy required checks; skipped checks may remain pending. Do not tag, create a
product GitHub Release, sign/publish a binary, change secrets/settings, or merge
release changes merely because local tests passed.
