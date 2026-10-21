# Repository agent instructions

Read the immutable [.AGENT.md](.AGENT.md) before work; do not edit it. This file
adds contribution and inference-release rules without replacing architecture,
evaluation, evidence, permission or manual-review requirements.

All agents must use the supported [contribution flow](CONTRIBUTING.md). For any
inference-node, gateway, packaging or release change, use the single
[PROVIS release pipeline](manuals_rag/docs/runbooks/provis_inference_cicd.md):

1. Verify repository/remotes, applicable instructions and a clean isolated
   `codex/` branch. Preserve unrelated work; never stage the whole checkout.
2. Run `python manuals_rag/scripts/provis_release_pipeline.py ci` plus relevant
   domain tests. Windows/Linux packaging checks must pass on the matching OS
   before a release. Record tested source SHA, commands, results and limitations.
3. Open a reviewed PR; do not merge your own product/release changes or bypass
   missing/skipped checks. Never mix another PC's uncommitted runtime changes.
4. Build reviewed candidates once, with exact source/tag/version and public-only
   managed configuration. Use the existing protected publisher for exact-byte
   signatures only after explicit human approval; never move keys into CI/repo.
5. Use only the pipeline's `verify` then explicitly approved `publish` commands
   for GitHub Releases and the separate PROVIS app/installer catalogs. Preserve
   signed approval, checksums and completion/partial-stage evidence receipts.
   Do not use ad-hoc uploads, `--clobber`, overwrite versions/assets, rebuild
   approved artifacts, move release tags or silently repair a partial release.

PR/build code must have no signing/catalog credentials and never run on a trusted
publisher runner. Never add `pull_request_target` or a privileged `workflow_run`
handoff that executes PR code. Publishing is a separate trusted manual workflow;
ordinary pushes cannot publish. Gateway deployment/restart, PC installation,
node rollout, model preparation and log cleanup need their own authorization.

Current bootstrap state: workflow jobs are deliberately hard-disabled. Do not
enable them, consume paid CI, provision runners/secrets, broaden tokens, change
environments/branch rules, or trigger publication without separate approval.
Use local tests and report the operational blocker honestly. Prose does not
enforce these rules against humans or other tools; protected environments,
review/rules and narrowly scoped credentials require maintainer setup.
