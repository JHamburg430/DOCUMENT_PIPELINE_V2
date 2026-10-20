# Dell reliability patch: local review only

Base: `5df482eaea183bcc92011bb6dc9a1c8c4ebd5637`.
Branch: `codex/dell-reliability-local-20261020`.
No deployment, runtime flag, signing, publication, release build, push, live acknowledgement retry, model operation, or credential/allowlist/network change is included.

## Durable terminal acknowledgement

The helper writes a private `healthy-ack.json` intent before committing `active.json`. It reports only the original signed plan's request ID, node ID and claim. A lost success response is safely repeated using that same claim and a fresh transport nonce; the existing server queue already makes identical terminal reports idempotent.

After restart, a freshly reconnected app verifies the active release signature/artifact, exact private plan, release version, node/config identity and matching nonce-bearing `healthy.json` before recovery. Manual installs without a queued plan do not create an acknowledgement. A probe cannot poll another update before its active-pointer commit. A process-scoped OS lock serializes helper and app recovery; crash-released locks cannot become permanent stale lock files.

Six persisted attempts across all restarts, a one-hour lifetime, at most three transmissions per pass, five-second request timeouts and capped exponential delays bound retries. Only transient transport errors and HTTP 408/429/5xx retry. Authentication, authorization, claim, transition, TLS-verification and malformed-response failures stop in a durable blocked state. Exhausted/blocked recovery does not poll new updates or change inference state. A committed healthy release is never rolled back because acknowledgement fails; an uncertain pointer write stops for inspection instead of killing the candidate.

This does not repair or retry the already denied live request `acf3a307-dc92-4ed6-bddc-3457c9acb8b8`. No live report was sent while preparing this patch.

## Optional exact-worker owner diagnostic

`diagnostic-inference` is disabled by default. Runtime opt-in is `INFERENCE_OWNER_DIAGNOSTICS_ENABLED=true`, and an existing Entra-authenticated Owner Agent Link is required. The body accepts exactly `target_node_id` and `model`; the server generates the request ID and fixed synthetic prompt. Normal inference never calls this action and does not require it.

The target must be currently enrolled in the same owner workspace and permitted by the existing `provis_employee_agent` worker allowlist, enrollment application approval and fresh signed heartbeat. It must be healthy, inference-enabled, idle, resource/capability eligible, and report both installed and loaded model state. The model must exist in the authoritative Ollama chat catalog. The scheduler is narrowed to exactly that authorized target; any different decision fails closed. No preparation, download, warm, pull or eviction command is issued.

The worker must advertise `diagnostic_loaded_only_v1`. It verifies the signed diagnostic constraint, fixed token budget and prompt, rechecks `/api/ps` under its request lock, and reuses the resident runner's reported context. Missing/unknown contexts or contexts outside 256–8192 are rejected instead of implicitly reloading with a different context. Older deployed `.3` workers lack this capability and cannot run the new action.

One global diagnostic slot, ten-second transport deadline, eight output tokens, bounded event/output sizes, no redirects, cancellation on owner disconnect, and unconditional reservation cleanup bound the operation. The receipt's worker identity/model/version come from the actual server decision and checked target, never model output. It is authenticated by the owner HTTPS response; it is not a portable cryptographic attestation.

## Production routing remains dynamic

Normal requests retain central application worker allowlists; caller-supplied node restrictions still cannot widen or force them. Owner workspace constraints apply to this optional management diagnostic, not to the company's authorized shared compute pool. Normal ingress identity, enrollment, nonce and application checks remain in the existing gateway paths.

Local scheduler repairs:

- Exclude unhealthy GPUs and backend/provider mismatches.
- Reject invalid/future/stale heartbeat times and enforce minimum GPU counts in single-device mode.
- Credit reusable resident model memory only with known per-GPU placement, capped by observed GPU usage and total memory. Unknown placement gets no invented credit.
- Bound active/queued/reserved occupancy, prefer lower measured CPU/GPU load for otherwise comparable workers, and keep missing load measurements optional.
- For OpenVINO GPUs with unknown dedicated VRAM, require fresh host available-memory telemetry and a one-GiB reserve. Track/release shared-memory reservations to prevent concurrent overcommit without fabricating VRAM numbers.

Tests demonstrate admission-time failover away from unhealthy, stale, occupied, incompatible, unapproved and under-resourced workers; exact diagnostics do not replace this dynamic path. Post-dispatch failures release reservations but are not automatically replayed on another worker, avoiding duplicate work after an ambiguous response.

## Validation and limits

Windows local fixture regression result: **357 passed, 1 skipped, 2 deselected** in 39.65 seconds. New focused modules account for 26 acknowledgement recovery tests, 44 owner diagnostic tests and 23 dynamic admission tests. See workspace `work/dell-reliability-tests.xml` and the accompanying local result record.

The run includes all `manuals_rag/tests/unit/test_inference*.py`, plus `test_updater_ack_recovery.py`, `test_owner_diagnostic.py`, `test_dynamic_worker_admission.py`, `test_host_resources.py`, `test_node_service_attachment.py` and `test_onnx_runtime_setup.py`. Python import roots are inference_fabric/src, provis_inference_node/src, manuals_rag/scripts, common/src, retrieval/src and manuals_rag. Tests use temporary fixtures and loopback servers only.

Platform exclusions: Linux peer-credential/flock relaunch test skips itself. The two Linux Ollama symlink activation tests (`test_activation_switches_atomically_and_health_checks`, `test_activation_rolls_back_when_health_check_fails`) were explicitly deselected because Windows denied symlink creation with WinError 1314. No Windows privilege/security setting was changed. POSIX lock/directory-fsync and Linux native activation still require a Linux test runner before any Linux release.

Other limits requiring an explicit product/rollout decision:

- Existing normal installed-model warmup policy is unchanged. Normal inference has pull/eviction flags but no separate warm-consent field; a distinct warm permission, if required, must be specified before treating normal warmup as separately authorized. The new diagnostic always forbids preparation.
- CPU-only ONNX workers still lack a CPU/RAM admission path. Do not represent an available OpenVINO CPU runtime as a GPU or promise Intel GPU execution where no Intel GPU is exposed.
- CPU/GPU load is a placement preference, not a preemption mechanism. Active/reservation counters are conservative and may overlap; no workload is killed or relocated.
- Disk telemetry is observable, not a disk reservation system for model downloads. No downloads were exercised here.
- Endpoint failure circuit-breaking and automatic post-dispatch retry are not implemented. Admission-time health/freshness failover is covered; ambiguous execution is not silently retried.
- The worker guards node-managed model operations; an external administrator changing Ollama residency concurrently is outside this lock. No atomic conditional-inference API is available from Ollama in this patch.
- `update-status` is not strictly read-only: its implementation expires stalled requests with an UPDATE/commit, and normal owner dispatch also performs mailbox housekeeping. Under the later instruction permitting a lookup only if strictly read-only, no new live status request was sent. The executor's reported `healthy_acknowledged` result is not independently confirmed here. Do not infer permission to retry acknowledgement or mutate the old request.
- No installed binary, real model, production owner action, or hardware GPU execution was tested with this source patch.

## Approval boundary

1. Review this local diff first. Building a new updater candidate requires explicit approval; then review its new version, exact hashes/sizes, publisher pin and test evidence before signing/publishing or installing on a named node with rollback. Existing `.3` artifacts are unchanged. Use a new controlled update request to test future recovery; do not replay the old denied request.
2. Diagnostic activation additionally requires approval of the exact gateway file manifest/restart/rollback, the new signed worker capability release, and the explicit opt-in flag. Preserve existing worker allowlists and require a loaded compatible model; approval is not authorization to download/warm one. A single named-target live probe and its budget should be approved separately at activation time.
3. Normal scheduler changes also require reviewed gateway deployment and bounded dynamic-routing validation. Enabling the owner diagnostic is optional and is never a prerequisite for ordinary inference.
4. No approval for any of these deployment steps is inferred from preparing this patch. Second employee verification remains last.
