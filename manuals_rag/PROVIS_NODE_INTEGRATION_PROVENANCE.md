# Protected node integration for release 2026.10.20.1

Base: 4472cb63b53b92231b00d65a7cfe92f5175337a8.
Verified upstream feature source: f277b09356bea6f847c4f11e2bb88ab4b488b956, branch codex/provis-production-updates-20261017.

This is a scoped integration, not the raw upstream tree. It carries the upstream commit's GPU UUID/index metadata, per-GPU placement, optional allowlisted ONNX backend and bundle/build support onto the protected source. Existing replay-resistant returning registration, signed update/installer validation, owner controls, listener refresh, diagnostics and broker source are retained.

Additional integration corrections: gate signed placement fields by explicit node heartbeat capability for legacy workers; preserve the deployed power_w field; construct new GPU metadata with keyword arguments so the retained power field cannot shift UUID/index/backend positions; apply the same placement-envelope compatibility rule to streaming.

The optional ONNX runtime is not installed or activated by default. An executable built without its optional dependencies must not be described as providing live ONNX/OpenVINO execution. No Entra permission, enrollment, worker authorization, publisher or signing-key change is included.

Verification before candidate build: 143 focused tests passed; one platform-specific skip. New tests cover two stable GPU UUIDs, telemetry field alignment, old/new signed envelopes and heartbeat parsing, in addition to existing replay and signed update checks.

No external push, signing, catalog publication, update queue operation, app execution or PC installation is implied by this source commit. The produced artifact requires independent bytecode/config/hash verification and an exact signing/publication approval.
