# Gateway handoff: node-brokered OpenAI-compatible inference

## Objective

Add the gateway side of the local PROVIS node broker. RAG and other local
applications must not receive gateway operator tokens or application tokens.
They call the connected node's local OpenAI-compatible endpoint; the node
authenticates to the gateway with its enrolled node secret; the gateway owns
model/resource admission and selects the inference node.

The local endpoint is:

```text
http://127.0.0.1:8767/v1
```

This is a loopback-only endpoint for applications running on the same PC.
Docker access is disabled in the signed desktop release. Do not configure
`host.docker.internal` as if it were a working endpoint: a container cannot
reach a service bound to the host's loopback address. Enabling Docker access
requires a separately approved host-only bind address, exact bridge CIDR
allowlist, and mandatory bearer authentication with a securely provisioned
container token. Do not bind to `0.0.0.0` or widen a firewall rule.

The signed desktop publisher builds the Manuals RAG inference client and its
bundled `manuals_rag_inference` package. The shared OpenAI broker implementation
is packaged from `provis_inference_node/src/provis_inference_node/openai_broker.py`;
the standalone app is not the signed release source. The deployed gateway must
still be updated: it currently returns `404` for the required route, so do
not claim end-to-end readiness until route and streaming checks pass against
the droplet deployment.

## 1. Add the node-authenticated gateway route

Implement:

```text
POST /v1/nodes/inference
```

Request body:

```json
{
  "node_id": "john-System-Product-Name",
  "issued_at": 1760000000.0,
  "request": {
    "request_id": "openai-...",
    "application": "manuals_rag",
    "workload": "openai_chat",
    "model": "qwen3.5:9b",
    "context_tokens": 32768,
    "required_capabilities": ["chat"],
    "max_latency_ms": 900000,
    "allow_model_pull": false,
    "allow_eviction": false
  },
  "operation": "chat",
  "payload": {
    "model": "qwen3.5:9b",
    "messages": [],
    "stream": false
  }
}
```

The node signs the exact canonical request bytes with its enrolled secret and
sends `X-Node-Signature`. The gateway must:

1. Look up `node_id` in persistent enrollment state.
2. Verify the signature against the exact received body.
3. Reject stale/replayed `issued_at` values.
4. Require a current healthy heartbeat and `inference_enabled=true`.
5. Require `request.application` in the node's approved application allowlist.
6. Resolve model size, VRAM, and capabilities from the gateway's authoritative
   model catalog; do not trust caller-supplied resource values.
7. Run normal scheduler admission and route to the selected node.
8. Forward to the selected node using the existing signed
   `/v1/node/inference` protocol. The ingress node and selected worker may be
   different.
9. Return the normal response shape:

```json
{
  "routing": {"request_id": "...", "node_id": "..."},
  "response": {"message": {"role": "assistant", "content": "..."}}
}
```

Do not require `X-Gateway-Token` or `X-Inference-Application-Token` on this
node-authenticated route. The enrolled node identity plus its application
allowlist is the authorization boundary. Keep the existing operator and
application-token authorization unchanged for direct `/v1/inference` callers.

Return controlled JSON errors (`401` for invalid node authentication, `403`
for ineligible/disallowed nodes, `400` for malformed requests, `503` for
admission/upstream failures). Never log prompts, tokens, HMACs, or raw model
responses.

## 2. Add true streaming; do not buffer it

The local OpenAI-compatible node accepts `stream=true`. The current node
client rejects streaming only because the gateway streaming contract is not
deployed yet. Implement streaming across all three layers:

```text
local OpenAI SSE
  -> node broker
  -> node-signed gateway request
  -> scheduler-selected node
  -> Ollama NDJSON stream
```

### Gateway streaming contract

The same `/v1/nodes/inference` route may carry `payload.stream=true`. For a
streaming request:

- Preserve the same node authentication, application allowlist, catalog
  resolution, scheduler reservation, model preparation, and timeout checks.
- Forward the signed request to the selected node without buffering.
- Use `Content-Type: application/x-ndjson` for the internal
  gateway-to-node stream.
- Release the scheduler reservation in a `finally` block after completion,
  disconnect, timeout, or upstream error.
- Stop forwarding on client disconnect and cancel/close the upstream response.
- Do not retry after any token has been emitted.
- If failure occurs before the first token, return a controlled `503`; after
  streaming begins, emit a terminal error event and close the stream.

The node's `/v1/node/inference` handler must stream Ollama's NDJSON lines when
`payload.stream=true`. It must not call `json.loads()` on the entire response.

### OpenAI SSE conversion

The local node converts each Ollama chunk to OpenAI-compatible SSE:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1760000000,"model":"qwen3.5:9b","choices":[{"index":0,"delta":{"role":"assistant","content":"token"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":1760000000,"model":"qwen3.5:9b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

For `/completions`, emit `text` deltas. Embeddings remain non-streaming and
must return `400` if `stream=true`. Honor standard `max_tokens`, temperature,
top-p, seed, and context options where supported.

Do not label a buffered response as streaming. The node now rejects streaming
only for unsupported operations such as embeddings; chat and completions must
remain incremental end-to-end.

## 3. Security and isolation requirements

- The RAG containers receive only the nonsecret node-broker URL.
- Gateway operator tokens, application tokens, node secrets, and Microsoft
  access tokens remain outside RAG and outside the local OpenAI request.
- Keep the local broker restricted to loopback. Docker bridge access has not
  been approved or enabled. Do not expose it on a LAN, public interface, or
  tailnet.
- Keep `X-ProVIS-Application`/`X-Inference-Application` constrained to the
  node's approved application list.
- Never use the OpenClaw employee portal URL as the inference gateway URL.
- Do not add a gateway-token fallback to the node broker.

## 4. Required tests

Add or run tests for:

1. Valid node signature and approved application are accepted.
2. Wrong node, wrong signature, stale timestamp, replay, unhealthy node,
   disabled node, and disallowed application are rejected.
3. Missing/unknown model is rejected from the authoritative catalog.
4. Ingress node may differ from selected worker node.
5. OpenAI JSON chat, completion, and embeddings responses are translated
   correctly.
6. `stream=true` produces incremental SSE chunks and `[DONE]`, not one
   buffered response.
7. Ollama disconnect, gateway timeout, client disconnect, and scheduler
   rejection release reservations and do not retry after token emission.
8. RAG can call the local endpoint without gateway credentials.
9. A local endpoint request cannot select an application outside the node's
   allowlist.
10. No prompts, access tokens, node secrets, or raw authentication responses
    appear in logs.

## 5. Deployment and handoff verification

Deploy only the dedicated inference-gateway service on the droplet. Do not
restart OpenClaw, RAG API/UI, databases, Qdrant, Redis, or node services as
part of the gateway route deployment.

Verify from a connected node:

```bash
curl --fail https://<verified-gateway>/health
```

Then verify the local node endpoint from a host application on the same PC:

```bash
curl http://127.0.0.1:8767/v1/models
```

Run one non-streaming and one streaming smoke request using an approved test
model and `X-ProVIS-Application: manuals_rag`. Confirm the response includes a
real scheduler-selected `routing.node_id`, and confirm node activity records
show the request without exposing prompt content.

Only after those checks pass should the RAG API/worker containers be recreated
with their nonsecret `INFERENCE_NODE_BROKER_URL` default and a one-question
UI-visible RAG smoke test be started.
