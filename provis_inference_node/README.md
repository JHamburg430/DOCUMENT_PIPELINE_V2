# PROVIS Inference Node

This is the standalone node application used by the central PROVIS inference
gateway. It is deliberately outside `manuals_rag/` and has no runtime import
or dependency on Manuals RAG. Its runtime dependencies are empty; it uses
Python’s standard library plus Tkinter for the desktop window.

## Linux quick start

On Ubuntu/Debian, install Python 3.10+ and the distribution Tk package (often
`python3-tk`), ensure Tailscale and local Ollama are running, then build or
extract the managed bundle and run:

```bash
./start-node-setup.sh
```

The launcher resolves its own directory, never the current working directory,
and uses the bundled managed HTTPS gateway configuration. Enrolled state is
stored at `$XDG_CONFIG_HOME/ProVIS/inference-node.json` or
`$HOME/.config/ProVIS/inference-node.json`.

For a Linux-native one-file executable, on the target Linux machine run:

```bash
python3 -m pip install --user 'pyinstaller>=6.11,<7'
./build-node-linux.sh
```

This produces `dist/PROVISInferenceNode`. PyInstaller builds are OS-specific.
No token, node secret, Microsoft access token, Docker socket, or Manuals RAG
package is included in the bundle.

## Build a managed bundle

The administrator must provide the verified deployed gateway URL; it is the
only value embedded:

```bash
python3 tools/build_node_bundle.py \
  --gateway-url https://verified-inference-gateway.example \
  --output dist/provis-inference-node.zip
```

The builder rejects empty, non-HTTPS, credential-bearing, query-bearing, or
fragment-bearing URLs. Do not use the OpenClaw employee portal URL.

Run the standalone checks with:

```bash
python3 -m unittest discover -s tests -v
```

## Local OpenAI-compatible endpoint

After the node connects, it exposes a local broker on port `8767`:

```text
http://127.0.0.1:8767/v1
```

Applications in the RAG Docker network use:

```text
http://host.docker.internal:8767/v1
```

The supported routes are `/chat/completions`, `/completions`, `/embeddings`,
and `/models`. Add `X-ProVIS-Application` when the caller is not using the
node's default application; the value must be in the node's approved
application allowlist. The node signs the request with its enrolled secret
and forwards it to the central gateway, which performs model/resource
admission and selects the inference node. Gateway credentials never leave the
node. Streaming is rejected until the gateway has a streaming contract.

The broker accepts only loopback and configured local Docker CIDRs by default.
For deployments requiring an additional application-level boundary, set
`INFERENCE_NODE_REQUIRE_LOCAL_API_TOKEN=true` and distribute the generated
local API token through the application's local secret store—not through the
gateway or source repository.
