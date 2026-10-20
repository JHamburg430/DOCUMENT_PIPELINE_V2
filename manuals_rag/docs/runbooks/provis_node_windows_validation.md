# PROVIS Inference Node: Windows validation

Use this checklist on a Windows GPU workstation after extracting the managed
node bundle. The workstation must already be joined to the organization's
Tailscale network, and Ollama must be installed locally at
`http://127.0.0.1:11434`.

## 1. Start the node

Open PowerShell in the extracted bundle directory:

```powershell
.\start-node-setup.ps1
```

Complete the Microsoft 365 enrollment flow. The node secret is stored in
`$env:LOCALAPPDATA\ProVIS\inference-node.json`; do not copy or print that
file. After enrollment, start the agent in a second PowerShell window:

```powershell
.\start-node-agent.ps1
```

The node defaults to `multi_gpu` placement. Keep these settings unless a
model explicitly requires one device:

```powershell
$env:INFERENCE_NODE_GPU_PLACEMENT_MODE = "multi_gpu"
$env:INFERENCE_ONNX_DEVICE = "AUTO"
```

Set these before starting `start-node-agent.ps1`, then restart the agent if
you change them.

## 2. Run local health checks

```powershell
$ip = (tailscale ip -4).Trim()
$health = Invoke-RestMethod "http://$($ip):8765/health"
$snapshot = Invoke-RestMethod "http://$($ip):8765/v1/node/snapshot"
$ollama = Invoke-RestMethod "http://127.0.0.1:11434/api/tags"

$health
$snapshot.node_id
$snapshot.inference_enabled
$snapshot.gpu_placement_mode
$snapshot.gpus | Select-Object gpu_id,gpu_index,name,total_vram_mb,used_vram_mb,backend
$snapshot.installed_models
$snapshot.onnx_providers
$ollama.models | Select-Object name

nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
```

Expected results:

- `/health` reports `status: ok`.
- `inference_enabled` is `True`.
- `gpu_placement_mode` is `multi_gpu`.
- Every NVIDIA GPU has a stable `gpu_id`/UUID and a `gpu_index`.
- Ollama models are visible in both `/api/tags` and the node snapshot.
- `nvidia-smi` and the node snapshot report the same GPU count.

## 3. Verify gateway heartbeat and multi-GPU admission

Check the node activity endpoint for `ollama: healthy`, `gpu: detected`,
`model_inventory: available`, and `heartbeat: accepted`:

```powershell
Invoke-RestMethod "http://$($ip):8765/v1/node/activity" |
  Select-Object -ExpandProperty events |
  Select-Object -Last 12
```

Ask the gateway operator for a test model catalog entry whose
`min_gpu_count` is `2` and whose `min_vram_mb` is larger than one local GPU's
available VRAM. Submit one normal inference request through the gateway and
inspect the returned `routing.placement` object. It should contain
`mode: multi_gpu`, two GPU UUIDs, and per-GPU VRAM allocations.

Do not test by setting `CUDA_VISIBLE_DEVICES` inside a running shared Ollama
process; that can affect unrelated requests. The scheduler's placement is
the admission reservation, while Ollama remains responsible for its own
multi-GPU execution.

## 4. Optional ONNX/OpenVINO test

The Python agent can load the optional dependencies from the bundle's
`requirements-inference-onnx.txt`:

```powershell
py -3 -m pip install -r .\requirements-inference-onnx.txt
```

Register a real, local ONNX model through an environment variable before
starting the agent. The path must be absolute:

```powershell
$manifest = @{
  "intel-encoder" = @{
    path = "C:\ProVIS\models\intel-encoder.onnx"
    capabilities = @("onnx", "embeddings")
  }
} | ConvertTo-Json -Compress
$env:INFERENCE_ONNX_MODELS_JSON = $manifest
$env:INFERENCE_ONNX_DEVICE = "AUTO"
.\start-node-agent.ps1
```

The snapshot should then include `onnx` in `capabilities`, list the ONNX
Runtime providers in `onnx_providers`, and show the registered model in
`installed_models`. A tensor request uses this shape through the gateway:

```json
{
  "model": "intel-encoder",
  "inputs": {"input_ids": [[1, 2, 3]]}
}
```

Use `GPU`, `GPU.0`, `MULTI`, or `HETERO` for a deliberate OpenVINO device
selection. `AUTO` is the recommended first test because it can choose an
available Intel GPU while retaining fallback behavior.

## 5. Stop and report results

Stop the agent with `Ctrl+C`. Report only the health status, node ID, GPU
count/UUIDs, provider names, and sanitized activity statuses. Never include
the node JSON, shared secret, Microsoft token, or gateway credentials.

If heartbeat is HTTP 400 while local checks pass, the gateway is probably
running an older inference-fabric schema and must be upgraded before the new
node can advertise placement metadata.
