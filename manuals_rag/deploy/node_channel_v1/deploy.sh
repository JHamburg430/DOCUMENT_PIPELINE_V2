#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_COMMIT="fffc77d75111f50255d2546d600f46937a108c34"
OVERLAY_COMMIT="537cd56c9b0df6daf8fcbab840060240961851f4"
PROTOCOL="provis-node-channel/1"
PKG="/opt/manuals-rag-inference/manuals_rag/packages/inference_fabric/src/manuals_rag_inference"
SERVICE="inference-fabric-gateway.service"
STAGE="$(mktemp -d /var/tmp/provis-node-channel-v1.XXXXXX)"
BACKUP="/var/backups/provis-node-channel-v1-$(date -u +%Y%m%dT%H%M%SZ)"
PATCH_URL="https://github.com/JHamburg430/DOCUMENT_PIPELINE_V2/commit/${SOURCE_COMMIT}.patch"
PATCH_SHA256="e3fdb51c81f8b99530be62b26e43957c6f229796c9d9081676371f4b182d30d6"
PRODUCTION_PATCH_SHA256="37c59633d24d312979764188ea75d50d9f46321742500d0800a4a76fa4fec0d0"
OVERLAY_URL="https://raw.githubusercontent.com/JHamburg430/DOCUMENT_PIPELINE_V2/codex/node-channel-v1-gateway/manuals_rag/deploy/node_channel_v1/agent_link.production.py"
OVERLAY_SHA256="687ff9601c9a31a8c499d41fc6b6d9e3d3d267b047d5425067521b2a7ddbfece"
LIVE_GATEWAY_SHA256="2465b129ed513c60a7aa13a7389f9b2a7b7629087a02d6f7061f37a2f9afb7d5"
LIVE_AGENT_LINK_SHA256="782eb831d5e6512969e6734946803cc56ef2eb78ae63577f97e3f9357f66ad50"
LIVE_AGENT_CLIENT_SHA256="c232167bc97bb08e1d20dfe2fcaf0ef579e43f2577c48f09ec12bbad78372291"
FILES=(gateway.py agent_link.py agent_link_client.py node_channel_protocol.py)
MUTATED=0

rollback() {
  local status=$?
  trap - ERR
  if [[ "$MUTATED" -eq 1 && -d "$BACKUP" ]]; then
    for file in "${FILES[@]}"; do
      if [[ -f "$BACKUP/$file" ]]; then
        install -o root -g root -m 0644 "$BACKUP/$file" "$PKG/$file"
      fi
    done
    systemctl restart "$SERVICE" || true
  fi
  echo "deployment_failed status=${status} backup=${BACKUP}" >&2
  exit "$status"
}
trap rollback ERR

test "$(id -u)" -eq 0
systemctl is-active --quiet "$SERVICE"

for file in gateway.py agent_link.py agent_link_client.py; do
  test -f "$PKG/$file"
done
test ! -e "$PKG/node_channel_protocol.py"
test "$(sha256sum "$PKG/gateway.py" | awk '{print $1}')" = "$LIVE_GATEWAY_SHA256"
test "$(sha256sum "$PKG/agent_link.py" | awk '{print $1}')" = "$LIVE_AGENT_LINK_SHA256"
test "$(sha256sum "$PKG/agent_link_client.py" | awk '{print $1}')" = "$LIVE_AGENT_CLIENT_SHA256"

curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  "$PATCH_URL" --output "$STAGE/full.patch"
SOURCE_PATCH="$STAGE/full.patch" OUTPUT_PATCH="$STAGE/node-channel-v1.patch" \
  /opt/manuals-rag-inference/.venv/bin/python - <<'PY'
import os
import re
from pathlib import Path

prefix = "manuals_rag/packages/inference_fabric/src/manuals_rag_inference/"
allowed = {
    prefix + "gateway.py",
    prefix + "agent_link.py",
    prefix + "agent_link_client.py",
    prefix + "node_channel_protocol.py",
}
source = Path(os.environ["SOURCE_PATCH"]).read_text(encoding="utf-8").splitlines(keepends=True)
selected = []
keep = False
for line in source:
    if line.startswith("diff --git a/"):
        parts = line.rstrip("\n").split()
        keep = len(parts) == 4 and parts[2][2:] in allowed and parts[3][2:] in allowed
    if keep:
        selected.append(line)
output = "".join(selected)
if not output or not all(f"a/{path}" in output for path in allowed):
    raise SystemExit("filtered Node Channel patch is incomplete")
Path(os.environ["OUTPUT_PATCH"]).write_text(output, encoding="utf-8")
PY
test "$(sha256sum "$STAGE/node-channel-v1.patch" | awk '{print $1}')" = "$PATCH_SHA256"

SOURCE_PATCH="$STAGE/node-channel-v1.patch" OUTPUT_PATCH="$STAGE/production.patch" \
  /opt/manuals-rag-inference/.venv/bin/python - <<'PY'
import os
import re
from pathlib import Path

source = Path(os.environ["SOURCE_PATCH"]).read_text(encoding="utf-8")
output = []
for section in re.split(r"(?=^diff --git )", source, flags=re.MULTILINE):
    if not section:
        continue
    header = section.splitlines()[0]
    if header.endswith("/agent_link.py b/manuals_rag/packages/inference_fabric/src/manuals_rag_inference/agent_link.py"):
        continue
    if header.endswith("/gateway.py b/manuals_rag/packages/inference_fabric/src/manuals_rag_inference/gateway.py"):
        first_hunk = section.find("@@ ")
        if first_hunk < 0:
            raise SystemExit("gateway diff has no hunks")
        prefix = section[:first_hunk]
        hunks = re.split(r"(?=^@@ )", section[first_hunk:], flags=re.MULTILINE)
        kept = [hunk for hunk in hunks if hunk and "from .node_channel_protocol import" not in hunk]
        if len(kept) != 3:
            raise SystemExit("unexpected gateway patch structure")
        section = prefix + "".join(kept)
    output.append(section)
result = "".join(output)
if "diff --git a/manuals_rag/packages/inference_fabric/src/manuals_rag_inference/agent_link.py" in result:
    raise SystemExit("production AgentLink must come from the reviewed overlay")
Path(os.environ["OUTPUT_PATCH"]).write_text(result, encoding="utf-8", newline="\n")
PY
test "$(sha256sum "$STAGE/production.patch" | awk '{print $1}')" = "$PRODUCTION_PATCH_SHA256"

curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
  "$OVERLAY_URL" --output "$STAGE/agent_link.py"
test "$(sha256sum "$STAGE/agent_link.py" | awk '{print $1}')" = "$OVERLAY_SHA256"

STAGED_ROOT="$STAGE/root"
STAGED_PKG="$STAGED_ROOT/manuals_rag/packages/inference_fabric/src/manuals_rag_inference"
install -d -m 0755 "$STAGED_PKG"
cp -a -- "$PKG/gateway.py" "$STAGED_PKG/gateway.py"
cp -a -- "$PKG/agent_link_client.py" "$STAGED_PKG/agent_link_client.py"
install -m 0644 "$STAGE/agent_link.py" "$STAGED_PKG/agent_link.py"
patch --batch --forward --no-backup-if-mismatch -p1 -d "$STAGED_ROOT" \
  < "$STAGE/production.patch"

GATEWAY="$STAGED_PKG/gateway.py" /opt/manuals-rag-inference/.venv/bin/python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["GATEWAY"])
text = path.read_text(encoding="utf-8")
needle = "from .streaming import NDJSONResponse, relay_ndjson, watch_disconnect\nfrom . import gateway_logging as gl\n"
replacement = """from .streaming import NDJSONResponse, relay_ndjson, watch_disconnect
from .node_channel_protocol import (
    PROTOCOL_HEADER,
    ProtocolVersionError,
    availability as node_channel_availability,
    require_protocol,
)
from .node_release import DEFAULT_VERSION
from . import gateway_logging as gl
"""
if text.count(needle) != 1:
    raise SystemExit("live gateway import seam changed")
path.write_text(text.replace(needle, replacement), encoding="utf-8", newline="\n")
PY

PYTHONDONTWRITEBYTECODE=1 /opt/manuals-rag-inference/.venv/bin/python -m py_compile \
  "$STAGED_PKG/gateway.py" "$STAGED_PKG/agent_link.py" \
  "$STAGED_PKG/agent_link_client.py" "$STAGED_PKG/node_channel_protocol.py"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight_succeeded source_commit=${SOURCE_COMMIT} overlay_commit=${OVERLAY_COMMIT}"
  sha256sum "$STAGED_PKG/gateway.py" "$STAGED_PKG/agent_link.py" \
    "$STAGED_PKG/agent_link_client.py" "$STAGED_PKG/node_channel_protocol.py"
  exit 0
fi

install -d -o root -g root -m 0755 "$BACKUP"
for file in "${FILES[@]}"; do
  if [[ -f "$PKG/$file" ]]; then
    cp -a -- "$PKG/$file" "$BACKUP/$file"
  else
    : > "$BACKUP/.absent-$file"
  fi
done

MUTATED=1
for file in "${FILES[@]}"; do
  install -o root -g root -m 0644 "$STAGED_PKG/$file" "$PKG/$file"
done

PYTHONDONTWRITEBYTECODE=1 /opt/manuals-rag-inference/.venv/bin/python -m py_compile \
  "$PKG/gateway.py" "$PKG/agent_link.py" "$PKG/agent_link_client.py" "$PKG/node_channel_protocol.py"

printf '%s\n' "$SOURCE_COMMIT" > "$BACKUP/deployed-source-commit.txt"
: > "$BACKUP/deployed-sha256.txt"
for file in "${FILES[@]}"; do
  sha256sum "$PKG/$file" >> "$BACKUP/deployed-sha256.txt"
done

systemctl restart "$SERVICE"
for _ in $(seq 1 30); do
  systemctl is-active --quiet "$SERVICE" && break
  sleep 1
done
systemctl is-active --quiet "$SERVICE"

availability="$(curl --fail --silent --show-error --max-time 10 \
  http://127.0.0.1:8780/v1/agent-link/availability)"
AVAILABILITY="$availability" EXPECTED_PROTOCOL="$PROTOCOL" \
  /opt/manuals-rag-inference/.venv/bin/python - <<'PY'
import json
import os

payload = json.loads(os.environ["AVAILABILITY"])
assert payload.get("protocol") == os.environ["EXPECTED_PROTOCOL"], payload
assert payload.get("available") is True, payload
assert payload.get("lifecycle", {}).get("heartbeat_seconds") == 6, payload
assert payload.get("lifecycle", {}).get("lease_seconds") == 30, payload
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
PY

trap - ERR
echo "deployment_succeeded source_commit=${SOURCE_COMMIT} backup=${BACKUP}"

