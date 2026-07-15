#!/usr/bin/env bash
# Render stage-2-pipeline.excalidraw → stage-2-pipeline.png
#
# Option 1 (Excalidraw+ workspace, requires API key + subscription):
#   1. Open https://excalidraw.com in browser, log in
#   2. File → Open → select stage-2-pipeline.excalidraw
#   3. File → Export image (PNG) → save as stage-2-pipeline.png
#
# Option 2 (CLI, requires npm):
#   npx -p @excalidraw/utils@latest excalidraw-convert \
#       stage-2-pipeline.excalidraw stage-2-pipeline.png
#
# Option 3 (Excalidraw+ API, programmatic upload — verified at implementation time):
#   K8s secret 'excalidraw-api-key' in your chosen namespace.
#   curl -X POST -H "Authorization: Bearer $(kubectl get secret ...)" ...

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v npx >/dev/null; then
    npx -p @excalidraw/utils@latest excalidraw-convert \
        stage-2-pipeline.excalidraw stage-2-pipeline.png
    echo "Rendered: stage-2-pipeline.png"
else
    echo "npm/npx not found. Use Excalidraw.com web UI to render — see top of this script for steps."
    exit 1
fi
