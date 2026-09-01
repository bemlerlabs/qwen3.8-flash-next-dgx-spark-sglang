#!/usr/bin/env bash
# ==============================================================================
# Qwen3.8-Flash-Next (180B MoE) SGLang Launch Script for NVIDIA DGX Spark
# Recipe by Michael Bemler (Bemler Labs)
# Hardware: Grace Blackwell GB10 (128GB Unified Memory, SM121)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration & Defaults
MODEL_ID="${MODEL_ID:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-7b719225242aacd3dbd3f9407468c2ee9a9d2594}"
PLE_DIR="${PLE_DIR:-${HOME}/flashnext-ple}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.config/qwen38/sglang-cache}"
CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-qwen38-flash:v1.5}"
API_KEY="${API_KEY:-local}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "================================================================="
echo " Starting Qwen3.8-Flash-Next on DGX Spark (SGLang + Fast Load)   "
echo "================================================================="
echo "Model ID:       ${MODEL_ID}"
echo "Revision:       ${MODEL_REVISION}"
echo "PLE Directory:  ${PLE_DIR}"
echo "Cache Dir:      ${CACHE_DIR}"
echo "HF Cache:       ${HF_CACHE_DIR}"
echo "Listening on:   http://${HOST}:${PORT}"
echo "================================================================="

mkdir -p "${PLE_DIR}" "${CACHE_DIR}" "${CONFIG_DIR}" "${HF_CACHE_DIR}"

# Check for chat template
if [ ! -f "${CONFIG_DIR}/chat-template-flashnext.jinja" ]; then
  echo "Error: Chat template not found at ${CONFIG_DIR}/chat-template-flashnext.jinja"
  exit 1
fi

# Background health monitor (disowned to prevent zombie subshell)
(
  for _ in $(seq 1 270); do
    sleep 10
    if curl -s -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "Server is healthy and ready on port ${PORT}."
      exit 0
    fi
  done
) </dev/null >/dev/null 2>&1 &
disown || true

# Launch production container with unthrottled page cache and 8-thread prefetching
exec /usr/bin/docker run --rm --name qwen38-flash --gpus all \
  --shm-size 16g --network host --ipc=host \
  -v "${HF_CACHE_DIR}":/root/.cache/huggingface \
  -v "${CONFIG_DIR}":/out \
  -v "${PLE_DIR}":/ple \
  -v "${CACHE_DIR}":/root/.cache/sglang \
  -e SGLANG_QWEN4_PLE_MMAP_DIR=/ple \
  -e TRITON_CACHE_DIR=/root/.cache/sglang/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/sglang/inductor \
  "${CONTAINER_IMAGE}" \
  python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --tp-size 1 \
    --served-model-name qwen3.8-flash-next \
    --quantization modelopt_fp4 \
    --prefill-attention-backend triton \
    --decode-attention-backend trtllm_mha \
    --ple-offload-embedding \
    --weight-loader-disable-mmap \
    --weight-loader-drop-cache-after-load \
    --weight-loader-prefetch-checkpoints \
    --weight-loader-prefetch-num-threads 8 \
    --mamba-radix-cache-strategy extra_buffer \
    --max-mamba-cache-size 48 \
    --mem-fraction-static 0.90 \
    --context-length 131072 \
    --chunked-prefill-size 4096 \
    --enable-mixed-chunk \
    --schedule-conservativeness 0.8 \
    --max-running-requests 4 \
    --allow-auto-truncate \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-draft-model-quantization unquant \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --chat-template /out/chat-template-flashnext.jinja \
    --api-key "${API_KEY}" \
    --host "${HOST}" \
    --port "${PORT}"
