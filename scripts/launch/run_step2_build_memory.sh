#!/usr/bin/env bash
set -euo pipefail

MEMORYCARD_ROOT="${MEMORYCARD_ROOT:-/path/to/MemoryCard}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STEP2_PY="${STEP2_PY:-${MEMORYCARD_ROOT}/memorycard/build_memory/build_memory.py}"
QWEN3_VL_REPO="${QWEN3_VL_REPO:-/path/to/Qwen3-VL-main}"

# Model path
VLM_MODEL_DIR="${VLM_MODEL_DIR:-/path/to/Qwen3-VL-8B-Instruct}"

# Data paths
DATA_JSONL="${DATA_JSONL:-/path/to/Video-MME/videomme/test-00000-of-00001.parquet}"
VIDEO_DATA_DIR="${VIDEO_DATA_DIR:-/path/to/Video-MME/data}"

# Output paths. This should be the same OUT_ROOT used in Step 1.
OUT_ROOT="${OUT_ROOT:-/path/to/output/videomme_memory}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"
TIMING_DIR="${TIMING_DIR:-${OUT_ROOT}/timing}"

# GPU sharding: one worker per GPU
VLM_GPUS_STR="${VLM_GPUS_STR:-0}"
read -r -a VLM_GPUS <<< "${VLM_GPUS_STR}"

# Memory construction hyperparameters
SELFREAD_MAX_NEW_TOKENS="${SELFREAD_MAX_NEW_TOKENS:-8192}"
FRAME_TIMEOUT_SEC="${FRAME_TIMEOUT_SEC:-30}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"   # sdpa | flash_attention_2 | eager | auto
OVERWRITE="${OVERWRITE:-0}"
KEEP_INTERMEDIATE_FRAMES="${KEEP_INTERMEDIATE_FRAMES:-0}"
JSONL_VIDEO_ID_FIELD="${JSONL_VIDEO_ID_FIELD:-video_id}"
JSONL_VIDEO_FILEID_FIELD="${JSONL_VIDEO_FILEID_FIELD:-videoID}"

check_path() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" || "${value}" == /path/to/* ]]; then
    echo "[ERROR] ${name} is not set: ${value}" >&2
    echo "Please edit the path section at the top of this script." >&2
    exit 1
  fi
}

check_path "MEMORYCARD_ROOT" "${MEMORYCARD_ROOT}"
check_path "STEP2_PY" "${STEP2_PY}"
check_path "QWEN3_VL_REPO" "${QWEN3_VL_REPO}"
check_path "VLM_MODEL_DIR" "${VLM_MODEL_DIR}"
check_path "DATA_JSONL" "${DATA_JSONL}"
check_path "VIDEO_DATA_DIR" "${VIDEO_DATA_DIR}"
check_path "OUT_ROOT" "${OUT_ROOT}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${TIMING_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

NUM_WORKERS="${#VLM_GPUS[@]}"
PIDS=()

cat <<INFO
==================================================
[STEP2] Qwen3-VL query-agnostic Memory Card construction
STEP2_PY=${STEP2_PY}
VLM_MODEL_DIR=${VLM_MODEL_DIR}
DATA_JSONL=${DATA_JSONL}
VIDEO_DATA_DIR=${VIDEO_DATA_DIR}
OUT_ROOT=${OUT_ROOT}
VLM_GPUS_STR=${VLM_GPUS_STR}
SELFREAD_MAX_NEW_TOKENS=${SELFREAD_MAX_NEW_TOKENS}
ATTN_IMPL=${ATTN_IMPL}
OVERWRITE=${OVERWRITE}
KEEP_INTERMEDIATE_FRAMES=${KEEP_INTERMEDIATE_FRAMES}
==================================================
INFO

for i in "${!VLM_GPUS[@]}"; do
  GPU_ID="${VLM_GPUS[$i]}"
  WORKER_ID="${i}"
  EXTRA_ARGS=()
  [[ "${OVERWRITE}" == "1" ]] && EXTRA_ARGS+=(--overwrite)
  [[ "${KEEP_INTERMEDIATE_FRAMES}" == "1" ]] && EXTRA_ARGS+=(--keep_intermediate_frames)

  PYTHONPATH_VALUE="${MEMORYCARD_ROOT}:${QWEN3_VL_REPO}:${PYTHONPATH:-}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH="${PYTHONPATH_VALUE}" \
  nohup "${PYTHON_BIN}" "${STEP2_PY}" \
    --out_root "${OUT_ROOT}" \
    --vlm_model_dir "${VLM_MODEL_DIR}" \
    --data_jsonl "${DATA_JSONL}" \
    --video_data_dir "${VIDEO_DATA_DIR}" \
    --jsonl_video_id_field "${JSONL_VIDEO_ID_FIELD}" \
    --jsonl_video_fileid_field "${JSONL_VIDEO_FILEID_FIELD}" \
    --worker_id "${WORKER_ID}" \
    --num_workers "${NUM_WORKERS}" \
    --selfread_max_new_tokens "${SELFREAD_MAX_NEW_TOKENS}" \
    --frame_timeout_sec "${FRAME_TIMEOUT_SEC}" \
    --attn_impl "${ATTN_IMPL}" \
    "${EXTRA_ARGS[@]}" \
    > "${LOG_DIR}/step2_worker${WORKER_ID}.log" 2>&1 &

  PIDS+=($!)
  echo "[STEP2] launched worker ${WORKER_ID} on GPU ${GPU_ID}, pid=${PIDS[-1]}"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then FAIL=1; fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[STEP2] some workers failed. Please check ${LOG_DIR}/step2_worker*.log" >&2
  exit 1
fi

echo "[STEP2] done"
