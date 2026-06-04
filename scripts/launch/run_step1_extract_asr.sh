#!/usr/bin/env bash
set -euo pipefail

# Project / script paths
MEMORYCARD_ROOT="${MEMORYCARD_ROOT:-/path/to/MemoryCard}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STEP1_PY="${STEP1_PY:-${MEMORYCARD_ROOT}/memorycard/build_memory/extract_asr.py}"
ASR_REPO_DIR="${ASR_REPO_DIR:-/path/to/Qwen3-ASR-main}"

# Model paths
QWEN3_ASR_MODEL_DIR="${QWEN3_ASR_MODEL_DIR:-/path/to/Qwen3-ASR-1.7B}"
FORCED_ALIGNER_MODEL_DIR="${FORCED_ALIGNER_MODEL_DIR:-/path/to/Qwen3-ForcedAligner-0.6B}"

# Data paths
# Note: the current ASR script expects a JSONL annotation file.
DATA_JSONL="${DATA_JSONL:-/path/to/Video-MME/test-00000-of-00001.jsonl}"
VIDEO_DATA_DIR="${VIDEO_DATA_DIR:-/path/to/Video-MME/data}"

# Output paths
OUT_ROOT="${OUT_ROOT:-/path/to/output/videomme_memory}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"
TIMING_DIR="${TIMING_DIR:-${OUT_ROOT}/timing}"
ASR_ROOT="${ASR_ROOT:-${OUT_ROOT}/asr}"
AUDIO_ROOT="${AUDIO_ROOT:-${OUT_ROOT}/audio_chunks}"

# GPU sharding: one worker per GPU
ASR_GPUS_STR="${ASR_GPUS_STR:-0}"
read -r -a ASR_GPUS <<< "${ASR_GPUS_STR}"

# ASR hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ASR_LANGUAGE="${ASR_LANGUAGE:-auto}"
ASR_HOTWORDS="${ASR_HOTWORDS:-}"
CHUNK_SEC="${CHUNK_SEC:-30}"
CHUNK_OVERLAP_SEC="${CHUNK_OVERLAP_SEC:-0}"
AUDIO_TIMEOUT_SEC="${AUDIO_TIMEOUT_SEC:-120}"
OVERWRITE="${OVERWRITE:-0}"
KEEP_WAV="${KEEP_WAV:-0}"
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
check_path "STEP1_PY" "${STEP1_PY}"
check_path "ASR_REPO_DIR" "${ASR_REPO_DIR}"
check_path "QWEN3_ASR_MODEL_DIR" "${QWEN3_ASR_MODEL_DIR}"
check_path "FORCED_ALIGNER_MODEL_DIR" "${FORCED_ALIGNER_MODEL_DIR}"
check_path "DATA_JSONL" "${DATA_JSONL}"
check_path "VIDEO_DATA_DIR" "${VIDEO_DATA_DIR}"
check_path "OUT_ROOT" "${OUT_ROOT}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${TIMING_DIR}" "${ASR_ROOT}" "${AUDIO_ROOT}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

NUM_WORKERS="${#ASR_GPUS[@]}"
PIDS=()

cat <<INFO
==================================================
[STEP1] 🎧 Qwen3-ASR chunked transcription
STEP1_PY=${STEP1_PY}
QWEN3_ASR_MODEL_DIR=${QWEN3_ASR_MODEL_DIR}
FORCED_ALIGNER_MODEL_DIR=${FORCED_ALIGNER_MODEL_DIR}
DATA_JSONL=${DATA_JSONL}
VIDEO_DATA_DIR=${VIDEO_DATA_DIR}
OUT_ROOT=${OUT_ROOT}
ASR_GPUS_STR=${ASR_GPUS_STR}
CHUNK_SEC=${CHUNK_SEC}
CHUNK_OVERLAP_SEC=${CHUNK_OVERLAP_SEC}
==================================================
INFO

for i in "${!ASR_GPUS[@]}"; do
  GPU_ID="${ASR_GPUS[$i]}"
  WORKER_ID="${i}"
  EXTRA_ARGS=()
  [[ "${OVERWRITE}" == "1" ]] && EXTRA_ARGS+=(--overwrite)
  [[ "${KEEP_WAV}" == "1" ]] && EXTRA_ARGS+=(--keep_wav)

  PYTHONPATH_VALUE="${MEMORYCARD_ROOT}:${ASR_REPO_DIR}:${PYTHONPATH:-}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH="${PYTHONPATH_VALUE}" \
  nohup "${PYTHON_BIN}" "${STEP1_PY}" \
    --out_root "${OUT_ROOT}" \
    --qwen3_asr_model_dir "${QWEN3_ASR_MODEL_DIR}" \
    --forced_aligner_model_dir "${FORCED_ALIGNER_MODEL_DIR}" \
    --data_jsonl "${DATA_JSONL}" \
    --video_data_dir "${VIDEO_DATA_DIR}" \
    --jsonl_video_id_field "${JSONL_VIDEO_ID_FIELD}" \
    --jsonl_video_fileid_field "${JSONL_VIDEO_FILEID_FIELD}" \
    --worker_id "${WORKER_ID}" \
    --num_workers "${NUM_WORKERS}" \
    --language "${ASR_LANGUAGE}" \
    --hotwords "${ASR_HOTWORDS}" \
    --chunk_sec "${CHUNK_SEC}" \
    --chunk_overlap_sec "${CHUNK_OVERLAP_SEC}" \
    --batch_size "${BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --audio_timeout_sec "${AUDIO_TIMEOUT_SEC}" \
    "${EXTRA_ARGS[@]}" \
    > "${LOG_DIR}/step1_worker${WORKER_ID}.log" 2>&1 &

  PIDS+=($!)
  echo "[STEP1] launched worker ${WORKER_ID} on GPU ${GPU_ID}, pid=${PIDS[-1]}"
done

FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then FAIL=1; fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[STEP1] some workers failed. Please check ${LOG_DIR}/step1_worker*.log" >&2
  exit 1
fi

echo "[STEP1] done"
