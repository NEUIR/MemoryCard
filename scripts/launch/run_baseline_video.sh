#!/usr/bin/env bash
set -euo pipefail

LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-/path/to/lmms-eval}"
PRETRAINED="${PRETRAINED:-/path/to/Qwen3-VL-8B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-/path/to/output/eval_logs}"
LOG_SUFFIX="${LOG_SUFFIX:-qwen3_vl_8b_video_videomme}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAIN_PORT="${MAIN_PORT:-8000}"
TASK_NAME="${TASK_NAME:-videomme}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-8}"

if [[ "${LMMS_EVAL_ROOT}" == /path/to/* || "${PRETRAINED}" == /path/to/* || "${OUTPUT_PATH}" == /path/to/* ]]; then
  echo "[ERROR] Please edit LMMS_EVAL_ROOT, PRETRAINED, and OUTPUT_PATH at the top of this script." >&2
  exit 1
fi

mkdir -p "${OUTPUT_PATH}"
cd "${LMMS_EVAL_ROOT}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PORT}" \
  -m lmms_eval \
  --model qwen3_vl \
  --model_args "pretrained=${PRETRAINED},max_num_frames=${MAX_NUM_FRAMES},frame_mode=video" \
  --task "${TASK_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --log_samples_suffix "${LOG_SUFFIX}" \
  --output_path "${OUTPUT_PATH}"
