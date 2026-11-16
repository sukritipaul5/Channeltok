#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DATA_ROOT="${DATA_ROOT:-/fs/cml-projects/gaze/data/imagenet100/imagenet100_merged}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch}"
CHECKPOINT="${CHECKPOINT:-${TOKENIZER_DIR}/checkpoints/best.ckpt}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-${TOKENIZER_DIR}/resolved_train_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/fs/cml-projects/gaze/channeltok_latents/imagenet100_16k_256ch}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-16}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}. Run 'uv sync --python 3.10' first." >&2
    exit 1
fi
for required in "${CHECKPOINT}" "${TOKENIZER_CONFIG}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required file not found: ${required}" >&2
        exit 1
    fi
done
if [[ ! -d "${DATA_ROOT}/train" || ! -d "${DATA_ROOT}/val" ]]; then
    echo "Expected ImageFolder train/ and val/ under ${DATA_ROOT}." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

"${PYTHON}" -m tokenizer.extract_latents \
    --checkpoint "${CHECKPOINT}" \
    --config "${TOKENIZER_CONFIG}" \
    --data_dir "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --splits val train \
    --inference_t 256 \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --device cuda

"${PYTHON}" "${REPO_ROOT}/scripts/verify_imagenet100_16k_latents.py" "${OUTPUT_DIR}"
