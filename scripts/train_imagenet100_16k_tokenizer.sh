#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
TORCHRUN="${REPO_ROOT}/.venv/bin/torchrun"
SOURCE_CONFIG="${SOURCE_CONFIG:-${REPO_ROOT}/configs/num_channels/ch_256_16k.yaml}"
DATA_ROOT="${DATA_ROOT:-/fs/cml-projects/gaze/data/imagenet100/imagenet100_merged}"
OUTPUT_DIR="${OUTPUT_DIR:-/fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
RUNTIME_CONFIG="${OUTPUT_DIR}/resolved_train_config.yaml"

if [[ ! -x "${PYTHON}" || ! -x "${TORCHRUN}" ]]; then
    echo "Missing .venv. Run 'uv sync --python 3.10' in ${REPO_ROOT}." >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}/train" || ! -d "${DATA_ROOT}/val" ]]; then
    echo "Expected ImageFolder train/ and val/ under ${DATA_ROOT}." >&2
    exit 1
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "No visible GPUs." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

"${PYTHON}" - "${SOURCE_CONFIG}" "${DATA_ROOT}" "${OUTPUT_DIR}" "${BATCH_SIZE}" "${MAX_EPOCHS}" "${RUNTIME_CONFIG}" <<'PY'
import sys
from omegaconf import OmegaConf
from utils.checkpoint import load_config

source, data_root, output_dir, batch_size, max_epochs, destination = sys.argv[1:]
config = load_config(source)
config.data.dataset = "imagefolder"
config.data.data_dir = data_root
config.data.batch_size = int(batch_size)
config.training.max_epochs = int(max_epochs)
config.output_dir = output_dir
OmegaConf.save(config, destination)
print(OmegaConf.to_yaml(config, resolve=True))
PY

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
echo "Launching tokenizer training on ${NUM_GPUS} GPU(s)"
echo "Resolved config: ${RUNTIME_CONFIG}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1; config resolved successfully, not launching training."
    exit 0
fi
"${TORCHRUN}" --standalone --nproc_per_node="${NUM_GPUS}" \
    -m tokenizer.train --config "${RUNTIME_CONFIG}"
