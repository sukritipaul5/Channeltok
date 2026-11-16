#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
TORCHRUN="${REPO_ROOT}/.venv/bin/torchrun"
MODEL="${MODEL:-GPT-B}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)}"
LATENT_DIR="${LATENT_DIR:-/fs/cml-projects/gaze/channeltok_latents/imagenet100_16k_256ch}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch}"
DECODER_CKPT="${DECODER_CKPT:-${TOKENIZER_DIR}/checkpoints/best.ckpt}"
DECODER_CONFIG="${DECODER_CONFIG:-${TOKENIZER_DIR}/resolved_train_config.yaml}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results_imagenet100k_16k}"
EPOCHS="${EPOCHS:-}"

case "${MODEL}" in
    GPT-B) MODEL_CONFIG="${REPO_ROOT}/configs/generation/gpt_b.yaml" ;;
    GPT-L) MODEL_CONFIG="${REPO_ROOT}/configs/generation/gpt_l.yaml" ;;
    *) echo "MODEL must be GPT-B or GPT-L." >&2; exit 1 ;;
esac

for required in "${LATENT_DIR}/train_latents.npz" "${DECODER_CKPT}" "${DECODER_CONFIG}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required input not found: ${required}" >&2
        exit 1
    fi
done
if [[ ! -x "${PYTHON}" || ! -x "${TORCHRUN}" ]]; then
    echo "Missing .venv. Run 'uv sync --python 3.10' first." >&2
    exit 1
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "No visible GPUs." >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
RUNTIME_CONFIG="${RESULTS_DIR}/resolved_${MODEL,,}_config.yaml"
cd "${REPO_ROOT}"

"${PYTHON}" - "${REPO_ROOT}/configs/generation/base.yaml" "${MODEL_CONFIG}" \
    "${LATENT_DIR}" "${DECODER_CKPT}" "${DECODER_CONFIG}" "${RESULTS_DIR}" \
    "${EPOCHS}" "${RUNTIME_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

base_path, model_path, latents, decoder_ckpt, decoder_config, results, epochs, destination = sys.argv[1:]
with open(base_path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
with open(model_path, encoding="utf-8") as handle:
    model_config = yaml.safe_load(handle)
model_config.pop("defaults", None)
config.update(model_config)
config.update(
    npz_path=latents,
    decoder_ckpt=decoder_ckpt,
    decoder_config=decoder_config,
    results_dir=results,
    cloud_save_path=str(Path(results) / "cloud"),
)
if epochs:
    config["epochs"] = int(epochs)
with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print(yaml.safe_dump(config, sort_keys=False))
PY

export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-online}"
echo "Launching ${MODEL} on ${NUM_GPUS} GPU(s)"
echo "Resolved config: ${RUNTIME_CONFIG}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1; config resolved successfully, not launching training."
    exit 0
fi
"${TORCHRUN}" --standalone --nproc_per_node="${NUM_GPUS}" \
    -m generation.train --config "${RUNTIME_CONFIG}"
