#!/usr/bin/env bash

set -euo pipefail
umask 002

DATA_PARENT="${DATA_PARENT:-/fs/cml-projects/gaze/data/imagenet100}"
ARCHIVE="${ARCHIVE:-${DATA_PARENT}/imagenet100.zip}"
RAW_DIR="${RAW_DIR:-${DATA_PARENT}/raw}"
DATA_ROOT="${DATA_ROOT:-${DATA_PARENT}/imagenet100_merged}"
URL="${IMAGENET100_URL:-https://www.kaggle.com/api/v1/datasets/download/ambityga/imagenet100}"

mkdir -p "${DATA_PARENT}" "${RAW_DIR}" "${DATA_ROOT}/train" "${DATA_ROOT}/val"

if [[ ! -f "${ARCHIVE}" ]]; then
    echo "Downloading ImageNet-100 to ${ARCHIVE}"
    curl -L --fail --retry 3 -o "${ARCHIVE}" "${URL}"
else
    echo "Reusing ${ARCHIVE}"
fi

if [[ ! -d "${RAW_DIR}/train.X1" || ! -d "${RAW_DIR}/val.X" ]]; then
    echo "Extracting ${ARCHIVE} into ${RAW_DIR}"
    unzip -q -o "${ARCHIVE}" -d "${RAW_DIR}"
else
    echo "Raw ImageNet-100 shards already exist in ${RAW_DIR}"
fi

for shard_root in train.X1 train.X2 train.X3 train.X4; do
    if [[ ! -d "${RAW_DIR}/${shard_root}" ]]; then
        echo "Missing expected shard: ${RAW_DIR}/${shard_root}" >&2
        exit 1
    fi
    for class_dir in "${RAW_DIR}/${shard_root}"/*; do
        [[ -d "${class_dir}" ]] || continue
        ln -sfn "${class_dir}" "${DATA_ROOT}/train/$(basename "${class_dir}")"
    done
done

if [[ ! -d "${RAW_DIR}/val.X" ]]; then
    echo "Missing expected validation shard: ${RAW_DIR}/val.X" >&2
    exit 1
fi
for class_dir in "${RAW_DIR}/val.X"/*; do
    [[ -d "${class_dir}" ]] || continue
    ln -sfn "${class_dir}" "${DATA_ROOT}/val/$(basename "${class_dir}")"
done

train_classes=$(find -L "${DATA_ROOT}/train" -mindepth 1 -maxdepth 1 -type d | wc -l)
val_classes=$(find -L "${DATA_ROOT}/val" -mindepth 1 -maxdepth 1 -type d | wc -l)
train_images=$(find -L "${DATA_ROOT}/train" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) | wc -l)
val_images=$(find -L "${DATA_ROOT}/val" -type f \( -iname '*.jpeg' -o -iname '*.jpg' -o -iname '*.png' \) | wc -l)

echo "ImageNet-100 is ready at ${DATA_ROOT}"
echo "Train: ${train_classes} classes, ${train_images} images"
echo "Val:   ${val_classes} classes, ${val_images} images"

if [[ "${train_classes}" -ne 100 || "${val_classes}" -ne 100 ]]; then
    echo "Expected 100 train and validation classes; inspect the extracted archive." >&2
    exit 1
fi
