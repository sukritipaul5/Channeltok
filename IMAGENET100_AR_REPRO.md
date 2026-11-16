# ImageNet-100 ChannelTok + AR reproduction

This reproduces the smaller unconditional generation experiment: train a 256-channel ChannelTok tokenizer with a 16,384-entry BSQ codebook, extract fixed 256-token sequences, and train GPT-B or GPT-L on ImageNet-100.

## Expected shared paths

The scripts use these defaults; override any of them with environment variables if your storage differs.

```text
/fs/cml-projects/gaze/data/imagenet100/imagenet100_merged/
├── train/<class>/*.JPEG
└── val/<class>/*.JPEG

/fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch/
/fs/cml-projects/gaze/channeltok_latents/imagenet100_16k_256ch/
```

## 1. Install

```bash
cd /fs/cml-projects/gaze/Channeltok
uv sync --python 3.10
```

Run the remaining commands inside a GPU allocation. `NUM_GPUS` means GPUs on one node; the supplied launchers use single-node `torchrun`.

## 2. Download ImageNet-100

The original experiment used the Kaggle `ambityga/imagenet100` archive: 100 classes, approximately 130K training images and 5K validation images. The preparation script downloads it and merges its four training shards into ImageFolder layout using symlinks.

```bash
bash scripts/prepare_imagenet100.sh
```

The final lines must report 100 train classes and 100 validation classes. To use another location:

```bash
DATA_PARENT=/shared/path/imagenet100 bash scripts/prepare_imagenet100.sh
```

## 3. Train the 256-channel/16K tokenizer

Start with the validated conservative batch of 16 images per GPU. The original H100 run used 32 per GPU; increase it only after a short memory check.

```bash
NUM_GPUS=4 BATCH_SIZE=16 \
  bash scripts/train_imagenet100_16k_tokenizer.sh
```

For a one-epoch environment and memory check:

```bash
NUM_GPUS=1 BATCH_SIZE=8 MAX_EPOCHS=1 WANDB_MODE=disabled \
  OUTPUT_DIR=/fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch_demo \
  bash scripts/train_imagenet100_16k_tokenizer.sh
```

The full recipe runs 100 epochs, starts GAN loss at epoch 25, and automatically resumes from `checkpoints/last.ckpt`. Expected outputs:

```text
/fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch/
├── resolved_train_config.yaml
└── checkpoints/
    ├── best.ckpt
    └── last.ckpt
```

Before extracting latents, reconstruct validation images at the full 256-channel budget and inspect them:

```bash
.venv/bin/python -m tokenizer.evaluate \
  --checkpoint /fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch/checkpoints/best.ckpt \
  --config /fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch/resolved_train_config.yaml \
  --data_dir /fs/cml-projects/gaze/data/imagenet100/imagenet100_merged \
  --inference_t 256
```

## 4. Extract and verify latents

The extractor runs validation first, then training, and fixes the token budget at 256 channels.

```bash
bash scripts/extract_imagenet100_16k_latents.sh
```

It creates `val_latents.npz` and `train_latents.npz`, then verifies:

- shape `[N, 256]`;
- one label per row;
- token range within `[0, 16383]`;
- dtype and file size.

The expected dataset sizes are approximately 5K validation and 130K training images. Treat a mismatch against the downloaded ImageFolder counts as an extraction failure.

## 5. Train the unconditional AR model

Run a one-epoch GPT-B check first:

```bash
MODEL=GPT-B NUM_GPUS=4 EPOCHS=1 WANDB_MODE=disabled \
  RESULTS_DIR=/fs/cml-projects/gaze/channeltok_ar/imagenet100_16k_demo \
  bash scripts/train_imagenet100_16k_ar.sh
```

Then launch the original full GPT-B recipe by omitting `EPOCHS`:

```bash
MODEL=GPT-B NUM_GPUS=8 bash scripts/train_imagenet100_16k_ar.sh
```

The original configuration is unconditional (`cls_token_num: 0`), uses fixed 256-token sequences, global batch 256, learning rate `4e-4`, cosine decay with 10% warmup, and 300 epochs. GPT-L is also available:

```bash
MODEL=GPT-L NUM_GPUS=8 bash scripts/train_imagenet100_16k_ar.sh
```

For reference, the historical GPT-B run reached loss `0.79` and token accuracy `0.755` after 152,100 steps. Early training should begin near `ln(16384) = 9.70` and move downward; stop and inspect if loss is NaN or flat.

## 6. Generate and inspect images

The preserved `scripts/generate_16k.py` produces unconditional samples and real-token completions. Supply the paths produced above:

```bash
.venv/bin/python scripts/generate_16k.py \
  --gpt-ckpt /path/to/results/000-GPT-B/checkpoints/latest.pt \
  --decoder-ckpt /fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch/checkpoints/best.ckpt \
  --decoder-config /fs/cml-projects/gaze/channeltok_ckpts/imagenet100_16k_256ch/resolved_train_config.yaml \
  --val-latents /fs/cml-projects/gaze/channeltok_latents/imagenet100_16k_256ch/val_latents.npz \
  --output-dir ./imagenet100_generation
```

The historical run used temperature `0.85` and top-k `200`. It produced coherent unconditional samples and progressively closer completions as the real-token prefix increased from 10% to 50%.

## Path overrides

The launchers accept these environment variables:

```text
DATA_ROOT, OUTPUT_DIR, TOKENIZER_DIR, CHECKPOINT, TOKENIZER_CONFIG,
LATENT_DIR, DECODER_CKPT, DECODER_CONFIG, RESULTS_DIR, NUM_GPUS,
BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS, EPOCHS, MODEL, WANDB_MODE, DRY_RUN
```

Each launcher prints its fully resolved YAML before training. Preserve that YAML with the resulting checkpoints; it records the exact paths and hyperparameters used.
