#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ImageNet-100 16K latent files")
    parser.add_argument("latent_dir", type=Path)
    args = parser.parse_args()

    for split in ("val", "train"):
        path = args.latent_dir / f"{split}_latents.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, mmap_mode="r") as data:
            indices = data["indices"]
            labels = data["labels"]
            if indices.ndim != 2 or indices.shape[1] != 256:
                raise ValueError(f"{path}: expected [N, 256], got {indices.shape}")
            if len(labels) != len(indices):
                raise ValueError(f"{path}: {len(labels)} labels for {len(indices)} rows")
            lo, hi = int(indices.min()), int(indices.max())
            if lo < 0 or hi >= 16384:
                raise ValueError(f"{path}: token range [{lo}, {hi}] is outside [0, 16383]")
            print(
                f"{split}: shape={indices.shape}, dtype={indices.dtype}, "
                f"range=[{lo}, {hi}], labels={len(labels)}, bytes={path.stat().st_size}"
            )


if __name__ == "__main__":
    main()
