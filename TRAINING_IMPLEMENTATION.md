## Training Implementation Summary

This document summarizes the Perceiver IO grid-to-grid implementation, the PyTorch training pipeline, fixes made during bring-up, current settings, and how to run and inspect outputs.


## Overview

- Task: map integer grids of shape 17x17 to integer grids of shape 17x17 (values in [0, 9]).
- Model: Perceiver IO built from repository core blocks, with task-specific grid adapters.
- Training: pure PyTorch (no Lightning), cross-entropy per cell, AdamW, optional AMP.
- Dataset: JSON file at `data/training_data.json` with `train` and `test` splits.


## Implemented files (new)

- `perceiver/model/grid/backend.py`
  - `GridInputAdapter`: embeds input integers and adds 2D Fourier position encodings; outputs `(B, H*W, C)`.
  - `GridClassificationOutputAdapter`: linear head to 10 classes per position; reshapes to `(B, H, W, 10)`.
  - `GridQueryProvider`: uses the encoder’s adapted input (`x_adapted`) as decoder queries (one per output position).
  - `GridPerceiverIO`: wires `PerceiverEncoder` + `PerceiverDecoder` for the grid task.

- `perceiver/model/grid/__init__.py`
  - Exports `GridPerceiverIO`, `GridInputAdapter`, `GridClassificationOutputAdapter`, `GridQueryProvider`.

- `examples/training/grid/train.py`
  - `GridDataset`: reads the JSON and yields `(LongTensor[H,W], LongTensor[H,W])` pairs.
  - Training loop: CE loss over per-cell logits; per-cell and exact-grid accuracy metrics; checkpointing best val acc.
  - Optional mixed precision (AMP) when CUDA is available.
  - After each epoch, dumps test predictions as JSON for later visualization (see “Outputs” below).
  - CLI:
    - `--data` (default `data/training_data.json`)
    - `--epochs` (default `100`)
    - `--batch-size` (default `32`)
    - `--lr` (default `1e-3`)
    - `--weight-decay` (default `1e-2`)
    - `--seed` (default `42`)
    - `--save-dir` (default `checkpoints`)
    - `--no-amp` (disable mixed precision on CUDA)

- `PLAN.md`
  - High-level plan describing the Perceiver IO approach and design decisions for this task.


## Architecture details

### Input adapter (GridInputAdapter)
- Input: `x ∈ ℤ^{B×H×W}` with values in `[0, num_value_embeddings)` (by default `[0..9]`).
- Value embedding: `nn.Embedding(num_value_embeddings, value_embedding_dim)` → `(B, H, W, E)`.
- 2D Fourier position encoding: `FourierPositionEncoding(input_shape=(H, W), num_frequency_bands=F)` → `(B, H*W, P)`.
- Concatenate along channel dimension after flattening: `(B, H*W, C)` with `C = E + P`.

### Encoder (PerceiverEncoder)
- Cross-attends from latent array `(B, N, D)` to adapted input `(B, H*W, C)`, followed by self-attention blocks.
- For head dimension consistency and stability:
  - Encoder cross-attn `num_qk_channels` and `num_v_channels` are left as `None` (default to latent channels `D`, divisible by heads).

### Decoder (PerceiverDecoder)
- Query provider uses `x_adapted` to produce one query per output position (mirrors the optical flow backend pattern).
- Output adapter produces `(B, H, W, 10)` logits (10 classes per grid cell).
- Decoder cross-attn `num_qk_channels` and `num_v_channels` are set to `num_latent_channels` (`D`), ensuring divisibility by `num_heads`.

### Tensor shapes (summary)
- Input grid: `(B, 17, 17)`
- Adapted input: `(B, 289, C)`
- Latent array: `(B, N, D)`
- Decoder queries: `(B, 289, C)` from `x_adapted`
- Output logits: `(B, 17, 17, 10)`


## Training pipeline

### Loss
- Cross-entropy per cell:
  - Reshape logits `(B, H, W, 10)` → `(B*H*W, 10)`
  - Reshape targets `(B, H, W)` → `(B*H*W,)`
  - `nn.CrossEntropyLoss()`

### Optimizer and regularization
- `AdamW(lr=1e-3, weight_decay=1e-2)` by default (tunable via CLI).
- Optional AMP (enabled by default on CUDA; use `--no-amp` to disable).
- Gradient clipping can be added if needed (not currently enabled).

### Metrics
- Per-cell accuracy: fraction of correct cells across all examples.
- Exact-grid accuracy: fraction of examples where all 289 cells match.

### Checkpointing
- Saves best validation (test split) per-cell accuracy checkpoint to `--save-dir` (default `checkpoints/`), including optimizer state and epoch.

### Outputs (per epoch)
- Predictions dump: `checkpoints/test_preds_epoch{epoch:03d}.json` with structure:
  ```json
  {
    "samples": [
      { "pred": [[... 17 ...], ... 17 ...], "target": [[...], ...] },
      ...
    ]
  }
  ```
  Use this for downstream visualizations or analysis.


## Fixes and bring-up notes

1) Cross-attention channel divisibility error
   - Error: “`num_qk_channels must be divisible by num_heads`”.
   - Fix: Default encoder cross-attn qk/v channels; set decoder qk/v channels to `num_latent_channels` (`D`) so they are divisible by the number of heads.

2) Tensor shape mismatch in adapter concatenation
   - Error when concatenating embeddings with Fourier positions.
   - Fix: Flatten embeddings to `(B, H*W, E)` and concatenate with Fourier encodings `(B, H*W, P)` before feeding to the encoder.

3) Test predictions JSON export
   - Added a helper to dump predictions each epoch to facilitate later visualization.


## How to run

Basic run:
```
python examples/training/grid/train.py --data data/training_data.json --epochs 100 --batch-size 32
```

Useful flags:
- `--save-dir checkpoints`
- `--no-amp` (if you want to disable mixed precision on CUDA)
- `--lr 1e-3 --weight-decay 1e-2`

Artifacts:
- Best checkpoint: `checkpoints/gridio-epoch{E}-acc{ACC}.pt`
- Predictions per epoch: `checkpoints/test_preds_epoch{E}.json`


## Current state and settings

- Training and evaluation run successfully after the fixes above.
- Example produced artifacts observed:
  - Checkpoint like `checkpoints/gridio-epoch009-acc0.813.pt`
  - Prediction dumps like `checkpoints/test_preds_epoch044.json`
- The training script currently uses parameters passed in-code or via CLI. Recent local modifications increased model capacity substantially (e.g., higher embedding dim, more heads/layers/blocks). Ensure that GPU memory is sufficient; otherwise, reduce:
  - `num_latents`, `num_latent_channels`
  - `num_self_attention_layers_per_block`, `num_self_attention_blocks`
  - `num_frequency_bands`, `value_embedding_dim`
  - `batch-size`


## Suggested next steps

- Add a learning-rate schedule (e.g., cosine decay with warmup) and optionally gradient clipping.
- Split `train` into train/val subsets to avoid selecting checkpoints on the test set; or add a proper `val` split in the JSON if available.
- Add simple visualization notebooks to render prediction grids vs. targets using the epoch JSON files.
- Save and load model hyperparameters alongside checkpoints for reproducibility.
- Add tests for adapter shapes and end-to-end forward on synthetic data for quick CI verification.


## Pointers to key code

- Core building blocks: `perceiver/model/core/modules.py`, `perceiver/model/core/adapter.py`, `perceiver/model/core/position.py`
- Grid backend: `perceiver/model/grid/backend.py`
- Training loop: `examples/training/grid/train.py`
- Reference dense-task wiring: `perceiver/model/vision/optical_flow/backend.py` (similar query pattern)


