## Plan: Grid-to-Grid Perceiver IO (PyTorch, no Lightning)

This plan describes how to implement and train a Perceiver IO model that maps 17x17 integer grids to 17x17 integer grids using only PyTorch building blocks from this repository. It focuses exclusively on the Perceiver IO variant.

- **Input**: 17x17 integers in [0, 9]
- **Output**: 17x17 integers in [0, 9]
- **Dataset**: `data/training_data.json` with keys `train` (len=219) and `test`

## Background: Perceiver IO in this repo

Perceiver IO is assembled from a generic encoder and decoder, plus task-specific adapters:
- `PerceiverEncoder` cross-attends from learned latents to adapted inputs, then applies self-attention blocks.
- `PerceiverDecoder` cross-attends from output queries to the latent array, then an output adapter maps to task logits.

Key references (for shapes and wiring):
- Encoder and decoder forward passes in `perceiver/model/core/modules.py`:
  - `PerceiverEncoder.forward`: returns either `x_latent` or `(x_latent, x_adapted)`
  - `PerceiverDecoder.forward`: `output_query = output_query_provider(x_adapted)` → cross-attn → `output_adapter(...)`
- Query and adapters in `perceiver/model/core/adapter.py`:
  - `TrainableQueryProvider` (learned queries), `ClassificationOutputAdapter` (linear over query channels)
- A dense prediction example mirroring our need is optical flow:
  - `perceiver/model/vision/optical_flow/backend.py` uses `x_adapted` as the decoder query input and reshapes output to image shape.

Implications for a 17x17 grid-to-grid task:
- Treat the grid as a 2D “image” with 1 semantic channel, add 2D Fourier position encodings, flatten to a sequence of length `O = 289`.
- Use `x_adapted` as decoder queries (one query per grid cell), so the decoder predicts per-cell outputs in parallel.

## Data loading and preprocessing

### File format
`data/training_data.json`:
- `{ "train": [ { "input": [[...17] x 17], "output": [[...17] x 17] }, ... ], "test": [...] }`
- Values are integers in [0, 9].

### Dataset class (PyTorch)
- Implement `GridDataset(torch.utils.data.Dataset)` that:
  - Loads the JSON once in `__init__`.
  - Stores a list of pairs `(input_grid, output_grid)`, each as `torch.LongTensor` of shape `(17, 17)`.
  - Optionally supports a `transform` hook (e.g., value normalization if doing regression, or augmentation if desired).
- `__getitem__` returns `(x: LongTensor[17,17], y: LongTensor[17,17])`.

### Collation
- Use default collation; batches become `LongTensor[b, 17, 17]` for both `x` and `y`.

### Encoding strategy
- Recommended: **Per-cell classification** with 10 classes.
  - Loss: cross-entropy over per-cell logits.
  - Benefits: discrete target modeling; natural for values in [0, 9].
- Alternative: Regression to scalar in [0, 9] with MSE/Huber + rounding at inference. Not recommended here.

## Model design (Perceiver IO)

We build a small custom Perceiver IO for grids, inspired by `ImageClassifier` and `OpticalFlow` backends.

### Input adapter: `GridInputAdapter`
- Purpose: Transform the `(B, 17, 17)` integer grid into a sequence with position encodings.
- Steps:
  - Value embedding: `nn.Embedding(num_embeddings=10, embedding_dim=E)` to embed each integer (E ∈ {32, 64}).
  - Position encoding: `FourierPositionEncoding(input_shape=(17, 17), num_frequency_bands=F)` (F ∈ {16, 32}).
  - Concatenate embedded values and position encodings along channels.
  - Flatten spatial dims: `(B, 17, 17, C) -> (B, 289, C)` where `C = E + PE`, and `PE = 2 * dims * F + dims` with `dims=2`.
  - Set `num_input_channels = C`.

Notes:
- See `FourierPositionEncoding` in `perceiver/model/core/position.py` for channel math and usage.
- `OpticalFlowInputAdapter` shows the pattern: project features, flatten, concatenate Fourier position encodings, then return `(B, M, C)`.

### Encoder: `PerceiverEncoder`
- Construct with:
  - `input_adapter = GridInputAdapter(...)`
  - `num_latents`: 64–256 (start with 128 for 219 examples)
  - `num_latent_channels`: 128–256 (start with 256)
  - Attention heads: 4
  - `num_self_attention_layers_per_block`: 4–6 (start with 4)
  - `num_self_attention_blocks`: 1
  - Dropout: 0.0–0.1 (start with 0.1 for regularization)
  - For cross-attention qk/v channels: default to adapter `num_input_channels`

Shape summary:
- Input: `(B, 17, 17)` ints → adapter → `(B, 289, C)`
- Encoder latent: `(B, N, D)` where `N = num_latents`, `D = num_latent_channels`
- If `return_adapted_input=True`, also returns `x_adapted: (B, 289, C)`

### Decoder: queries and output adapter
- Queries: Use `x_adapted` as in optical flow (one query per grid position).
  - Implement a `GridQueryProvider` equivalent to `OpticalFlowQueryProvider`, which receives `x_adapted` and returns it as queries.
  - Then `num_output_query_channels = C`.
- Output adapter:
  - For classification: `GridClassificationOutputAdapter(num_classes=10, num_output_query_channels=C)` → linear to 10 classes per position and reshape to `(B, 17, 17, 10)`.
  - For regression: `GridRegressionOutputAdapter(num_output_query_channels=C)` → linear to 1 per position and reshape to `(B, 17, 17)`.

End-to-end forward:
1) `x_latent, x_adapted = encoder(x, return_adapted_input=True)`
2) `logits_or_values = decoder(x_latent, x_adapted=x_adapted)`

### Suggested default hyperparameters (small data)
- `E` (value embedding dim): 32
- `F` (Fourier bands): 16
- `C = E + (2 * F + 1) * 2 = 32 + (33 * 2) = 98`
- `num_latents`: 128
- `num_latent_channels`: 256
- Heads: 4
- Self-attn layers per block: 4
- Blocks: 1
- Cross/self widening factors: 1–2 (start with 1)
- Dropout in attention/MLP: 0.1

## Training (pure PyTorch)

### Loss and targets
- Classification (recommended):
  - Model output: `(B, 17, 17, 10)` logits
  - Targets: `(B, 17, 17)` `LongTensor` in `[0..9]`
  - Compute loss by reshaping:
    - `logits.view(B*289, 10)` and `targets.view(B*289)`
    - `nn.CrossEntropyLoss()`

### Optimizer, schedule, and regularization
- Optimizer: `AdamW(lr=1e-3, weight_decay=0.01)`
- Learning rate schedule (optional): cosine decay or step LR after warmup (e.g., 200–500 steps) given small dataset.
- Gradient clipping: clip global norm at 1.0–2.0.
- Mixed precision: `torch.cuda.amp.autocast` + `GradScaler` if on GPU.
- Dropout: 0.1 in attention and MLP for regularization.

### Dataloaders
- Split: use provided `train` and `test` lists.
- Batch size: start with 32 (adjust to memory); shuffle train, no shuffle test.
- Num workers: 2–4.

### Training loop skeleton
1) Seed RNGs for reproducibility.
2) Create `GridDataset` for train/test; `DataLoader` with batch size and workers.
3) Construct model (encoder+decoder) and move to device.
4) Configure optimizer, optional scheduler, AMP scaler.
5) For each epoch:
   - Train loop: forward, compute CE loss, backward, clip, optimizer step, scheduler step.
   - Eval loop on test: compute per-cell accuracy and loss.
   - Save best checkpoint by validation loss or accuracy.

### Checkpointing
- Save: model `state_dict`, optimizer state, scheduler state, epoch, global step, best metric.
- Filename pattern: `checkpoints/gridio-epoch{E}-acc{ACC:.3f}.pt`.

## Evaluation and metrics
- Per-cell accuracy: fraction of correct cells over all cells and examples.
- Exact-grid accuracy: fraction where all 289 cells match.
- Per-class accuracy (optional) and confusion matrix (flattened over all cells).
- For regression alternative: MAE/MSE per cell; but classification is preferred here.

## Implementation details (concrete steps and files)

Minimal additions (keep changes small and task-specific):
1) Implement adapters and model assembly (new module under `perceiver/model/vision/` or `perceiver/model/grid/`):
   - `GridInputAdapter` (embedding + Fourier positions + flatten)
   - `GridClassificationOutputAdapter` (linear to 10, reshape to `(B, 17, 17, 10)`)
   - `GridQueryProvider` (like `OpticalFlowQueryProvider`: identity over `x_adapted`)
   - `GridPerceiverIO` (wire encoder/decoder similarly to `OpticalFlow`)
2) Implement `GridDataset` and a small train script (e.g., `examples/training/grid/train.py`) that uses pure PyTorch training loop.
3) Add a config dataclass like other backends (`PerceiverIOConfig[GridEncoderConfig, GridDecoderConfig]`) if you want CLI/config parity; otherwise pass params directly.

Where to look for examples:
- Adapter patterns and wiring from `perceiver/model/vision/optical_flow/backend.py`.
- Core building blocks in `perceiver/model/core/modules.py` and `perceiver/model/core/adapter.py`.

## Tensor shapes (summary)
- Input grid: `(B, 17, 17)` ints in `[0..9]`
- After embedding: `(B, 17, 17, E)`
- Fourier positions: `(B, 17, 17, PE)` where `PE = 2 * (2 * F) + 2`
- Concatenated: `(B, 17, 17, C=E+PE)` → flatten → `(B, 289, C)`
- Encoder latent: `(B, N, D)`
- Decoder queries (`x_adapted`): `(B, 289, C)`
- Decoder output (classification): `(B, 17, 17, 10)`

## Default hyperparameters for this dataset size
- Value embedding dim `E`: 32
- Fourier bands `F`: 16
- `C`: 98 (approx., per formula above)
- `num_latents`: 128
- `num_latent_channels`: 256
- Heads: 4
- Self-attn layers per block: 4
- Blocks: 1
- Dropout: 0.1
- Optimizer: AdamW, lr=1e-3, wd=0.01
- Epochs: 50–200 (use early stopping on test set or hold out val from train if desired)
- Batch size: 32 (tune)

## Risks and mitigations
- Small train set (219): risk of overfitting
  - Use small model, dropout, weight decay, and early stopping.
  - Optionally perform K-fold cross-validation over the `train` list.
- Class imbalance: compute per-class stats; consider class weights in CE if skewed.
- Over-parameterized `C`: if memory constrained, reduce embedding dim `E` or Fourier bands `F`.

## Appendix: Relevant code references in this repo

- Encoder/decoder IO wiring (`PerceiverEncoder`, `PerceiverDecoder`, `PerceiverIO`):

```610:689:perceiver/model/core/modules.py
class PerceiverDecoder(nn.Module):
    ...
    def forward(self, x_latent, x_adapted=None, **kwargs):
        output_query = self.output_query_provider(x_adapted)
        output = self.cross_attn(output_query, x_latent).last_hidden_state
        return self.output_adapter(output, **kwargs)

class PerceiverIO(nn.Sequential):
    def __init__(self, encoder: PerceiverEncoder, decoder: PerceiverDecoder):
        super().__init__(encoder, decoder)
```

- Optical flow backend (dense queries via `x_adapted` and output reshape):

```39:79:perceiver/model/vision/optical_flow/backend.py
class OpticalFlowInputAdapter(InputAdapter):
    ...
    def forward(self, x):
        ...
        x = self.linear(x)
        x = rearrange(x, "b ... c -> b (...) c")
        pos_enc = self.position_encoding(b)
        return torch.cat([x, pos_enc], dim=-1)

class OpticalFlowOutputAdapter(OutputAdapter):
    ...
    def forward(self, x):
        x = self.linear(x) / self.rescale_factor
        return rearrange(x, "b (h w) c -> b h w c", h=self.image_shape[0])
```

- Query provider pattern:

```81:93:perceiver/model/vision/optical_flow/backend.py
class OpticalFlowQueryProvider(nn.Module, QueryProvider):
    ...
    def forward(self, x):
        assert x.shape[-1] == self.num_query_channels
        return x
```

This grid-to-grid model mirrors the optical flow pattern with different input features and output heads (10-way classification per grid cell). With small architectural sizes and standard PyTorch training, it should be straightforward to implement and iterate. 


