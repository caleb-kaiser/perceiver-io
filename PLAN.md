## Plan: Episodic Perceiver IO for ARC-AGI Grid Puzzles

### Scope and goal
- Build an episodic, meta-learning variant of Perceiver IO that induces a puzzle-specific transformation from a small support set of input/output grid pairs and applies it to a new query input grid.
- Focus exclusively on Perceiver IO (not Perceiver/AR). Keep changes minimal and localized to a new backend and a dedicated training script (no Lightning).

Target setting (default, configurable):
- Each episode = one puzzle
- Support = K examples (default K=3)
- Query = 1 input grid (supervised training has its target; inference doesn’t)
- Grid size = 17×17; values in [0, 9] (vocabulary can be generalized)


### Guiding design principle
Use Perceiver IO’s encoder latent array as an episodic memory written from the support examples. The decoder then answers the query by cross-attending from output-position queries (derived from the query input grid) into that support-written latent memory.


### Architectural overview
We extend the existing grid backend to an episodic variant:

- Encoder input: a single concatenated token sequence containing
  - K support input grids
  - K support output grids
  - 1 query input grid
- Decoder query: tokens derived only from the query input grid (one query per output cell)
- Output: per-cell classification logits for the query output grid

This preserves the Perceiver IO pattern:
- Encoder: latent array cross-attends to a long, task-specific input sequence (here: all support and query-input tokens), then self-attends.
- Decoder: output queries (here: query-input tokens) cross-attend into latents and are mapped to per-cell classes.


### Key components to add
1) Episodic input adapter
   - `EpisodicGridInputAdapter` (new):
     - Reuses the single-grid embedding routine (value embedding + 2D Fourier position encoding) to encode any grid into `(B, H*W, C_base)`.
     - Adds:
       - Role embeddings: one of {support_input, support_output, query_input}.
       - Pair-index embeddings for support examples: {0, …, K-1}.
     - Returns a concatenated sequence:
       `[support_0_in, support_0_out, support_1_in, support_1_out, ..., support_{K-1}_in, support_{K-1}_out, query_in]`
       Each segment has shape `(B, H*W, C_base + C_role + C_pair)`.
     - Variable K is supported by concatenation; pair-index embeddings can be implemented as a small learned table up to a configurable `max_support` (default 5), or via sinusoidal embedding of the scalar pair index.

2) Query-only input adapter
   - Reuse the existing `GridInputAdapter` for the query’s output queries (one token per output position).
   - Its channel count defines `num_output_query_channels` for the decoder and output adapter.

3) Episodic backend
   - `EpisodicGridPerceiverIO` (new):
     - Encoder: `PerceiverEncoder(input_adapter=EpisodicGridInputAdapter, ...)`.
     - Decoder:
       - `output_query_provider = GridQueryProvider(num_query_channels=query_adapter.num_input_channels)`
       - `output_adapter = GridClassificationOutputAdapter(grid_shape, num_output_query_channels=query_adapter.num_input_channels, num_classes)`
     - Forward signature (train): `forward(support_x: LongTensor[B,K,H,W], support_y: LongTensor[B,K,H,W], query_x: LongTensor[B,H,W]) -> logits[B,H,W,num_classes]`
       - Build episodic input (support_x, support_y, query_x) and feed to encoder.
       - Build query-only adapted tokens via `GridInputAdapter(query_x)` and pass as `x_adapted` to the decoder.
       - Decoder returns per-cell logits; compute CE against `query_y` in the training loop.
     - Forward signature (inference): `forward_episode(support_x, support_y, query_x)` with no targets.

4) Dataset and training loop (pure PyTorch)
   - `EpisodicGridDataset`:
     - Each item loads a puzzle with >= K+1 examples.
     - Samples/augments K support pairs `(x_i, y_i)` and 1 query `(x_q, y_q)` (during training `y_q` exists; at test-time you may withhold).
     - Returns tensors shaped `(K,H,W)` for support_x/support_y and `(H,W)` for query_x/query_y.
   - Training:
     - Loss: per-cell CrossEntropy on predicted `y_q` (flatten `(B*H*W, C)` vs `(B*H*W,)`), same as current grid training.
     - Metrics: per-cell accuracy; exact-grid accuracy on the query prediction.
     - AMP, AdamW, LR-schedule/grad clipping optional.


### Data and episode format
Recommended JSON structure for episodic data:
```json
{
  "puzzles": [
    {
      "examples": [
        { "input": [[...17...], ...17...], "output": [[...17...], ...17...] },
        ...
      ]
    },
    ...
  ]
}
```
- Training sampler enforces each episode has at least K+1 examples.
- Optional augmentations per episode:
  - 90° rotations and flips
  - Color-value permutation (bijection on [0..9])
  - Crops/pads if you later generalize beyond 17×17


### Tensor shapes (summary)
- Base single-grid adapter (`GridInputAdapter`):
  - Input grid `(B,H,W)` → `(B, H*W, C_base)`
- Episodic adapter (`EpisodicGridInputAdapter`):
  - Support input `(B,K,H,W)` → `(B, K*H*W, C_base + C_role + C_pair)`
  - Support output `(B,K,H,W)` → `(B, K*H*W, C_base + C_role + C_pair)`
  - Query input `(B,H,W)` → `(B, H*W, C_base + C_role + C_pair)`
  - Concatenated episodic input: `(B, M_epi, C_epi)` with `M_epi = (2K+1)*H*W` and `C_epi = C_base + C_role + C_pair`
- Encoder latents: `(B, N, D)`
- Decoder queries (query-only adapter): `(B, H*W, C_query)`
- Output logits: `(B, H, W, num_classes)`


### How it fits existing code
- Encoder/decoder are unchanged; we only supply:
  - A new input adapter that understands episodic structure and returns a long concatenated sequence.
  - A decoder query provider that uses query-only adapted tokens (same `GridQueryProvider` pattern used today).
- The pattern matches the existing grid backend and optical-flow backend: decoder queries are derived from inputs while the encoder writes context into latents.


### Training pipeline
1) Dataloader
   - Batch of episodes: collate lists of `(support_x, support_y, query_x, query_y)` to tensors:
     - `support_x`: `(B,K,H,W)`
     - `support_y`: `(B,K,H,W)`
     - `query_x`: `(B,H,W)`
     - `query_y`: `(B,H,W)`
   - Optional per-episode random augmentation and color remapping.

2) Forward
   - `logits = model(support_x, support_y, query_x)` → `(B,H,W,num_classes)`

3) Loss and metrics
   - CE over flattened cells; per-cell and exact-grid accuracy.

4) Optimization
   - AdamW; consider cosine schedule with warmup; optional grad clipping (global norm or value clip).

5) Checkpointing and logging
   - Track best validation per-cell accuracy; also report exact-grid accuracy.
   - Optionally dump per-episode predictions for visualization (as done in the single-grid trainer).


### Minimal implementation plan
Files to add (keeping modifications isolated):
- `perceiver/model/grid/episodic.py`
  - `EpisodicGridInputAdapter`
    - Reuse components from `GridInputAdapter` (value embedding and Fourier position encoding).
    - Add `nn.Embedding` for roles (size 3) and for pair indices (size `max_support`).
    - Concatenate role and pair embeddings to each token.
    - Forward accepts a struct (e.g., dict) with `support_in`, `support_out`, `query_in` and returns the concatenated sequence.
  - `EpisodicGridPerceiverIO`
    - Build encoder with `EpisodicGridInputAdapter`.
    - Build a separate `GridInputAdapter` for query-only decoding.
    - Use `GridQueryProvider` and `GridClassificationOutputAdapter` like the single-grid model, with `num_output_query_channels` from the query-only adapter.
    - `forward(support_x, support_y, query_x)` encodes the episode; decodes from query-only adapted tokens.

- `examples/training/grid/train_episodic.py`
  - `EpisodicGridDataset` and training loop mirroring `examples/training/grid/train.py`, adapted to episode sampling.
  - CLI: `--data`, `--episodes-per-epoch`, `--support-k`, `--batch-size`, `--lr`, `--epochs`, `--no-amp`, `--save-dir`, augmentation flags.

Optional (later):
- `perceiver/model/grid/__init__.py` export of the new backend class.


### Hyperparameters and defaults
- `K` (support size): 3
- `num_latents` `N`: 17×17 (289) or smaller (e.g., 128) depending on budget
- `num_latent_channels` `D`: 256–512
- Heads/layers:
  - Cross/self heads: 8–16
  - Self-attn layers per block: 8–16
  - Self-attn blocks: 4–16 (weight sharing can reduce params)
- Dropout: 0.1–0.2
- Embedding sizes:
  - Value embedding: 32–64
  - Positional Fourier bands: 16–32
  - Role embedding: 8–16
  - Pair embedding: 8–16


### Ablations and variants
- Where to put role/pair signals:
  - Concatenate as channels (proposed), or add as bias via small MLP.
  - Try shared vs separate value embeddings for inputs and outputs.
  - Try omitting support outputs (use inputs-only memory) as a control.

- Episode tokenization layout:
  - Interleave per-pair `[x_i, y_i]` vs grouping all X then all Y. The per-pair interleave should aid locality; keep as default.

- Query provider options:
  - Use query-only adapter (proposed).
  - Alternative: learned queries plus conditioning on query input via an additional encoder pass (heavier).

- Few-shot size:
  - Train with K∈{1,2,3} via sampling to improve robustness.

- Augmentation:
  - Study color remap and rotations to encourage permutation/rotation invariance.


### Risks and mitigations
- Latent capacity: When `M_epi = (2K+1) H W` grows, ensure `N` and `D` are sufficient. Mitigate with more self-attn depth and adequate `N` (≥ H*W is safe but expensive).
- Leakage of query target: Ensure the episodic adapter never encodes `query_y` during training.
- Overfitting to color indices: Use color permutation augmentation.
- Memory: Use AMP, reduce `N`, `D`, or layers if OOM; consider gradient checkpointing already supported by core modules.


### Milestones
1) Backend and adapter
   - Implement `EpisodicGridInputAdapter` and `EpisodicGridPerceiverIO`.
   - Unit test on synthetic episodes (shape checks; forward pass).
2) Training loop
   - Implement `EpisodicGridDataset`, sampler, collate, augmentation.
   - Train sanity-check runs; verify loss decreases; evaluate per-cell/exact-grid metrics.
3) Evaluation/inference
   - Script path to load support examples and predict for a withheld query input; JSON or image visualization.
4) Ablations
   - Toggle role/pair embeddings, K, and layout; track metrics.


### Minimal surface changes (commit plan)
- Add `perceiver/model/grid/episodic.py`
- Add `examples/training/grid/train_episodic.py`
- Optionally export from `perceiver/model/grid/__init__.py`
- No edits to core modules are required.


### API sketch (for reference)
Episodic backend construction (defaults omitted):
```python
model = EpisodicGridPerceiverIO(
    grid_shape=(17, 17),
    num_classes=10,
    max_support=5,   # for pair-index embeddings
    # encoder/decoder hyperparams...
)
```
Training step:
```python
logits = model(support_x, support_y, query_x)   # (B,H,W,C)
loss = ce(logits.view(B*H*W, C), query_y.view(B*H*W))
```


### What success looks like
- On held-out episodes, exact-grid accuracy for the query output significantly exceeds single-pair training baselines.
- Robustness to color permutations and geometric transforms via augmentation.
- Clean integration with existing Perceiver IO code paths; small, isolated code surface area.


