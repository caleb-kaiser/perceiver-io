Perceiver IO Episodic + ACT: Implementation Plan

Overview

- Goal: Add an Adaptive Computation Time (ACT) loop around the latent backbone of the episodic, meta-learning Perceiver IO model (`EpisodicGridPerceiverIO`) and adapt the training pipeline to supervise both prediction accuracy and computation budget.
- Scope: Only the Perceiver IO variant and its episodic grid model. Keep changes minimal and contained; avoid modifying generic core modules unless strictly necessary.
- High-level idea: Perform the initial cross-attention once to seed the latent array from the episodic input sequence, then iterate latent self-attention with an ACT controller that decides, per example, how many refinement steps are needed before halting. Aggregate intermediate latent states using ACT weighting and decode once into the output grid queries.

Context Recap (Current Design)

- Episodic input encoding:
  - `EpisodicGridInputAdapter` builds a single sequence by concatenating K support input grids, K support output grids, and the single query input grid. Roles and pair embeddings distinguish segments; 2D Fourier features encode positions.
- Encoder/decoder backbone (Perceiver IO):
  - Encoder seeds a latent array via cross-attention, then refines with self-attention blocks; decoder cross-attends from output queries (built from the query input grid) to latents and projects to class logits per grid cell.
- Episodic model specifics (`perceiver/model/grid/episodic.py`):
  - To avoid double adaptation, `EpisodicGridPerceiverIO.forward` manually routes through `encoder.latent_provider`, `encoder.cross_attn_1`, and (shared) self-attention blocks.

Why ACT and Where to Apply It

- Motivation: Different puzzles vary in difficulty. A fixed number of self-attention blocks either over-computes simple cases or under-computes complex ones. ACT allows dynamic depth per example with a penalty for extra computation.
- Placement: Wrap ACT around the latent refinement stage (self-attention). Keep the seeding cross-attention fixed at step 0 (always run once to initialize latents). Decode after ACT completes.

Design Decisions

1) Halting granularity
- Start simple with example-wise halting (one halting decision per batch element, shared across all latents). This keeps code localized and avoids complex variable-length masks per latent token.
- Optionally extend later to per-latent halting (token-wise ACT) for finer control.

2) What constitutes a “step”
- One step = one “self-attention block” application (the same unit already repeated in the encoder). If the model uses shared self-attention weights across blocks, we reuse the shared block; otherwise we reuse the first block weights for steps beyond the first to minimize changes.

3) Aggregation of intermediate states
- Use standard ACT aggregation: the final latent state is the weighted sum of intermediate latent states with weights given by the step-wise halting probabilities, including the remainder on the final step.

4) Halting head
- A small projection from the pooled latents to a scalar per example: p_t = sigmoid(Linear(LayerNorm(mean_pool(latents)))) with optional temperature. Keep it minimal to avoid destabilizing training.

5) Losses
- Task loss: unchanged (episodic grid classification over the output grid).
- Ponder loss: λ_ponder * expected_steps (per example), averaged across the batch. Expected steps is the ACT aggregate of step usage.

6) Mixed precision and numerics
- Maintain halting accumulators in float32 even under AMP; use `.float()` casts where necessary. Keep dropout and model precision unchanged.

Public API and Configuration

- Add optional ACT parameters to `EpisodicGridPerceiverIO`:
  - `act_enabled: bool = False`
  - `act_max_steps: int = num_self_attention_blocks` (upper bound)
  - `act_threshold: float = 1.0 - 1e-2` (halting mass needed to stop)
  - `act_epsilon: float = 1e-2` (stability slack)
  - `act_ponder_cost: float = 1e-3` (λ_ponder; tune)
  - `act_min_steps: int = 1` (optional warmup steps before halting allowed)
  - `act_temperature: float = 1.0` (optional scaling for halting logits)
- Surface these in the training script via flags; default to ACT off to preserve current behavior.

Core Algorithm (Example-wise ACT over self-attention)

Initialization

1) Build episodic adapted inputs X_adapted and initialize latents L0 via cross-attention as today.
2) Set:
   - step = 0
   - halting_acc = zeros([B], float32)
   - remainder = zeros([B], float32)
   - act_weights_sum = zeros([B, 1, 1], float32)  // optional for sanity checks
   - L_agg = zeros_like(L0, float32)              // weighted sum of latents

Loop

For t in 1..act_max_steps:

1) Compute new latent state: L_t = SelfAttentionBlock(L_{t-1}).
2) Compute halting probability per example:
   - h_t = sigmoid(W · LN(mean_pool(L_t)) / act_temperature)
   - p_t = clamp(h_t, 0, 1)
3) Determine still-active examples: m_active = (halting_acc < act_threshold) as float32.
4) Compute new mass to add:
   - new_mass = m_active * (1 - halting_acc)
   - weight_t = where(halting_acc + p_t >= act_threshold, new_mass, p_t)
5) Update aggregate latent and accumulators:
   - L_agg += weight_t.view(B, 1, 1) * L_t
   - halting_acc += weight_t
   - act_weights_sum += weight_t.view(B, 1, 1)
6) Early exit if all examples have halting_acc >= act_threshold and t >= act_min_steps.

Output

- Use L_agg (ACT-aggregated latents) as the encoder output to the Perceiver decoder.
- Ponder metrics:
  - expected_steps = sum_t weight_t  (per example)
  - remainder = 1 - sum_t weights if < threshold before max steps (implicitly captured by weight_t on the last step)

Training Objective

Loss = task_loss + act_ponder_cost * mean(expected_steps)

- task_loss: unchanged episodic loss (per-token CE + episode-level weighting) already used in your script.
- expected_steps: computed per example from ACT; average across batch.

Minimal Code Changes (File-local)

- Keep changes contained to the episodic model and training script:
  1) `perceiver/model/grid/episodic.py`
     - Add ACT config parameters to `EpisodicGridPerceiverIO.__init__` (with defaults).
     - Add halting head modules: `LayerNorm(num_latent_channels)` + `Linear(num_latent_channels, 1)`.
     - Implement an internal `_act_refine_latents(latents)` that runs the loop above and returns `(latents_agg, expected_steps)`.
     - In `forward`, after the initial cross-attention:
       - If `act_enabled`: call `_act_refine_latents` to obtain `x_latents` and `expected_steps`.
       - Else: keep existing fixed self-attention pass (current behavior).
     - Return logits as today; attach `expected_steps` onto a small return structure only if you prefer, or stash into a model attribute (e.g., `self._last_act_stats`) to avoid breaking callers.
  2) `examples/training/grid/train_episodic.py`
     - Add CLI flags for ACT params and `--act-enabled` toggle.
     - During training:
       - Fetch `expected_steps` from the model (e.g., `model._last_act_stats`) if ACT is enabled; compute `ponder_loss = lambda * expected_steps.mean()` and add to task loss.
       - Log average `expected_steps`, halting coverage (fraction halted before max steps), and final step histogram for monitoring.

Notes on Using Existing Blocks

- Self-attention block reuse: Today the model either shares or not shares weights between blocks. For ACT, we can repeatedly apply:
  - If `encoder.extra_self_attention_block` is False: use `encoder.self_attn_1` every step (shared weights).
  - Else: use `encoder.self_attn_n` for steps ≥ 2 (already how the current loop is structured for multiple blocks). For ACT, consistently reuse one of them to form a recurrent “transition” function; prefer the shared one for stability.
- Cross-attention refresh: Keep it out of the loop for v1 (simpler and matches “seed then refine”). Consider periodic cross-attn refresh later if needed.

Compatibility and Defaults

- With `act_enabled=False`, behavior and outputs remain identical to current implementation.
- With `act_enabled=True`, the only change to the training step is the added ponder loss and the dynamic number of latent refinements.

Hyperparameters and Recommendations

- act_max_steps: start at the current `num_self_attention_blocks`. If blocks are “macro” (each with multiple layers), consider increasing `max_steps` and reducing per-block depth to create finer ACT control.
- act_threshold and act_epsilon: use threshold=0.99 and epsilon=1e-2 initially.
- act_ponder_cost (λ): start small (1e-4 to 1e-3). Increase if model overcomputes; decrease if undercomputes.
- act_temperature: 1.0 initially; adjust if halting saturates too early or too late.
- act_min_steps: set to 1–2 for stability during warmup.

Metrics and Logging

- Add to training logs (if ACT enabled):
  - mean_expected_steps, median_expected_steps, fraction_halted_before_max, final_step_histogram
  - task metrics unchanged (per-cell accuracy, per-grid accuracy; existing logs remain).

Evaluation and Inference

- Same ACT loop at inference; no ponder loss term. Optionally clamp `act_min_steps` higher during evaluation for more stable performance.
- For deterministic latency budgets, you can disable ACT at inference and run at a fixed step budget equal to a high-percentile of training `expected_steps`—this is optional.

Testing Strategy (Recommended)

- Unit tests:
  - Check halting convergence on synthetic data (monotonic p_t → halt in ≤ 3 steps).
  - Check aggregation weights sum to ~1.0 (within epsilon) per example.
  - Check shape invariants and device/dtype handling under AMP.
- Regression tests:
  - With ACT disabled, ensure identical outputs to pre-ACT model given the same seed and weights.

Potential Extensions (Later)

- Per-latent ACT: token-wise halting with masks and partial updates; more flexible but more complex.
- Curriculum or schedule for λ_ponder and act_temperature.
- Occasional cross-attention refresh during ACT steps (every N steps) for long inputs.
- Alternative halting heads (e.g., attention pooling over latents).

Implementation Checklist

1) Episodic model updates (`perceiver/model/grid/episodic.py`)
   - [ ] Add ACT params and halting head
   - [ ] Implement `_act_refine_latents`
   - [ ] Wire into `forward` with opt-in flag
2) Training script updates (`examples/training/grid/train_episodic.py`)
   - [ ] Add CLI flags for ACT
   - [ ] Add ponder loss to the objective when ACT is enabled
   - [ ] Log ACT metrics
3) Validation
   - [ ] Smoke test on a small subset (1–2 epochs) to validate halting behavior and loss decreases
   - [ ] Compare accuracy/latency tradeoff across λ_ponder values

Minimality and Containment

- All logic is localized to the episodic model file and the episodic training script. No changes to generic core modules (`perceiver/model/core/*`) are required for v1.


