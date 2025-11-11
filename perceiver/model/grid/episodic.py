from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from perceiver.model.core import (
    InputAdapter,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
    QueryProvider,
    FourierPositionEncoding,
)
from .backend import GridClassificationOutputAdapter, GridQueryProvider, GridInputAdapter


class EpisodicGridInputAdapter(InputAdapter):
    def __init__(
        self,
        grid_shape: Tuple[int, int] = (17, 17),
        num_value_embeddings: int = 10,
        value_embedding_dim: int = 32,
        num_frequency_bands: int = 16,
        role_embedding_dim: int = 16,
        pair_embedding_dim: int = 16,
        max_support: int = 5,
    ):
        # Base per-grid encoding = value embed + 2D Fourier pos enc
        position_encoding = FourierPositionEncoding(input_shape=grid_shape, num_frequency_bands=num_frequency_bands)
        base_channels = value_embedding_dim + position_encoding.num_position_encoding_channels()
        super().__init__(num_input_channels=base_channels + role_embedding_dim + pair_embedding_dim)

        self.grid_shape = grid_shape
        self.value_embedding = nn.Embedding(num_value_embeddings, value_embedding_dim)
        self.position_encoding = position_encoding

        # Roles: 0 = support_input, 1 = support_output, 2 = query_input
        self.role_embedding = nn.Embedding(3, role_embedding_dim)
        self.max_support = max_support
        self.pair_embedding = nn.Embedding(max_support, pair_embedding_dim)

    def _encode_grid(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, h, w) long
        b, h, w = x.shape
        if (h, w) != self.grid_shape:
            raise ValueError(f"Input grid shape {(h, w)} does not match adapter grid_shape {self.grid_shape}")
        x_emb = self.value_embedding(x)  # (b, h, w, E)
        x_emb = rearrange(x_emb, "b h w c -> b (h w) c")
        pos_enc = self.position_encoding(b)  # (b, hw, P)
        return torch.cat([x_emb, pos_enc], dim=-1)  # (b, hw, base_channels)

    def forward(
        self,
        support_in: torch.Tensor,
        support_out: torch.Tensor,
        query_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param support_in: (b, k, h, w) LongTensor
        :param support_out: (b, k, h, w) LongTensor
        :param query_in: (b, h, w) LongTensor
        :return: concatenated episodic sequence (b, (2k+1)*h*w, C)
        """
        if support_in.dim() != 4 or support_out.dim() != 4:
            raise ValueError("support_in and support_out must be shaped (b, k, h, w)")
        if query_in.dim() != 3:
            raise ValueError("query_in must be shaped (b, h, w)")

        b, k, h, w = support_in.shape
        if (h, w) != self.grid_shape:
            raise ValueError(f"Support grid shape {(h, w)} does not match adapter grid_shape {self.grid_shape}")
        if support_out.shape != support_in.shape:
            raise ValueError("support_in and support_out must have the same shape")
        if k > self.max_support:
            raise ValueError(f"Number of support pairs k={k} exceeds max_support={self.max_support}")

        # Encode supports per pair and label with role/pair embeddings
        segments = []
        for i in range(k):
            x_in = self._encode_grid(support_in[:, i])  # (b, hw, base)
            x_out = self._encode_grid(support_out[:, i])  # (b, hw, base)

            role_in = self.role_embedding.weight[0]  # (r,)
            role_out = self.role_embedding.weight[1]  # (r,)
            pair_i = self.pair_embedding.weight[i]  # (p,)

            role_in_b = repeat(role_in, "r -> b n r", b=b, n=x_in.shape[1])
            role_out_b = repeat(role_out, "r -> b n r", b=b, n=x_out.shape[1])
            pair_b_in = repeat(pair_i, "p -> b n p", b=b, n=x_in.shape[1])
            pair_b_out = repeat(pair_i, "p -> b n p", b=b, n=x_out.shape[1])

            segments.append(torch.cat([x_in, role_in_b, pair_b_in], dim=-1))
            segments.append(torch.cat([x_out, role_out_b, pair_b_out], dim=-1))

        # Encode query input and label with role embedding; no pair embedding
        x_q = self._encode_grid(query_in)
        role_q = self.role_embedding.weight[2]
        role_q_b = repeat(role_q, "r -> b n r", b=b, n=x_q.shape[1])
        # zero pair embedding for query (same dim), detach to keep as constant
        zero_pair = torch.zeros(self.pair_embedding.embedding_dim, device=x_q.device, dtype=x_q.dtype)
        zero_pair_b = repeat(zero_pair, "p -> b n p", b=b, n=x_q.shape[1])
        x_q = torch.cat([x_q, role_q_b, zero_pair_b], dim=-1)

        segments.append(x_q)
        return torch.cat(segments, dim=1)  # (b, (2k+1)*hw, C)


class EpisodicGridPerceiverIO(PerceiverIO):
    def __init__(
        self,
        grid_shape: Tuple[int, int] = (17, 17),
        num_classes: int = 10,
        # Base features
        num_value_embeddings: int = 10,
        value_embedding_dim: int = 32,
        num_frequency_bands: int = 16,
        # Episodic features
        role_embedding_dim: int = 16,
        pair_embedding_dim: int = 16,
        max_support: int = 5,
        # Latents
        num_latents: int = 128,
        num_latent_channels: int = 256,
        # Attention
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_layers_per_block: int = 4,
        num_self_attention_blocks: int = 1,
        dropout: float = 0.1,
        activation_checkpointing: bool = False,
        activation_offloading: bool = False,
        # ACT (Adaptive Computation Time)
        act_enabled: bool = False,
        act_max_steps: int = None,
        act_threshold: float = 0.99,
        act_epsilon: float = 1e-2,
        act_min_steps: int = 1,
        act_temperature: float = 1.0,
    ):
        episodic_input_adapter = EpisodicGridInputAdapter(
            grid_shape=grid_shape,
            num_value_embeddings=num_value_embeddings,
            value_embedding_dim=value_embedding_dim,
            num_frequency_bands=num_frequency_bands,
            role_embedding_dim=role_embedding_dim,
            pair_embedding_dim=pair_embedding_dim,
            max_support=max_support,
        )

        # Separate adapter for decoder queries from the query input grid
        query_input_adapter = GridInputAdapter(
            grid_shape=grid_shape,
            num_value_embeddings=num_value_embeddings,
            value_embedding_dim=value_embedding_dim,
            num_frequency_bands=num_frequency_bands,
        )

        encoder = PerceiverEncoder(
            input_adapter=episodic_input_adapter,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            num_cross_attention_heads=num_cross_attention_heads,
            num_cross_attention_qk_channels=None,
            num_cross_attention_v_channels=None,
            num_self_attention_heads=num_self_attention_heads,
            num_self_attention_qk_channels=None,
            num_self_attention_v_channels=None,
            num_self_attention_layers_per_block=num_self_attention_layers_per_block,
            num_self_attention_blocks=num_self_attention_blocks,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
            activation_offloading=activation_offloading,
        )

        output_query_provider = GridQueryProvider(num_query_channels=query_input_adapter.num_input_channels)
        output_adapter = GridClassificationOutputAdapter(
            grid_shape=grid_shape,
            num_output_query_channels=query_input_adapter.num_input_channels,
            num_classes=num_classes,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            output_query_provider=output_query_provider,
            num_latent_channels=num_latent_channels,
            num_cross_attention_heads=num_cross_attention_heads,
            num_cross_attention_qk_channels=num_latent_channels,
            num_cross_attention_v_channels=num_latent_channels,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
            activation_offloading=activation_offloading,
        )

        super().__init__(encoder, decoder)
        self._query_input_adapter = query_input_adapter
        # ACT config
        self.act_enabled = act_enabled
        self.act_max_steps = act_max_steps if act_max_steps is not None else max(1, num_self_attention_blocks)
        self.act_threshold = act_threshold
        self.act_epsilon = act_epsilon
        self.act_min_steps = act_min_steps
        self.act_temperature = act_temperature
        # Halting head
        self._halt_norm = nn.LayerNorm(num_latent_channels)
        self._halt_proj = nn.Linear(num_latent_channels, 1)
        # Stats holder (read by training loop if needed)
        self._last_act_stats = {}

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param support_x: (b, k, h, w) LongTensor
        :param support_y: (b, k, h, w) LongTensor
        :param query_x: (b, h, w) LongTensor
        :return: logits (b, h, w, num_classes)
        """
        # Encode episodic input into latents
        x_latent = self.encoder.input_adapter(
            support_in=support_x, support_out=support_y, query_in=query_x
        )
        # run encoder with already-adapted episodic sequence by calling encoder modules explicitly
        # to avoid double-adaptation, we pass through the encoder stack manually
        # Note: PerceiverEncoder expects raw x; replicate minimal forward with pre-adapted x
        # by using its latent provider and attention blocks directly.
        # Fetch underlying encoder
        encoder = self.encoder
        x_latents = encoder.latent_provider()
        x_latents = encoder.cross_attn_1(x_latents, x_latent).last_hidden_state
        if self.act_enabled:
            x_latents, expected_steps = self._act_refine_latents(
                x_latents,
                transition_block=encoder.self_attn_1,
                max_steps=self.act_max_steps,
                threshold=self.act_threshold,
                epsilon=self.act_epsilon,
                min_steps=self.act_min_steps,
                temperature=self.act_temperature,
            )
            # Save for training loop (avoid changing public API)
            self._last_act_stats = {"expected_steps": expected_steps.detach()}
        else:
            # Fixed-depth refinement as before
            x_latents = encoder.self_attn_1(x_latents).last_hidden_state
            cross_attn_n = encoder.cross_attn_n if encoder.extra_cross_attention_layer else encoder.cross_attn_1
            self_attn_n = encoder.self_attn_n if encoder.extra_self_attention_block else encoder.self_attn_1
            for i in range(1, encoder.num_self_attention_blocks):
                if i < encoder.num_cross_attention_layers:
                    x_latents = cross_attn_n(x_latents, x_latent).last_hidden_state
                x_latents = self_attn_n(x_latents).last_hidden_state

        # Build query-only adapted tokens and decode
        x_adapted_query = self._query_input_adapter(query_x)  # (b, hw, Cq)
        return self.decoder(x_latents, x_adapted=x_adapted_query)

    def _act_refine_latents(
        self,
        x_latents: torch.Tensor,
        transition_block: nn.Module,
        max_steps: int,
        threshold: float,
        epsilon: float,
        min_steps: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive refinement of latents using a recurrent application of a self-attention block.
        Returns:
          - aggregated latents (B, N, D)
          - expected_steps per example (B,)
        """
        b, n, d = x_latents.shape
        device = x_latents.device

        # Maintain accumulators in float32 for numerical stability
        agg = torch.zeros((b, n, d), device=device, dtype=torch.float32)
        halting_acc = torch.zeros((b,), device=device, dtype=torch.float32)
        exp_steps = torch.zeros((b,), device=device, dtype=torch.float32)

        latents = x_latents

        for t in range(1, max_steps + 1):
            # Transition
            latents = transition_block(latents).last_hidden_state
            # Halting probability per example
            pooled = latents.mean(dim=1)  # (B, D)
            halted_logits = self._halt_proj(self._halt_norm(pooled)).squeeze(-1)  # (B,)
            if temperature != 1.0:
                halted_logits = halted_logits / float(temperature)
            p_t = torch.sigmoid(halted_logits).to(torch.float32)  # (B,)

            # Active mask
            m_active = (halting_acc < threshold).to(torch.float32)  # (B,)

            remaining = (1.0 - halting_acc)  # (B,)
            # How much mass to take this step
            new_mass = torch.minimum(p_t, remaining)
            # If we cross threshold this step, take the remainder
            weight_t = torch.where(halting_acc + p_t >= threshold, remaining, new_mass)  # (B,)
            weight_t = weight_t * m_active

            # Accumulate weighted latents
            agg = agg + weight_t.view(b, 1, 1) * latents.to(torch.float32)
            halting_acc = halting_acc + weight_t
            exp_steps = exp_steps + weight_t

            # Early exit if everyone halted and min_steps satisfied
            if t >= min_steps and torch.all(halting_acc >= (threshold - 1e-6)):
                break

        # Normalize in case sum(weights) < 1 due to epsilon; avoid division by zero
        normalizer = torch.clamp(halting_acc, min=epsilon).view(b, 1, 1)
        agg = agg / normalizer

        # Cast back to original dtype
        agg = agg.to(x_latents.dtype)
        return agg, exp_steps


