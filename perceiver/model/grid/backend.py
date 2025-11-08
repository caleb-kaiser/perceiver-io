from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from perceiver.model.core import (
    InputAdapter,
    OutputAdapter,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
    QueryProvider,
    FourierPositionEncoding,
)


class GridInputAdapter(InputAdapter):
    def __init__(
        self,
        grid_shape: Tuple[int, int] = (30, 30),
        num_value_embeddings: int = 10,
        value_embedding_dim: int = 32,
        num_frequency_bands: int = 16,
    ):
        # Position encoding channels depend on grid shape and number of bands
        position_encoding = FourierPositionEncoding(input_shape=grid_shape, num_frequency_bands=num_frequency_bands)
        num_input_channels = value_embedding_dim + position_encoding.num_position_encoding_channels()
        super().__init__(num_input_channels)

        self.grid_shape = grid_shape
        self.value_embedding = nn.Embedding(num_value_embeddings, value_embedding_dim)
        self.position_encoding = position_encoding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, h, w) with integer values in [0..num_value_embeddings)
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (b, h, w); got {tuple(x.shape)}")

        b, h, w = x.shape
        if (h, w) != self.grid_shape:
            raise ValueError(f"Input grid shape {(h, w)} does not match adapter grid_shape {self.grid_shape}")

        x_emb = self.value_embedding(x)  # (b, h, w, E)
        x_emb = rearrange(x_emb, "b h w c -> b (h w) c")  # (b, hw, E)
        pos_enc = self.position_encoding(b)  # (b, hw, P)

        x_adapted = torch.cat([x_emb, pos_enc], dim=-1)  # (b, hw, E+P)
        return x_adapted


class GridClassificationOutputAdapter(OutputAdapter):
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        num_output_query_channels: int,
        num_classes: int = 10,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, o, f) where o = h*w and f = num_output_query_channels
        b, o, _ = x.shape
        h, w = self.grid_shape
        if o != h * w:
            raise ValueError(f"Output length {o} does not match grid size {h*w}")
        x = self.linear(x)  # (b, o, num_classes)
        return rearrange(x, "b (h w) c -> b h w c", h=h, w=w)


class GridQueryProvider(nn.Module, QueryProvider):
    def __init__(self, num_query_channels: int):
        super().__init__()
        self._num_query_channels = num_query_channels

    @property
    def num_query_channels(self) -> int:
        return self._num_query_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x to be the adapted input of shape (b, o, num_query_channels)
        if x is None:
            raise ValueError("GridQueryProvider requires x_adapted from the encoder; got None")
        if x.shape[-1] != self.num_query_channels:
            raise ValueError(
                f"Query channels mismatch: got {x.shape[-1]}, expected {self.num_query_channels}"
            )
        return x


class GridPerceiverIO(PerceiverIO):
    def __init__(
        self,
        grid_shape: Tuple[int, int] = (17, 17),
        num_classes: int = 10,
        # Input features
        num_value_embeddings: int = 10,
        value_embedding_dim: int = 32,
        num_frequency_bands: int = 16,
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
    ):
        input_adapter = GridInputAdapter(
            grid_shape=grid_shape,
            num_value_embeddings=num_value_embeddings,
            value_embedding_dim=value_embedding_dim,
            num_frequency_bands=num_frequency_bands,
        )

        # Make qk/v default explicit to input channels for stable dims
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
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

        output_query_provider = GridQueryProvider(num_query_channels=input_adapter.num_input_channels)
        output_adapter = GridClassificationOutputAdapter(
            grid_shape=grid_shape,
            num_output_query_channels=input_adapter.num_input_channels,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, h, w) LongTensor of token values
        x_latent, x_adapted = self.encoder(x, return_adapted_input=True)
        return self.decoder(x_latent, x_adapted=x_adapted)


