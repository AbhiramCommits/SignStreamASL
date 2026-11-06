"""CPU-friendly pose-sequence transformer with temporal attention access."""

from __future__ import annotations

import torch
from torch import nn


class PoseSequenceTransformer(nn.Module):
    """Transformer encoder over per-frame 183-dimensional pose features.

    The model is intentionally small (<1.5M parameters) so the full experiment
    suite can run on a laptop CPU. Attention-masked mean and max pooling keep
    variable-length and padded inputs well defined.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 183,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 60,
        max_len: int = 48,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.max_len = int(max_len)
        self.num_classes = int(num_classes)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, d_model)
        # A light depthwise temporal convolution gives the transformer a local
        # motion prior; the self-attention layers still model long-range signing.
        self.temporal_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.pool_norm = nn.LayerNorm(d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def _prepare(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        if x.ndim != 3 or x.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected [B, T, {self.feature_dim}] features, got {tuple(x.shape)}"
            )
        if x.shape[1] > self.max_len:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_len={self.max_len}")
        if padding_mask is not None:
            if padding_mask.shape != x.shape[:2]:
                raise ValueError("padding_mask must have shape [B, T]")
            padding_mask = padding_mask.to(dtype=torch.bool)
        return x, padding_mask

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits with shape ``[B, num_classes]``."""
        x, padding_mask = self._prepare(x, padding_mask)
        hidden = self.input_projection(self.input_norm(x))
        conv = self.temporal_conv(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden + self.conv_norm(conv)
        hidden = hidden + self.positional_encoding[:, : hidden.shape[1], :]
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            valid = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        else:
            valid = ~padding_mask
        weights = valid.unsqueeze(-1).to(hidden.dtype)
        mean_pool = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        masked = hidden.masked_fill(~valid.unsqueeze(-1), torch.finfo(hidden.dtype).min)
        max_pool = masked.max(dim=1).values
        pooled = self.pool_norm(torch.cat([mean_pool, max_pool], dim=-1))
        return self.classifier(self.dropout(pooled))

    @torch.no_grad()
    def temporal_attention_weights(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the last encoder layer's averaged attention ``[B, T, T]``.

        This is an interpretability hook for the API and README; it does not
        change the classifier's forward path.
        """
        x, padding_mask = self._prepare(x, padding_mask)
        hidden = self.input_projection(self.input_norm(x))
        conv = self.temporal_conv(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = hidden + self.conv_norm(conv)
        hidden = hidden + self.positional_encoding[:, : hidden.shape[1], :]
        for layer in self.encoder.layers[:-1]:
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
        final = self.encoder.layers[-1]
        normed = final.norm1(hidden) if final.norm_first else hidden
        _, attention = final.self_attn(
            normed,
            normed,
            normed,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return attention

    @property
    def parameter_count(self) -> int:
        """Number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

