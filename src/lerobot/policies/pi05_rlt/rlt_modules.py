#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RLT (RL Token) modules for PI05.

Implements the components described in "RL Token: Bootstrapping Online RL with
Vision-Language-Action Models" (Physical Intelligence, 2026):

- ``RLTokenEncoder``: a lightweight transformer encoder that appends a learned
  ``<rl>`` token to the VLA's final-layer prefix embeddings and reads out the
  RL token ``z_rl`` at the special-token position (Eq. 1).
- ``RLTokenDecoder``: a transformer decoder that autoregressively reconstructs
  the (stop-gradient) prefix embeddings from ``z_rl`` with teacher forcing
  (Eq. 2). Only used during Stage 1 training.
- ``RLTActor``: an MLP producing a full action chunk (NOT a residual),
  conditioned on ``z_rl``, proprioceptive state, and the VLA's reference
  action chunk. Supports reference-action dropout: when the reference is
  dropped, it is zero-masked at the input and does not leak into the output.
- ``RLTCritic`` / ``TwinCritic``: chunk-level Q functions Q(x, a_{1:C}).
"""

import copy
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(num_layers):
        layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RLTokenEncoder(nn.Module):
    """Encoder transformer that summarizes VLA prefix embeddings into the RL token.

    Follows Eq. (1) of the RLT paper: a learned embedding ``e_rl`` is appended to
    the sequence of (projected) VLA final-layer token embeddings, the augmented
    sequence is processed with self-attention, and the output at the special-token
    position is the RL token ``z_rl``.
    """

    def __init__(
        self,
        embedding_dim: int,
        width: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.width = width
        self.in_proj = nn.Linear(embedding_dim, width) if embedding_dim != width else nn.Identity()
        self.rl_token_embedding = nn.Parameter(torch.randn(1, 1, width) / math.sqrt(width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=ff_mult * width,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, embeddings: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            embeddings: [B, M, embedding_dim] VLA final-layer token embeddings.
            pad_mask: [B, M] bool, True for valid tokens. None = all valid.

        Returns:
            z_rl: [B, width]
        """
        bsize, seq_len, _ = embeddings.shape
        x = self.in_proj(embeddings.to(self.rl_token_embedding.dtype))
        rl_tok = self.rl_token_embedding.expand(bsize, 1, self.width)
        x = torch.cat([x, rl_tok], dim=1)

        if pad_mask is None:
            key_padding_mask = None
        else:
            valid = torch.cat(
                [pad_mask.bool(), torch.ones(bsize, 1, dtype=torch.bool, device=pad_mask.device)], dim=1
            )
            # nn.Transformer* expects True = ignore
            key_padding_mask = ~valid

        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return out[:, -1]


class RLTokenDecoder(nn.Module):
    """Autoregressive decoder reconstructing prefix embeddings from the RL token.

    Follows Eq. (2): the target sequence is ``[z_rl, sg(z_1), ..., sg(z_{M-1})]``
    (teacher forcing with a causal mask), the memory is ``z_rl``, and a linear
    head predicts ``sg(z_i)`` at position i. The reconstruction loss forces the
    RL token to act as an information bottleneck.
    """

    def __init__(
        self,
        embedding_dim: int,
        width: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.width = width
        self.in_proj = nn.Linear(embedding_dim, width) if embedding_dim != width else nn.Identity()
        layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=ff_mult * width,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_head = nn.Linear(width, embedding_dim)

    def reconstruction_loss(self, z_rl: Tensor, embeddings: Tensor, pad_mask: Tensor | None = None) -> Tensor:
        """Masked MSE between reconstructed and stop-gradient target embeddings.

        Args:
            z_rl: [B, width]
            embeddings: [B, M, embedding_dim] targets (will be detached here).
            pad_mask: [B, M] bool, True for valid tokens.

        Returns:
            Scalar loss.
        """
        targets = embeddings.detach().to(z_rl.dtype)
        bsize, seq_len, _ = targets.shape

        # Teacher-forced input: [z_rl, z_1, ..., z_{M-1}]
        shifted = self.in_proj(targets[:, :-1])
        tgt = torch.cat([z_rl[:, None, :], shifted], dim=1)  # [B, M, width]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt.device)
        memory = z_rl[:, None, :]

        if pad_mask is None:
            tgt_key_padding_mask = None
        else:
            # Position i predicts token i; padded target positions are excluded from
            # attention as queries/keys. The first position (z_rl) is always valid.
            valid = torch.cat(
                [
                    torch.ones(bsize, 1, dtype=torch.bool, device=targets.device),
                    pad_mask.bool()[:, :-1],
                ],
                dim=1,
            )
            tgt_key_padding_mask = ~valid

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        recon = self.out_head(out)  # [B, M, embedding_dim]

        per_token = F.mse_loss(recon, targets, reduction="none").mean(dim=-1)  # [B, M]
        if pad_mask is None:
            return per_token.mean()
        mask = pad_mask.to(per_token.dtype)
        return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


class RLTActor(nn.Module):
    """Gaussian actor over full action chunks, conditioned on the VLA reference chunk.

    Follows Eq. (4): pi_theta(a_{1:C} | x, ref) = N(mu_theta(x, ref), sigma^2 I) with a
    small fixed sigma. The reference chunk is an INPUT (pass-through conditioning),
    not an additive residual base, so reference-action dropout (zeroing the ref
    input) yields a genuinely independent action-generation pathway.
    """

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.05,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.fixed_std = fixed_std
        in_dim = z_dim + proprio_dim + chunk_size * action_dim
        self.net = _build_mlp(in_dim, hidden_dim, chunk_size * action_dim, num_layers)

    def forward(self, z_rl: Tensor, proprio: Tensor, ref_chunk: Tensor, ref_mask: Tensor | None = None):
        """
        Args:
            z_rl: [B, z_dim]
            proprio: [B, proprio_dim]
            ref_chunk: [B, C, action_dim] VLA reference chunk.
            ref_mask: [B] bool, True = reference visible. False rows get a zeroed
                reference input (reference-action dropout).

        Returns:
            mu: [B, C, action_dim]
        """
        bsize = z_rl.shape[0]
        ref = ref_chunk.reshape(bsize, -1).to(z_rl.dtype)
        if ref_mask is not None:
            ref = ref * ref_mask.to(ref.dtype)[:, None]
        x = torch.cat([z_rl, proprio.to(z_rl.dtype), ref], dim=-1)
        mu = self.net(x)
        return mu.reshape(bsize, self.chunk_size, self.action_dim)

    def sample(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        ref_chunk: Tensor,
        ref_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> Tensor:
        mu = self(z_rl, proprio, ref_chunk, ref_mask)
        if deterministic or self.fixed_std <= 0:
            return mu
        return mu + self.fixed_std * torch.randn_like(mu)


class RLTCritic(nn.Module):
    """Chunk-level Q function Q(x, a_{1:C})."""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        in_dim = z_dim + proprio_dim + chunk_size * action_dim
        self.net = _build_mlp(in_dim, hidden_dim, 1, num_layers)

    def forward(self, z_rl: Tensor, proprio: Tensor, action_chunk: Tensor) -> Tensor:
        bsize = z_rl.shape[0]
        a = action_chunk.reshape(bsize, -1).to(z_rl.dtype)
        x = torch.cat([z_rl, proprio.to(z_rl.dtype), a], dim=-1)
        return self.net(x).squeeze(-1)


class TwinCritic(nn.Module):
    """Two independent Q functions; targets use the minimum (TD3 / clipped double-Q)."""

    def __init__(self, **critic_kwargs):
        super().__init__()
        self.q1 = RLTCritic(**critic_kwargs)
        self.q2 = RLTCritic(**critic_kwargs)

    def forward(self, z_rl: Tensor, proprio: Tensor, action_chunk: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(z_rl, proprio, action_chunk), self.q2(z_rl, proprio, action_chunk)

    def min_q(self, z_rl: Tensor, proprio: Tensor, action_chunk: Tensor) -> Tensor:
        q1, q2 = self(z_rl, proprio, action_chunk)
        return torch.minimum(q1, q2)


class RLTModules(nn.Module):
    """Container for all RLT components attached to a PI05RLT policy."""

    def __init__(
        self,
        embedding_dim: int,
        width: int,
        proprio_dim: int,
        action_dim: int,
        chunk_size: int,
        encoder_layers: int = 2,
        encoder_heads: int = 8,
        decoder_layers: int = 2,
        ff_mult: int = 4,
        actor_hidden_dim: int = 256,
        actor_num_layers: int = 2,
        critic_hidden_dim: int = 256,
        critic_num_layers: int = 2,
        fixed_std: float = 0.05,
    ):
        super().__init__()
        self.encoder = RLTokenEncoder(
            embedding_dim=embedding_dim,
            width=width,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            ff_mult=ff_mult,
        )
        self.decoder = RLTokenDecoder(
            embedding_dim=embedding_dim,
            width=width,
            num_layers=decoder_layers,
            num_heads=encoder_heads,
            ff_mult=ff_mult,
        )
        self.actor = RLTActor(
            z_dim=width,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dim=actor_hidden_dim,
            num_layers=actor_num_layers,
            fixed_std=fixed_std,
        )
        critic_kwargs = {
            "z_dim": width,
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
            "chunk_size": chunk_size,
            "hidden_dim": critic_hidden_dim,
            "num_layers": critic_num_layers,
        }
        self.critic = TwinCritic(**critic_kwargs)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)

    def sync_target(self, tau: float = 1.0) -> None:
        """Soft-update the target critic: theta' <- tau * theta + (1 - tau) * theta'."""
        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters(), strict=True):
                p_t.mul_(1.0 - tau).add_(p, alpha=tau)
            for b, b_t in zip(self.critic.buffers(), self.critic_target.buffers(), strict=True):
                b_t.copy_(b)
