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

"""PI05 + RLT (RL Token) policy.

Wraps a frozen ``pi05`` model with the RLT recipe from "RL Token: Bootstrapping
Online RL with Vision-Language-Action Models" (Physical Intelligence, 2026):

1. Stage 1 (``train_stage="rl_token"``): an encoder-decoder transformer is
   trained to reconstruct the frozen VLA's final-layer prefix embeddings
   through a bottleneck — the RL token ``z_rl``. Runs with `lerobot-train`.
2. Stage 2 (``train_stage="online"``): a small actor-critic operates on
   ``(z_rl, proprio)`` and refines the VLA's reference action chunks with
   TD3-style off-policy RL (see ``lerobot.scripts.rlt.train_pi05_rlt_online``).

The actor outputs FULL action chunks conditioned on the reference chunk
(pass-through conditioning + BC regularization + reference dropout), not a
residual. With ``rlt_enabled=false`` the policy is byte-for-byte pi05.
"""

import builtins
from pathlib import Path
from typing import Unpack

import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from ..pi05.modeling_pi05 import (
    ActionSelectKwargs,
    PI05Policy,
    get_gemma_config,
    make_att_2d_masks,
    pad_vector,
)
from ..pretrained import T
from .configuration_pi05_rlt import PI05RLTConfig
from .rlt_modules import RLTModules


class PI05RLTPolicy(PI05Policy):
    """PI05 policy augmented with RLT modules. The pi05 backbone is always frozen."""

    config_class = PI05RLTConfig
    name = "pi05_rlt"

    def __init__(self, config: PI05RLTConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        # The pi05 backbone is never trained in this policy (the paper's optional
        # joint VLA SFT is replaced by starting from a task-SFT'd pi05 checkpoint).
        self.model.requires_grad_(False)
        self.model.eval()

        embedding_dim = get_gemma_config(config.paligemma_variant).width
        original_action_dim = config.output_features[ACTION].shape[0]

        self.rlt = RLTModules(
            embedding_dim=embedding_dim,
            width=config.rlt_width,
            proprio_dim=config.max_state_dim,
            action_dim=original_action_dim,
            chunk_size=config.rl_chunk_size,
            encoder_layers=config.rlt_encoder_layers,
            encoder_heads=config.rlt_encoder_heads,
            decoder_layers=config.rlt_decoder_layers,
            ff_mult=config.rlt_ff_mult,
            actor_hidden_dim=config.actor_hidden_dim,
            actor_num_layers=config.actor_num_layers,
            critic_hidden_dim=config.critic_hidden_dim,
            critic_num_layers=config.critic_num_layers,
            fixed_std=config.rlt_fixed_std,
        )
        self.rlt.to(dtype=torch.float32)
        if config.device is not None:
            self.rlt.to(config.device)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the frozen backbone in eval mode regardless of the policy mode.
        self.model.eval()
        return self

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """Load with ``strict=False`` by default so that a plain pi05 checkpoint
        (which has no ``rlt.*`` keys) can seed the frozen backbone."""
        if config is not None and not isinstance(config, PI05RLTConfig):
            raise ValueError(
                "PI05RLTPolicy requires a PI05RLTConfig. Use --policy.type=pi05_rlt with "
                "--policy.pretrained_path pointing at a pi05 or pi05_rlt checkpoint."
            )
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            strict=strict,
            **kwargs,
        )

    def get_optim_params(self) -> dict:
        if self.config.train_stage == "rl_token":
            return list(self.rlt.encoder.parameters()) + list(self.rlt.decoder.parameters())
        return list(self.rlt.actor.parameters()) + list(self.rlt.critic.parameters())

    # ------------------------------------------------------------------
    # Prefix embedding capture
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_prefix(self, batch: dict[str, Tensor]):
        """Run the frozen VLA prefix (VLM) forward and return its final-layer
        token embeddings together with padding masks and prefill cache inputs.

        Returns:
            prefix_output: [B, M, embedding_dim] final-layer prefix embeddings.
            prefix_pad_masks: [B, M] bool validity mask.
            past_key_values: prefill KV cache (for subsequent denoising).
        """
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

        model = self.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_output, _), past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_output, prefix_pad_masks, past_key_values

    @torch.no_grad()
    def _denoise_from_prefill(
        self, prefix_pad_masks: Tensor, past_key_values, bsize: int, device, noise: Tensor | None = None
    ) -> Tensor:
        """Flow-matching denoising loop, identical to ``PI05Pytorch.sample_actions``
        after the prefix prefill (RTC is not supported with RLT)."""
        model = self.model
        num_steps = model.config.num_inference_steps

        if noise is None:
            actions_shape = (bsize, model.config.chunk_size, model.config.max_action_dim)
            noise = model.sample_noise(actions_shape, device)

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
        return x_t

    def extract_rl_token(self, batch: dict[str, Tensor]) -> Tensor:
        """Extract z_rl from the frozen VLA's final-layer prefix embeddings."""
        prefix_output, prefix_pad_masks, _ = self._compute_prefix(batch)
        return self.rlt.encoder(prefix_output.to(torch.float32), prefix_pad_masks)

    def _get_proprio(self, batch: dict[str, Tensor]) -> Tensor:
        """Normalized proprioceptive state padded to max_state_dim, float32."""
        state = batch[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim).to(torch.float32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_rlt_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, deterministic: bool | None = None
    ) -> dict[str, Tensor]:
        """Full RLT inference step used by both ``predict_action_chunk`` and the
        online RL trainer.

        Returns a dict with (all normalized action space, float32):
            actions:   [B, H, action_dim] chunk to execute (first C steps come
                       from the actor when ``rlt_actor_mode="actor"``).
            ref_chunk: [B, H, action_dim] the VLA's reference chunk.
            z_rl:      [B, rlt_width]
            proprio:   [B, max_state_dim]
        """
        self.eval()
        if self._rtc_enabled():
            raise NotImplementedError("RTC is not supported together with RLT.")

        tokens = batch[OBS_LANGUAGE_TOKENS]
        bsize, device = tokens.shape[0], tokens.device

        prefix_output, prefix_pad_masks, past_key_values = self._compute_prefix(batch)
        ref_padded = self._denoise_from_prefill(prefix_pad_masks, past_key_values, bsize, device, noise)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        ref = ref_padded[:, :, :original_action_dim].to(torch.float32)

        z_rl = self.rlt.encoder(prefix_output.to(torch.float32), prefix_pad_masks)
        proprio = self._get_proprio(batch)

        actions = ref.clone()
        if self.config.rlt_actor_mode == "actor":
            c = self.config.rl_chunk_size
            deterministic = (not self.config.rlt_actor_stochastic) if deterministic is None else deterministic
            actor_chunk = self.rlt.actor.sample(
                z_rl, proprio, ref[:, :c], ref_mask=None, deterministic=deterministic
            )
            actions[:, :c] = actor_chunk

        return {"actions": actions, "ref_chunk": ref, "z_rl": z_rl, "proprio": proprio}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        if not self.config.rlt_enabled:
            return super().predict_action_chunk(batch, **kwargs)
        out = self.predict_rlt_chunk(batch, noise=kwargs.get("noise"))
        return out["actions"]

    # ------------------------------------------------------------------
    # Stage 1 training (reconstruction) — runs under `lerobot-train`
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.train_stage != "rl_token":
            raise NotImplementedError(
                "PI05RLTPolicy.forward only implements Stage 1 (train_stage='rl_token'). "
                "Stage 2 uses lerobot.scripts.rlt.train_pi05_rlt_online."
            )

        prefix_output, prefix_pad_masks, _ = self._compute_prefix(batch)
        prefix_output = prefix_output.to(torch.float32)

        z_rl = self.rlt.encoder(prefix_output, prefix_pad_masks)
        loss = self.rlt.decoder.reconstruction_loss(z_rl, prefix_output, prefix_pad_masks)

        loss_dict = {"loss": loss.item(), "reconstruction_loss": loss.item()}
        return loss, loss_dict
