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

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_rlt")
@dataclass
class PI05RLTConfig(PI05Config):
    """Configuration for PI05 + RLT (RL Token) online RL refinement.

    Extends ``PI05Config`` with the RLT components from "RL Token: Bootstrapping
    Online RL with Vision-Language-Action Models" (Physical Intelligence, 2026).
    The base pi05 model is always kept frozen; only the RLT modules train.
    """

    # --- Runtime behavior ---
    # False = bypass all RLT processing; the policy behaves exactly like pi05.
    rlt_enabled: bool = True
    # "reference": execute the VLA reference chunk as-is (warmup / pass-through).
    # "actor": execute the RL actor's chunk (Stage 2 and after).
    rlt_actor_mode: str = "reference"
    # Sample actor actions (exploration noise) instead of using the mean.
    # The online RL trainer enables this during rollouts; keep False for eval.
    rlt_actor_stochastic: bool = False

    # --- RL action chunking ---
    # C in the paper: the RL policy operates on the first C steps of the VLA's
    # H-step chunk (C < H for reactivity). Paper uses C=10 with H=50.
    rl_chunk_size: int = 10

    # --- RL token encoder/decoder (Stage 1) ---
    # Width of the encoder/decoder transformers and of z_rl. Paper: 2048.
    rlt_width: int = 2048
    rlt_encoder_layers: int = 2
    rlt_encoder_heads: int = 8
    rlt_decoder_layers: int = 2
    rlt_ff_mult: int = 4

    # --- Actor / critic networks (Stage 2) ---
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    # Small fixed std of the Gaussian actor (paper App. B).
    rlt_fixed_std: float = 0.05
    # Reference-action dropout probability during training (paper: 0.5).
    ref_dropout_prob: float = 0.5
    # beta: BC regularization weight in L_pi = -Q + beta * ||a - ref||^2 (Eq. 5).
    bc_beta: float = 1.0
    # Chunk-level TD parameters.
    rl_gamma: float = 0.99
    rl_tau: float = 0.005

    # --- Training stage selection ---
    # "rl_token": policy.forward() returns the Stage 1 reconstruction loss and
    #   get_optim_params() returns encoder+decoder parameters (usable with
    #   the standard `lerobot-train` script).
    # "online": RLT actor/critic training (handled by the online RL trainer).
    train_stage: str = "rl_token"

    def __post_init__(self):
        super().__post_init__()

        if self.rlt_actor_mode not in ("reference", "actor"):
            raise ValueError(f"Invalid rlt_actor_mode: {self.rlt_actor_mode}")
        if self.train_stage not in ("rl_token", "online"):
            raise ValueError(f"Invalid train_stage: {self.train_stage}")
        if not (0 < self.rl_chunk_size <= self.chunk_size):
            raise ValueError(
                f"rl_chunk_size ({self.rl_chunk_size}) must be in (0, chunk_size={self.chunk_size}]"
            )
        if not (0.0 <= self.ref_dropout_prob <= 1.0):
            raise ValueError(f"ref_dropout_prob must be in [0, 1], got {self.ref_dropout_prob}")

        # When RLT is active the policy replans every C steps: only the first C
        # actions of each chunk are produced by the RL actor.
        if self.rlt_enabled:
            self.n_action_steps = min(self.n_action_steps, self.rl_chunk_size)

    @classmethod
    def from_pi05_config(cls, base: PI05Config, **overrides) -> "PI05RLTConfig":
        """Build a PI05RLTConfig carrying over all fields of an existing pi05 config
        (e.g. one loaded from a pretrained pi05 checkpoint)."""
        from dataclasses import fields as dc_fields

        rlt_field_names = {f.name for f in dc_fields(cls) if f.init}
        base_kwargs = {
            f.name: getattr(base, f.name)
            for f in dc_fields(type(base))
            if f.init and f.name in rlt_field_names
        }
        base_kwargs.update(overrides)
        return cls(**base_kwargs)
