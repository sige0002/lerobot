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

"""Online RL machinery for PI05-RLT (Stage 2).

Implements, following the RLT paper:

- ``RLTReplayBuffer``: chunk-level transitions ``<x, a_{1:C}, ref, r, x', ref'>``
  where ``x = (z_rl, proprio)``. Compact tensor ring buffer (no images — the
  frozen VLA already digested them into ``z_rl``).
- ``ChunkTransitionAssembler``: turns per-step rollout records into chunk-level
  transitions, including stride subsampling (paper: stride 2 → transitions at
  ``<x_0, a_0:C>, <x_2, a_2:C+2>, ...``), chunk-level discounted rewards
  ``R = sum_{t'=1..C} gamma^{t'-1} r_{t'}``, and terminal handling.
- ``RLTOnlineUpdater``: TD3-style updates. Critic: chunk-level C-step TD with
  twin-min targets (Eq. 3). Actor: ``L_pi = -Q1(x, a) + beta * ||a - ref||^2``
  with reference-action dropout on the actor input (Eq. 5).
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from .modeling_pi05_rlt import PI05RLTPolicy


class RLTReplayBuffer:
    """Ring buffer of chunk-level transitions, stored on CPU as float32."""

    def __init__(self, capacity: int, z_dim: int, proprio_dim: int, action_dim: int, chunk_size: int):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        c, d = chunk_size, action_dim
        self.z = torch.zeros(capacity, z_dim)
        self.proprio = torch.zeros(capacity, proprio_dim)
        self.action = torch.zeros(capacity, c, d)
        self.ref = torch.zeros(capacity, c, d)
        self.reward = torch.zeros(capacity)  # discounted C-step reward
        self.done = torch.zeros(capacity)  # 1.0 = no bootstrap
        self.next_z = torch.zeros(capacity, z_dim)
        self.next_proprio = torch.zeros(capacity, proprio_dim)
        self.next_ref = torch.zeros(capacity, c, d)
        self.source = torch.zeros(capacity, dtype=torch.int8)  # 0=warmup/reference, 1=rl

    def add(
        self,
        z: Tensor,
        proprio: Tensor,
        action: Tensor,
        ref: Tensor,
        reward: float,
        done: bool,
        next_z: Tensor,
        next_proprio: Tensor,
        next_ref: Tensor,
        source: int = 1,
    ) -> None:
        i = self.pos
        self.z[i] = z.detach().float().cpu()
        self.proprio[i] = proprio.detach().float().cpu()
        self.action[i] = action.detach().float().cpu()
        self.ref[i] = ref.detach().float().cpu()
        self.reward[i] = float(reward)
        self.done[i] = float(done)
        self.next_z[i] = next_z.detach().float().cpu()
        self.next_proprio[i] = next_proprio.detach().float().cpu()
        self.next_ref[i] = next_ref.detach().float().cpu()
        self.source[i] = source
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> dict[str, Tensor]:
        idx = torch.randint(0, self.size, (batch_size,))
        return {
            "z": self.z[idx].to(device),
            "proprio": self.proprio[idx].to(device),
            "action": self.action[idx].to(device),
            "ref": self.ref[idx].to(device),
            "reward": self.reward[idx].to(device),
            "done": self.done[idx].to(device),
            "next_z": self.next_z[idx].to(device),
            "next_proprio": self.next_proprio[idx].to(device),
            "next_ref": self.next_ref[idx].to(device),
        }

    def __len__(self) -> int:
        return self.size


@dataclass
class _StepRecord:
    action: Tensor  # [d] executed action, normalized space
    reward: float
    done: bool
    z: Tensor | None = None  # [z_dim], only at stride-aligned steps
    proprio: Tensor | None = None  # [proprio_dim]


@dataclass
class ChunkTransitionAssembler:
    """Accumulates one episode of per-step records and emits chunk transitions.

    Usage per episode:
        1. ``add_boundary(step, ref_h)`` whenever the VLA samples a new H-step
           reference chunk (every C env steps).
        2. ``add_step(...)`` for every executed env step. ``z``/``proprio`` must
           be provided for steps where ``step % stride == 0``.
        3. ``finish_episode()`` → list of transition dicts for the replay buffer.

    The reference window for a state at step ``t`` is the slice
    ``ref_h[t - b : t - b + C]`` of the H-step chunk sampled at the latest
    boundary ``b <= t`` (H >= 2C guarantees the slice exists).
    """

    chunk_size: int
    gamma: float
    stride: int = 2
    steps: list[_StepRecord] = field(default_factory=list)
    boundaries: list[tuple[int, Tensor]] = field(default_factory=list)  # (step, ref_h [H, d])

    def add_boundary(self, step: int, ref_h: Tensor) -> None:
        self.boundaries.append((step, ref_h.detach().float().cpu()))

    def add_step(
        self,
        action: Tensor,
        reward: float,
        done: bool,
        z: Tensor | None = None,
        proprio: Tensor | None = None,
    ) -> None:
        self.steps.append(
            _StepRecord(
                action=action.detach().float().cpu(),
                reward=float(reward),
                done=bool(done),
                z=None if z is None else z.detach().float().cpu(),
                proprio=None if proprio is None else proprio.detach().float().cpu(),
            )
        )

    def _ref_window(self, t: int) -> Tensor:
        b, ref_h = next((bb, rr) for bb, rr in reversed(self.boundaries) if bb <= t)
        offset = t - b
        c = self.chunk_size
        window = ref_h[offset : offset + c]
        if window.shape[0] < c:  # H exhausted near episode end; repeat last ref action
            pad = window[-1:].expand(c - window.shape[0], -1)
            window = torch.cat([window, pad], dim=0)
        return window

    def finish_episode(self, source: int = 1) -> list[dict]:
        transitions = []
        num_steps = len(self.steps)
        c = self.chunk_size
        for t in range(0, num_steps, self.stride):
            rec = self.steps[t]
            if rec.z is None:
                continue
            end = min(t + c, num_steps)
            window = self.steps[t:end]

            # Discounted chunk reward over available steps.
            reward = 0.0
            for k, w in enumerate(window):
                reward += (self.gamma**k) * w.reward
            done_within = any(w.done for w in window)

            # Executed action chunk, padded by repeating the last action if the
            # episode ended inside the window.
            actions = torch.stack([w.action for w in window])
            if actions.shape[0] < c:
                pad = actions[-1:].expand(c - actions.shape[0], -1)
                actions = torch.cat([actions, pad], dim=0)

            if done_within:
                next_z = torch.zeros_like(rec.z)
                next_proprio = torch.zeros_like(rec.proprio)
                next_ref = torch.zeros(c, actions.shape[-1])
            else:
                if t + c >= num_steps or self.steps[t + c].z is None:
                    # Next state not observed with z (episode truncated before
                    # t+C was recorded) — skip this window.
                    continue
                nxt = self.steps[t + c]
                next_z = nxt.z
                next_proprio = nxt.proprio
                next_ref = self._ref_window(t + c)

            transitions.append(
                {
                    "z": rec.z,
                    "proprio": rec.proprio,
                    "action": actions,
                    "ref": self._ref_window(t),
                    "reward": reward,
                    "done": done_within,
                    "next_z": next_z,
                    "next_proprio": next_proprio,
                    "next_ref": next_ref,
                    "source": source,
                }
            )
        self.steps.clear()
        self.boundaries.clear()
        return transitions


class RLTOnlineUpdater:
    """TD3-style updater for the RLT actor and twin critic."""

    def __init__(
        self,
        policy: PI05RLTPolicy,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_delay: int = 2,
    ):
        self.policy = policy
        self.rlt = policy.rlt
        cfg = policy.config
        self.gamma = cfg.rl_gamma
        self.tau = cfg.rl_tau
        self.beta = cfg.bc_beta
        self.ref_dropout_prob = cfg.ref_dropout_prob
        self.chunk_size = cfg.rl_chunk_size
        self.actor_delay = actor_delay
        self.critic_updates = 0

        self.critic_optimizer = torch.optim.Adam(self.rlt.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.rlt.actor.parameters(), lr=actor_lr)

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        metrics = self._update_critic(batch)
        self.critic_updates += 1
        if self.critic_updates % self.actor_delay == 0:
            metrics.update(self._update_actor(batch))
            self.rlt.sync_target(self.tau)
        return metrics

    def _update_critic(self, batch: dict[str, Tensor]) -> dict[str, float]:
        rlt = self.rlt
        with torch.no_grad():
            next_a = rlt.actor.sample(
                batch["next_z"], batch["next_proprio"], batch["next_ref"], deterministic=False
            )
            target_q = rlt.critic_target.min_q(batch["next_z"], batch["next_proprio"], next_a)
            bootstrap = (1.0 - batch["done"]) * (self.gamma**self.chunk_size)
            q_hat = batch["reward"] + bootstrap * target_q

        q1, q2 = rlt.critic(batch["z"], batch["proprio"], batch["action"])
        critic_loss = torch.nn.functional.mse_loss(q1, q_hat) + torch.nn.functional.mse_loss(q2, q_hat)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "target_q_mean": q_hat.mean().item(),
        }

    def _update_actor(self, batch: dict[str, Tensor]) -> dict[str, float]:
        rlt = self.rlt
        bsize = batch["z"].shape[0]
        # Reference-action dropout: zero the reference INPUT for a random subset,
        # while the BC regularization target stays the true reference (Eq. 5).
        ref_mask = torch.rand(bsize, device=batch["z"].device) >= self.ref_dropout_prob

        mu = rlt.actor(batch["z"], batch["proprio"], batch["ref"], ref_mask=ref_mask)
        a = mu + rlt.actor.fixed_std * torch.randn_like(mu)

        q1 = rlt.critic.q1(batch["z"], batch["proprio"], a)
        bc = ((a - batch["ref"]) ** 2).sum(dim=(1, 2)).mean() / (a.shape[1] * a.shape[2])
        actor_loss = -q1.mean() + self.beta * bc

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "actor_q1_mean": q1.mean().item(),
            "bc_mse": bc.item(),
            "ref_deviation": (mu - batch["ref"]).abs().mean().item(),
        }


def compute_z_batched(
    policy: PI05RLTPolicy, processed_batches: list[dict[str, Tensor]]
) -> tuple[Tensor, Tensor]:
    """Compute z_rl and proprio for several single-sample processed batches in one
    frozen-VLA prefix forward (used for stride-subsampled intermediate steps)."""
    keys = processed_batches[0].keys()
    merged: dict[str, Tensor] = {}
    for key in keys:
        vals = [b[key] for b in processed_batches]
        if isinstance(vals[0], torch.Tensor):
            merged[key] = torch.cat(vals, dim=0)
        else:
            merged[key] = np.concatenate([np.asarray(v) for v in vals]).tolist()
    z = policy.extract_rl_token(merged)
    proprio = policy._get_proprio(merged)
    return z, proprio
