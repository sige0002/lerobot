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

"""Stage 2 online RL trainer for PI05-RLT (TD3-style, chunk-level TD).

Example:

```bash
uv run python -m lerobot.scripts.rlt.train_pi05_rlt_online \
    --policy.type=pi05_rlt \
    --policy.pretrained_path=outputs/rlt_stage1/checkpoints/last/pretrained_model \
    --env.type=libero --env.task=libero_object --env.task_ids='[0]' \
    --episodes=150 --warmup_episodes=15 \
    --output_dir=outputs/rlt_stage2
```

The loop follows the RLT paper: warmup rollouts execute the VLA reference
chunks to pre-fill the replay buffer; afterwards the stochastic actor collects
experience and TD3-style updates run at ``utd`` gradient steps per collected
transition (critic:actor = ``actor_delay``:1, twin-min targets, chunk-level
``gamma^C`` bootstrapping, stride-2 subsampled transitions, sparse binary
success reward).
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy
from lerobot.policies.pi05_rlt.online_rl import (
    ChunkTransitionAssembler,
    RLTOnlineUpdater,
    RLTReplayBuffer,
    compute_z_batched,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class TrainRLTOnlineConfig:
    policy: PreTrainedConfig | None = None
    env: EnvConfig | None = None
    output_dir: Path = Path("outputs/rlt_stage2")
    job_name: str = "pi05_rlt_online"
    seed: int = 42

    # Rollout
    episodes: int = 150
    warmup_episodes: int = 15
    stride: int = 2

    # Updates
    utd: int = 5
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_delay: int = 2
    buffer_capacity: int = 200_000
    min_buffer_size: int = 200

    # Eval / checkpointing
    eval_freq_episodes: int = 25
    eval_episodes: int = 10
    save_freq_episodes: int = 50

    def __post_init__(self):
        if self.policy is None or self.env is None:
            raise ValueError("Both --policy.* and --env.* must be provided")
        if not isinstance(self.policy, PI05RLTConfig):
            raise ValueError(f"--policy.type must be pi05_rlt, got {self.policy.type}")
        if not self.policy.pretrained_path:
            raise ValueError("--policy.pretrained_path is required (pi05 or pi05_rlt checkpoint)")
        if self.policy.rl_chunk_size % self.stride != 0:
            raise ValueError(
                f"stride ({self.stride}) must divide rl_chunk_size ({self.policy.rl_chunk_size})"
            )


def _extract_success(info: dict, num_envs: int = 1) -> bool:
    """Extract is_success from a (possibly vectorized) env info dict."""
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            is_success = final_info.get("is_success", [False] * num_envs)
            return bool(is_success[0]) if hasattr(is_success, "__len__") else bool(is_success)
        for item in final_info:
            if isinstance(item, dict) and "is_success" in item:
                return bool(item["is_success"])
        return False
    if "is_success" in info:
        is_success = info["is_success"]
        return bool(is_success[0]) if hasattr(is_success, "__len__") else bool(is_success)
    return False


class OnlineRLTTrainer:
    def __init__(self, cfg: TrainRLTOnlineConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.policy.device if cfg.policy.device else "cuda")

        # --- Environment (single task, single env) ---
        suites = make_env(cfg.env, n_envs=1, use_async_envs=False)
        flat = [(s, tid, e) for s, tasks in suites.items() for tid, e in tasks.items()]
        if len(flat) > 1:
            logging.warning(
                "make_env returned %d task envs; online RL uses only the first one. "
                "Restrict with --env.task_ids='[k]'.",
                len(flat),
            )
        self.suite, self.task_id, self.env = flat[0]
        self.max_steps = self.env.call("_max_episode_steps")[0]
        logging.info(f"Training on {self.suite} task {self.task_id} (max {self.max_steps} steps)")

        # --- Policy and processors ---
        self.policy: PI05RLTPolicy = make_policy(cfg.policy, env_cfg=cfg.env)
        self.policy.config.train_stage = "online"
        self.policy.config.rlt_enabled = True
        self.policy.eval()

        preprocessor_overrides = {"device_processor": {"device": str(self.device)}}
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

        # --- RL machinery ---
        pcfg: PI05RLTConfig = self.policy.config
        action_dim = pcfg.output_features[ACTION].shape[0]
        self.chunk_size = pcfg.rl_chunk_size
        self.buffer = RLTReplayBuffer(
            capacity=cfg.buffer_capacity,
            z_dim=pcfg.rlt_width,
            proprio_dim=pcfg.max_state_dim,
            action_dim=action_dim,
            chunk_size=self.chunk_size,
        )
        self.updater = RLTOnlineUpdater(
            self.policy,
            actor_lr=cfg.actor_lr,
            critic_lr=cfg.critic_lr,
            actor_delay=cfg.actor_delay,
        )

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = cfg.output_dir / "log.jsonl"
        self.total_env_steps = 0
        self.total_updates = 0

    # ------------------------------------------------------------------

    def _log(self, record: dict) -> None:
        record = {"time": time.time(), **record}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _build_batch(self, observation) -> dict:
        obs = preprocess_observation(observation)
        try:
            obs["task"] = list(self.env.call("task_description"))
        except (AttributeError, NotImplementedError):
            obs["task"] = [""]
        obs = self.env_pre(obs)
        obs = self.pre(obs)
        return obs

    def _env_step(self, normalized_action: torch.Tensor):
        """Unnormalize a [1, d] normalized action and step the environment."""
        action = self.post(normalized_action)
        transition = self.env_post({ACTION: action})
        action_numpy = transition[ACTION].to("cpu").numpy()
        return self.env.step(action_numpy)

    # ------------------------------------------------------------------

    def run_episode(
        self, mode: str, stochastic: bool, collect: bool = True, seed: int | None = None
    ) -> dict:
        """Run one episode; returns episode stats and (optionally) fills the buffer."""
        cfg = self.cfg
        self.policy.config.rlt_actor_mode = mode
        self.policy.reset()
        observation, _ = self.env.reset(seed=seed)

        assembler = ChunkTransitionAssembler(
            chunk_size=self.chunk_size, gamma=self.policy.config.rl_gamma, stride=cfg.stride
        )
        step = 0
        done = False
        success = False
        pending_z: list[tuple[int, dict]] = []  # (episode step, processed batch)
        step_records: list[dict] = []

        while not done and step < self.max_steps:
            batch = self._build_batch(observation)
            with torch.inference_mode():
                out = self.policy.predict_rlt_chunk(batch, deterministic=not stochastic)
            if collect:
                assembler.add_boundary(step, out["ref_chunk"][0])
                step_records.append(
                    {"step": step, "z": out["z_rl"][0].clone(), "proprio": out["proprio"][0].clone()}
                )

            for j in range(self.chunk_size):
                normalized_action = out["actions"][:, j]
                observation, _reward, terminated, truncated, info = self._env_step(normalized_action)
                step_success = _extract_success(info)
                success = success or step_success
                reward = 1.0 if step_success else 0.0
                done = bool(terminated[0] or truncated[0])
                if collect:
                    assembler.add_step(
                        action=normalized_action[0], reward=reward, done=done, z=None, proprio=None
                    )
                step += 1
                self.total_env_steps += 1
                if done:
                    break
                # Queue intermediate observations for batched z computation.
                if collect and j + 1 < self.chunk_size and step % cfg.stride == 0:
                    pending_z.append((step, self._build_batch(observation)))

            # Batched z_rl for intermediate (stride-subsampled) steps of this chunk.
            if collect and pending_z:
                z_batch, prop_batch = compute_z_batched(self.policy, [b for _, b in pending_z])
                for i, (s, _) in enumerate(pending_z):
                    step_records.append({"step": s, "z": z_batch[i].clone(), "proprio": prop_batch[i].clone()})
                pending_z.clear()

        transitions = []
        if collect:
            # Attach z/proprio to the assembler's step records.
            by_step = {r["step"]: r for r in step_records}
            for s, rec in enumerate(assembler.steps):
                if s in by_step:
                    rec.z = by_step[s]["z"].detach().float().cpu()
                    rec.proprio = by_step[s]["proprio"].detach().float().cpu()
            transitions = assembler.finish_episode(source=0 if mode == "reference" else 1)
            for tr in transitions:
                self.buffer.add(**tr)

        return {
            "steps": step,
            "success": success,
            "transitions": len(transitions),
        }

    def evaluate(self, n_episodes: int, seed_base: int = 10_000) -> dict:
        successes, lengths = [], []
        for i in range(n_episodes):
            stats = self.run_episode(mode="actor", stochastic=False, collect=False, seed=seed_base + i)
            successes.append(stats["success"])
            lengths.append(stats["steps"])
        return {
            "success_rate": float(sum(successes)) / max(len(successes), 1),
            "mean_episode_steps": float(sum(lengths)) / max(len(lengths), 1),
            "episodes": n_episodes,
        }

    def save_checkpoint(self, tag: str) -> Path:
        ckpt_dir = self.cfg.output_dir / "checkpoints" / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Persist an eval-ready configuration.
        self.policy.config.rlt_actor_mode = "actor"
        self.policy.config.rlt_actor_stochastic = False
        self.policy.save_pretrained(ckpt_dir)
        self.pre.save_pretrained(ckpt_dir)
        self.post.save_pretrained(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}")
        return ckpt_dir

    # ------------------------------------------------------------------

    def train(self) -> None:
        cfg = self.cfg
        start = time.time()

        for episode in range(cfg.episodes):
            is_warmup = episode < cfg.warmup_episodes
            mode = "reference" if is_warmup else "actor"
            stats = self.run_episode(mode=mode, stochastic=not is_warmup, collect=True, seed=cfg.seed + episode)

            # Gradient updates: utd steps per collected transition.
            update_metrics: dict[str, float] = {}
            n_updates = 0
            if not is_warmup and len(self.buffer) >= max(cfg.min_buffer_size, cfg.batch_size):
                n_updates = stats["transitions"] * cfg.utd
                for _ in range(n_updates):
                    batch = self.buffer.sample(cfg.batch_size, device=self.device)
                    update_metrics = self.updater.update(batch)
                    self.total_updates += 1

            record = {
                "type": "episode",
                "episode": episode,
                "mode": mode,
                "success": stats["success"],
                "steps": stats["steps"],
                "transitions": stats["transitions"],
                "buffer_size": len(self.buffer),
                "updates": n_updates,
                "total_updates": self.total_updates,
                "total_env_steps": self.total_env_steps,
                **{f"last_{k}": v for k, v in update_metrics.items()},
            }
            self._log(record)
            logging.info(
                f"ep {episode:4d} [{mode:9s}] success={stats['success']} steps={stats['steps']:4d} "
                f"buffer={len(self.buffer):6d} updates={self.total_updates:7d}"
            )

            if (episode + 1) % cfg.eval_freq_episodes == 0 and not is_warmup:
                eval_stats = self.evaluate(cfg.eval_episodes)
                self._log({"type": "eval", "episode": episode, **eval_stats})
                logging.info(
                    f"[eval @ep{episode}] success_rate={eval_stats['success_rate']:.2%} "
                    f"mean_steps={eval_stats['mean_episode_steps']:.1f}"
                )

            if (episode + 1) % cfg.save_freq_episodes == 0:
                self.save_checkpoint(f"ep{episode + 1:04d}")

        # Final eval + checkpoint
        eval_stats = self.evaluate(cfg.eval_episodes)
        self._log({"type": "eval", "episode": cfg.episodes, "final": True, **eval_stats})
        logging.info(
            f"[final eval] success_rate={eval_stats['success_rate']:.2%} "
            f"mean_steps={eval_stats['mean_episode_steps']:.1f}"
        )
        self.save_checkpoint("last")
        self._log({"type": "done", "wall_time_s": time.time() - start})
        self.env.close()


@parser.wrap()
def main(cfg: TrainRLTOnlineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    set_seed(cfg.seed)
    trainer = OnlineRLTTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
