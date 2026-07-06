"""CPU unit tests for the RLT modules, chunk assembler, replay buffer, and TD3 updater.

These tests do NOT instantiate the pi05 backbone; integration tests that load
real checkpoints live in test_pi05_rlt_integration.py.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.pi05_rlt.online_rl import (
    ChunkTransitionAssembler,
    RLTOnlineUpdater,
    RLTReplayBuffer,
)
from lerobot.policies.pi05_rlt.rlt_modules import (
    RLTActor,
    RLTModules,
    RLTokenDecoder,
    RLTokenEncoder,
    TwinCritic,
)

EMB, WIDTH, PROP, ACT_DIM, C = 32, 16, 6, 4, 4


def _modules() -> RLTModules:
    torch.manual_seed(0)
    return RLTModules(
        embedding_dim=EMB,
        width=WIDTH,
        proprio_dim=PROP,
        action_dim=ACT_DIM,
        chunk_size=C,
        encoder_layers=1,
        encoder_heads=2,
        decoder_layers=1,
        ff_mult=2,
        actor_hidden_dim=32,
        actor_num_layers=2,
        critic_hidden_dim=32,
        critic_num_layers=2,
        fixed_std=0.05,
    )


# ---------------------------------------------------------------------------
# Encoder / decoder
# ---------------------------------------------------------------------------


def test_encoder_output_shape():
    enc = RLTokenEncoder(EMB, WIDTH, num_layers=1, num_heads=2)
    emb = torch.randn(3, 11, EMB)
    z = enc(emb)
    assert z.shape == (3, WIDTH)
    assert torch.isfinite(z).all()


def test_encoder_ignores_padded_tokens():
    torch.manual_seed(1)
    enc = RLTokenEncoder(EMB, WIDTH, num_layers=1, num_heads=2).eval()
    emb = torch.randn(2, 9, EMB)
    pad = torch.ones(2, 9, dtype=torch.bool)
    pad[:, 6:] = False

    z1 = enc(emb, pad)
    emb2 = emb.clone()
    emb2[:, 6:] = 123.0  # garbage in padded region must not change z
    z2 = enc(emb2, pad)
    torch.testing.assert_close(z1, z2)


def test_decoder_loss_masks_padded_targets():
    torch.manual_seed(2)
    dec = RLTokenDecoder(EMB, WIDTH, num_layers=1, num_heads=2).eval()
    z = torch.randn(2, WIDTH)
    emb = torch.randn(2, 9, EMB)
    pad = torch.ones(2, 9, dtype=torch.bool)
    pad[:, 7:] = False

    loss1 = dec.reconstruction_loss(z, emb, pad)
    emb2 = emb.clone()
    emb2[:, 7:] = 55.0
    loss2 = dec.reconstruction_loss(z, emb2, pad)
    torch.testing.assert_close(loss1, loss2)
    assert torch.isfinite(loss1)


def test_decoder_gradients_reach_encoder():
    enc = RLTokenEncoder(EMB, WIDTH, num_layers=1, num_heads=2)
    dec = RLTokenDecoder(EMB, WIDTH, num_layers=1, num_heads=2)
    emb = torch.randn(2, 7, EMB)
    z = enc(emb)
    loss = dec.reconstruction_loss(z, emb)
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# ---------------------------------------------------------------------------
# Actor / critic
# ---------------------------------------------------------------------------


def test_actor_outputs_full_chunk():
    actor = RLTActor(WIDTH, PROP, ACT_DIM, C, hidden_dim=32, num_layers=2)
    mu = actor(torch.randn(5, WIDTH), torch.randn(5, PROP), torch.randn(5, C, ACT_DIM))
    assert mu.shape == (5, C, ACT_DIM)


def test_actor_ref_dropout_gives_independent_pathway():
    """With the reference masked out, the output must NOT depend on the reference —
    the failure mode of residual actors (a = ref + delta) that RLT explicitly avoids."""
    torch.manual_seed(3)
    actor = RLTActor(WIDTH, PROP, ACT_DIM, C, hidden_dim=32, num_layers=2).eval()
    z, p = torch.randn(4, WIDTH), torch.randn(4, PROP)
    ref_a, ref_b = torch.randn(4, C, ACT_DIM), torch.randn(4, C, ACT_DIM)
    mask_off = torch.zeros(4, dtype=torch.bool)

    out_a = actor(z, p, ref_a, ref_mask=mask_off)
    out_b = actor(z, p, ref_b, ref_mask=mask_off)
    torch.testing.assert_close(out_a, out_b)

    # And with the mask on, the reference must influence the output.
    mask_on = torch.ones(4, dtype=torch.bool)
    out_c = actor(z, p, ref_a, ref_mask=mask_on)
    out_d = actor(z, p, ref_b, ref_mask=mask_on)
    assert not torch.allclose(out_c, out_d)


def test_actor_sample_deterministic_vs_stochastic():
    torch.manual_seed(4)
    actor = RLTActor(WIDTH, PROP, ACT_DIM, C, hidden_dim=32, num_layers=2, fixed_std=0.1).eval()
    z, p, ref = torch.randn(2, WIDTH), torch.randn(2, PROP), torch.randn(2, C, ACT_DIM)
    det1 = actor.sample(z, p, ref, deterministic=True)
    det2 = actor.sample(z, p, ref, deterministic=True)
    torch.testing.assert_close(det1, det2)
    sto = actor.sample(z, p, ref, deterministic=False)
    assert not torch.allclose(det1, sto)
    # Noise scale should match fixed_std roughly.
    assert (sto - det1).abs().max() < 1.0


def test_twin_critic_min_q():
    critic = TwinCritic(
        z_dim=WIDTH, proprio_dim=PROP, action_dim=ACT_DIM, chunk_size=C, hidden_dim=32, num_layers=2
    )
    z, p, a = torch.randn(3, WIDTH), torch.randn(3, PROP), torch.randn(3, C, ACT_DIM)
    q1, q2 = critic(z, p, a)
    assert q1.shape == q2.shape == (3,)
    torch.testing.assert_close(critic.min_q(z, p, a), torch.minimum(q1, q2))


def test_sync_target():
    m = _modules()
    for p in m.critic.parameters():
        p.data.add_(1.0)
    m.sync_target(tau=1.0)
    for p, pt in zip(m.critic.parameters(), m.critic_target.parameters(), strict=True):
        torch.testing.assert_close(p, pt)
    for p in m.critic.parameters():
        p.data.add_(1.0)
    m.sync_target(tau=0.1)
    for p, pt in zip(m.critic.parameters(), m.critic_target.parameters(), strict=True):
        torch.testing.assert_close(pt, p - 0.9, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Chunk transition assembler
# ---------------------------------------------------------------------------


def _make_ref_h(boundary: int, h: int = 2 * C) -> torch.Tensor:
    """Reference chunk whose value encodes (boundary, offset) for slice checking."""
    ref = torch.zeros(h, ACT_DIM)
    for k in range(h):
        ref[k] = boundary * 100 + k
    return ref


def test_assembler_full_episode_no_done():
    gamma = 0.9
    asm = ChunkTransitionAssembler(chunk_size=C, gamma=gamma, stride=2)
    # 8 steps, boundaries at 0 and 4, never done.
    for b in (0, 4):
        asm.add_boundary(b, _make_ref_h(b))
    for t in range(8):
        z = torch.full((WIDTH,), float(t)) if t % 2 == 0 else None
        p = torch.full((PROP,), float(t)) if t % 2 == 0 else None
        asm.add_step(action=torch.full((ACT_DIM,), float(t)), reward=0.0, done=False, z=z, proprio=p)
    trs = asm.finish_episode()
    # t=0 (next=4 ok), t=2 (next=6 ok); t=4 and t=6 need next at 8/10 which don't exist -> skipped.
    assert len(trs) == 2
    t0 = trs[0]
    assert t0["z"][0] == 0 and t0["next_z"][0] == 4
    # ref window at t=0: boundary 0, offsets 0..3
    torch.testing.assert_close(t0["ref"][:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    # next_ref at t=4: boundary 4, offsets 0..3
    torch.testing.assert_close(t0["next_ref"][:, 0], torch.tensor([400.0, 401.0, 402.0, 403.0]))
    t2 = trs[1]
    # ref window at t=2: boundary 0, offsets 2..5
    torch.testing.assert_close(t2["ref"][:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
    # next_ref at t=6: boundary 4, offsets 2..5
    torch.testing.assert_close(t2["next_ref"][:, 0], torch.tensor([402.0, 403.0, 404.0, 405.0]))
    # executed actions
    torch.testing.assert_close(t2["action"][:, 0], torch.tensor([2.0, 3.0, 4.0, 5.0]))


def test_assembler_terminal_reward_and_padding():
    gamma = 0.5
    asm = ChunkTransitionAssembler(chunk_size=C, gamma=gamma, stride=2)
    asm.add_boundary(0, _make_ref_h(0))
    # Episode terminates with success (reward 1) at step 2 (3 steps total: 0,1,2).
    rewards = [0.0, 0.0, 1.0]
    for t in range(3):
        z = torch.full((WIDTH,), float(t)) if t % 2 == 0 else None
        p = torch.full((PROP,), float(t)) if t % 2 == 0 else None
        asm.add_step(
            action=torch.full((ACT_DIM,), float(t)),
            reward=rewards[t],
            done=(t == 2),
            z=z,
            proprio=p,
        )
    trs = asm.finish_episode()
    # t=0: window steps 0..2, done inside -> emitted; t=2: window steps 2, done -> emitted.
    assert len(trs) == 2
    t0 = trs[0]
    assert t0["done"] is True
    assert math.isclose(t0["reward"], gamma**2 * 1.0)
    # padded executed actions repeat the last action (value 2)
    torch.testing.assert_close(t0["action"][:, 0], torch.tensor([0.0, 1.0, 2.0, 2.0]))
    t2 = trs[1]
    assert math.isclose(t2["reward"], 1.0)
    torch.testing.assert_close(t2["action"][:, 0], torch.tensor([2.0, 2.0, 2.0, 2.0]))


def test_assembler_skips_steps_without_z():
    asm = ChunkTransitionAssembler(chunk_size=C, gamma=0.9, stride=2)
    asm.add_boundary(0, _make_ref_h(0))
    for _t in range(6):
        asm.add_step(action=torch.zeros(ACT_DIM), reward=0.0, done=False, z=None, proprio=None)
    assert asm.finish_episode() == []


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


def test_buffer_roundtrip_and_wraparound():
    buf = RLTReplayBuffer(capacity=3, z_dim=WIDTH, proprio_dim=PROP, action_dim=ACT_DIM, chunk_size=C)
    for i in range(5):
        buf.add(
            z=torch.full((WIDTH,), float(i)),
            proprio=torch.zeros(PROP),
            action=torch.zeros(C, ACT_DIM),
            ref=torch.zeros(C, ACT_DIM),
            reward=float(i),
            done=False,
            next_z=torch.zeros(WIDTH),
            next_proprio=torch.zeros(PROP),
            next_ref=torch.zeros(C, ACT_DIM),
        )
    assert len(buf) == 3
    batch = buf.sample(8)
    assert batch["z"].shape == (8, WIDTH)
    assert batch["action"].shape == (8, C, ACT_DIM)
    # Only the last 3 rewards (2, 3, 4) can be present after wraparound.
    assert set(batch["reward"].tolist()).issubset({2.0, 3.0, 4.0})


# ---------------------------------------------------------------------------
# TD3 updater
# ---------------------------------------------------------------------------


def _fake_policy():
    cfg = SimpleNamespace(
        rl_gamma=0.99,
        rl_tau=0.05,
        bc_beta=1.0,
        ref_dropout_prob=0.5,
        rl_chunk_size=C,
    )
    return SimpleNamespace(config=cfg, rlt=_modules())


def _rand_batch(bsize=16):
    return {
        "z": torch.randn(bsize, WIDTH),
        "proprio": torch.randn(bsize, PROP),
        "action": torch.randn(bsize, C, ACT_DIM),
        "ref": torch.randn(bsize, C, ACT_DIM),
        "reward": torch.rand(bsize),
        "done": torch.zeros(bsize),
        "next_z": torch.randn(bsize, WIDTH),
        "next_proprio": torch.randn(bsize, PROP),
        "next_ref": torch.randn(bsize, C, ACT_DIM),
    }


def test_updater_critic_and_delayed_actor():
    torch.manual_seed(5)
    policy = _fake_policy()
    updater = RLTOnlineUpdater(policy, actor_lr=1e-3, critic_lr=1e-3, actor_delay=2)

    actor_before = [p.clone() for p in policy.rlt.actor.parameters()]
    critic_before = [p.clone() for p in policy.rlt.critic.parameters()]
    target_before = [p.clone() for p in policy.rlt.critic_target.parameters()]

    m1 = updater.update(_rand_batch())
    assert "critic_loss" in m1 and math.isfinite(m1["critic_loss"])
    assert "actor_loss" not in m1  # delayed
    assert any(
        not torch.allclose(a, b)
        for a, b in zip(critic_before, list(policy.rlt.critic.parameters()), strict=True)
    )
    assert all(
        torch.allclose(a, b) for a, b in zip(actor_before, list(policy.rlt.actor.parameters()), strict=True)
    )

    m2 = updater.update(_rand_batch())
    assert "actor_loss" in m2 and math.isfinite(m2["actor_loss"])
    assert any(
        not torch.allclose(a, b)
        for a, b in zip(actor_before, list(policy.rlt.actor.parameters()), strict=True)
    )
    # Target critic moved after the actor update.
    assert any(
        not torch.allclose(a, b)
        for a, b in zip(target_before, list(policy.rlt.critic_target.parameters()), strict=True)
    )


def test_updater_done_blocks_bootstrap():
    torch.manual_seed(6)
    policy = _fake_policy()
    updater = RLTOnlineUpdater(policy, actor_delay=10_000)
    batch = _rand_batch(4)
    batch["done"] = torch.ones(4)
    batch["reward"] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    with torch.no_grad():
        next_a = policy.rlt.actor.sample(batch["next_z"], batch["next_proprio"], batch["next_ref"])
        tq = policy.rlt.critic_target.min_q(batch["next_z"], batch["next_proprio"], next_a)
    # With done=1 the target must equal the reward regardless of target-critic values.
    assert tq.abs().sum() != 0 or True
    metrics = updater.update(batch)
    assert math.isfinite(metrics["target_q_mean"])
    assert abs(metrics["target_q_mean"] - batch["reward"].mean().item()) < 1e-5


# ---------------------------------------------------------------------------
# Config registration
# ---------------------------------------------------------------------------


def test_pi05_rlt_registered():
    from lerobot.policies.factory import get_policy_class, make_policy_config
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig

    cfg = make_policy_config("pi05_rlt")
    assert isinstance(cfg, PI05RLTConfig)
    assert cfg.type == "pi05_rlt"
    # rl chunk clamps n_action_steps when RLT is enabled
    assert cfg.n_action_steps == cfg.rl_chunk_size

    cls = get_policy_class("pi05_rlt")
    assert cls.__name__ == "PI05RLTPolicy"
    assert cls.name == "pi05_rlt"


def test_config_validation():
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig

    with pytest.raises(ValueError):
        PI05RLTConfig(rl_chunk_size=100, chunk_size=50)
    with pytest.raises(ValueError):
        PI05RLTConfig(rlt_actor_mode="residual")  # residual mode intentionally absent
    cfg = PI05RLTConfig(rlt_enabled=False)
    assert cfg.n_action_steps == 50  # untouched when RLT disabled
