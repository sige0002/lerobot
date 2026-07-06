"""GPU integration tests for PI05-RLT against a real pi05 checkpoint.

Heavy: downloads ``lerobot/pi05_libero_finetuned`` (~7 GB) and requires CUDA.
Enable explicitly with:

    RLT_INTEGRATION=1 uv run pytest tests/policies/pi05_rlt/test_pi05_rlt_integration.py -svv
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RLT_INTEGRATION") != "1",
    reason="Set RLT_INTEGRATION=1 to run heavy pi05_rlt integration tests (GPU + checkpoint download)",
)

CHECKPOINT = os.environ.get("RLT_CHECKPOINT", "lerobot/pi05_libero_finetuned")


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def base_setup(device):
    """Load the plain pi05 policy + its processors from the checkpoint."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
    cfg.pretrained_path = CHECKPOINT
    cfg.device = str(device)
    policy = PI05Policy.from_pretrained(CHECKPOINT, config=cfg)
    policy.to(device)
    policy.eval()

    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return cfg, policy, pre, post


@pytest.fixture(scope="module")
def rlt_policy(base_setup, device):
    from lerobot.policies.pi05_rlt.configuration_pi05_rlt import PI05RLTConfig
    from lerobot.policies.pi05_rlt.modeling_pi05_rlt import PI05RLTPolicy

    base_cfg, _, _, _ = base_setup
    cfg = PI05RLTConfig.from_pi05_config(base_cfg, rlt_enabled=True, rlt_actor_mode="reference")
    cfg.pretrained_path = CHECKPOINT
    cfg.device = str(device)
    policy = PI05RLTPolicy.from_pretrained(CHECKPOINT, config=cfg)
    policy.to(device)
    policy.eval()
    return policy


@pytest.fixture(scope="module")
def batch(base_setup, device):
    """A synthetic LIBERO-like observation pushed through the real preprocessor."""
    _, _, pre, _ = base_setup
    torch.manual_seed(0)
    obs = {
        "observation.images.image": torch.rand(3, 224, 224),
        "observation.images.image2": torch.rand(3, 224, 224),
        "observation.state": torch.randn(8) * 0.1,
        "task": "pick up the alphabet soup and place it in the basket",
    }
    return pre(obs)


def _fixed_noise(policy, device):
    g = torch.Generator(device="cpu").manual_seed(1234)
    noise = torch.randn(
        1, policy.config.chunk_size, policy.config.max_action_dim, generator=g, dtype=torch.float32
    )
    return noise.to(device)


def test_backbone_weights_identical(base_setup, rlt_policy):
    _, base_policy, _, _ = base_setup
    base_sd = base_policy.model.state_dict()
    rlt_sd = rlt_policy.model.state_dict()
    assert base_sd.keys() == rlt_sd.keys()
    for k in base_sd:
        assert torch.equal(base_sd[k], rlt_sd[k]), f"backbone weight mismatch: {k}"


def test_backbone_frozen(rlt_policy):
    assert all(not p.requires_grad for p in rlt_policy.model.parameters())
    assert any(p.requires_grad for p in rlt_policy.rlt.parameters())


def test_parity_rlt_disabled(base_setup, rlt_policy, batch, device):
    """rlt_enabled=false must reproduce pi05 exactly (same weights, same noise)."""
    _, base_policy, _, _ = base_setup
    noise = _fixed_noise(base_policy, device)
    base_actions = base_policy.predict_action_chunk(dict(batch), noise=noise)

    rlt_policy.config.rlt_enabled = False
    try:
        rlt_actions = rlt_policy.predict_action_chunk(dict(batch), noise=noise)
    finally:
        rlt_policy.config.rlt_enabled = True
    torch.testing.assert_close(rlt_actions, base_actions, atol=0.0, rtol=0.0)


def test_parity_reference_mode(base_setup, rlt_policy, batch, device):
    """Reference (pass-through) mode must match pi05: the custom prefill+denoise
    path replicates PI05Pytorch.sample_actions op-for-op."""
    _, base_policy, _, _ = base_setup
    noise = _fixed_noise(base_policy, device)
    base_actions = base_policy.predict_action_chunk(dict(batch), noise=noise)

    rlt_policy.config.rlt_actor_mode = "reference"
    out = rlt_policy.predict_rlt_chunk(dict(batch), noise=noise)
    torch.testing.assert_close(out["actions"], base_actions, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out["ref_chunk"], base_actions, atol=1e-4, rtol=1e-4)


def test_rl_token_extraction(rlt_policy, batch):
    z = rlt_policy.extract_rl_token(dict(batch))
    assert z.shape == (1, rlt_policy.config.rlt_width)
    assert z.dtype == torch.float32
    assert torch.isfinite(z).all()
    # Deterministic given the same observation.
    z2 = rlt_policy.extract_rl_token(dict(batch))
    torch.testing.assert_close(z, z2)


def test_actor_mode_changes_only_first_c_steps(rlt_policy, batch, device):
    noise = _fixed_noise(rlt_policy, device)
    rlt_policy.config.rlt_actor_mode = "actor"
    try:
        out = rlt_policy.predict_rlt_chunk(dict(batch), noise=noise, deterministic=True)
    finally:
        rlt_policy.config.rlt_actor_mode = "reference"
    c = rlt_policy.config.rl_chunk_size
    # Beyond C the chunk is the untouched reference.
    torch.testing.assert_close(out["actions"][:, c:], out["ref_chunk"][:, c:])
    # The first C steps come from the (randomly initialized) actor and should differ.
    assert not torch.allclose(out["actions"][:, :c], out["ref_chunk"][:, :c])


def test_stage1_forward_and_learning(rlt_policy, batch):
    rlt_policy.config.train_stage = "rl_token"
    params = rlt_policy.get_optim_params()
    optim = torch.optim.Adam(params, lr=1e-4)

    loss0, _ = rlt_policy.forward(dict(batch))
    assert torch.isfinite(loss0)
    losses = []
    for _ in range(10):
        loss, _ = rlt_policy.forward(dict(batch))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"reconstruction loss did not decrease: {losses}"


def test_backbone_untouched_by_stage1(base_setup, rlt_policy):
    """After Stage 1 optimizer steps in the previous test, pi05 weights must be unchanged."""
    _, base_policy, _, _ = base_setup
    base_sd = base_policy.model.state_dict()
    rlt_sd = rlt_policy.model.state_dict()
    for k in base_sd:
        assert torch.equal(base_sd[k], rlt_sd[k]), f"backbone changed during Stage 1: {k}"
