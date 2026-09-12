# LeRobot agent guidance

LeRobot is a PyTorch library for robotics policies, datasets, training, evaluation, and hardware control.
Use Python 3.12+ and `uv run`; dependency and tool settings come from `pyproject.toml` and `uv.lock`.

## Implementation constraints

- Extend the existing policy/config/processor/dataset structure. Preserve compatibility with training and evaluation entrypoints.
- Optional policy, environment, and hardware dependencies must remain guarded or lazily imported. Check the relevant extra in `pyproject.toml`.
- Configs use dataclasses and draccus registration. Policies inherit `PreTrainedPolicy` and use the existing factory and processor contracts.
- Type checking is gradual; use the module-specific settings in `pyproject.toml` when changing annotations.
- Choose focused tests for the changed behavior. Dataset video tests may need ffmpeg; hardware tests use the existing skip helpers. GPU, E2E, and hardware runs require the relevant environment and task scope.

## Context by task

- SO-101 setup, recording, policy selection, training duration, or evaluation help:
  read the relevant section of [AGENT_GUIDE.md](AGENT_GUIDE.md).
- Repository navigation, environment setup, or broader validation:
  use [development reference](docs/agent/development.md).
- VLA policy implementation, PyTorch ports, or RLT extensions:
  use [lerobot-vla-pytorch](.agents/skills/lerobot-vla-pytorch/SKILL.md).
- Recording or revisiting a reusable failure and its fix:
  use [issue-log](.agents/skills/issue-log/SKILL.md).

Read the context needed for the change. Routine edits do not require installing every extra,
running CUDA E2E tests, or loading all hardware documentation.
Skill sources live in `.agents/skills/`; `.claude/skills/` retains compatibility links.
`CLAUDE.md` links to this file.
