# Development reference

Use this for repository navigation, environment preparation, or validation that spans components.
All code paths below are relative to the repository root.

## Architecture

| Location | Responsibility |
|---|---|
| `src/lerobot/scripts/` | CLI entrypoints registered in `pyproject.toml [project.scripts]` |
| `src/lerobot/configs/` | draccus dataclasses; `TrainPipelineConfig`, `PreTrainedConfig`, ChoiceRegistry subclasses |
| `src/lerobot/policies/` | Per-policy folders, `PreTrainedPolicy` (`nn.Module` + `HubMixin`), lazy factory |
| `src/lerobot/processor/` | Registered `ProcessorStep` implementations and data/policy pipelines |
| `src/lerobot/datasets/` | `LeRobotDataset`, metadata, episode sampling, video decoding |
| `src/lerobot/envs/` | `EnvConfig`, environment factory, `gym_kwargs` and `create_envs()` |
| `src/lerobot/robots/`, `motors/`, `cameras/`, `teleoperators/` | Hardware abstractions under `src/lerobot/` |
| `src/lerobot/types.py`, `configs/types.py` | Shared types; second path is under `src/lerobot/` |
| `tests/` | Module tests, `fixtures/`, `mocks/`, hardware skip helpers in `utils.py` |
| `docs/source/` | HF documentation in MDX; separate docs dependencies/build |
| `examples/`, `benchmarks/`, `docker/` | User workflows, performance tools, user/CI containers |

Read existing implementations in the affected component before changing registration or contracts.
The repository uses PyTorch, HF datasets/Hub/accelerate, draccus, and Gymnasium; select optional
dependencies through the project's existing extras.

## Setup and checks

Use only the dependencies and artifacts needed for the task:

```bash
uv sync --locked
uv sync --locked --extra test --extra dev
```

`uv sync --locked --extra all` is available when all integrations are actually needed.
If tests require LFS artifacts, use the repository's LFS setup and fetch those artifacts;
do not pull large assets for a documentation-only change.

Start with the relevant pytest target. For broad changes, the existing commands include:

```bash
uv run pytest tests -svv --maxfail=10
pre-commit run --all-files
```

`DEVICE=cuda make test-end-to-end` runs GPU E2E tests and writes results under `tests/outputs/`.
Use it only when that validation is required and execution is authorized.

CI definitions in `.github/workflows/` are the source for required checks:
`quality.yml`, `fast_tests.yml`, and conditional `full_tests.yml` cover quality and tests;
`latest_deps_tests.yml`, `security.yml`, and `release.yml` handle separate maintenance/release workflows.
Check current files instead of treating a copied CI schedule as authoritative.
