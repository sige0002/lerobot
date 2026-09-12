---
name: lerobot-vla-pytorch
description: Implement or adapt VLA policies in LeRobot's PyTorch policy and processor structure, including pi0/pi0.5 ports and RLT extensions. Use for VLA architecture, action normalization, or training/evaluation compatibility work.
---

# LeRobot VLA PyTorch

Preserve the requested policy and LeRobot's dataset, training, evaluation, and simulation contracts.
Prefer a focused extension in PyTorch; keep JAX interop where the existing implementation requires it.

## Follow the affected path

Inspect the policy registration, config, model, and relevant processor/action normalization before
changing their contracts. Follow the path into training or inference entrypoints when the change
affects them. Do not require unrelated hardware or every policy implementation as initial context.

For an experimental RLT extension, use a distinct policy name where it protects the upstream
baseline; `pi05_rlt` is an example, not a mandatory name for every task.
A bug fix to an existing policy should remain scoped to that policy when that is what the user requested.

Implement only the requested RLT features, such as rollout data, loss, advantage/return conditioning,
or a value model. Expose the needed choices through existing config conventions and keep the
unmodified baseline available for comparisons.

## Verify contracts

Check the failure modes relevant to the change:

- Dataset keys, image/token shapes, action dimensions and sequence layout.
- Normalization statistics and action scale through both training and evaluation.
- Checkpoint assumptions across JAX/PyTorch.
- Continuous action heads versus autoregressive decoding.
- Forward output, loss, and action selection through the affected policy path.

Use focused tests or a small dummy/dataset smoke run before full training.
LIBERO or SO-101 validation is appropriate when requested and available; the skill does not require
starting a simulator, training job, or real robot for every policy edit.
When adding a training/evaluation workflow, document its configuration and baseline comparison.
