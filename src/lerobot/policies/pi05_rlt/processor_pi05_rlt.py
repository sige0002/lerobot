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

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from ..pi05.processor_pi05 import make_pi05_pre_post_processors
from .configuration_pi05_rlt import PI05RLTConfig


def make_pi05_rlt_pre_post_processors(
    config: PI05RLTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """PI05RLT uses the exact same processing pipelines as pi05.

    The RLT actor and replay buffer operate in the same normalized action space
    the pi05 model outputs; un-normalization stays in the shared post-processor.
    """
    return make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
