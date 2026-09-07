# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every DSpark model class the V2 speculator can load must expose the
sampling hooks it calls. The greedy branch of DSparkSpeculator._sample_logits
calls model.map_draft_to_target on every profile_run (draft_logits is None
during profiling), so a missing hook is a cold-boot blocker, not an edge case
-- upstream #49969 added the hooks to amd/, xpu/, kimi_k3 and qwen3_dspark
but not the nvidia deepseek_v4 class (reported by alexbi29 in
vllm-project/vllm#41834)."""

import inspect
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.nvidia import dspark as nvidia_dspark
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def test_nvidia_dspark_exposes_v2_speculator_hooks():
    cls = nvidia_dspark.DSparkDeepseekV4ForCausalLM
    for hook in ("map_draft_to_target", "compute_draft_logits"):
        assert hasattr(cls, hook), (
            f"{cls.__name__} lacks {hook}; the V2 DSpark speculator dies in "
            "profile_run on the first cold boot"
        )


def test_map_draft_to_target_is_identity_for_full_vocab():
    src = inspect.getsource(
        nvidia_dspark.DSparkDeepseekV4ForCausalLM.map_draft_to_target
    )
    assert "return draft_ids" in src


def test_sequential_sampling_keeps_logits_in_reduced_draft_vocabulary():
    """Markov bias is draft-sized; target-sized logits cannot be added to it."""
    draft_logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
    d2t = torch.tensor([7, 11, 19])
    model = SimpleNamespace(
        compute_draft_logits=lambda hidden: draft_logits,
        compute_logits=lambda hidden: torch.zeros(2, 20),
        markov_embed=lambda previous: previous,
        markov_bias=lambda embedding: torch.zeros(1, 3),
    )
    speculator = SimpleNamespace(
        _draft_topk=None,
        num_speculative_steps=2,
        sample_indices=torch.arange(2),
        sample_idx_mapping=torch.zeros(2, dtype=torch.long),
        sample_pos=torch.tensor([5, 6]),
        input_buffers=SimpleNamespace(input_ids=torch.tensor([4, 0])),
        _anchor_idx=torch.tensor([0]),
        model=model,
        enable_adaptive_verification=False,
        _sample_logits=lambda logits, *args: d2t[logits.argmax(dim=-1)],
        draft_tokens=torch.empty(1, 2, dtype=torch.long),
    )
    DSparkSpeculator._sample_sequential(speculator, 1, torch.zeros(2, 4))
    assert speculator.draft_tokens.tolist() == [[11, 7]]
