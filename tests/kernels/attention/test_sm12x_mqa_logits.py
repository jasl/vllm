# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x MQA logits: Triton kernel and torch fallback must stay interchangeable.

The sparse-attn-indexer prefill path calls ``_fp8_mqa_logits_sm12x`` with
``clean_logits=False``. The Triton kernel cleans unconditionally (writes -inf
outside ``[ks, ke)``), a strict superset of that contract, so the dispatch must
route both clean_logits values to it. A regression here silently reroutes long
prefills onto the torch fallback, which degenerates to head_chunk_size=1 at
large ``seq_len_kv`` and stalls the engine for minutes per chunk.
"""

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks
from vllm.models.deepseek_v4.nvidia.ops.sm12x_deep_gemm_fallbacks import (
    _fp8_mqa_logits_sm12x,
    _fp8_mqa_logits_torch,
)


def _make_inputs(seq_len, seq_len_kv, num_heads, head_dim, device):
    torch.manual_seed(0)
    q = torch.randn(seq_len, num_heads, head_dim, device=device).to(
        torch.float8_e4m3fn
    )
    k = torch.randn(seq_len_kv, head_dim, device=device).to(torch.float8_e4m3fn)
    scales = (torch.rand(seq_len_kv, device=device) * 0.5 + 0.75).float()
    weights = torch.rand(seq_len, num_heads, device=device, dtype=torch.float32)
    ks = (torch.rand(seq_len, device=device) * (seq_len_kv // 4)).to(torch.int32)
    ke = (
        torch.rand(seq_len, device=device) * (seq_len_kv // 2) + seq_len_kv // 2
    ).to(torch.int32)
    return q, k, scales, weights, ks, ke


def test_dispatch_routes_unclean_logits_to_triton(monkeypatch):
    """clean_logits=False (the indexer prefill contract) must hit the Triton
    kernel, not the torch fallback."""
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_mqa

    hit = {}

    def fake_triton(q_values, kv, weights, cu_ks, cu_ke):
        hit["triton"] = True
        return torch.zeros(
            (q_values.shape[0], kv[0].shape[0]), dtype=torch.float32
        )

    monkeypatch.setattr(sm12x_mqa, "fp8_mqa_logits_triton", fake_triton)
    q = torch.zeros(4, 8, 16).to(torch.float8_e4m3fn)
    k = torch.zeros(32, 16).to(torch.float8_e4m3fn)
    scales = torch.ones(32)
    weights = torch.ones(4, 8)
    ks = torch.zeros(4, dtype=torch.int32)
    ke = torch.full((4,), 32, dtype=torch.int32)

    for clean_logits in (True, False):
        hit.clear()
        _fp8_mqa_logits_sm12x(
            (q, None), (k, scales), weights, ks, ke, clean_logits
        )
        assert hit.get("triton"), (
            f"clean_logits={clean_logits} fell back to the torch path"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "seq_len,seq_len_kv,num_heads,head_dim",
    [(64, 2048, 32, 128), (17, 511, 64, 128)],
)
def test_triton_matches_torch_fallback(seq_len, seq_len_kv, num_heads, head_dim):
    """Numerics: Triton output equals the torch reference on the valid window
    and applies the identical -inf mask outside it."""
    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import fp8_mqa_logits_triton

    q, k, scales, weights, ks, ke = _make_inputs(
        seq_len, seq_len_kv, num_heads, head_dim, "cuda"
    )
    ref = _fp8_mqa_logits_torch(
        (q, None), (k, scales), weights, ks, ke, clean_logits=True
    )
    out = fp8_mqa_logits_triton(q, (k, scales), weights, ks, ke)[
        :seq_len, :seq_len_kv
    ]
    assert torch.equal(torch.isfinite(ref), torch.isfinite(out))
    finite = torch.isfinite(ref)
    torch.testing.assert_close(
        out[finite], ref[finite], rtol=1e-4, atol=1e-4
    )
