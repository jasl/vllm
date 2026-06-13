# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v4 import sparse_mla
from vllm.models.deepseek_v4.nvidia import flashmla


def test_c128a_effective_topk_width_uses_current_positions() -> None:
    assert (
        sparse_mla._c128a_effective_topk_width(
            positions=torch.tensor([0, 126], dtype=torch.int64),
            compress_ratio=128,
            max_compressed_tokens=4096,
            alignment=128,
        )
        == 128
    )
    assert (
        sparse_mla._c128a_effective_topk_width(
            positions=torch.tensor([127, 1023], dtype=torch.int64),
            compress_ratio=128,
            max_compressed_tokens=4096,
            alignment=128,
        )
        == 128
    )
    assert (
        sparse_mla._c128a_effective_topk_width(
            positions=torch.tensor([524287], dtype=torch.int64),
            compress_ratio=128,
            max_compressed_tokens=8192,
            alignment=128,
        )
        == 4096
    )
    assert (
        sparse_mla._c128a_effective_topk_width(
            positions=torch.tensor([1048575], dtype=torch.int64),
            compress_ratio=128,
            max_compressed_tokens=8192,
            alignment=128,
        )
        == 8192
    )


def test_indexed_d512_split_topk_keeps_small_c128a_prefills() -> None:
    assert not flashmla._is_indexed_d512_split_topk(128)
    assert flashmla._is_indexed_d512_split_topk(256)
    assert flashmla._is_indexed_d512_split_topk(512)
    assert flashmla._is_indexed_d512_split_topk(1152)
    assert not flashmla._is_indexed_d512_split_topk(1280)


def test_indexed_d512_split_prefill_respects_min_token_env(monkeypatch) -> None:
    monkeypatch.setattr(
        flashmla.envs,
        "VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL",
        True,
    )
    monkeypatch.setattr(
        flashmla.envs,
        "VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS",
        8192,
        raising=False,
    )
    kwargs = {
        "compress_ratio": 4,
        "head_dim": 512,
        "num_prefills": 1,
        "combined_topk": 640,
        "max_prefill_seq_len": 4096,
        "swa_only": False,
    }
    assert not flashmla._use_indexed_d512_split_prefill(**kwargs)

    monkeypatch.setattr(
        flashmla.envs,
        "VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS",
        4096,
        raising=False,
    )
    assert flashmla._use_indexed_d512_split_prefill(**kwargs)


def test_indexed_d512_split_prefill_default_allows_runtime_4096(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS",
        raising=False,
    )
    monkeypatch.setattr(
        flashmla.envs,
        "VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL",
        True,
    )
    monkeypatch.delattr(
        flashmla.envs,
        "VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS",
        raising=False,
    )
    kwargs = {
        "compress_ratio": 4,
        "head_dim": 512,
        "num_prefills": 1,
        "combined_topk": 640,
        "max_prefill_seq_len": 4096,
        "swa_only": False,
    }

    with override_forward_context(
        ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
    ):
        assert flashmla._use_indexed_d512_split_prefill(**kwargs)
        assert flashmla._use_indexed_d512_chunked_prefill(
            **{**kwargs, "combined_topk": 1280}
        )


def test_indexed_d512_split_prefill_skips_dummy_profile_context(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        flashmla.envs,
        "VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL",
        True,
    )
    monkeypatch.setattr(
        flashmla.envs,
        "VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS",
        4096,
        raising=False,
    )
    kwargs = {
        "compress_ratio": 4,
        "head_dim": 512,
        "num_prefills": 1,
        "combined_topk": 640,
        "max_prefill_seq_len": 4096,
        "swa_only": False,
    }

    with override_forward_context(
        ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            additional_kwargs={"is_dummy_run": True, "is_profile": True},
        )
    ):
        assert not flashmla._use_indexed_d512_split_prefill(**kwargs)
        assert not flashmla._use_indexed_d512_chunked_prefill(
            **{**kwargs, "combined_topk": 1280}
        )


def test_indexed_d512_fused_sink_prefill_defaults_off() -> None:
    assert not flashmla._use_indexed_d512_fused_sink_prefill(
        split_prefill=False,
    )
    assert not flashmla._use_indexed_d512_fused_sink_prefill(
        split_prefill=True,
    )


def test_prefill_has_cached_prefix_detects_extend_rows() -> None:
    assert not flashmla._prefill_has_cached_prefix(
        seq_lens_cpu=torch.tensor([6, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 6, 10], dtype=torch.int32),
        num_decodes=0,
        num_prefills=2,
    )
    assert flashmla._prefill_has_cached_prefix(
        seq_lens_cpu=torch.tensor([1, 12, 8], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 5, 9], dtype=torch.int32),
        num_decodes=1,
        num_prefills=2,
    )


def test_prefill_has_cached_prefix_accepts_prefill_only_seq_lens() -> None:
    assert not flashmla._prefill_has_cached_prefix(
        seq_lens_cpu=torch.tensor([4, 4, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 6, 10, 14], dtype=torch.int32),
        num_decodes=1,
        num_prefills=3,
    )
    assert flashmla._prefill_has_cached_prefix(
        seq_lens_cpu=torch.tensor([4, 9, 4], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2, 6, 10, 14], dtype=torch.int32),
        num_decodes=1,
        num_prefills=3,
    )
