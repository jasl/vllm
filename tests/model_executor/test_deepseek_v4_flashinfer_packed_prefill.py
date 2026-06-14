# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.models.deepseek_v4.common.ops import cache_utils
from vllm.models.deepseek_v4.nvidia import flashmla as flashmla_mod


def test_flashinfer_packed_prefill_env_defaults_disabled(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_DEEPSEEK_V4_FLASHINFER_PACKED_PREFILL", raising=False)

    assert envs.VLLM_DEEPSEEK_V4_FLASHINFER_PACKED_PREFILL is False


@pytest.mark.parametrize("value, expected", [("0", False), ("1", True)])
def test_flashinfer_packed_prefill_env_parses_bool(
    monkeypatch,
    value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("VLLM_DEEPSEEK_V4_FLASHINFER_PACKED_PREFILL", value)

    assert envs.VLLM_DEEPSEEK_V4_FLASHINFER_PACKED_PREFILL is expected


def test_flashinfer_packed_prefill_import_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(flashmla_mod.importlib, "import_module", lambda _: object())

    assert flashmla_mod._flashinfer_packed_sparse_mla_attention() is None


def test_flashinfer_packed_prefill_gate_requires_env_and_packed_shape(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        flashmla_mod.envs,
        "VLLM_DEEPSEEK_V4_FLASHINFER_PACKED_PREFILL",
        False,
        raising=False,
    )

    assert not flashmla_mod._use_flashinfer_packed_prefill(
        compress_ratio=128,
        head_dim=512,
        swa_only=False,
        query_tokens=256,
        compressed_block_size=2,
        swa_block_size=64,
        q_device=torch.device("cuda"),
    )

    monkeypatch.setattr(
        flashmla_mod.envs,
        "VLLM_DEEPSEEK_V4_FLASHINFER_PACKED_PREFILL",
        True,
        raising=False,
    )

    assert flashmla_mod._use_flashinfer_packed_prefill(
        compress_ratio=128,
        head_dim=512,
        swa_only=False,
        query_tokens=256,
        compressed_block_size=2,
        swa_block_size=64,
        q_device=torch.device("cuda"),
    )
    assert not flashmla_mod._use_flashinfer_packed_prefill(
        compress_ratio=128,
        head_dim=512,
        swa_only=False,
        query_tokens=64,
        compressed_block_size=2,
        swa_block_size=64,
        q_device=torch.device("cuda"),
    )
    assert not flashmla_mod._use_flashinfer_packed_prefill(
        compress_ratio=128,
        head_dim=512,
        swa_only=False,
        query_tokens=256,
        compressed_block_size=4,
        swa_block_size=64,
        q_device=torch.device("cuda"),
    )


def test_flashinfer_mixed_sparse_indices_reuses_workspace(monkeypatch) -> None:
    class FakeKernel:
        def __init__(self) -> None:
            self.grid = None
            self.kwargs = None

        def __getitem__(self, grid):
            self.grid = grid

            def launch(*args, **kwargs):
                self.kwargs = kwargs

            return launch

    fake_kernel = FakeKernel()
    monkeypatch.setattr(
        cache_utils, "_build_flashinfer_mixed_sparse_indices_kernel", fake_kernel
    )

    decode_swa_indices = torch.empty((0, 4), dtype=torch.int32)
    prefill_topk_indices = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    seq_lens = torch.tensor([10], dtype=torch.int32)
    token_to_req_indices = torch.zeros(2, dtype=torch.int32)
    block_table = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    sparse_indices_workspace = torch.full((4, 12), -1, dtype=torch.int32)
    sparse_lens_workspace = torch.full((4,), -1, dtype=torch.int32)

    sparse_indices, sparse_lens = cache_utils.build_flashinfer_mixed_sparse_indices(
        decode_swa_indices=decode_swa_indices,
        decode_compressed_indices=None,
        decode_compressed_topk_lens=None,
        prefill_topk_indices=prefill_topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        token_to_req_indices=token_to_req_indices,
        swa_block_table=block_table,
        swa_block_size=64,
        compressed_block_table=block_table,
        compressed_block_size=2,
        window_size=4,
        compress_ratio=128,
        topk=3,
        sparse_indices=sparse_indices_workspace,
        sparse_topk_lens=sparse_lens_workspace,
    )

    assert sparse_indices.shape == (2, 12)
    assert sparse_lens.shape == (2,)
    assert sparse_indices.data_ptr() == sparse_indices_workspace.data_ptr()
    assert sparse_lens.data_ptr() == sparse_lens_workspace.data_ptr()
    assert fake_kernel.grid == (2,)
    assert fake_kernel.kwargs["PADDED_TOP_K"] == 8
