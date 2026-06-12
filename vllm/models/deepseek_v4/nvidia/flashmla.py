# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    dequantize_combined_sparse_mla_decode_kv,
    dequantize_global_slots_k_cache,
    sparse_prefill_combined_topk_size,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.v1.attention.backends.mla.sparse_mla_env import (
    is_triton_sparse_mla_enabled,
    is_triton_sparse_mla_enabled_for_platform,
    triton_sparse_mla_matmul_decode_enabled,
    triton_sparse_mla_prefill_topk_chunk_size,
    triton_sparse_mla_query_chunk_size,
    triton_sparse_mla_topk_chunk_size,
)
from vllm.v1.attention.backends.mla.sparse_mla_kernels import (
    accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead,
    accumulate_indexed_d512_chunked_sparse_mla_attention,
    accumulate_indexed_d512_split_sparse_mla_attention,
    accumulate_indexed_sparse_mla_attention_chunk,
    build_combined_sparse_mla_decode_valid_mask,
    finish_sparse_mla_attention_with_sink,
    finish_two_sparse_mla_attention_states_with_sink,
    fp8ds_global_paged_sparse_mla_attention_with_sink_multihead,
    fp8ds_paged_sparse_mla_attention_with_sink_multihead,
    matmul_sparse_mla_attention_with_sink,
    sparse_mla_decode_head_block_size,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


_sparse_mla_prefill_stats_disable_depth = 0
_INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS = 8192
_INDEXED_D512_SPLIT_PREFILL_MIN_TOPK = 256
_INDEXED_D512_SPLIT_PREFILL_MAX_TOPK = 1152


@contextmanager
def _disable_sparse_mla_prefill_stats() -> Iterator[None]:
    global _sparse_mla_prefill_stats_disable_depth
    _sparse_mla_prefill_stats_disable_depth += 1
    try:
        yield
    finally:
        _sparse_mla_prefill_stats_disable_depth -= 1


def _sparse_mla_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _sparse_mla_cuda_device() -> int | None:
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return None


def _sparse_mla_prefill_stats_path() -> Path | None:
    raw_path = envs.VLLM_DEEPSEEK_V4_SPARSE_MLA_STATS_PATH
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / f"rank{_sparse_mla_rank()}.jsonl"


def _sparse_mla_prefill_stats_enabled() -> bool:
    return (
        _sparse_mla_prefill_stats_disable_depth <= 0
        and bool(envs.VLLM_DEEPSEEK_V4_SPARSE_MLA_STATS_PATH)
    )


def _sparse_mla_prefill_stage_timing_enabled() -> bool:
    return (
        _sparse_mla_prefill_stats_enabled()
        and envs.VLLM_DEEPSEEK_V4_SPARSE_MLA_STATS_STAGE_TIMING
    )


class _SparseMLAPrefillStageTimer:
    def __init__(self) -> None:
        self.enabled = (
            _sparse_mla_prefill_stage_timing_enabled()
            and torch.cuda.is_available()
        )
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events.append((name, start, end))

    def elapsed_ms(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for name, start, end in self._events:
            end.synchronize()
            totals[name] = totals.get(name, 0.0) + float(start.elapsed_time(end))
        return totals


def _sparse_mla_lens_summary(combined_lens: torch.Tensor) -> dict[str, int]:
    lens = combined_lens.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
    count = int(lens.numel())
    if count == 0:
        return {
            "count": 0,
            "min": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
            "sum": 0,
        }
    lens, _ = torch.sort(lens)

    def percentile(q: float) -> int:
        idx = min(count - 1, int(math.ceil(q * (count - 1))))
        return int(lens[idx].item())

    return {
        "count": count,
        "min": int(lens[0].item()),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": int(lens[-1].item()),
        "sum": int(lens.sum().item()),
    }


def _sparse_mla_candidate_overlap_groups(
    per_row: list[list[int]],
) -> dict[str, dict[str, float | int]]:
    rows = len(per_row)
    group_summaries: dict[str, dict[str, float | int]] = {}
    for group_size in (2, 4, 8, 16, 32):
        groups = 0
        total_valid = 0
        total_unique = 0
        for start in range(0, rows, group_size):
            group = per_row[start : start + group_size]
            if len(group) < group_size:
                break
            flattened: list[int] = []
            for values in group:
                flattened.extend(values)
            if not flattened:
                continue
            groups += 1
            total_valid += len(flattened)
            total_unique += len(set(flattened))
        group_summaries[str(group_size)] = {
            "groups": groups,
            "valid_candidates": total_valid,
            "unique_candidates": total_unique,
            "unique_to_valid_ratio": (
                float(total_unique) / float(total_valid) if total_valid else 0.0
            ),
        }
    return group_summaries


def _sparse_mla_candidate_rows(
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    sample_rows: int,
) -> list[list[int]]:
    rows = min(
        int(sample_rows),
        int(combined_lens.numel()),
        int(combined_indices.shape[0]),
    )
    if rows <= 0:
        return []
    indices_cpu = combined_indices[:rows].detach().to(device="cpu", dtype=torch.int64)
    lens_cpu = combined_lens[:rows].detach().to(device="cpu", dtype=torch.int64)
    per_row: list[list[int]] = []
    for row_idx in range(rows):
        valid_len = max(0, min(int(lens_cpu[row_idx].item()), indices_cpu.shape[1]))
        row_values = indices_cpu[row_idx, :valid_len]
        per_row.append([int(x) for x in row_values.tolist() if int(x) >= 0])
    return per_row


def _sparse_mla_candidate_overlap_summary(
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    sample_rows: int,
) -> dict[str, object]:
    if sample_rows <= 0:
        return {}
    per_row = _sparse_mla_candidate_rows(
        combined_indices=combined_indices,
        combined_lens=combined_lens,
        sample_rows=sample_rows,
    )
    return {
        "sample_rows": len(per_row),
        "groups": _sparse_mla_candidate_overlap_groups(per_row),
    }


def _sparse_mla_candidate_region_overlap_summary(
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    sample_rows: int,
    gather_region_size: int,
    swa_region_offset: int,
) -> dict[str, object]:
    if sample_rows <= 0 or gather_region_size <= 0:
        return {}
    per_row = _sparse_mla_candidate_rows(
        combined_indices=combined_indices,
        combined_lens=combined_lens,
        sample_rows=sample_rows,
    )
    compressed_rows: list[list[int]] = []
    swa_rows: list[list[int]] = []
    for values in per_row:
        compressed: list[int] = []
        swa: list[int] = []
        for value in values:
            local = value % gather_region_size
            if local < swa_region_offset:
                compressed.append(value)
            else:
                swa.append(value)
        compressed_rows.append(compressed)
        swa_rows.append(swa)
    return {
        "sample_rows": len(per_row),
        "compressed": _sparse_mla_candidate_overlap_groups(compressed_rows),
        "swa": _sparse_mla_candidate_overlap_groups(swa_rows),
    }


def _sparse_mla_candidate_region_work_summary(
    *,
    query_tokens: int,
    combined_topk: int,
    compressed_region_width: int,
    swa_region_width: int,
    compressed_candidate_visits: int | None,
    swa_candidate_visits: int | None,
) -> dict[str, dict[str, float | int]]:
    if query_tokens <= 0 or combined_topk <= 0:
        return {}
    if compressed_candidate_visits is None and swa_candidate_visits is None:
        return {}

    def summarize_region(slots: int, effective: int) -> dict[str, float | int]:
        padding = max(0, slots - effective)
        return {
            "candidate_slots": slots,
            "effective_candidate_visits": effective,
            "padding_candidate_visits": padding,
            "padding_ratio": float(padding) / float(slots) if slots else 0.0,
        }

    summary: dict[str, dict[str, float | int]] = {}
    compressed_width = max(0, int(compressed_region_width))
    swa_width = max(0, int(swa_region_width))
    if compressed_width > 0 and compressed_candidate_visits is not None:
        summary["compressed"] = summarize_region(
            slots=int(query_tokens) * compressed_width,
            effective=max(0, int(compressed_candidate_visits)),
        )
    if swa_width > 0 and swa_candidate_visits is not None:
        summary["swa"] = summarize_region(
            slots=int(query_tokens) * swa_width,
            effective=max(0, int(swa_candidate_visits)),
        )
    alignment_width = max(
        0,
        int(combined_topk) - compressed_width - swa_width,
    )
    if alignment_width > 0:
        summary["alignment_padding"] = summarize_region(
            slots=int(query_tokens) * alignment_width,
            effective=0,
        )
    return summary


def _sparse_mla_prefill_candidate_region_visits(
    *,
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_decodes: int,
    chunk_start: int,
    chunk_end: int,
    top_k: int,
    compress_ratio: int,
    window_size: int,
) -> tuple[int, int]:
    compressed_visits = 0
    swa_visits = 0
    safe_compress_ratio = max(1, int(compress_ratio))
    safe_top_k = max(0, int(top_k))
    safe_window_size = max(0, int(window_size))
    for req_idx in range(chunk_start, chunk_end):
        query_start = int(query_start_loc_cpu[num_decodes + req_idx].item())
        query_end = int(query_start_loc_cpu[num_decodes + req_idx + 1].item())
        query_len = max(0, query_end - query_start)
        seq_len = int(seq_lens_cpu[req_idx].item())
        start_pos = seq_len - query_len
        for token_offset in range(query_len):
            pos = start_pos + token_offset
            token_len = max(0, pos + 1)
            compressed_visits += min(token_len // safe_compress_ratio, safe_top_k)
            swa_visits += min(token_len, safe_window_size)
    return compressed_visits, swa_visits


def _write_sparse_mla_prefill_stats(
    *,
    layer_type: str,
    layer_prefix: str,
    compress_ratio: int,
    num_prefills: int,
    query_tokens: int,
    combined_topk: int,
    combined_lens: torch.Tensor,
    combined_indices: torch.Tensor | None = None,
    gather_region_size: int = 0,
    swa_region_offset: int = 0,
    compressed_region_width: int = 0,
    swa_region_width: int = 0,
    compressed_candidate_visits: int | None = None,
    swa_candidate_visits: int | None = None,
    stage_timings_ms: dict[str, float] | None = None,
) -> None:
    if not _sparse_mla_prefill_stats_enabled():
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    try:
        path = _sparse_mla_prefill_stats_path()
    except OSError:
        return
    if path is None:
        return

    try:
        lens_summary = _sparse_mla_lens_summary(combined_lens)
        candidate_slots = int(query_tokens) * int(combined_topk)
        effective_visits = int(lens_summary["sum"])
        padding_visits = max(0, candidate_slots - effective_visits)
        row: dict[str, object] = {
            "kind": "deepseek_v4_sparse_mla_prefill_stats",
            "version": 1,
            "rank": _sparse_mla_rank(),
            "cuda_device": _sparse_mla_cuda_device(),
            "layer_type": layer_type,
            "layer_prefix": layer_prefix,
            "compress_ratio": int(compress_ratio),
            "num_prefills": int(num_prefills),
            "query_tokens": int(query_tokens),
            "combined_topk": int(combined_topk),
            "candidate_slots": candidate_slots,
            "effective_candidate_visits": effective_visits,
            "padding_candidate_visits": padding_visits,
            "combined_lens": lens_summary,
        }
        region_work = _sparse_mla_candidate_region_work_summary(
            query_tokens=int(query_tokens),
            combined_topk=int(combined_topk),
            compressed_region_width=int(compressed_region_width),
            swa_region_width=int(swa_region_width),
            compressed_candidate_visits=compressed_candidate_visits,
            swa_candidate_visits=swa_candidate_visits,
        )
        if region_work:
            row["candidate_region_work"] = region_work
        if stage_timings_ms:
            row["stage_timings_ms"] = {
                str(name): float(value)
                for name, value in sorted(stage_timings_ms.items())
            }
        overlap_rows = envs.VLLM_DEEPSEEK_V4_SPARSE_MLA_STATS_OVERLAP_ROWS
        if combined_indices is not None and overlap_rows > 0:
            row["candidate_overlap"] = _sparse_mla_candidate_overlap_summary(
                combined_indices=combined_indices,
                combined_lens=combined_lens,
                sample_rows=overlap_rows,
            )
            row["candidate_region_overlap"] = (
                _sparse_mla_candidate_region_overlap_summary(
                    combined_indices=combined_indices,
                    combined_lens=combined_lens,
                    sample_rows=overlap_rows,
                    gather_region_size=gather_region_size,
                    swa_region_offset=swa_region_offset,
                )
            )
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception:
        # Diagnostic stats must never affect inference.
        return


def _use_indexed_d512_split_prefill(
    *,
    compress_ratio: int,
    head_dim: int,
    num_prefills: int,
    combined_topk: int,
    max_prefill_seq_len: int,
    swa_only: bool,
) -> bool:
    return (
        envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL
        and not swa_only
        and compress_ratio in (4, 128)
        and head_dim == 512
        and num_prefills == 1
        and _is_indexed_d512_split_topk(combined_topk)
        and max_prefill_seq_len >= _INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS
    )


def _is_indexed_d512_split_topk(combined_topk: int) -> bool:
    return (
        _INDEXED_D512_SPLIT_PREFILL_MIN_TOPK
        <= combined_topk
        <= _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK
    )


def _use_indexed_d512_chunked_prefill(
    *,
    compress_ratio: int,
    head_dim: int,
    num_prefills: int,
    combined_topk: int,
    max_prefill_seq_len: int,
    swa_only: bool,
) -> bool:
    return (
        envs.VLLM_DEEPSEEK_V4_INDEXED_D512_CHUNKED_PREFILL
        and envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL
        and not swa_only
        and compress_ratio in (4, 128)
        and head_dim == 512
        and num_prefills == 1
        and combined_topk > _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK
        and max_prefill_seq_len >= _INDEXED_D512_SPLIT_PREFILL_MIN_TOKENS
    )


def _sparse_mla_prefill_gather_len_upper_bound(
    *,
    max_model_len: int,
    max_num_batched_tokens: int,
    window_size: int,
) -> tuple[int, int]:
    max_query_chunk_tokens = max(1, min(max_model_len, max_num_batched_tokens))
    max_prefix_len = max(max_model_len - max_query_chunk_tokens, 0)
    max_gather_len = max_query_chunk_tokens + min(
        max_prefix_len,
        max(window_size - 1, 0),
    )
    return max_query_chunk_tokens, max_gather_len


class DeepseekV4FlashMLAAttention(DeepseekV4Attention):
    """FlashMLA sparse MLA attention layer for DeepSeek V4 (CUDA)."""

    backend_cls = DeepseekV4FlashMLABackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128.
        if num_heads > 128:
            raise ValueError(
                f"DeepseekV4 FlashMLA does not support {num_heads} heads "
                "(FP8 decode kernel requires h_q in {64, 128})."
            )
        return 64 if num_heads <= 64 else 128

    @classmethod
    def _prefill_workspace_topk_bound(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
    ) -> int:
        if layer.compress_ratio <= 1:
            return 0
        if (
            layer.topk_indices_buffer is not None
            and layer.topk_indices_buffer.ndim > 0
            and layer.topk_indices_buffer.shape[-1] > 0
        ):
            return int(layer.topk_indices_buffer.shape[-1])
        indexer_topk = getattr(layer.indexer, "topk_tokens", None)
        if indexer_topk is not None:
            return int(indexer_topk)
        return 2048

    @classmethod
    def _prefill_stats_layer_type(
        cls,
        *,
        triton_sparse_mla_enabled: bool,
        indexed_d512_split_prefill: bool,
        indexed_d512_chunked_prefill: bool = False,
    ) -> str:
        if not triton_sparse_mla_enabled:
            return "mla_prefill_flashmla"
        if indexed_d512_chunked_prefill:
            return "mla_prefill_indexed_d512_chunked"
        if indexed_d512_split_prefill:
            return "mla_prefill_indexed_d512"
        return "mla_prefill_chunk"

    @classmethod
    def _prefill_workspace_reservation_specs(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        max_model_len = max(1, int(layer.max_model_len))
        max_num_batched_tokens = max(1, int(layer.max_num_batched_tokens))
        window_size = max(1, int(layer.window_size))
        compress_ratio = max(1, int(layer.compress_ratio))
        head_dim = int(layer.head_dim)
        num_heads = int(layer.n_local_heads)

        max_query_chunk_tokens, max_gather_len = (
            _sparse_mla_prefill_gather_len_upper_bound(
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                window_size=window_size,
            )
        )
        if compress_ratio <= 1:
            m_bound = max_gather_len
        else:
            compressed_region_size = max_model_len // compress_ratio
            m_bound = compressed_region_size + max_gather_len

        combined_topk = sparse_prefill_combined_topk_size(
            cls._prefill_workspace_topk_bound(layer),
            window_size,
        )
        specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            ((layer.PREFILL_CHUNK_SIZE, m_bound, head_dim), torch.bfloat16),
            ((max_query_chunk_tokens, combined_topk), torch.int32),
            ((max_query_chunk_tokens,), torch.int32),
        ]
        if is_triton_sparse_mla_enabled_for_platform():
            query_chunk_size = min(
                max_query_chunk_tokens,
                triton_sparse_mla_query_chunk_size(),
            )
            specs.extend(
                [
                    ((query_chunk_size, num_heads), torch.float32),
                    ((query_chunk_size, num_heads), torch.float32),
                    ((query_chunk_size, num_heads, head_dim), torch.float32),
                ]
            )
            if _use_indexed_d512_split_prefill(
                compress_ratio=compress_ratio,
                head_dim=head_dim,
                num_prefills=1,
                combined_topk=combined_topk,
                max_prefill_seq_len=max_model_len,
                swa_only=False,
            ):
                specs.append(
                    ((query_chunk_size, num_heads, combined_topk), torch.float32)
                )
            elif _use_indexed_d512_chunked_prefill(
                compress_ratio=compress_ratio,
                head_dim=head_dim,
                num_prefills=1,
                combined_topk=combined_topk,
                max_prefill_seq_len=max_model_len,
                swa_only=False,
            ):
                chunked_score_width = min(
                    combined_topk,
                    _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK,
                )
                specs.extend(
                    (
                        (
                            (query_chunk_size, num_heads, chunked_score_width),
                            torch.float32,
                        ),
                        ((query_chunk_size, num_heads), torch.float32),
                        ((query_chunk_size, num_heads), torch.float32),
                        ((query_chunk_size, num_heads, head_dim), torch.float32),
                    )
                )
        return tuple(specs)

    @classmethod
    def _reserve_prefill_workspace(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
    ) -> None:
        try:
            workspace_manager = current_workspace_manager()
        except AssertionError:
            return
        workspace_manager.get_simultaneous(
            *cls._prefill_workspace_reservation_specs(layer)
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same graph-stable
            # workspace shapes _forward_prefill can use, but skip real kernels.
            self._reserve_prefill_workspace(self)
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    @classmethod
    def _forward_sparse_mla_swa_decode_triton(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
        q: torch.Tensor,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        if not mtp_decode:
            fp8ds_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                block_size=swa_metadata.block_size,
                candidate_offset=0,
                num_candidates=max_swa_len,
                scale=layer.scale,
                attn_sink=layer.attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=layer.n_local_heads,
            )
            if output.shape[1] > layer.n_local_heads:
                output[:, layer.n_local_heads :].zero_()
            return

        (
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads, q.shape[-1]), torch.float32),
        )
        swa_max_score.fill_(float("-inf"))
        swa_denom.zero_()
        swa_acc.zero_()
        accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
            q=q,
            k_cache=swa_k_cache,
            slot_ids=swa_indices,
            lens=swa_lens,
            block_size=swa_metadata.block_size,
            scale=layer.scale,
            max_score=swa_max_score,
            denom=swa_denom,
            acc=swa_acc,
            head_block_size=head_block_size,
        )
        finish_sparse_mla_attention_with_sink(
            swa_max_score,
            swa_denom,
            swa_acc,
            layer.attn_sink,
            output=output,
        )
        if output.shape[1] > layer.n_local_heads:
            output[:, layer.n_local_heads :].zero_()

    @classmethod
    def _forward_sparse_mla_compressed_decode_triton(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor,
        swa_k_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_lens: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata,
        output: torch.Tensor,
    ) -> None:
        if layer.compress_ratio not in (4, 128):
            raise NotImplementedError(
                "Triton sparse MLA compressed decode currently supports "
                f"compress_ratio=4 or 128, got {layer.compress_ratio}"
            )

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        mtp_decode = num_decode_tokens != num_decodes

        max_swa_len = swa_metadata.decode_swa_indices.shape[-1]
        compressed_block_size = attn_metadata.block_size // layer.compress_ratio
        compressed_topk = topk_indices.shape[-1]
        topk_chunk_size = min(
            compressed_topk,
            triton_sparse_mla_topk_chunk_size(),
        )
        compressed_slot_ids = topk_indices[:, 0, :]
        swa_lens = swa_metadata.decode_swa_lens[:num_decode_tokens]
        swa_indices = swa_metadata.decode_swa_indices[:num_decode_tokens]
        head_block_size = sparse_mla_decode_head_block_size(num_decode_tokens)
        if (
            compressed_topk <= topk_chunk_size
            and triton_sparse_mla_matmul_decode_enabled()
        ):
            total_candidates = compressed_topk + max_swa_len
            (
                combined_kv,
                valid_tokens,
                score_buffer,
            ) = current_workspace_manager().get_simultaneous(
                ((num_decode_tokens, total_candidates, q.shape[-1]), torch.bfloat16),
                ((num_decode_tokens, total_candidates), torch.bool),
                (
                    (num_decode_tokens, layer.n_local_heads, total_candidates),
                    torch.bfloat16,
                ),
            )
            if mtp_decode:
                dequantize_global_slots_k_cache(
                    combined_kv[:, :compressed_topk],
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                )
                dequantize_global_slots_k_cache(
                    combined_kv[:, compressed_topk:],
                    swa_k_cache,
                    swa_indices,
                    swa_metadata.block_size,
                )
            else:
                dequantize_combined_sparse_mla_decode_kv(
                    combined_kv,
                    compressed_k_cache,
                    compressed_slot_ids,
                    compressed_block_size,
                    swa_k_cache,
                    swa_metadata.seq_lens[:num_decodes],
                    swa_lens,
                    swa_metadata.block_table[:num_decodes],
                    swa_metadata.block_size,
                )

            build_combined_sparse_mla_decode_valid_mask(
                valid_tokens,
                compressed_slot_ids,
                topk_lens,
                swa_lens,
            )
            use_dot_finish = num_decode_tokens <= 16
            matmul_sparse_mla_attention_with_sink(
                q=q,
                kv=combined_kv,
                valid_tokens=valid_tokens,
                scale=layer.scale,
                attn_sink=layer.attn_sink,
                output=output,
                num_heads=layer.n_local_heads,
                score_buffer=score_buffer,
                value_block_size=512 if use_dot_finish else 256,
                candidate_block_size=128 if use_dot_finish else None,
            )
            return

        if not mtp_decode and compressed_topk <= topk_chunk_size:
            fp8ds_global_paged_sparse_mla_attention_with_sink_multihead(
                q=q,
                compressed_k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids,
                topk_lens=topk_lens,
                compressed_block_size=compressed_block_size,
                swa_k_cache=swa_k_cache,
                seq_lens=swa_metadata.seq_lens[:num_decodes],
                gather_lens=swa_lens,
                block_table=swa_metadata.block_table[:num_decodes],
                swa_block_size=swa_metadata.block_size,
                num_compressed_candidates=compressed_topk,
                num_swa_candidates=max_swa_len,
                scale=layer.scale,
                attn_sink=layer.attn_sink,
                output=output,
                head_block_size=head_block_size,
                num_heads=layer.n_local_heads,
            )
            if output.shape[1] > layer.n_local_heads:
                output[:, layer.n_local_heads :].zero_()
            return

        (
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
        ) = current_workspace_manager().get_simultaneous(
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads, q.shape[-1]), torch.float32),
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads), torch.float32),
            ((num_decode_tokens, layer.n_local_heads, q.shape[-1]), torch.float32),
        )
        comp_max_score.fill_(float("-inf"))
        comp_denom.zero_()
        comp_acc.zero_()
        swa_max_score.fill_(float("-inf"))
        swa_denom.zero_()
        swa_acc.zero_()

        for chunk_start in range(0, compressed_topk, topk_chunk_size):
            chunk_end = min(chunk_start + topk_chunk_size, compressed_topk)
            accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
                q=q,
                k_cache=compressed_k_cache,
                slot_ids=compressed_slot_ids[:, chunk_start:chunk_end],
                lens=topk_lens,
                block_size=compressed_block_size,
                candidate_offset=chunk_start,
                scale=layer.scale,
                max_score=comp_max_score,
                denom=comp_denom,
                acc=comp_acc,
                head_block_size=head_block_size,
            )
        accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
            q=q,
            k_cache=swa_k_cache,
            slot_ids=swa_indices,
            lens=swa_lens,
            block_size=swa_metadata.block_size,
            scale=layer.scale,
            max_score=swa_max_score,
            denom=swa_denom,
            acc=swa_acc,
            head_block_size=head_block_size,
        )
        finish_two_sparse_mla_attention_states_with_sink(
            comp_max_score,
            comp_denom,
            comp_acc,
            swa_max_score,
            swa_denom,
            swa_acc,
            layer.attn_sink,
            output=output,
        )
        if output.shape[1] > layer.n_local_heads:
            output[:, layer.n_local_heads :].zero_()

    @classmethod
    def _forward_sparse_mla_prefill_triton(
        cls,
        layer: "DeepseekV4FlashMLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        combined_indices: torch.Tensor,
        combined_lens: torch.Tensor,
        output: torch.Tensor,
        state_buffers: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        kv_flat = kv.reshape(-1, q.shape[-1])
        topk_chunk_size = triton_sparse_mla_prefill_topk_chunk_size(
            combined_topk_size=combined_indices.shape[-1],
            compress_ratio=int(layer.compress_ratio),
            request_count=kv.shape[0],
        )
        query_chunk_size = min(
            q.shape[0],
            triton_sparse_mla_query_chunk_size(),
        )
        if state_buffers is None:
            (
                max_score_buffer,
                denom_buffer,
                output_buffer,
            ) = current_workspace_manager().get_simultaneous(
                ((query_chunk_size, layer.n_local_heads), torch.float32),
                ((query_chunk_size, layer.n_local_heads), torch.float32),
                ((query_chunk_size, layer.n_local_heads, q.shape[-1]), torch.float32),
            )
        else:
            max_score_buffer, denom_buffer, output_buffer = state_buffers[:3]
        indexed_d512_scores = None
        indexed_d512_chunked_buffers = None
        if (
            state_buffers is not None
            and envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL
            and layer.compress_ratio in (4, 128)
            and q.shape[-1] == 512
            and kv.shape[0] == 1
            and _is_indexed_d512_split_topk(combined_indices.shape[-1])
            and len(state_buffers) == 4
        ):
            indexed_d512_scores = state_buffers[3]
        elif (
            state_buffers is not None
            and envs.VLLM_DEEPSEEK_V4_INDEXED_D512_CHUNKED_PREFILL
            and envs.VLLM_DEEPSEEK_V4_INDEXED_D512_SPLIT_PREFILL
            and layer.compress_ratio in (4, 128)
            and q.shape[-1] == 512
            and kv.shape[0] == 1
            and combined_indices.shape[-1] > _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK
            and len(state_buffers) == 7
        ):
            indexed_d512_chunked_buffers = state_buffers[3:7]

        for token_start in range(0, q.shape[0], query_chunk_size):
            token_end = min(token_start + query_chunk_size, q.shape[0])
            q_chunk = q[token_start:token_end]
            indices_chunk_full = combined_indices[token_start:token_end]
            lens_chunk = combined_lens[token_start:token_end]
            num_tokens = token_end - token_start
            max_score = max_score_buffer[:num_tokens]
            denom = denom_buffer[:num_tokens]
            subset_acc = output_buffer[:num_tokens]
            can_use_indexed_d512_scores = (
                indexed_d512_scores is not None
                and indexed_d512_scores.shape[0] >= num_tokens
                and indexed_d512_scores.shape[2] >= combined_indices.shape[-1]
            )
            can_use_indexed_d512_chunked = (
                indexed_d512_chunked_buffers is not None
                and indexed_d512_chunked_buffers[0].shape[0] >= num_tokens
            )
            if can_use_indexed_d512_scores:
                assert indexed_d512_scores is not None
                accumulate_indexed_d512_split_sparse_mla_attention(
                    q=q_chunk,
                    kv_flat=kv_flat,
                    indices=indices_chunk_full,
                    lens=lens_chunk,
                    scale=layer.scale,
                    max_score=max_score,
                    denom=denom,
                    acc=subset_acc,
                    scores=indexed_d512_scores[
                        :num_tokens, :, : combined_indices.shape[-1]
                    ],
                )
            elif can_use_indexed_d512_chunked:
                assert indexed_d512_chunked_buffers is not None
                (
                    indexed_d512_scores,
                    chunk_max_score,
                    chunk_denom,
                    chunk_acc,
                ) = indexed_d512_chunked_buffers
                accumulate_indexed_d512_chunked_sparse_mla_attention(
                    q=q_chunk,
                    kv_flat=kv_flat,
                    indices=indices_chunk_full,
                    lens=lens_chunk,
                    scale=layer.scale,
                    max_score=max_score,
                    denom=denom,
                    acc=subset_acc,
                    scores=indexed_d512_scores[:num_tokens],
                    chunk_max_score=chunk_max_score[:num_tokens],
                    chunk_denom=chunk_denom[:num_tokens],
                    chunk_acc=chunk_acc[:num_tokens],
                )
            else:
                max_score.fill_(float("-inf"))
                denom.zero_()
                subset_acc.zero_()

                for index_start in range(
                    0, combined_indices.shape[-1], topk_chunk_size
                ):
                    index_end = min(
                        index_start + topk_chunk_size,
                        combined_indices.shape[-1],
                    )
                    accumulate_indexed_sparse_mla_attention_chunk(
                        q=q_chunk,
                        kv_flat=kv_flat,
                        indices=indices_chunk_full[:, index_start:index_end],
                        lens=lens_chunk,
                        candidate_offset=index_start,
                        scale=layer.scale,
                        max_score=max_score,
                        denom=denom,
                        acc=subset_acc,
                    )

            finish_sparse_mla_attention_with_sink(
                max_score,
                denom,
                subset_acc,
                layer.attn_sink,
                output=output[token_start:token_end],
            )
            if output.shape[1] > layer.n_local_heads:
                output[token_start:token_end, layer.n_local_heads :].zero_()

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        compressed_k_cache = kv_cache
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        if is_triton_sparse_mla_enabled(q.device):
            if swa_only:
                self._forward_sparse_mla_swa_decode_triton(
                    layer=self,
                    q=q,
                    swa_k_cache=self.swa_cache_layer.kv_cache,
                    swa_metadata=swa_metadata,
                    output=output,
                )
                return
            if self.compress_ratio in (4, 128):
                assert compressed_k_cache is not None
                assert attn_metadata is not None
                assert topk_indices is not None
                assert topk_lens is not None
                self._forward_sparse_mla_compressed_decode_triton(
                    layer=self,
                    q=q,
                    compressed_k_cache=compressed_k_cache,
                    swa_k_cache=self.swa_cache_layer.kv_cache,
                    topk_indices=topk_indices,
                    topk_lens=topk_lens,
                    swa_metadata=swa_metadata,
                    attn_metadata=attn_metadata,
                    output=output,
                )
                return

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        seq_lens_cpu = swa_metadata.prefill_seq_lens_cpu
        gather_lens_cpu = swa_metadata.prefill_gather_lens_cpu
        assert seq_lens is not None
        assert gather_lens is not None
        assert seq_lens_cpu is not None
        assert gather_lens_cpu is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = int((seq_lens_cpu // self.compress_ratio).max().item())
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + int(gather_lens_cpu.max().item())
        chunk_size_const = self.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const
        max_query_chunk_tokens = 0
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            max_query_chunk_tokens = max(
                max_query_chunk_tokens, int(query_end - query_start)
            )
        combined_topk = sparse_prefill_combined_topk_size(top_k, self.window_size)

        workspace_manager = current_workspace_manager()
        triton_sparse_mla_enabled = is_triton_sparse_mla_enabled(q.device)
        indexed_d512_split_prefill = False
        indexed_d512_chunked_prefill = False
        if triton_sparse_mla_enabled:
            query_chunk_size = min(
                max_query_chunk_tokens,
                triton_sparse_mla_query_chunk_size(),
            )
            indexed_d512_split_prefill = _use_indexed_d512_split_prefill(
                compress_ratio=int(self.compress_ratio),
                head_dim=int(self.head_dim),
                num_prefills=int(num_prefills),
                combined_topk=int(combined_topk),
                max_prefill_seq_len=int(seq_lens_cpu.max().item()),
                swa_only=swa_only,
            )
            if not indexed_d512_split_prefill:
                indexed_d512_chunked_prefill = _use_indexed_d512_chunked_prefill(
                    compress_ratio=int(self.compress_ratio),
                    head_dim=int(self.head_dim),
                    num_prefills=int(num_prefills),
                    combined_topk=int(combined_topk),
                    max_prefill_seq_len=int(seq_lens_cpu.max().item()),
                    swa_only=swa_only,
                )
            extra_specs: list[tuple[tuple[int, ...], torch.dtype]] = []
            if indexed_d512_split_prefill:
                extra_specs.append(
                    (
                        (query_chunk_size, self.n_local_heads, combined_topk),
                        torch.float32,
                    )
                )
            elif indexed_d512_chunked_prefill:
                chunked_score_width = min(
                    combined_topk,
                    _INDEXED_D512_SPLIT_PREFILL_MAX_TOPK,
                )
                extra_specs.extend(
                    (
                        (
                            (query_chunk_size, self.n_local_heads, chunked_score_width),
                            torch.float32,
                        ),
                        ((query_chunk_size, self.n_local_heads), torch.float32),
                        ((query_chunk_size, self.n_local_heads), torch.float32),
                        (
                            (query_chunk_size, self.n_local_heads, q.shape[-1]),
                            torch.float32,
                        ),
                    )
                )
            (
                kv,
                combined_indices_buffer,
                combined_lens_buffer,
                max_score_buffer,
                denom_buffer,
                output_buffer,
                *extra_state_buffers,
            ) = workspace_manager.get_simultaneous(
                ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
                ((max_query_chunk_tokens, combined_topk), torch.int32),
                ((max_query_chunk_tokens,), torch.int32),
                ((query_chunk_size, self.n_local_heads), torch.float32),
                ((query_chunk_size, self.n_local_heads), torch.float32),
                ((query_chunk_size, self.n_local_heads, q.shape[-1]), torch.float32),
                *extra_specs,
            )
            prefill_state_buffers = (
                max_score_buffer,
                denom_buffer,
                output_buffer,
                *extra_state_buffers,
            )
        else:
            (
                kv,
                combined_indices_buffer,
                combined_lens_buffer,
            ) = workspace_manager.get_simultaneous(
                ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
                ((max_query_chunk_tokens, combined_topk), torch.int32),
                ((max_query_chunk_tokens,), torch.int32),
            )
            prefill_state_buffers = None
        for chunk_idx in range(num_chunks):
            write_stats = _sparse_mla_prefill_stats_enabled()
            stage_timer = (
                _SparseMLAPrefillStageTimer()
                if _sparse_mla_prefill_stage_timing_enabled()
                else None
            )
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                with (
                    stage_timer.stage("gather_compressed_kv")
                    if stage_timer is not None
                    else nullcontext()
                ):
                    dequantize_and_gather_k_cache(
                        kv[:chunk_size],
                        compressed_k_cache,
                        seq_lens=(
                            seq_lens[chunk_start:chunk_end] // self.compress_ratio
                        ),
                        gather_lens=None,
                        block_table=block_table[chunk_start:chunk_end],
                        block_size=attn_metadata.block_size // self.compress_ratio,
                        offset=0,
                    )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            with (
                stage_timer.stage("gather_swa_kv")
                if stage_timer is not None
                else nullcontext()
            ):
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    swa_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end],
                    gather_lens=gather_lens[chunk_start:chunk_end],
                    block_table=swa_block_table[chunk_start:chunk_end],
                    block_size=swa_metadata.block_size,
                    offset=N,
                )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            with (
                stage_timer.stage("combine_indices")
                if stage_timer is not None
                else nullcontext()
            ):
                combined_indices, combined_lens = combine_topk_swa_indices(
                    topk_indices[query_start:query_end],
                    query_start_loc[
                        num_decodes + chunk_start : num_decodes + chunk_end + 1
                    ],
                    seq_lens[chunk_start:chunk_end],
                    gather_lens[chunk_start:chunk_end],
                    self.window_size,
                    self.compress_ratio,
                    top_k,
                    M,
                    N,
                    combined_indices=combined_indices_buffer,
                    combined_lens=combined_lens_buffer,
                )
            if triton_sparse_mla_enabled:
                with (
                    stage_timer.stage("sparse_accumulate")
                    if stage_timer is not None
                    else nullcontext()
                ):
                    self._forward_sparse_mla_prefill_triton(
                        self,
                        q=q[query_start:query_end],
                        kv=kv[:chunk_size],
                        combined_indices=combined_indices,
                        combined_lens=combined_lens,
                        output=output[query_start:query_end],
                        state_buffers=prefill_state_buffers,
                    )
            else:
                with (
                    stage_timer.stage("sparse_accumulate")
                    if stage_timer is not None
                    else nullcontext()
                ):
                    flash_mla_sparse_fwd(
                        q=q[query_start:query_end],
                        kv=kv.view(-1, 1, q.shape[-1]),
                        indices=combined_indices.unsqueeze(1),
                        sm_scale=self.scale,
                        attn_sink=self.attn_sink,
                        topk_length=combined_lens,
                        out=output[query_start:query_end],
                    )
            if write_stats:
                compressed_visits, swa_visits = (
                    _sparse_mla_prefill_candidate_region_visits(
                        query_start_loc_cpu=query_start_loc_cpu,
                        seq_lens_cpu=seq_lens_cpu,
                        num_decodes=num_decodes,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        top_k=top_k,
                        compress_ratio=self.compress_ratio,
                        window_size=self.window_size,
                    )
                )
                _write_sparse_mla_prefill_stats(
                    layer_type=self._prefill_stats_layer_type(
                        triton_sparse_mla_enabled=triton_sparse_mla_enabled,
                        indexed_d512_split_prefill=indexed_d512_split_prefill,
                        indexed_d512_chunked_prefill=(
                            indexed_d512_chunked_prefill
                        ),
                    ),
                    layer_prefix=self.prefix,
                    compress_ratio=self.compress_ratio,
                    num_prefills=chunk_size,
                    query_tokens=int(query_end - query_start),
                    combined_topk=combined_indices.shape[-1],
                    combined_lens=combined_lens,
                    combined_indices=combined_indices,
                    gather_region_size=M,
                    swa_region_offset=N,
                    compressed_region_width=top_k,
                    swa_region_width=self.window_size,
                    compressed_candidate_visits=compressed_visits,
                    swa_candidate_visits=swa_visits,
                    stage_timings_ms=(
                        stage_timer.elapsed_ms()
                        if stage_timer is not None
                        else None
                    ),
                )
