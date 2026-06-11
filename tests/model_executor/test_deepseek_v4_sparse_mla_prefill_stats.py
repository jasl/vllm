# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import torch

from vllm.models.deepseek_v4.nvidia import flashmla as flashmla_mod


def _set_stats_env(monkeypatch, path: str) -> None:
    monkeypatch.setattr(
        flashmla_mod.envs,
        "VLLM_DEEPSEEK_V4_SPARSE_MLA_STATS_PATH",
        path,
    )
    monkeypatch.setattr(
        flashmla_mod.envs,
        "VLLM_DEEPSEEK_V4_SPARSE_MLA_STATS_OVERLAP_ROWS",
        2,
    )
    monkeypatch.setattr(
        flashmla_mod.envs,
        "VLLM_DEEPSEEK_V4_SPARSE_MLA_STATS_STAGE_TIMING",
        False,
    )


def test_sparse_mla_prefill_stats_disable_context_blocks_writer(
    tmp_path, monkeypatch
) -> None:
    _set_stats_env(monkeypatch, str(tmp_path))

    with flashmla_mod._disable_sparse_mla_prefill_stats():
        flashmla_mod._write_sparse_mla_prefill_stats(
            layer_type="mla_prefill_chunk",
            layer_prefix="model.layers.0.self_attn",
            compress_ratio=128,
            num_prefills=1,
            query_tokens=1,
            combined_topk=2,
            combined_lens=torch.tensor([2], dtype=torch.int32),
        )

    assert not list(tmp_path.glob("*.jsonl"))


def test_sparse_mla_prefill_stats_writer_emits_region_and_overlap(
    tmp_path, monkeypatch
) -> None:
    _set_stats_env(monkeypatch, str(tmp_path))

    flashmla_mod._write_sparse_mla_prefill_stats(
        layer_type="mla_prefill_chunk",
        layer_prefix="model.layers.0.self_attn",
        compress_ratio=128,
        num_prefills=1,
        query_tokens=2,
        combined_topk=4,
        combined_lens=torch.tensor([4, 3], dtype=torch.int32),
        combined_indices=torch.tensor(
            [
                [0, 1, 4, 5],
                [0, 2, 4, 7],
            ],
            dtype=torch.int32,
        ),
        gather_region_size=8,
        swa_region_offset=4,
        compressed_region_width=2,
        swa_region_width=2,
        compressed_candidate_visits=3,
        swa_candidate_visits=4,
        stage_timings_ms={"combine_indices": 1.25},
    )

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(tmp_path.glob("*.jsonl"))
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "deepseek_v4_sparse_mla_prefill_stats"
    assert row["candidate_slots"] == 8
    assert row["effective_candidate_visits"] == 7
    assert row["candidate_region_work"]["compressed"]["effective_candidate_visits"] == 3
    assert row["candidate_region_work"]["swa"]["effective_candidate_visits"] == 4
    assert row["stage_timings_ms"] == {"combine_indices": 1.25}
    assert row["candidate_overlap"]["groups"]["2"]["unique_candidates"] == 5
    assert row["candidate_region_overlap"]["compressed"]["2"][
        "unique_to_valid_ratio"
    ] == 0.75
    assert row["candidate_region_overlap"]["swa"]["2"][
        "unique_to_valid_ratio"
    ] == 2 / 3
