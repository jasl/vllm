from types import SimpleNamespace

from vllm import envs
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.quantization import mxfp4


def test_mxfp4_postload_memory_debug_env_defaults_to_off(monkeypatch):
    monkeypatch.delenv("VLLM_MXFP4_MOE_POST_LOAD_MEMORY_DEBUG", raising=False)
    envs.disable_envs_cache()

    assert envs.VLLM_MXFP4_MOE_POST_LOAD_MEMORY_DEBUG is False


def test_mxfp4_postload_memory_debug_logs_gated_snapshots(
    monkeypatch,
):
    layer = SimpleNamespace(layer_name="model.layers.3.mlp.experts.routed_experts")
    messages = []
    monkeypatch.setattr(
        mxfp4,
        "_get_mxfp4_moe_post_load_memory_stats",
        lambda: {
            "allocated_gib": 1.0,
            "reserved_gib": 2.0,
            "free_gib": 3.0,
            "total_gib": 4.0,
            "host_available_gib": 5.0,
            "host_total_gib": 6.0,
        },
    )
    monkeypatch.setattr(
        mxfp4.logger,
        "info",
        lambda message, *args: messages.append(message % args),
    )

    monkeypatch.setenv("VLLM_MXFP4_MOE_POST_LOAD_MEMORY_DEBUG", "0")
    envs.disable_envs_cache()
    mxfp4._log_mxfp4_moe_post_load_memory(layer, "before_setup")
    assert not messages

    monkeypatch.setenv("VLLM_MXFP4_MOE_POST_LOAD_MEMORY_DEBUG", "1")
    envs.disable_envs_cache()
    mxfp4._log_mxfp4_moe_post_load_memory(layer, "before_setup")

    assert "stage=before_setup" in messages[0]
    assert "layer=model.layers.3.mlp.experts.routed_experts" in messages[0]
    assert "allocated=1.00GiB" in messages[0]
    assert "host_available=5.00GiB" in messages[0]


def test_mxfp4_postload_process_weights_records_and_releases_boundaries(
    monkeypatch,
):
    layer = SimpleNamespace(
        layer_name="model.layers.3.mlp.experts.routed_experts",
        w13_weight=object(),
        w2_weight=object(),
        w13_weight_scale=object(),
        w2_weight_scale=object(),
    )
    method = object.__new__(mxfp4.Mxfp4MoEMethod)
    method.mxfp4_backend = Mxfp4MoeBackend.MARLIN

    events = []
    monkeypatch.setattr(
        mxfp4,
        "_log_mxfp4_moe_post_load_memory",
        lambda layer, stage: events.append(f"log:{stage}"),
    )
    monkeypatch.setattr(
        mxfp4,
        "_maybe_release_mxfp4_moe_post_load_cache",
        lambda stage, *, force=False: events.append(f"release:{stage}:{force}"),
    )
    monkeypatch.setattr(
        method,
        "_setup_kernel",
        lambda *args: events.append("setup"),
    )

    method.process_weights_after_loading(layer)

    assert events == [
        "log:before_setup",
        "release:before_setup:False",
        "setup",
        "log:after_setup",
        "release:after_setup:True",
        "log:after_cache_release",
    ]
