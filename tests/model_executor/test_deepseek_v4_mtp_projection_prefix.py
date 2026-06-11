# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.models.deepseek_v4.amd.mtp import (
    _mtp_projection_prefix as amd_mtp_projection_prefix,
)
from vllm.models.deepseek_v4.nvidia.mtp import (
    _mtp_projection_prefix as nvidia_mtp_projection_prefix,
)


class _QuantConfig:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


def test_deepseek_v4_mtp_projection_prefix_for_compressed_tensors():
    quant_config = _QuantConfig("compressed-tensors")

    for projection_prefix in (amd_mtp_projection_prefix, nvidia_mtp_projection_prefix):
        assert (
            projection_prefix(quant_config, "model.layers.61", "e_proj")
            == "model.layers.61.e_proj"
        )
        assert (
            projection_prefix(quant_config, "model.layers.61", "h_proj")
            == "model.layers.61.h_proj"
        )


def test_deepseek_v4_mtp_projection_prefix_keeps_fp8_legacy_path():
    for quant_config in (_QuantConfig("fp8"), None):
        for projection_prefix in (
            amd_mtp_projection_prefix,
            nvidia_mtp_projection_prefix,
        ):
            assert projection_prefix(quant_config, "model.layers.61", "e_proj") == ""
            assert projection_prefix(quant_config, "model.layers.61", "h_proj") == ""
