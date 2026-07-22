import pytest
from app.runtime.pytorch_model_runner import PyTorchModelRunner
import torch

import gc

from app.runtime.device_capabilities import (
    detect_device_capabilities,
)


def test_cpu_capabilities() -> None:
    capabilities = detect_device_capabilities(
        "cpu"
    )

    assert capabilities.compute_capability is None
    assert capabilities.supports_fp16 is False
    assert capabilities.supports_bf16 is False
    assert (
        capabilities.hardware_fp8_candidate
        is False
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)
def test_cuda_capabilities_match_pytorch() -> None:
    capabilities = detect_device_capabilities(
        "cuda"
    )

    assert capabilities.compute_capability == (
        torch.cuda.get_device_capability()
    )

    assert capabilities.supports_fp16 is True

    assert capabilities.supports_bf16 == (
        torch.cuda.is_bf16_supported()
    )


from app.runtime.pytorch_model_runner import PyTorchModelRunner


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.integration
def test_runner_uses_requested_dtype(
    dtype: torch.dtype,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("Current CUDA device does not support BF16")

    runner = PyTorchModelRunner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        dtype=dtype,
    )

    try:
        parameter = next(
            parameter
            for parameter in runner.model.parameters()
            if parameter.is_floating_point()
        )

        assert parameter.dtype == dtype

    finally:
        del runner
        gc.collect()
        torch.cuda.empty_cache()