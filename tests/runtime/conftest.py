import pytest
import torch

from app.runtime.pytorch_model_runner import PyTorchModelRunner

@pytest.fixture(scope="session")
def real_runner() -> PyTorchModelRunner:
    return PyTorchModelRunner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        dtype=torch.float16,

    )

