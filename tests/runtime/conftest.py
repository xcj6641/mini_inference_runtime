import pytest
from app.runtime.batch_builder import BatchBuilder
from app.runtime.kv_block_manager import KVBlockManager
from app.runtime.paged_kv_cache import PagedKVCache
import torch

from app.runtime.pytorch_model_runner import PyTorchModelRunner
from app.runtime.continuous_scheduler import ContinuousScheduler
from tests.runtime.fake_runner import FakeRunner


@pytest.fixture(scope="session")
def real_runner() -> PyTorchModelRunner:
    return PyTorchModelRunner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        dtype=torch.float16,
    )

@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def block_manager() -> KVBlockManager:
    return KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

@pytest.fixture
def batch_builder() -> BatchBuilder:
    return BatchBuilder()

@pytest.fixture
def paged_kv_cache() -> PagedKVCache:
    return PagedKVCache(
        num_layers=1,
        num_blocks=32,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

@pytest.fixture
def scheduler(
    real_runner,
    batch_builder,
    block_manager,
    paged_kv_cache
) -> ContinuousScheduler:
    
    return ContinuousScheduler(
        runner=real_runner,
        batch_builder=batch_builder,
        block_manager=block_manager,
        max_prefill_batch_size=4,
        max_decode_batch_size=4,
        paged_kv_cache=paged_kv_cache
    )
