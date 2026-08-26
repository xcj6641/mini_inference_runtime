import pytest
from app.runtime.batch_builder import BatchBuilder
from app.runtime.kv_block_manager import KVBlockManager
from app.runtime.paged_kv_cache import PagedKVCache
from app.runtime.prefix_cache import PrefixCache
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
def fake_runner_all_dim(
    paged_kv_cache: PagedKVCache,
) -> FakeRunner:
    return FakeRunner(
        num_layers=paged_kv_cache.num_layers,
        num_kv_heads=paged_kv_cache.num_kv_heads,
        head_dim=paged_kv_cache.head_dim,
    )


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
def paged_kv_cache_qwen(
        real_runner: PyTorchModelRunner,
)->PagedKVCache:
    
    config = real_runner.model.config
    return PagedKVCache(
        num_layers=config.num_hidden_layers,
        num_blocks=32,
        num_kv_heads=config.num_key_value_heads,
        block_size=4,
        head_dim=(
            config.hidden_size // config.num_attention_heads
        ),
        dtype=real_runner.dtype,
        device=real_runner.device,
    )

@pytest.fixture
def make_paged_kv_cache_qwen(
    real_runner,
):
    def _make():
        config = real_runner.model.config

        return PagedKVCache(
            num_layers=config.num_hidden_layers,
            num_blocks=32,
            num_kv_heads=config.num_key_value_heads,
            block_size=4,
            head_dim=(
                config.hidden_size
                // config.num_attention_heads
            ),
            dtype=real_runner.dtype,
            device=real_runner.device,
        )

    return _make

@pytest.fixture
def prefix_cache(
    block_manager: KVBlockManager
) -> PrefixCache:
    return PrefixCache(
        block_manager=block_manager,
    )

@pytest.fixture
def scheduler(
    real_runner,
    batch_builder,
    block_manager: KVBlockManager,
    paged_kv_cache_qwen: PagedKVCache,
    prefix_cache : PrefixCache,
) -> ContinuousScheduler:
    
    return ContinuousScheduler(
        runner=real_runner,
        batch_builder=batch_builder,
        block_manager=block_manager,
        max_prefill_batch_size=4,
        max_decode_batch_size=4,
        paged_kv_cache=paged_kv_cache_qwen,
        prefix_cache=prefix_cache,
    )

@pytest.fixture
def scheduler_fake_runner(
    fake_runner_all_dim,
    batch_builder,
    block_manager: KVBlockManager,
    paged_kv_cache: PagedKVCache,
    prefix_cache : PrefixCache,
) -> ContinuousScheduler:
    
    return ContinuousScheduler(
        runner=fake_runner_all_dim,
        batch_builder=batch_builder,
        block_manager=block_manager,
        max_prefill_batch_size=4,
        max_decode_batch_size=4,
        paged_kv_cache=paged_kv_cache,
        prefix_cache=prefix_cache,
    )
