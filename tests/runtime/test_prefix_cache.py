import torch
from app.runtime.request import Request, RequestState

from app.runtime.prefix_cache import PrefixCache
from app.runtime.paged_kv_cache import PagedKVCache
from app.runtime.kv_block_manager import KVBlockManager
from app.runtime.continuous_scheduler import ContinuousScheduler
from app.runtime.batch_builder import BatchBuilder


def test_prefix_cache_miss_returns_none(block_manager) -> None:
    cache = PrefixCache(block_manager)

    result = cache.lookup([10, 11, 12, 13])

    assert result is None


def test_prefix_cache_insert_then_lookup(
    block_manager,
) -> None:
    request = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13],
    )

    block_manager.ensure_capacity(
        request=request,
        total_tokens=4,
    )

    block_id = request.block_table[0]

    cache = PrefixCache(block_manager)

    cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[block_id],
    )

    result = cache.lookup(
        [10, 11, 12, 13]
    )

    assert result is not None
    assert result.token_ids == (
        10, 11, 12, 13
    )
    assert result.block_ids == (
        block_id,
    )


def test_prefix_cache_different_prefix_misses(
    block_manager,
) -> None:
    request = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13],
    )

    block_manager.ensure_capacity(
        request=request,
        total_tokens=4,
    )

    block_id = request.block_table[0]

    cache = PrefixCache(block_manager)

    cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[block_id],
    )

    result = cache.lookup(
        [10, 11, 12, 99]
    )

    assert result is None


def test_prefix_cache_same_prefix_hits(
    block_manager,
) -> None:
    request = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13],
    )

    block_manager.ensure_capacity(
        request=request,
        total_tokens=4,
    )

    block_id = request.block_table[0]

    cache = PrefixCache(block_manager)

    cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[block_id],
    )

    result = cache.lookup(
        [10, 11, 12, 13]
    )

    assert result is not None
    assert result.block_ids == (
        block_id,
    )

def test_cacheable_prefix_uses_only_full_blocks() -> None:
    assert (
        PrefixCache.get_cacheable_prefix_length(
            num_tokens=6,
            block_size=4,
        )
        == 4
    )


def test_cacheable_prefix_can_include_multiple_full_blocks() -> None:
    assert (
        PrefixCache.get_cacheable_prefix_length(
            num_tokens=8,
            block_size=4,
        )
        == 8
    )


def test_cacheable_prefix_zero_when_no_full_block() -> None:
    assert (
        PrefixCache.get_cacheable_prefix_length(
            num_tokens=3,
            block_size=4,
        )
        == 0
    )


# test paged kv cache integration with continuous scheduler
def make_request(
    request_id: str,
    max_new_tokens: int = 4,
    input_ids: list[int] | None = None,
) -> Request:
    if input_ids is None:
        input_ids = [1, 2, 3]

    return Request(
        request_id=request_id,
        input_ids=list(input_ids),
        max_new_tokens=max_new_tokens,
    )


def test_retain_block_increments_ref_count(
    block_manager,
) -> None:
    request = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
    )

    block_manager.ensure_capacity(
        request=request,
        total_tokens=4,
    )

    block_id = request.block_table[0]

    assert (
        block_manager.get_ref_count(block_id)
        == 1
    )

    block_manager.retain_blocks([block_id])

    assert (
        block_manager.get_ref_count(block_id)
        == 2
    )

def test_release_shared_block_does_not_free_it(
    block_manager,
) -> None:
    request = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
    )

    block_manager.ensure_capacity(
        request=request,
        total_tokens=4,
    )

    block_id = request.block_table[0]

    block_manager.retain_blocks([block_id])

    block_manager.release_blocks([block_id])

    assert (
        block_manager.get_ref_count(block_id)
        == 1
    )

    assert block_id not in block_manager._free_blocks

def test_final_release_returns_block_to_free_list(
    block_manager,
) -> None:
    
    request = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
    )

    # allocate
    block_manager.ensure_capacity(
        request=request,
        total_tokens=4,
    )

    block_id = request.block_table[0]

    assert (
        block_manager.get_ref_count(block_id)
        == 1
    )

    # retain
    block_manager.retain_blocks([block_id])
    assert(
        block_manager.get_ref_count(block_id)
        == 2
    )

    # release
    block_manager.release_blocks([block_id])
    assert (
        block_manager.get_ref_count(block_id)
        == 1
    )
    # release
    block_manager.release_blocks([block_id])
    assert (
        block_manager.get_ref_count(block_id)
        == 0
    )

# test evicts
def test_cache_eviction_releases_final_block_reference(block_manager)-> None:
        initial_free_blocks = len(block_manager._free_blocks)
        request = make_request(
            request_id="A",
            input_ids=[1, 2, 3, 4],
        )
    
        # A allocates block 2
        # ref = 1
        block_manager.ensure_capacity(
            request=request,
            total_tokens=4,
        )
    
        block_id = request.block_table[0]
    
        assert (
            block_manager.get_ref_count(block_id)
            == 1
        )

        # PrefixCache inserts block 2
        # ref = 2
        prefix_cache = PrefixCache(block_manager)
        prefix_cache.insert(
            token_ids=[1,2,3,4],
            block_ids=[block_id],
        )
        assert(
            block_manager.get_ref_count(block_id)
            == 2
        )

        # A releases block 2
        # ref = 1
        block_manager.release_blocks([block_id])
        assert(
            block_manager.get_ref_count(block_id) == 1
        )
        assert block_id not in block_manager._free_blocks

        # cache evicts
        # ref = 0
        # block returned to free list
        prefix_cache.evict(
            token_ids = [1,2,3,4]
        )
        assert(
            block_manager.get_ref_count(block_id) == 0
        )
        assert block_id in block_manager._free_blocks
        assert block_id not in block_manager._ref_counts

        assert(len(block_manager._free_blocks) == initial_free_blocks)

def test_second_request_reuses_cached_prefix_block(
    block_manager,
) -> None:
    request_a = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13, 14, 15],
    )

    block_manager.ensure_capacity(
        request=request_a,
        total_tokens=6,
    )

    # A owns two blocks:
    #
    # block 0 -> tokens 0..3
    # block 1 -> tokens 4..5
    #
    prefix_block_id = request_a.block_table[0]

    assert (
        block_manager.get_ref_count(
            prefix_block_id
        )
        == 1
    )

    prefix_cache = PrefixCache(
        block_manager
    )

    # Cache only the first full block.
    prefix_cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[prefix_block_id],
    )

    # A + PrefixCache
    assert (
        block_manager.get_ref_count(
            prefix_block_id
        )
        == 2
    )

    request_b = make_request(
        request_id="B",
        input_ids=[10, 11, 12, 13, 99, 100],
    )

    entry = prefix_cache.lookup(
        [10, 11, 12, 13]
    )

    assert entry is not None

    # Attach the shared prefix to B.
    request_b.block_table.extend(
        entry.block_ids
    )

    request_b.kv_tokens = len(
        entry.token_ids
    )

    block_manager.retain_blocks(
        entry.block_ids
    )

    # A and B point to the exact same
    # physical prefix block.
    assert (
        request_a.block_table[0]
        == request_b.block_table[0]
    )

    assert (
        request_b.block_table[0]
        == prefix_block_id
    )

    # Owners:
    # A + PrefixCache + B
    assert (
        block_manager.get_ref_count(
            prefix_block_id
        )
        == 3
    )

    assert request_b.kv_tokens == 4

def test_releasing_first_request_keeps_shared_prefix_alive(
    block_manager,
) -> None:
    request_a = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13],
    )

    block_manager.ensure_capacity(
        request=request_a,
        total_tokens=4,
    )

    prefix_block_id = (
        request_a.block_table[0]
    )

    prefix_cache = PrefixCache(
        block_manager
    )

    prefix_cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[prefix_block_id],
    )

    request_b = make_request(
        request_id="B",
        input_ids=[10, 11, 12, 13, 99, 100],
    )

    entry = prefix_cache.lookup(
        [10, 11, 12, 13]
    )

    assert entry is not None

    request_b.block_table.extend(
        entry.block_ids
    )

    request_b.kv_tokens = len(
        entry.token_ids
    )

    block_manager.retain_blocks(
        entry.block_ids
    )

    # A + cache + B
    assert (
        block_manager.get_ref_count(
            prefix_block_id
        )
        == 3
    )

    # A finishes.
    block_manager.release_blocks(
        [prefix_block_id]
    )

    # cache + B still reference it.
    assert (
        block_manager.get_ref_count(
            prefix_block_id
        )
        == 2
    )

    assert (
        prefix_block_id
        not in block_manager._free_blocks
    )

    # B still points to the same block.
    assert (
        request_b.block_table[0]
        == prefix_block_id
    )

def test_releasing_b_still_keeps_prefix_for_cache(
    block_manager,
) -> None:
    request_a = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13],
    )

    block_manager.ensure_capacity(
        request=request_a,
        total_tokens=4,
    )

    prefix_block_id = (
        request_a.block_table[0]
    )

    prefix_cache = PrefixCache(
        block_manager
    )

    prefix_cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[prefix_block_id],
    )

    request_b = make_request(
        request_id="B",
        input_ids=[10, 11, 12, 13, 99],
    )

    entry = prefix_cache.lookup(
        [10, 11, 12, 13]
    )

    assert entry is not None

    request_b.block_table.extend(
        entry.block_ids
    )

    request_b.kv_tokens = len(
        entry.token_ids
    )

    block_manager.retain_blocks(
        entry.block_ids
    )

    # A + cache + B
    assert (
        block_manager.get_ref_count(
            prefix_block_id
        )
        == 3
    )

    # A releases.
    block_manager.release_blocks(
        [prefix_block_id]
    )

    assert (
        block_manager.get_ref_count(
            prefix_block_id
        )
        == 2
    )

    # B releases.
    block_manager.release_blocks(
        [prefix_block_id]
    )

    # Cache still owns the block.
    assert (
        block_manager.get_ref_count(
            prefix_block_id
        )
        == 1
    )

    assert (
        prefix_block_id
        not in block_manager._free_blocks
    )

    # Cache eviction releases final reference.
    prefix_cache.evict(
        token_ids=[10, 11, 12, 13]
    )

    assert (
        block_manager.get_ref_count(
            prefix_block_id
        )
        == 0
    )

    assert (
        prefix_block_id
        in block_manager._free_blocks
    )

def test_shared_prefix_block_but_private_suffix_blocks(
    block_manager,
) -> None:
    request_a = make_request(
        request_id = "A",
        input_ids = [10,11,12,13,14,15],
    )
    block_manager.ensure_capacity(
        request=request_a,
        total_tokens=6,
    )

    assert len(request_a.block_table) == 2

    prefix_cache = PrefixCache(block_manager)

    prefix_id_a = request_a.block_table[0]
    suffix_id_a = request_a.block_table[1]

    prefix_cache.insert(
        token_ids=[10,11,12,13],
        block_ids=[prefix_id_a],
    )

    request_b = make_request(
        request_id = "B",
        input_ids = [10,11,12,13,99,100],
    )
    entry = prefix_cache.lookup([10,11,12,13])

    assert entry is not None

    request_b.block_table.extend(entry.block_ids)
    block_manager.retain_blocks(entry.block_ids)
    request_b.kv_tokens = len(entry.token_ids)

    block_manager.ensure_capacity(
        request = request_b,
        total_tokens=6,)

    assert len(request_b.block_table) == 2

    prefix_id_b = request_b.block_table[0]
    suffix_id_b = request_b.block_table[1]


    assert prefix_id_a == prefix_id_b
    assert suffix_id_a != suffix_id_b

    assert block_manager.get_ref_count(prefix_id_a) == 3
    assert block_manager.get_ref_count(suffix_id_a) == 1
    assert block_manager.get_ref_count(suffix_id_b) == 1

# test prefix cache integrate with scheduler
def test_select_prefill_request_uses_cached_prefix_blocks(
    scheduler,
    block_manager,
    prefix_cache,
) -> None:
    request_a = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13],
    )

    # Let scheduler create the KV and
    # automatically publish the prefix.
    scheduler.add_request(request_a)
    scheduler.step()

    shared_block = (
        request_a.block_table[0]
    )

    entry = prefix_cache.lookup(
        [10, 11, 12, 13]
    )

    assert entry is not None

    assert entry.block_ids == (
        shared_block,
    )

    request_b = make_request(
        request_id="B",
        input_ids=[
            10, 11, 12, 13,
            99, 100,
        ],
    )

    scheduler.add_request(request_b)

    selected = (
        scheduler._select_prefill_requests()
    )

    assert request_b in selected

    assert (
        request_b.block_table[0]
        == shared_block
    )

    assert request_b.kv_tokens == 4

    assert len(
        request_b.block_table
    ) == 2

    assert (
        request_b.block_table[1]
        != shared_block
    )

def test_select_prefill_rolls_back_cached_prefix_when_no_capacity(
    scheduler,
    block_manager,
    prefix_cache,
) -> None:
    request_a = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13],
    )

    scheduler.add_request(request_a)
    scheduler.step()

    shared_block = request_a.block_table[0]

    entry = prefix_cache.lookup(
        [10, 11, 12, 13]
    )

    assert entry is not None
    assert entry.block_ids == (
        shared_block,
    )

    initial_ref_count = (
        block_manager.get_ref_count(
            shared_block
        )
    )

    # Consume all remaining free blocks.
    while block_manager.num_free_blocks > 0:
        dummy = make_request(
            request_id=f"dummy-{block_manager.num_free_blocks}",
            input_ids=[1, 2, 3, 4],
        )

        block_manager.ensure_capacity(
            request=dummy,
            total_tokens=4,
        )

    request_b = make_request(
        request_id="B",
        input_ids=[10, 11, 12, 13, 99, 100],
    )

    scheduler.add_request(request_b)

    selected = scheduler._select_prefill_requests()

    assert request_b not in selected

    # Prefix attachment must have been undone.
    assert request_b.block_table == []

    assert request_b.kv_tokens == 0

    # Temporary retain must also be undone.
    assert (
        block_manager.get_ref_count(
            shared_block
        )
        == initial_ref_count
    )

def test_select_prefill_uses_shorter_cached_prefix_when_longest_misses(
        scheduler,
        block_manager,
        prefix_cache,
    ) -> None:
    request_a = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13],
    )

    scheduler.add_request(request_a)
    scheduler.step()

    shared_block = request_a.block_table[0]

    entry = prefix_cache.lookup(
        [10, 11, 12, 13]
    )

    assert entry is not None
    assert entry.block_ids == (
        shared_block,
    )

    request_b = make_request(
        request_id="B",
        input_ids=[
            10, 11, 12, 13,
            14, 15, 16, 17,
            18,
        ],
    )

    scheduler.add_request(request_b)

    selected = scheduler._select_prefill_requests()

    assert request_b in selected

    # 8-token prefix is not cached,
    # but 4-token prefix is.
    assert request_b.kv_tokens == 4

    assert (
        request_b.block_table[0]
        == shared_block
    )

    # 9 tokens / block_size 4
    # = 3 total blocks.
    #
    # 1 reused + 2 newly allocated.
    assert len(request_b.block_table) == 3

# test write kv cache into paged kv cache with shared blocks
def test_write_request_kv_prefix_cache_writes_only_suffix() -> None:
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=4,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    block_table = [0, 1]

    # Step 1:
    # Pretend block 0 already contains the cached prefix
    # for logical tokens 0..3.
    prefix_key = torch.tensor(
        [[[
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
            [13.0, 13.0],
        ]]]
    )

    prefix_value = torch.tensor(
        [[[
            [110.0, 110.0],
            [111.0, 111.0],
            [112.0, 112.0],
            [113.0, 113.0],
        ]]]
    )

    paged_kv_cache.write_request_kv(
        block_table=block_table,
        past_key_values=(
            (prefix_key, prefix_value),
        ),
        num_tokens=4,
        source_start=0,
    )

    # Save the shared prefix block before suffix write.
    prefix_key_before = (
        paged_kv_cache.key_cache[
            0,
            0,
        ].clone()
    )

    prefix_value_before = (
        paged_kv_cache.value_cache[
            0,
            0,
        ].clone()
    )

    # Step 2:
    # Runner returns the full KV:
    #
    # logical tokens:
    # 0   1   2   3   4    5
    # 10  11  12  13  99   100
    full_key = torch.tensor(
        [[[
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
            [13.0, 13.0],
            [99.0, 99.0],
            [100.0, 100.0],
        ]]]
    )

    full_value = torch.tensor(
        [[[
            [110.0, 110.0],
            [111.0, 111.0],
            [112.0, 112.0],
            [113.0, 113.0],
            [199.0, 199.0],
            [200.0, 200.0],
        ]]]
    )

    # Only write logical tokens 4 and 5.
    paged_kv_cache.write_request_kv_prefix_cache(
        block_table=block_table,
        past_key_values=(
            (full_key, full_value),
        ),
        num_tokens=2,
        source_start=4,
        destination_start=4,
    )

    # Step 3:
    # Shared prefix block must not change.
    torch.testing.assert_close(
        paged_kv_cache.key_cache[
            0,
            0,
        ],
        prefix_key_before,
    )

    torch.testing.assert_close(
        paged_kv_cache.value_cache[
            0,
            0,
        ],
        prefix_value_before,
    )

    # Step 4:
    # Logical token 4 should be written to:
    # block_table[1] = physical block 1, slot 0.
    torch.testing.assert_close(
        paged_kv_cache.key_cache[
            0,
            1,
            :,
            0,
            :,
        ],
        torch.tensor(
            [[99.0, 99.0]]
        ),
    )

    torch.testing.assert_close(
        paged_kv_cache.value_cache[
            0,
            1,
            :,
            0,
            :,
        ],
        torch.tensor(
            [[199.0, 199.0]]
        ),
    )

    # Logical token 5 should go to slot 1.
    torch.testing.assert_close(
        paged_kv_cache.key_cache[
            0,
            1,
            :,
            1,
            :,
        ],
        torch.tensor(
            [[100.0, 100.0]]
        ),
    )

    torch.testing.assert_close(
        paged_kv_cache.value_cache[
            0,
            1,
            :,
            1,
            :,
        ],
        torch.tensor(
            [[200.0, 200.0]]
        ),
    )

def test_write_request_kv_prefix_cache_with_padding() -> None:
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=4,
        num_kv_heads=1,
        block_size=4,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    block_table = [0, 1]

    # Existing cached prefix:
    # logical tokens 0..3
    prefix_key = torch.tensor(
        [[[
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
            [13.0, 13.0],
        ]]]
    )

    prefix_value = prefix_key + 100

    paged_kv_cache.write_request_kv(
        block_table=block_table,
        past_key_values=(
            (prefix_key, prefix_value),
        ),
        num_tokens=4,
        source_start=0,
    )

    prefix_key_before = (
        paged_kv_cache.key_cache[
            0, 0
        ].clone()
    )

    # Runner-produced physical KV:
    #
    # idx:
    # 0    1    2   3   4   5   6   7
    #
    # PAD  PAD  10  11  12  13  99 100
    full_key = torch.tensor(
        [[[
            [-1.0, -1.0],
            [-2.0, -2.0],
            [10.0, 10.0],
            [11.0, 11.0],
            [12.0, 12.0],
            [13.0, 13.0],
            [99.0, 99.0],
            [100.0, 100.0],
        ]]]
    )

    full_value = full_key + 100

    paged_kv_cache.write_request_kv_prefix_cache(
        block_table=block_table,
        past_key_values=(
            (full_key, full_value),
        ),
        num_tokens=2,

        # Read physical indices 6 and 7.
        source_start=6,

        # Write logical indices 4 and 5.
        destination_start=4,
    )

    # Cached prefix block must remain unchanged.
    torch.testing.assert_close(
        paged_kv_cache.key_cache[
            0, 0
        ],
        prefix_key_before,
    )

    # Logical token 4 -> block 1, slot 0
    torch.testing.assert_close(
        paged_kv_cache.key_cache[
            0,
            1,
            :,
            0,
            :,
        ],
        torch.tensor(
            [[99.0, 99.0]]
        ),
    )

    # Logical token 5 -> block 1, slot 1
    torch.testing.assert_close(
        paged_kv_cache.key_cache[
            0,
            1,
            :,
            1,
            :,
        ],
        torch.tensor(
            [[100.0, 100.0]]
        ),
    )

    # PAD values must never appear in the
    # logical suffix positions.
    assert not torch.equal(
        paged_kv_cache.key_cache[
            0,
            1,
            :,
            0,
            :,
        ],
        torch.tensor(
            [[-1.0, -1.0]]
        ),
    )

def test_prefix_hit_prefill_does_not_overwrite_shared_prefix_block(
            scheduler,
            block_manager,
            paged_kv_cache,
            prefix_cache,
        ) -> None:
    # A has one full cacheable block plus a suffix.
    request_a = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13, 14, 15],
    )

    scheduler.add_request(request_a)

    # Prefill A and publish its prefix.
    scheduler.step()

    assert len(request_a.block_table) >= 2

    shared_block_id = request_a.block_table[0]

    entry = prefix_cache.lookup(
    [10, 11, 12, 13]
)

    assert entry is not None

    assert entry.block_ids == (
        shared_block_id,
    )

    # Save the physical contents of the shared block.
    shared_key_before = (
        paged_kv_cache.key_cache[
            :,
            shared_block_id,
        ].clone()
    )

    shared_value_before = (
        paged_kv_cache.value_cache[
            :,
            shared_block_id,
        ].clone()
    )

    # Verify the prefix is actually cached.
    entry = prefix_cache.lookup(
        [10, 11, 12, 13]
    )

    assert entry is not None

    assert (
        entry.block_ids[0]
        == shared_block_id
    )

    # B shares A's first full block,
    # but has a different suffix.
    request_b = make_request(
        request_id="B",
        input_ids=[10, 11, 12, 13, 99, 100],
    )

    scheduler.add_request(request_b)

    scheduler.step()

    # B should reuse the same physical prefix block.
    assert (
        request_b.block_table[0]
        == shared_block_id
    )
    cached_prefix_length = 4
    suffix_input_ids = request_b.input_ids[
        cached_prefix_length:
    ]
    assert suffix_input_ids == [
        99,
        100,
    ]

    # But the shared physical KV block must
    # remain bit-for-bit unchanged.
    torch.testing.assert_close(
        paged_kv_cache.key_cache[
            :,
            shared_block_id,
        ],
        shared_key_before,
    )

    torch.testing.assert_close(
        paged_kv_cache.value_cache[
            :,
            shared_block_id,
        ],
        shared_value_before,
    )

    # B should also have its own suffix block.
    assert len(request_b.block_table) >= 2

    assert (
        request_b.block_table[1]
        != shared_block_id
    )

# runner implement prefill_with_past()
import pytest
import torch

from app.runtime.kv_cache_utils import (
    get_kv_sequence_length,
)
from app.runtime.pytorch_model_runner import (
    PyTorchModelRunner,
)


@pytest.mark.integration
def test_prefill_with_past_matches_full_prefill(
        real_runner: PyTorchModelRunner,
    ) -> None:
    # -------------------------
    # Build a real token sequence
    # -------------------------
    full_input_ids = (
        real_runner.encode_prompt(
            "The capital of France is Paris"
        )
        .squeeze(0)
    )

    assert full_input_ids.ndim == 1
    assert full_input_ids.shape[0] >= 2

    # Split into:
    #
    # prefix + suffix
    #
    # Keep at least one token in suffix.
    prefix_length = (
        full_input_ids.shape[0] - 1
    )

    # [batch_size, sequence_length]
    prefix_input_ids = (
        full_input_ids[:prefix_length]
        .unsqueeze(0)
        .to(real_runner.device)
    )

    suffix_input_ids = (
        full_input_ids[prefix_length:]
        .unsqueeze(0)
        .to(real_runner.device)
    )

    full_input_ids_batch = (
        full_input_ids
        .unsqueeze(0)
        .to(real_runner.device)
    )

    full_length = (
        full_input_ids_batch.shape[1]
    )

    suffix_length = (
        suffix_input_ids.shape[1]
    )

    assert (
        prefix_length + suffix_length
        == full_length
    )

    # ==================================================
    # Path A:
    # Full prefill in one model call
    # ==================================================
    full_attention_mask = torch.ones(
        (
            1,
            full_length,
        ),
        dtype=torch.long,
        device=real_runner.device,
    )

    full_position_ids = torch.arange(
        full_length,
        dtype=torch.long,
        device=real_runner.device,
    ).unsqueeze(0)

    with torch.no_grad():
        full_output = real_runner.model(
            input_ids=full_input_ids_batch,
            attention_mask=full_attention_mask,
            position_ids=full_position_ids,
            use_cache=True,
        )

    full_next_token_id = int(
        torch.argmax(
            full_output.logits[:, -1, :],
            dim=-1,
        ).item()
    )

    full_kv = full_output.past_key_values

    assert (
        get_kv_sequence_length(full_kv)
        == full_length
    )

    # ==================================================
    # Path B, Step 1:
    # Prefill only the prefix
    # ==================================================
    prefix_attention_mask = torch.ones(
        (
            1,
            prefix_length,
        ),
        dtype=torch.long,
        device=real_runner.device,
    )

    prefix_position_ids = torch.arange(
        prefix_length,
        dtype=torch.long,
        device=real_runner.device,
    ).unsqueeze(0)

    with torch.no_grad():
        prefix_output = real_runner.model(
            input_ids=prefix_input_ids,
            attention_mask=prefix_attention_mask,
            position_ids=prefix_position_ids,
            use_cache=True,
        )

    cached_past_key_values = (
        prefix_output.past_key_values
    )

    assert (
        get_kv_sequence_length(
            cached_past_key_values
        )
        == prefix_length
    )

    # ==================================================
    # Path B, Step 2:
    # Run only the suffix using cached prefix KV
    # ==================================================
    suffix_attention_mask = torch.ones(
        (
            1,
            prefix_length + suffix_length,
        ),
        dtype=torch.long,
        device=real_runner.device,
    )

    suffix_position_ids = torch.arange(
        prefix_length,
        prefix_length + suffix_length,
        dtype=torch.long,
        device=real_runner.device,
    ).unsqueeze(0)

    (
        cached_path_next_token_id,
        updated_kv,
    ) = real_runner.prefill_with_past(
        input_ids=suffix_input_ids,
        past_key_values=(
            cached_past_key_values
        ),
        attention_mask=(
            suffix_attention_mask
        ),
        position_ids=(
            suffix_position_ids
        ),
        suffix_lengths=[suffix_length],
    )

    # ==================================================
    # Assertions
    # ==================================================

    # Both execution paths should predict
    # the same next token.
    assert (
        cached_path_next_token_id[0]
        == full_next_token_id
    )

    # Prefix KV + suffix should produce
    # the same final logical KV length.
    assert (
        get_kv_sequence_length(
            updated_kv
        )
        == full_length
    )

    # Same number of layers.
    assert (
        len(updated_kv)
        == len(full_kv)
    )

    # Same shapes for every K/V layer.
    for (
        full_layer,
        cached_layer,
    ) in zip(
        full_kv,
        updated_kv,
    ):
        full_key, full_value = (
            full_layer
        )

        cached_key, cached_value = (
            cached_layer
        )

        assert (
            full_key.shape
            == cached_key.shape
        )

        assert (
            full_value.shape
            == cached_value.shape
        )

@pytest.mark.integration
def test_scheduler_cached_prefix_matches_full_prefill(
        real_runner,
        make_paged_kv_cache_qwen,
    ) -> None:
    # ==================================================
    # Path A: baseline full prefill
    # ==================================================
    baseline_block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

    baseline_prefix_cache = PrefixCache(
        baseline_block_manager
    )

    baseline_paged_kv_cache = (
        make_paged_kv_cache_qwen()
    )

    baseline_scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
        block_manager=baseline_block_manager,
        paged_kv_cache=baseline_paged_kv_cache,
        prefix_cache=baseline_prefix_cache,
    )

    input_ids = (
        real_runner.encode_prompt(
            "The capital of France is Paris"
        )
        .squeeze(0)
        .tolist()
    )

    baseline_request = Request(
        request_id="baseline",
        input_ids=input_ids,
        max_new_tokens=2,
    )

    baseline_scheduler.add_request(
        baseline_request
    )

    baseline_scheduler.step()

    baseline_first_token = (
        baseline_request.generated_ids[0]
    )

    # ==================================================
    # Path B: prefix-cache execution
    # ==================================================
    block_manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

    prefix_cache = PrefixCache(
        block_manager
    )

    paged_kv_cache = (
        make_paged_kv_cache_qwen()
    )

    scheduler = ContinuousScheduler(
        runner=real_runner,
        batch_builder=BatchBuilder(),
        max_prefill_batch_size=1,
        max_decode_batch_size=1,
        block_manager=block_manager,
        paged_kv_cache=paged_kv_cache,
        prefix_cache=prefix_cache,
    )

    # Exactly one full cacheable block.
    prefix_ids = input_ids[:4]

    prefix_request = Request(
        request_id="prefix",
        input_ids=prefix_ids,
        max_new_tokens=2,
    )

    scheduler.add_request(
        prefix_request
    )

    scheduler.step()

    shared_prefix_block = (
        prefix_request.block_table[0]
    )

    # Scheduler should automatically publish
    # the full-block prefix.
    entry = prefix_cache.lookup(
        prefix_ids
    )

    assert entry is not None

    assert entry.block_ids == (
        shared_prefix_block,
    )

    # Save shared block contents before reuse.
    shared_key_before = (
        paged_kv_cache.key_cache[
            :,
            shared_prefix_block,
        ].clone()
    )

    shared_value_before = (
        paged_kv_cache.value_cache[
            :,
            shared_prefix_block,
        ].clone()
    )

    cached_request = Request(
        request_id="cached",
        input_ids=input_ids,
        max_new_tokens=2,
    )

    scheduler.add_request(
        cached_request
    )

    scheduler.step()

    # ==================================================
    # Assertions
    # ==================================================

    # Functional equivalence:
    # cached path predicts the same first token
    # as normal full prefill.
    assert (
        cached_request.generated_ids[0]
        == baseline_first_token
    )

    # Full prompt KV should now be valid.
    assert (
        cached_request.kv_tokens
        == len(cached_request.input_ids)
    )

    # Cached request really reused the same
    # physical prefix block.
    assert (
        cached_request.block_table[0]
        == shared_prefix_block
    )

    # Shared prefix KV must remain immutable.
    torch.testing.assert_close(
        paged_kv_cache.key_cache[
            :,
            shared_prefix_block,
        ],
        shared_key_before,
    )

    torch.testing.assert_close(
        paged_kv_cache.value_cache[
            :,
            shared_prefix_block,
        ],
        shared_value_before,
    )

def test_cache_hit_extends_prefix_cache_with_new_full_block(
        scheduler,
        block_manager,
        paged_kv_cache,
        prefix_cache,
    ) -> None:
    # Step 1:
    # Seed one full cached block.
    request_a = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
    )

    scheduler.add_request(request_a)
    scheduler.step()

    shared_block = request_a.block_table[0]

    # If automatic insertion is already implemented,
    # this should already exist.
    entry_4 = prefix_cache.lookup(
        [1, 2, 3, 4]
    )

    assert entry_4 is not None
    assert entry_4.block_ids == (
        shared_block,
    )

    # Step 2:
    # New request hits the 4-token prefix,
    # then computes five more tokens.
    request_b = make_request(
        request_id="B",
        input_ids=[
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,
        ],
    )

    scheduler.add_request(request_b)
    scheduler.step()

    # It should reuse the existing first block.
    assert (
        request_b.block_table[0]
        == shared_block
    )

    # 9 tokens with block_size=4:
    # [0] tokens 1..4
    # [1] tokens 5..8
    # [2] token 9
    assert len(request_b.block_table) == 3

    second_block = request_b.block_table[1]

    # Step 3:
    # The newly completed 8-token prefix
    # should now be published.
    entry_8 = prefix_cache.lookup(
        [1, 2, 3, 4, 5, 6, 7, 8]
    )

    assert entry_8 is not None

    assert entry_8.block_ids == (
        shared_block,
        second_block,
    )

    # The partial 9-token prefix should NOT
    # be cached.
    entry_9 = prefix_cache.lookup(
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    assert entry_9 is None

# test overlapping prefix entries and reference counts.
def test_overlapping_prefix_entries_keep_shared_block_alive(
        block_manager,
    ) -> None:
    request = make_request(
        request_id="A",
        input_ids=[
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],
    )

    block_manager.ensure_capacity(
        request=request,
        total_tokens=8,
    )

    assert len(request.block_table) == 2

    block_0 = request.block_table[0]
    block_1 = request.block_table[1]

    prefix_cache = PrefixCache(
        block_manager
    )

    # Request owns both blocks:
    #
    # block_0 ref = 1
    # block_1 ref = 1
    assert (
        block_manager.get_ref_count(block_0)
        == 1
    )

    assert (
        block_manager.get_ref_count(block_1)
        == 1
    )

    # Cache the 4-token prefix.
    prefix_cache.insert(
        token_ids=[1, 2, 3, 4],
        block_ids=[block_0],
    )

    # block_0:
    # request + 4-token cache entry
    assert (
        block_manager.get_ref_count(block_0)
        == 2
    )

    assert (
        block_manager.get_ref_count(block_1)
        == 1
    )

    # Cache the 8-token prefix.
    prefix_cache.insert(
        token_ids=[
            1, 2, 3, 4,
            5, 6, 7, 8,
        ],
        block_ids=[
            block_0,
            block_1,
        ],
    )

    # block_0:
    # request
    # + 4-token entry
    # + 8-token entry
    #
    # ref = 3
    assert (
        block_manager.get_ref_count(block_0)
        == 3
    )

    # block_1:
    # request
    # + 8-token entry
    #
    # ref = 2
    assert (
        block_manager.get_ref_count(block_1)
        == 2
    )

    # Request finishes.
    block_manager.release_blocks(
        request.block_table
    )

    # Only cache references remain.
    assert (
        block_manager.get_ref_count(block_0)
        == 2
    )

    assert (
        block_manager.get_ref_count(block_1)
        == 1
    )

    # Evict shorter prefix first.
    prefix_cache.evict(
        token_ids=[1, 2, 3, 4]
    )

    # block_0 must NOT be freed because
    # the 8-token entry still references it.
    assert (
        block_manager.get_ref_count(block_0)
        == 1
    )

    assert (
        block_0
        not in block_manager._free_blocks
    )

    # block_1 is still referenced by
    # the 8-token entry too.
    assert (
        block_manager.get_ref_count(block_1)
        == 1
    )

    assert (
        block_1
        not in block_manager._free_blocks
    )

    # Now evict the 8-token prefix.
    prefix_cache.evict(
        token_ids=[
            1, 2, 3, 4,
            5, 6, 7, 8,
        ]
    )

    # All references are gone.
    assert (
        block_manager.get_ref_count(block_0)
        == 0
    )

    assert (
        block_manager.get_ref_count(block_1)
        == 0
    )

    assert (
        block_0
        in block_manager._free_blocks
    )

    assert (
        block_1
        in block_manager._free_blocks
    )

def test_scheduler_created_overlapping_prefixes_have_correct_ref_counts(
        scheduler,
        block_manager,
        prefix_cache,
    ) -> None:
    # Step 1:
    # Request A creates the first cached prefix:
    #
    # [1,2,3,4] -> [block_0]
    request_a = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
        max_new_tokens=4,
    )

    scheduler.add_request(request_a)
    scheduler.step()

    entry_4 = prefix_cache.lookup(
        [1, 2, 3, 4]
    )

    assert entry_4 is not None

    block_0 = entry_4.block_ids[0]

    # Step 2:
    # Request B hits the 4-token prefix and extends it:
    #
    # [1,2,3,4,5,6,7,8,9]
    #
    # After prefill:
    #
    # [1..4] -> [block_0]
    # [1..8] -> [block_0, block_1]
    request_b = make_request(
        request_id="B",
        input_ids=[
            1, 2, 3, 4,
            5, 6, 7, 8,
            9,
        ],
        max_new_tokens=4,
    )

    scheduler.add_request(request_b)
    scheduler.step()

    entry_8 = prefix_cache.lookup(
        [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ]
    )

    assert entry_8 is not None

    assert entry_8.block_ids[0] == block_0
    assert len(entry_8.block_ids) == 2

    block_1 = entry_8.block_ids[1]

    # Both cache entries reference block_0.
    #
    # We don't assert an exact refcount yet,
    # because A/B may still hold request references
    # depending on scheduler lifecycle.
    ref_count_block_0_before = (
        block_manager.get_ref_count(
            block_0
        )
    )

    ref_count_block_1_before = (
        block_manager.get_ref_count(
            block_1
        )
    )

    assert ref_count_block_0_before >= 2
    assert ref_count_block_1_before >= 1

    # Step 3:
    # Evict only the shorter prefix.
    prefix_cache.evict(
        token_ids=[1, 2, 3, 4]
    )

    # block_0 must still be alive because
    # the 8-token entry still references it.
    assert (
        block_manager.get_ref_count(
            block_0
        )
        == ref_count_block_0_before - 1
    )

    assert (
        block_0
        not in block_manager._free_blocks
    )

    # block_1 was not referenced by the
    # 4-token entry, so its refcount should
    # not change.
    assert (
        block_manager.get_ref_count(
            block_1
        )
        == ref_count_block_1_before
    )

    # Step 4:
    # Evict the longer prefix too.
    prefix_cache.evict(
        token_ids=[
            1, 2, 3, 4,
            5, 6, 7, 8,
        ]
    )

    # Each block loses exactly one more
    # cache-held reference.
    assert (
        block_manager.get_ref_count(
            block_0
        )
        == ref_count_block_0_before - 2
    )

    assert (
        block_manager.get_ref_count(
            block_1
        )
        == ref_count_block_1_before - 1
    )

def test_finished_request_releases_own_reference_but_cache_keeps_block_alive(
        scheduler,
        block_manager,
        prefix_cache,
    ) -> None:
    request = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
        max_new_tokens=2,
    )

    scheduler.add_request(request)

    # Step 1:
    # A is selected for prefill.
    # Prefill generates token #1.
    prefill_result = scheduler.step()

    assert request.generated_tokens_count == 1
    assert request.state == RequestState.DECODING

    entry = prefix_cache.lookup(
        [1, 2, 3, 4]
    )

    assert entry is not None
    assert len(entry.block_ids) == 1

    cached_block = entry.block_ids[0]

    # At this point:
    # request + PrefixCache both own the block.
    assert (
        block_manager.get_ref_count(
            cached_block
        )
        == 2
    )

    assert (
        cached_block
        not in block_manager._free_blocks
    )

    # Step 2:
    # A is now selected for decode.
    # Decode generates token #2 and A reaches
    # max_new_tokens.
    decode_result = scheduler.step()

    assert request.generated_tokens_count == 2
    assert request.state == RequestState.FINISHED
    assert request.finish_reason == "length"

    assert (
        request.request_id
        in scheduler.completed
    )

    assert (
        request.request_id
        not in scheduler.active
    )

    # Request-owned reference should be gone.
    # PrefixCache still owns the cached block.
    assert (
        block_manager.get_ref_count(
            cached_block
        )
        == 1
    )

    assert (
        cached_block
        not in block_manager._free_blocks
    )

    # Finally remove the cache's reference.
    prefix_cache.evict(
        token_ids=[1, 2, 3, 4]
    )

    assert (
        block_manager.get_ref_count(
            cached_block
        )
        == 0
    )

    assert (
        cached_block
        in block_manager._free_blocks
    )

def test_new_request_reuses_prefix_after_original_request_finishes(
        scheduler,
        block_manager,
        prefix_cache,
    ) -> None:
    # ----------------------------------
    # Step 1:
    # A creates a cacheable prefix.
    # ----------------------------------
    request_a = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
        max_new_tokens=2,
    )

    scheduler.add_request(request_a)

    # Step 1:
    # prefill A, generate token #1
    scheduler.step()

    entry = prefix_cache.lookup(
        [1, 2, 3, 4]
    )

    assert entry is not None

    cached_block = entry.block_ids[0]

    # A + PrefixCache
    assert (
        block_manager.get_ref_count(
            cached_block
        )
        == 2
    )

    # ----------------------------------
    # Step 2:
    # Decode A once.
    # max_new_tokens=2, so A finishes.
    # ----------------------------------
    scheduler.step()

    assert (
        request_a.state
        == RequestState.FINISHED
    )

    # A released its reference.
    # PrefixCache still owns the block.
    assert (
        block_manager.get_ref_count(
            cached_block
        )
        == 1
    )

    assert (
        cached_block
        not in block_manager._free_blocks
    )

    # PrefixCache entry must still exist.
    entry_after_a_finishes = (
        prefix_cache.lookup(
            [1, 2, 3, 4]
        )
    )

    assert (
        entry_after_a_finishes
        is not None
    )

    assert (
        entry_after_a_finishes.block_ids
        == (cached_block,)
    )

    # ----------------------------------
    # Step 3:
    # B arrives after A has finished.
    # ----------------------------------
    request_b = make_request(
        request_id="B",
        input_ids=[
            1, 2, 3, 4,
            9, 10,
        ],
        max_new_tokens=4,
    )

    scheduler.add_request(request_b)

    # Only select prefill first so we can inspect
    # the attachment before execution.
    selected = (
        scheduler._select_prefill_requests()
    )

    assert request_b in selected

    # B should reuse A's old cached block.
    assert (
        request_b.block_table[0]
        == cached_block
    )

    assert request_b.kv_tokens == 4

    # PrefixCache + B
    assert (
        block_manager.get_ref_count(
            cached_block
        )
        == 2
    )

    # B also needs a private suffix block.
    assert len(request_b.block_table) == 2

    assert (
        request_b.block_table[1]
        != cached_block
    )

# test batch prefill 
def test_split_prefill_requests_by_cache_hit(
        scheduler,
    ) -> None:
    request_a = make_request(
        request_id="A",
        input_ids=[1, 2, 3, 4],
    )

    request_b = make_request(
        request_id="B",
        input_ids=[1, 2, 3, 4, 5, 6],
    )

    request_c = make_request(
        request_id="C",
        input_ids=[9, 10, 11],
    )

    # Simulate that B already attached
    # a 4-token cached prefix.
    request_b.kv_tokens = 4

    cache_misses, cache_hits = (
        scheduler._split_prefill_requests_by_cache_hit(
            [
                request_a,
                request_b,
                request_c,
            ]
        )
    )

    assert cache_misses == [
        request_a,
        request_c,
    ]

    assert cache_hits == [
        request_b,
    ]

def test_mixed_prefill_batch_handles_cache_miss_and_hit(
        scheduler,
        fake_runner_all_dim,
    ) -> None:
    # Seed the prefix cache first.
    cached_request = make_request(
        request_id="cached-source",
        input_ids=[10, 11, 12, 13],
    )

    scheduler.add_request(cached_request)

    scheduler.step()

    # Finish/release the original request while keeping
    # the prefix cache entry alive.
    while (
        cached_request.state
        != RequestState.FINISHED
    ):
        scheduler.step()

    # A is a full cache miss.
    request_a = make_request(
        request_id="A",
        input_ids=[50, 51, 52, 53],
    )

    # B shares the cached prefix and has a new suffix.
    request_b = make_request(
        request_id="B",
        input_ids=[10, 11, 12, 13, 99],
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    result = scheduler.step()

    assert request_a.request_id in (
        result.prefetched_request_ids
    )
    assert request_b.request_id in (
        result.prefetched_request_ids
    )

    assert request_a.state == RequestState.DECODING
    assert request_b.state == RequestState.DECODING

    # A had no reusable KV before prefill.
    # Its entire prompt should now be represented.
    assert request_a.kv_tokens == len(
        request_a.input_ids
    )

    # B should also end up with KV for its entire prompt:
    #
    # cached:
    # [10, 11, 12, 13]
    #
    # suffix:
    # [99]
    assert request_b.kv_tokens == len(
        request_b.input_ids
    )

    assert (
        request_a.request_id
        in result.generated_token_ids
    )
    assert (
        request_b.request_id
        in result.generated_token_ids
    )

def test_multiple_cache_misses_remain_batched(
        scheduler,
        fake_runner_all_dim,
    ) -> None:
    request_a = make_request(
        request_id="A",
        input_ids=[10, 11, 12],
    )

    request_b = make_request(
        request_id="B",
        input_ids=[20, 21, 22],
    )

    scheduler.add_request(request_a)
    scheduler.add_request(request_b)

    result = scheduler.step()

    assert request_a.state == RequestState.DECODING
    assert request_b.state == RequestState.DECODING

    assert request_a.request_id in (
        result.prefetched_request_ids
    )
    assert request_b.request_id in (
        result.prefetched_request_ids
    )

    assert request_a.request_id in (
        result.generated_token_ids
    )
    assert request_b.request_id in (
        result.generated_token_ids
    )

def test_mixed_prefill_handles_one_miss_and_multiple_hits(
        scheduler_fake_runner,
        fake_runner_all_dim,
    ) -> None:
    # Seed reusable prefix.
    source = make_request(
        request_id="source",
        input_ids=[10, 11, 12, 13],
    )

    scheduler_fake_runner.add_request(source)
    scheduler_fake_runner.step()

    while source.state != RequestState.FINISHED:
        scheduler_fake_runner.step()

    request_a = make_request(
        request_id="A",
        input_ids=[50, 51, 52],
    )

    request_b = make_request(
        request_id="B",
        input_ids=[10, 11, 12, 13, 99],
    )

    request_c = make_request(
        request_id="C",
        input_ids=[10, 11, 12, 13, 100],
    )

    scheduler_fake_runner.add_request(request_a)
    scheduler_fake_runner.add_request(request_b)
    scheduler_fake_runner.add_request(request_c)

    result = scheduler_fake_runner.step()

    assert request_a.state == RequestState.DECODING
    assert request_b.state == RequestState.DECODING
    assert request_c.state == RequestState.DECODING

    assert request_a.request_id in (
        result.prefetched_request_ids
    )
    assert request_b.request_id in (
        result.prefetched_request_ids
    )
    assert request_c.request_id in (
        result.prefetched_request_ids
    )

    assert request_a.request_id in (
        result.generated_token_ids
    )
    assert request_b.request_id in (
        result.generated_token_ids
    )
    assert request_c.request_id in (
        result.generated_token_ids
    )

    assert request_a.kv_tokens == 3
    assert request_b.kv_tokens == 5
    assert request_c.kv_tokens == 5

def test_two_equal_length_cache_hits_use_one_batched_prefill_with_past(
        scheduler_fake_runner,
        fake_runner_all_dim,
    ) -> None:
    assert scheduler_fake_runner.runner is fake_runner_all_dim
    
    source = make_request(
        request_id="source",
        input_ids=[10, 11, 12, 13],
    )

    scheduler_fake_runner.add_request(source)

    scheduler_fake_runner.step()

    while source.state != RequestState.FINISHED:
        scheduler_fake_runner.step()

    # Ignore calls made while creating the
    # prefix-cache entry.
    fake_runner_all_dim.prefill_with_past_batch_sizes.clear()

    request_b = make_request(
        request_id="B",
        input_ids=[
            10,
            11,
            12,
            13,
            99,
        ],
    )

    request_c = make_request(
        request_id="C",
        input_ids=[
            10,
            11,
            12,
            13,
            100,
        ],
    )

    scheduler_fake_runner.add_request(request_b)
    scheduler_fake_runner.add_request(request_c)

    result = scheduler_fake_runner.step()

    assert request_b.request_id in (
        result.prefetched_request_ids
    )

    assert request_c.request_id in (
        result.prefetched_request_ids
    )

    assert request_b.state == RequestState.DECODING
    assert request_c.state == RequestState.DECODING

    assert request_b.kv_tokens == 5
    assert request_c.kv_tokens == 5

    # Current implementation:
    #
    # B -> prefill_with_past(batch_size=1)
    # C -> prefill_with_past(batch_size=1)
    #
    # Therefore there are two runner calls.
    assert (
        fake_runner_all_dim.prefill_with_past_batch_sizes
        == [2]
    )

def test_cached_prefill_batches_variable_suffix_lengths(
        scheduler_fake_runner,
        fake_runner_all_dim,
    ) -> None:
    scheduler = scheduler_fake_runner

    assert scheduler.runner is fake_runner_all_dim

    # Seed reusable prefix:
    # [10, 11, 12, 13]
    source = make_request(
        request_id="source",
        input_ids=[10, 11, 12, 13],
    )

    scheduler.add_request(source)
    scheduler.step()

    while source.state != RequestState.FINISHED:
        scheduler.step()

    fake_runner_all_dim.prefill_with_past_batch_sizes.clear()

    # B:
    # cached prefix length = 4
    # suffix length = 1
    request_b = make_request(
        request_id="B",
        input_ids=[
            10,
            11,
            12,
            13,
            99,
        ],
    )

    # C:
    # cached prefix length = 4
    # suffix length = 3
    request_c = make_request(
        request_id="C",
        input_ids=[
            10,
            11,
            12,
            13,
            100,
            101,
            102,
        ],
    )

    scheduler.add_request(request_b)
    scheduler.add_request(request_c)

    result = scheduler.step()

    # Both requests should be handled
    # during this scheduler tick.
    assert request_b.request_id in (
        result.prefetched_request_ids
    )

    assert request_c.request_id in (
        result.prefetched_request_ids
    )

    assert request_b.request_id in (
        result.generated_token_ids
    )

    assert request_c.request_id in (
        result.generated_token_ids
    )

    # Both should now be ready for decode.
    assert request_b.state == RequestState.DECODING
    assert request_c.state == RequestState.DECODING

    # Logical KV lengths must reflect each
    # request's REAL prompt length.
    assert request_b.kv_tokens == 5
    assert request_c.kv_tokens == 7

    # Most important batching assertion:
    #
    # both cache-hit requests went through
    # one prefill_with_past() runner call.
    assert (
        fake_runner_all_dim.prefill_with_past_batch_sizes
        == [2]
    )

def test_cached_prefill_batches_variable_prefix_lengths(
        scheduler_fake_runner,
        fake_runner_all_dim,
    ) -> None:
    scheduler = scheduler_fake_runner

    assert scheduler.runner is fake_runner_all_dim

    # ----------------------------------------
    # Seed short prefix:
    # [10, 11, 12, 13]
    # cached length = 4
    # ----------------------------------------
    source_short = make_request(
        request_id="source-short",
        input_ids=[
            10,
            11,
            12,
            13,
        ],
    )

    scheduler.add_request(source_short)
    scheduler.step()

    while source_short.state != RequestState.FINISHED:
        scheduler.step()

    # ----------------------------------------
    # Seed long prefix:
    # [20, 21, 22, 23, 24, 25, 26, 27]
    # cached length = 8
    # ----------------------------------------
    source_long = make_request(
        request_id="source-long",
        input_ids=[
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ],
    )

    scheduler.add_request(source_long)
    scheduler.step()

    while source_long.state != RequestState.FINISHED:
        scheduler.step()

    fake_runner_all_dim.prefill_with_past_batch_sizes.clear()

    # B:
    # cached prefix = 4
    # suffix = [99]
    request_b = make_request(
        request_id="B",
        input_ids=[
            10,
            11,
            12,
            13,
            99,
        ],
    )

    # C:
    # cached prefix = 8
    # suffix = [100]
    request_c = make_request(
        request_id="C",
        input_ids=[
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            100,
        ],
    )

    scheduler.add_request(request_b)
    scheduler.add_request(request_c)

    result = scheduler.step()

    assert request_b.request_id in (
        result.prefetched_request_ids
    )

    assert request_c.request_id in (
        result.prefetched_request_ids
    )

    assert request_b.request_id in (
        result.generated_token_ids
    )

    assert request_c.request_id in (
        result.generated_token_ids
    )

    assert request_b.state == RequestState.DECODING
    assert request_c.state == RequestState.DECODING

    # Logical KV lengths.
    assert request_b.kv_tokens == 5
    assert request_c.kv_tokens == 9

    # One batched cached-prefill runner call.
    assert (
        fake_runner_all_dim.prefill_with_past_batch_sizes
        == [2]
    )


def test_cached_prefill_variable_prefix_lengths_are_order_independent(
        scheduler_fake_runner,
        fake_runner_all_dim,
    ) -> None:
    scheduler = scheduler_fake_runner

    source_short = make_request(
        request_id="source-short",
        input_ids=[
            10,
            11,
            12,
            13,
        ],
    )

    scheduler.add_request(source_short)
    scheduler.step()

    while source_short.state != RequestState.FINISHED:
        scheduler.step()

    source_long = make_request(
        request_id="source-long",
        input_ids=[
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ],
    )

    scheduler.add_request(source_long)
    scheduler.step()

    while source_long.state != RequestState.FINISHED:
        scheduler.step()

    fake_runner_all_dim.prefill_with_past_batch_sizes.clear()

    # Add LONG prefix request first.
    request_long = make_request(
        request_id="long",
        input_ids=[
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            100,
        ],
    )

    # Add SHORT prefix request second.
    request_short = make_request(
        request_id="short",
        input_ids=[
            10,
            11,
            12,
            13,
            99,
        ],
    )

    scheduler.add_request(request_long)
    scheduler.add_request(request_short)

    result = scheduler.step()

    assert request_long.request_id in (
        result.prefetched_request_ids
    )

    assert request_short.request_id in (
        result.prefetched_request_ids
    )

    assert request_long.state == RequestState.DECODING
    assert request_short.state == RequestState.DECODING

    assert request_long.kv_tokens == 9
    assert request_short.kv_tokens == 5

    assert (
        fake_runner_all_dim.prefill_with_past_batch_sizes
        == [2]
    )

def test_cached_prefill_copies_suffix_from_physical_boundary_to_logical_position(
        scheduler_fake_runner,
        fake_runner_all_dim,
        paged_kv_cache,
    ) -> None:
    scheduler = scheduler_fake_runner

    # --------------------------------------------------
    # Seed a 4-token prefix cache.
    # --------------------------------------------------
    source_short = make_request(
        request_id="source-short",
        input_ids=[
            10,
            11,
            12,
            13,
        ],
    )

    scheduler.add_request(source_short)

    while source_short.state != RequestState.FINISHED:
        scheduler.step()

    # --------------------------------------------------
    # Seed an 8-token prefix cache.
    #
    # This causes the cached-prefill batch below to have:
    #
    # max_cached_prefix_length = 8
    # --------------------------------------------------
    source_long = make_request(
        request_id="source-long",
        input_ids=[
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
        ],
    )

    scheduler.add_request(source_long)

    while source_long.state != RequestState.FINISHED:
        scheduler.step()

    fake_runner_all_dim.prefill_with_past_batch_sizes.clear()

    # --------------------------------------------------
    # Short cached prefix:
    #
    # logical:
    #
    # [10 11 12 13] [99]
    #        4 KV       1 suffix
    #
    # physical cached batch:
    #
    # [PAD PAD PAD PAD 10 11 12 13] [99]
    #
    # Therefore:
    #
    # physical suffix source = 8
    # logical suffix destination = 4
    # --------------------------------------------------
    request_short = make_request(
        request_id="short",
        input_ids=[
            10,
            11,
            12,
            13,
            99,
        ],
    )

    # --------------------------------------------------
    # Long cached prefix:
    #
    # [20 21 22 23 24 25 26 27] [100]
    #
    # cached prefix = 8
    # --------------------------------------------------
    request_long = make_request(
        request_id="long",
        input_ids=[
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            100,
        ],
    )

    scheduler.add_request(request_short)
    scheduler.add_request(request_long)

    result = scheduler.step()

    assert (
        fake_runner_all_dim.prefill_with_past_batch_sizes
        == [2]
    )

    assert request_short.request_id in (
        result.prefetched_request_ids
    )

    assert request_long.request_id in (
        result.prefetched_request_ids
    )

    # Logical prompt KV lengths after cached prefill.
    assert request_short.kv_tokens == 5
    assert request_long.kv_tokens == 9

    # --------------------------------------------------
    # Materialize SHORT request's logical KV.
    #
    # It should have length 5:
    #
    # logical:
    #   0   1   2   3   4
    # [prefix ........] [suffix]
    #
    # The suffix at logical position 4 should have been
    # copied from PHYSICAL position 8.
    # --------------------------------------------------
    short_kv = (
        paged_kv_cache.materialize_request_kv(
            block_table=request_short.block_table,
            num_tokens=request_short.kv_tokens,
        )
    )

    assert (
        get_kv_sequence_length(short_kv)
        == 5
    )

    short_key = short_kv[0][0]

    # Shape:
    #
    # [1, num_kv_heads, sequence_length, head_dim]
    #
    suffix_key = short_key[
        0,
        0,
        4,
        0,
    ]

    assert suffix_key.item() == 8.0

@pytest.mark.integration
def test_batched_prefill_with_different_cached_prefix_lengths_matches_full_prefill(
    real_runner: PyTorchModelRunner,
) -> None:
    # ==================================================
    # Build two real token sequences
    # ==================================================
    full_a = (
        real_runner.encode_prompt(
            "The capital of France is Paris"
        )
        .squeeze(0)
    )

    full_b = (
        real_runner.encode_prompt(
            "Machine learning systems need efficient inference"
        )
        .squeeze(0)
    )

    assert full_a.ndim == 1
    assert full_b.ndim == 1

    # We want different cached-prefix lengths.
    prefix_a_length = 3
    prefix_b_length = 5

    assert full_a.shape[0] > prefix_a_length
    assert full_b.shape[0] > prefix_b_length

    suffix_a = full_a[prefix_a_length:]
    suffix_b = full_b[prefix_b_length:]

    suffix_a_length = int(
        suffix_a.shape[0]
    )

    suffix_b_length = int(
        suffix_b.shape[0]
    )

    assert suffix_a_length > 0
    assert suffix_b_length > 0

    # ==================================================
    # Reference path:
    # independently full-prefill both requests
    # ==================================================
    def full_prefill_next_token(
        input_ids: torch.Tensor,
    ) -> int:
        batch = (
            input_ids
            .unsqueeze(0)
            .to(real_runner.device)
        )

        sequence_length = int(
            batch.shape[1]
        )

        attention_mask = torch.ones(
            (
                1,
                sequence_length,
            ),
            dtype=torch.long,
            device=real_runner.device,
        )

        position_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=real_runner.device,
        ).unsqueeze(0)

        with torch.no_grad():
            output = real_runner.model(
                input_ids=batch,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        return int(
            torch.argmax(
                output.logits[:, -1, :],
                dim=-1,
            ).item()
        )

    expected_a = full_prefill_next_token(
        full_a
    )

    expected_b = full_prefill_next_token(
        full_b
    )

    # ==================================================
    # Prefill prefixes independently
    # ==================================================
    def build_prefix_kv(
        input_ids: torch.Tensor,
    ):
        batch = (
            input_ids
            .unsqueeze(0)
            .to(real_runner.device)
        )

        sequence_length = int(
            batch.shape[1]
        )

        attention_mask = torch.ones(
            (
                1,
                sequence_length,
            ),
            dtype=torch.long,
            device=real_runner.device,
        )

        position_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=real_runner.device,
        ).unsqueeze(0)

        with torch.no_grad():
            output = real_runner.model(
                input_ids=batch,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        return output.past_key_values

    kv_a = build_prefix_kv(
        full_a[:prefix_a_length]
    )

    kv_b = build_prefix_kv(
        full_b[:prefix_b_length]
    )

    assert (
        get_kv_sequence_length(kv_a)
        == prefix_a_length
    )

    assert (
        get_kv_sequence_length(kv_b)
        == prefix_b_length
    )

    # ==================================================
    # Left-pad cached KV to common physical length
    #
    # A:
    # [PAD PAD | real real real]
    #
    # B:
    # [real real real real real]
    # ==================================================
    max_prefix_length = max(
        prefix_a_length,
        prefix_b_length,
    )

    def left_pad_single_kv(
        past_key_values,
        target_length: int,
    ):
        current_length = (
            get_kv_sequence_length(
                past_key_values
            )
        )

        pad_length = (
            target_length
            - current_length
        )

        padded_layers = []

        for key, value in past_key_values:
            if pad_length > 0:
                key_padding = torch.zeros(
                    (
                        key.shape[0],
                        key.shape[1],
                        pad_length,
                        key.shape[3],
                    ),
                    dtype=key.dtype,
                    device=key.device,
                )

                value_padding = torch.zeros(
                    (
                        value.shape[0],
                        value.shape[1],
                        pad_length,
                        value.shape[3],
                    ),
                    dtype=value.dtype,
                    device=value.device,
                )

                key = torch.cat(
                    (
                        key_padding,
                        key,
                    ),
                    dim=2,
                )

                value = torch.cat(
                    (
                        value_padding,
                        value,
                    ),
                    dim=2,
                )

            padded_layers.append(
                (
                    key,
                    value,
                )
            )

        return tuple(
            padded_layers
        )

    padded_kv_a = left_pad_single_kv(
        kv_a,
        max_prefix_length,
    )

    padded_kv_b = left_pad_single_kv(
        kv_b,
        max_prefix_length,
    )

    # ==================================================
    # Concatenate the two padded caches into one batch
    # ==================================================
    batched_layers = []

    for (
        (key_a, value_a),
        (key_b, value_b),
    ) in zip(
        padded_kv_a,
        padded_kv_b,
    ):
        batched_layers.append(
            (
                torch.cat(
                    (
                        key_a,
                        key_b,
                    ),
                    dim=0,
                ),
                torch.cat(
                    (
                        value_a,
                        value_b,
                    ),
                    dim=0,
                ),
            )
        )

    batched_past_key_values = tuple(
        batched_layers
    )

    assert (
        get_kv_sequence_length(
            batched_past_key_values
        )
        == max_prefix_length
    )

    # ==================================================
    # Right-pad suffix input_ids
    # ==================================================
    max_suffix_length = max(
        suffix_a_length,
        suffix_b_length,
    )

    input_ids = torch.full(
        (
            2,
            max_suffix_length,
        ),
        fill_value=real_runner.pad_token_id,
        dtype=torch.long,
        device=real_runner.device,
    )

    input_ids[
        0,
        :suffix_a_length,
    ] = suffix_a.to(
        real_runner.device
    )

    input_ids[
        1,
        :suffix_b_length,
    ] = suffix_b.to(
        real_runner.device
    )

    # ==================================================
    # Position IDs
    #
    # Important:
    #
    # positions are LOGICAL positions,
    # not physical padded-KV positions.
    # ==================================================
    position_ids = torch.zeros(
        (
            2,
            max_suffix_length,
        ),
        dtype=torch.long,
        device=real_runner.device,
    )

    position_ids[
        0,
        :suffix_a_length,
    ] = torch.arange(
        prefix_a_length,
        prefix_a_length + suffix_a_length,
        dtype=torch.long,
        device=real_runner.device,
    )

    position_ids[
        1,
        :suffix_b_length,
    ] = torch.arange(
        prefix_b_length,
        prefix_b_length + suffix_b_length,
        dtype=torch.long,
        device=real_runner.device,
    )

    # ==================================================
    # Attention mask
    #
    # A:
    # [0 0 1 1 1 | suffix... | 0 0]
    #
    # B:
    # [1 1 1 1 1 | suffix...]
    # ==================================================
    attention_mask = torch.zeros(
        (
            2,
            max_prefix_length
            + max_suffix_length,
        ),
        dtype=torch.long,
        device=real_runner.device,
    )

    # Request A cached prefix
    prefix_a_start = (
        max_prefix_length
        - prefix_a_length
    )

    attention_mask[
        0,
        prefix_a_start:max_prefix_length,
    ] = 1

    attention_mask[
        0,
        max_prefix_length:
        max_prefix_length + suffix_a_length,
    ] = 1

    # Request B cached prefix
    prefix_b_start = (
        max_prefix_length
        - prefix_b_length
    )

    attention_mask[
        1,
        prefix_b_start:max_prefix_length,
    ] = 1

    attention_mask[
        1,
        max_prefix_length:
        max_prefix_length + suffix_b_length,
    ] = 1

    # ==================================================
    # One real batched cached-prefill
    # ==================================================
    (
        actual_next_token_ids,
        updated_kv,
    ) = real_runner.prefill_with_past(
        input_ids=input_ids,
        past_key_values=batched_past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        suffix_lengths=[
            suffix_a_length,
            suffix_b_length,
        ],
    )

    # ==================================================
    # Verify semantic equivalence
    # ==================================================
    assert len(actual_next_token_ids) == 2

    assert (
        actual_next_token_ids[0]
        == expected_a
    )

    assert (
        actual_next_token_ids[1]
        == expected_b
    )

    # Physical output KV contains:
    #
    # max padded prefix
    # +
    # max padded suffix
    #
    assert (
        get_kv_sequence_length(updated_kv)
        ==
        max_prefix_length
        + max_suffix_length
    )

    ############# end-to-end scheduler integration test using the real runner ##########
@pytest.mark.integration
def test_scheduler_cached_prefill_with_variable_prefix_lengths_real_runner(
        scheduler,
        real_runner,
    ) -> None:
    # ----------------------------------------
    # Seed short cached prefix.
    # ----------------------------------------
    short_prefix_ids = (
        real_runner.encode_prompt(
            "The capital of France"
        )
        .squeeze(0)
        .tolist()
    )

    # Make sure the prefix contains at least
    # one full cache block.
    assert len(short_prefix_ids) >= 4

    short_prefix_ids = short_prefix_ids[:4]

    source_short = make_request(
        request_id="source-short",
        input_ids=short_prefix_ids,
        max_new_tokens=1,
    )

    scheduler.add_request(source_short)

    while source_short.state != RequestState.FINISHED:
        scheduler.step()

    # ----------------------------------------
    # Seed long cached prefix.
    # ----------------------------------------
    long_prefix_ids = (
        real_runner.encode_prompt(
            "Machine learning inference systems "
            "need efficient memory management"
        )
        .squeeze(0)
        .tolist()
    )

    assert len(long_prefix_ids) >= 8

    long_prefix_ids = long_prefix_ids[:8]

    source_long = make_request(
        request_id="source-long",
        input_ids=long_prefix_ids,
        max_new_tokens=1,
    )

    scheduler.add_request(source_long)

    while source_long.state != RequestState.FINISHED:
        scheduler.step()

    # ----------------------------------------
    # Build cache-hit requests.
    #
    # Different:
    #   cached prefix lengths
    #   suffix lengths
    # ----------------------------------------
    request_short = make_request(
        request_id="short",
        input_ids=(
            short_prefix_ids
            + [1234]
        ),
        max_new_tokens=3,
    )

    request_long = make_request(
        request_id="long",
        input_ids=(
            long_prefix_ids
            + [2345, 3456]
        ),
        max_new_tokens=3,
    )

    scheduler.add_request(request_short)
    scheduler.add_request(request_long)

    # ----------------------------------------
    # Cached prefill should happen together.
    # ----------------------------------------
    result = scheduler.step()

    assert (
        request_short.request_id
        in result.prefetched_request_ids
    )

    assert (
        request_long.request_id
        in result.prefetched_request_ids
    )

    # Scheduler should have completed
    # logical prompt KV construction.
    assert (
        request_short.kv_tokens
        == len(request_short.input_ids)
    )

    assert (
        request_long.kv_tokens
        == len(request_long.input_ids)
    )

    assert (
        request_short.state
        == RequestState.DECODING
    )

    assert (
        request_long.state
        == RequestState.DECODING
    )


