import torch
from app.runtime.request import Request

from app.runtime.prefix_cache import PrefixCache
from app.runtime.paged_kv_cache import PagedKVCache


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
def test_evicts_blocks_when_capacity_exceeded(block_manager)-> None:
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
    # A owns a full cached prefix block.
    request_a = make_request(
        request_id="A",
        input_ids=[10, 11, 12, 13],
    )

    block_manager.ensure_capacity(
        request=request_a,
        total_tokens=4,
    )

    shared_block = request_a.block_table[0]

    prefix_cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[shared_block],
    )

    # B needs 6 tokens total:
    # 4 cached + 2 uncached.
    request_b = make_request(
        request_id="B",
        input_ids=[10, 11, 12, 13, 99, 100],
    )

    scheduler.add_request(request_b)

    selected = scheduler._select_prefill_requests()

    assert request_b in selected

    assert request_b.block_table[0] == shared_block

    assert request_b.kv_tokens == 4

    # One shared block + one private suffix block.
    assert len(request_b.block_table) == 2

    assert request_b.block_table[1] != shared_block

    # A + cache + B
    assert (
        block_manager.get_ref_count(shared_block)
        == 3
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

    block_manager.ensure_capacity(
        request=request_a,
        total_tokens=4,
    )

    shared_block = request_a.block_table[0]

    prefix_cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[shared_block],
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

    block_manager.ensure_capacity(
        request=request_a,
        total_tokens=4,
    )

    shared_block = request_a.block_table[0]

    prefix_cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[shared_block],
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

    prefix_cache.insert(
        token_ids=[10, 11, 12, 13],
        block_ids=[shared_block_id],
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
