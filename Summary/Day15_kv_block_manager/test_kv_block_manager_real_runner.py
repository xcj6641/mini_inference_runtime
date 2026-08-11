import pytest

from app.runtime.kv_block_manager import (
    KVBlockManager,
    BlockAllocationError
)

from app.runtime.request import Request

def test_block_manager_initial_state() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=8,
    )

    assert manager.num_blocks == 4
    assert manager.block_size == 8

    assert manager.num_free_blocks == 4
    assert manager.num_allocated_blocks == 0
    assert manager.utilization == 0.0

    assert (
        manager.num_free_blocks
        + manager.num_allocated_blocks
        == manager.num_blocks
    )


@pytest.mark.parametrize(
    ("num_blocks", "block_size"),
    [
        (0, 8),
        (-1, 8),
        (4, 0),
        (4, -1),
    ],
)
def test_block_manager_rejects_invalid_configuration(
    num_blocks: int,
    block_size: int,
) -> None:
    with pytest.raises(ValueError):
        KVBlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )

@pytest.mark.parametrize(
    ("num_tokens", "expected_blocks"),
    [
        (0, 0),
        (1, 1),
        (7, 1),
        (8, 1),
        (9, 2),
        (16, 2),
        (17, 3),
    ],
)
def test_blocks_required_for_tokens(
    num_tokens: int,
    expected_blocks: int,
) -> None:
    manager = KVBlockManager(
        num_blocks=8,
        block_size=8,
    )

    actual_blocks = (
        manager.blocks_required_for_tokens(
            num_tokens
        )
    )

    assert actual_blocks == expected_blocks


def test_blocks_required_rejects_negative_tokens() -> None:
    manager = KVBlockManager(
        num_blocks=8,
        block_size=8,
    )

    with pytest.raises(
        ValueError,
        match="num_tokens must not be negative",
    ):
        manager.blocks_required_for_tokens(-1)

def test_request_starts_with_empty_block_table() -> None:
    request = Request(
        request_id="request-a",
        input_ids=[1, 2, 3],
        max_new_tokens=4,
    )

    assert request.block_table == []

def test_can_allocate_blocks() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=8,
    )

    assert manager.can_allocate(0)
    assert manager.can_allocate(4)
    assert not manager.can_allocate(5)


def test_can_allocate_tokens() -> None:
    manager = KVBlockManager(
        num_blocks=2,
        block_size=8,
    )

    assert manager.can_allocate_tokens(16)
    assert not manager.can_allocate_tokens(17)


def test_can_allocate_rejects_negative_blocks() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=8,
    )

    with pytest.raises(
        ValueError,
        match="num_blocks must not be negative",
    ):
        manager.can_allocate(-1)

def make_request(
    request_id: str = "request-a",
) -> Request:
    return Request(
        request_id=request_id,
        input_ids=[1, 2, 3],
        max_new_tokens=4,
    )

def test_allocate_blocks() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=8,
    )

    request = make_request()

    allocated = manager.allocate(
        request=request,
        num_blocks=2,
    )

    assert allocated == [0, 1]
    assert request.block_table == [0, 1]

    assert manager.num_free_blocks == 2
    assert manager.num_allocated_blocks == 2

    assert (
        manager.num_free_blocks
        + manager.num_allocated_blocks
        == manager.num_blocks
    )

def test_allocate_zero_blocks_changes_nothing() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=8,
    )

    request = make_request()

    allocated = manager.allocate(
        request=request,
        num_blocks=0,
    )

    assert allocated == []
    assert request.block_table == []

    assert manager.num_free_blocks == 4
    assert manager.num_allocated_blocks == 0

def test_allocation_failure_is_atomic() -> None:
    manager = KVBlockManager(
        num_blocks=2,
        block_size=4,
    )

    request = make_request()

    with pytest.raises(
        BlockAllocationError,
        match="Insufficient KV blocks",
    ):
        manager.allocate(
            request=request,
            num_blocks=3,
        )

    assert request.block_table == []

    assert manager.num_free_blocks == 2
    assert manager.num_allocated_blocks == 0

def test_requests_receive_disjoint_blocks() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=8,
    )

    request_a = make_request("request-a")
    request_b = make_request("request-b")

    manager.allocate(
        request=request_a,
        num_blocks=2,
    )

    manager.allocate(
        request=request_b,
        num_blocks=2,
    )

    assert request_a.block_table == [0, 1]
    assert request_b.block_table == [2, 3]

    assert set(
        request_a.block_table
    ).isdisjoint(
        request_b.block_table
    )

    assert manager.num_free_blocks == 0
    assert manager.num_allocated_blocks == 4

    assert manager.owner_of(0) == "request-a"
    assert manager.owner_of(2) == "request-b"


def test_ensure_capacity_allocates_incrementally() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request = make_request()

    first = manager.ensure_capacity(
        request=request,
        total_tokens=3,
    )

    assert first == [0]
    assert request.block_table == [0]
    assert manager.num_free_blocks == 3

    second = manager.ensure_capacity(
        request=request,
        total_tokens=4,
    )

    assert second == []
    assert request.block_table == [0]
    assert manager.num_free_blocks == 3

    third = manager.ensure_capacity(
        request=request,
        total_tokens=5,
    )

    assert third == [1]
    assert request.block_table == [0, 1]
    assert manager.num_free_blocks == 2


def test_ensure_capacity_does_not_double_allocate() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request = make_request()

    manager.ensure_capacity(
        request=request,
        total_tokens=6,
    )

    first_table = list(
        request.block_table
    )

    manager.ensure_capacity(
        request=request,
        total_tokens=6,
    )

    assert request.block_table == first_table
    assert request.block_table == [0, 1]

    assert manager.num_allocated_blocks == 2
    assert manager.num_free_blocks == 2

def test_ensure_capacity_does_not_shrink_blocks() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request = make_request()

    manager.ensure_capacity(
        request=request,
        total_tokens=8,
    )

    assert request.block_table == [0, 1]

    allocated = manager.ensure_capacity(
        request=request,
        total_tokens=2,
    )

    assert allocated == []
    assert request.block_table == [0, 1]

    assert manager.num_allocated_blocks == 2
    assert manager.num_free_blocks == 2

def test_ensure_capacity_failure_is_atomic() -> None:
    manager = KVBlockManager(
        num_blocks=2,
        block_size=4,
    )

    request = make_request()

    manager.ensure_capacity(
        request=request,
        total_tokens=4,
    )

    assert request.block_table == [0]

    with pytest.raises(
        BlockAllocationError,
        match="Insufficient KV blocks",
    ):
        manager.ensure_capacity(
            request=request,
            total_tokens=12,
        )

    assert request.block_table == [0]
    assert manager.num_allocated_blocks == 1
    assert manager.num_free_blocks == 1

def test_ensure_capacity_rejects_negative_tokens() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="total_tokens must not be negative",
    ):
        manager.ensure_capacity(
            request=request,
            total_tokens=-1,
        )

    assert request.block_table == []
    assert manager.num_free_blocks == 4

def test_additional_blocks_required() -> None:
    manager = KVBlockManager(
        num_blocks=8,
        block_size=4,
    )

    request = make_request()

    assert (
        manager.additional_blocks_required(
            request,
            total_tokens=0,
        )
        == 0
    )

    assert (
        manager.additional_blocks_required(
            request,
            total_tokens=5,
        )
        == 2
    )

    manager.ensure_capacity(
        request,
        total_tokens=5,
    )

    assert (
        manager.additional_blocks_required(
            request,
            total_tokens=8,
        )
        == 0
    )

    assert (
        manager.additional_blocks_required(
            request,
            total_tokens=9,
        )
        == 1
    )

def test_free_releases_request_blocks() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request = make_request()

    manager.allocate(
        request=request,
        num_blocks=2,
    )

    released = manager.free(request)

    assert released == [0, 1]
    assert request.block_table == []

    assert manager.num_free_blocks == 4
    assert manager.num_allocated_blocks == 0

    assert (
        manager.num_free_blocks
        + manager.num_allocated_blocks
        == manager.num_blocks
    )

def test_free_empty_request_is_idempotent() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request = make_request()

    manager.allocate(
        request=request,
        num_blocks=2,
    )

    first_release = manager.free(request)
    second_release = manager.free(request)

    assert first_release == [0, 1]
    assert second_release == []

    assert request.block_table == []
    assert manager.num_free_blocks == 4
    assert manager.num_allocated_blocks == 0

def test_released_blocks_are_reusable() -> None:
    manager = KVBlockManager(
        num_blocks=2,
        block_size=4,
    )

    request_a = make_request("request-a")
    request_b = make_request("request-b")

    allocated_a = manager.allocate(
        request=request_a,
        num_blocks=2,
    )

    assert allocated_a == [0, 1]
    assert manager.num_free_blocks == 0

    released = manager.free(request_a)

    assert released == [0, 1]
    assert manager.num_free_blocks == 2

    allocated_b = manager.allocate(
        request=request_b,
        num_blocks=2,
    )

    assert allocated_b == [0, 1]
    assert request_b.block_table == [0, 1]

def test_free_preserves_other_request_blocks() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request_a = make_request("request-a")
    request_b = make_request("request-b")

    manager.allocate(
        request=request_a,
        num_blocks=2,
    )

    manager.allocate(
        request=request_b,
        num_blocks=2,
    )

    released = manager.free(request_a)

    assert released == [0, 1]
    assert request_a.block_table == []

    assert request_b.block_table == [2, 3]

    assert manager.num_free_blocks == 2
    assert manager.num_allocated_blocks == 2

def test_free_rejects_wrong_block_owner() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request_a = make_request("request-a")
    request_b = make_request("request-b")

    manager.allocate(
        request=request_a,
        num_blocks=1,
    )

    manager.allocate(
        request=request_b,
        num_blocks=1,
    )

    assert request_a.block_table == [0]
    assert request_b.block_table == [1]

    request_a.block_table.append(1)

    with pytest.raises(
        RuntimeError,
        match="owned by request-b",
    ):
        manager.free(request_a)

    # Free must fail atomically.
    assert request_a.block_table == [0, 1]
    assert request_b.block_table == [1]

    assert manager.num_free_blocks == 2
    assert manager.num_allocated_blocks == 2

def test_free_rejects_unowned_block() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request = make_request()

    request.block_table.append(3)

    with pytest.raises(
        RuntimeError,
        match="Block 3 has no owner",
    ):
        manager.free(request)

    assert request.block_table == [3]
    assert manager.num_free_blocks == 4
    assert manager.num_allocated_blocks == 0

def test_ensure_batch_capacity_allocates_for_all_requests() -> None:
    manager = KVBlockManager(
        num_blocks=6,
        block_size=4,
    )

    request_a = make_request("request-a")
    request_b = make_request("request-b")

    allocated = manager.ensure_batch_capacity(
        [
            (request_a, 5),
            (request_b, 8),
        ]
    )

    assert allocated == {
        "request-a": [0, 1],
        "request-b": [2, 3],
    }

    assert request_a.block_table == [0, 1]
    assert request_b.block_table == [2, 3]

    assert manager.num_allocated_blocks == 4
    assert manager.num_free_blocks == 2

def test_ensure_batch_capacity_allocates_only_missing_blocks() -> None:
    manager = KVBlockManager(
        num_blocks=6,
        block_size=4,
    )

    request_a = make_request("request-a")
    request_b = make_request("request-b")

    manager.ensure_capacity(
        request=request_a,
        total_tokens=4,
    )

    manager.ensure_capacity(
        request=request_b,
        total_tokens=8,
    )

    assert request_a.block_table == [0]
    assert request_b.block_table == [1, 2]

    allocated = manager.ensure_batch_capacity(
        [
            (request_a, 5),
            (request_b, 9),
        ]
    )

    assert allocated == {
        "request-a": [3],
        "request-b": [4],
    }

    assert request_a.block_table == [0, 3]
    assert request_b.block_table == [1, 2, 4]

    assert manager.num_allocated_blocks == 5
    assert manager.num_free_blocks == 1

def test_ensure_batch_capacity_failure_is_atomic() -> None:
    manager = KVBlockManager(
        num_blocks=3,
        block_size=4,
    )

    request_a = make_request("request-a")
    request_b = make_request("request-b")

    manager.ensure_capacity(
        request=request_a,
        total_tokens=4,
    )

    assert request_a.block_table == [0]
    assert request_b.block_table == []

    with pytest.raises(
        BlockAllocationError,
        match="Insufficient KV blocks for batch",
    ):
        manager.ensure_batch_capacity(
            [
                (request_a, 9),
                (request_b, 5),
            ]
        )

    assert request_a.block_table == [0]
    assert request_b.block_table == []

    assert manager.num_allocated_blocks == 1
    assert manager.num_free_blocks == 2

def test_ensure_batch_capacity_changes_nothing_when_capacity_exists() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request_a = make_request("request-a")
    request_b = make_request("request-b")

    manager.ensure_capacity(
        request=request_a,
        total_tokens=6,
    )

    manager.ensure_capacity(
        request=request_b,
        total_tokens=3,
    )

    before_a = list(
        request_a.block_table
    )
    before_b = list(
        request_b.block_table
    )

    allocated = manager.ensure_batch_capacity(
        [
            (request_a, 7),
            (request_b, 4),
        ]
    )

    assert allocated == {
        "request-a": [],
        "request-b": [],
    }

    assert request_a.block_table == before_a
    assert request_b.block_table == before_b

    assert manager.num_allocated_blocks == 3
    assert manager.num_free_blocks == 1

def test_ensure_batch_capacity_accepts_empty_batch() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    allocated = manager.ensure_batch_capacity(
        []
    )

    assert allocated == {}
    assert manager.num_free_blocks == 4
    assert manager.num_allocated_blocks == 0

def test_ensure_batch_capacity_rejects_duplicate_request() -> None:
    manager = KVBlockManager(
        num_blocks=4,
        block_size=4,
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Duplicate request in batch",
    ):
        manager.ensure_batch_capacity(
            [
                (request, 4),
                (request, 8),
            ]
        )

    assert request.block_table == []
    assert manager.num_free_blocks == 4
    assert manager.num_allocated_blocks == 0