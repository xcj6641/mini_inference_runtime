# tests/attention/test_reference_paged_attention.py

import torch

from app.attention.reference_paged_attention import (
    contiguous_attention_reference,
    paged_attention_reference,
    read_paged_kv_token,
    read_paged_kv_token_from_cache,
    paged_attention_reference_from_cache,
)
from app.runtime.paged_kv_cache import PagedKVCache

def test_contiguous_attention_reference() -> None:
    query = torch.tensor(
        [1.0, 0.0],
    )

    key = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    value = torch.tensor(
        [
            [10.0, 0.0],
            [0.0, 20.0],
        ]
    )

    output = contiguous_attention_reference(
        query=query,
        key=key,
        value=value,
    )

    assert output.shape == (2,)

    assert torch.isfinite(output).all()

def test_contiguous_attention_matches_manual_calculation() -> None:
    query = torch.tensor(
        [1.0, 2.0],
    )

    key = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    value = torch.tensor(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
        ]
    )

    output = contiguous_attention_reference(
        query=query,
        key=key,
        value=value,
    )

    scores = (
        query @ key.transpose(0, 1)
        / (query.shape[0] ** 0.5)
    )

    expected = (
        torch.softmax(scores, dim=-1)
        @ value
    )

    torch.testing.assert_close(
        output,
        expected,
    )

def test_read_paged_kv_token_with_scrambled_blocks() -> None:
    # Logical KV:
    #
    # token 0 = [10, 11]
    # token 1 = [20, 21]
    # token 2 = [30, 31]
    # token 3 = [40, 41]
    # token 4 = [50, 51]
    #
    # block_size = 2
    #
    # logical block 0 = tokens 0,1
    # logical block 1 = tokens 2,3
    # logical block 2 = token 4

    key_cache = torch.tensor(
        [
            # physical block 0:
            # logical tokens 2,3
            [
                [30.0, 31.0],
                [40.0, 41.0],
            ],

            # physical block 1:
            # logical token 4
            [
                [50.0, 51.0],
                [-1.0, -1.0],
            ],

            # physical block 2:
            # logical tokens 0,1
            [
                [10.0, 11.0],
                [20.0, 21.0],
            ],
        ]
    )

    value_cache = key_cache + 100.0

    block_table = [2, 0, 1]

    expected_keys = [
        [10.0, 11.0],
        [20.0, 21.0],
        [30.0, 31.0],
        [40.0, 41.0],
        [50.0, 51.0],
    ]

    for token_index in range(5):
        key, value = read_paged_kv_token(
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            token_index=token_index,
            block_size=2,
        )

        torch.testing.assert_close(
            key,
            torch.tensor(
                expected_keys[token_index]
            ),
        )

        torch.testing.assert_close(
            value,
            torch.tensor(
                expected_keys[token_index]
            ) + 100.0,
        )

def test_paged_attention_matches_contiguous_attention() -> None:
    query = torch.tensor(
        [
            [1.0, 2.0]
        ],
    )

    contiguous_key = torch.tensor(
        [
            [[10.0, 11.0]],
            [[20.0, 21.0]],
            [[30.0, 31.0]],
            [[40.0, 41.0]],
            [[50.0, 51.0]],
        ]
    )

    contiguous_value = torch.tensor(
        [
            [[100.0, 101.0]],
            [[200.0, 201.0]],
            [[300.0, 301.0]],
            [[400.0, 401.0]],
            [[500.0, 501.0]],
        ]
    )
    key_cache = torch.tensor(
        [
            # physical block 0
            # logical block 1
            [
                [[30.0, 31.0]],
                [[40.0, 41.0]],
            ],

            # physical block 1
            # logical block 2
            [
                [[50.0, 51.0]],
                [[-999.0, -999.0]],
            ],

            # physical block 2
            # logical block 0
            [
                [[10.0, 11.0]],
                [[20.0, 21.0]],
            ],
        ]
    )
    value_cache = torch.tensor(
        [
            [
                [[300.0, 301.0]],
                [[400.0, 401.0]],
            ],
            [
                [[500.0, 501.0]],
                [[-999.0, -999.0]],
            ],
            [
                [[100.0, 101.0]],
                [[200.0, 201.0]],
            ],
        ]
    )

    block_table = [2, 0, 1]

    expected = contiguous_attention_reference(
        query=query[0],
        key=contiguous_key[:,0,:],
        value=contiguous_value[:,0,:],
    )

    actual = paged_attention_reference(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_len=5,
        block_size=2,
    )

    torch.testing.assert_close(
        actual[0],
        expected,
    )

def test_multi_head_paged_attention() -> None:
    query = torch.tensor(
        [
            [1.0, 0.0],  # head 0
            [0.0, 1.0],  # head 1
        ]
    )

    # Shape:
    # [num_blocks=2, block_size=2, num_heads=2, head_dim=2]

    key_cache = torch.tensor(
        [
            [
                # token 0
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                # token 1
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
            ],
            [
                # token 2
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ],
                # unused
                [
                    [-999.0, -999.0],
                    [-999.0, -999.0],
                ],
            ],
        ]
    )

    value_cache = torch.tensor(
        [
            [
                [
                    [10.0, 11.0],
                    [100.0, 101.0],
                ],
                [
                    [20.0, 21.0],
                    [200.0, 201.0],
                ],
            ],
            [
                [
                    [30.0, 31.0],
                    [300.0, 301.0],
                ],
                [
                    [-999.0, -999.0],
                    [-999.0, -999.0],
                ],
            ],
        ]
    )

    output = paged_attention_reference(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=[0, 1],
        seq_len=3,
        block_size=2,
    )
    assert output.shape == (2, 2)

def test_multi_head_paged_attention_matches_contiguous_attention() -> None:
    query = torch.tensor(
        [
            [1.0, 0.0],  # head 0
            [0.0, 1.0],  # head 1
        ]
    )

    # Logical contiguous KV:
    # shape = [seq_len=3, num_heads=2, head_dim=2]
    contiguous_key = torch.tensor(
        [
            # token 0
            [
                [1.0, 0.0],  # head 0
                [0.0, 1.0],  # head 1
            ],
            # token 1
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            # token 2
            [
                [1.0, 1.0],
                [1.0, 1.0],
            ],
        ]
    )

    contiguous_value = torch.tensor(
        [
            # token 0
            [
                [10.0, 11.0],
                [100.0, 101.0],
            ],
            # token 1
            [
                [20.0, 21.0],
                [200.0, 201.0],
            ],
            # token 2
            [
                [30.0, 31.0],
                [300.0, 301.0],
            ],
        ]
    )

    # Physical layout is intentionally scrambled.
    #
    # block_size = 2
    #
    # logical block 0:
    #   token 0
    #   token 1
    #
    # logical block 1:
    #   token 2
    #
    # physical block 0 stores logical block 1
    # physical block 1 stores logical block 0
    #
    # Therefore:
    # block_table = [1, 0]

    key_cache = torch.tensor(
        [
            # physical block 0
            # logical block 1
            [
                # token 2
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                ],
                # unused slot
                [
                    [-999.0, -999.0],
                    [-999.0, -999.0],
                ],
            ],

            # physical block 1
            # logical block 0
            [
                # token 0
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                # token 1
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
            ],
        ]
    )

    value_cache = torch.tensor(
        [
            # physical block 0
            [
                # token 2
                [
                    [30.0, 31.0],
                    [300.0, 301.0],
                ],
                # unused slot
                [
                    [-999.0, -999.0],
                    [-999.0, -999.0],
                ],
            ],

            # physical block 1
            [
                # token 0
                [
                    [10.0, 11.0],
                    [100.0, 101.0],
                ],
                # token 1
                [
                    [20.0, 21.0],
                    [200.0, 201.0],
                ],
            ],
        ]
    )

    actual = paged_attention_reference(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=[1, 0],
        seq_len=3,
        block_size=2,
    )

    expected_outputs = []

    for head_index in range(query.shape[0]):
        head_output = contiguous_attention_reference(
            query=query[head_index],
            key=contiguous_key[:, head_index, :],
            value=contiguous_value[:, head_index, :],
        )

        expected_outputs.append(head_output)

    expected = torch.stack(
        expected_outputs,
        dim=0,
    )

    assert actual.shape == (2, 2)

    torch.testing.assert_close(
        actual,
        expected,
    )

def test_gqa_paged_attention_matches_contiguous_attention() -> None:
    # 4 query heads, 2 KV heads.
    #
    # Q0, Q1 -> KV head 0
    # Q2, Q3 -> KV head 1
    query = torch.tensor(
        [
            [1.0, 0.0],   # Q head 0
            [0.5, 1.0],   # Q head 1
            [0.0, 1.0],   # Q head 2
            [1.0, 1.0],   # Q head 3
        ]
    )

    # Logical KV layout.
    #
    # shape:
    # [seq_len=3, num_kv_heads=2, head_dim=2]
    contiguous_key = torch.tensor(
        [
            # token 0
            [
                [1.0, 0.0],  # KV head 0
                [0.0, 1.0],  # KV head 1
            ],
            # token 1
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            # token 2
            [
                [1.0, 1.0],
                [2.0, 1.0],
            ],
        ]
    )

    contiguous_value = torch.tensor(
        [
            # token 0
            [
                [10.0, 11.0],    # KV head 0
                [100.0, 101.0],  # KV head 1
            ],
            # token 1
            [
                [20.0, 21.0],
                [200.0, 201.0],
            ],
            # token 2
            [
                [30.0, 31.0],
                [300.0, 301.0],
            ],
        ]
    )

    # block_size = 2
    #
    # Logical block 0:
    #   token 0
    #   token 1
    #
    # Logical block 1:
    #   token 2
    #
    # Physical placement is intentionally scrambled:
    #
    # physical block 0 -> logical block 1
    # physical block 1 -> logical block 0
    #
    # Therefore:
    # block_table = [1, 0]

    # shape:
    # [num_blocks=2, block_size=2, num_kv_heads=2, head_dim=2]
    key_cache = torch.tensor(
        [
            # physical block 0
            # stores logical block 1
            [
                # token 2
                [
                    [1.0, 1.0],
                    [2.0, 1.0],
                ],
                # unused slot
                [
                    [-999.0, -999.0],
                    [-999.0, -999.0],
                ],
            ],

            # physical block 1
            # stores logical block 0
            [
                # token 0
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                # token 1
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                ],
            ],
        ]
    )

    value_cache = torch.tensor(
        [
            # physical block 0
            [
                # token 2
                [
                    [30.0, 31.0],
                    [300.0, 301.0],
                ],
                # unused slot
                [
                    [-999.0, -999.0],
                    [-999.0, -999.0],
                ],
            ],

            # physical block 1
            [
                # token 0
                [
                    [10.0, 11.0],
                    [100.0, 101.0],
                ],
                # token 1
                [
                    [20.0, 21.0],
                    [200.0, 201.0],
                ],
            ],
        ]
    )

    actual = paged_attention_reference(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=[1, 0],
        seq_len=3,
        block_size=2,
    )

    num_attention_heads = query.shape[0]
    num_kv_heads = contiguous_key.shape[1]

    assert num_attention_heads % num_kv_heads == 0

    num_queries_per_kv_head = (
        num_attention_heads // num_kv_heads
    )

    expected_outputs = []

    for attention_head_index in range(
        num_attention_heads
    ):
        kv_head_index = (
            attention_head_index
            // num_queries_per_kv_head
        )

        head_output = contiguous_attention_reference(
            query=query[attention_head_index],
            key=contiguous_key[
                :,
                kv_head_index,
                :,
            ],
            value=contiguous_value[
                :,
                kv_head_index,
                :,
            ],
        )

        expected_outputs.append(head_output)

    expected = torch.stack(
        expected_outputs,
        dim=0,
    )

    assert actual.shape == (4, 2)

    torch.testing.assert_close(
        actual,
        expected,
    )

################# intergration with paged KV Cache
def test_read_paged_kv_token_from_real_cache() -> None:
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=3,
        num_kv_heads=2,
        block_size=2,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    block_table = [2, 0, 1]

    paged_kv_cache.key_cache[
        0,  # layer
        2,  # physical block
        :,  # KV heads
        0,  # block offset
        :,
    ] = torch.tensor(
        [
            [10.0, 11.0],
            [110.0, 111.0],
        ]
    )

    paged_kv_cache.key_cache[
        0,
        2,
        :,
        1,
        :,
    ] = torch.tensor(
        [
            [20.0, 21.0],
            [120.0, 121.0],
        ]
    )

    paged_kv_cache.key_cache[
        0,
        0,
        :,
        0,
        :,
    ] = torch.tensor(
        [
            [30.0, 31.0],
            [130.0, 131.0],
        ]
    )

    paged_kv_cache.key_cache[
        0,
        0,
        :,
        1,
        :,
    ] = torch.tensor(
        [
            [40.0, 41.0],
            [140.0, 141.0],
        ]
    )

    paged_kv_cache.key_cache[
        0,
        1,
        :,
        0,
        :,
    ] = torch.tensor(
        [
            [50.0, 51.0],
            [150.0, 151.0],
        ]
    )

    paged_kv_cache.value_cache.copy_(
        paged_kv_cache.key_cache + 1000.0
    )

    expected_keys = [
        [
            [10.0, 11.0],
            [110.0, 111.0],
        ],
        [
            [20.0, 21.0],
            [120.0, 121.0],
        ],
        [
            [30.0, 31.0],
            [130.0, 131.0],
        ],
        [
            [40.0, 41.0],
            [140.0, 141.0],
        ],
        [
            [50.0, 51.0],
            [150.0, 151.0],
        ],
    ]

    for token_index in range(5):
        key, value = read_paged_kv_token_from_cache(
            paged_kv_cache=paged_kv_cache,
            block_table=block_table,
            layer_index=0,
            token_index=token_index,
        )

        expected_key = torch.tensor(
            expected_keys[token_index]
        )

        torch.testing.assert_close(
            key,
            expected_key,
        )

        torch.testing.assert_close(
            value,
            expected_key + 1000.0,
        )

    assert key.shape == (2, 2)

def test_paged_attention_from_real_cache_matches_contiguous_attention() -> None:
    paged_kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=3,
        num_kv_heads=2,
        block_size=2,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )

    # 4 query heads, 2 KV heads:
    #
    # Q0, Q1 -> KV head 0
    # Q2, Q3 -> KV head 1
    query = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    # Logical KV:
    # [seq_len=5, num_kv_heads=2, head_dim=2]
    contiguous_key = torch.tensor(
        [
            [
                [1.0, 0.0],
                [10.0, 0.0],
            ],
            [
                [0.0, 1.0],
                [0.0, 10.0],
            ],
            [
                [1.0, 1.0],
                [10.0, 10.0],
            ],
            [
                [2.0, 1.0],
                [20.0, 10.0],
            ],
            [
                [1.0, 2.0],
                [10.0, 20.0],
            ],
        ]
    )

    contiguous_value = torch.tensor(
        [
            [
                [10.0, 11.0],
                [100.0, 101.0],
            ],
            [
                [20.0, 21.0],
                [200.0, 201.0],
            ],
            [
                [30.0, 31.0],
                [300.0, 301.0],
            ],
            [
                [40.0, 41.0],
                [400.0, 401.0],
            ],
            [
                [50.0, 51.0],
                [500.0, 501.0],
            ],
        ]
    )

    # block_size = 2
    #
    # logical block 0 -> tokens 0,1
    # logical block 1 -> tokens 2,3
    # logical block 2 -> token 4
    #
    # Physical placement is scrambled:
    #
    # physical block 0 -> logical block 1
    # physical block 1 -> logical block 2
    # physical block 2 -> logical block 0
    #
    # Therefore:
    block_table = [2, 0, 1]

    # --------------------------------------------------
    # physical block 2 stores logical tokens 0 and 1
    # --------------------------------------------------

    paged_kv_cache.key_cache[
        0, 2, :, 0, :
    ] = contiguous_key[0]

    paged_kv_cache.value_cache[
        0, 2, :, 0, :
    ] = contiguous_value[0]

    paged_kv_cache.key_cache[
        0, 2, :, 1, :
    ] = contiguous_key[1]

    paged_kv_cache.value_cache[
        0, 2, :, 1, :
    ] = contiguous_value[1]

    # --------------------------------------------------
    # physical block 0 stores logical tokens 2 and 3
    # --------------------------------------------------

    paged_kv_cache.key_cache[
        0, 0, :, 0, :
    ] = contiguous_key[2]

    paged_kv_cache.value_cache[
        0, 0, :, 0, :
    ] = contiguous_value[2]

    paged_kv_cache.key_cache[
        0, 0, :, 1, :
    ] = contiguous_key[3]

    paged_kv_cache.value_cache[
        0, 0, :, 1, :
    ] = contiguous_value[3]

    # --------------------------------------------------
    # physical block 1 stores logical token 4
    # --------------------------------------------------

    paged_kv_cache.key_cache[
        0, 1, :, 0, :
    ] = contiguous_key[4]

    paged_kv_cache.value_cache[
        0, 1, :, 0, :
    ] = contiguous_value[4]

    # Unused second slot of the final block.
    # This MUST NOT affect the attention result.
    paged_kv_cache.key_cache[
        0, 1, :, 1, :
    ] = torch.tensor(
        [
            [-999.0, -999.0],
            [-999.0, -999.0],
        ]
    )

    paged_kv_cache.value_cache[
        0, 1, :, 1, :
    ] = torch.tensor(
        [
            [-999.0, -999.0],
            [-999.0, -999.0],
        ]
    )

    actual = paged_attention_reference_from_cache(
        query=query,
        paged_kv_cache=paged_kv_cache,
        block_table=block_table,
        layer_index=0,
        seq_len=5,
    )

    # --------------------------------------------------
    # Build expected result using contiguous attention.
    # --------------------------------------------------

    num_attention_heads = query.shape[0]
    num_kv_heads = contiguous_key.shape[1]

    assert num_attention_heads % num_kv_heads == 0

    num_queries_per_kv_head = (
        num_attention_heads // num_kv_heads
    )

    expected_outputs = []

    for attention_head_index in range(
        num_attention_heads
    ):
        kv_head_index = (
            attention_head_index
            // num_queries_per_kv_head
        )

        expected_outputs.append(
            contiguous_attention_reference(
                query=query[
                    attention_head_index
                ],
                key=contiguous_key[
                    :,
                    kv_head_index,
                    :,
                ],
                value=contiguous_value[
                    :,
                    kv_head_index,
                    :,
                ],
            )
        )

    expected = torch.stack(
        expected_outputs,
        dim=0,
    )

    assert actual.shape == (4, 2)

    torch.testing.assert_close(
        actual,
        expected,
    )

import math

import pytest
import torch

from app.runtime.paged_kv_cache import PagedKVCache
from app.attention.reference_paged_attention import (
    contiguous_attention_reference,
    paged_attention_reference_from_cache,
)


@pytest.mark.parametrize(
    (
        "seq_len",
        "block_size",
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
    ),
    [
        # Smallest case.
        (1, 2, 1, 1, 4),

        # MHA:
        # one query head per KV head.
        (3, 2, 2, 2, 4),

        # GQA:
        # two query heads share one KV head.
        (5, 2, 4, 2, 8),

        # MQA-like case:
        # all query heads share one KV head.
        (7, 4, 4, 1, 8),

        # Stronger GQA case:
        # four query heads share one KV head.
        (9, 4, 8, 2, 4),
    ],
)
def test_randomized_paged_attention_matches_contiguous_attention(
    seq_len: int,
    block_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    # Make the random test deterministic.
    torch.manual_seed(1234)

    num_logical_blocks = math.ceil(
        seq_len / block_size
    )

    # Add extra unused physical blocks so that:
    #
    # logical block count != total physical block count
    #
    # This makes the storage setup more realistic.
    num_physical_blocks = (
        num_logical_blocks + 2
    )

    paged_kv_cache = PagedKVCache(
        num_layers=1,
        num_blocks=num_physical_blocks,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
    )

    # --------------------------------------------------
    # Step 1:
    # Build random logical contiguous Q / K / V.
    # --------------------------------------------------

    query = torch.randn(
        num_attention_heads,
        head_dim,
    )

    contiguous_key = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
    )

    contiguous_value = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
    )

    # --------------------------------------------------
    # Step 2:
    # Randomly assign logical blocks to physical blocks.
    # --------------------------------------------------

    physical_block_order = torch.randperm(
        num_physical_blocks
    ).tolist()

    block_table = physical_block_order[
        :num_logical_blocks
    ]

    assert len(block_table) == (
        num_logical_blocks
    )

    assert len(set(block_table)) == (
        num_logical_blocks
    )

    # --------------------------------------------------
    # Step 3:
    # Fill the whole physical cache with garbage first.
    #
    # This helps catch bugs where attention accidentally
    # reads:
    #
    # - unused blocks
    # - unused slots
    # - wrong physical blocks
    # --------------------------------------------------

    paged_kv_cache.key_cache.fill_(
        -999.0
    )

    paged_kv_cache.value_cache.fill_(
        -777.0
    )

    # --------------------------------------------------
    # Step 4:
    # Scatter logical K / V into physical blocks.
    # --------------------------------------------------

    for token_index in range(seq_len):
        logical_block = (
            token_index // block_size
        )

        block_offset = (
            token_index % block_size
        )

        physical_block = block_table[
            logical_block
        ]

        paged_kv_cache.key_cache[
            0,
            physical_block,
            :,
            block_offset,
            :,
        ] = contiguous_key[
            token_index
        ]

        paged_kv_cache.value_cache[
            0,
            physical_block,
            :,
            block_offset,
            :,
        ] = contiguous_value[
            token_index
        ]

    # --------------------------------------------------
    # Step 5:
    # Run reference PagedAttention on the real cache.
    # --------------------------------------------------

    actual = (
        paged_attention_reference_from_cache(
            query=query,
            paged_kv_cache=paged_kv_cache,
            block_table=block_table,
            layer_index=0,
            seq_len=seq_len,
        )
    )

    # --------------------------------------------------
    # Step 6:
    # Compute expected output using logical contiguous KV.
    # --------------------------------------------------

    assert (
        num_attention_heads
        % num_kv_heads
        == 0
    )

    num_queries_per_kv_head = (
        num_attention_heads
        // num_kv_heads
    )

    expected_outputs = []

    for attention_head_index in range(
        num_attention_heads
    ):
        kv_head_index = (
            attention_head_index
            // num_queries_per_kv_head
        )

        head_output = (
            contiguous_attention_reference(
                query=query[
                    attention_head_index
                ],
                key=contiguous_key[
                    :,
                    kv_head_index,
                    :,
                ],
                value=contiguous_value[
                    :,
                    kv_head_index,
                    :,
                ],
            )
        )

        expected_outputs.append(
            head_output
        )

    expected = torch.stack(
        expected_outputs,
        dim=0,
    )

    # --------------------------------------------------
    # Step 7:
    # Compare.
    # --------------------------------------------------

    assert actual.shape == (
        num_attention_heads,
        head_dim,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

