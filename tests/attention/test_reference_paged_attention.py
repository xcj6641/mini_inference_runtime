# tests/attention/test_reference_paged_attention.py

import torch

from app.attention.reference_paged_attention import (
    contiguous_attention_reference,
    paged_attention_reference,
    read_paged_kv_token,
)

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

