import pytest
import torch

from app.runtime.batch_builder import BatchBuilder
from app.runtime.request import Request


def make_request(
    request_id: str,
    input_ids: list[int],
    max_new_tokens: int = 32,
) -> Request:
    return Request(
        request_id=request_id,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
    )

def test_build_equal_length_prefill_batch() -> None:
    requests = [
        make_request("A", [10, 11, 12]),
        make_request("B", [20, 21, 22]),
        make_request("C", [30, 31, 32]),
    ]

    builder = BatchBuilder()

    batch = builder.build_equal_length_prefill_batch(
        requests
    )

    assert batch.request_ids == ["A", "B", "C"]
    assert batch.batch_size == 3
    assert batch.sequence_length == 3

    assert batch.input_ids.shape == (3, 3)
    assert batch.attention_mask.shape == (3, 3)

    assert torch.equal(
        batch.input_ids[0],
        torch.tensor([10, 11, 12]),
    )
    assert torch.equal(
        batch.input_ids[1],
        torch.tensor([20, 21, 22]),
    )
    assert torch.equal(
        batch.input_ids[2],
        torch.tensor([30, 31, 32]),
    )

    assert torch.equal(
        batch.attention_mask,
        torch.ones((3, 3), dtype=torch.long),
    )

def test_equal_length_prefill_rejects_empty_list() -> None:
    builder = BatchBuilder()

    with pytest.raises(
        ValueError,
        match="requests cannot be empty",
    ):
        builder.build_equal_length_prefill_batch([])
    
def test_equal_length_prefill_rejects_different_lengths() -> None:
    requests = [
        make_request("A", [10, 11]),
        make_request("B", [20, 21, 22]),
    ]

    builder = BatchBuilder()

    with pytest.raises(
        ValueError,
        match="equal prompt length",
    ):
        builder.build_equal_length_prefill_batch(
            requests
        )
    
