from types import SimpleNamespace

import torch

from app.runtime.generation import generate


def make_fake_kv_cache(
    *,
    batch_size: int,
    sequence_length: int,
):
    key = torch.zeros(
        (
            batch_size,
            1,  # num_kv_heads
            sequence_length,
            1,  # head_dim
        ),
        dtype=torch.float32,
    )

    value = torch.zeros(
        (
            batch_size,
            1,
            sequence_length,
            1,
        ),
        dtype=torch.float32,
    )

    # Legacy Hugging Face KV format:
    #
    # (
    #     (layer_0_key, layer_0_value),
    #     (layer_1_key, layer_1_value),
    #     ...
    # )
    return (
        (
            key,
            value,
        ),
    )


class EosRunner:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.eos_token_ids = {2}

    def prefill(
        self,
        input_ids: torch.Tensor,
    ):
        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]

        return SimpleNamespace(
            next_token_id=2,
            past_key_values=make_fake_kv_cache(
                batch_size=batch_size,
                sequence_length=sequence_length,
            ),
            logits=torch.zeros(
                batch_size,
                sequence_length,
                10,
            ),
        )

    def decode_tokens(
        self,
        token_ids,
        skip_special_tokens=True,
    ):
        return ""


def test_stop_on_eos() -> None:
    runner = EosRunner()

    input_ids = torch.tensor(
        [[1, 3, 4]]
    )

    result = generate(
        runner=runner,
        input_ids=input_ids,
        max_new_tokens=10,
    )

    assert result.generated_token_ids == [2]
    assert result.finish_reason == "eos"