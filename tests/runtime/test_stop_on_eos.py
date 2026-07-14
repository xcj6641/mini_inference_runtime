from types import SimpleNamespace

import torch

from app.runtime.generation import generate


class EosRunner:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.eos_token_ids = {2}

    def prefill(self, input_ids):
        return SimpleNamespace(
            next_token_id=2,
            past_key_values=object(),
            logits=torch.zeros(1, input_ids.shape[1], 10),
        )

    def decode_tokens(
        self,
        token_ids,
        skip_special_tokens=True,
    ):
        return ""


def test_stop_on_eos() -> None:
    runner = EosRunner()
    input_ids = torch.tensor([[1, 3, 4]])

    result = generate(
        runner=runner,
        input_ids=input_ids,
        max_new_tokens=10,
    )

    assert result.generated_token_ids == [2]
    assert result.finish_reason == "eos"