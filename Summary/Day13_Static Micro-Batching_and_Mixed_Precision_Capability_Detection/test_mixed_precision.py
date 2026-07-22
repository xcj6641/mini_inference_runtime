import pytest
import torch

import gc


from app.runtime.pytorch_model_runner import PyTorchModelRunner
from test_batch_builder import make_request
from app.runtime.batch_builder import BatchBuilder

@pytest.mark.integration
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
def test_batched_prefill_is_finite_for_dtype(
    dtype,
) -> None:
    if (
        dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("BF16 is not supported")

    runner = PyTorchModelRunner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
        dtype=dtype,
    )

    prompt_a = "The capital of France is"
    prompt_b = "The capital of Germany is"

    input_ids_a_tensor = runner.encode_prompt(prompt_a)# 2d tensor
    input_ids_b_tensor = runner.encode_prompt(prompt_b)


    #1d list, fit Class Request
    input_ids_a = (
        input_ids_a_tensor
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    input_ids_b = (
        input_ids_b_tensor
        .squeeze(0)
        .detach()
        .cpu()
        .tolist()
    )

    requests = [
        make_request("A", input_ids_a),
        make_request("B", input_ids_b),
    ]

    batch = (
        BatchBuilder()
        .build_equal_length_prefill_batch(
            requests
        )
    )

    output = runner.prefill_batch(batch)

    assert torch.isfinite(output.logits).all()
    assert len(output.next_token_ids) == 2