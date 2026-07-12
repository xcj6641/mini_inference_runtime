# Environment
source /home/ubuntu/vllm-env/bin/activate

# pytest
## run pytest through the same Python
python -m pytest -m integration -v
python -m pytest test/runtime -m integration -v
python -m pytest \
test/runtime/test_model_runner.py::test_real_prefill_returns_logits \
-v

## verify the markers
python -m pytest --collect-only -q
python -m pytest --collect-only -m "not integration" -q