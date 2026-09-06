from pathlib import Path

from torch.utils.cpp_extension import load


CUDA_DIR = Path(__file__).resolve().parent


# torch.utils.cpp_extension.load() does two things for us:

# .cpp + .cu
#    ↓
# compile
#    ↓
# shared library
#    ↓
# load into Python
# returns a Python-accessible extension module.

# Can I say that load() firstly creates a module named paged_attention_cuda,
# then compiles the C++ and CUDA code.
# When compling, the module will be the parameter m of PYBIND11_MODULE(TORCH_EXTENSION_NAME, m),
# and TORCH_EXTENSION_NAME  is the name of the module, which is paged_attention_cuda.
# Then in PYBIND11_MODULE(TORCH_EXTENSION_NAME, m), 
# the function m.def() will register the function paged_attention_cuda.forward to the module.
paged_attention_cuda = load(
    name="paged_attention_cuda",
    sources=[
        str(CUDA_DIR / "paged_attention.cpp"),
        str(CUDA_DIR / "paged_attention_kernel.cu"),
    ],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)