import os
import platform
from pathlib import Path

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

__all__ = [
    "get_ext_modules",
    "get_build_ext",
]

_THIS_DIR = Path(__file__).parent.resolve()
_ROOT_DIR = _THIS_DIR.parent.parent.resolve()
_CSRC_DIR = _ROOT_DIR / "src" / "libtorchaudio"


def _get_build(var, default=False):
    if var not in os.environ:
        return default

    val = os.environ.get(var, "0")
    trues = ["1", "true", "TRUE", "on", "ON", "yes", "YES"]
    falses = ["0", "false", "FALSE", "off", "OFF", "no", "NO"]
    if val in trues:
        return True
    if val not in falses:
        print(f"WARNING: Unexpected environment variable value `{var}={val}`. " f"Expected one of {trues + falses}")
    return False


_BUILD_CPP_TEST = _get_build("BUILD_CPP_TEST", False)
_BUILD_RNNT = _get_build("BUILD_RNNT", True)
_USE_ROCM = _get_build("USE_ROCM", torch.backends.cuda.is_built() and torch.version.hip is not None)
_USE_CUDA = _get_build("USE_CUDA", torch.backends.cuda.is_built() and torch.version.hip is None)
_BUILD_ALIGN = _get_build("BUILD_ALIGN", True)
_BUILD_CUDA_CTC_DECODER = _get_build("BUILD_CUDA_CTC_DECODER", _USE_CUDA)
_USE_OPENMP = _get_build("USE_OPENMP", True) and "ATen parallel backend: OpenMP" in torch.__config__.parallel_info()
_TORCH_CUDA_ARCH_LIST = os.environ.get("TORCH_CUDA_ARCH_LIST", None)


def get_build_ext():
    return BuildExtension.with_options(no_python_abi_suffix=True)


def get_ext_modules():
    extra_compile_args = {
        "cxx": [
            "-DTORCH_TARGET_VERSION=0x020a000000000000",
        ],
    }

    if platform.system() != "Windows":
        extra_compile_args["cxx"].append("-fdiagnostics-color=always")

    extension = CppExtension

    if _USE_CUDA:
        extension = CUDAExtension
        extra_compile_args["cxx"].append("-DUSE_CUDA")
        extra_compile_args["nvcc"] = ["-O2", "-DUSE_CUDA"]

    sources = [
        "libtorchaudio.cpp",
        "utils.cpp",
        "lfilter.cpp",
        "overdrive.cpp",
    ]

    if _USE_CUDA:
        sources.append("iir_cuda.cu")

    if _BUILD_RNNT:
        sources.extend(
            [
                "rnnt/cpu/compute.cpp",
                "rnnt/compute.cpp",
            ]
        )
        if _USE_CUDA:
            sources.append("rnnt/gpu/compute.cu")

    if _BUILD_ALIGN:
        extra_compile_args["cxx"].append("-DINCLUDE_ALIGN")
        sources.extend(
            [
                "forced_align/cpu/compute.cpp",
                "forced_align/compute.cpp",
            ]
        )
        if _USE_CUDA:
            sources.append("forced_align/gpu/compute.cu")

    modules = [
        extension(
            name="torchaudio.lib._torchaudio",
            sources=[
                _CSRC_DIR / "_torchaudio.cpp",
                _CSRC_DIR / "utils.cpp",
            ],
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            include_dirs=[_CSRC_DIR.parent],
        ),
        extension(
            name="torchaudio.lib.libtorchaudio",
            sources=[_CSRC_DIR / s for s in sources],
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
            include_dirs=[_CSRC_DIR.parent],
        ),
    ]
    if _BUILD_CUDA_CTC_DECODER:
        modules.extend(
            [
                extension(
                    name="torchaudio.lib.libctc_prefix_decoder",
                    sources=[
                        _CSRC_DIR / "cuctc" / "src" / s
                        for s in ["ctc_prefix_decoder.cpp", "ctc_prefix_decoder_kernel_v2.cu", "python_binding.cpp"]
                    ],
                    py_limited_api=True,
                    extra_compile_args=extra_compile_args,
                    include_dirs=[_CSRC_DIR / "cuctc", _CSRC_DIR.parent],
                ),
            ]
        )

    return modules
