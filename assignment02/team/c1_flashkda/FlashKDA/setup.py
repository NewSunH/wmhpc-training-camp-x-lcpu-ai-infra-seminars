import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.run(["git", "submodule", "update", "--init", "cutlass"])


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


def get_c1_vsplit_args():
    """Keep the C1 two-CTA recurrence prototype opt-in at build time."""
    return ["-DC1_VSPLIT_K2=1"] if is_flag_set("FLASH_KDA_C1_VSPLIT_K2") else []


def get_c1_state_only_args():
    """Build the R7 recurrence probe without K2 output materialization."""
    return ["-DC1_K2_STATE_ONLY=1"] if is_flag_set("FLASH_KDA_C1_STATE_ONLY") else []


def get_c1_output_stage_args():
    """Optionally override the K2 output pipeline depth for R8 ablations."""
    value = os.getenv("FLASH_KDA_C1_OUTPUT_STAGES")
    if value is None:
        return []
    if value not in {"1", "2", "3"}:
        raise ValueError("FLASH_KDA_C1_OUTPUT_STAGES must be 1, 2, or 3")
    return [f"-DC1_K2_OUTPUT_STAGES={value}"]


def get_c1_out_fp32_args():
    """Enable the R8 output FP32-accumulation epilogue probe."""
    return ["-DC1_K2_OUT_FP32_ACCUM=1"] if is_flag_set("FLASH_KDA_C1_OUT_FP32") else []


def get_c1_fuse_out_add_args():
    """Enable the R8 fused output conversion/add probe."""
    return ["-DC1_K2_FUSE_OUT_ADD=1"] if is_flag_set("FLASH_KDA_C1_FUSE_OUT_ADD") else []


SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]


def detect_cuda_arch():
    import torch

    if not torch.cuda.is_available():
        return None

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"{major}{minor}a"


def get_arch_flags():
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"

    requested = os.getenv("FLASH_KDA_CUDA_ARCHS", "auto").lower()
    if requested == "auto":
        arch = detect_cuda_arch()
        if arch is None:
            raise RuntimeError(
                "FLASH_KDA_CUDA_ARCHS=auto requires a visible CUDA device. "
                "Set FLASH_KDA_CUDA_ARCHS=all to build all supported archs."
            )
        archs = [arch]
    elif requested == "all":
        archs = SUPPORTED_CUDA_ARCHS
    else:
        archs = [arch.strip() for arch in requested.split(",") if arch.strip()]

    flags = []
    for arch in archs:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags


ext_modules = [
    CUDAExtension(
        name='flash_kda_C',
        sources=[
            'csrc/flash_kda.cpp',
            'csrc/smxx/fwd_launch.cu',
        ],
        include_dirs=[
            os.path.join(this_dir, 'cutlass', 'include'),
            os.path.join(this_dir, 'cutlass', 'examples', 'common'),
            os.path.join(this_dir, 'cutlass', 'tools', 'util', 'include'),
            os.path.join(this_dir, 'csrc'),
        ],
        extra_compile_args={
            'cxx': ['-O3', '-Wno-psabi'],
            'nvcc': [
                '-O3',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
                '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '--use_fast_math',
                '--ptxas-options=-v,--register-usage-level=10,--warn-on-spills',
                '-lineinfo',
                *get_nvcc_thread_args(),
                *get_arch_flags(),
                *get_c1_vsplit_args(),
                *get_c1_state_only_args(),
                *get_c1_output_stage_args(),
                *get_c1_out_fp32_args(),
                *get_c1_fuse_out_add_args(),
            ],
        },
    )
]
cmdclass = {"build_ext": BuildExtension}

rev = os.getenv("FLASH_KDA_VERSION_SUFFIX", "")
if not rev:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        rev = "+" + subprocess.check_output(cmd, cwd=this_dir).decode("ascii").rstrip()
    except Exception:
        rev = ""

setup(
    name='flash_kda',
    version='0.0.1' + rev,
    description='FlashKDA: Flash Kimi Delta Attention',
    ext_modules=ext_modules,
    packages=['flash_kda'],
    cmdclass=cmdclass,
    zip_safe=False,
)
