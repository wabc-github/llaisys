import os
import sys
import ctypes
from pathlib import Path
from ctypes import *


from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops

# 定义函数指针类型
LlaisysQwen2Model = ctypes.c_void_p
LlaisysQwen2Weights = ctypes.c_void_p

# 定义LlaisysQwen2Meta结构体
class LlaisysQwen2Meta(ctypes.Structure):
    _fields_ = [
        ("dtype", c_int),                    # llaisysDataType_t
        ("nlayer", c_size_t),                # number of layers
        ("hs", c_size_t),                    # hidden size
        ("nh", c_size_t),                    # number of heads
        ("nkvh", c_size_t),                  # number of key-value heads
        ("dh", c_size_t),                    # dimension per head
        ("di", c_size_t),                    # intermediate dimension
        ("maxseq", c_size_t),                # maximum sequence length
        ("voc", c_size_t),                   # vocabulary size
        ("epsilon", c_float),                # epsilon for normalization
        ("theta", c_float),                  # theta for RoPE
        ("end_token", c_int64)               # end token id
    ]

def load_shared_library():
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)


__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "LlaisysQwen2Meta",  
    "LlaisysQwen2Model",  
    "LlaisysQwen2Weights",  
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
]
