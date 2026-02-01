from ctypes import *
import ctypes
import os

# 定义函数指针类型
LlaisysQwen2Model = c_void_p
LlaisysQwen2Weights = c_void_p
LlaisysTensor_t = c_void_p
LlaisysQwen2Meta = c_void_p


# 从共享库加载函数
# LIB_LLAISYS = CDLL("path_to_libllaisys")

# 尝试从环境变量或相对路径找到正确的库文件
lib_path = os.environ.get("LLAISYS_LIB_PATH", "./libllaisys.so")  # Linux
if os.name == 'nt':  # Windows
    lib_path = os.environ.get("LLAISYS_LIB_PATH", "./llaisys.dll")

# 尝试从安装目录查找
if not os.path.exists(lib_path):
    # 在当前目录的父级目录中查找
    for root, dirs, files in os.walk(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))):
        for file in files:
            if file.startswith("libllaisys") and (file.endswith(".so") or file.endswith(".dll") or file.endswith(".dylib")):
                lib_path = os.path.join(root, file)
                break

try:
    LIB_LLAISYS = CDLL(lib_path)
except OSError as e:
    print(f"Error loading library {lib_path}: {e}")
    print("Make sure the llaisys library is built and the path is correct.")
    raise

# 为头文件中定义的每个C API函数设置签名
LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
    POINTER(LlaisysQwen2Meta), 
    c_int, 
    POINTER(c_int), 
    c_int
]
LIB_LLAISYS.llaisysQwen2ModelCreate.restype = LlaisysQwen2Model

LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [LlaisysQwen2Model]
LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None

LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
    LlaisysQwen2Model, 
    POINTER(c_int64), 
    c_size_t
]
LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64

# 添加采样推理函数的签名
LIB_LLAISYS.llaisysQwen2ModelInferWithSampling.argtypes = [
    LlaisysQwen2Model, 
    POINTER(c_int64), 
    c_size_t,
    c_int,      # top_k
    c_float,    # top_p
    c_float     # temperature
]
LIB_LLAISYS.llaisysQwen2ModelInferWithSampling.restype = c_int64

# 获取模型权重的函数
LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [LlaisysQwen2Model]
LIB_LLAISYS.llaisysQwen2ModelWeights.restype = LlaisysQwen2Weights

# 权重加载接口（从内存加载单张量到权重结构体）
LIB_LLAISYS.llaisysQwen2WeightsLoadTensor.argtypes = [
    LlaisysQwen2Weights,      # weights
    c_char_p,                 # tensor_name
    LlaisysTensor_t           # tensor
]
LIB_LLAISYS.llaisysQwen2WeightsLoadTensor.restype = None

# 初始化KV-Cache（每轮推理前调用）
LIB_LLAISYS.llaisysQwen2ModelInitKVCache.argtypes = [LlaisysQwen2Model]
LIB_LLAISYS.llaisysQwen2ModelInitKVCache.restype = None

# 释放KV-Cache（推理结束后调用）
LIB_LLAISYS.llaisysQwen2ModelFreeKVCache.argtypes = [LlaisysQwen2Model]
LIB_LLAISYS.llaisysQwen2ModelFreeKVCache.restype = None