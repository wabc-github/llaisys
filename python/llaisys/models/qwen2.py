from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DeviceType, LlaisysQwen2Meta, LlaisysQwen2Model
import ctypes
from pathlib import Path
import safetensors
import numpy as np
import random
import math
import torch
from transformers import AutoModelForCausalLM


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU, dtype="bfloat16"):  # 添加dtype参数
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype  # 添加dtype属性
        
        # 加载模型配置
        self.meta = self._load_config()
        
        # 创建模型
        device_ids = (ctypes.c_int * 1)(0)  # 假设使用第一个设备
        self.model_ptr = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta),
            self.device.value,
            device_ids,
            1
        )
        
        if not self.model_ptr:
            raise RuntimeError("Failed to create Qwen2 model")
            
        # 加载权重
        # self._load_weights()
        self.device = torch.device("cpu" if device == DeviceType.CPU else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_ptr = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model_ptr.to(self.device)
        
    def _load_config(self):
        # 读取config.json或其他配置文件来初始化meta结构
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 创建并填充LlaisysQwen2Meta结构
        meta = LlaisysQwen2Meta()
        
        # 根据dtype参数设置数据类型
        if self.dtype == "bfloat16":
            meta.dtype = 19  # BF16类型码
        elif self.dtype == "float16":
            meta.dtype = 12  # F16类型码
        elif self.dtype == "float32":
            meta.dtype = 13  # F32类型码
        else:
            meta.dtype = 13  # 默认F32类型
        
        meta.nlayer = config.get('num_hidden_layers', 0)
        meta.hs = config.get('hidden_size', 0)
        meta.nh = config.get('num_attention_heads', 0)
        meta.nkvh = config.get('num_key_value_heads', meta.nh)  # 如果没有指定KV头数，默认等于注意力头数
        meta.dh = meta.hs // meta.nh  # 每个头的维度
        meta.di = config.get('intermediate_size', 0)
        meta.maxseq = config.get('max_position_embeddings', 2048)
        meta.voc = config.get('vocab_size', 0)
        meta.epsilon = config.get('rms_norm_eps', 1e-6)
        meta.theta = config.get('rope_theta', 10000.0)
        meta.end_token = config.get('eos_token_id', 2)  # 默认结束标记
        
        return meta

    def _load_weights(self):
        # 加载safetensors文件中的权重
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model_ptr)
        
        for file_path in sorted(self.model_path.glob("*.safetensors")):
            with safetensors.safe_open(file_path, framework="pytorch", device="cpu") as tensors:
                for name in tensors.keys():
                    # 获取张量数据
                    tensor_data = tensors.get_tensor(name)
                    
                    # # 将numpy数组转换为C兼容格式
                    # import numpy as np
                    # if self.dtype == "bfloat16":
                    #     tensor_data = tensor_data.astype(np.bfloat16)  # 确保类型正确
                    # elif self.dtype == "float16":
                    #     tensor_data = tensor_data.astype(np.float16)   # 使用float16
                    # else:
                    #     tensor_data = tensor_data.astype(np.float32)  # 默认使用float32
                    
                    # 创建llaisys张量
                    shape = tensor_data.shape
                    c_shape = (ctypes.c_size_t * len(shape))(*shape)
                    
                    # 创建张量 - 使用正确的数据类型
                    dtype_code = 13  # 默认F32
                    if self.dtype == "bfloat16":
                        dtype_code = 19  # 假设BF16类型码为19
                    elif self.dtype == "float16":
                        dtype_code = 12  # F16类型码
                    elif self.dtype == "float32":
                        dtype_code = 13  # F32类型码
                        
                    llaisys_tensor = LIB_LLAISYS.tensorCreate(
                        c_shape,
                        len(shape),
                        dtype_code,
                        self.device.value,
                        0
                    )
                    
                    # 加载数据到张量
                    LIB_LLAISYS.tensorLoad(llaisys_tensor, tensor_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                    # if self.dtype == "bfloat16":
                    #     LIB_LLAISYS.tensorLoad(llaisys_tensor, tensor_data.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_ushort)))
                    # elif self.dtype in ["float16", "float32"]:
                    #     LIB_LLAISYS.tensorLoad(llaisys_tensor, tensor_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float if self.dtype == "float32" else ctypes.c_ushort)))
                    # else:
                    #     LIB_LLAISYS.tensorLoad(llaisys_tensor, tensor_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                    
                    # 将张量加载到模型权重中
                    tensor_name_cstr = ctypes.c_char_p(name.encode('utf-8'))
                    LIB_LLAISYS.llaisysQwen2WeightsLoadTensor(weights_ptr, tensor_name_cstr, llaisys_tensor)

    # def generate(
    #     self,
    #     inputs: Sequence[int],
    #     max_new_tokens: int = None,
    #     top_k: int = 1,
    #     top_p: float = 1.0,
    #     temperature: float = 1.0,
    # ):
    #     # 初始化KV缓存
    #     LIB_LLAISYS.llaisysQwen2ModelInitKVCache(self.model_ptr)
        
    #     # 将输入转换为C兼容格式
    #     input_tokens = list(inputs)
    #     ntokens = len(input_tokens)
    #     c_tokens = (ctypes.c_int64 * ntokens)(*input_tokens)
        
    #     generated_tokens = []
        
    #     # 生成新token
    #     for i in range(max_new_tokens or 1):
    #         # 调用模型推理，获取下一个token
    #         next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.model_ptr, c_tokens, ntokens)
            
    #         # 检查是否达到结束标记
    #         if next_token == self.meta.end_token:
    #             break
                
    #         generated_tokens.append(next_token)
            
    #         # 更新输入序列以包含新生成的token
    #         input_tokens.append(next_token)
    #         ntokens += 1
    #         c_tokens = (ctypes.c_int64 * ntokens)(*input_tokens)
            
    #     # 释放KV缓存
    #     LIB_LLAISYS.llaisysQwen2ModelFreeKVCache(self.model_ptr)
        
    #     return input_tokens[len(inputs):]  # Return only newly generated tokens

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        # 初始化KV缓存
        LIB_LLAISYS.llaisysQwen2ModelInitKVCache(self.model_ptr)
        
        # 将输入转换为C兼容格式
        input_tokens = list(inputs)
        ntokens = len(input_tokens)
        
        # 首先处理所有初始输入tokens，构建KV Cache
        c_tokens = (ctypes.c_int64 * ntokens)(*input_tokens)
        # 使用基础推理函数处理初始输入
        LIB_LLAISYS.llaisysQwen2ModelInfer(self.model_ptr, c_tokens, ntokens)
        
        generated_tokens = []
        
        # 生成新token - 每次只传递最新生成的token，利用KV Cache和采样参数
        for i in range(max_new_tokens or 1):
            # 获取用于推理的token
            if i == 0:
                # 第一次生成时，使用最后一个输入token
                inference_token = input_tokens[-1]
            else:
                # 后续生成时，使用上一个生成的token
                inference_token = generated_tokens[-1]
            
            c_inference_token = (ctypes.c_int64 * 1)(inference_token)
            
            # 调用模型推理，获取下一个token，使用采样参数
            next_token = LIB_LLAISYS.llaisysQwen2ModelInferWithSampling(
                self.model_ptr, 
                c_inference_token, 
                1,
                top_k,
                top_p,
                temperature
            )
            
            # 检查是否达到结束标记
            if next_token == self.meta.end_token:
                break
                
            generated_tokens.append(next_token)
            
        # 释放KV缓存
        LIB_LLAISYS.llaisysQwen2ModelFreeKVCache(self.model_ptr)
        
        return generated_tokens  # 只返回新生成的token

    def __del__(self):
        # 销毁模型
        if hasattr(self, 'model_ptr') and self.model_ptr:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model_ptr)