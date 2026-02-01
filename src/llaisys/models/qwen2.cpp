#include "qwen2.hpp"
#include "llaisys/ops.h"
#include "llaisys/tensor.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <random>

namespace llaisys {
namespace models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta* meta, llaisysDeviceType_t device, int* device_ids, int ndevice) {
    // 1. 拷贝元信息
    memcpy(&_meta, meta, sizeof(LlaisysQwen2Meta));
    // 2. 设备信息
    _device = device;
    _device_ids.resize(ndevice);
    memcpy(_device_ids.data(), device_ids, ndevice * sizeof(int));
    // 3. 初始化权重数组（按nlayer分配）
    size_t nlayer = meta->nlayer;
    _weights.attn_norm_w = new llaisysTensor_t[nlayer];
    _weights.attn_q_w = new llaisysTensor_t[nlayer];
    _weights.attn_q_b = new llaisysTensor_t[nlayer];
    _weights.attn_k_w = new llaisysTensor_t[nlayer];
    _weights.attn_k_b = new llaisysTensor_t[nlayer];
    _weights.attn_v_w = new llaisysTensor_t[nlayer];
    _weights.attn_v_b = new llaisysTensor_t[nlayer];
    _weights.attn_o_w = new llaisysTensor_t[nlayer];
    _weights.mlp_norm_w = new llaisysTensor_t[nlayer];
    _weights.mlp_gate_w = new llaisysTensor_t[nlayer];
    _weights.mlp_up_w = new llaisysTensor_t[nlayer];
    _weights.mlp_down_w = new llaisysTensor_t[nlayer];
    // 4. 初始化KV-Cache
    _kv_caches.resize(nlayer);
}

Qwen2Model::~Qwen2Model() {
    // 释放权重数组
    delete[] _weights.attn_norm_w;
    delete[] _weights.attn_q_w;
    delete[] _weights.attn_q_b;
    delete[] _weights.attn_k_w;
    delete[] _weights.attn_k_b;
    delete[] _weights.attn_v_w;
    delete[] _weights.attn_v_b;
    delete[] _weights.attn_o_w;
    delete[] _weights.mlp_norm_w;
    delete[] _weights.mlp_gate_w;
    delete[] _weights.mlp_up_w;
    delete[] _weights.mlp_down_w;
    // 释放KV-Cache
    free_kv_cache();
}

LlaisysQwen2Weights* Qwen2Model::weights() {
    return &_weights;
}


void Qwen2Model::init_kv_cache() {
    // 为每层创建KV-Cache张量
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto& cache = _kv_caches[i];
        cache.cur_len = 0;
        // K/V缓存shape：[maxseq, nkvh, dh]
        size_t shape_k[] = {_meta.maxseq, _meta.nkvh, _meta.dh};
        cache.k_cache = tensorCreate(shape_k, 3, _meta.dtype, _device, _device_ids[0]);
        cache.v_cache = tensorCreate(shape_k, 3, _meta.dtype, _device, _device_ids[0]);
    }
}

void Qwen2Model::free_kv_cache() {
    for (auto& cache : _kv_caches) {
        if (cache.k_cache) tensorDestroy(cache.k_cache);
        if (cache.v_cache) tensorDestroy(cache.v_cache);
        cache.cur_len = 0;
    }
}

// 核心推理逻辑
int64_t Qwen2Model::infer(int64_t* token_ids, size_t ntoken) {
    // 1. Embedding层
    llaisysTensor_t x = forward_embedding(token_ids, ntoken);
    // 2. 遍历所有层
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        // 2.1 RMSNorm（attention输入）
        llaisysTensor_t x_norm = forward_rms_norm(x, _weights.attn_norm_w[i], _meta.epsilon);
        // 2.2 SelfAttention（带KV-Cache）
        llaisysTensor_t attn_out = forward_attention(x_norm, i, ntoken);
        // 2.3 残差连接
        // 获取x的形状信息用于创建输出张量
        size_t x_ndim = tensorGetNdim(x);
        size_t* x_shape = new size_t[x_ndim];
        tensorGetShape(x, x_shape);
        llaisysDataType_t x_dtype = tensorGetDataType(x);
        llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
        int x_device_id = tensorGetDeviceId(x);
        
        llaisysTensor_t x_residual_result = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
        llaisysAdd(x_residual_result, x, attn_out);
        tensorDestroy(x);
        x = x_residual_result;
        delete[] x_shape; // 释放动态分配的形状数组
        
        tensorDestroy(x_norm);
        tensorDestroy(attn_out);
        
        // 2.4 RMSNorm（MLP输入）
        llaisysTensor_t x_mlp_norm = forward_rms_norm(x, _weights.mlp_norm_w[i], _meta.epsilon);
        // 2.5 MLP（SwiGLU）
        llaisysTensor_t mlp_out = forward_mlp(x_mlp_norm, i);
        // 2.6 残差连接
        // 获取x的形状信息用于创建输出张量
        size_t x_ndim2 = tensorGetNdim(x);
        size_t* x_shape2 = new size_t[x_ndim2];
        tensorGetShape(x, x_shape2);
        llaisysDataType_t x_dtype2 = tensorGetDataType(x);
        llaisysDeviceType_t x_device_type2 = tensorGetDeviceType(x);
        int x_device_id2 = tensorGetDeviceId(x);
        
        llaisysTensor_t x_mlp_result = tensorCreate(x_shape2, x_ndim2, x_dtype2, x_device_type2, x_device_id2);
        llaisysAdd(x_mlp_result, x, mlp_out);
        tensorDestroy(x);
        x = x_mlp_result;
        delete[] x_shape2; // 释放动态分配的形状数组
        
        tensorDestroy(x_mlp_norm);
        tensorDestroy(mlp_out);
    }
    // 3. 最后的RMSNorm
    llaisysTensor_t x_final = forward_rms_norm(x, _weights.out_norm_w, _meta.epsilon);
    // 4. 输出Embedding（linear层）
    // 获取x_final的形状信息用于创建logits张量
    size_t x_final_ndim = tensorGetNdim(x_final);
    size_t* x_final_shape = new size_t[x_final_ndim];
    tensorGetShape(x_final, x_final_shape);
    // 修改最后一维的大小为词汇表大小
    x_final_shape[x_final_ndim-1] = _meta.voc;
    llaisysDataType_t x_final_dtype = tensorGetDataType(x_final);
    llaisysDeviceType_t x_final_device_type = tensorGetDeviceType(x_final);
    int x_final_device_id = tensorGetDeviceId(x_final);
    
    llaisysTensor_t logits = tensorCreate(x_final_shape, x_final_ndim, x_final_dtype, x_final_device_type, x_final_device_id);
    delete[] x_final_shape;
    
    llaisysLinear(logits, x_final, _weights.out_embed, nullptr);  
    
    // 5. ArgMax取最大概率token
    // 为ArgMax创建输出张量
    size_t argmax_shape[] = {_meta.voc}; 
    llaisysTensor_t max_idx = tensorCreate(argmax_shape, 1, LLAISYS_DTYPE_I64, x_final_device_type, x_final_device_id);
    llaisysTensor_t max_val = tensorCreate(argmax_shape, 1, x_final_dtype, x_final_device_type, x_final_device_id);
    
    llaisysArgmax(max_idx, max_val, logits);  
    
    int64_t next_token = *reinterpret_cast<int64_t*>(tensorGetData(max_idx));
    
    // 6. 释放临时张量
    tensorDestroy(x);
    tensorDestroy(x_final);
    tensorDestroy(logits);
    tensorDestroy(max_idx);
    tensorDestroy(max_val);
    return next_token;
}

// 添加采样推理的实现
int64_t Qwen2Model::infer_with_sampling(int64_t* token_ids, size_t ntoken, int top_k, float top_p, float temperature) {
    // 1. Embedding层
    llaisysTensor_t x = forward_embedding(token_ids, ntoken);
    // 2. 遍历所有层
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        // 2.1 RMSNorm（attention输入）
        llaisysTensor_t x_norm = forward_rms_norm(x, _weights.attn_norm_w[i], _meta.epsilon);
        // 2.2 SelfAttention（带KV-Cache）
        llaisysTensor_t attn_out = forward_attention(x_norm, i, ntoken);
        // 2.3 残差连接
        // 获取x的形状信息用于创建输出张量
        size_t x_ndim = tensorGetNdim(x);
        size_t* x_shape = new size_t[x_ndim];
        tensorGetShape(x, x_shape);
        llaisysDataType_t x_dtype = tensorGetDataType(x);
        llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
        int x_device_id = tensorGetDeviceId(x);
        
        llaisysTensor_t x_residual_result = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
        llaisysAdd(x_residual_result, x, attn_out);
        tensorDestroy(x);
        x = x_residual_result;
        delete[] x_shape; // 释放动态分配的形状数组
        
        tensorDestroy(x_norm);
        tensorDestroy(attn_out);
        
        // 2.4 RMSNorm（MLP输入）
        llaisysTensor_t x_mlp_norm = forward_rms_norm(x, _weights.mlp_norm_w[i], _meta.epsilon);
        // 2.5 MLP（SwiGLU）
        llaisysTensor_t mlp_out = forward_mlp(x_mlp_norm, i);
        // 2.6 残差连接
        // 获取x的形状信息用于创建输出张量
        size_t x_ndim2 = tensorGetNdim(x);
        size_t* x_shape2 = new size_t[x_ndim2];
        tensorGetShape(x, x_shape2);
        llaisysDataType_t x_dtype2 = tensorGetDataType(x);
        llaisysDeviceType_t x_device_type2 = tensorGetDeviceType(x);
        int x_device_id2 = tensorGetDeviceId(x);
        
        llaisysTensor_t x_mlp_result = tensorCreate(x_shape2, x_ndim2, x_dtype2, x_device_type2, x_device_id2);
        llaisysAdd(x_mlp_result, x, mlp_out);
        tensorDestroy(x);
        x = x_mlp_result;
        delete[] x_shape2; // 释放动态分配的形状数组
        
        tensorDestroy(x_mlp_norm);
        tensorDestroy(mlp_out);
    }
    // 3. 最后的RMSNorm
    llaisysTensor_t x_final = forward_rms_norm(x, _weights.out_norm_w, _meta.epsilon);
    // 4. 输出Embedding（linear层）
    // 获取x_final的形状信息用于创建logits张量
    size_t x_final_ndim = tensorGetNdim(x_final);
    size_t* x_final_shape = new size_t[x_final_ndim];
    tensorGetShape(x_final, x_final_shape);
    // 修改最后一维的大小为词汇表大小
    x_final_shape[x_final_ndim-1] = _meta.voc;
    llaisysDataType_t x_final_dtype = tensorGetDataType(x_final);
    llaisysDeviceType_t x_final_device_type = tensorGetDeviceType(x_final);
    int x_final_device_id = tensorGetDeviceId(x_final);
    
    llaisysTensor_t logits = tensorCreate(x_final_shape, x_final_ndim, x_final_dtype, x_final_device_type, x_final_device_id);
    delete[] x_final_shape;
    
    llaisysLinear(logits, x_final, _weights.out_embed, nullptr);  

    // 5. 应用温度缩放
    if (temperature != 1.0f) {
        // 对logits应用温度缩放: logits = logits / temperature
        // 这需要一个标量除法操作
        // 为了简化，我们先获取数据指针直接操作
        float* logits_data = reinterpret_cast<float*>(tensorGetData(logits));
        for (size_t i = 0; i < _meta.voc; ++i) {
            logits_data[i] /= temperature;
        }
    }

    // 6. Top-K 和 Top-P 采样
    // 这里需要实现Top-K和Top-P采样的逻辑
    // 为了实现Top-K和Top-P，我们需要获取logits数据并实现采样算法
    float* logits_data = reinterpret_cast<float*>(tensorGetData(logits));
    
    // 创建索引数组并根据logits值排序
    std::vector<std::pair<float, int>> logit_indices;
    for (size_t i = 0; i < _meta.voc; ++i) {
        logit_indices.push_back({logits_data[i], static_cast<int>(i)});
    }
    
    // 按logits值降序排序
    std::sort(logit_indices.begin(), logit_indices.end(), 
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });
    
    // 应用Top-K
    if (top_k > 0 && top_k < static_cast<int>(_meta.voc)) {
        logit_indices.resize(top_k);
    }
    
    // 应用Top-P (nucleus sampling)
    if (top_p > 0.0f && top_p < 1.0f) {
        // 计算softmax概率
        std::vector<float> probs;
        float max_logit = logit_indices[0].first;  // 用于数值稳定性
        float sum_probs = 0.0f;
        
        for (auto& item : logit_indices) {
            float prob = std::exp(item.first - max_logit);
            probs.push_back(prob);
            sum_probs += prob;
        }
        
        // 归一化概率
        for (auto& prob : probs) {
            prob /= sum_probs;
        }
        
        // 应用Top-P - 选择累积概率超过top_p的最小集合
        float cum_prob = 0.0f;
        size_t cutoff_idx = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cum_prob += probs[i];
            if (cum_prob >= top_p) {
                cutoff_idx = i + 1;
                break;
            }
        }
        
        if (cutoff_idx > 0) {
            logit_indices.resize(cutoff_idx);
            probs.resize(cutoff_idx);
        }
    }
    
    // 从筛选后的候选中随机采样
    // 使用标准库的随机数生成器
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    // 重新计算筛选后候选的softmax概率
    std::vector<float> final_probs;
    if (!logit_indices.empty()) {
        float max_logit = logit_indices[0].first;
        float sum_probs = 0.0f;
        
        for (auto& item : logit_indices) {
            float prob = std::exp(item.first - max_logit);
            final_probs.push_back(prob);
            sum_probs += prob;
        }
        
        // 归一化
        for (auto& prob : final_probs) {
            prob /= sum_probs;
        }
        
        // 使用累积概率进行随机采样
        std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
        int selected_idx = dist(gen);
        
        int64_t next_token = logit_indices[selected_idx].second;
        
        // 6. 释放临时张量
        tensorDestroy(x);
        tensorDestroy(x_final);
        tensorDestroy(logits);
        
        return next_token;
    } else {
        // 如果没有候选词，则返回第一个词（退化情况）
        tensorDestroy(x);
        tensorDestroy(x_final);
        tensorDestroy(logits);
        return 0;
    }
}

// 其他辅助函数（forward_embedding/forward_attention/forward_mlp/forward_rms_norm）

llaisysTensor_t Qwen2Model::forward_embedding(int64_t* token_ids, size_t ntoken) {
    llaisysTensor_t index_tensor = tensorCreate(&ntoken, 1, LLAISYS_DTYPE_I64, _device, _device_ids[0]);
    tensorLoad(index_tensor, token_ids);
    

    size_t embedding_shape[2] = {ntoken, _meta.hs};
    llaisysTensor_t out = tensorCreate(embedding_shape, 2, _meta.dtype, _device, _device_ids[0]);
    llaisysEmbedding(out, index_tensor, _weights.in_embed);
    tensorDestroy(index_tensor);
    return out;
}


llaisysTensor_t Qwen2Model::forward_attention(const llaisysTensor_t& x, size_t layer_idx, size_t seq_len) {
    // 获取当前层的权重参数
    llaisysTensor_t wq = _weights.attn_q_w[layer_idx];
    llaisysTensor_t wk = _weights.attn_k_w[layer_idx];
    llaisysTensor_t wv = _weights.attn_v_w[layer_idx];
    llaisysTensor_t wo = _weights.attn_o_w[layer_idx];
    
    // 获取x的形状信息
    size_t x_ndim = tensorGetNdim(x);
    size_t* x_shape = new size_t[x_ndim];
    tensorGetShape(x, x_shape);
    llaisysDataType_t x_dtype = tensorGetDataType(x);
    llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
    int x_device_id = tensorGetDeviceId(x);
    
    // 计算Q、K、V的输出形状
    size_t q_shape[3] = {x_shape[0], x_shape[1], _meta.hs};
    size_t k_shape[3] = {x_shape[0], x_shape[1], (_meta.hs/_meta.nh)*_meta.nkvh};
    size_t v_shape[3] = {x_shape[0], x_shape[1], (_meta.hs/_meta.nh)*_meta.nkvh};
    
    // 创建Q、K、V张量
    llaisysTensor_t q = tensorCreate(q_shape, 3, x_dtype, x_device_type, x_device_id);
    llaisysTensor_t k = tensorCreate(k_shape, 3, x_dtype, x_device_type, x_device_id);
    llaisysTensor_t v = tensorCreate(v_shape, 3, x_dtype, x_device_type, x_device_id);
    
    // 计算Q、K、V
    llaisysLinear(q, x, wq, _weights.attn_q_b[layer_idx]);
    llaisysLinear(k, x, wk, _weights.attn_k_b[layer_idx]);
    llaisysLinear(v, x, wv, _weights.attn_v_b[layer_idx]);
    
    // 获取KV Cache
    auto& cache = _kv_caches[layer_idx];
    size_t cur_len = cache.cur_len;
    
    // 检查序列长度是否超出限制
    if (cur_len + x_shape[1] > _meta.maxseq) {
        throw std::runtime_error("Sequence length exceeds maximum allowed length");
    }
    
    
    // 获取数据指针
    void* k_cache_ptr = tensorGetData(cache.k_cache);
    void* v_cache_ptr = tensorGetData(cache.v_cache);
    void* k_ptr = tensorGetData(k);
    void* v_ptr = tensorGetData(v);
    
    // 根据数据类型计算元素大小
    size_t element_size;
    switch (_meta.dtype) {
        case 12: // float16
        case 19: // bfloat16
            element_size = sizeof(uint16_t);
            break;
        case 13: // float32
            element_size = sizeof(float);
            break;
        default:
            element_size = sizeof(float); // 默认
    }
    
    // 计算要复制的数据大小和目标偏移
    size_t k_elements_per_seq = ((_meta.hs/_meta.nh)*_meta.nkvh) * _meta.dh;
    size_t v_elements_per_seq = k_elements_per_seq;
    
    size_t k_dst_offset = cur_len * k_elements_per_seq;
    size_t v_dst_offset = cur_len * v_elements_per_seq;
    
    // 计算总复制字节数
    size_t k_copy_bytes = x_shape[1] * k_elements_per_seq * element_size;
    size_t v_copy_bytes = x_shape[1] * v_elements_per_seq * element_size;
    
    // 将当前K、V复制到cache的正确位置
    memcpy((char*)k_cache_ptr + k_dst_offset * element_size, k_ptr, k_copy_bytes);
    memcpy((char*)v_cache_ptr + v_dst_offset * element_size, v_ptr, v_copy_bytes);
    
    // 更新cache长度
    cache.cur_len += x_shape[1];
    
    // 使用当前Q和完整的KV缓存进行注意力计算
    size_t attn_out_shape[3] = {x_shape[0], x_shape[1], _meta.hs};
    llaisysTensor_t temp_attn = tensorCreate(attn_out_shape, 3, x_dtype, x_device_type, x_device_id);
    
    // 使用当前Q和完整的KV缓存进行注意力计算
    llaisysSelfAttention(temp_attn, q, cache.k_cache, cache.v_cache, 1.0f/_meta.hs);
    
    // 应用输出投影
    llaisysTensor_t output = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    llaisysLinear(output, temp_attn, wo, nullptr);
    
    // 清理临时张量（不要清理cache中的张量）
    delete[] x_shape;
    tensorDestroy(q);
    tensorDestroy(k);  // K已经被复制到cache，可以安全销毁
    tensorDestroy(v);  // V已经被复制到cache，可以安全销毁
    tensorDestroy(temp_attn);
    
    return output;
}
llaisysTensor_t Qwen2Model::forward_mlp(const llaisysTensor_t& x, size_t layer_idx) {
    // 获取当前层的MLP权重参数
    llaisysTensor_t gate_w = _weights.mlp_gate_w[layer_idx];
    llaisysTensor_t up_w = _weights.mlp_up_w[layer_idx];
    llaisysTensor_t down_w = _weights.mlp_down_w[layer_idx];
    
    // 获取x的形状信息
    size_t x_ndim = tensorGetNdim(x);
    size_t* x_shape = new size_t[x_ndim];
    tensorGetShape(x, x_shape);
    llaisysDataType_t x_dtype = tensorGetDataType(x);
    llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
    int x_device_id = tensorGetDeviceId(x);
    
    // 计算中间层的形状
    size_t intermediate_shape[2];
    for(size_t i = 0; i < x_ndim - 1; i++) {
        intermediate_shape[i] = x_shape[i];
    }
    intermediate_shape[x_ndim - 1] = _meta.di; // 中间层维度
    
    // 创建中间张量
    llaisysTensor_t gate_proj = tensorCreate(intermediate_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    llaisysTensor_t up_proj = tensorCreate(intermediate_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    
    // 执行SwiGLU操作：Swish(x*W_g) * (x*W_u)
    llaisysLinear(gate_proj, x, gate_w, nullptr);
    llaisysLinear(up_proj, x, up_w, nullptr);
    
    // 使用SwiGLU算子进行计算
    size_t output_shape[2];
    for(size_t i = 0; i < x_ndim - 1; i++) {
        output_shape[i] = x_shape[i];
    }
    output_shape[x_ndim - 1] = _meta.hs; // 输出维度等于隐藏层大小
    
    llaisysTensor_t swiglu_output = tensorCreate(output_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    llaisysSwiGLU(swiglu_output, gate_proj, up_proj);
    
    // 创建最终输出张量
    llaisysTensor_t output = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    
    // 通过down projection得到最终输出
    llaisysLinear(output, swiglu_output, down_w, nullptr);
    
    // 清理临时张量
    delete[] x_shape;
    tensorDestroy(gate_proj);
    tensorDestroy(up_proj);
    tensorDestroy(swiglu_output);
    
    return output;
}

llaisysTensor_t Qwen2Model::forward_rms_norm(const llaisysTensor_t& x, const llaisysTensor_t& weight, float eps) {
    // 获取x的形状信息用于创建输出张量
    size_t x_ndim = tensorGetNdim(x);
    size_t* x_shape = new size_t[x_ndim];
    tensorGetShape(x, x_shape);
    llaisysDataType_t x_dtype = tensorGetDataType(x);
    llaisysDeviceType_t x_device_type = tensorGetDeviceType(x);
    int x_device_id = tensorGetDeviceId(x);
    
    llaisysTensor_t output = tensorCreate(x_shape, x_ndim, x_dtype, x_device_type, x_device_id);
    delete[] x_shape;
    llaisysRmsNorm(output, x, weight, eps);  
    return output;
}

} // namespace models
} // namespace llaisys