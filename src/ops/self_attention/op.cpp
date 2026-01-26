#include "op.hpp"
#include <cmath>
#include <algorithm>


namespace llaisys::ops {

template <typename T>
void self_attention_kernel(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 获取张量形状
    size_t qlen = q->shape()[0];      // 查询序列长度
    size_t nhead = q->shape()[1];     // 查询头数
    size_t qd = q->shape()[2];        // 查询维度
    size_t klen = k->shape()[0];      // 键序列长度
    size_t nkvh = k->shape()[1];      // 键值头数
    size_t kd = k->shape()[2];        // 键维度
    size_t vd = v->shape()[2];        // 值维度

    // 确保维度匹配
    if (qd != kd || qd != vd) {
        throw std::runtime_error("Query, Key, and Value dimensions must match");
    }

    
    size_t group_size = nhead / nkvh;

    // 获取数据指针
    const T* q_ptr = reinterpret_cast<const T*>(q->data());
    const T* k_ptr = reinterpret_cast<const T*>(k->data());
    const T* v_ptr = reinterpret_cast<const T*>(v->data());
    T* out_ptr = reinterpret_cast<T*>(attn_val->data());

    // 遍历查询序列
    for (size_t i = 0; i < qlen; ++i) {
        // 遍历查询头
        for (size_t h = 0; h < nhead; ++h) {
            // 计算当前查询头对应的键值头
            size_t kv_head = h / group_size;

            // 计算注意力分数
            std::vector<float> scores(klen);

            for (size_t j = 0; j < klen; ++j) {
                // 计算 Q[i,h,:] * K[j,kv_head,:]
                float score = 0.0f;
                for (size_t d = 0; d < qd; ++d) {
                    size_t q_idx = i * nhead * qd + h * qd + d;
                    size_t k_idx = j * nkvh * kd + kv_head * kd + d;
                    score += utils::cast<float>(q_ptr[q_idx]) * utils::cast<float>(k_ptr[k_idx]);
                }
                score *= scale;

                // 应用因果掩码
                if (j > i + (klen - qlen)) {
                    score = -std::numeric_limits<float>::infinity();
                }

                scores[j] = score;
            }

            // 计算带掩码的softmax
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < klen; ++j) {
                if (scores[j] != -std::numeric_limits<float>::infinity() && scores[j] > max_score) {
                    max_score = scores[j];
                }
            }

            // 计算exp并求和
            std::vector<float> exp_scores(klen);
            float sum_exp = 0.0f;
            for (size_t j = 0; j < klen; ++j) {
                if (scores[j] == -std::numeric_limits<float>::infinity()) {
                    exp_scores[j] = 0.0f;
                } else {
                    exp_scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += exp_scores[j];
                }
            }

            // 归一化
            for (size_t j = 0; j < klen; ++j) {
                if (sum_exp != 0.0f) {
                    exp_scores[j] /= sum_exp;
                }
            }

            // 计算输出: attention_weights * V
            for (size_t d = 0; d < vd; ++d) {
                float result = 0.0f;
                for (size_t j = 0; j < klen; ++j) {
                    size_t v_idx = j * nkvh * vd + kv_head * vd + d;
                    result += exp_scores[j] * utils::cast<float>(v_ptr[v_idx]);
                }
                
                size_t out_idx = i * nhead * vd + h * vd + d;
                out_ptr[out_idx] = utils::cast<T>(result);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    llaisysDataType_t type = q->dtype();

    if (type == LLAISYS_DTYPE_F32) {
        self_attention_kernel<float>(attn_val, q, k, v, scale);
    } 
    else if (type == LLAISYS_DTYPE_F16) {
        self_attention_kernel<fp16_t>(attn_val, q, k, v, scale);
    } 
    else if (type == LLAISYS_DTYPE_BF16) {
        self_attention_kernel<bf16_t>(attn_val, q, k, v, scale);
    }
    else {
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops