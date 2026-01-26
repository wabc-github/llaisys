#include "op.hpp"
#include <cmath>


namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 获取输入张量的维度信息
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    
    // 确保head_dim是偶数
    if (head_dim % 2 != 0) {
        throw std::runtime_error("Head dimension must be even for RoPE");
    }
    
    size_t half_head_dim = head_dim / 2;
    
    // 获取位置ID数组
    const int64_t* pos_ids_data = reinterpret_cast<const int64_t*>(pos_ids->data());
    
    // 获取数据类型
    llaisysDataType_t dtype = in->dtype();
    
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            const float* input_data = reinterpret_cast<const float*>(in->data());
            float* output_data = reinterpret_cast<float*>(out->data());
            
            for (size_t i = 0; i < seq_len; ++i) {
                int64_t pos_id = pos_ids_data[i];
                
                for (size_t h = 0; h < n_heads; ++h) {
                    for (size_t j = 0; j < half_head_dim; ++j) {
                        // 计算频率
                        float freq = pos_id / std::pow(theta, static_cast<float>(2 * j) / head_dim);
                        
                        // 计算旋转角度
                        float cos_val = std::cos(freq);
                        float sin_val = std::sin(freq);
                        
                        // 获取输入的a_i,j 和 b_i,j
                        size_t base_idx = i * n_heads * head_dim + h * head_dim;
                        float a_ij = input_data[base_idx + j];
                        float b_ij = input_data[base_idx + j + half_head_dim];
                        
                        // 应用旋转公式
                        // a'_ij = a_ij * cos(phi) - b_ij * sin(phi)
                        // b'_ij = b_ij * cos(phi) + a_ij * sin(phi)
                        output_data[base_idx + j] = a_ij * cos_val - b_ij * sin_val;
                        output_data[base_idx + j + half_head_dim] = b_ij * cos_val + a_ij * sin_val;
                    }
                }
            }
            break;
        }
        case LLAISYS_DTYPE_F16: {
            const fp16_t* input_data = reinterpret_cast<const fp16_t*>(in->data());
            fp16_t* output_data = reinterpret_cast<fp16_t*>(out->data());
            
            for (size_t i = 0; i < seq_len; ++i) {
                int64_t pos_id = pos_ids_data[i];
                
                for (size_t h = 0; h < n_heads; ++h) {
                    for (size_t j = 0; j < half_head_dim; ++j) {
                        // 计算频率
                        float freq = pos_id / std::pow(theta, static_cast<float>(2 * j) / head_dim);
                        
                        // 计算旋转角度
                        float cos_val = std::cos(freq);
                        float sin_val = std::sin(freq);
                        
                        // 获取输入的a_i,j 和 b_i,j，并转换为FP32进行计算
                        size_t base_idx = i * n_heads * head_dim + h * head_dim;
                        float a_ij = utils::cast<float>(input_data[base_idx + j]);
                        float b_ij = utils::cast<float>(input_data[base_idx + j + half_head_dim]);
                        
                        // 应用旋转公式
                        // a'_ij = a_ij * cos(phi) - b_ij * sin(phi)
                        // b'_ij = b_ij * cos(phi) + a_ij * sin(phi)
                        float a_prime = a_ij * cos_val - b_ij * sin_val;
                        float b_prime = b_ij * cos_val + a_ij * sin_val;
                        
                        // 将结果转换回FP16并存储
                        output_data[base_idx + j] = utils::cast<fp16_t>(a_prime);
                        output_data[base_idx + j + half_head_dim] = utils::cast<fp16_t>(b_prime);
                    }
                }
            }
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            const bf16_t* input_data = reinterpret_cast<const bf16_t*>(in->data());
            bf16_t* output_data = reinterpret_cast<bf16_t*>(out->data());
            
            for (size_t i = 0; i < seq_len; ++i) {
                int64_t pos_id = pos_ids_data[i];
                
                for (size_t h = 0; h < n_heads; ++h) {
                    for (size_t j = 0; j < half_head_dim; ++j) {
                        // 计算频率
                        float freq = pos_id / std::pow(theta, static_cast<float>(2 * j) / head_dim);
                        
                        // 计算旋转角度
                        float cos_val = std::cos(freq);
                        float sin_val = std::sin(freq);
                        
                        // 获取输入的a_i,j 和 b_i,j，并转换为FP32进行计算
                        size_t base_idx = i * n_heads * head_dim + h * head_dim;
                        float a_ij = utils::cast<float>(input_data[base_idx + j]);
                        float b_ij = utils::cast<float>(input_data[base_idx + j + half_head_dim]);
                        
                        // 应用旋转公式
                        // a'_ij = a_ij * cos(phi) - b_ij * sin(phi)
                        // b'_ij = b_ij * cos(phi) + a_ij * sin(phi)
                        float a_prime = a_ij * cos_val - b_ij * sin_val;
                        float b_prime = b_ij * cos_val + a_ij * sin_val;
                        
                        // 将结果转换回BF16并存储
                        output_data[base_idx + j] = utils::cast<bf16_t>(a_prime);
                        output_data[base_idx + j + half_head_dim] = utils::cast<bf16_t>(b_prime);
                    }
                }
            }
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops
