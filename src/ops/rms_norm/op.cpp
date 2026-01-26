#include "op.hpp"
#include <cmath>


namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 获取输入张量的维度
    size_t batch_size = in->shape()[0];  // 行数
    size_t feature_size = in->shape()[1]; // 每行的元素数 (d)
    
    // 获取数据类型
    llaisysDataType_t dtype = in->dtype();
    
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            const float* input_data = reinterpret_cast<const float*>(in->data());
            const float* weight_data = reinterpret_cast<const float*>(weight->data());
            float* output_data = reinterpret_cast<float*>(out->data());
            
            for (size_t i = 0; i < batch_size; ++i) {
                // 计算当前行的RMS (Root Mean Square)
                float sum_sq = 0.0f;
                for (size_t j = 0; j < feature_size; ++j) {
                    float val = input_data[i * feature_size + j];
                    sum_sq += val * val;
                }
                
                // 计算均方根并加eps防止除零
                float mean_sq = sum_sq / static_cast<float>(feature_size);
                float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
                
                // 应用RMS Norm公式: Y_i = W_i * X_i * inv_rms
                for (size_t j = 0; j < feature_size; ++j) {
                    size_t idx = i * feature_size + j;
                    output_data[idx] = weight_data[j] * input_data[idx] * inv_rms;
                }
            }
            break;
        }
        case LLAISYS_DTYPE_F16: {
            const fp16_t* input_data = reinterpret_cast<const fp16_t*>(in->data());
            const fp16_t* weight_data = reinterpret_cast<const fp16_t*>(weight->data());
            fp16_t* output_data = reinterpret_cast<fp16_t*>(out->data());
            
            for (size_t i = 0; i < batch_size; ++i) {
                // 计算当前行的RMS (Root Mean Square)，使用FP32进行中间计算
                float sum_sq = 0.0f;
                for (size_t j = 0; j < feature_size; ++j) {
                    float val = utils::cast<float>(input_data[i * feature_size + j]);
                    sum_sq += val * val;
                }
                
                // 计算均方根并加eps防止除零
                float mean_sq = sum_sq / static_cast<float>(feature_size);
                float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
                
                // 应用RMS Norm公式: Y_i = W_i * X_i * inv_rms
                for (size_t j = 0; j < feature_size; ++j) {
                    size_t idx = i * feature_size + j;
                    float input_val = utils::cast<float>(input_data[idx]);
                    float weight_val = utils::cast<float>(weight_data[j]);
                    output_data[idx] = utils::cast<fp16_t>(weight_val * input_val * inv_rms);
                }
            }
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            const bf16_t* input_data = reinterpret_cast<const bf16_t*>(in->data());
            const bf16_t* weight_data = reinterpret_cast<const bf16_t*>(weight->data());
            bf16_t* output_data = reinterpret_cast<bf16_t*>(out->data());
            
            for (size_t i = 0; i < batch_size; ++i) {
                // 计算当前行的RMS (Root Mean Square)，使用FP32进行中间计算
                float sum_sq = 0.0f;
                for (size_t j = 0; j < feature_size; ++j) {
                    float val = utils::cast<float>(input_data[i * feature_size + j]);
                    sum_sq += val * val;
                }
                
                // 计算均方根并加eps防止除零
                float mean_sq = sum_sq / static_cast<float>(feature_size);
                float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
                
                // 应用RMS Norm公式: Y_i = W_i * X_i * inv_rms
                for (size_t j = 0; j < feature_size; ++j) {
                    size_t idx = i * feature_size + j;
                    float input_val = utils::cast<float>(input_data[idx]);
                    float weight_val = utils::cast<float>(weight_data[j]);
                    output_data[idx] = utils::cast<bf16_t>(weight_val * input_val * inv_rms);
                }
            }
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops
