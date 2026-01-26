#include "op.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 获取输入参数维度信息
    size_t batch_size = in->shape()[0];  // M
    size_t in_features = in->shape()[1]; // K
    size_t out_features = weight->shape()[0]; // N
    size_t weight_features = weight->shape()[1]; // K
    
    // 确保输入特征数量匹配
    if (in_features != weight_features) {
        throw std::runtime_error("Input features and weight features must match");
    }
    
    // 获取数据类型
    llaisysDataType_t dtype = in->dtype();
    
    // 对于不同的数据类型执行矩阵乘法和加偏置
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            const float* input_data = reinterpret_cast<const float*>(in->data());
            const float* weight_data = reinterpret_cast<const float*>(weight->data());
            float* output_data = reinterpret_cast<float*>(out->data());
            const float* bias_data = bias ? reinterpret_cast<const float*>(bias->data()) : nullptr;
            
            // 执行矩阵乘法 Y = X * W^T
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < in_features; ++k) {
                        sum += input_data[i * in_features + k] * weight_data[j * in_features + k];
                    }
                    
                    // 加上偏置（如果存在）
                    if (bias_data) {
                        sum += bias_data[j];
                    }
                    
                    output_data[i * out_features + j] = sum;
                }
            }
            break;
        }
        case LLAISYS_DTYPE_F16: {
            const fp16_t* input_data = reinterpret_cast<const fp16_t*>(in->data());
            const fp16_t* weight_data = reinterpret_cast<const fp16_t*>(weight->data());
            fp16_t* output_data = reinterpret_cast<fp16_t*>(out->data());
            const fp16_t* bias_data = bias ? reinterpret_cast<const fp16_t*>(bias->data()) : nullptr;
            
            // 执行矩阵乘法 Y = X * W^T，使用FP32中间计算
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < in_features; ++k) {
                        sum += utils::cast<float>(input_data[i * in_features + k]) * 
                               utils::cast<float>(weight_data[j * in_features + k]);
                    }
                    
                    // 加上偏置（如果存在）
                    if (bias_data) {
                        sum += utils::cast<float>(bias_data[j]);
                    }
                    
                    output_data[i * out_features + j] = utils::cast<fp16_t>(sum);
                }
            }
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            const bf16_t* input_data = reinterpret_cast<const bf16_t*>(in->data());
            const bf16_t* weight_data = reinterpret_cast<const bf16_t*>(weight->data());
            bf16_t* output_data = reinterpret_cast<bf16_t*>(out->data());
            const bf16_t* bias_data = bias ? reinterpret_cast<const bf16_t*>(bias->data()) : nullptr;
            
            // 执行矩阵乘法 Y = X * W^T，使用FP32中间计算
            for (size_t i = 0; i < batch_size; ++i) {
                for (size_t j = 0; j < out_features; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < in_features; ++k) {
                        sum += utils::cast<float>(input_data[i * in_features + k]) * 
                               utils::cast<float>(weight_data[j * in_features + k]);
                    }
                    
                    // 加上偏置（如果存在）
                    if (bias_data) {
                        sum += utils::cast<float>(bias_data[j]);
                    }
                    
                    output_data[i * out_features + j] = utils::cast<bf16_t>(sum);
                }
            }
            break;
        }
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops