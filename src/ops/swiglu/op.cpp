#include "op.hpp"
#include <cmath>

namespace llaisys::ops {

template<typename T>
void swiglu_kernel(tensor_t out, tensor_t gate, tensor_t up) {
    size_t numel = gate->numel();
    
    const T* gate_data = reinterpret_cast<const T*>(gate->data());
    const T* up_data = reinterpret_cast<const T*>(up->data());
    T* out_data = reinterpret_cast<T*>(out->data());
    
    for (size_t i = 0; i < numel; ++i) {
        float gate_val = utils::cast<float>(gate_data[i]);
        float up_val = utils::cast<float>(up_data[i]);
        
        // 计算 Swish: gate / (1 + exp(-gate))
        float swish_val = gate_val / (1.0f + std::exp(-gate_val));
        
        // 计算最终结果: up * Swish
        float result = up_val * swish_val;
        out_data[i] = utils::cast<T>(result);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    llaisysDataType_t dtype = gate->dtype();
    
    if (dtype == LLAISYS_DTYPE_F32) {
        swiglu_kernel<float>(out, gate, up);
    } 
    else if (dtype == LLAISYS_DTYPE_F16) {
        swiglu_kernel<fp16_t>(out, gate, up);
    } 
    else if (dtype == LLAISYS_DTYPE_BF16) {
        swiglu_kernel<bf16_t>(out, gate, up);
    }
    else {
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops
