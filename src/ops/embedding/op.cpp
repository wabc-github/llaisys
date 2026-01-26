#include "op.hpp"
#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    size_t out_rows = out->shape()[0];
    size_t out_cols = out->shape()[1];
    size_t weight_cols = weight->shape()[1];
    
    size_t elem_size = utils::dsize(out->dtype());
    
    const int64_t* indices = reinterpret_cast<const int64_t*>(index->data());
    const void* weight_data = weight->data();
    void* out_data = out->data();
    
    for (size_t i = 0; i < out_rows; ++i) {
        int64_t row_idx = indices[i];
        const char* src_row = reinterpret_cast<const char*>(weight_data) + row_idx * weight_cols * elem_size;
        char* dst_row = reinterpret_cast<char*>(out_data) + i * out_cols * elem_size;
        std::memcpy(dst_row, src_row, weight_cols * elem_size);
    }
}
} // namespace llaisys::ops
