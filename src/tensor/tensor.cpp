#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    // 空张量默认连续
    if (_meta.shape.empty()) return true;
    
    // 从最后一维开始校验步长：连续张量的步长满足「当前维步长 = 下一维步长 * 下一维大小」
    ptrdiff_t expected_stride = 1;
    for (size_t i = _meta.shape.size() - 1; i >= 0; --i) {
        if (_meta.strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= _meta.shape[i];
    }    
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    // 验证输入的维度顺序是否有效
    if (order.size() != _meta.shape.size()) {
        throw std::invalid_argument("Permutation order size must match tensor dimension");
    }

    // 检查 order 是否包含所有维度索引，且无重复
    std::vector<bool> seen(_meta.shape.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] >= _meta.shape.size()) {
            throw std::invalid_argument("Invalid dimension index in permutation order");
        }
        if (seen[order[i]]) {
            throw std::invalid_argument("Duplicate dimension index in permutation order");
        }
        seen[order[i]] = true;
    }

    // 构建新形状和新步长
    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());

    for (size_t i = 0; i < order.size(); ++i) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }

    // 创建新的元数据
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};

    // 创建并返回新的张量，使用相同的存储空间和偏移量
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 计算原始张量和新形状的元素总数
    size_t original_numel = this->numel();
    size_t new_numel = 1;
    for (size_t dim : shape) {
        new_numel *= dim;
    }

    // 检查元素总数是否匹配
    if (original_numel != new_numel) {
        throw std::runtime_error("New shape must have the same number of elements as the original tensor");
    }

    // 检查张量是否连续，只有连续的张量才能安全地改变视图
    if (!this->isContiguous()) {
        throw std::runtime_error("Cannot view non-contiguous tensor, call contiguous() first");
    }

    // 计算新的步长
    std::vector<ptrdiff_t> new_strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = shape.size() - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= shape[i];
    }

    // 创建新的元数据
    TensorMeta new_meta{_meta.dtype, shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 验证维度索引是否有效
    if (dim >= _meta.shape.size()) {
        throw std::invalid_argument("Dimension index out of range");
    }

    // 验证切片范围是否有效
    if (start >= end || end > _meta.shape[dim]) {
        throw std::invalid_argument("Invalid slice range");
    }

    // 计算新形状：将指定维度的大小更新为切片后的大小
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;

    // 计算新的偏移量：原偏移量加上切片起始位置的偏移
    size_t new_offset = _offset + start * _meta.strides[dim] * utils::dsize(_meta.dtype);

    // 步长保持不变，因为切片不改变维度间的步长关系
    std::vector<ptrdiff_t> new_strides = _meta.strides;

    // 创建新的元数据
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};

    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    printf("Tensor::load()\n");
    size_t total_bytes = this->numel() * this->elementSize();
    
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // 如果目标是CPU设备，直接内存拷贝
        std::memcpy(this->data(), src_, total_bytes);
    } else {
        // 如果目标是GPU或其他设备，需要从主机内存复制到设备内存
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            this->data(),           // 目标地址（设备内存）
            src_,                   // 源地址（主机内存）
            total_bytes,            // 要复制的字节数
            LLAISYS_MEMCPY_H2D      // 从主机到设备的内存复制
        );
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
