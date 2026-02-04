#include "llaisys/models/qwen2.h"
#include "models/qwen2.hpp"
#include <cstring>
#include <stdexcept>
#include <string> 

using namespace llaisys::models;

__C {
    // 模型创建
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta, 
        llaisysDeviceType_t device, 
        int *device_ids, 
        int ndevice
    ) {
        try {
            auto model = new Qwen2Model(meta, device, device_ids, ndevice);
            return reinterpret_cast<LlaisysQwen2Model*>(model);
        } catch (...) {
            return nullptr;
        }
    }

    // 模型销毁
    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (model) {
            auto cpp_model = reinterpret_cast<Qwen2Model*>(model);
            delete cpp_model;
        }
    }

    // 获取权重结构体
    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        if (!model) return nullptr;
        auto cpp_model = reinterpret_cast<Qwen2Model*>(model);
        return cpp_model->weights();
    }

    // 推理
    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken) {
        if (!model || !token_ids || ntoken == 0) return -1;
        auto cpp_model = reinterpret_cast<Qwen2Model*>(model);
        return cpp_model->infer(token_ids, ntoken);
    }

    // 修改推理函数，增加采样参数
    int64_t llaisysQwen2ModelInferWithSampling(
        struct LlaisysQwen2Model * model, 
        int64_t * token_ids, 
        size_t ntoken,
        int top_k,
        float top_p,
        float temperature
    ) {
        if (!model || !token_ids || ntoken == 0) return -1;
        auto cpp_model = reinterpret_cast<Qwen2Model*>(model);
        return cpp_model->infer_with_sampling(token_ids, ntoken, top_k, top_p, temperature);
    }

    

    // 权重加载（根据tensor_name匹配权重字段）
    void llaisysQwen2WeightsLoadTensor(
        struct LlaisysQwen2Weights *weights,
        const char *tensor_name,
        llaisysTensor_t tensor
    ) {
        if (!weights || !tensor_name || !tensor) return;
        
        // 解析tensor_name并匹配到对应的权重字段
        std::string name(tensor_name);
        
        if (name == "model.embed_tokens.weight") {
            weights->in_embed = tensor;
        } else if (name == "lm_head.weight") {
            weights->out_embed = tensor;
        } else if (name == "model.norm.weight") {
            weights->out_norm_w = tensor;
        } else {
            // 处理层特定权重，例如：model.layers.0.self_attn.q_proj.weight
            if (name.find("model.layers.") != std::string::npos) {
                // 解析层索引
                size_t layer_pos = name.find("layers.");
                size_t layer_start = layer_pos + 7; // "layers." length
                size_t layer_end = name.find(".", layer_start);
                if (layer_end != std::string::npos) {
                    int layer_idx = std::stoi(name.substr(layer_start, layer_end - layer_start));
                    
                    // 根据权重名称匹配到对应字段
                    if (name.find(".input_layernorm.weight") != std::string::npos) {
                        weights->attn_norm_w[layer_idx] = tensor;
                    } else if (name.find(".self_attn.q_proj.weight") != std::string::npos) {
                        weights->attn_q_w[layer_idx] = tensor;
                    } else if (name.find(".self_attn.q_proj.bias") != std::string::npos) {
                        weights->attn_q_b[layer_idx] = tensor;
                    } else if (name.find(".self_attn.k_proj.weight") != std::string::npos) {
                        weights->attn_k_w[layer_idx] = tensor;
                    } else if (name.find(".self_attn.k_proj.bias") != std::string::npos) {
                        weights->attn_k_b[layer_idx] = tensor;
                    } else if (name.find(".self_attn.v_proj.weight") != std::string::npos) {
                        weights->attn_v_w[layer_idx] = tensor;
                    } else if (name.find(".self_attn.v_proj.bias") != std::string::npos) {
                        weights->attn_v_b[layer_idx] = tensor;
                    } else if (name.find(".self_attn.o_proj.weight") != std::string::npos) {
                        weights->attn_o_w[layer_idx] = tensor;
                    } else if (name.find(".post_attention_layernorm.weight") != std::string::npos) {
                        weights->mlp_norm_w[layer_idx] = tensor;
                    } else if (name.find(".mlp.gate_proj.weight") != std::string::npos) {
                        weights->mlp_gate_w[layer_idx] = tensor;
                    } else if (name.find(".mlp.up_proj.weight") != std::string::npos) {
                        weights->mlp_up_w[layer_idx] = tensor;
                    } else if (name.find(".mlp.down_proj.weight") != std::string::npos) {
                        weights->mlp_down_w[layer_idx] = tensor;
                    }
                }
            }
        }
    }

    // 初始化KV-Cache
    void llaisysQwen2ModelInitKVCache(struct LlaisysQwen2Model *model) {
        if (!model) return;
        auto cpp_model = reinterpret_cast<Qwen2Model*>(model);
        cpp_model->init_kv_cache();
    }

    // 释放KV-Cache
    void llaisysQwen2ModelFreeKVCache(struct LlaisysQwen2Model *model) {
        if (!model) return;
        auto cpp_model = reinterpret_cast<Qwen2Model*>(model);
        cpp_model->free_kv_cache();
    }
}