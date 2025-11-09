// include/layers/attention.hpp
#pragma once

#include "solvers/accelerate.hpp"
#include "utils/logger.hpp"
#include "utils/weight_init.hpp"
#include "utils/activations.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/core/xmath.hpp>
#include <memory>
#include <limits>

namespace transformer {
namespace layers {

using namespace xt;

/**
 * @brief 多头注意力机制
 */
template<typename T = float>
class MultiHeadAttention {
public:
    /**
     * @param h 注意力头数
     * @param d_model 模型维度
     * @param dropout Dropout 概率
     */
    MultiHeadAttention(size_t h, size_t d_model, T dropout = 0.1)
        : h_(h)
        , d_model_(d_model)
        , d_k_(d_model / h)
        , dropout_(dropout)
    {
        assert(d_model_ % h_ == 0);
        
        W_q_ = utils::transformer_init<T>({d_model_, d_model_}, 1);
        W_k_ = utils::transformer_init<T>({d_model_, d_model_}, 2);
        W_v_ = utils::transformer_init<T>({d_model_, d_model_}, 3);
        W_o_ = utils::transformer_init<T>({d_model_, d_model_}, 4);
        
        LOG_DEBUG("MultiHeadAttention initialized: heads={}, d_model={}, d_k={}", 
                 h, d_model, d_k_);
    }
    
    /**
     * @brief 前向传播
     * @param query 查询张量
     * @param key 键张量
     * @param value 值张量
     * @param mask 注意力 mask
     * @param suppress_empty_mask_warning 是否抑制空 mask 警告
     */
    xarray<T> forward(const xarray<T>& query, 
                      const xarray<T>& key, 
                      const xarray<T>& value,
                      const xarray<T>& mask = xarray<T>(),
                      bool suppress_empty_mask_warning = false)
    {
        LOG_TRACE("=== MultiHeadAttention::forward ===");
        
        // 1. 线性投影
        auto Q = linear_projection(query, W_q_);
        auto K = linear_projection(key, W_k_);
        auto V = linear_projection(value, W_v_);
        
        auto Q_shape = Q.shape();
        size_t batch_size = Q_shape[0];
        size_t seq_len = Q_shape[1];
        
        LOG_TRACE("After linear projection - Q.shape = [{}, {}, {}]", 
                 batch_size, seq_len, Q_shape[2]);
        
        // 2. Reshape 到多头格式
        std::vector<size_t> multi_head_shape = {batch_size, seq_len, h_, d_k_};
        auto Q_reshaped = xt::reshape_view(Q, multi_head_shape);
        auto K_reshaped = xt::reshape_view(K, multi_head_shape);
        auto V_reshaped = xt::reshape_view(V, multi_head_shape);
        
        // 3. Transpose 到 [batch, heads, seq, d_k]
        auto Q_heads = xt::transpose(Q_reshaped, {0, 2, 1, 3});
        auto K_heads = xt::transpose(K_reshaped, {0, 2, 1, 3});
        auto V_heads = xt::transpose(V_reshaped, {0, 2, 1, 3});
        
        xarray<T> Q_multi = Q_heads;
        xarray<T> K_multi = K_heads;
        xarray<T> V_multi = V_heads;
        
        // 4. 执行缩放点积注意力
        auto attn_output = scaled_dot_product_attention(
            Q_multi, K_multi, V_multi, mask, suppress_empty_mask_warning);
        
        // 5. 转置并合并多头
        xarray<T> attn_output_arr = attn_output;
        auto attn_transposed = xt::transpose(attn_output_arr, {0, 2, 1, 3});
        
        std::vector<size_t> concat_shape = {batch_size, seq_len, d_model_};
        auto attn_concat = xt::reshape_view(attn_transposed, concat_shape);
        xarray<T> attn_concat_arr = attn_concat;
        
        // 6. 最后的线性投影
        return linear_projection(attn_concat_arr, W_o_);
    }

private:
    size_t h_;        // 头数
    size_t d_model_;  // 模型维度
    size_t d_k_;      // 每个头的维度
    T dropout_;
    
    xarray<T> W_q_, W_k_, W_v_, W_o_;
    
    /**
     * @brief 线性投影辅助函数
     */
    auto linear_projection(const xarray<T>& x, const xarray<T>& weight) {
        auto orig_shape = x.shape();
        std::vector<std::ptrdiff_t> new_shape = {-1, static_cast<std::ptrdiff_t>(orig_shape.back())};
        auto x_2d = xt::reshape_view(x, new_shape);
        
        xarray<T> x_matrix = x_2d;
        auto result = accelerate::Backend::matmul_auto(x_matrix, weight);
        
        std::vector<size_t> target_shape(orig_shape.begin(), orig_shape.end() - 1);
        target_shape.push_back(weight.shape()[1]);
        return xt::reshape_view(result, target_shape);
    }

    /**
     * @brief 应用注意力 mask
     */
    xarray<T> apply_attention_mask(const xarray<T>& scores, 
                                    const xarray<T>& mask,
                                    bool suppress_empty_mask_warning = false) {
        if (mask.size() == 0 || mask.dimension() == 0) {
            if (!suppress_empty_mask_warning) {
                LOG_TRACE("Empty mask detected, no masking applied");
            }
            return scores;
        }
        
        auto scores_shape = scores.shape();
        auto mask_shape = mask.shape();
        
        // 检查 mask 有效性
        if (mask.dimension() > 4) {
            LOG_WARN("Invalid mask dimension: {}. Skipping mask.", mask.dimension());
            return scores;
        }
        
        for (size_t i = 0; i < mask.dimension(); ++i) {
            if (mask_shape[i] == 0 || mask_shape[i] > 100000) {
                LOG_WARN("Invalid mask shape detected: {} at dimension {}. Skipping mask.",
                        mask_shape[i], i);
                return scores;
            }
        }
        
        size_t scores_seq_q = scores_shape[scores.dimension() - 2];
        size_t scores_seq_k = scores_shape[scores.dimension() - 1];
        size_t mask_last_dim = mask_shape[mask.dimension() - 1];
        
        if (mask_last_dim != scores_seq_k) {
            LOG_WARN("Mask shape incompatible with scores. Mask last dim={}, but scores needs {}. Skipping mask.",
                    mask_last_dim, scores_seq_k);
            return scores;
        }
        
        xarray<T> mask_arr = mask;
        
        if (mask.dimension() == scores.dimension()) {
            return xt::where(xt::equal(mask_arr, static_cast<T>(0)), 
                            std::numeric_limits<T>::lowest(), 
                            scores);
        } else if (mask.dimension() == 3 && scores.dimension() == 4) {
            std::vector<size_t> new_shape = {mask_shape[0], 1, mask_shape[1], mask_shape[2]};
            auto mask_reshaped = xt::reshape_view(mask_arr, new_shape);
            xarray<T> mask_expanded = mask_reshaped;
            
            return xt::where(xt::equal(mask_expanded, static_cast<T>(0)), 
                            std::numeric_limits<T>::lowest(), 
                            scores);
        }
        
        LOG_WARN("Unsupported mask dimensions. Skipping mask.");
        return scores;
    }
    
    /**
     * @brief 缩放点积注意力
     */
    xarray<T> scaled_dot_product_attention(
        const xarray<T>& Q,
        const xarray<T>& K,
        const xarray<T>& V,
        const xarray<T>& mask,
        bool suppress_empty_mask_warning = false)
    {
        LOG_TRACE("=== scaled_dot_product_attention ===");
        
        // K^T
        auto K_T = xt::transpose(K, {0, 1, 3, 2});
        
        xarray<T> Q_arr = Q;
        xarray<T> K_T_arr = K_T;
        
        // Q @ K^T
        auto scores = accelerate::Backend::matmul_auto(Q_arr, K_T_arr);
        
        // 裁剪到合理范围
        scores = xt::clip(scores, static_cast<T>(-50.0), static_cast<T>(50.0));
        
        auto scores_min = xt::amin(scores)();
        auto scores_max = xt::amax(scores)();
        LOG_TRACE("scores range: [{:.3f}, {:.3f}]", scores_min, scores_max);
        
        if (std::isnan(scores_min) || std::isnan(scores_max)) {
            LOG_ERROR("NaN detected in attention scores!");
            throw std::runtime_error("NaN in attention scores");
        }
        
        // 缩放
        scores = scores / std::sqrt(static_cast<T>(d_k_));
        
        // 应用 mask
        if (mask.size() > 0) {
            scores = apply_attention_mask(scores, mask, suppress_empty_mask_warning);
        }
        
        // Softmax
        auto attn_weights = utils::softmax(scores, -1);
        
        xarray<T> attn_arr = attn_weights;
        xarray<T> V_arr = V;
        
        // attn @ V
        auto output = accelerate::Backend::matmul_auto(attn_arr, V_arr);
        
        return output;
    }
};

} // namespace layers
} // namespace transformer