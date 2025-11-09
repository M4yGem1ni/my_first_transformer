// include/core/encoder_decoder.hpp
#pragma once

#include "xtensor_accelerate.hpp"
#include "utils/logger.hpp" 
#include <xtensor/containers/xarray.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xindex_view.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <memory>
#include <cmath>
#include <random>

namespace transformer {

using namespace xt;

// ============================================
// 前向声明
// ============================================
class Encoder;
class Decoder;
class Embeddings;
class Generator;

// ============================================
// 权重初始化函数
// ============================================

template<typename T = float>
xarray<T> xavier_uniform(const std::vector<size_t>& shape, uint32_t seed = 42) {
    size_t fan_in = shape[0];
    size_t fan_out = shape[1];
    T limit = std::sqrt(6.0 / (fan_in + fan_out));
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(-limit, limit);
    
    xarray<T> result = zeros<T>(shape);
    for (auto& val : result) {
        val = dist(gen);
    }
    
    LOG_TRACE("xavier_uniform initialized: shape=[{}, {}], limit={:.4f}", 
              shape[0], shape[1], limit);
    
    return result;
}

template<typename T = float>
xarray<T> randn(const std::vector<size_t>& shape, T mean = 0.0, T stddev = 1.0, uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(mean, stddev);
    
    xarray<T> result = zeros<T>(shape);
    for (auto& val : result) {
        val = dist(gen);
    }
    
    LOG_TRACE("randn initialized: shape=[{}, {}], mean={:.4f}, stddev={:.4f}", 
              shape[0], shape[1], mean, stddev);
    
    return result;
}

template<typename T = float>
xarray<T> transformer_init(const std::vector<size_t>& shape, uint32_t seed = 42) {
    // 使用更小的标准差，避免数值爆炸
    T stddev = std::sqrt(T(2.0) / T(shape[0] + shape[1]));
    stddev = std::min(stddev, T(0.02));  // 限制最大值
    
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(T(0), stddev);
    
    xarray<T> result = zeros<T>(shape);
    for (auto& val : result) {
        val = dist(gen);
    }
    
    LOG_TRACE("transformer_init: shape=[{}, {}], stddev={:.4f}", 
              shape[0], shape[1], stddev);
    
    return result;
}

template<typename T = float>
xarray<T> glorot_uniform(const std::vector<size_t>& shape) {
    T limit = std::sqrt(6.0 / (shape[0] + shape[1]));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-limit, limit);
    
    xarray<T> result = zeros<T>(shape);
    for (auto& val : result) {
        val = dist(gen);
    }
    
    LOG_TRACE("glorot_uniform initialized: shape=[{}, {}], limit={:.4f}", 
              shape[0], shape[1], limit);
    
    return result;
}

// ============================================
// Softmax 实现
// ============================================

template<typename T>
xt::xarray<T> softmax(const xt::xarray<T>& x, int axis = -1) {
    using namespace xt;
    
    size_t ndim = x.dimension();
    size_t pos_axis = (axis < 0) ? ndim + axis : static_cast<size_t>(axis);
    
    // 构建形状字符串
    std::string shape_str = "[";
    for (size_t i = 0; i < ndim; ++i) {
        shape_str += std::to_string(x.shape()[i]);
        if (i < ndim - 1) shape_str += ", ";
    }
    shape_str += "]";
    
    LOG_TRACE("softmax - input shape: {}, axis={}", shape_str, pos_axis);
    
    try {
        auto x_max = amax(x, {pos_axis});
        LOG_TRACE("x_max computed");
        
        auto x_max_expanded = expand_dims(x_max, pos_axis);
        LOG_TRACE("x_max expanded");
        
        auto x_shifted = x - x_max_expanded;
        LOG_TRACE("x_shifted computed");
        
        auto exp_x = exp(x_shifted);
        LOG_TRACE("exp_x computed");
        
        auto sum_exp = sum(exp_x, {pos_axis});
        LOG_TRACE("sum_exp computed");
        
        auto sum_exp_expanded = expand_dims(sum_exp, pos_axis);
        LOG_TRACE("sum_exp expanded");
        
        auto result = exp_x / sum_exp_expanded;
        LOG_TRACE("division done");
        
        xarray<T> result_arr = result;
        
        LOG_TRACE("softmax completed successfully");
        
        return result_arr;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error in softmax: {}", e.what());
        throw;
    }
}

// ============================================
// EncoderDecoder 主类
// ============================================

template<typename T = float>
class EncoderDecoder {
public:
    EncoderDecoder(
        std::shared_ptr<Encoder> encoder,
        std::shared_ptr<Decoder> decoder,
        std::shared_ptr<Embeddings> src_embed,
        std::shared_ptr<Embeddings> tgt_embed,
        std::shared_ptr<Generator> generator)
        : encoder_(encoder)
        , decoder_(decoder)
        , src_embed_(src_embed)
        , tgt_embed_(tgt_embed)
        , generator_(generator)
    {
        LOG_DEBUG("EncoderDecoder model assembled");
    }
    
    xarray<T> forward(
        const xarray<T>& src,
        const xarray<T>& tgt,
        const xarray<T>& src_mask,
        const xarray<T>& tgt_mask);
    
    xarray<T> encode(const xarray<T>& src, const xarray<T>& src_mask);
    
    xarray<T> decode(
        const xarray<T>& memory,
        const xarray<T>& src_mask,
        const xarray<T>& tgt,
        const xarray<T>& tgt_mask);

private:
    std::shared_ptr<Encoder> encoder_;
    std::shared_ptr<Decoder> decoder_;
    std::shared_ptr<Embeddings> src_embed_;
    std::shared_ptr<Embeddings> tgt_embed_;
    std::shared_ptr<Generator> generator_;
};

// ============================================
// Layer Normalization
// ============================================

template<typename T = float>
class LayerNorm {
public:
    LayerNorm(size_t features, T eps = 1e-6)
        : features_(features)
        , eps_(eps)
        , gamma_(ones<T>({features}))
        , beta_(zeros<T>({features}))
    {
        LOG_TRACE("LayerNorm initialized: features={}, eps={}", features, eps);
    }
    
    xarray<T> forward(const xarray<T>& x) {
        auto mean = xt::mean(x, {x.dimension() - 1}, keep_dims);
        std::vector<std::size_t> axes = {x.dimension() - 1};
        auto var = xt::variance(x, axes, xt::keep_dims);
        
        auto x_norm = (x - mean) / xt::sqrt(var + eps_);
        
        return gamma_ * x_norm + beta_;
    }
    
private:
    size_t features_;
    T eps_;
    xarray<T> gamma_;
    xarray<T> beta_;
};

// ============================================
// 残差连接
// ============================================

template<typename T = float>
class SublayerConnection {
public:
    SublayerConnection(size_t size, T dropout = 0.1)
        : norm_(size)
        , dropout_(dropout)
    {
        LOG_TRACE("SublayerConnection initialized: size={}, dropout={}", size, dropout);
    }
    
    template<typename Sublayer>
    xarray<T> forward(const xarray<T>& x, Sublayer sublayer) {
        return x + apply_dropout(sublayer(norm_.forward(x)));
    }

private:
    LayerNorm<T> norm_;
    T dropout_;
    
    xarray<T> apply_dropout(const xarray<T>& x) {
        return x;  // 推理时不使用 dropout
    }
};

// ============================================
// 多头注意力机制
// ============================================

template<typename T = float>
class MultiHeadAttention {
public:
    MultiHeadAttention(size_t h, size_t d_model, T dropout = 0.1)
        : h_(h)
        , d_model_(d_model)
        , d_k_(d_model / h)
        , dropout_(dropout)
    {
        assert(d_model_ % h_ == 0);
        
        W_q_ = transformer_init<T>({d_model_, d_model_}, 1);
        W_k_ = transformer_init<T>({d_model_, d_model_}, 2);
        W_v_ = transformer_init<T>({d_model_, d_model_}, 3);
        W_o_ = transformer_init<T>({d_model_, d_model_}, 4);
        
        LOG_DEBUG("MultiHeadAttention initialized: heads={}, d_model={}, d_k={}", 
                 h, d_model, d_k_);
    }
    
    // 🔴 添加 suppress_empty_mask_warning 参数
    xarray<T> forward(const xarray<T>& query, 
                      const xarray<T>& key, 
                      const xarray<T>& value,
                      const xarray<T>& mask = xarray<T>(),
                      bool suppress_empty_mask_warning = false)  // 🔴 新参数
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
        
        // 2-3. Reshape 和 Transpose
        std::vector<size_t> multi_head_shape = {batch_size, seq_len, h_, d_k_};
        auto Q_reshaped = xt::reshape_view(Q, multi_head_shape);
        auto K_reshaped = xt::reshape_view(K, multi_head_shape);
        auto V_reshaped = xt::reshape_view(V, multi_head_shape);
        
        auto Q_heads = xt::transpose(Q_reshaped, {0, 2, 1, 3});
        auto K_heads = xt::transpose(K_reshaped, {0, 2, 1, 3});
        auto V_heads = xt::transpose(V_reshaped, {0, 2, 1, 3});
        
        xarray<T> Q_multi = Q_heads;
        xarray<T> K_multi = K_heads;
        xarray<T> V_multi = V_heads;
        
        LOG_TRACE("After reshape/transpose - Q_multi.shape = [{}, {}, {}, {}]",
                 Q_multi.shape()[0], Q_multi.shape()[1], 
                 Q_multi.shape()[2], Q_multi.shape()[3]);
        
        // 4. 执行缩放点积注意力 - 🔴 传递标志
        auto attn_output = scaled_dot_product_attention(
            Q_multi, K_multi, V_multi, mask, suppress_empty_mask_warning);
        
        LOG_TRACE("After attention - attn_output.shape = [{}, {}, {}, {}]",
                 attn_output.shape()[0], attn_output.shape()[1],
                 attn_output.shape()[2], attn_output.shape()[3]);
        
        // 5-6. 转置回来并合并多头
        xarray<T> attn_output_arr = attn_output;
        auto attn_transposed = xt::transpose(attn_output_arr, {0, 2, 1, 3});
        
        LOG_TRACE("After transpose - attn_transposed.shape = [{}, {}, {}, {}]",
                 attn_transposed.shape()[0], attn_transposed.shape()[1],
                 attn_transposed.shape()[2], attn_transposed.shape()[3]);
        
        std::vector<size_t> concat_shape = {batch_size, seq_len, d_model_};
        auto attn_concat = xt::reshape_view(attn_transposed, concat_shape);
        xarray<T> attn_concat_arr = attn_concat;
        
        LOG_TRACE("After concat - attn_concat_arr.shape = [{}, {}, {}]",
                 attn_concat_arr.shape()[0], attn_concat_arr.shape()[1],
                 attn_concat_arr.shape()[2]);
        
        // 7. 最后的线性投影
        return linear_projection(attn_concat_arr, W_o_);
    }

private:
    size_t h_;
    size_t d_model_;
    size_t d_k_;
    T dropout_;
    
    xarray<T> W_q_, W_k_, W_v_, W_o_;
    
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

    // 🔴 修改 apply_attention_mask，添加标志参数
    xarray<T> apply_attention_mask(const xarray<T>& scores, 
                                    const xarray<T>& mask,
                                    bool suppress_empty_mask_warning = false) {
        // 🔴 检查空 mask
        if (mask.size() == 0 || mask.dimension() == 0) {
            if (!suppress_empty_mask_warning) {
                LOG_TRACE("Empty mask detected, no masking applied");
            }
            return scores;
        }
        
        auto scores_shape = scores.shape();
        auto mask_shape = mask.shape();
        
        // 构建形状字符串
        std::string scores_shape_str = "[";
        for (size_t i = 0; i < scores.dimension(); ++i) {
            scores_shape_str += std::to_string(scores_shape[i]);
            if (i < scores.dimension() - 1) scores_shape_str += ", ";
        }
        scores_shape_str += "]";
        
        std::string mask_shape_str = "[";
        for (size_t i = 0; i < mask.dimension(); ++i) {
            mask_shape_str += std::to_string(mask_shape[i]);
            if (i < mask.dimension() - 1) mask_shape_str += ", ";
        }
        mask_shape_str += "]";
        
        LOG_TRACE("apply_attention_mask: scores={}, mask={}", 
                 scores_shape_str, mask_shape_str);
        
        // 检查 mask 的有效性
        if (mask.dimension() > 4) {
            LOG_WARN("Invalid mask dimension: {}. Skipping mask.", mask.dimension());
            return scores;
        }
        
        // 检查 mask 形状是否合理
        bool mask_shape_valid = true;
        for (size_t i = 0; i < mask.dimension(); ++i) {
            if (mask_shape[i] == 0 || mask_shape[i] > 100000) {
                LOG_WARN("Invalid mask shape detected: {} at dimension {}. Skipping mask.",
                        mask_shape[i], i);
                mask_shape_valid = false;
                break;
            }
        }
        
        if (!mask_shape_valid) {
            return scores;
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
            LOG_TRACE("Applying mask with matching dimensions");
            return xt::where(xt::equal(mask_arr, static_cast<T>(0)), 
                            std::numeric_limits<T>::lowest(), 
                            scores);
        } else if (mask.dimension() == 2 && scores.dimension() == 4) {
            if (mask_shape[0] != scores_seq_q || mask_shape[1] != scores_seq_k) {
                LOG_WARN("2D mask dimensions don't match scores. Skipping mask.");
                return scores;
            }
            
            std::vector<size_t> new_shape = {1, 1, mask_shape[0], mask_shape[1]};
            auto mask_reshaped = xt::reshape_view(mask_arr, new_shape);
            xarray<T> mask_expanded = mask_reshaped;
            
            LOG_TRACE("Broadcasting 2D mask to 4D: [1, 1, {}, {}]", 
                     mask_shape[0], mask_shape[1]);
            
            return xt::where(xt::equal(mask_expanded, static_cast<T>(0)), 
                            std::numeric_limits<T>::lowest(), 
                            scores);
        } else if (mask.dimension() == 3 && scores.dimension() == 4) {
            if (mask_shape[1] != scores_seq_q || mask_shape[2] != scores_seq_k) {
                LOG_WARN("3D mask dimensions don't match scores. Skipping mask.");
                return scores;
            }
            
            std::vector<size_t> new_shape = {mask_shape[0], 1, mask_shape[1], mask_shape[2]};
            auto mask_reshaped = xt::reshape_view(mask_arr, new_shape);
            xarray<T> mask_expanded = mask_reshaped;
            
            LOG_TRACE("Broadcasting 3D mask to 4D: [{}, 1, {}, {}]",
                     mask_shape[0], mask_shape[1], mask_shape[2]);
            
            return xt::where(xt::equal(mask_expanded, static_cast<T>(0)), 
                            std::numeric_limits<T>::lowest(), 
                            scores);
        } else {
            LOG_WARN("Unsupported mask dimensions: mask={}, scores={}. Skipping mask.",
                    mask.dimension(), scores.dimension());
            return scores;
        }
    }
    
    // 🔴 修改 scaled_dot_product_attention，添加标志参数
    xarray<T> scaled_dot_product_attention(
        const xarray<T>& Q,
        const xarray<T>& K,
        const xarray<T>& V,
        const xarray<T>& mask,
        bool suppress_empty_mask_warning = false)
    {
        LOG_TRACE("=== scaled_dot_product_attention ===");
        LOG_TRACE("Q.shape = [{}, {}, {}, {}]", 
                 Q.shape()[0], Q.shape()[1], Q.shape()[2], Q.shape()[3]);
        LOG_TRACE("K.shape = [{}, {}, {}, {}]",
                 K.shape()[0], K.shape()[1], K.shape()[2], K.shape()[3]);
        
        // K^T
        auto K_T = xt::transpose(K, {0, 1, 3, 2});
        LOG_TRACE("K_T.shape = [{}, {}, {}, {}]",
                 K_T.shape()[0], K_T.shape()[1], K_T.shape()[2], K_T.shape()[3]);
        
        xarray<T> Q_arr = Q;
        xarray<T> K_T_arr = K_T;
        
        // Q @ K^T
        auto scores = accelerate::Backend::matmul_auto(Q_arr, K_T_arr);
        
        // 裁剪到合理范围
        scores = xt::clip(scores, static_cast<T>(-50.0), static_cast<T>(50.0));
        
        auto scores_min = xt::amin(scores)();
        auto scores_max = xt::amax(scores)();
        LOG_TRACE("scores range (after clipping): [{:.3f}, {:.3f}]", scores_min, scores_max);
        
        if (std::isnan(scores_min) || std::isnan(scores_max)) {
            LOG_ERROR("NaN detected in attention scores!");
            throw std::runtime_error("NaN in attention scores");
        }
        
        // 缩放
        scores = scores / std::sqrt(static_cast<T>(d_k_));
        
        // 应用 mask - 🔴 传递标志
        if (mask.size() > 0) {
            scores = apply_attention_mask(scores, mask, suppress_empty_mask_warning);
        }
        
        // Softmax
        auto attn_weights = softmax(scores, -1);
        
        LOG_TRACE("attn_weights.shape = [{}, {}, {}, {}]",
                 attn_weights.shape()[0], attn_weights.shape()[1],
                 attn_weights.shape()[2], attn_weights.shape()[3]);
        
        xarray<T> attn_arr = attn_weights;
        xarray<T> V_arr = V;
        
        // attn @ V
        auto output = accelerate::Backend::matmul_auto(attn_arr, V_arr);
        
        LOG_TRACE("output.shape = [{}, {}, {}, {}]",
                 output.shape()[0], output.shape()[1],
                 output.shape()[2], output.shape()[3]);
        
        return output;
    }
};

// ============================================
// 位置前馈网络
// ============================================

template<typename T = float>
class PositionwiseFeedForward {
public:
    PositionwiseFeedForward(size_t d_model, size_t d_ff, T dropout = 0.1)
        : d_model_(d_model)
        , d_ff_(d_ff)
        , dropout_(dropout)
    {
        W1_ = transformer_init<T>({d_model_, d_ff_}, 5);
        b1_ = zeros<T>({d_ff_});
        W2_ = transformer_init<T>({d_ff_, d_model_}, 6);
        b2_ = zeros<T>({d_model_});
        
        LOG_DEBUG("FFN initialized: d_model={}, d_ff={}", d_model, d_ff);
    }
    
    xarray<T> forward(const xarray<T>& x) {
        size_t batch = x.shape()[0];
        size_t seq_len = x.shape()[1];
        
        auto x_2d = reshape_view(x, {batch * seq_len, d_model_});
        xarray<T> x_2d_arr = x_2d;
        auto hidden = accelerate::Backend::matmul_auto(x_2d_arr, W1_);
        hidden = hidden + b1_;
        hidden = xt::maximum(hidden, T(0));  // ReLU
        
        auto output = accelerate::Backend::matmul_auto(hidden, W2_);
        output = output + b2_;
        
        return reshape_view(output, {batch, seq_len, d_model_});
    }

private:
    size_t d_model_;
    size_t d_ff_;
    T dropout_;
    xarray<T> W1_, b1_, W2_, b2_;
};

// ============================================
// 编码器层
// ============================================

template<typename T = float>
class EncoderLayer {
public:
    EncoderLayer(size_t d_model, size_t heads, size_t d_ff, T dropout = 0.1)
        : self_attn_(heads, d_model, dropout)
        , feed_forward_(d_model, d_ff, dropout)
        , sublayer1_(d_model, dropout)
        , sublayer2_(d_model, dropout)
    {
        LOG_TRACE("EncoderLayer created");
    }
    
    xarray<T> forward(const xarray<T>& x, const xarray<T>& mask) {
        // 自注意力子层
        auto attn_fn = [this, &mask](const xarray<T>& x_in) {
            return self_attn_.forward(x_in, x_in, x_in, mask, false);  // 🔴 不抑制警告
        };
        auto x1 = sublayer1_.forward(x, attn_fn);
        
        // 前馈子层
        auto ff_fn = [this](const xarray<T>& x_in) {
            return feed_forward_.forward(x_in);
        };
        return sublayer2_.forward(x1, ff_fn);
    }

private:
    MultiHeadAttention<T> self_attn_;
    PositionwiseFeedForward<T> feed_forward_;
    SublayerConnection<T> sublayer1_;
    SublayerConnection<T> sublayer2_;
};

// ============================================
// 编码器
// ============================================

class Encoder {
public:
    template<typename T = float>
    Encoder(size_t d_model, size_t heads, size_t d_ff, size_t num_layers, T dropout = 0.1)
        : d_model_(d_model)
        , norm_(d_model)
    {
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.push_back(
                std::make_shared<EncoderLayer<T>>(d_model, heads, d_ff, dropout)
            );
        }
        LOG_DEBUG("Encoder created: {} layers", num_layers);
    }
    
    template<typename T = float>
    xarray<T> forward(const xarray<T>& x, const xarray<T>& mask) {
        auto output = x;
        for (auto& layer : layers_) {
            auto enc_layer = std::static_pointer_cast<EncoderLayer<T>>(layer);
            output = enc_layer->forward(output, mask);
        }
        return norm_.forward(output);
    }

private:
    size_t d_model_;
    std::vector<std::shared_ptr<void>> layers_;
    LayerNorm<float> norm_;
};

// ============================================
// 解码器层
// ============================================

template<typename T = float>
class DecoderLayer {
public:
    DecoderLayer(size_t d_model, size_t heads, size_t d_ff, T dropout = 0.1)
        : self_attn_(heads, d_model, dropout)
        , src_attn_(heads, d_model, dropout)
        , feed_forward_(d_model, d_ff, dropout)
        , sublayer1_(d_model, dropout)
        , sublayer2_(d_model, dropout)
        , sublayer3_(d_model, dropout)
    {
        LOG_TRACE("DecoderLayer created");
    }

    xarray<T> forward(
        const xarray<T>& x,
        const xarray<T>& memory,
        const xarray<T>& src_mask,
        const xarray<T>& tgt_mask)
    {
        LOG_TRACE("DecoderLayer::forward - x.shape=[{}, {}, {}]", 
                 x.shape()[0], x.shape()[1], x.shape()[2]);
        LOG_TRACE("DecoderLayer::forward - memory.shape=[{}, {}, {}]",
                 memory.shape()[0], memory.shape()[1], memory.shape()[2]);
        
        // 1. Self-attention (使用 tgt_mask)
        auto self_attn_fn = [this, &tgt_mask](const xarray<T>& x_in) {
            return self_attn_.forward(x_in, x_in, x_in, tgt_mask, false);  // 不抑制警告
        };
        auto x1 = sublayer1_.forward(x, self_attn_fn);
        
        // 2. Cross-attention (使用空 mask)
        // 🔴 传递 true 来抑制空 mask 警告
        auto src_attn_fn = [this, &memory](const xarray<T>& x_in) {
            xarray<T> empty_mask;
            return src_attn_.forward(x_in, memory, memory, empty_mask, true);  // 🔴 抑制警告
        };
        auto x2 = sublayer2_.forward(x1, src_attn_fn);
        
        // 3. Feed-forward
        auto ff_fn = [this](const xarray<T>& x_in) {
            return feed_forward_.forward(x_in);
        };
        return sublayer3_.forward(x2, ff_fn);
    }

private:
    MultiHeadAttention<T> self_attn_;
    MultiHeadAttention<T> src_attn_;
    PositionwiseFeedForward<T> feed_forward_;
    SublayerConnection<T> sublayer1_;
    SublayerConnection<T> sublayer2_;
    SublayerConnection<T> sublayer3_;
};

// ============================================
// 解码器
// ============================================

class Decoder {
public:
    template<typename T = float>
    Decoder(size_t d_model, size_t heads, size_t d_ff, size_t num_layers, T dropout = 0.1)
        : d_model_(d_model)
        , norm_(d_model)
    {
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.push_back(
                std::make_shared<DecoderLayer<T>>(d_model, heads, d_ff, dropout)
            );
        }
        LOG_DEBUG("Decoder created: {} layers", num_layers);
    }
    
    template<typename T = float>
    xarray<T> forward(
        const xarray<T>& x,
        const xarray<T>& memory,
        const xarray<T>& src_mask,
        const xarray<T>& tgt_mask)
    {
        LOG_TRACE("Decoder::forward - x.shape=[{}, {}, {}]",
                 x.shape()[0], x.shape()[1], x.shape()[2]);
        LOG_TRACE("Decoder::forward - memory.shape=[{}, {}, {}]",
                 memory.shape()[0], memory.shape()[1], memory.shape()[2]);
        
        // 🔴 添加 mask 验证
        if (tgt_mask.size() > 0) {
            std::string tgt_mask_shape = "[";
            for (size_t i = 0; i < tgt_mask.dimension(); ++i) {
                tgt_mask_shape += std::to_string(tgt_mask.shape()[i]);
                if (i < tgt_mask.dimension() - 1) tgt_mask_shape += ", ";
            }
            tgt_mask_shape += "]";
            LOG_TRACE("Decoder::forward - tgt_mask.shape={}", tgt_mask_shape);
        }
        
        if (src_mask.size() > 0) {
            std::string src_mask_shape = "[";
            for (size_t i = 0; i < src_mask.dimension(); ++i) {
                src_mask_shape += std::to_string(src_mask.shape()[i]);
                if (i < src_mask.dimension() - 1) src_mask_shape += ", ";
            }
            src_mask_shape += "]";
            LOG_TRACE("Decoder::forward - src_mask.shape={}", src_mask_shape);
        }
        
        auto output = x;
        for (auto& layer : layers_) {
            auto dec_layer = std::static_pointer_cast<DecoderLayer<T>>(layer);
            output = dec_layer->forward(output, memory, src_mask, tgt_mask);
        }
        return norm_.forward(output);
    }

private:
    size_t d_model_;
    std::vector<std::shared_ptr<void>> layers_;
    LayerNorm<float> norm_;
};

// ============================================
// 词嵌入
// ============================================

class Embeddings {
public:
    template<typename T = float>
    Embeddings(size_t vocab_size, size_t d_model)
        : vocab_size_(vocab_size)
        , d_model_(d_model)
        , embedding_table_(transformer_init<T>({vocab_size, d_model}, 7))
    {
        LOG_DEBUG("Embeddings initialized: vocab_size={}, d_model={}", vocab_size, d_model);
    }
    
    template<typename T = float>
    xarray<T> forward(const xarray<int>& x) {
        size_t batch = x.shape()[0];
        size_t seq_len = x.shape()[1];
        
        xarray<T> output = zeros<T>({batch, seq_len, d_model_});
        
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                int idx = x(b, s);
                if (idx >= 0 && idx < static_cast<int>(vocab_size_)) {
                    view(output, b, s, all()) = view(embedding_table_, idx, all());
                }
            }
        }
        
        return output;
    }

private:
    size_t vocab_size_;
    size_t d_model_;
    xarray<float> embedding_table_;
};

// ============================================
// 生成器 (输出层)
// ============================================

class Generator {
public:
    template<typename T = float>
    Generator(size_t d_model, size_t vocab_size)
        : d_model_(d_model)
        , vocab_size_(vocab_size)
        , proj_(transformer_init<T>({d_model, vocab_size}, 8))
    {
        LOG_DEBUG("Generator initialized: d_model={}, vocab_size={}", d_model, vocab_size);
    }
    
    template<typename T = float>
    xarray<T> forward(const xarray<T>& x) {
        size_t batch = x.shape()[0];
        size_t seq_len = x.shape()[1];
        
        auto x_2d = reshape_view(x, {batch * seq_len, d_model_});
        auto logits = accelerate::Backend::matmul_auto(x_2d, proj_);
        
        // Log softmax
        auto max_logits = xt::amax(logits, {-1}, keep_dims);
        auto exp_logits = xt::exp(logits - max_logits);
        auto log_sum_exp = xt::log(xt::sum(exp_logits, {-1}, keep_dims));
        auto log_probs = logits - max_logits - log_sum_exp;
        
        return reshape_view(log_probs, {batch, seq_len, vocab_size_});
    }

private:
    size_t d_model_;
    size_t vocab_size_;
    xarray<float> proj_;
};

// ============================================
// EncoderDecoder 实现
// ============================================

template<typename T>
xarray<T> EncoderDecoder<T>::forward(
    const xarray<T>& src,
    const xarray<T>& tgt,
    const xarray<T>& src_mask,
    const xarray<T>& tgt_mask)
{
    LOG_TRACE("EncoderDecoder::forward - encoding source");
    auto memory = encode(src, src_mask);
    
    LOG_TRACE("EncoderDecoder::forward - decoding target");
    return decode(memory, src_mask, tgt, tgt_mask);
}

template<typename T>
xarray<T> EncoderDecoder<T>::encode(
    const xarray<T>& src,
    const xarray<T>& src_mask)
{
    auto embedded = src_embed_->template forward<T>(src);
    return encoder_->template forward<T>(embedded, src_mask);
}

template<typename T>
xarray<T> EncoderDecoder<T>::decode(
    const xarray<T>& memory,
    const xarray<T>& src_mask,
    const xarray<T>& tgt,
    const xarray<T>& tgt_mask)
{
    auto embedded = tgt_embed_->template forward<T>(tgt);
    return decoder_->template forward<T>(embedded, memory, src_mask, tgt_mask);
}

} // namespace transformer