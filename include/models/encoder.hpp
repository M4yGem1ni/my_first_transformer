// include/core/models/encoder.hpp
#pragma once

#include "layers/attention.hpp"
#include "layers/feedforward.hpp"
#include "layers/layer_norm.hpp"
#include "layers/sublayer.hpp"
#include "utils/logger.hpp"
#include <xtensor/containers/xarray.hpp>
#include <memory>
#include <vector>

namespace transformer {
namespace models {

using namespace xt;
using namespace layers;

/**
 * @brief 编码器层
 */
template<typename T = float>
class EncoderLayer {
public:
    /**
     * @param d_model 模型维度
     * @param heads 注意力头数
     * @param d_ff 前馈网络隐藏层维度
     * @param dropout Dropout 概率
     */
    EncoderLayer(size_t d_model, size_t heads, size_t d_ff, T dropout = 0.1)
        : self_attn_(heads, d_model, dropout)
        , feed_forward_(d_model, d_ff, dropout)
        , sublayer1_(d_model, dropout)
        , sublayer2_(d_model, dropout)
    {
        LOG_TRACE("EncoderLayer created");
    }
    
    /**
     * @brief 前向传播
     * @param x 输入 [batch, seq_len, d_model]
     * @param mask 注意力 mask [batch, seq_len, seq_len]
     */
    xarray<T> forward(const xarray<T>& x, const xarray<T>& mask) {
        // 自注意力子层
        auto attn_fn = [this, &mask](const xarray<T>& x_in) {
            return self_attn_.forward(x_in, x_in, x_in, mask, false);
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

/**
 * @brief 编码器（多层编码器层堆叠）
 */
class Encoder {
public:
    /**
     * @param d_model 模型维度
     * @param heads 注意力头数
     * @param d_ff 前馈网络隐藏层维度
     * @param num_layers 编码器层数
     * @param dropout Dropout 概率
     */
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
    
    /**
     * @brief 前向传播
     * @param x 输入 [batch, seq_len, d_model]
     * @param mask 注意力 mask [batch, seq_len, seq_len]
     */
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

} // namespace models
} // namespace transformer