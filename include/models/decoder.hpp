// include/core/models/decoder.hpp
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
 * @brief 解码器层
 */
template<typename T = float>
class DecoderLayer {
public:
    /**
     * @param d_model 模型维度
     * @param heads 注意力头数
     * @param d_ff 前馈网络隐藏层维度
     * @param dropout Dropout 概率
     */
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

    /**
     * @brief 前向传播
     * @param x 输入 [batch, tgt_len, d_model]
     * @param memory 编码器输出 [batch, src_len, d_model]
     * @param src_mask 源序列 mask（未使用）
     * @param tgt_mask 目标序列 mask [batch, tgt_len, tgt_len]
     */
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
            return self_attn_.forward(x_in, x_in, x_in, tgt_mask, false);
        };
        auto x1 = sublayer1_.forward(x, self_attn_fn);
        
        // 2. Cross-attention (使用空 mask，抑制警告)
        auto src_attn_fn = [this, &memory](const xarray<T>& x_in) {
            xarray<T> empty_mask;
            return src_attn_.forward(x_in, memory, memory, empty_mask, true);
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

/**
 * @brief 解码器（多层解码器层堆叠）
 */
class Decoder {
public:
    /**
     * @param d_model 模型维度
     * @param heads 注意力头数
     * @param d_ff 前馈网络隐藏层维度
     * @param num_layers 解码器层数
     * @param dropout Dropout 概率
     */
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
    
    /**
     * @brief 前向传播
     * @param x 输入 [batch, tgt_len, d_model]
     * @param memory 编码器输出 [batch, src_len, d_model]
     * @param src_mask 源序列 mask
     * @param tgt_mask 目标序列 mask [batch, tgt_len, tgt_len]
     */
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

} // namespace models
} // namespace transformer