// include/models/encoder_decoder.hpp
#pragma once

#include "models/encoder.hpp"
#include "models/decoder.hpp"
#include "models/embeddings.hpp"
#include "models/generator.hpp"
#include "utils/logger.hpp"
#include <xtensor/containers/xarray.hpp>
#include <memory>

namespace transformer {
namespace models {

using namespace xt;

/**
 * @brief 完整的 Encoder-Decoder Transformer 模型
 */
template<typename T = float>
class EncoderDecoder {
public:
    /**
     * @param encoder 编码器
     * @param decoder 解码器
     * @param src_embed 源语言嵌入层
     * @param tgt_embed 目标语言嵌入层
     * @param generator 输出生成器
     */
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
    
    /**
     * @brief 完整的前向传播
     * @param src 源序列 [batch, src_len]
     * @param tgt 目标序列 [batch, tgt_len]
     * @param src_mask 源序列 mask [batch, src_len, src_len]
     * @param tgt_mask 目标序列 mask [batch, tgt_len, tgt_len]
     * @return 解码器输出 [batch, tgt_len, d_model]
     */
    xarray<T> forward(
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
    
    /**
     * @brief 编码
     * @param src 源序列 [batch, src_len]
     * @param src_mask 源序列 mask
     * @return 编码器输出 [batch, src_len, d_model]
     */
    xarray<T> encode(const xarray<T>& src, const xarray<T>& src_mask)
    {
        auto embedded = src_embed_->template forward<T>(src);
        return encoder_->template forward<T>(embedded, src_mask);
    }
    
    /**
     * @brief 解码
     * @param memory 编码器输出 [batch, src_len, d_model]
     * @param src_mask 源序列 mask
     * @param tgt 目标序列 [batch, tgt_len]
     * @param tgt_mask 目标序列 mask
     * @return 解码器输出 [batch, tgt_len, d_model]
     */
    xarray<T> decode(
        const xarray<T>& memory,
        const xarray<T>& src_mask,
        const xarray<T>& tgt,
        const xarray<T>& tgt_mask)
    {
        auto embedded = tgt_embed_->template forward<T>(tgt);
        return decoder_->template forward<T>(embedded, memory, src_mask, tgt_mask);
    }

private:
    std::shared_ptr<Encoder> encoder_;
    std::shared_ptr<Decoder> decoder_;
    std::shared_ptr<Embeddings> src_embed_;
    std::shared_ptr<Embeddings> tgt_embed_;
    std::shared_ptr<Generator> generator_;
};

} // namespace models
} // namespace transformer