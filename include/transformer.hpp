// include/transformer.hpp
#pragma once

/**
 * @file transformer.hpp
 * @brief Transformer 模型主入口头文件
 * @author M4yGem1ni
 * @date 2025-11-09
 * 
 * 这个头文件包含了整个 Transformer 模型所需的所有组件
 */

// 工具函数
#include "utils/weight_init.hpp"
#include "utils/activations.hpp"

// 基础层
#include "layers/layer_norm.hpp"
#include "layers/sublayer.hpp"
#include "layers/attention.hpp"
#include "layers/feedforward.hpp"

// 模型组件
#include "models/embeddings.hpp"
#include "models/generator.hpp"
#include "models/encoder.hpp"
#include "models/decoder.hpp"
#include "models/encoder_decoder.hpp"

// 命名空间别名，方便使用
namespace transformer {

// 导出常用类型
using EncoderDecoder = models::EncoderDecoder<float>;
using Encoder = models::Encoder;
using Decoder = models::Decoder;
using Embeddings = models::Embeddings;
using Generator = models::Generator;

template<typename T = float>
using MultiHeadAttention = layers::MultiHeadAttention<T>;

template<typename T = float>
using PositionwiseFeedForward = layers::PositionwiseFeedForward<T>;

template<typename T = float>
using LayerNorm = layers::LayerNorm<T>;

} // namespace transformer

/**
 * @example 基本使用示例
 * 
 * @code
 * #include "core/transformer.hpp"
 * 
 * using namespace transformer;
 * 
 * // 创建模型
 * auto encoder = std::make_shared<Encoder>(512, 8, 2048, 6);
 * auto decoder = std::make_shared<Decoder>(512, 8, 2048, 6);
 * auto src_embed = std::make_shared<Embeddings>(1000, 512);
 * auto tgt_embed = std::make_shared<Embeddings>(1000, 512);
 * auto generator = std::make_shared<Generator>(512, 1000);
 * 
 * auto model = std::make_shared<EncoderDecoder>(
 *     encoder, decoder, src_embed, tgt_embed, generator
 * );
 * 
 * // 前向传播
 * auto output = model->forward(src, tgt, src_mask, tgt_mask);
 * @endcode
 */