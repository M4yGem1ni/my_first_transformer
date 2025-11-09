// include/models/generator.hpp
#pragma once

#include "solvers/accelerate.hpp"
#include "utils/logger.hpp"
#include "utils/weight_init.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xreducer.hpp>

namespace transformer {
namespace models {

using namespace xt;

/**
 * @brief 输出生成器
 * 
 * 将解码器输出投影到词汇表大小，并应用 log_softmax
 */
class Generator {
public:
    /**
     * @param d_model 模型维度
     * @param vocab_size 词汇表大小
     */
    template<typename T = float>
    Generator(size_t d_model, size_t vocab_size)
        : d_model_(d_model)
        , vocab_size_(vocab_size)
        , proj_(utils::transformer_init<T>({d_model, vocab_size}, 8))
    {
        LOG_DEBUG("Generator initialized: d_model={}, vocab_size={}", d_model, vocab_size);
    }
    
    /**
     * @brief 前向传播
     * @param x 解码器输出 [batch, seq_len, d_model]
     * @return log 概率 [batch, seq_len, vocab_size]
     */
    template<typename T = float>
    xarray<T> forward(const xarray<T>& x) {
        size_t batch = x.shape()[0];
        size_t seq_len = x.shape()[1];
        
        // 投影到词汇表大小
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

} // namespace models
} // namespace transformer