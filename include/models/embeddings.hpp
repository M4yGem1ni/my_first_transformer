// include/models/embeddings.hpp
#pragma once

#include "utils/logger.hpp"
#include "utils/weight_init.hpp"
#include "solvers/accelerate.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/views/xview.hpp>

namespace transformer {
namespace models {

using namespace xt;

/**
 * @brief 词嵌入层
 * 
 * 将词汇表索引映射到 d_model 维的向量
 */
class Embeddings {
public:
    /**
     * @param vocab_size 词汇表大小
     * @param d_model 嵌入维度
     */
    template<typename T = float>
    Embeddings(size_t vocab_size, size_t d_model)
        : vocab_size_(vocab_size)
        , d_model_(d_model)
        , embedding_table_(utils::transformer_init<T>({vocab_size, d_model}, 7))
    {
        LOG_DEBUG("Embeddings initialized: vocab_size={}, d_model={}", vocab_size, d_model);
    }
    
    /**
     * @brief 前向传播：查找嵌入向量
     * @param x 输入索引 [batch, seq_len]
     * @return 嵌入向量 [batch, seq_len, d_model]
     */
    template<typename T = float>
    xarray<T> forward(const xarray<int>& x) {
        return accelerate::Backend::embedding_lookup(x, embedding_table_);
    }

private:
    size_t vocab_size_;
    size_t d_model_;
    xarray<float> embedding_table_;
};

} // namespace models
} // namespace transformer