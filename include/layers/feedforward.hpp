// include/layers/feedforward.hpp
#pragma once

#include "solvers/accelerate.hpp"
#include "utils/logger.hpp"
#include "utils/weight_init.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <xtensor/core/xmath.hpp>

namespace transformer {
namespace layers {

using namespace xt;

/**
 * @brief 位置前馈网络（Position-wise Feed-Forward Network）
 * 
 * FFN(x) = max(0, xW1 + b1)W2 + b2
 */
template<typename T = float>
class PositionwiseFeedForward {
public:
    /**
     * @param d_model 模型维度
     * @param d_ff 前馈网络隐藏层维度
     * @param dropout Dropout 概率
     */
    PositionwiseFeedForward(size_t d_model, size_t d_ff, T dropout = 0.1)
        : d_model_(d_model)
        , d_ff_(d_ff)
        , dropout_(dropout)
    {
        W1_ = utils::transformer_init<T>({d_model_, d_ff_}, 5);
        b1_ = zeros<T>({d_ff_});
        W2_ = utils::transformer_init<T>({d_ff_, d_model_}, 6);
        b2_ = zeros<T>({d_model_});
        
        LOG_DEBUG("FFN initialized: d_model={}, d_ff={}", d_model, d_ff);
    }
    
    /**
     * @brief 前向传播
     */
    xarray<T> forward(const xarray<T>& x) {
        size_t batch = x.shape()[0];
        size_t seq_len = x.shape()[1];
        
        // 第一层：线性变换 + ReLU
        auto x_2d = reshape_view(x, {batch * seq_len, d_model_});
        xarray<T> x_2d_arr = x_2d;
        auto hidden = accelerate::Backend::matmul_auto(x_2d_arr, W1_);
        hidden = hidden + b1_;
        hidden = xt::maximum(hidden, T(0));  // ReLU
        
        // 第二层：线性变换
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

} // namespace layers
} // namespace transformer