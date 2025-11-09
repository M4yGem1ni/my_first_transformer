// include/layers/sublayer.hpp
#pragma once

#include "layers/layer_norm.hpp"
#include "utils/logger.hpp"
#include <xtensor/containers/xarray.hpp>

namespace transformer {
namespace layers {

using namespace xt;

/**
 * @brief 残差连接与 Layer Normalization
 * 
 * 实现 x + Sublayer(LayerNorm(x))
 */
template<typename T = float>
class SublayerConnection {
public:
    /**
     * @param size 特征维度大小
     * @param dropout Dropout 概率（推理时不使用）
     */
    SublayerConnection(size_t size, T dropout = 0.1)
        : norm_(size)
        , dropout_(dropout)
    {
        LOG_TRACE("SublayerConnection initialized: size={}, dropout={}", size, dropout);
    }
    
    /**
     * @brief 前向传播
     * @param x 输入
     * @param sublayer 子层函数（lambda 或 functor）
     */
    template<typename Sublayer>
    xarray<T> forward(const xarray<T>& x, Sublayer sublayer) {
        return x + apply_dropout(sublayer(norm_.forward(x)));
    }

private:
    LayerNorm<T> norm_;
    T dropout_;
    
    xarray<T> apply_dropout(const xarray<T>& x) {
        // 推理时不使用 dropout
        return x;
    }
};

} // namespace layers
} // namespace transformer