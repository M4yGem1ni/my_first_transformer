// include/layers/layer_norm.hpp
#pragma once

#include "utils/logger.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xreducer.hpp>

namespace transformer {
namespace layers {

using namespace xt;

/**
 * @brief Layer Normalization
 * 
 * 对输入的最后一个维度进行归一化
 */
template<typename T = float>
class LayerNorm {
public:
    /**
     * @param features 特征维度大小
     * @param eps 防止除零的小值
     */
    LayerNorm(size_t features, T eps = 1e-6)
        : features_(features)
        , eps_(eps)
        , gamma_(ones<T>({features}))
        , beta_(zeros<T>({features}))
    {
        LOG_TRACE("LayerNorm initialized: features={}, eps={}", features, eps);
    }
    
    /**
     * @brief 前向传播
     */
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
    xarray<T> gamma_;  // 缩放参数
    xarray<T> beta_;   // 平移参数
};

} // namespace layers
} // namespace transformer