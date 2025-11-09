// include/utils/activations.hpp
#pragma once

#include "utils/logger.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/reducers/xreducer.hpp>
#include <xtensor/misc/xmanipulation.hpp>
#include <string>

namespace transformer {
namespace utils {

using namespace xt;

/**
 * @brief Softmax 激活函数
 * @param x 输入张量
 * @param axis 应用 softmax 的轴（默认 -1，即最后一个维度）
 */
template<typename T>
xarray<T> softmax(const xarray<T>& x, int axis = -1) {
    size_t ndim = x.dimension();
    size_t pos_axis = (axis < 0) ? ndim + axis : static_cast<size_t>(axis);
    
    std::string shape_str = "[";
    for (size_t i = 0; i < ndim; ++i) {
        shape_str += std::to_string(x.shape()[i]);
        if (i < ndim - 1) shape_str += ", ";
    }
    shape_str += "]";
    
    LOG_TRACE("softmax - input shape: {}, axis={}", shape_str, pos_axis);
    
    try {
        auto x_max = amax(x, {pos_axis});
        auto x_max_expanded = expand_dims(x_max, pos_axis);
        auto x_shifted = x - x_max_expanded;
        auto exp_x = exp(x_shifted);
        auto sum_exp = sum(exp_x, {pos_axis});
        auto sum_exp_expanded = expand_dims(sum_exp, pos_axis);
        auto result = exp_x / sum_exp_expanded;
        
        xarray<T> result_arr = result;
        LOG_TRACE("softmax completed successfully");
        
        return result_arr;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error in softmax: {}", e.what());
        throw;
    }
}

} // namespace utils
} // namespace transformer