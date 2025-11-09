// include/utils/weight_init.hpp
#pragma once

#include "utils/logger.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <vector>
#include <random>
#include <cmath>

namespace transformer {
namespace utils {

using namespace xt;

/**
 * @brief Xavier/Glorot uniform 初始化
 */
template<typename T = float>
xarray<T> xavier_uniform(const std::vector<size_t>& shape, uint32_t seed = 42) {
    size_t fan_in = shape[0];
    size_t fan_out = shape[1];
    T limit = std::sqrt(6.0 / (fan_in + fan_out));
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dist(-limit, limit);
    
    xarray<T> result = zeros<T>(shape);
    for (auto& val : result) {
        val = dist(gen);
    }
    
    LOG_TRACE("xavier_uniform initialized: shape=[{}, {}], limit={:.4f}", 
              shape[0], shape[1], limit);
    
    return result;
}

/**
 * @brief 正态分布初始化
 */
template<typename T = float>
xarray<T> randn(const std::vector<size_t>& shape, T mean = 0.0, T stddev = 1.0, uint32_t seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(mean, stddev);
    
    xarray<T> result = zeros<T>(shape);
    for (auto& val : result) {
        val = dist(gen);
    }
    
    LOG_TRACE("randn initialized: shape=[{}, {}], mean={:.4f}, stddev={:.4f}", 
              shape[0], shape[1], mean, stddev);
    
    return result;
}

/**
 * @brief Transformer 专用初始化（小标准差）
 */
template<typename T = float>
xarray<T> transformer_init(const std::vector<size_t>& shape, uint32_t seed = 42) {
    T stddev = std::sqrt(T(2.0) / T(shape[0] + shape[1]));
    stddev = std::min(stddev, T(0.02));
    
    std::mt19937 gen(seed);
    std::normal_distribution<T> dist(T(0), stddev);
    
    xarray<T> result = zeros<T>(shape);
    for (auto& val : result) {
        val = dist(gen);
    }
    
    LOG_TRACE("transformer_init: shape=[{}, {}], stddev={:.4f}", 
              shape[0], shape[1], stddev);
    
    return result;
}

/**
 * @brief Glorot/Xavier uniform 初始化（随机种子）
 */
template<typename T = float>
xarray<T> glorot_uniform(const std::vector<size_t>& shape) {
    T limit = std::sqrt(6.0 / (shape[0] + shape[1]));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-limit, limit);
    
    xarray<T> result = zeros<T>(shape);
    for (auto& val : result) {
        val = dist(gen);
    }
    
    LOG_TRACE("glorot_uniform initialized: shape=[{}, {}], limit={:.4f}", 
              shape[0], shape[1], limit);
    
    return result;
}

} // namespace utils
} // namespace transformer