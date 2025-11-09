// include/solvers/accelerate.hpp
#pragma once

#include <cstddef>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include <vector>
#include "utils/logger.hpp"

/**
 * @file accelerate.hpp
 * @brief xtensor 的 Apple Accelerate 后端
 * @author M4yGem1ni
 * @date 2025-11-09
 * 
 * 提供基于 Apple Accelerate Framework 的高性能矩阵运算
 */

namespace xt {
namespace accelerate {

/**
 * @brief Apple Accelerate 后端类
 * 
 * 封装了基于 BLAS 的矩阵乘法操作
 */
class Backend {
public:
    // ============================================
    // 2D 矩阵乘法 - 单精度
    // ============================================
    
    /**
     * @brief 2D 矩阵乘法（单精度浮点）
     * @param A 矩阵 A [M x K]
     * @param B 矩阵 B [K x N]
     * @param alpha 缩放因子（默认 1.0）
     * @param beta 输出矩阵的缩放因子（默认 0.0）
     * @return 结果矩阵 C = alpha * A @ B + beta * C [M x N]
     */
    static xarray<float> matmul_2d(
        const xarray<float>& A,
        const xarray<float>& B,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        validate_matmul_2d_dims(A, B);
        
        size_t M = A.shape()[0];
        size_t K = A.shape()[1];
        size_t N = B.shape()[1];
        
        LOG_TRACE("matmul_2d<float>: [{} x {}] @ [{} x {}]", M, K, K, N);
        
        xarray<float> C = zeros<float>({M, N});
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
            alpha,
            A.data(), static_cast<int>(K),
            B.data(), static_cast<int>(N),
            beta,
            C.data(), static_cast<int>(N)
        );
        
        #pragma clang diagnostic pop
        
        return C;
    }

    // ============================================
    // 批量矩阵乘法 - 单精度
    // ============================================
    
    /**
     * @brief 批量矩阵乘法（单精度浮点）
     * @param A 矩阵 A [..., M, K]
     * @param B 矩阵 B [..., K, N]
     * @param alpha 缩放因子（默认 1.0）
     * @param beta 输出矩阵的缩放因子（默认 0.0）
     * @return 结果矩阵 C [..., M, N]
     * 
     * 支持广播：如果 A 和 B 的批量维度不同，会自动广播
     * 例如：A[1,8,5,64] @ B[1,8,64,5] -> C[1,8,5,5]
     */
    static xarray<float> batch_matmul(
        const xarray<float>& A,
        const xarray<float>& B,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        auto A_shape = A.shape();
        auto B_shape = B.shape();
        
        size_t A_ndim = A.dimension();
        size_t B_ndim = B.dimension();
        
        size_t M = A_shape[A_ndim - 2];
        size_t K = A_shape[A_ndim - 1];
        size_t K2 = B_shape[B_ndim - 2];
        size_t N = B_shape[B_ndim - 1];
        
        if (K != K2) {
            LOG_ERROR("Matrix multiplication dimensions incompatible: A[...,{},{}] @ B[...,{},{}]", 
                     M, K, K2, N);
            throw std::invalid_argument("Matrix multiplication dimensions incompatible");
        }
        
        // 计算批量大小
        size_t A_batch_size = 1;
        for (size_t i = 0; i < A_ndim - 2; ++i) {
            A_batch_size *= A_shape[i];
        }
        
        size_t B_batch_size = 1;
        for (size_t i = 0; i < B_ndim - 2; ++i) {
            B_batch_size *= B_shape[i];
        }
        
        size_t output_batch_size = std::max(A_batch_size, B_batch_size);
        
        // 构建输出形状
        std::vector<size_t> output_shape;
        
        if (A_batch_size >= B_batch_size) {
            for (size_t i = 0; i < A_ndim - 2; ++i) {
                output_shape.push_back(A_shape[i]);
            }
        } else {
            for (size_t i = 0; i < B_ndim - 2; ++i) {
                output_shape.push_back(B_shape[i]);
            }
        }
        
        output_shape.push_back(M);
        output_shape.push_back(N);
        
        LOG_TRACE("batch_matmul<float>: batch_size={}, [{} x {}] @ [{} x {}]",
                 output_batch_size, M, K, K, N);
        
        xarray<float> C = zeros<float>(output_shape);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        
        // 对每个批次执行矩阵乘法
        for (size_t b = 0; b < output_batch_size; ++b) {
            size_t a_idx = (A_batch_size == 1) ? 0 : (b % A_batch_size);
            size_t b_idx = (B_batch_size == 1) ? 0 : (b % B_batch_size);
            
            size_t A_offset = a_idx * M * K;
            size_t B_offset = b_idx * K * N;
            size_t C_offset = b * M * N;
            
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                alpha,
                A.data() + A_offset, static_cast<int>(K),
                B.data() + B_offset, static_cast<int>(N),
                beta,
                C.data() + C_offset, static_cast<int>(N)
            );
        }
        
        #pragma clang diagnostic pop
        
        return C;
    }

    // ============================================
    // 自动矩阵乘法 - 根据维度自动选择
    // ============================================
    
    /**
     * @brief 自动选择合适的矩阵乘法实现
     * @param A 矩阵 A
     * @param B 矩阵 B
     * @param alpha 缩放因子（默认 1.0）
     * @param beta 输出矩阵的缩放因子（默认 0.0）
     * @return 结果矩阵 C = alpha * A @ B + beta * C
     * 
     * 根据输入维度自动选择：
     * - 如果 A 和 B 都是 2D，使用 matmul_2d
     * - 否则使用 batch_matmul
     */
    template<typename T>
    static xarray<T> matmul_auto(
        const xarray<T>& A,
        const xarray<T>& B,
        T alpha = T(1),
        T beta = T(0)) 
    {
        if (A.dimension() < 2 || B.dimension() < 2) {
            LOG_ERROR("Both inputs must be at least 2D for matrix multiplication");
            throw std::invalid_argument("Both inputs must be at least 2D for matrix multiplication");
        }
        
        if (A.dimension() == 2 && B.dimension() == 2) {
            return matmul_2d(A, B, alpha, beta);
        } else {
            return batch_matmul(A, B, alpha, beta);
        }
    }

private:
    /**
     * @brief 验证 2D 矩阵乘法的维度
     */
    template<typename T>
    static void validate_matmul_2d_dims(const xarray<T>& A, const xarray<T>& B) {
        if (A.dimension() != 2 || B.dimension() != 2) {
            LOG_ERROR("Both inputs must be 2D matrices");
            throw std::invalid_argument("Both inputs must be 2D matrices");
        }
        if (A.shape()[1] != B.shape()[0]) {
            LOG_ERROR("Matrix dimensions incompatible: ({} x {}) @ ({} x {})",
                     A.shape()[0], A.shape()[1], B.shape()[0], B.shape()[1]);
            throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
        }
    }
};

} // namespace accelerate
} // namespace xt