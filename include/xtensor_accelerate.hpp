#pragma once

#include <cstddef>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/views/xstrided_view.hpp>
#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "utils/logger.hpp" 

namespace xt {
namespace accelerate {

// 类型特征辅助
template<typename T>
struct blas_traits;

template<>
struct blas_traits<float> {
    using value_type = float;
    static constexpr bool is_supported = true;
};

template<>
struct blas_traits<double> {
    using value_type = double;
    static constexpr bool is_supported = true;
};

// 主要的 Accelerate 后端类
class Backend {
public:
    // ============================================
    // 矩阵乘法 (GEMM) - 2D 版本 - 双精度
    // ============================================
    
    static xarray<double> matmul_2d(
        const xarray<double>& A,
        const xarray<double>& B,
        double alpha = 1.0,
        double beta = 0.0)
    {
        validate_matmul_2d_dims(A, B);
        
        size_t M = A.shape()[0];
        size_t K = A.shape()[1];
        size_t N = B.shape()[1];
        
        LOG_TRACE("matmul_2d: [{} x {}] @ [{} x {}]", M, K, K, N);
        
        // 创建输出数组
        xarray<double> C = zeros<double>({M, N});
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        
        cblas_dgemm(
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
    // 矩阵乘法 (GEMM) - 2D 版本 - 单精度
    // ============================================
    
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
        
        LOG_TRACE("matmul_2d: [{} x {}] @ [{} x {}]", M, K, K, N);
        
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
        
        // 使用 DEBUG 级别记录详细信息
        if (transformer::Logger::instance().get_level() <= transformer::Logger::Level::DEBUG) {
            std::string shape_str = "[";
            for (size_t i = 0; i < output_shape.size(); ++i) {
                shape_str += std::to_string(output_shape[i]);
                if (i < output_shape.size() - 1) shape_str += ", ";
            }
            shape_str += "]";
            LOG_DEBUG("batch_matmul output_shape = {}", shape_str);
        }
        
        xarray<float> C = zeros<float>(output_shape);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        
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
    // 批量矩阵乘法 - 双精度
    // ============================================

    static xarray<double> batch_matmul(
        const xarray<double>& A,
        const xarray<double>& B,
        double alpha = 1.0,
        double beta = 0.0)
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
        
        size_t A_batch_size = 1;
        for (size_t i = 0; i < A_ndim - 2; ++i) {
            A_batch_size *= A_shape[i];
        }
        
        size_t B_batch_size = 1;
        for (size_t i = 0; i < B_ndim - 2; ++i) {
            B_batch_size *= B_shape[i];
        }
        
        size_t output_batch_size = std::max(A_batch_size, B_batch_size);
        
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
        
        xarray<double> C = zeros<double>(output_shape);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        
        for (size_t b = 0; b < output_batch_size; ++b) {
            size_t a_idx = (A_batch_size == 1) ? 0 : (b % A_batch_size);
            size_t b_idx = (B_batch_size == 1) ? 0 : (b % B_batch_size);
            
            size_t A_offset = a_idx * M * K;
            size_t B_offset = b_idx * K * N;
            size_t C_offset = b * M * N;
            
            cblas_dgemm(
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

    // ============================================
    // 兼容性接口
    // ============================================
    
    static xarray<double> matmul(
        const xarray<double>& A,
        const xarray<double>& B,
        double alpha = 1.0,
        double beta = 0.0)
    {
        return matmul_auto(A, B, alpha, beta);
    }
    
    static xarray<float> matmul(
        const xarray<float>& A,
        const xarray<float>& B,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        return matmul_auto(A, B, alpha, beta);
    }
    
    // ============================================
    // 其他 BLAS 操作（省略详细实现，与原来相同）
    // ============================================
    
    static double dot(const xarray<double>& x, const xarray<double>& y) {
        validate_vector_dims(x, y);
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        double result = cblas_ddot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
        #pragma clang diagnostic pop
        return result;
    }
    
    static float dot(const xarray<float>& x, const xarray<float>& y) {
        validate_vector_dims(x, y);
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        float result = cblas_sdot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
        #pragma clang diagnostic pop
        return result;
    }
    
    static double norm(const xarray<double>& x) {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        double result = cblas_dnrm2(static_cast<int>(x.size()), x.data(), 1);
        #pragma clang diagnostic pop
        return result;
    }
    
    static float norm(const xarray<float>& x) {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        float result = cblas_snrm2(static_cast<int>(x.size()), x.data(), 1);
        #pragma clang diagnostic pop
        return result;
    }
    
    static void axpy(double alpha, const xarray<double>& x, xarray<double>& y) {
        validate_vector_dims(x, y);
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        cblas_daxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
        #pragma clang diagnostic pop
    }
    
    static void axpy(float alpha, const xarray<float>& x, xarray<float>& y) {
        validate_vector_dims(x, y);
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        cblas_saxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
        #pragma clang diagnostic pop
    }
    
    static xarray<double> matvec(const xarray<double>& A, const xarray<double>& x,
                                 double alpha = 1.0, double beta = 0.0) {
        if (A.dimension() != 2 || x.dimension() != 1) {
            LOG_ERROR("Invalid dimensions for matvec");
            throw std::invalid_argument("Invalid dimensions for matvec");
        }
        
        size_t M = A.shape()[0];
        size_t N = A.shape()[1];
        
        if (N != x.size()) {
            LOG_ERROR("Matrix-vector dimension mismatch");
            throw std::invalid_argument("Matrix-vector dimension mismatch");
        }
        
        xarray<double> y = zeros<double>({M});
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(M), static_cast<int>(N),
                   alpha, A.data(), static_cast<int>(N), x.data(), 1, beta, y.data(), 1);
        #pragma clang diagnostic pop
        
        return y;
    }
    
    static xarray<float> matvec(const xarray<float>& A, const xarray<float>& x,
                                float alpha = 1.0f, float beta = 0.0f) {
        if (A.dimension() != 2 || x.dimension() != 1) {
            LOG_ERROR("Invalid dimensions for matvec");
            throw std::invalid_argument("Invalid dimensions for matvec");
        }
        
        size_t M = A.shape()[0];
        size_t N = A.shape()[1];
        
        if (N != x.size()) {
            LOG_ERROR("Matrix-vector dimension mismatch");
            throw std::invalid_argument("Matrix-vector dimension mismatch");
        }
        
        xarray<float> y = zeros<float>({M});
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(M), static_cast<int>(N),
                   alpha, A.data(), static_cast<int>(N), x.data(), 1, beta, y.data(), 1);
        #pragma clang diagnostic pop
        
        return y;
    }
    
private:
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
    
    template<typename T>
    static void validate_vector_dims(const xarray<T>& x, const xarray<T>& y) {
        if (x.size() != y.size()) {
            LOG_ERROR("Vector sizes don't match: {} vs {}", x.size(), y.size());
            throw std::invalid_argument("Vector sizes don't match");
        }
    }
};

} // namespace accelerate
} // namespace xt