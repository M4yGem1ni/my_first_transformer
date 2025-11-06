// include/xtensor_accelerate.hpp
#pragma once

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

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
    // 矩阵乘法 (GEMM) - 双精度
    // ============================================
    
    static xarray<double> matmul(
        const xarray<double>& A,
        const xarray<double>& B,
        double alpha = 1.0,
        double beta = 0.0)
    {
        validate_matmul_dims(A, B);
        
        size_t M = A.shape()[0];
        size_t K = A.shape()[1];
        size_t N = B.shape()[1];
        
        // 使用 vector 创建实际的内存
        std::vector<double> C_data(M * N, 0.0);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
            alpha,
            A.data(), static_cast<int>(A.shape()[1]),
            B.data(), static_cast<int>(B.shape()[1]),
            beta,
            C_data.data(), static_cast<int>(N)
        );
        
        #pragma clang diagnostic pop
        
        // 转换为 xarray
        return adapt(C_data, {M, N});
    }
    
    // ============================================
    // 矩阵乘法 (GEMM) - 单精度
    // ============================================
    
    static xarray<float> matmul(
        const xarray<float>& A,
        const xarray<float>& B,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        validate_matmul_dims(A, B);
        
        size_t M = A.shape()[0];
        size_t K = A.shape()[1];
        size_t N = B.shape()[1];
        
        std::vector<float> C_data(M * N, 0.0f);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
            alpha,
            A.data(), static_cast<int>(A.shape()[1]),
            B.data(), static_cast<int>(B.shape()[1]),
            beta,
            C_data.data(), static_cast<int>(N)
        );
        
        #pragma clang diagnostic pop
        
        return adapt(C_data, {M, N});
    }
    
    // ============================================
    // 向量点积 - 双精度
    // ============================================
    
    static double dot(const xarray<double>& x, const xarray<double>& y) {
        validate_vector_dims(x, y);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        double result = cblas_ddot(
            static_cast<int>(x.size()),
            x.data(), 1, y.data(), 1
        );
        #pragma clang diagnostic pop
        
        return result;
    }
    
    // ============================================
    // 向量点积 - 单精度
    // ============================================
    
    static float dot(const xarray<float>& x, const xarray<float>& y) {
        validate_vector_dims(x, y);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        float result = cblas_sdot(
            static_cast<int>(x.size()),
            x.data(), 1, y.data(), 1
        );
        #pragma clang diagnostic pop
        
        return result;
    }
    
    // ============================================
    // 向量范数 - 双精度
    // ============================================
    
    static double norm(const xarray<double>& x) {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        double result = cblas_dnrm2(static_cast<int>(x.size()), x.data(), 1);
        #pragma clang diagnostic pop
        return result;
    }
    
    // ============================================
    // 向量范数 - 单精度
    // ============================================
    
    static float norm(const xarray<float>& x) {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        float result = cblas_snrm2(static_cast<int>(x.size()), x.data(), 1);
        #pragma clang diagnostic pop
        return result;
    }
    
    // ============================================
    // AXPY: y = alpha * x + y - 双精度
    // ============================================
    
    static void axpy(double alpha, const xarray<double>& x, xarray<double>& y) {
        validate_vector_dims(x, y);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        cblas_daxpy(
            static_cast<int>(x.size()),
            alpha, x.data(), 1, y.data(), 1
        );
        #pragma clang diagnostic pop
    }
    
    // ============================================
    // AXPY: y = alpha * x + y - 单精度
    // ============================================
    
    static void axpy(float alpha, const xarray<float>& x, xarray<float>& y) {
        validate_vector_dims(x, y);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        cblas_saxpy(
            static_cast<int>(x.size()),
            alpha, x.data(), 1, y.data(), 1
        );
        #pragma clang diagnostic pop
    }
    
    // ============================================
    // 矩阵-向量乘法 (GEMV) - 双精度
    // ============================================
    
    static xarray<double> matvec(
        const xarray<double>& A,
        const xarray<double>& x,
        double alpha = 1.0,
        double beta = 0.0)
    {
        if (A.dimension() != 2) {
            throw std::invalid_argument("A must be 2D");
        }
        if (x.dimension() != 1) {
            throw std::invalid_argument("x must be 1D");
        }
        
        size_t M = A.shape()[0];
        size_t N = A.shape()[1];
        
        if (N != x.size()) {
            throw std::invalid_argument("Matrix-vector dimension mismatch");
        }
        
        std::vector<double> y_data(M, 0.0);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        
        cblas_dgemv(
            CblasRowMajor, CblasNoTrans,
            static_cast<int>(M), static_cast<int>(N),
            alpha, A.data(), static_cast<int>(N),
            x.data(), 1,
            beta, y_data.data(), 1
        );
        
        #pragma clang diagnostic pop
        
        return adapt(y_data, {M});
    }
    
    // ============================================
    // 矩阵-向量乘法 (GEMV) - 单精度
    // ============================================
    
    static xarray<float> matvec(
        const xarray<float>& A,
        const xarray<float>& x,
        float alpha = 1.0f,
        float beta = 0.0f)
    {
        if (A.dimension() != 2 || x.dimension() != 1) {
            throw std::invalid_argument("Invalid dimensions for matvec");
        }
        
        size_t M = A.shape()[0];
        size_t N = A.shape()[1];
        
        if (N != x.size()) {
            throw std::invalid_argument("Matrix-vector dimension mismatch");
        }
        
        std::vector<float> y_data(M, 0.0f);
        
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        
        cblas_sgemv(
            CblasRowMajor, CblasNoTrans,
            static_cast<int>(M), static_cast<int>(N),
            alpha, A.data(), static_cast<int>(N),
            x.data(), 1,
            beta, y_data.data(), 1
        );
        
        #pragma clang diagnostic pop
        
        return adapt(y_data, {M});
    }

    template<typename T>
    static xarray<T> matmul_auto(
        const xarray<T>& A,
        const xarray<T>& B,
        T alpha = T(1),
        T beta = T(0)) {
        if constexpr (std::is_same_v<T, float>) {
            return matmul(A, B, alpha, beta);
        } else if constexpr (std::is_same_v<T, double>) {
            return matmul(A, B, alpha, beta);
        } else {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                         "Only float and double are supported");
        }
    }
    
private:
    template<typename T>
    static void validate_matmul_dims(const xarray<T>& A, const xarray<T>& B) {
        if (A.dimension() != 2 || B.dimension() != 2) {
            throw std::invalid_argument("Both inputs must be 2D matrices");
        }
        if (A.shape()[1] != B.shape()[0]) {
            throw std::invalid_argument(
                "Matrix dimensions incompatible for multiplication: (" +
                std::to_string(A.shape()[0]) + "x" + std::to_string(A.shape()[1]) + ") @ (" +
                std::to_string(B.shape()[0]) + "x" + std::to_string(B.shape()[1]) + ")"
            );
        }
    }
    
    template<typename T>
    static void validate_vector_dims(const xarray<T>& x, const xarray<T>& y) {
        if (x.size() != y.size()) {
            throw std::invalid_argument(
                "Vector sizes don't match: " +
                std::to_string(x.size()) + " vs " + std::to_string(y.size())
            );
        }
    }
};

} // namespace accelerate
} // namespace xt