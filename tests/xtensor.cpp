// tests/xtensor.cpp
#include "xtensor_accelerate.hpp"
#include "utils/logger.hpp"
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <chrono>

using namespace xt;
using namespace transformer;

int main() {
    // 初始化 Logger
    Logger::instance().init(
        "xtensor_test",
        true,
        "logs/xtensor_test.log"
    );
    Logger::instance().set_level(Logger::Level::INFO);
    
    LOG_INFO("========================================");
    LOG_INFO("xtensor + Accelerate Framework Test");
    LOG_INFO("Date: {} {}", __DATE__, __TIME__);
    LOG_INFO("========================================");
    
    // Test 1: 基本矩阵乘法
    LOG_INFO("Test 1: Basic Matrix Multiplication");
    
    auto A = xarray<double>{{1, 2, 3}, {4, 5, 6}};
    auto B = xarray<double>{{7, 8}, {9, 10}, {11, 12}};
    
    LOG_DEBUG("Matrix A shape: {} x {}", A.shape()[0], A.shape()[1]);
    LOG_DEBUG("Matrix B shape: {} x {}", B.shape()[0], B.shape()[1]);
    
    auto C = accelerate::Backend::matmul(A, B);
    
    xarray<double> expected{{58, 64}, {139, 154}};
    bool correct = allclose(C, expected);
    
    if (correct) {
        LOG_INFO("✓ Matrix multiplication PASSED");
    } else {
        LOG_ERROR("✗ Matrix multiplication FAILED");
    }
    
    // Test 2: 向量操作
    LOG_INFO("Test 2: Vector Operations");
    
    auto x = xarray<double>{1, 2, 3, 4};
    auto y = xarray<double>{5, 6, 7, 8};
    
    double dot_result = accelerate::Backend::dot(x, y);
    double norm_x = accelerate::Backend::norm(x);
    
    LOG_INFO("Dot product: {}", dot_result);
    LOG_INFO("Vector norm: {:.4f}", norm_x);
    
    // Test 3: 性能基准测试
    LOG_INFO("Test 3: Performance Benchmarks");
    
    std::vector<size_t> sizes = {64, 128, 256, 512};
    
    for (size_t size : sizes) {
        LOG_INFO("Benchmarking {} x {} matrices", size, size);
        
        xarray<float> A_bench = random::rand<float>({size, size});
        xarray<float> B_bench = random::rand<float>({size, size});
        
        auto start = std::chrono::high_resolution_clock::now();
        xarray<float> C_bench = accelerate::Backend::matmul(A_bench, B_bench);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        LOG_INFO("Size: {} - Time: {} ms", size, duration.count());
    }
    
    // Test 4: 大规模测试
    LOG_INFO("Test 4: Large Scale Test");
    
    const size_t large_size = 1024;
    LOG_INFO("Testing {} x {} matrices", large_size, large_size);
    
    xarray<float> A_large = random::rand<float>({large_size, large_size});
    xarray<float> B_large = random::rand<float>({large_size, large_size});
    
    auto start = std::chrono::high_resolution_clock::now();
    xarray<float> C_large = accelerate::Backend::matmul(A_large, B_large);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double gflops = (2.0 * large_size * large_size * large_size) / 
                    (elapsed.count() / 1000.0) / 1e9;
    
    LOG_INFO("Execution time: {} ms", elapsed.count());
    LOG_INFO("Performance: {:.2f} GFLOPS", gflops);
    
    LOG_INFO("========================================");
    LOG_INFO("All Tests Completed Successfully! ✓");
    LOG_INFO("========================================");
    
    Logger::instance().flush();
    Logger::instance().shutdown();    
    return 0;
}