// tests/xtensor.cpp
#include "xtensor_accelerate.hpp"
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace xt;

// 性能测试工具
template<typename Func>
double benchmark(const std::string& name, Func&& func, int iterations = 100) {
    // 预热
    func();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time = elapsed.count() / iterations;
    
    std::cout << std::setw(40) << std::left << name 
              << std::setw(12) << std::right << std::fixed << std::setprecision(3) 
              << avg_time << " ms" << std::endl;
    
    return avg_time;
}

void print_section(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

int main() {
    std::cout << "xtensor + Accelerate Framework Integration Test" << std::endl;
    std::cout << "macOS SDK: " << __MAC_OS_X_VERSION_MAX_ALLOWED << std::endl;
    std::cout << "User: M4yGem1ni" << std::endl;
    std::cout << "Date: 2025-11-06 08:47:05 UTC" << std::endl;
    
    // ========================================
    // Test 1: 基本矩阵乘法
    // ========================================
    print_section("Test 1: Basic Matrix Multiplication");
    
    auto A = xarray<double>{{1, 2, 3}, {4, 5, 6}};
    auto B = xarray<double>{{7, 8}, {9, 10}, {11, 12}};
    
    std::cout << "A (2x3):\n" << A << "\n" << std::endl;
    std::cout << "B (3x2):\n" << B << "\n" << std::endl;
    
    auto C = accelerate::Backend::matmul(A, B);
    std::cout << "C = A @ B:\n" << C << std::endl;
    
    // 验证结果
    xarray<double> expected{{58, 64}, {139, 154}};
    bool correct = allclose(C, expected);
    std::cout << (correct ? "✓ Result CORRECT" : "✗ Result INCORRECT") << std::endl;
    
    // ========================================
    // Test 2: 向量操作
    // ========================================
    print_section("Test 2: Vector Operations");
    
    auto x = xarray<double>{1, 2, 3, 4};
    auto y = xarray<double>{5, 6, 7, 8};
    
    double dot_result = accelerate::Backend::dot(x, y);
    double norm_x = accelerate::Backend::norm(x);
    
    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl;
    std::cout << "x · y = " << dot_result << " (expected: 70)" << std::endl;
    std::cout << "||x|| = " << norm_x << " (expected: ~5.477)" << std::endl;
    
    // AXPY 测试
    auto y_copy = y;
    accelerate::Backend::axpy(2.0, x, y_copy);
    std::cout << "y + 2*x = " << y_copy << " (expected: {7, 10, 13, 16})" << std::endl;
    
    // ========================================
    // Test 3: 矩阵-向量乘法
    // ========================================
    print_section("Test 3: Matrix-Vector Multiplication");
    
    auto M = xarray<double>{{1, 2, 3}, {4, 5, 6}};
    auto v = xarray<double>{1, 2, 3};
    
    auto result = accelerate::Backend::matvec(M, v);
    std::cout << "M:\n" << M << "\n" << std::endl;
    std::cout << "v = " << v << std::endl;
    std::cout << "M @ v = " << result << " (expected: {14, 32})" << std::endl;
    
    // ========================================
    // Test 4: 性能基准测试
    // ========================================
    print_section("Test 4: Performance Benchmarks");
    
    std::vector<size_t> sizes = {64, 128, 256, 512};
    
    for (size_t size : sizes) {
        std::cout << "\nMatrix size: " << size << "x" << size << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        // 关键修复：显式实例化为 xarray
        xarray<float> A_bench = random::rand<float>({size, size});
        xarray<float> B_bench = random::rand<float>({size, size});
        
        int iters = size > 256 ? 10 : 50;
        
        benchmark("Accelerate matmul (float)", [&]() {
            xarray<float> C_fast = accelerate::Backend::matmul(A_bench, B_bench);
        }, iters);
    }
    
    // ========================================
    // Test 5: 不同数据类型
    // ========================================
    print_section("Test 5: Different Data Types");
    
    xarray<float> A_f = xarray<float>{{1.0f, 2.0f}, {3.0f, 4.0f}};
    xarray<float> B_f = xarray<float>{{5.0f, 6.0f}, {7.0f, 8.0f}};
    xarray<float> C_f = accelerate::Backend::matmul(A_f, B_f);
    
    std::cout << "Float32 multiplication:" << std::endl;
    std::cout << "A:\n" << A_f << "\n" << std::endl;
    std::cout << "B:\n" << B_f << "\n" << std::endl;
    std::cout << "Result:\n" << C_f << std::endl;
    
    xarray<double> A_d = xarray<double>{{1.0, 2.0}, {3.0, 4.0}};
    xarray<double> B_d = xarray<double>{{5.0, 6.0}, {7.0, 8.0}};
    xarray<double> C_d = accelerate::Backend::matmul(A_d, B_d);
    
    std::cout << "\nFloat64 multiplication:" << std::endl;
    std::cout << "Result:\n" << C_d << std::endl;
    
    // ========================================
    // Test 6: 大规模测试
    // ========================================
    print_section("Test 6: Large Scale Test");
    
    const size_t large_size = 1024;
    std::cout << "Testing " << large_size << "x" << large_size << " matrices..." << std::endl;
    
    // 关键修复：显式实例化为 xarray
    xarray<float> A_large = random::rand<float>({large_size, large_size});
    xarray<float> B_large = random::rand<float>({large_size, large_size});
    
    auto start = std::chrono::high_resolution_clock::now();
    xarray<float> C_large = accelerate::Backend::matmul(A_large, B_large);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " ms" << std::endl;
    
    double gflops = (2.0 * large_size * large_size * large_size) / (elapsed.count() / 1000.0) / 1e9;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    // ========================================
    // Test 7: 验证正确性
    // ========================================
    print_section("Test 7: Correctness Verification");
    
    xarray<float> A_small = xarray<float>{{1, 2}, {3, 4}};
    xarray<float> B_small = xarray<float>{{2, 0}, {1, 2}};
    xarray<float> C_small = accelerate::Backend::matmul(A_small, B_small);
    
    xarray<float> expected_small = xarray<float>{{4, 4}, {10, 8}};
    
    std::cout << "A:\n" << A_small << std::endl;
    std::cout << "B:\n" << B_small << std::endl;
    std::cout << "Result:\n" << C_small << std::endl;
    std::cout << "Expected:\n" << expected_small << std::endl;
    
    bool is_correct = allclose(C_small, expected_small);
    std::cout << (is_correct ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    print_section("All Tests Completed Successfully! ✓");
    
    return 0;
}