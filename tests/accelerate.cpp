// tests/accelerate.cpp
/**
 * @file accelerate.cpp
 * @brief solvers/accelerate 后端测试套件
 * @author M4yGem1ni
 * @date 2025-11-09
 * 
 * 测试 Apple Accelerate 后端的矩阵运算功能
 */

#include "solvers/accelerate.hpp"
#include "utils/logger.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>           // 🔴 添加 view 支持
#include <xtensor/generators/xrandom.hpp>         // 🔴 添加随机数支持
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace xt;
using namespace transformer;

/**
 * @brief 辅助函数：将 xtensor 对象转换为字符串
 */
template<typename E>
std::string xarray_to_string(const E& expr) {
    std::ostringstream oss;
    oss << expr;
    return oss.str();
}

// ============================================
// 测试辅助函数
// ============================================

/**
 * @brief 检查两个数组是否近似相等
 */
template<typename T>
bool arrays_close(const xarray<T>& a, const xarray<T>& b, T tolerance = 1e-5) {
    if (a.shape() != b.shape()) {
        return false;
    }
    
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a.data()[i] - b.data()[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 打印测试结果
 */
void print_test_result(const std::string& test_name, bool passed) {
    if (passed) {
        LOG_INFO("  ✅ {}", test_name);
    } else {
        LOG_ERROR("  ❌ {}", test_name);
    }
}

/**
 * @brief 打印数组形状
 */
template<typename T>
std::string shape_str(const xarray<T>& arr) {
    std::string result = "[";
    for (size_t i = 0; i < arr.dimension(); ++i) {
        result += std::to_string(arr.shape()[i]);
        if (i < arr.dimension() - 1) result += ", ";
    }
    result += "]";
    return result;
}

// ============================================
// 测试用例
// ============================================

/**
 * @brief 测试 1: 简单 2D 矩阵乘法
 */
bool test_simple_2d_matmul() {
    LOG_INFO("\n=== Test 1: Simple 2D Matrix Multiplication ===");
    
    // 准备测试数据
    xarray<float> A = {{1.0f, 2.0f}, 
                       {3.0f, 4.0f}};
    
    xarray<float> B = {{5.0f, 6.0f}, 
                       {7.0f, 8.0f}};
    
    // 预期结果: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //          = [[19, 22], [43, 50]]
    xarray<float> expected = {{19.0f, 22.0f}, 
                              {43.0f, 50.0f}};
    
    LOG_INFO("A = \n{}", xarray_to_string(A));
    LOG_INFO("B = \n{}", xarray_to_string(B));
    
    // 执行矩阵乘法
    auto C = accelerate::Backend::matmul_2d(A, B);
    
    LOG_INFO("C = A @ B = \n{}", xarray_to_string(C));
    LOG_INFO("Expected = \n{}", xarray_to_string(expected));
    
    // 验证结果
    bool passed = arrays_close(C, expected, 1e-5f);
    print_test_result("Simple 2D matmul", passed);
    
    return passed;
}

/**
 * @brief 测试 2: 不同维度的 2D 矩阵乘法
 */
bool test_rectangular_2d_matmul() {
    LOG_INFO("\n=== Test 2: Rectangular 2D Matrix Multiplication ===");
    
    // A: 3x2, B: 2x4 -> C: 3x4
    xarray<float> A = {{1.0f, 2.0f}, 
                       {3.0f, 4.0f}, 
                       {5.0f, 6.0f}};
    
    xarray<float> B = {{1.0f, 2.0f, 3.0f, 4.0f}, 
                       {5.0f, 6.0f, 7.0f, 8.0f}};
    
    LOG_INFO("A shape: {}", shape_str(A));
    LOG_INFO("B shape: {}", shape_str(B));
    
    auto C = accelerate::Backend::matmul_2d(A, B);
    
    LOG_INFO("C shape: {}", shape_str(C));
    LOG_INFO("C = \n{}", xarray_to_string(C));
    
    // 验证形状
    bool shape_correct = (C.shape()[0] == 3 && C.shape()[1] == 4);
    
    // 验证第一个元素: 1*1 + 2*5 = 11
    bool value_correct = std::abs(C(0, 0) - 11.0f) < 1e-5f;
    
    bool passed = shape_correct && value_correct;
    print_test_result("Rectangular 2D matmul", passed);
    
    return passed;
}

/**
 * @brief 测试 3: 批量矩阵乘法（3D）
 */
bool test_3d_batch_matmul() {
    LOG_INFO("\n=== Test 3: 3D Batch Matrix Multiplication ===");
    
    // 批量大小 2, 每个矩阵 2x2
    xarray<float> A = {{{1.0f, 2.0f}, 
                        {3.0f, 4.0f}},
                       {{5.0f, 6.0f}, 
                        {7.0f, 8.0f}}};
    
    xarray<float> B = {{{1.0f, 0.0f}, 
                        {0.0f, 1.0f}},  // 单位矩阵
                       {{2.0f, 0.0f}, 
                        {0.0f, 2.0f}}}; // 2倍单位矩阵
    
    LOG_INFO("A shape: {}", shape_str(A));
    LOG_INFO("B shape: {}", shape_str(B));
    
    auto C = accelerate::Backend::batch_matmul(A, B);
    
    LOG_INFO("C shape: {}", shape_str(C));
    LOG_INFO("C[0] = \n{}", xarray_to_string(view(C, 0, all(), all())));
    LOG_INFO("C[1] = \n{}", xarray_to_string(view(C, 1, all(), all())));

    // 第一批: A[0] @ I = A[0]
    xarray<float> expected_0 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    bool batch_0_correct = arrays_close(
        xarray<float>(view(C, 0, all(), all())), 
        expected_0, 
        1e-5f
    );
    
    // 第二批: A[1] @ 2I = 2*A[1]
    xarray<float> expected_1 = {{10.0f, 12.0f}, {14.0f, 16.0f}};
    bool batch_1_correct = arrays_close(
        xarray<float>(view(C, 1, all(), all())), 
        expected_1, 
        1e-5f
    );
    
    bool passed = batch_0_correct && batch_1_correct;
    print_test_result("3D batch matmul", passed);
    
    return passed;
}

/**
 * @brief 测试 4: 4D 批量矩阵乘法（Attention 场景）
 */
bool test_4d_batch_matmul_attention() {
    LOG_INFO("\n=== Test 4: 4D Batch Matrix Multiplication (Attention) ===");
    
    // 模拟 Attention: Q @ K^T
    // Q: [batch=1, heads=2, seq_q=3, d_k=4]
    // K_T: [batch=1, heads=2, d_k=4, seq_k=3]
    // Result: [batch=1, heads=2, seq_q=3, seq_k=3]
    
    xarray<float> Q = zeros<float>({1, 2, 3, 4});
    xarray<float> K_T = zeros<float>({1, 2, 4, 3});
    
    // 填充一些值
    for (size_t i = 0; i < Q.size(); ++i) {
        Q.data()[i] = static_cast<float>(i % 10) / 10.0f;
    }
    
    for (size_t i = 0; i < K_T.size(); ++i) {
        K_T.data()[i] = static_cast<float>((i + 5) % 10) / 10.0f;
    }
    
    LOG_INFO("Q shape: {}", shape_str(Q));
    LOG_INFO("K_T shape: {}", shape_str(K_T));
    
    auto scores = accelerate::Backend::batch_matmul(Q, K_T);
    
    LOG_INFO("Scores shape: {}", shape_str(scores));
    
    // 验证形状
    bool shape_correct = (
        scores.shape()[0] == 1 && 
        scores.shape()[1] == 2 && 
        scores.shape()[2] == 3 && 
        scores.shape()[3] == 3
    );

    LOG_INFO("Scores[0,0] = \n{}", xarray_to_string(view(scores, 0, 0, all(), all())));

    bool passed = shape_correct;
    print_test_result("4D batch matmul (Attention)", passed);
    
    return passed;
}

/**
 * @brief 测试 5: 广播（批量大小不匹配）
 */
bool test_broadcast_batch_matmul() {
    LOG_INFO("\n=== Test 5: Broadcast Batch Matrix Multiplication ===");
    
    // A: [1, 2, 2] - 批量大小 1
    // B: [3, 2, 2] - 批量大小 3
    // Result: [3, 2, 2] - 广播 A 到批量大小 3
    
    xarray<float> A = {{{1.0f, 2.0f}, 
                        {3.0f, 4.0f}}};  // 批量 1
    
    xarray<float> B = {{{1.0f, 0.0f}, 
                        {0.0f, 1.0f}},
                       {{2.0f, 0.0f}, 
                        {0.0f, 2.0f}},
                       {{3.0f, 0.0f}, 
                        {0.0f, 3.0f}}};  // 批量 3
    
    LOG_INFO("A shape: {}", shape_str(A));
    LOG_INFO("B shape: {}", shape_str(B));
    
    auto C = accelerate::Backend::batch_matmul(A, B);
    
    LOG_INFO("C shape: {}", shape_str(C));
    
    // 验证形状
    bool shape_correct = (
        C.shape()[0] == 3 && 
        C.shape()[1] == 2 && 
        C.shape()[2] == 2
    );
    
    // 验证第一批: A @ I = A
    xarray<float> expected_0 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    bool batch_0_correct = arrays_close(
        xarray<float>(view(C, 0, all(), all())), 
        expected_0, 
        1e-5f
    );
    
    // 验证第二批: A @ 2I = 2A
    xarray<float> expected_1 = {{2.0f, 4.0f}, {6.0f, 8.0f}};
    bool batch_1_correct = arrays_close(
        xarray<float>(view(C, 1, all(), all())), 
        expected_1, 
        1e-5f
    );
    
    bool passed = shape_correct && batch_0_correct && batch_1_correct;
    print_test_result("Broadcast batch matmul", passed);
    
    return passed;
}

/**
 * @brief 测试 6: matmul_auto 自动选择
 */
bool test_matmul_auto() {
    LOG_INFO("\n=== Test 6: matmul_auto (Automatic Selection) ===");
    
    bool all_passed = true;
    
    // 测试 2D 自动选择
    xarray<float> A_2d = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    xarray<float> B_2d = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    
    auto C_2d = accelerate::Backend::matmul_auto(A_2d, B_2d);
    LOG_INFO("2D auto: shape = {}", shape_str(C_2d));
    
    bool test_2d = (C_2d.dimension() == 2);
    print_test_result("matmul_auto for 2D", test_2d);
    all_passed &= test_2d;
    
    // 测试 3D 自动选择
    xarray<float> A_3d = zeros<float>({2, 3, 4});
    xarray<float> B_3d = zeros<float>({2, 4, 5});
    
    auto C_3d = accelerate::Backend::matmul_auto(A_3d, B_3d);
    LOG_INFO("3D auto: shape = {}", shape_str(C_3d));
    
    bool test_3d = (
        C_3d.shape()[0] == 2 && 
        C_3d.shape()[1] == 3 && 
        C_3d.shape()[2] == 5
    );
    print_test_result("matmul_auto for 3D", test_3d);
    all_passed &= test_3d;
    
    // 测试 4D 自动选择
    xarray<float> A_4d = zeros<float>({1, 8, 10, 64});
    xarray<float> B_4d = zeros<float>({1, 8, 64, 10});
    
    auto C_4d = accelerate::Backend::matmul_auto(A_4d, B_4d);
    LOG_INFO("4D auto: shape = {}", shape_str(C_4d));
    
    bool test_4d = (
        C_4d.shape()[0] == 1 && 
        C_4d.shape()[1] == 8 && 
        C_4d.shape()[2] == 10 && 
        C_4d.shape()[3] == 10
    );
    print_test_result("matmul_auto for 4D", test_4d);
    all_passed &= test_4d;
    
    return all_passed;
}

/**
 * @brief 测试 7: 数值精度测试
 */
bool test_numerical_precision() {
    LOG_INFO("\n=== Test 7: Numerical Precision ===");
    
    // 创建一个已知结果的矩阵乘法
    xarray<float> A = {{0.1f, 0.2f, 0.3f}, 
                       {0.4f, 0.5f, 0.6f}};
    
    xarray<float> B = {{1.0f, 2.0f}, 
                       {3.0f, 4.0f}, 
                       {5.0f, 6.0f}};
    
    // 手动计算预期结果
    // C[0,0] = 0.1*1 + 0.2*3 + 0.3*5 = 0.1 + 0.6 + 1.5 = 2.2
    // C[0,1] = 0.1*2 + 0.2*4 + 0.3*6 = 0.2 + 0.8 + 1.8 = 2.8
    // C[1,0] = 0.4*1 + 0.5*3 + 0.6*5 = 0.4 + 1.5 + 3.0 = 4.9
    // C[1,1] = 0.4*2 + 0.5*4 + 0.6*6 = 0.8 + 2.0 + 3.6 = 6.4
    
    xarray<float> expected = {{2.2f, 2.8f}, 
                              {4.9f, 6.4f}};
    
    auto C = accelerate::Backend::matmul_2d(A, B);

    LOG_INFO("C = \n{}", xarray_to_string(C));
    LOG_INFO("Expected = \n{}", xarray_to_string(expected));

    // 检查每个元素的精度
    bool all_close = true;
    float max_error = 0.0f;
    
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            float error = std::abs(C(i, j) - expected(i, j));
            max_error = std::max(max_error, error);
            if (error > 1e-5f) {
                all_close = false;
                LOG_WARN("Element ({}, {}) error: {}", i, j, error);
            }
        }
    }
    
    LOG_INFO("Maximum error: {:.2e}", max_error);
    
    bool passed = all_close && (max_error < 1e-5f);
    print_test_result("Numerical precision", passed);
    
    return passed;
}

/**
 * @brief 测试 8: 错误处理
 */
bool test_error_handling() {
    LOG_INFO("\n=== Test 8: Error Handling ===");
    
    bool all_passed = true;
    
    // 测试 1: 维度不匹配
    try {
        xarray<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};  // 2x2
        xarray<float> B = {{1.0f, 2.0f, 3.0f}};          // 1x3
        
        auto C = accelerate::Backend::matmul_2d(A, B);  // 应该抛出异常
        
        LOG_ERROR("Should have thrown dimension mismatch exception!");
        all_passed = false;
    } catch (const std::invalid_argument& e) {
        LOG_INFO("Correctly caught dimension mismatch: {}", e.what());
        print_test_result("Dimension mismatch detection", true);
    }
    
    // 测试 2: 非 2D 输入到 matmul_2d
    try {
        xarray<float> A = zeros<float>({2, 3, 4});  // 3D
        xarray<float> B = zeros<float>({4, 5});      // 2D
        
        auto C = accelerate::Backend::matmul_2d(A, B);  // 应该抛出异常
        
        LOG_ERROR("Should have thrown non-2D exception!");
        all_passed = false;
    } catch (const std::invalid_argument& e) {
        LOG_INFO("Correctly caught non-2D input: {}", e.what());
        print_test_result("Non-2D input detection", true);
    }
    
    // 测试 3: 批量乘法维度不匹配
    try {
        xarray<float> A = zeros<float>({2, 3, 5});  // [..., 3, 5]
        xarray<float> B = zeros<float>({2, 4, 6});  // [..., 4, 6]
        
        auto C = accelerate::Backend::batch_matmul(A, B);  // 应该抛出异常
        
        LOG_ERROR("Should have thrown batch dimension mismatch exception!");
        all_passed = false;
    } catch (const std::invalid_argument& e) {
        LOG_INFO("Correctly caught batch dimension mismatch: {}", e.what());
        print_test_result("Batch dimension mismatch detection", true);
    }
    
    return all_passed;
}

/**
 * @brief 测试 9: 性能基准测试
 */
bool test_performance_benchmark() {
    LOG_INFO("\n=== Test 9: Performance Benchmark ===");
    
    const size_t warmup_runs = 5;
    const size_t benchmark_runs = 20;
    
    // 测试不同大小的矩阵
    std::vector<size_t> sizes = {64, 128, 256, 512};
    
    for (size_t size : sizes) {
        xarray<float> A = xt::random::randn<float>({size, size});
        xarray<float> B = xt::random::randn<float>({size, size});
        
        // 预热
        for (size_t i = 0; i < warmup_runs; ++i) {
            auto C = accelerate::Backend::matmul_2d(A, B);
        }
        
        // 基准测试
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < benchmark_runs; ++i) {
            auto C = accelerate::Backend::matmul_2d(A, B);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_time_ms = duration.count() / static_cast<double>(benchmark_runs) / 1000.0;
        double gflops = (2.0 * size * size * size) / (avg_time_ms * 1e6);
        
        LOG_INFO("Matrix size: {}x{}", size, size);
        LOG_INFO("  Average time: {:.3f} ms", avg_time_ms);
        LOG_INFO("  Performance: {:.2f} GFLOPS", gflops);
    }
    
    print_test_result("Performance benchmark completed", true);
    return true;
}

/**
 * @brief 测试 10: 大规模批量乘法（真实 Transformer 场景）
 */
bool test_transformer_realistic_scenario() {
    LOG_INFO("\n=== Test 10: Transformer Realistic Scenario ===");
    
    // 模拟真实的 Transformer 参数
    const size_t batch_size = 2;
    const size_t num_heads = 8;
    const size_t seq_len = 128;
    const size_t d_k = 64;
    
    LOG_INFO("Simulating Transformer Attention:");
    LOG_INFO("  batch_size: {}", batch_size);
    LOG_INFO("  num_heads: {}", num_heads);
    LOG_INFO("  seq_len: {}", seq_len);
    LOG_INFO("  d_k: {}", d_k);
    
    // Q @ K^T
    xarray<float> Q = xt::random::randn<float>({batch_size, num_heads, seq_len, d_k});
    xarray<float> K_T = xt::random::randn<float>({batch_size, num_heads, d_k, seq_len});
    
    LOG_INFO("\nComputing Q @ K^T...");
    auto start = std::chrono::high_resolution_clock::now();
    
    auto scores = accelerate::Backend::batch_matmul(Q, K_T);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    LOG_INFO("Scores shape: {}", shape_str(scores));
    LOG_INFO("Time: {} ms", duration.count());
    
    // 验证形状
    bool shape_correct = (
        scores.shape()[0] == batch_size &&
        scores.shape()[1] == num_heads &&
        scores.shape()[2] == seq_len &&
        scores.shape()[3] == seq_len
    );
    
    // Attention @ V
    xarray<float> V = xt::random::randn<float>({batch_size, num_heads, seq_len, d_k});
    
    LOG_INFO("\nComputing Attention @ V...");
    start = std::chrono::high_resolution_clock::now();
    
    auto output = accelerate::Backend::batch_matmul(scores, V);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    LOG_INFO("Output shape: {}", shape_str(output));
    LOG_INFO("Time: {} ms", duration.count());
    
    // 验证输出形状
    bool output_shape_correct = (
        output.shape()[0] == batch_size &&
        output.shape()[1] == num_heads &&
        output.shape()[2] == seq_len &&
        output.shape()[3] == d_k
    );
    
    bool passed = shape_correct && output_shape_correct;
    print_test_result("Transformer realistic scenario", passed);
    
    return passed;
}

// ============================================
// 主函数
// ============================================

int main() {
    try {
        // 初始化日志系统
        Logger::instance().init(
            "xtensor_accelerate_test",
            true,
            "logs/xtensor_accelerate_test.log",
            1024 * 1024 * 10,
            3
        );
        
        Logger::instance().set_level(Logger::Level::INFO);
        
        LOG_INFO("╔═══════════════════════════════════════════════════════════╗");
        LOG_INFO("║  xtensor_accelerate Backend Test Suite                   ║");
        LOG_INFO("╚═══════════════════════════════════════════════════════════╝");
        LOG_INFO("");
        LOG_INFO("Date: 2025-11-09 12:40:34 (UTC)");
        LOG_INFO("User: M4yGem1ni");
        LOG_INFO("Platform: Apple Accelerate Framework");
        LOG_INFO("");
        
        // 运行所有测试
        std::vector<std::pair<std::string, std::function<bool()>>> tests = {
            {"Simple 2D matmul", test_simple_2d_matmul},
            {"Rectangular 2D matmul", test_rectangular_2d_matmul},
            {"3D batch matmul", test_3d_batch_matmul},
            {"4D batch matmul (Attention)", test_4d_batch_matmul_attention},
            {"Broadcast batch matmul", test_broadcast_batch_matmul},
            {"matmul_auto", test_matmul_auto},
            {"Numerical precision", test_numerical_precision},
            {"Error handling", test_error_handling},
            {"Performance benchmark", test_performance_benchmark},
            {"Transformer realistic scenario", test_transformer_realistic_scenario}
        };
        
        int passed_count = 0;
        int total_count = tests.size();
        
        for (const auto& [name, test_func] : tests) {
            try {
                if (test_func()) {
                    passed_count++;
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Test '{}' threw exception: {}", name, e.what());
            }
        }
        
        // 打印测试总结
        LOG_INFO("");
        LOG_INFO("╔═══════════════════════════════════════════════════════════╗");
        LOG_INFO("║  Test Summary                                             ║");
        LOG_INFO("╚═══════════════════════════════════════════════════════════╝");
        LOG_INFO("Total tests: {}", total_count);
        LOG_INFO("Passed: {}", passed_count);
        LOG_INFO("Failed: {}", total_count - passed_count);
        LOG_INFO("Success rate: {:.1f}%", 
                (100.0 * passed_count) / total_count);
        
        if (passed_count == total_count) {
            LOG_INFO("");
            LOG_INFO("🎉 All tests passed! xtensor_accelerate is working correctly!");
            Logger::instance().shutdown();
            return 0;
        } else {
            LOG_ERROR("");
            LOG_ERROR("❌ Some tests failed. Please review the output above.");
            Logger::instance().shutdown();
            return 1;
        }
        
    } catch (const std::exception& e) {
        LOG_CRITICAL("Fatal error: {}", e.what());
        Logger::instance().shutdown();
        return 1;
    }
}