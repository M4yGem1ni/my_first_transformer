// tests/logger.cpp
#include "utils/logger.hpp"
#include <thread>
#include <chrono>
#include <vector>

using namespace transformer;

void test_basic_logging() {
    LOG_INFO("=== Test 1: Basic Logging ===");
    
    LOG_TRACE("This is a trace message");
    LOG_DEBUG("This is a debug message");
    LOG_INFO("This is an info message");
    LOG_WARN("This is a warning message");
    LOG_ERROR("This is an error message");
    LOG_CRITICAL("This is a critical message");
}

void test_formatted_logging() {
    LOG_INFO("=== Test 2: Formatted Logging ===");
    
    int x = 42;
    double y = 3.14159;
    std::string name = "Transformer";
    
    LOG_INFO("Integer: {}, Double: {:.4f}, String: {}", x, y, name);
    LOG_DEBUG("Matrix dimensions: {} x {}", 512, 512);
    LOG_WARN("Memory usage: {:.2f} MB", 1024.5);
}

void test_log_levels() {
    LOG_INFO("=== Test 3: Log Levels ===");
    
    LOG_INFO("Current log level: {}", 
        static_cast<int>(Logger::instance().get_level()));
    
    LOG_INFO("Setting log level to WARN...");
    Logger::instance().set_level(Logger::Level::WARN);
    
    LOG_TRACE("This should NOT appear (TRACE)");
    LOG_DEBUG("This should NOT appear (DEBUG)");
    LOG_INFO("This should NOT appear (INFO)");
    LOG_WARN("This SHOULD appear (WARN)");
    LOG_ERROR("This SHOULD appear (ERROR)");
    
    // 恢复默认级别
    Logger::instance().set_level(Logger::Level::INFO);
    LOG_INFO("Log level restored to INFO");
}

void worker_thread(int id) {
    for (int i = 0; i < 5; ++i) {
        LOG_INFO("Thread {} iteration {}", id, i);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void test_multithreading() {
    LOG_INFO("=== Test 4: Multithreading ===");
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back(worker_thread, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    LOG_INFO("All threads completed");
}

void test_conditional_logging() {
    LOG_INFO("=== Test 5: Conditional Logging ===");
    
    int value = 42;
    LOG_IF(value > 40, INFO, "Value is greater than 40: {}", value);
    LOG_IF(value < 40, WARN, "This should NOT appear");
    
    bool is_training = true;
    LOG_IF(is_training, DEBUG, "Model is in training mode");
}

void expensive_computation() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void test_performance_timing() {
    LOG_INFO("=== Test 6: Performance Timing ===");
    
    // 方法 1: 使用开始/结束宏
    LOG_TIME_START(task1);
    expensive_computation();
    LOG_TIME_END(task1);
    
    // 方法 2: 再次测试
    LOG_TIME_START(task2);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    LOG_TIME_END(task2);
}

void test_custom_pattern() {
    LOG_INFO("=== Test 7: Custom Pattern ===");
    
    LOG_INFO("Default pattern");
    
    Logger::instance().set_pattern("[%H:%M:%S] [%^%l%$] %v");
    LOG_INFO("Simple pattern - no file info");
    LOG_ERROR("Error with simple pattern");
    
    // 恢复默认模式
    Logger::instance().set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
    LOG_INFO("Pattern restored");
}

void simulate_model_training() {
    LOG_INFO("=== Test 8: Simulating Model Training ===");
    
    LOG_INFO("Initializing transformer model...");
    LOG_DEBUG("Model config: d_model=512, num_heads=8, num_layers=6");
    
    for (int epoch = 1; epoch <= 3; ++epoch) {
        LOG_INFO("Epoch {} / {}", epoch, 3);
        
        for (int batch = 1; batch <= 5; ++batch) {
            double loss = 2.5 / (epoch * batch);
            LOG_DEBUG("Batch {} loss: {:.4f}", batch, loss);
            
            if (loss < 0.5) {
                LOG_WARN("Loss is getting very low: {:.4f}", loss);
            }
        }
        
        LOG_INFO("Epoch {} completed. Avg loss: {:.4f}", epoch, 0.5 / epoch);
    }
    
    LOG_INFO("Training completed successfully!");
}

int main() {
    // 初始化 Logger
    Logger::instance().init(
        "transformer",              // logger name
        true,                       // console output
        "logs/transformer.log",     // log file
        1024 * 1024 * 5,           // 5 MB per file
        3                          // keep 3 log files
    );
    
    // 设置日志级别
    Logger::instance().set_level(Logger::Level::TRACE);
    
    LOG_INFO("========================================");
    LOG_INFO("Transformer Logger Test Suite");
    LOG_INFO("Date: 2025-11-06 09:14:13 UTC");
    LOG_INFO("User: M4yGem1ni");
    LOG_INFO("========================================");
    
    test_basic_logging();
    LOG_INFO("");
    
    test_formatted_logging();
    LOG_INFO("");
    
    test_log_levels();
    LOG_INFO("");
    
    test_multithreading();
    LOG_INFO("");
    
    test_conditional_logging();
    LOG_INFO("");
    
    test_performance_timing();
    LOG_INFO("");
    
    test_custom_pattern();
    LOG_INFO("");
    
    simulate_model_training();
    LOG_INFO("");
    
    LOG_INFO("========================================");
    LOG_INFO("All logger tests completed successfully!");
    LOG_INFO("========================================");
    
    // 刷新日志
    Logger::instance().flush();
    
    return 0;
}