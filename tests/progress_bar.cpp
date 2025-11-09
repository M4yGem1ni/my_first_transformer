// tests/test_progress_bar.cpp
#include "utils/progress_bar.hpp"
#include "utils/metrics.hpp"
#include "utils/logger.hpp"
#include <thread>
#include <random>

using namespace transformer::utils;
using namespace transformer;

int main() {
    Logger::instance().init("progress_test", true);
    Logger::instance().set_level(Logger::Level::INFO);
    
    LOG_INFO("=== Training Simulation ===\n");
    
    const size_t num_epochs = 5;
    const size_t num_batches = 100;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> loss_dist(0.5, 2.0);
    
    // 外层循环：Epoch
    ProgressBar epoch_bar(num_epochs, "Epoch");
    
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        // 内层循环：Batch
        ProgressBar batch_bar(num_batches, "  Batch");
        MetricsCollector metrics;
        
        double epoch_loss = 0.0;
        
        for (size_t batch = 0; batch < num_batches; ++batch) {
            // 模拟训练
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // 计算损失
            double loss = loss_dist(gen) / (epoch + 1);
            epoch_loss += loss;
            
            // 更新指标
            metrics.update("loss", loss);
            metrics.update("lr", 0.001 * std::exp(-0.1 * epoch));
            
            // 更新进度条
            batch_bar.set_postfix(metrics.format());
            batch_bar.update(batch + 1);
        }
        
        batch_bar.finish();
        
        // 计算平均损失
        double avg_loss = epoch_loss / num_batches;
        
        // 更新 epoch 进度条
        MetricsCollector epoch_metrics;
        epoch_metrics.update("avg_loss", avg_loss);
        epoch_bar.set_postfix(epoch_metrics.format());
        epoch_bar.update(epoch + 1);
        
        LOG_INFO("Epoch {} completed - Avg Loss: {:.4f}\n", epoch + 1, avg_loss);
    }
    
    epoch_bar.finish();
    
    LOG_INFO("\n🎉 Training completed!");
    
    Logger::instance().shutdown();
    return 0;
}