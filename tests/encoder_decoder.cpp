// tests/encoder_decoder.cpp
#include "core/encoder_decoder.hpp"
#include "utils/logger.hpp"  // 🔴 添加 logger
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>
#include <iostream>

using namespace xt;
using namespace transformer;

// ============================================
// 辅助函数
// ============================================

// 创建因果mask（下三角矩阵）
template<typename T = float>
xarray<T> create_causal_mask(size_t size) {
    std::vector<size_t> shape = {1, size, size};
    xarray<T> mask = ones<T>(shape);
    
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i + 1; j < size; ++j) {
            mask.data()[i * size + j] = static_cast<T>(0);
        }
    }
    
    LOG_TRACE("Created causal mask: size={}", size);
    return mask;
}

// 创建 padding mask
template<typename T = float>
xarray<T> create_padding_mask(size_t batch_size, size_t seq_len) {
    std::vector<size_t> shape = {batch_size, seq_len, seq_len};
    xarray<T> mask = ones<T>(shape);
    
    LOG_TRACE("Created padding mask: batch_size={}, seq_len={}", batch_size, seq_len);
    return mask;
}

// ============================================
// 工厂函数：创建完整的 Transformer 模型
// ============================================
template<typename T = float>
std::shared_ptr<EncoderDecoder<T>> make_model(
    size_t src_vocab,
    size_t tgt_vocab,
    size_t d_model = 512,
    size_t num_heads = 8,
    size_t num_layers = 6,
    size_t d_ff = 2048,
    T dropout = 0.1f)
{
    LOG_INFO("Creating Transformer model components...");
    
    auto encoder = std::make_shared<Encoder>(d_model, num_heads, d_ff, num_layers, dropout);
    LOG_INFO("  ✓ Encoder created");
    
    auto decoder = std::make_shared<Decoder>(d_model, num_heads, d_ff, num_layers, dropout);
    LOG_INFO("  ✓ Decoder created");
    
    auto src_embed = std::make_shared<Embeddings>(src_vocab, d_model);
    LOG_INFO("  ✓ Source embeddings created");
    
    auto tgt_embed = std::make_shared<Embeddings>(tgt_vocab, d_model);
    LOG_INFO("  ✓ Target embeddings created");
    
    auto generator = std::make_shared<Generator>(d_model, tgt_vocab);
    LOG_INFO("  ✓ Generator created");
    
    auto model = std::make_shared<EncoderDecoder<T>>(
        encoder, decoder, src_embed, tgt_embed, generator
    );
    LOG_INFO("  ✓ Model assembled");
    
    return model;
}

// ============================================
// 辅助函数：打印数组信息
// ============================================
template<typename T>
std::string shape_to_string(const xarray<T>& arr) {
    std::string result = "[";
    for (size_t i = 0; i < arr.dimension(); ++i) {
        result += std::to_string(arr.shape()[i]);
        if (i < arr.dimension() - 1) result += ", ";
    }
    result += "]";
    return result;
}

// ============================================
// 主测试函数
// ============================================
int main() {
    try {
        // 🔴 初始化 Logger
        Logger::instance().init(
            "transformer_test",       // logger 名称
            true,                      // 控制台输出
            "logs/transformer_test.log",  // 日志文件
            1024 * 1024 * 10,         // 10MB
            3                          // 保留3个文件
        );
        
        // 🔴 设置日志级别
        // Logger::instance().set_level(Logger::Level::TRACE);  // 超详细（调试用）
        // Logger::instance().set_level(Logger::Level::DEBUG);  // 详细
        Logger::instance().set_level(Logger::Level::INFO);    // 正常（推荐）
        // Logger::instance().set_level(Logger::Level::WARN);   // 只显示警告
        
        LOG_INFO("=== Transformer Encoder-Decoder Test ===");
        LOG_INFO("Current Date and Time (UTC): 2025-01-09 11:55:42");
        LOG_INFO("Current User's Login: M4yGem1ni");
        LOG_INFO("");  // 空行
        
        // ============================================
        // 模型参数
        // ============================================
        const size_t src_vocab_size = 1000;
        const size_t tgt_vocab_size = 1000;
        const size_t d_model = 512;
        const size_t num_heads = 8;
        const size_t num_layers = 6;
        const size_t d_ff = 2048;
        const float dropout = 0.1f;
        
        LOG_DEBUG("Model hyperparameters:");
        LOG_DEBUG("  src_vocab_size: {}", src_vocab_size);
        LOG_DEBUG("  tgt_vocab_size: {}", tgt_vocab_size);
        LOG_DEBUG("  d_model: {}", d_model);
        LOG_DEBUG("  num_heads: {}", num_heads);
        LOG_DEBUG("  num_layers: {}", num_layers);
        LOG_DEBUG("  d_ff: {}", d_ff);
        LOG_DEBUG("  dropout: {:.2f}", dropout);
        
        // ============================================
        // 创建模型
        // ============================================
        auto model = make_model<float>(
            src_vocab_size,
            tgt_vocab_size,
            d_model,
            num_heads,
            num_layers,
            d_ff,
            dropout
        );
        
        LOG_INFO("");
        LOG_INFO("✓ Model created successfully!");
        LOG_INFO("");
        
        // ============================================
        // 准备测试数据
        // ============================================
        const size_t batch_size = 1;
        const size_t src_seq_len = 5;
        const size_t tgt_seq_len = 4;
        
        LOG_INFO("=== Preparing Test Data ===");
        
        std::vector<size_t> src_shape = {batch_size, src_seq_len};
        std::vector<size_t> tgt_shape = {batch_size, tgt_seq_len};
        
        xarray<int> src = zeros<int>(src_shape);
        xarray<int> tgt = zeros<int>(tgt_shape);
        
        // 填充测试数据
        for (size_t i = 0; i < src_seq_len; ++i) {
            src.data()[i] = static_cast<int>(i + 1);
        }
        for (size_t i = 0; i < tgt_seq_len; ++i) {
            tgt.data()[i] = static_cast<int>(i + 1);
        }
        
        LOG_INFO("  src: [{}, {}]", batch_size, src_seq_len);
        LOG_INFO("  tgt: [{}, {}]", batch_size, tgt_seq_len);
        
        // 只在 DEBUG 级别打印具体的 token
        if (Logger::instance().get_level() <= Logger::Level::DEBUG) {
            std::stringstream ss_src, ss_tgt;
            ss_src << src;
            ss_tgt << tgt;
            LOG_DEBUG("  src tokens: {}", ss_src.str());
            LOG_DEBUG("  tgt tokens: {}", ss_tgt.str());
        }
        
        // ============================================
        // 创建 Masks
        // ============================================
        LOG_INFO("");
        LOG_INFO("=== Creating Masks ===");
        
        auto src_mask = create_padding_mask<float>(batch_size, src_seq_len);
        LOG_INFO("  src_mask: {}", shape_to_string(src_mask));
        
        auto tgt_mask = create_causal_mask<float>(tgt_seq_len);
        LOG_INFO("  tgt_mask: {}", shape_to_string(tgt_mask));
        
        // 只在 DEBUG 级别打印 mask 内容
        if (Logger::instance().get_level() <= Logger::Level::DEBUG) {
            std::stringstream ss_mask;
            ss_mask << tgt_mask;
            LOG_DEBUG("  tgt_mask content (causal):\n{}", ss_mask.str());
        }
        
        // ============================================
        // 前向传播
        // ============================================
        LOG_INFO("");
        LOG_INFO("=== Running Forward Pass ===");
        
        // 🔴 使用日志计时宏
        LOG_TIME_START(forward_pass);
        
        auto output = model->forward(src, tgt, src_mask, tgt_mask);
        
        LOG_TIME_END(forward_pass);
        
        // ============================================
        // 输出结果
        // ============================================
        LOG_INFO("");
        LOG_INFO("=== Forward Pass Complete ===");
        LOG_INFO("Output shape: {}", shape_to_string(output));
        
        // 计算统计信息
        auto output_min = xt::amin(output)();
        auto output_max = xt::amax(output)();
        auto output_mean = xt::mean(output)();
        
        LOG_INFO("Output statistics:");
        LOG_INFO("  min:  {:.6f}", output_min);
        LOG_INFO("  max:  {:.6f}", output_max);
        LOG_INFO("  mean: {:.6f}", output_mean);
        
        // 检查输出是否有效
        if (std::isnan(output_min) || std::isnan(output_max) || std::isnan(output_mean)) {
            LOG_ERROR("Output contains NaN values!");
            LOG_ERROR("❌ Test FAILED!");
            Logger::instance().shutdown();
            return 1;
        }
        
        if (std::isinf(output_min) || std::isinf(output_max)) {
            LOG_ERROR("Output contains Inf values!");
            LOG_ERROR("❌ Test FAILED!");
            Logger::instance().shutdown();
            return 1;
        }
        
        // ============================================
        // 测试成功
        // ============================================
        LOG_INFO("");
        LOG_INFO("✅ Test completed successfully!");
        LOG_INFO("🎉 Transformer model works correctly!");
        
        // 关闭日志系统
        Logger::instance().shutdown();
        
        return 0;
        
    } catch (const std::exception& e) {
        LOG_CRITICAL("Unhandled exception: {}", e.what());
        LOG_CRITICAL("❌ Test FAILED!");
        Logger::instance().shutdown();
        return 1;
    }
}