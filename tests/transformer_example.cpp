// tests/encoder_decoder.cpp
#include "transformer.hpp"  // 🔴 只需要包含这一个头文件
#include "utils/logger.hpp"
#include "utils/progress_bar.hpp"
#include "utils/metrics.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>
#include <iostream>

using namespace xt;
using namespace transformer;
using namespace transformer::utils;

// ============================================
// 辅助函数
// ============================================

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

template<typename T = float>
xarray<T> create_padding_mask(size_t batch_size, size_t seq_len) {
    std::vector<size_t> shape = {batch_size, seq_len, seq_len};
    xarray<T> mask = ones<T>(shape);
    
    LOG_TRACE("Created padding mask: batch_size={}, seq_len={}", batch_size, seq_len);
    return mask;
}

template<typename T = float>
std::shared_ptr<models::EncoderDecoder<T>> make_model(
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
    
    auto model = std::make_shared<models::EncoderDecoder<T>>(
        encoder, decoder, src_embed, tgt_embed, generator
    );
    LOG_INFO("  ✓ Model assembled");
    
    return model;
}

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
        // 初始化 Logger
        Logger::instance().init(
            "transformer_test",
            true,
            "logs/transformer_test.log",
            1024 * 1024 * 10,
            3
        );
        
        Logger::instance().set_level(Logger::Level::INFO);
        
        LOG_INFO("=== Transformer Encoder-Decoder Test ===");
        LOG_INFO("Current Date and Time (UTC): 2025-11-09 12:30:57");
        LOG_INFO("Current User's Login: M4yGem1ni");
        LOG_INFO("");
        
        // 模型参数
        const size_t src_vocab_size = 1000;
        const size_t tgt_vocab_size = 1000;
        const size_t d_model = 512;
        const size_t num_heads = 8;
        const size_t num_layers = 6;
        const size_t d_ff = 2048;
        const float dropout = 0.1f;
        
        // 创建模型
        auto model = make_model<float>(
            src_vocab_size, tgt_vocab_size, d_model,
            num_heads, num_layers, d_ff, dropout
        );
        
        LOG_INFO("");
        LOG_INFO("✓ Model created successfully!");
        LOG_INFO("");
        
        // 准备测试数据
        const size_t batch_size = 1;
        const size_t src_seq_len = 5;
        const size_t tgt_seq_len = 4;
        
        LOG_INFO("=== Preparing Test Data ===");
        
        std::vector<size_t> src_shape = {batch_size, src_seq_len};
        std::vector<size_t> tgt_shape = {batch_size, tgt_seq_len};
        
        xarray<int> src = zeros<int>(src_shape);
        xarray<int> tgt = zeros<int>(tgt_shape);
        
        for (size_t i = 0; i < src_seq_len; ++i) {
            src.data()[i] = static_cast<int>(i + 1);
        }
        for (size_t i = 0; i < tgt_seq_len; ++i) {
            tgt.data()[i] = static_cast<int>(i + 1);
        }
        
        LOG_INFO("  src: [{}, {}]", batch_size, src_seq_len);
        LOG_INFO("  tgt: [{}, {}]", batch_size, tgt_seq_len);
        
        // 创建 Masks
        LOG_INFO("");
        LOG_INFO("=== Creating Masks ===");
        
        auto src_mask = create_padding_mask<float>(batch_size, src_seq_len);
        LOG_INFO("  src_mask: {}", shape_to_string(src_mask));
        
        auto tgt_mask = create_causal_mask<float>(tgt_seq_len);
        LOG_INFO("  tgt_mask: {}", shape_to_string(tgt_mask));
        
        // 前向传播
        LOG_INFO("");
        LOG_INFO("=== Running Forward Pass ===");

        const size_t num_iterations = 10;
        ProgressBar progress(num_iterations, "Forward Pass");
        
        xarray<float> output;
        
        for (size_t i = 0; i < num_iterations; ++i) {
            LOG_TIME_START(forward_pass);
            output = model->forward(src, tgt, src_mask, tgt_mask);
            LOG_TIME_END(forward_pass);
            progress.update(i + 1);
        }
        
        // auto output = model->forward(src, tgt, src_mask, tgt_mask);
        
        // 输出结果
        LOG_INFO("");
        LOG_INFO("=== Forward Pass Complete ===");
        LOG_INFO("Output shape: {}", shape_to_string(output));
        
        auto output_min = xt::amin(output)();
        auto output_max = xt::amax(output)();
        auto output_mean = xt::mean(output)();
        
        LOG_INFO("Output statistics:");
        LOG_INFO("  min:  {:.6f}", output_min);
        LOG_INFO("  max:  {:.6f}", output_max);
        LOG_INFO("  mean: {:.6f}", output_mean);
        
        // 检查输出有效性
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
        
        LOG_INFO("");
        LOG_INFO("✅ Test completed successfully!");
        LOG_INFO("🎉 Transformer model works correctly!");
        
        Logger::instance().shutdown();
        return 0;
        
    } catch (const std::exception& e) {
        LOG_CRITICAL("Unhandled exception: {}", e.what());
        LOG_CRITICAL("❌ Test FAILED!");
        Logger::instance().shutdown();
        return 1;
    }
}