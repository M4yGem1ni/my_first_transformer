// include/utils/progress_bar.hpp
#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <sstream>

namespace transformer {
namespace utils {

/**
 * @brief 简单而美观的进度条
 * 
 * 用法：
 * ProgressBar bar(100, "Training");
 * for (int i = 0; i < 100; ++i) {
 *     // do work
 *     bar.update(i + 1);
 * }
 * bar.finish();
 */
class ProgressBar {
public:
    /**
     * @param total 总步数
     * @param description 描述文字
     * @param bar_width 进度条宽度（字符数）
     */
    ProgressBar(size_t total, 
                const std::string& description = "", 
                size_t bar_width = 50)
        : total_(total)
        , current_(0)
        , description_(description)
        , bar_width_(bar_width)
        , start_time_(std::chrono::steady_clock::now())
        , last_update_time_(start_time_)
    {
        if (!description_.empty()) {
            description_ += ": ";
        }
    }
    
    /**
     * @brief 更新进度
     * @param current 当前完成的步数
     */
    void update(size_t current) {
        current_ = current;
        
        // 限制更新频率（避免闪烁）
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_update_time_).count();
        
        if (elapsed < 100 && current < total_) {
            return;  // 100ms 内不重复更新
        }
        
        last_update_time_ = now;
        display();
    }
    
    /**
     * @brief 完成进度条
     */
    void finish() {
        current_ = total_;
        display();
        std::cout << std::endl;
    }
    
    /**
     * @brief 设置后缀信息（如 loss, accuracy 等）
     */
    void set_postfix(const std::string& postfix) {
        postfix_ = postfix;
    }
    
private:
    size_t total_;
    size_t current_;
    std::string description_;
    std::string postfix_;
    size_t bar_width_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_update_time_;
    
    void display() {
        float progress = static_cast<float>(current_) / static_cast<float>(total_);
        size_t filled = static_cast<size_t>(progress * bar_width_);
        
        // 计算已用时间和预计剩余时间
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - start_time_).count();
        
        double avg_time_per_step = static_cast<double>(elapsed) / static_cast<double>(current_);
        size_t remaining_steps = total_ - current_;
        size_t eta = static_cast<size_t>(avg_time_per_step * remaining_steps);
        
        // 构建进度条字符串
        std::cout << "\r" << description_;
        std::cout << "[";
        
        for (size_t i = 0; i < bar_width_; ++i) {
            if (i < filled) {
                std::cout << "█";
            } else if (i == filled) {
                std::cout << "▌";
            } else {
                std::cout << " ";
            }
        }
        
        std::cout << "] ";
        std::cout << std::setw(3) << static_cast<int>(progress * 100) << "% ";
        std::cout << current_ << "/" << total_;
        
        // 显示时间信息
        std::cout << " [" << format_time(elapsed);
        if (current_ < total_) {
            std::cout << "<" << format_time(eta);
        }
        std::cout << "]";
        
        // 显示后缀信息
        if (!postfix_.empty()) {
            std::cout << " " << postfix_;
        }
        
        std::cout << std::flush;
    }
    
    /**
     * @brief 格式化时间显示
     */
    std::string format_time(size_t seconds) const {
        std::ostringstream oss;
        
        if (seconds < 60) {
            oss << seconds << "s";
        } else if (seconds < 3600) {
            size_t minutes = seconds / 60;
            size_t secs = seconds % 60;
            oss << minutes << "m" << std::setw(2) << std::setfill('0') << secs << "s";
        } else {
            size_t hours = seconds / 3600;
            size_t minutes = (seconds % 3600) / 60;
            oss << hours << "h" << std::setw(2) << std::setfill('0') << minutes << "m";
        }
        
        return oss.str();
    }
};

/**
 * @brief 多进度条管理器（用于嵌套循环）
 * 
 * 用法：
 * MultiProgressBar bars;
 * auto& epoch_bar = bars.add_bar(10, "Epoch");
 * for (int epoch = 0; epoch < 10; ++epoch) {
 *     auto& batch_bar = bars.add_bar(100, "Batch");
 *     for (int batch = 0; batch < 100; ++batch) {
 *         // do work
 *         batch_bar.update(batch + 1);
 *     }
 *     batch_bar.finish();
 *     epoch_bar.update(epoch + 1);
 * }
 * epoch_bar.finish();
 */
class MultiProgressBar {
public:
    ProgressBar& add_bar(size_t total, const std::string& description = "") {
        bars_.emplace_back(total, description);
        return bars_.back();
    }
    
    void clear() {
        bars_.clear();
    }
    
private:
    std::vector<ProgressBar> bars_;
};

} // namespace utils
} // namespace transformer