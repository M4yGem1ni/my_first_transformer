// include/utils/logger.hpp
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>  // 添加这一行

namespace transformer {

/**
 * @brief Transformer 项目的日志封装类
 * 基于 spdlog 库，提供简化的接口
 */
class Logger {
public:
    // 日志级别枚举
    enum class Level {
        TRACE = 0,
        DEBUG = 1,
        INFO = 2,
        WARN = 3,
        ERROR = 4,
        CRITICAL = 5,
        OFF = 6
    };
    
    /**
     * @brief 获取全局 Logger 实例（单例模式）
     */
    static Logger& instance() {
        static Logger logger;
        return logger;
    }
    
    // 禁止拷贝和赋值
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    /**
     * @brief 初始化日志系统
     * @param log_name 日志器名称
     * @param console 是否输出到控制台
     * @param file_path 日志文件路径（可选）
     * @param max_file_size 单个日志文件最大大小（默认 10MB）
     * @param max_files 最多保留的日志文件数（默认 3 个）
     */
    void init(
        const std::string& log_name = "transformer",
        bool console = true,
        const std::string& file_path = "",
        size_t max_file_size = 1024 * 1024 * 10,  // 10 MB
        size_t max_files = 3
    ) {
        try {
            std::vector<spdlog::sink_ptr> sinks;
            
            // 控制台输出（带颜色）
            if (console) {
                auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                console_sink->set_level(spdlog::level::trace);
                console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
                sinks.push_back(console_sink);
            }
            
            // 文件输出（轮转日志）
            if (!file_path.empty()) {
                auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                    file_path, max_file_size, max_files);
                file_sink->set_level(spdlog::level::trace);
                file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");
                sinks.push_back(file_sink);
            }
            
            // 创建 logger
            logger_ = std::make_shared<spdlog::logger>(log_name, sinks.begin(), sinks.end());
            logger_->set_level(spdlog::level::info);
            logger_->flush_on(spdlog::level::warn);  // WARN 及以上级别自动刷新
            
            // 设置为默认 logger
            spdlog::set_default_logger(logger_);
            
            initialized_ = true;
            
        } catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        }
    }
    
    /**
     * @brief 设置日志级别
     */
    void set_level(Level level) {
        if (!initialized_) {
            init();
        }
        logger_->set_level(to_spdlog_level(level));
    }
    
    /**
     * @brief 获取当前日志级别
     */
    Level get_level() const {
        if (!initialized_) {
            return Level::INFO;
        }
        return from_spdlog_level(logger_->level());
    }
    
    /**
     * @brief 设置日志输出格式
     * @param pattern 格式字符串
     * 
     * 格式说明：
     * %v - 消息内容
     * %l - 日志级别
     * %L - 日志级别（简写）
     * %t - 线程 ID
     * %P - 进程 ID
     * %n - Logger 名称
     * %Y-%m-%d %H:%M:%S.%e - 时间戳
     * %s - 源文件名
     * %# - 行号
     * %! - 函数名
     */
    void set_pattern(const std::string& pattern) {
        if (!initialized_) {
            init();
        }
        logger_->set_pattern(pattern);
    }
    
    /**
     * @brief 立即刷新日志缓冲区
     */
    void flush() {
        if (initialized_) {
            logger_->flush();
        }
    }
    
    /**
     * @brief 获取底层的 spdlog logger（高级用法）
     */
    std::shared_ptr<spdlog::logger> get_logger() {
        if (!initialized_) {
            init();
        }
        return logger_;
    }
    
    /**
     * @brief 检查是否已初始化
     */
    bool is_initialized() const {
        return initialized_;
    }

    /**
     * @brief 关闭日志系统，释放资源
     */
    void shutdown() {
        if (initialized_) {
            logger_->flush();
            spdlog::shutdown();
            initialized_ = false;
        }
    }
    
private:
    Logger() : initialized_(false) {}
    
    ~Logger() {
        if (initialized_) {
            logger_->flush();
        }
    }
    
    // 转换日志级别
    spdlog::level::level_enum to_spdlog_level(Level level) const {
        switch (level) {
            case Level::TRACE:    return spdlog::level::trace;
            case Level::DEBUG:    return spdlog::level::debug;
            case Level::INFO:     return spdlog::level::info;
            case Level::WARN:     return spdlog::level::warn;
            case Level::ERROR:    return spdlog::level::err;
            case Level::CRITICAL: return spdlog::level::critical;
            case Level::OFF:      return spdlog::level::off;
            default:              return spdlog::level::info;
        }
    }
    
    Level from_spdlog_level(spdlog::level::level_enum level) const {
        switch (level) {
            case spdlog::level::trace:    return Level::TRACE;
            case spdlog::level::debug:    return Level::DEBUG;
            case spdlog::level::info:     return Level::INFO;
            case spdlog::level::warn:     return Level::WARN;
            case spdlog::level::err:      return Level::ERROR;
            case spdlog::level::critical: return Level::CRITICAL;
            case spdlog::level::off:      return Level::OFF;
            default:                      return Level::INFO;
        }
    }
    
    std::shared_ptr<spdlog::logger> logger_;
    bool initialized_;
};

} // namespace transformer

// ============================================
// 便捷的日志宏定义
// ============================================

// 确保 Logger 已初始化
#define ENSURE_LOGGER_INIT() \
    do { \
        if (!transformer::Logger::instance().is_initialized()) { \
            transformer::Logger::instance().init(); \
        } \
    } while(0)

// 基础日志宏
#define LOG_TRACE(...) \
    ENSURE_LOGGER_INIT(); \
    SPDLOG_TRACE(__VA_ARGS__)

#define LOG_DEBUG(...) \
    ENSURE_LOGGER_INIT(); \
    SPDLOG_DEBUG(__VA_ARGS__)

#define LOG_INFO(...) \
    ENSURE_LOGGER_INIT(); \
    SPDLOG_INFO(__VA_ARGS__)

#define LOG_WARN(...) \
    ENSURE_LOGGER_INIT(); \
    SPDLOG_WARN(__VA_ARGS__)

#define LOG_ERROR(...) \
    ENSURE_LOGGER_INIT(); \
    SPDLOG_ERROR(__VA_ARGS__)

#define LOG_CRITICAL(...) \
    ENSURE_LOGGER_INIT(); \
    SPDLOG_CRITICAL(__VA_ARGS__)

// 条件日志宏
#define LOG_IF(condition, level, ...) \
    do { \
        if (condition) { \
            LOG_##level(__VA_ARGS__); \
        } \
    } while(0)

// 断言宏
#define LOG_ASSERT(condition, ...) \
    do { \
        if (!(condition)) { \
            ENSURE_LOGGER_INIT(); \
            SPDLOG_CRITICAL("Assertion failed: {} - {}", #condition, fmt::format(__VA_ARGS__)); \
            std::abort(); \
        } \
    } while(0)

// 性能计时宏（简化版）
#define LOG_SCOPE_DURATION(name) \
    auto __log_start_##name = std::chrono::high_resolution_clock::now(); \
    auto __log_guard_##name = [&, start = __log_start_##name](const char* n) { \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); \
        ENSURE_LOGGER_INIT(); \
        SPDLOG_INFO("{} took {} ms", n, duration.count()); \
    }; \
    std::shared_ptr<void> __guard_ptr_##name(nullptr, [&](void*){ __log_guard_##name(#name); })

// 简单的计时宏
#define LOG_TIME_START(name) \
    auto __timer_##name = std::chrono::high_resolution_clock::now()

#define LOG_TIME_END(name) \
    do { \
        auto __end_##name = std::chrono::high_resolution_clock::now(); \
        auto __duration_##name = std::chrono::duration_cast<std::chrono::milliseconds>( \
            __end_##name - __timer_##name); \
        LOG_INFO("{} took {} ms", #name, __duration_##name.count()); \
    } while(0)