// include/utils/metrics.hpp
#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <map>

namespace transformer {
namespace utils {

/**
 * @brief 训练指标收集器
 */
class MetricsCollector {
public:
    /**
     * @brief 添加或更新指标
     */
    void update(const std::string& name, double value) {
        metrics_[name] = value;
    }
    
    /**
     * @brief 获取指标值
     */
    double get(const std::string& name) const {
        auto it = metrics_.find(name);
        if (it != metrics_.end()) {
            return it->second;
        }
        return 0.0;
    }
    
    /**
     * @brief 格式化为字符串（用于进度条后缀）
     */
    std::string format() const {
        std::ostringstream oss;
        bool first = true;
        
        for (const auto& [name, value] : metrics_) {
            if (!first) {
                oss << ", ";
            }
            oss << name << "=";
            oss << std::fixed << std::setprecision(4) << value;
            first = false;
        }
        
        return oss.str();
    }
    
    /**
     * @brief 清空指标
     */
    void clear() {
        metrics_.clear();
    }
    
private:
    std::map<std::string, double> metrics_;
};

} // namespace utils
} // namespace transformer