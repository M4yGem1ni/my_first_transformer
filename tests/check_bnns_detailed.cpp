// tests/check_bnns_detailed.cpp
#include <Accelerate/Accelerate.h>
#include <iostream>

int main() {
    std::cout << "=== BNNS API Check ===" << std::endl;
    std::cout << "macOS SDK version: " << __MAC_OS_X_VERSION_MAX_ALLOWED << std::endl;
    std::cout << "SDK: macOS " << __MAC_OS_X_VERSION_MAX_ALLOWED / 10000 << "." 
              << (__MAC_OS_X_VERSION_MAX_ALLOWED / 100) % 100 << std::endl;
    
    // 检查具体的 BNNS 类型和函数
    BNNSNDArrayDescriptor desc = {};
    
    std::cout << "\n=== BNNSNDArrayDescriptor fields ===" << std::endl;
    std::cout << "Size of BNNSNDArrayDescriptor: " << sizeof(desc) << " bytes" << std::endl;
    
    // 测试创建一个简单的数组描述符
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    desc.data = data;
    desc.size[0] = 2;
    desc.size[1] = 2;
    desc.stride[0] = 2;
    desc.stride[1] = 1;
    
    std::cout << "Successfully created BNNSNDArrayDescriptor" << std::endl;
    
    // 检查 BLAS 函数可用性
    std::cout << "\n=== BLAS Functions ===" << std::endl;
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0, 0, 0, 0};
    
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 2, 1.0f, A, 2, B, 2, 0.0f, C, 2);
    
    #pragma clang diagnostic pop
    
    std::cout << "BLAS sgemm result: [" << C[0] << ", " << C[1] << ", " 
              << C[2] << ", " << C[3] << "]" << std::endl;
    
    return 0;
}