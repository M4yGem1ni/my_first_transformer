// check_bnns.cpp
#include <Accelerate/Accelerate.h>
#include <iostream>

int main() {
    std::cout << "macOS SDK version: " << __MAC_OS_X_VERSION_MAX_ALLOWED << std::endl;
    
    #ifdef BNNS_API_AVAILABLE
    std::cout << "BNNS API is available" << std::endl;
    #endif
    
    return 0;
}