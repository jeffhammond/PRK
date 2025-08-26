#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

int main() {
    std::cout << "=== CUDA Version Information ===" << std::endl;
    
    // Get CUDA Runtime Version
    int runtimeVersion;
    cudaError_t runtimeResult = cudaRuntimeGetVersion(&runtimeVersion);
    if (runtimeResult == cudaSuccess) {
        int runtimeMajor = runtimeVersion / 1000;
        int runtimeMinor = (runtimeVersion % 1000) / 10;
        std::cout << "CUDA Runtime Version: " << runtimeMajor << "." << runtimeMinor 
                  << " (raw: " << runtimeVersion << ")" << std::endl;
    } else {
        std::cout << "Error getting CUDA Runtime version: " 
                  << cudaGetErrorString(runtimeResult) << std::endl;
    }
    
    // Get CUDA Driver Version
    int driverVersion;
    cudaError_t driverResult = cudaDriverGetVersion(&driverVersion);
    if (driverResult == cudaSuccess) {
        int driverMajor = driverVersion / 1000;
        int driverMinor = (driverVersion % 1000) / 10;
        std::cout << "CUDA Driver Version: " << driverMajor << "." << driverMinor 
                  << " (raw: " << driverVersion << ")" << std::endl;
    } else {
        std::cout << "Error getting CUDA Driver version: " 
                  << cudaGetErrorString(driverResult) << std::endl;
    }
    
    // Check compatibility
    if (driverResult == cudaSuccess && runtimeResult == cudaSuccess) {
        std::cout << "\nCompatibility Check:" << std::endl;
        if (driverVersion >= runtimeVersion) {
            std::cout << "✓ Driver and runtime versions are compatible" << std::endl;
        } else {
            std::cout << "✗ WARNING: Driver version is older than runtime!" << std::endl;
            std::cout << "  This may cause cudaErrorInsufficientDriver errors" << std::endl;
        }
    }
    
    // Get device information
    int deviceCount;
    cudaError_t deviceResult = cudaGetDeviceCount(&deviceCount);
    if (deviceResult == cudaSuccess) {
        std::cout << "\n=== Device Information ===" << std::endl;
        std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
        
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaError_t propResult = cudaGetDeviceProperties(&prop, i);
            if (propResult == cudaSuccess) {
                std::cout << "Device " << i << ": " << prop.name << std::endl;
                std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
                std::cout << "  Total Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
                std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
            }
        }
    } else {
        std::cout << "Error getting device count: " 
                  << cudaGetErrorString(deviceResult) << std::endl;
    }
    
    // Alternative method using CUDA Driver API directly
    std::cout << "\n=== Alternative Driver API Check ===" << std::endl;
    CUresult cuResult = cuInit(0);
    if (cuResult == CUDA_SUCCESS) {
        int cuDriverVersion;
        cuResult = cuDriverGetVersion(&cuDriverVersion);
        if (cuResult == CUDA_SUCCESS) {
            int cuDriverMajor = cuDriverVersion / 1000;
            int cuDriverMinor = (cuDriverVersion % 1000) / 10;
            std::cout << "CUDA Driver Version (Driver API): " << cuDriverMajor << "." << cuDriverMinor 
                      << " (raw: " << cuDriverVersion << ")" << std::endl;
        }
    } else {
        std::cout << "Failed to initialize CUDA Driver API" << std::endl;
    }
    
    return 0;
}
