#include "CUDAConfig.h"
#include <iostream>

CUDAManager& CUDAManager::getInstance() {
    static CUDAManager instance;
    return instance;
}

bool CUDAManager::initialize() {
    if (initialized) return true;

    try {
        
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }

        
        CUDA_CHECK(cudaSetDevice(deviceId));

        
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

        std::cout << "Using CUDA device: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

        initialized = true;
        return true;
    }
    catch (...) {
        std::cerr << "Failed to initialize CUDA" << std::endl;
        return false;
    }
}

void CUDAManager::cleanup() {
    if (initialized) {
        CUDA_CHECK(cudaDeviceReset());
        initialized = false;
    }
}

template<typename T>
T* CUDAManager::allocateGPU(size_t count) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
void CUDAManager::copyToGPU(T* gpu_ptr, const T* cpu_ptr, size_t count) {
    CUDA_CHECK(cudaMemcpy(gpu_ptr, cpu_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void CUDAManager::copyToCPU(T* cpu_ptr, const T* gpu_ptr, size_t count) {
    CUDA_CHECK(cudaMemcpy(cpu_ptr, gpu_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void CUDAManager::freeGPU(T* gpu_ptr) {
    if (gpu_ptr) {
        CUDA_CHECK(cudaFree(gpu_ptr));
    }
}


template float* CUDAManager::allocateGPU<float>(size_t);
template int* CUDAManager::allocateGPU<int>(size_t);
template CUDANodeData* CUDAManager::allocateGPU<CUDANodeData>(size_t);
template CUDASplitInfo* CUDAManager::allocateGPU<CUDASplitInfo>(size_t);

template void CUDAManager::copyToGPU<float>(float*, const float*, size_t);
template void CUDAManager::copyToGPU<int>(int*, const int*, size_t);
template void CUDAManager::copyToGPU<CUDANodeData>(CUDANodeData*, const CUDANodeData*, size_t);

template void CUDAManager::copyToCPU<float>(float*, const float*, size_t);
template void CUDAManager::copyToCPU<int>(int*, const int*, size_t);
template void CUDAManager::copyToCPU<CUDASplitInfo>(CUDASplitInfo*, const CUDASplitInfo*, size_t);

template void CUDAManager::freeGPU<float>(float*);
template void CUDAManager::freeGPU<int>(int*);
template void CUDAManager::freeGPU<CUDANodeData>(CUDANodeData*);
template void CUDAManager::freeGPU<CUDASplitInfo>(CUDASplitInfo*);