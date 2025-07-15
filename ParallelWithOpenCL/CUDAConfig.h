#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#define CUDA_BLOCK_SIZE 256
#define CUDA_MAX_THREADS_PER_BLOCK 1024
#define CUDA_MAX_BLOCKS 65535
#define MAX_NODES_PARALLEL 1024
#define MAX_SPLITS_PARALLEL 8192


#define PARALLELIZE_ON_TREES_CUDA
//
//#define PARALLELIZE_ON_NODES_CUDA

//#define PARALLELIZE_ON_SPLITS_CUDA


#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)


struct CUDANodeData {
    int feature_index;
    float threshold;
    int class_label;
    int subtree_max_depth;
    int min_samples_split;
    int dataset_offset;
    int dataset_len;
    int is_leaf;
    int process_flag;  
};

struct CUDASplitInfo {
    int feature_index;
    float split_point;
    float gain_ratio;
    int valid;
};


class CUDAManager {
public:
    static CUDAManager& getInstance();

    bool initialize();
    void cleanup();

    
    template<typename T>
    T* allocateGPU(size_t count);

    template<typename T>
    void copyToGPU(T* gpu_ptr, const T* cpu_ptr, size_t count);

    template<typename T>
    void copyToCPU(T* cpu_ptr, const T* gpu_ptr, size_t count);

    template<typename T>
    void freeGPU(T* gpu_ptr);

private:
    CUDAManager() = default;
    bool initialized = false;
    int deviceId = 0;
};
