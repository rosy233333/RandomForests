#include "CUDAConfig.h"
#include "Dataset.h"
#include <device_launch_parameters.h>
#include <curand_kernel.h>

// CUDA kernel: parallel computation of gain rate at split points
__global__ void computeGainRatiosKernel(
    const float* features,      // [instance_count * feature_count]
    const int* labels,          // [instance_count]
    const float* split_points,  // [split_count]
    float* gain_ratios,         // [split_count] - output
    int instance_count,
    int feature_index,
    int split_count,
    int class_count
) {
    int split_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (split_id >= split_count) return;
    
    float threshold = split_points[split_id];
    
    // Optimizing tag counting with shared memory
    __shared__ int shared_left_count[11];   // Assuming a maximum of 11 categories
    __shared__ int shared_right_count[11];
    
    // Initializing shared memory
    if (threadIdx.x < class_count) {
        shared_left_count[threadIdx.x] = 0;
        shared_right_count[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Counting the distribution of labels in the left and right subsets
    for (int i = 0; i < instance_count; i++) {
        float feature_value = features[i * 11 + feature_index];  // Assuming 11 features
        int label = labels[i];
        
        if (feature_value <= threshold) {
            atomicAdd(&shared_left_count[label], 1);
        } else {
            atomicAdd(&shared_right_count[label], 1);
        }
    }
    __syncthreads();
    
    // Calculate total
    int total_left = 0, total_right = 0;
    for (int c = 0; c < class_count; c++) {
        total_left += shared_left_count[c];
        total_right += shared_right_count[c];
    }
    
    if (total_left == 0 || total_right == 0) {
        gain_ratios[split_id] = 0.0f;
        return;
    }
    
    // Calculate entropy
    float entropy_left = 0.0f, entropy_right = 0.0f;
    
    for (int c = 0; c < class_count; c++) {
        if (shared_left_count[c] > 0) {
            float p = (float)shared_left_count[c] / total_left;
            entropy_left -= p * log2f(p);
        }
        if (shared_right_count[c] > 0) {
            float p = (float)shared_right_count[c] / total_right;
            entropy_right -= p * log2f(p);
        }
    }
    
    // Calculate the total entropy
    int total_count[11] = {0};
    for (int c = 0; c < class_count; c++) {
        total_count[c] = shared_left_count[c] + shared_right_count[c];
    }
    
    float entropy_total = 0.0f;
    int total = total_left + total_right;
    for (int c = 0; c < class_count; c++) {
        if (total_count[c] > 0) {
            float p = (float)total_count[c] / total;
            entropy_total -= p * log2f(p);
        }
    }
    
    // Calculate information gain
    float weighted_entropy = (float)total_left / total * entropy_left + 
                            (float)total_right / total * entropy_right;
    float gain = entropy_total - weighted_entropy;
    
    // Calculate split information
    float p_left = (float)total_left / total;
    float p_right = (float)total_right / total;
    float split_info = -p_left * log2f(p_left) - p_right * log2f(p_right);
    
    // Calculating gain ratio
    gain_ratios[split_id] = (split_info > 0) ? gain / split_info : 0.0f;
}

// CUDA kernel: parallel processing of decision tree nodes
__global__ void processNodesKernel(
    CUDANodeData* nodes,           // Node data
    const float* all_features,     // All feature data
    const int* all_labels,         // All label data
    CUDASplitInfo* best_splits,    // Output optimal split information
    curandState* rand_states,      // random number state
    int node_count,
    int total_instances,
    int feature_count,
    int class_count
) {
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_id >= node_count) return;
    
    CUDANodeData& node = nodes[node_id];
    
    // Check if this node needs to be processed
    if (node.process_flag == 0) return;
    
    // Check if it should be a leaf node
    if (node.dataset_len < node.min_samples_split || node.subtree_max_depth <= 0) {
        node.is_leaf = 1;
        
        // Calculate the most common class labels
        int label_count[11] = {0};  // Assuming a maximum of 11 categories
        for (int i = 0; i < node.dataset_len; i++) {
            int instance_idx = node.dataset_offset + i;
            label_count[all_labels[instance_idx]]++;
        }
        
        int max_count = 0;
        int most_common_label = 0;
        for (int c = 0; c < class_count; c++) {
            if (label_count[c] > max_count) {
                max_count = label_count[c];
                most_common_label = c;
            }
        }
        node.class_label = most_common_label;
        return;
    }
    
    // Find the optimal split point for this node
    float best_gain = 0.0f;
    int best_feature = -1;
    float best_threshold = 0.0f;
    
    // Simplified feature selection and split point calculation
    curandState local_state = rand_states[node_id];
    
    for (int feature_idx = 0; feature_idx < feature_count; feature_idx++) {
        // Randomly selecting a number of split points for evaluation
        for (int split_attempt = 0; split_attempt < min(node.dataset_len, 32); split_attempt++) {
            int rand_instance = node.dataset_offset + (curand(&local_state) % node.dataset_len);
            float threshold = all_features[rand_instance * feature_count + feature_idx];
            
            // Calculate the gain rate at this split point
            // Simplified version of gain rate calculation
            int left_count = 0, right_count = 0;
            for (int i = 0; i < node.dataset_len; i++) {
                int instance_idx = node.dataset_offset + i;
                if (all_features[instance_idx * feature_count + feature_idx] <= threshold) {
                    left_count++;
                } else {
                    right_count++;
                }
            }
            
            if (left_count > 0 && right_count > 0) {
                // Simplified gain calculation (the full message gain should actually be calculated)
                float gain = (float)(left_count * right_count) / (node.dataset_len * node.dataset_len);
                
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }
    }
    
    // Update random number status
    rand_states[node_id] = local_state;
    
    // Storing optimal split information
    best_splits[node_id].feature_index = best_feature;
    best_splits[node_id].split_point = best_threshold;
    best_splits[node_id].gain_ratio = best_gain;
    best_splits[node_id].valid = (best_gain > 0) ? 1 : 0;
    
    if (best_gain == 0) {
        node.is_leaf = 1;
        // Setting leaf node labels logic as above
    }
}

// Initialize kernel for random number state
__global__ void initRandomStates(curandState* states, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &states[id]);
    }
}

// C++ wrappers for CUDA utility functions
extern "C" {
    void launchComputeGainRatios(
        const float* d_features,
        const int* d_labels,
        const float* d_split_points,
        float* d_gain_ratios,
        int instance_count,
        int feature_index,
        int split_count,
        int class_count
    ) {
        dim3 blockSize(CUDA_BLOCK_SIZE);
        dim3 gridSize((split_count + blockSize.x - 1) / blockSize.x);
        
        computeGainRatiosKernel<<<gridSize, blockSize>>>(
            d_features, d_labels, d_split_points, d_gain_ratios,
            instance_count, feature_index, split_count, class_count
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void launchProcessNodes(
        CUDANodeData* d_nodes,
        const float* d_features,
        const int* d_labels,
        CUDASplitInfo* d_best_splits,
        curandState* d_rand_states,
        int node_count,
        int total_instances,
        int feature_count,
        int class_count
    ) {
        dim3 blockSize(CUDA_BLOCK_SIZE);
        dim3 gridSize((node_count + blockSize.x - 1) / blockSize.x);
        
        processNodesKernel<<<gridSize, blockSize>>>(
            d_nodes, d_features, d_labels, d_best_splits, d_rand_states,
            node_count, total_instances, feature_count, class_count
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void initializeCudaRandomStates(curandState* d_states, int count) {
        dim3 blockSize(CUDA_BLOCK_SIZE);
        dim3 gridSize((count + blockSize.x - 1) / blockSize.x);
        
        initRandomStates<<<gridSize, blockSize>>>(d_states, time(NULL), count);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}