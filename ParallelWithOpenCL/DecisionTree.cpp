#include "DecisionTree.h"
#include "CUDAConfig.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>

extern "C" {
    void launchComputeGainRatios(
        const float* d_features, const int* d_labels,
        const float* d_split_points, float* d_gain_ratios,
        int instance_count, int feature_index,
        int split_count, int class_count
    );

    void launchProcessNodes(
        CUDANodeData* d_nodes, const float* d_features,
        const int* d_labels, CUDASplitInfo* d_best_splits,
        curandState* d_rand_states, int node_count,
        int total_instances, int feature_count, int class_count
    );
    


    void initializeCudaRandomStates(curandState* d_states, int count);
    
}

DecisionTree::~DecisionTree()
{
    if (this->root != nullptr) {
        delete this->root;
    }
}

int cmpfunc(const void* a, const void* b)
{
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}


std::vector<float> computeGainRatiosCUDA(
    const std::vector<float>& features,
    const std::vector<int>& labels,
    const std::vector<float>& split_points,
    int feature_index
) {
#ifdef PARALLELIZE_ON_SPLITS_CUDA
    CUDAManager& cudaManager = CUDAManager::getInstance();
    if (!cudaManager.initialize()) {

        std::vector<float> results(split_points.size(), 0.0f);
        return results;
    }

    try {

        float* d_features = cudaManager.allocateGPU<float>(features.size());
        int* d_labels = cudaManager.allocateGPU<int>(labels.size());
        float* d_split_points = cudaManager.allocateGPU<float>(split_points.size());
        float* d_gain_ratios = cudaManager.allocateGPU<float>(split_points.size());


        cudaManager.copyToGPU(d_features, features.data(), features.size());
        cudaManager.copyToGPU(d_labels, labels.data(), labels.size());
        cudaManager.copyToGPU(d_split_points, split_points.data(), split_points.size());


        launchComputeGainRatios(
            d_features, d_labels, d_split_points, d_gain_ratios,
            static_cast<int>(labels.size()), feature_index,
            static_cast<int>(split_points.size()), CLASS_NUM
        );


        std::vector<float> results(split_points.size());
        cudaManager.copyToCPU(results.data(), d_gain_ratios, split_points.size());


        cudaManager.freeGPU(d_features);
        cudaManager.freeGPU(d_labels);
        cudaManager.freeGPU(d_split_points);
        cudaManager.freeGPU(d_gain_ratios);

        return results;
    }
    catch (...) {
        std::cerr << "CUDA error in computeGainRatiosCUDA" << std::endl;

        std::vector<float> results(split_points.size(), 0.0f);
        return results;
    }
#else

    std::vector<float> results(split_points.size());
    for (size_t i = 0; i < split_points.size(); i++) {

        results[i] = 0.0f;
    }
    return results;
#endif
}

void DecisionTree::train(Dataset* dataset)
{
    CUDAManager& cudaManager = CUDAManager::getInstance();
    cudaManager.initialize();

    this->root = new TreeNode(this->max_depth - 1, this->min_samples_split);


    NodeDataset* node_queue = new NodeDataset[1024];
    node_queue[0].node = this->root;
    node_queue[0].dataset = new Dataset(dataset);
    int producer_index = 1;
    int consumer_index = 0;

    while (consumer_index < producer_index) {
        if (consumer_index >= 1024) {
            printf("consumer: node_queue run out of space.");
            exit(-1);
        }

        TreeNode* node = node_queue[consumer_index].node;
        Dataset* node_dataset = node_queue[consumer_index].dataset;
        consumer_index++;

        if (node_dataset->len < this->min_samples_split || node->subtree_max_depth <= 0) {
            node->class_label = node_dataset->get_most_common_class_label();
        }
        else {
#ifdef PARALLELIZE_ON_SPLITS_CUDA
            processNodeWithCUDA(node, node_dataset, node_queue, producer_index);
#else
            processNodeCPU(node, node_dataset, node_queue, producer_index);
#endif
        }

        delete node_dataset;
    }

    delete[] node_queue;
}

void DecisionTree::processNodeWithCUDA(TreeNode* node, Dataset* dataset,
    NodeDataset* node_queue, int& producer_index) {
    struct split_info {
        int feature_index;
        float split_point;
    };
    split_info best_split[10];
    float best_gain_ratio = 0;
    int best_split_count = 0;


    std::vector<float> features(dataset->len * FEATURE_NUM);
    std::vector<int> labels(dataset->len);

    for (int i = 0; i < dataset->len; i++) {
        for (int j = 0; j < FEATURE_NUM; j++) {
            features[i * FEATURE_NUM + j] = dataset->data[i].feature[j];
        }
        labels[i] = dataset->data[i].label;
    }


    for (int i = 0; i < FEATURE_NUM; i++) {

        std::vector<float> feature_values;
        for (int j = 0; j < dataset->len; j++) {
            feature_values.push_back(dataset->data[j].feature[i]);
        }
        std::sort(feature_values.begin(), feature_values.end());

        std::vector<float> split_points;
        for (int j = 0; j < dataset->len - 1; j++) {
            if (feature_values[j] != feature_values[j + 1]) {
                split_points.push_back((feature_values[j] + feature_values[j + 1]) / 2);
            }
        }

        if (split_points.empty()) continue;


        std::vector<float> gain_ratios = computeGainRatiosCUDA(
            features, labels, split_points, i);


        for (size_t j = 0; j < split_points.size(); j++) {
            float current_gain_ratio = gain_ratios[j];
            if (current_gain_ratio > best_gain_ratio) {
                best_gain_ratio = current_gain_ratio;
                best_split[0].feature_index = i;
                best_split[0].split_point = split_points[j];
                best_split_count = 1;
            }
            else if (current_gain_ratio == best_gain_ratio) {
                if (best_split_count < 10) {
                    best_split[best_split_count].feature_index = i;
                    best_split[best_split_count].split_point = split_points[j];
                    best_split_count++;
                }
            }
        }
    }

    if (best_gain_ratio == 0) {
        node->class_label = dataset->get_most_common_class_label();
    }
    else {

        int best_split_index_final = rand() % best_split_count;
        int feature_index = best_split[best_split_index_final].feature_index;
        float split_point = best_split[best_split_index_final].split_point;

        node->feature_index = feature_index;
        node->threshold = split_point;
        node->left_child = new TreeNode(node->subtree_max_depth - 1, node->min_samples_split);
        node->right_child = new TreeNode(node->subtree_max_depth - 1, node->min_samples_split);

        Dataset* left_dataset = new Dataset();
        Dataset* right_dataset = new Dataset();
        dataset->spilt(feature_index, split_point, left_dataset, right_dataset);

        if (producer_index >= 1023) {
            printf("producer: node_queue run out of space.");
            exit(-1);
        }
        node_queue[producer_index].node = node->left_child;
        node_queue[producer_index].dataset = left_dataset;
        node_queue[producer_index + 1].node = node->right_child;
        node_queue[producer_index + 1].dataset = right_dataset;
        producer_index += 2;
    }
}

void DecisionTree::processNodeCPU(TreeNode* node, Dataset* dataset,
    NodeDataset* node_queue, int& producer_index) {


    struct split_info {
        int feature_index;
        float split_point;
    };
    split_info best_split[10];
    float best_gain_ratio = 0;
    int best_split_count = 0;


    for (int i = 0; i < FEATURE_NUM; i++) {

        float* sorted_values = new float[dataset->len];
        for (int j = 0; j < dataset->len; j++) {
            sorted_values[j] = dataset->data[j].feature[i];
        }
        qsort(sorted_values, dataset->len, sizeof(float), cmpfunc);

        for (int j = 0; j < dataset->len - 1; j++) {
            if (sorted_values[j] == sorted_values[j + 1]) {
                continue;
            }
            float split_point = (sorted_values[j] + sorted_values[j + 1]) / 2;
            float current_gain_ratio = dataset->compute_gain_ratio(i, split_point);
            if (current_gain_ratio > best_gain_ratio) {
                best_gain_ratio = current_gain_ratio;
                best_split[0].feature_index = i;
                best_split[0].split_point = split_point;
                best_split_count = 1;
            }
            else if (current_gain_ratio == best_gain_ratio) {
                if (best_split_count < 10) {
                    best_split[best_split_count].feature_index = i;
                    best_split[best_split_count].split_point = split_point;
                    best_split_count++;
                }
            }
        }
        delete[] sorted_values;
    }

    if (best_gain_ratio == 0) {
        node->class_label = dataset->get_most_common_class_label();
    }
    else {

        int best_split_index_final = rand() % best_split_count;
        int feature_index = best_split[best_split_index_final].feature_index;
        float split_point = best_split[best_split_index_final].split_point;

        node->feature_index = feature_index;
        node->threshold = split_point;
        node->left_child = new TreeNode(node->subtree_max_depth - 1, node->min_samples_split);
        node->right_child = new TreeNode(node->subtree_max_depth - 1, node->min_samples_split);

        Dataset* left_dataset = new Dataset();
        Dataset* right_dataset = new Dataset();
        dataset->spilt(feature_index, split_point, left_dataset, right_dataset);

        if (producer_index >= 1023) {
            printf("producer: node_queue run out of space.");
            exit(-1);
        }
        node_queue[producer_index].node = node->left_child;
        node_queue[producer_index].dataset = left_dataset;
        node_queue[producer_index + 1].node = node->right_child;
        node_queue[producer_index + 1].dataset = right_dataset;
        producer_index += 2;
    }
}

int* DecisionTree::test(Dataset* dataset)
{
    int* result = new int[dataset->len];
    for (int i = 0; i < dataset->len; i++) {
        Instance* instance = &dataset->data[i];

        TreeNode* current_node = this->root;
        while (current_node->class_label == -1) {
            if (instance->feature[current_node->feature_index] <= current_node->threshold) {
                current_node = current_node->left_child;
            }
            else {
                current_node = current_node->right_child;
            }
        }
        result[i] = current_node->class_label;
    }
    return result;
}

char* DecisionTree::to_string()
{
    if (this->root == nullptr) {
        char* empty_tree = (char*)malloc(20 * sizeof(char));
        if (empty_tree == nullptr) {
            return nullptr;
        }
        strcpy(empty_tree, "Empty Tree");
        return empty_tree;
    }
    else {
        return this->root->to_string();
    }
}

TreeNode::~TreeNode()
{
    if (this->left_child != nullptr) {
        delete this->left_child;
    }
    if (this->right_child != nullptr) {
        delete this->right_child;
    }
}

char* TreeNode::to_string()
{
    if (this->class_label != -1) {
        char* leaf_str = (char*)malloc(20 * sizeof(char));
        if (leaf_str == nullptr) {
            return nullptr;
        }
        sprintf(leaf_str, "Leaf: Label %d\n", this->class_label);
        return leaf_str;
    }
    else {
        char* node_str = (char*)malloc(5000 * sizeof(char));
        if (node_str == nullptr) {
            return nullptr;
        }

        size_t offset = 0;
        int len = sprintf(node_str, "Node: Feature %d, Threshold %.2f\n", this->feature_index, this->threshold);
        if (len > 0) {
            offset += static_cast<size_t>(len);
        }

        if (this->left_child != nullptr) {
            char* left_str = this->left_child->to_string();
            if (left_str != nullptr) {
                char* left_copy = new char[strlen(left_str) + 1];
                strcpy(left_copy, left_str);
                const char* split_token = "\n";
                char* current_line = strtok(left_copy, split_token);
                while (current_line != nullptr && offset < 4990) {
                    int line_len = sprintf(node_str + offset, "\t%s\n", current_line);
                    if (line_len > 0) {
                        offset += static_cast<size_t>(line_len);
                    }
                    current_line = strtok(nullptr, split_token);
                }
                delete[] left_copy;
                delete[] left_str;
            }
        }

        if (this->right_child != nullptr) {
            char* right_str = this->right_child->to_string();
            if (right_str != nullptr) {
                char* right_copy = new char[strlen(right_str) + 1];
                strcpy(right_copy, right_str);
                const char* split_token = "\n";
                char* current_line = strtok(right_copy, split_token);
                while (current_line != nullptr && offset < 4990) {
                    int line_len = sprintf(node_str + offset, "\t%s\n", current_line);
                    if (line_len > 0) {
                        offset += static_cast<size_t>(line_len);
                    }
                    current_line = strtok(nullptr, split_token);
                }
                delete[] right_copy;
                delete[] right_str;
            }
        }
        return node_str;
    }
}