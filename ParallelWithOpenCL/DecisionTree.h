#pragma once
#include <cstddef>
#include "Dataset.h"

class TreeNode
{
public:
	int feature_index; 
	float threshold; 
	TreeNode* left_child;
	TreeNode* right_child;
	int class_label;   
	int subtree_max_depth; 
	int min_samples_split; 

	TreeNode(int subtree_max_depth = 0, int min_samples_split = 0) : feature_index(-1), threshold(0.0f), left_child(NULL), right_child(NULL), class_label(-1), subtree_max_depth(subtree_max_depth), min_samples_split(min_samples_split) {}
	~TreeNode();
	char* to_string(); 
};

class DecisionTree
{
public:

	struct NodeDataset {
		TreeNode* node;
		Dataset* dataset;
	};
	// config
	int max_depth;
	int min_samples_split;

	DecisionTree(int max_depth = 10, int min_samples_split = 5) : max_depth(max_depth), min_samples_split(min_samples_split), root(NULL) {}
	~DecisionTree();
	void train(Dataset* dataset);
	int* test(Dataset* dataset);
	char* to_string(); 
private:
	TreeNode* root;



	void processNodeWithCUDA(TreeNode* node, Dataset* dataset, NodeDataset* node_queue, int& producer_index);
	void processNodeCPU(TreeNode* node, Dataset* dataset, NodeDataset* node_queue, int& producer_index);
};

