#pragma once
#include <cstddef>
#include "Dataset.h"

class TreeNode
{
public:
	int feature_index; // Feature used in splitting
	float threshold; // Threshold for splitting, less than or equal to goes to the left node, greater than goes to the right node
	TreeNode* left_child;
	TreeNode* right_child;
	int class_label;   // Class label if it is a leaf node; -1 if it is not a leaf node. Distinguish leaf nodes from non-leaf nodes with this field
	int subtree_max_depth; // Maximum depth of subtree for pruning
	int min_samples_split; // Minimum sample size segmentation threshold for pruning

	TreeNode(int subtree_max_depth = 0, int min_samples_split = 0) : feature_index(-1), threshold(0.0f), left_child(NULL), right_child(NULL), class_label(-1), subtree_max_depth(subtree_max_depth), min_samples_split(min_samples_split) {}
	~TreeNode();
	char* to_string(); // Returns a string representation of the node (recursively outputs child nodes) for debugging and visualization. You need to use delete[] to release the returned pointer. Only for debugging, can not be used in multi-threaded environment.
};

class DecisionTree
{
public:
	// config
	int max_depth;
	int min_samples_split;

	DecisionTree(int max_depth = 10, int min_samples_split = 5) : max_depth(max_depth), min_samples_split(min_samples_split), root(NULL) {}
	~DecisionTree();
	void train(Dataset* dataset);
	int* test(Dataset* dataset); // Returns an array of predicted class labels, which the caller must delete
	char* to_string(); // Returns a string representation of the decision tree for debugging and visualization. Use delete[] to release the returned pointer. Only for debugging, cannot be used in multi-threaded environment.
private:
	TreeNode* root;
};

