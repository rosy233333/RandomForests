#pragma once
#include <cstddef>
#include "Dataset.h"

class TreeNode
{
public:
	int feature_index; // 分割时使用的特征
	float threshold; // 分割的阈值，小于等于则进入左节点，大于则进入右节点
	TreeNode* left_child;
	TreeNode* right_child;
	int class_label;   // 如果是叶子节点，则为类标签；如果不是叶子节点，则为-1。以该字段区分叶子节点和非叶子节点
	int subtree_max_depth; // 子树的最大深度，用于剪枝
	int min_samples_split; // 最小样本数分割阈值，用于剪枝

	TreeNode(int subtree_max_depth = 0, int min_samples_split = 0) : feature_index(-1), threshold(0.0f), left_child(NULL), right_child(NULL), class_label(-1), subtree_max_depth(subtree_max_depth), min_samples_split(min_samples_split) {}
	~TreeNode();
	char* to_string(); // 返回该节点的字符串表示（递归输出子节点），便于调试和可视化。需使用delete[]释放返回的指针。仅用于调试，不能在多线程环境下使用
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
	int* test(Dataset* dataset); // 返回一个预测的类标签数组，调用者必须删除该数组
	char* to_string(); // 返回决策树的字符串表示，便于调试和可视化。需使用delete[]释放返回的指针。仅用于调试，不能在多线程环境下使用
private:
	TreeNode* root;
};

