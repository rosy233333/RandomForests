#pragma once
#include "DecisionTree.h"
// 使用C4.5算法实现的随机森林分类器
class RandomForestClassifier
{
public:
	// config
	int tree_num;
	int tree_max_depth;
	int tree_min_samples_split;
	int bootstrap_data_num;
	float bootstrap_feature_ratio;

	RandomForestClassifier(int tree_num = 100, int tree_max_depth = 10, int tree_min_samples_split = 5, int bootstrap_data_num = 100, float bootstrap_feature_ratio = 0.75);
	~RandomForestClassifier();
	void train(Dataset* dataset);
	int* test(Dataset* dataset); // 返回一个预测的类别标签数组，调用者必须删除该数组
	//private:
	DecisionTree* trees;
};

