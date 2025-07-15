#pragma once
#include "DecisionTree.h"

class RandomForestClassifier
{
public:

	int tree_num;
	int tree_max_depth;
	int tree_min_samples_split;
	int bootstrap_data_num;
	float bootstrap_feature_ratio;

	RandomForestClassifier(int tree_num = 100, int tree_max_depth = 10, int tree_min_samples_split = 5, int bootstrap_data_num = 100, float bootstrap_feature_ratio = 0.75);
	~RandomForestClassifier();
	void train(Dataset* dataset);
	int* test(Dataset* dataset);

	DecisionTree* trees;
};

