#include "RandomForestClassifier.h"
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "ParallelConfig.h"

RandomForestClassifier::RandomForestClassifier(int tree_num, int tree_max_depth, int tree_min_samples_split, int bootstrap_data_num, float bootstrap_feature_ratio)
	: tree_num(tree_num), tree_max_depth(tree_max_depth), tree_min_samples_split(tree_min_samples_split), bootstrap_data_num(bootstrap_data_num), bootstrap_feature_ratio(bootstrap_feature_ratio)
{
	trees = new DecisionTree[tree_num];
	for (int i = 0; i < tree_num; i++)
	{
		trees[i] = DecisionTree(tree_max_depth, tree_min_samples_split);
	}
}

RandomForestClassifier::~RandomForestClassifier()
{
	delete[] this->trees;
}

void RandomForestClassifier::train(Dataset* dataset)
{
#ifdef PARALLELIZE_ON_TREES
#pragma omp parallel for
#endif
	for (int i = 0; i < tree_num; i++) // Parallelizable, no data contention
	{
		Dataset* bootstrap_sample = dataset->bootstrap(bootstrap_data_num, bootstrap_feature_ratio);
		this->trees[i].train(bootstrap_sample);
		delete bootstrap_sample;
	}
}

int* RandomForestClassifier::test(Dataset* dataset)
{
	int* results = new int[dataset->len];
	int(*votes)[CLASS_NUM] = (int (*)[CLASS_NUM])new int[CLASS_NUM * dataset->len]; // 2-dimensional array, first dimension represents instances, second dimension represents vote results
	int* max_votes = new int[dataset->len]; // Store the maximum number of votes per instance
	memset(votes, 0, sizeof(int) * CLASS_NUM * dataset->len);
	memset(max_votes, 0, sizeof(int) * dataset->len);
	for (int i = 0; i < this->tree_num; i++) // Parallelizable with data competition for max_votes
	{
		int* current_results = this->trees[i].test(dataset);
		for (int j = 0; j < dataset->len; j++) // Parallelizable, no data contention
		{
			votes[j][current_results[j]]++;
			if (votes[j][current_results[j]] > max_votes[j]) {
				max_votes[j] = votes[j][current_results[j]];
			}
		}
		delete[] current_results;
	}

	for (int i = 0; i < dataset->len; i++) // Parallelizable, no data contention
	{
		int max_vote_class[CLASS_NUM];
		int max_vote_count = 0;
		for (int j = 0; j < CLASS_NUM; j++)
		{
			if (votes[i][j] == max_votes[i]) {
				max_vote_class[max_vote_count] = j;
				max_vote_count++;
				break;
			}
		}
		int max_vote_class_final = max_vote_class[rand() % max_vote_count]; // Randomly select the category with the largest number of votes
		results[i] = max_vote_class_final;
	}

	return results;
}