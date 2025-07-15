//#include "RandomForestClassifier.h"
//#include <cstring>
//#include <cstdlib>
//#include <ctime>
//#include "ParallelConfig.h"
//
//
//RandomForestClassifier::RandomForestClassifier(int tree_num, int tree_max_depth, int tree_min_samples_split, int bootstrap_data_num, float bootstrap_feature_ratio)
//	: tree_num(tree_num), tree_max_depth(tree_max_depth), tree_min_samples_split(tree_min_samples_split), bootstrap_data_num(bootstrap_data_num), bootstrap_feature_ratio(bootstrap_feature_ratio)
//{
//	trees = new DecisionTree[tree_num];
//	for (int i = 0; i < tree_num; i++)
//	{
//		trees[i] = DecisionTree(tree_max_depth, tree_min_samples_split);
//	}
//}
//
//RandomForestClassifier::~RandomForestClassifier()
//{
//	delete[] this->trees;
//}
//
//void RandomForestClassifier::train(Dataset* dataset)
//{
//	for (int i = 0; i < tree_num; i++)
//	{
//		Dataset* bootstrap_sample = dataset->bootstrap(bootstrap_data_num, bootstrap_feature_ratio);
//		this->trees[i].train(bootstrap_sample);
//		delete bootstrap_sample;
//	}
//}
//
//int* RandomForestClassifier::test(Dataset* dataset)
//{
//	int* results = new int[dataset->len];
//	int(*votes)[CLASS_NUM] = (int (*)[CLASS_NUM])new int[CLASS_NUM * dataset->len];
//	int* max_votes = new int[dataset->len];
//	memset(votes, 0, sizeof(int) * CLASS_NUM * dataset->len);
//	memset(max_votes, 0, sizeof(int) * dataset->len);
//	for (int i = 0; i < this->tree_num; i++)
//	{
//		int* current_results = this->trees[i].test(dataset);
//		for (int j = 0; j < dataset->len; j++)
//		{
//			votes[j][current_results[j]]++;
//			if (votes[j][current_results[j]] > max_votes[j]) {
//				max_votes[j] = votes[j][current_results[j]];
//			}
//		}
//		delete[] current_results;
//	}
//
//	for (int i = 0; i < dataset->len; i++)
//	{
//		int max_vote_class[CLASS_NUM];
//		int max_vote_count = 0;
//		for (int j = 0; j < CLASS_NUM; j++)
//		{
//			if (votes[i][j] == max_votes[i]) {
//				max_vote_class[max_vote_count] = j;
//				max_vote_count++;
//				break;
//			}
//		}
//		int max_vote_class_final = max_vote_class[rand() % max_vote_count];
//		results[i] = max_vote_class_final;
//	}
//
//	return results;
//}

#include "RandomForestClassifier.h"
#include "ParallelConfig.h"
#include <mpi.h>
#include <ctime>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

RandomForestClassifier::RandomForestClassifier(int tree_num, int tree_max_depth, int tree_min_samples_split, int bootstrap_data_num, float bootstrap_feature_ratio)
    : tree_num(tree_num), tree_max_depth(tree_max_depth), tree_min_samples_split(tree_min_samples_split), bootstrap_data_num(bootstrap_data_num), bootstrap_feature_ratio(bootstrap_feature_ratio)
{
    trees = new DecisionTree[tree_num];
    for (int i = 0; i < tree_num; i++) {
        trees[i] = DecisionTree(tree_max_depth, tree_min_samples_split);
    }
}

RandomForestClassifier::~RandomForestClassifier() {
    delete[] trees;
}

void RandomForestClassifier::train(Dataset* dataset) {
    int rank = 0, size = 1;
#if PARALLEL_TREE
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    int trees_per_proc = tree_num / size;
    int extra = tree_num % size;
    int start = rank * trees_per_proc + std::min(rank, extra);
    int end = start + trees_per_proc + (rank < extra ? 1 : 0);

    for (int i = start; i < end; i++) {
        Dataset* sample = dataset->bootstrap(bootstrap_data_num, bootstrap_feature_ratio);
        trees[i].train(sample);
        delete sample;
    }

#if PARALLEL_TREE
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank != 0) {
        MPI_Send(&trees[start], (end - start) * sizeof(DecisionTree), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
    else {
        for (int src = 1; src < size; src++) {
            int s = src * trees_per_proc + std::min(src, extra);
            int e = s + trees_per_proc + (src < extra ? 1 : 0);
            MPI_Recv(&trees[s], (e - s) * sizeof(DecisionTree), MPI_BYTE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
#endif
}

int* RandomForestClassifier::test(Dataset* dataset) {
    int* votes = new int[dataset->len * CLASS_NUM]();
    int* result = new int[dataset->len];

    for (int i = 0; i < tree_num; i++) {
        int* pred = trees[i].test(dataset);
        for (int j = 0; j < dataset->len; j++) {
            votes[j * CLASS_NUM + pred[j]]++;
        }
        delete[] pred;
    }

    for (int i = 0; i < dataset->len; i++) {
        int max_vote = -1, label = -1;
        for (int c = 0; c < CLASS_NUM; c++) {
            if (votes[i * CLASS_NUM + c] > max_vote) {
                max_vote = votes[i * CLASS_NUM + c];
                label = c;
            }
        }
        result[i] = label;
    }

    delete[] votes;
    return result;
}
