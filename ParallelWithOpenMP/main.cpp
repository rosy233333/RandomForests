#include <cstdio>
#include "Dataset.h"
#include "RandomForestClassifier.h"
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "ParallelConfig.h"

int main()
{
	srand((unsigned)time(NULL));

	omp_set_nested(1); // Enable Parallel Nesting
	omp_set_num_threads(THREAD_NUM);

	Dataset* train_dataset = Dataset::from_file("..\\wine-red-train.txt");
	Dataset* test_dataset = Dataset::from_file("..\\wine-red-test.txt");
	RandomForestClassifier* classifier = new RandomForestClassifier(100, 10, 5, 1000, 0.63);

	clock_t start_time = clock();
	classifier->train(train_dataset);
	clock_t end_time = clock();
	printf("Training time: %.4f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

	//printf("Trees:\n");
	//for (int i = 0; i < classifier->tree_num; i++) {
	//	printf("Tree %d:\n", i + 1);
	//	char* tree_str = classifier->trees[i].to_string();
	//	printf("%s", tree_str);
	//	delete[] tree_str;
	//}

	//printf("Train dataset:\n");
	//int* train_result = classifier->test(train_dataset);
	//for (int i = 0; i < train_dataset->len; i++) {
	//	printf("predict: %d, actual: %d\n", train_result[i], train_dataset->data[i].label);
	//}

	//printf("Test dataset:\n");
	//int* test_result = classifier->test(test_dataset);
	//for (int i = 0; i < test_dataset->len; i++) {
	//	printf("predict: %d, actual: %d\n", test_result[i], test_dataset->data[i].label);
	//}

	int* train_result = classifier->test(train_dataset);
	printf("Train dataset accuracy: %.2f%%\n", train_dataset->validate(train_result) * 100);

	int* test_result = classifier->test(test_dataset);
	printf("Test dataset accuracy: %.2f%%\n", test_dataset->validate(test_result) * 100);

	delete[] train_result;
	delete[] test_result;
	delete classifier;
	delete train_dataset;
	delete test_dataset;
	return 0;
}
