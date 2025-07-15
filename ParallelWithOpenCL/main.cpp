#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "Dataset.h"
#include "RandomForestClassifier.h"
#include "CUDAConfig.h"

int main()
{
    srand((unsigned)time(NULL));

    
    CUDAManager& cudaManager = CUDAManager::getInstance();
    if (!cudaManager.initialize()) {
        fprintf(stderr, "Failed to initialize CUDA. Falling back to CPU.\n");
        return -1;
    }

    Dataset* train_dataset = Dataset::from_file("..\\wine-red-train.txt");
    Dataset* test_dataset = Dataset::from_file("..\\wine-red-test.txt");
    RandomForestClassifier* classifier = new RandomForestClassifier(100, 10, 5, 1000, 0.63);

    clock_t start_time = clock();
    classifier->train(train_dataset);
    clock_t end_time = clock();
    printf("Training time: %.4f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    int* train_result = classifier->test(train_dataset);
    printf("Train dataset accuracy: %.2f%%\n", train_dataset->validate(train_result) * 100);

    int* test_result = classifier->test(test_dataset);
    printf("Test dataset accuracy: %.2f%%\n", test_dataset->validate(test_result) * 100);

    delete[] train_result;
    delete[] test_result;
    delete classifier;
    delete train_dataset;
    delete test_dataset;

    // clear CUDA 
    cudaManager.cleanup();

    return 0;
}