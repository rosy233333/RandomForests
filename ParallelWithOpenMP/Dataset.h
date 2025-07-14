#pragma once

#define INSTANCE_MAX_NUM (1600) // Maximum number of instances
#define FEATURE_NUM (11) // Number of feature dimensions
#define CLASS_NUM (11) // Number of class labels

// Change the values of these constants in Dataset.cpp
extern const char* CLASS_LABELS[CLASS_NUM]; // The string corresponding to the class label
extern const char* SPLIT_TOKEN; // Separator for each row of data in the dataset file

class Instance
{
public:
	Instance(float* feature, int label);
	Instance() {} // Uninitialized
	float feature[FEATURE_NUM];
	int label; // Value 0~CLASS_NUM-1

	char* to_str(); // Use delete[] to release the returned pointer
};

class Dataset
{
public:
	Instance* data; // Variable-length arrays of Instance
	int len;

	Dataset() : data(NULL), len(0) {}
	Dataset(Dataset* d);
	~Dataset();

	static Dataset* from_file(const char* filename); // Use delete to free the returned pointer

	Dataset* shuffle();                        // Use delete to free the returned pointer
	// Sampling with playback data_num times
	// Only some of the feature dimensions will be selected, and the selection ratio is determined by the feature_ratio
	// Use delete to free the returned pointer
	Dataset* bootstrap(int data_num, float feature_ratio);
	float validate(int* result); // Validate the results given by the classifier and return the correctness rate

	// Operations used to implement decision trees
	float compute_gain_ratio(int feature_index, float threshold); // Calculate the gain ratio at a certain threshold for a feature
	void spilt(int feature_index, float threshold, Dataset* left, Dataset* right); // Split the dataset, left and right are output parameters that store instances less than or equal to and greater than the threshold, respectively, and the data for left and right should be NULL when passed in.
	int get_most_common_class_label(); // Get the class label with the most occurrences in the dataset
};