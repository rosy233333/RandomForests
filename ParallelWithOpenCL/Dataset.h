#pragma once
#include <cstddef>

#define INSTANCE_MAX_NUM (1600) 
#define FEATURE_NUM (11) 
#define CLASS_NUM (11) 


extern const char* CLASS_LABELS[CLASS_NUM]; 
extern const char* SPLIT_TOKEN; 

class Instance
{
public:
	Instance(float* feature, int label);
	Instance() :label(0) {
		for (int i = 0; i < FEATURE_NUM; i++) {
			feature[i] = 0.0f;
		}
	}
	float feature[FEATURE_NUM];
	int label;

	char* to_str();
};

class Dataset
{
public:
	Instance* data; 
	int len;

	Dataset() : data(NULL), len(0) {}
	Dataset(Dataset* d);
	~Dataset();

	static Dataset* from_file(const char* filename); 

	Dataset* shuffle();                        
	
	Dataset* bootstrap(int data_num, float feature_ratio);
	float validate(int* result); 

	float compute_gain_ratio(int feature_index, float threshold); 
	void spilt(int feature_index, float threshold, Dataset* left, Dataset* right); 
	int get_most_common_class_label(); 
};