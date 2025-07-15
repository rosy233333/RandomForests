#define _USE_MATH_DEFINES
#include "Dataset.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstddef>
#include <algorithm>

const char* CLASS_LABELS[CLASS_NUM] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" }; 
const char* SPLIT_TOKEN = ",;\n"; 

Dataset* Dataset::from_file(const char* filename)
{
	FILE* file = fopen(filename, "r");
	if (file == NULL)
	{
		printf("Error opening file: %s\n", filename);
		exit(-1);
	}
	float feature[FEATURE_NUM];
	char label[100];
	Instance* buffer = new Instance[INSTANCE_MAX_NUM];
	int count = 0;
	for (;; count++)
	{
		char line[200];
		if (fgets(line, sizeof(line), file) == NULL)
		{
			break; 
		}
		else
		{
			
			char* token = strtok(line, SPLIT_TOKEN);
			int label_; 
			for (int i = 0; i < FEATURE_NUM; i++)
			{
				if (token == NULL)
				{
					printf("Error parsing line: %s\n", line);
					fclose(file);
					delete[] buffer;
					exit(-1);
				}
				feature[i] = atof(token); 
				token = strtok(NULL, SPLIT_TOKEN);
			}

			for (int i = 0;; i++)
			{
				if (token == NULL)
				{
					printf("Error parsing line: %s\n", line);
					fclose(file);
					delete[] buffer;
					exit(-1);
				}
				if (strcmp(token, CLASS_LABELS[i]) == 0)
				{
					label_ = i; 
					break;
				}

				if (i == CLASS_NUM - 1) 
				{
					printf("Invalid label: %s\n", token);
					fclose(file);
					delete[] buffer;
					exit(-1);
				}
			}

			buffer[count] = Instance(feature, label_);
		}
	}

	Dataset* dataset = new Dataset();
	dataset->len = count;
	dataset->data = new Instance[count];
	memcpy(dataset->data, buffer, count * sizeof(Instance));

	delete[] buffer; 
	return dataset;
}

Dataset* Dataset::shuffle()
{
	int data_num = this->len;
	int order[INSTANCE_MAX_NUM];
	for (int i = 0; i < data_num; i++)
	{
		int current;
		while (true)
		{
			current = rand() % data_num;
			
			bool flag = true;
			for (int j = 0; j < i; j++)
			{
				if (order[j] == current)
				{
					flag = false;
				}
			}
			if (flag)
			{
				break;
			}
		}
		order[i] = current;
	}

	Dataset* dataset = new Dataset();
	dataset->len = data_num;
	dataset->data = new Instance[data_num];
	for (int i = 0; i < data_num; i++)
	{
		memcpy(&(dataset->data[i]), &(this->data[order[i]]), sizeof(Instance));
	}

	return dataset;
}

Dataset* Dataset::bootstrap(int data_num, float feature_ratio)
{
	if (feature_ratio <= 0 || feature_ratio > 1)
	{
		printf("invalid feature_ratio: %f", feature_ratio);
		exit(-1);
	}

	
	int feature_num = (int)ceil(FEATURE_NUM * feature_ratio);
	int selected_feature_mask[FEATURE_NUM] = { 0 };
	for (int i = 0; i < feature_num;)
	{
		int current = rand() % FEATURE_NUM;
		if (selected_feature_mask[current] == 0)
		{
			selected_feature_mask[current] = 1;
			i++;
		}
	}

	
	Dataset* dataset = new Dataset();
	dataset->len = data_num;
	dataset->data = new Instance[data_num];
	for (int i = 0; i < data_num; i++)
	{
		int selected = rand() % this->len;
		for (int j = 0; j < FEATURE_NUM; j++) {
			dataset->data[i].feature[j] = this->data[selected].feature[j] * selected_feature_mask[j];
		}
		dataset->data[i].label = this->data[selected].label;
	}

	return dataset;
}

float Dataset::validate(int* result)
{
	if (this->len == 0)
	{
		printf("Dataset is empty.\n");
		exit(-1);
	}

	int correct_count = 0; 
	for (int i = 0; i < this->len; i++) {
		if (result[i] == this->data[i].label) {
			correct_count++;
		}
	}
	return 1.0 * correct_count / this->len;
}


float xlog2x(float x)
{
	if (x == 0)
	{
		return 0.0f;
	}
	else
	{
		return x * log2f(x);
	}
}

float Dataset::compute_gain_ratio(int feature_index, float threshold)
{
	int label_count[CLASS_NUM] = { 0 }; 
	int label_count_left[CLASS_NUM] = { 0 }; 
	int label_count_right[CLASS_NUM] = { 0 }; 

	for (int i = 0; i < this->len; i++)
	{
		label_count[this->data[i].label]++;
		if (this->data[i].feature[feature_index] <= threshold)
		{
			label_count_left[this->data[i].label]++;
		}
		else
		{
			label_count_right[this->data[i].label]++;
		}
	}

	int count = this->len;
	int left_count = 0;
	for (int i = 0; i < CLASS_NUM; i++)
	{
		left_count += label_count_left[i];
	}
	int right_count = 0;
	for (int i = 0; i < CLASS_NUM; i++)
	{
		right_count += label_count_right[i];
	}

	
	float entropy = 0.0f;
	for (int i = 0; i < CLASS_NUM; i++)
	{
		entropy -= xlog2x(1.0 * label_count[i] / count);
	}
	float entropy_left = 0.0f;
	for (int i = 0; i < CLASS_NUM; i++)
	{
		entropy_left -= xlog2x(1.0 * label_count_left[i] / left_count);
	}
	float entropy_right = 0.0f;
	for (int i = 0; i < CLASS_NUM; i++)
	{
		entropy_right -= xlog2x(1.0 * label_count_right[i] / right_count);
	}
	float gain = entropy - (entropy_left * left_count / count + entropy_right * right_count / count);
	float split_info = -xlog2x(1.0 * left_count / count) - xlog2x(1.0 * right_count / count);
	float gain_ratio = gain / split_info;
	return gain_ratio;
}

void Dataset::spilt(int feature_index, float threshold, Dataset* left, Dataset* right)
{
	
	int left_count = 0, right_count = 0;
	for (int i = 0; i < this->len; i++)
	{
		if (this->data[i].feature[feature_index] <= threshold)
		{
			left_count++;
		}
		else
		{
			right_count++;
		}
	}

	
	left->len = left_count;
	left->data = new Instance[left_count];
	right->len = right_count;
	right->data = new Instance[right_count];

	
	int left_index = 0, right_index = 0;
	for (int i = 0; i < this->len; i++)
	{
		if (this->data[i].feature[feature_index] <= threshold)
		{
			left->data[left_index] = this->data[i];
			left_index++;
		}
		else
		{
			right->data[right_index] = this->data[i];
			right_index++;
		}
	}
}

int Dataset::get_most_common_class_label()
{
	if (this->len == 0)
	{
		printf("Dataset is empty.\n");
		exit(-1);
	}
	int label_count[CLASS_NUM] = { 0 }; 
	for (int i = 0; i < this->len; i++)
	{
		label_count[this->data[i].label]++;
	}

	int max_label[CLASS_NUM]; 
	int max_count = 0; 
	int max_label_count = 0; 
	for (int i = 0; i < CLASS_NUM; i++)
	{
		if (label_count[i] > max_count)
		{
			max_count = label_count[i];
			max_label_count = 1;
			max_label[0] = i;
		}
		else if (label_count[i] == max_count)
		{
			max_label[max_label_count] = i;
			max_label_count++;
		}
	}

	int max_label_final = max_label[rand() % max_label_count]; 
	return max_label_final;
}

Dataset::Dataset(Dataset* d)
{
	this->len = d->len;
	this->data = new Instance[this->len];
	for (int i = 0; i < this->len; i++)
	{
		this->data[i] = d->data[i];
	}
}

Dataset::~Dataset()
{
	if (this->len != 0)
	{
		delete[] this->data;
	}
}

char* Instance::to_str()
{
	char* str = new char[200];
	int offset = 0;
	char label_name[20];
	strcpy(label_name, CLASS_LABELS[this->label]); 
	for (int i = 0; i < CLASS_NUM; i++) 
	{
		if (strlen(label_name) > 19)
		{
			label_name[19] = '\0'; 
			break;
		}
	}

	for (int i = 0; i < FEATURE_NUM; i++) 
	{
		offset += sprintf(str + offset, "%f,", this->feature[i]);
	}
	sprintf(str + offset, "%s\n", label_name);
	return str;
}

Instance::Instance(float* feature, int label)
{
	for (int i = 0; i < FEATURE_NUM; i++)
	{
		this->feature[i] = feature[i];
	}
	this->label = label;
}
