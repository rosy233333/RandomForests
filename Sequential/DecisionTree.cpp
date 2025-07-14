#include "DecisionTree.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>

DecisionTree::~DecisionTree()
{
	if (this->root != NULL) {
		delete this->root;
	}
}

int cmpfunc(const void* a, const void* b)
{
	float fa = *(const float*)a;
	float fb = *(const float*)b;
	return (fa > fb) - (fa < fb);
}

void DecisionTree::train(Dataset* dataset)
{
	// Setting up the root node
	this->root = new TreeNode(this->max_depth - 1, this->min_samples_split);

	// Initialize the queue of pending nodes
	struct NodeDataset {
		TreeNode* node;
		Dataset* dataset;
	};
	NodeDataset* node_queue = new NodeDataset[1024];
	node_queue[0].node = this->root;
	node_queue[0].dataset = new Dataset(dataset);
	int producer_index = 1; // Points to the first empty space that can be placed
	int consumer_index = 0; // Points to the first object to be processed

	// Start processing each pending node (split node)
	while (consumer_index < producer_index) // Parallelizable with data contention for node_queue, producer_index and consumer_index. Therefore parallelization requires a lock on the node queue.
	{
		if (consumer_index >= 1024)
		{
			printf("consumer: node_queue run out of space.");
			exit(-1);
		}

		TreeNode* node = node_queue[consumer_index].node;
		Dataset* dataset = node_queue[consumer_index].dataset;
		consumer_index++;

		if (dataset->len < this->min_samples_split || node->subtree_max_depth <= 0)
		{
			// If the length of the dataset is less than the minimum number of split samples, or the maximum depth of the subtree is less than or equal to 0, set the leaf node
			node->class_label = dataset->get_most_common_class_label();
		}
		else
		{
			// Otherwise, perform the split operation
			struct spilt_info {
				int feature_index;
				float split_point;
			};
			spilt_info best_split[10]; // Optimal split point information
			float best_gain_ratio = 0;
			int best_split_count = 0;

			// Finding the optimal split point
			for (int i = 0; i < FEATURE_NUM; i++) {
				// Order the sample values within this feature
				float* sorted_values = new float[dataset->len];
				for (int j = 0; j < dataset->len; j++) {
					sorted_values[j] = dataset->data[j].feature[i];
				}
				qsort(sorted_values, dataset->len, sizeof(float), cmpfunc);

				// Finding split points based on sorted values
				for (int j = 0; j < dataset->len - 1; j++) // The number of split points is less than the number of sample values by 1  // Parallelizable, no data contention
				{
					if (sorted_values[j] == sorted_values[j + 1])
					{
						continue;
					}
					float spilt_point = (sorted_values[j] + sorted_values[j + 1]) / 2;
					float current_gain_ratio = dataset->compute_gain_ratio(i, spilt_point);
					if (current_gain_ratio > best_gain_ratio)
					{
						best_gain_ratio = current_gain_ratio;
						best_split[0].feature_index = i;
						best_split[0].split_point = spilt_point;
						best_split_count = 1;
					}
					else if (current_gain_ratio == best_gain_ratio)
					{
						if (best_split_count < 10)
						{
							best_split[best_split_count].feature_index = i;
							best_split[best_split_count].split_point = spilt_point;
							best_split_count++;
						}
					}
				}

				delete[] sorted_values;
			} // best_split_count must be > 0 at the end of the search process?

			if (best_gain_ratio == 0)
			{
				// Split fails to provide information gain, then set leaf nodes
				node->class_label = dataset->get_most_common_class_label();
			}
			else
			{
				// Finding the optimal split point
				int best_spilt_index_final = rand() % best_split_count;
				int feature_index = best_split[best_spilt_index_final].feature_index;
				float split_point = best_split[best_spilt_index_final].split_point;

				// Splitting the current node
				node->feature_index = feature_index;
				node->threshold = split_point;
				node->left_child = new TreeNode(node->subtree_max_depth - 1, node->min_samples_split);
				node->right_child = new TreeNode(node->subtree_max_depth - 1, node->min_samples_split);
				Dataset* left_dataset = new Dataset();
				Dataset* right_dataset = new Dataset();
				dataset->spilt(feature_index, split_point, left_dataset, right_dataset);
				if (producer_index >= 1023)
				{
					printf("producer: node_queue run out of space.");
					exit(-1);
				}
				node_queue[producer_index].node = node->left_child;
				node_queue[producer_index].dataset = left_dataset;
				node_queue[producer_index + 1].node = node->right_child;
				node_queue[producer_index + 1].dataset = right_dataset;
				producer_index += 2;
			}
		}

		delete dataset; // The dataset needs to be freed, while the node doesn't need to be freed because it is part of the tree
	}

	delete[] node_queue;
}

int* DecisionTree::test(Dataset* dataset)
{
	int* result = new int[dataset->len];
	for (int i = 0; i < dataset->len; i++) // Parallelizable, no data contention
	{
		Instance* instance = &dataset->data[i]; // Tests on this instance

		TreeNode* current_node = this->root;
		while (current_node->class_label == -1)
		{
			if (instance->feature[current_node->feature_index] <= current_node->threshold)
			{
				current_node = current_node->left_child;
			}
			else
			{
				current_node = current_node->right_child;
			}
		}
		result[i] = current_node->class_label; // Class labels for leaf nodes
	}
	return result;
}

char* DecisionTree::to_string()
{
	if (this->root == NULL) {
		char* empty_tree = (char*)malloc(20 * sizeof(char));
		strcat(empty_tree, "Empty Tree");
		return empty_tree;
	}
	else {
		return this->root->to_string();
	}
}

TreeNode::~TreeNode()
{
	if (this->left_child != NULL) {
		delete this->left_child;
	}
	if (this->right_child != NULL) {
		delete this->right_child;
	}
}

char* TreeNode::to_string()
{
	if (this->class_label != -1) {
		char* leaf_str = (char*)malloc(20 * sizeof(char));
		sprintf(leaf_str, "Leaf: Label %d\n", this->class_label);
		return leaf_str;
	}
	else {
		char* node_str = (char*)malloc(5000 * sizeof(char));
		int offset = 0;
		sprintf(node_str, "Node: Feature %d, Threshold %.2f\n", this->feature_index, this->threshold);
		offset += strlen(node_str);

		if (this->left_child != NULL) {
			char* left_str = this->left_child->to_string();
			const char split_token[2] = "\n";
			char* current_line = strtok(left_str, split_token);
			do {
				sprintf(node_str + offset, "\t%s\n", current_line); // Add tab indentation
				offset += strlen(current_line) + 2;
				current_line = strtok(NULL, split_token);
			} while (current_line != NULL);
			delete[] left_str;
		}

		if (this->right_child != NULL) {
			char* right_str = this->right_child->to_string();
			const char split_token[2] = "\n";
			char* current_line = strtok(right_str, split_token);
			do {
				sprintf(node_str + offset, "\t%s\n", current_line); // Add tab indentation
				offset += strlen(current_line) + 2;
				current_line = strtok(NULL, split_token);
			} while (current_line != NULL);
			delete[] right_str;
		}
		return node_str;
	}
	return nullptr;
}
