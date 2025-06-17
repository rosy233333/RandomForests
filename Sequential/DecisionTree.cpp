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
	// 设置根节点
	this->root = new TreeNode(this->max_depth - 1, this->min_samples_split);

	// 初始化待处理节点的队列
	struct NodeDataset {
		TreeNode* node;
		Dataset* dataset;
	};
	NodeDataset* node_queue = new NodeDataset[1024];
	node_queue[0].node = this->root;
	node_queue[0].dataset = new Dataset(dataset);
	int producer_index = 1; // 指向可以放入的第一个空位
	int consumer_index = 0; // 指向需要处理的第一个对象

	// 开始处理每个待处理节点（分裂节点）
	while (consumer_index < producer_index) // 可并行化，有对node_queue、producer_index和consumer_index的数据竞争。因此并行化时需要对节点队列加锁。
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
			// 如果数据集长度小于最小分裂样本数，或者子树最大深度小于等于0，则设置叶节点
			node->class_label = dataset->get_most_common_class_label();
		}
		else
		{
			// 否则，进行分裂操作
			struct spilt_info {
				int feature_index;
				float split_point;
			};
			spilt_info best_split[10]; // 最佳分裂点信息
			float best_gain_ratio = 0;
			int best_split_count = 0;

			// 寻找最佳分裂点
			for (int i = 0; i < FEATURE_NUM; i++) {
				// 在该特征内部对样本值排序
				float* sorted_values = new float[dataset->len];
				for (int j = 0; j < dataset->len; j++) {
					sorted_values[j] = dataset->data[j].feature[i];
				}
				qsort(sorted_values, dataset->len, sizeof(float), cmpfunc);

				// 根据排序后的值寻找分裂点
				for (int j = 0; j < dataset->len - 1; j++) // 分裂点的数量比样本值的数量少1  // 可并行化，无数据竞争
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
			} // 寻找过程结束后，best_split_count一定>0？

			if (best_gain_ratio == 0)
			{
				// 分裂无法提供信息增益，则设置叶节点
				node->class_label = dataset->get_most_common_class_label();
			}
			else
			{
				// 找到最佳分裂点
				int best_spilt_index_final = rand() % best_split_count;
				int feature_index = best_split[best_spilt_index_final].feature_index;
				float split_point = best_split[best_spilt_index_final].split_point;

				// 分裂当前节点
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

		delete dataset; // dataset需要释放，而node不需释放，因为它是树的一部分
	}

	delete[] node_queue;
}

int* DecisionTree::test(Dataset* dataset)
{
	int* result = new int[dataset->len];
	for (int i = 0; i < dataset->len; i++) // 可并行化，无数据竞争
	{
		Instance* instance = &dataset->data[i]; // 对该实例进行测试

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
		result[i] = current_node->class_label; // 叶节点的类标签
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
				sprintf(node_str + offset, "\t%s\n", current_line); // 添加制表符缩进
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
				sprintf(node_str + offset, "\t%s\n", current_line); // 添加制表符缩进
				offset += strlen(current_line) + 2;
				current_line = strtok(NULL, split_token);
			} while (current_line != NULL);
			delete[] right_str;
		}
		return node_str;
	}
	return nullptr;
}
