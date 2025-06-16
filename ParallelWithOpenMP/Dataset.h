#pragma once

#define INSTANCE_MAX_NUM (1600) // 实例最大数目
#define FEATURE_NUM (11) // 特征维度数目
#define CLASS_NUM (11) // 类标签数目

// 在Dataset.cpp中修改这些常量的值
extern const char* CLASS_LABELS[CLASS_NUM]; // 类标签对应的字符串
extern const char* SPLIT_TOKEN; // 数据集文件中每行数据的分隔符

class Instance
{
public:
	Instance(float* feature, int label);
	Instance() {} // 未初始化
	float feature[FEATURE_NUM];
	int label; // 取值0~CLASS_NUM-1

	char* to_str(); // 使用delete[]释放返回的指针
};

class Dataset
{
public:
	Instance* data; // 作为Instance的可变长数组
	int len;

	Dataset() : data(NULL), len(0) {}
	Dataset(Dataset* d);
	~Dataset();

	static Dataset* from_file(const char* filename); // 使用delete释放返回的指针

	Dataset* shuffle();                        // 使用delete释放返回的指针
	// 有放回抽样data_num次
	// 同时会只选取部分特征维度，选取比例由feature_ratio决定
	// 使用delete释放返回的指针
	Dataset* bootstrap(int data_num, float feature_ratio);
	float validate(int* result); // 验证分类器给出的结果，返回正确率

	// 用于实现决策树的操作
	float compute_gain_ratio(int feature_index, float threshold); // 计算在某个特征某个阈值上的增益率
	void spilt(int feature_index, float threshold, Dataset* left, Dataset* right); // 分割数据集，left和right是输出参数，分别存储小于等于和大于阈值的实例，传入时left和right的data应为NULL。
	int get_most_common_class_label(); // 获取数据集中出现次数最多的类标签
};