#pragma once

#define THREAD_NUM (8)

// 通过注释和取消注释来控制并行化的方式
// 在训练不同的决策树时并行化
#define PARALLELIZE_ON_TREES 
// 在训练决策树的不同节点时并行化
#define PARALLELIZE_ON_NODES
// 在选取不同分裂点、计算不同分裂点的增益时并行化
#define PARALLELIZE_ON_SPLITS
