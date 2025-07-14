#pragma once

#define THREAD_NUM (8)

// Controlling parallelization through annotation and unannotation
// Parallelization in training different decision trees
#define PARALLELIZE_ON_TREES 
// Parallelization in training different nodes of a decision tree
#define PARALLELIZE_ON_NODES
// Parallelization in selecting different splitting points and calculating the gain of different splitting points
#define PARALLELIZE_ON_SPLITS
