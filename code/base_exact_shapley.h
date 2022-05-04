#include <stdlib.h>
#include <stdint.h>

#include "utils.h"
#include "benchmark.h"

uint64_t run_shapley(void *context);
void get_true_KNN(context_t *context);

void compute_single_unweighted_knn_class_shapley(context_t *context);