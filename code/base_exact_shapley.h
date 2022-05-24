#include <stdlib.h>
#include <stdint.h>

#include "utils.h"
#include "benchmark.h"

uint64_t run_shapley(void *context);
void get_true_exact_KNN(void *context);
void compute_single_unweighted_knn_class_shapley(void *context);
void compute_transposed_single_unweighted_knn_class_shapley(void *context);
void current_opt_compute_single_unweighted_knn_class_shapley(void *context);
void single_unweighted_knn_class_shapley_opt(void *context);
void single_unweighted_knn_class_shapley_opt1(void *context);
void single_unweighted_knn_class_shapley_opt2(void *context);
void single_unweighted_knn_class_shapley_opt3(void *context);
void single_unweighted_knn_class_shapley_opt4(void *context);
void single_unweighted_knn_class_shapley_opt5(void *context);
void single_unweighted_knn_class_shapley_opt6(void *context);
void single_unweighted_knn_class_shapley_opt7(void *context);
void single_unweighted_knn_class_shapley_opt8(void *context);